"""Evaluation metrics (PAPER §6). Pure functions over UIR dicts / note lists.

All operate on the ``mambo.utterance.v1`` JSON shape so the same code scores the
pipeline and any future end-to-end model.
"""

from __future__ import annotations

from typing import Any

import mir_eval
import numpy as np


def _hz(midi) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def bootstrap_ci(values, *, iters: int = 5000, alpha: float = 0.05,
                 seed: int = 1234) -> tuple:
    """Percentile bootstrap CI for the mean of per-item scores (e.g. per-clip 0/1
    success). Returns (mean, lo, hi); deterministic (fixed seed) so the reported
    interval is reproducible. Small-n honesty: with n items the CI is wide and
    that is the point — it states the uncertainty a single point estimate hides."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion k/n. The correct CI for our
    pass-rate H-gates: at the boundary the bootstrap of binary outcomes degenerates
    (6/6 -> [1.0, 1.0]) whereas Wilson gives the honest ~[0.61, 1.0]. Returns
    (p, lo, hi)."""
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- #
# Melody: note onset+pitch F1 (mir_eval), key top-k.
# --------------------------------------------------------------------------- #


def note_prf(ref_notes: list[dict], est_notes: list[dict], *,
             onset_tol: float = 0.05, pitch_tol: float = 50.0) -> tuple[float, float, float]:
    """Onset+pitch P/R/F (offsets ignored), 50 ms / 50 cent default (PAPER §6).

    Notes are dicts with ``t0``, ``dur``, ``midi``. Times must be in the same
    reference frame (caller offsets to span-local or global consistently).
    """
    def to_arrays(notes):
        if not notes:
            return np.zeros((0, 2)), np.zeros(0)
        iv = np.array([[n["t0"], n["t0"] + n["dur"]] for n in notes])
        return iv, _hz([n["midi"] for n in notes])

    ref_iv, ref_p = to_arrays(ref_notes)
    est_iv, est_p = to_arrays(est_notes)
    if len(ref_iv) == 0 and len(est_iv) == 0:
        return 1.0, 1.0, 1.0
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=onset_tol, pitch_tolerance=pitch_tol, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def key_in_topk(gt_key: str, est_candidates: list[dict], k: int = 2) -> bool:
    return gt_key in [c["key"] for c in est_candidates[:k]]


# --------------------------------------------------------------------------- #
# Router: segment-level P/R/F1 per kind + boundary error.
# --------------------------------------------------------------------------- #


def _overlap(a: dict, b: dict) -> float:
    lo, hi = max(a["t0"], b["t0"]), min(a["t1"], b["t1"])
    return max(0.0, hi - lo)


def segment_prf(ref: list[dict], est: list[dict], *, iou_min: float = 0.5
                ) -> tuple[float, float, float, list[float]]:
    """Segment detection P/R/F1 (a match = same kind + temporal IoU >= iou_min)
    plus the list of boundary errors (s) over matched segments.

    Greedy best-overlap matching. ``ambiguous`` est segments match a ref of
    either ``speech`` or ``melody`` (the planner gets both decodings).
    """
    used_est: set[int] = set()
    boundary_errs: list[float] = []
    tp = 0
    for rseg in ref:
        best_j, best_iou = -1, 0.0
        for j, eseg in enumerate(est):
            if j in used_est:
                continue
            if not _kind_compatible(rseg["kind"], eseg["kind"]):
                continue
            inter = _overlap(rseg, eseg)
            union = (rseg["t1"] - rseg["t0"]) + (eseg["t1"] - eseg["t0"]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_min:
            used_est.add(best_j)
            tp += 1
            boundary_errs.append(abs(rseg["t0"] - est[best_j]["t0"]))
            boundary_errs.append(abs(rseg["t1"] - est[best_j]["t1"]))
    fp = len(est) - len(used_est)
    fn = len(ref) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, boundary_errs


def _kind_compatible(ref_kind: str, est_kind: str) -> bool:
    if est_kind == "ambiguous":
        return ref_kind in ("speech", "melody", "ambiguous")
    return ref_kind == est_kind


# --------------------------------------------------------------------------- #
# Hallucination containment: ASR text leaking onto melody time spans.
# --------------------------------------------------------------------------- #


def hallucination_on_melody(ref: list[dict], est: list[dict], *, min_overlap: float = 0.3) -> bool:
    """True if any est segment carrying ``text`` overlaps a GT melody span by
    >= ``min_overlap`` of the melody span — i.e. ASR words leaked onto a hum.
    """
    melodies = [s for s in ref if s["kind"] == "melody"]
    texted = [s for s in est if s.get("text")]
    for m in melodies:
        mlen = m["t1"] - m["t0"]
        for s in texted:
            if mlen > 0 and _overlap(m, s) / mlen >= min_overlap:
                return True
    return False


# --------------------------------------------------------------------------- #
# Speech: WER over matched speech spans.
# --------------------------------------------------------------------------- #


def wer(ref_text: str, est_text: str) -> float:
    import jiwer

    ref_text = (ref_text or "").strip().lower()
    est_text = (est_text or "").strip().lower()
    if not ref_text:
        return 0.0 if not est_text else 1.0
    return float(jiwer.wer(ref_text, est_text))


# --------------------------------------------------------------------------- #
# Aggregation helpers.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Action-plan accuracy (PAPER §6): does the plan capture the golden intent?
# --------------------------------------------------------------------------- #

_PRIMARY_OPS = ("insert_notes", "change_track_volume", "mute_track", "solo_track",
                "transport", "create_track", "set_project_tempo", "ask_user")


def _key_action(plan: dict) -> Optional[dict]:
    """The plan's primary intent action (prefer a concrete op over ask_user)."""
    acts = plan.get("actions", [])
    for a in acts:
        if a["op"] in _PRIMARY_OPS and a["op"] != "ask_user":
            return a
    for a in acts:
        if a["op"] == "ask_user":
            return a
    return acts[0] if acts else None


def _args_match(op: str, g: dict, e: dict, *, tempo_tol: float = 0.20) -> bool:
    if op == "insert_notes":
        if g.get("notes_ref") != e.get("notes_ref"):
            return False
        gt, et = g.get("tempo_bpm"), e.get("tempo_bpm")
        return gt and et and abs(et - gt) <= tempo_tol * gt
    if op == "change_track_volume":
        gd, ed = g.get("delta_db", 0), e.get("delta_db", 0)
        if (gd > 0) != (ed > 0):  # sign must match
            return False
        gi = (g.get("track") or {}).get("index")
        ei = (e.get("track") or {}).get("index")
        return gi is None or gi == ei
    if op == "set_project_tempo":
        gb, eb = g.get("bpm"), e.get("bpm")
        return gb and eb and abs(eb - gb) <= tempo_tol * gb
    if op == "transport":
        return g.get("action") == e.get("action")
    if op in ("mute_track", "solo_track"):
        gi = (g or {}).get("index")
        ei = (e or {}).get("index")
        return gi is None or gi == ei
    return True  # create_track / ask_user: op match is enough


def plan_correct(golden: dict, est: dict) -> bool:
    """An est plan is correct if it contains an action matching the golden plan's
    key action (same op + args within tolerance). Extra/auxiliary ops are fine."""
    key = _key_action(golden)
    if key is None:
        return not est.get("actions")
    for a in est.get("actions", []):
        if a["op"] == key["op"] and _args_match(key["op"], key.get("args", {}), a.get("args", {})):
            return True
    return False


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "min": None, "median": None}
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "min": float(a.min()),
            "median": float(np.median(a))}
