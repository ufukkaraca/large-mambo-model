"""Held-out real-voice benchmark — the harness that turns the synthetic, in-distribution
headline into a defensible main-conference result (the v2 benchmark plan).

For every real-voice clip it runs the full pipeline and scores **whichever metrics
the committed ground truth supports**, reporting per-voice + held-out-pooled with
confidence intervals (Wilson for rates, bootstrap for means). It is designed to
*scale with the labels*: add richer GT fields to a speaker's `manifest.jsonl` and
the corresponding metric lights up; until then it is reported PENDING, honestly.

Ground-truth fields read from each manifest entry (all optional except the count):
  intended_notes / verified_notes  -> note-count ±tol            (always available)
  gt_pitches: [midi,...]           -> transposition-invariant pitch-sequence F1
  gt_notes:   [{midi,t0,dur},...]  -> note onset+pitch F1 (mir_eval-style)
  gt_segments:[{kind,t0,t1},...]   -> segmentation F1
  lyric: "..."  (kind sung)        -> dual-decode lyric word-recall / WER + containment

Speakers under fixtures/human/voices/<name>/ and fixtures/human/sung/<name>/ are
HELD-OUT (nothing tuned on them); the operator (`sung/ufuk`) is flagged in-distribution.

    cd lab && uv run python -m mambo_lab.eval.voicebench
Writes runs/<ts>-voicebench/results.json.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import metrics
from .b6_omni import pitch_prf
from .. import dual_decode, fuse, melody

REPO = Path(__file__).resolve().parents[3]
SOURCES = [REPO / "fixtures" / "human" / "voices", REPO / "fixtures" / "human" / "sung"]
OPERATOR = {"ufuk"}  # in-distribution voice — reported but excluded from held-out pools
RUNS = REPO / "runs"


def _load(p: Path):
    a, sr = sf.read(str(p), dtype="float32")
    return (a.mean(axis=1) if a.ndim > 1 else a), sr


def _entries(d: Path) -> list[dict]:
    man = d / "manifest.jsonl"
    if not man.exists():
        return []
    out = []
    for ln in man.read_text().splitlines():
        if ln.strip():
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "wav" in e and (d / e["wav"]).exists():
                out.append(e)
    return out


def score_clip(path: Path, e: dict, key: str | None) -> dict:
    audio, sr = _load(path)
    uir = fuse.file_to_uir(audio, sr, strategy="joint", utterance_id=e["wav"]).to_dict()
    segs = uir["segments"]
    det_notes = [n for s in segs if s["kind"] in ("melody", "ambiguous") for n in s.get("notes", [])]
    det_pitches = [n["midi"] for n in det_notes]
    mel = [s for s in segs if s["kind"] == "melody"]
    r: dict = {"wav": e["wav"], "kind": e.get("kind", "note")}

    # note-count ±tol (always available)
    gt_count = e.get("verified_notes", e.get("intended_notes"))
    if gt_count is not None:
        r["note_count_ok"] = int(abs(len(det_notes) - int(gt_count)) <= int(e.get("tol", 1)))
        r["det_notes"], r["gt_notes_count"] = len(det_notes), int(gt_count)
    # transposition-invariant pitch-sequence F1 (needs gt_pitches)
    if e.get("gt_pitches"):
        r["pitch_seq_f1"] = pitch_prf(det_pitches, list(e["gt_pitches"]))["f1"]
    # full note onset+pitch F1 (needs gt_notes events)
    if e.get("gt_notes"):
        est = [{"midi": n["midi"], "t0": n["t0"], "dur": n["dur"]} for n in det_notes]
        r["note_f1"] = metrics.note_prf(e["gt_notes"], est)["f1"]
    # segmentation F1 (needs gt_segments)
    if e.get("gt_segments"):
        est = [{"kind": s["kind"], "t0": s["t0"], "t1": s["t1"]} for s in segs]
        r["segment_f1"] = metrics.segment_prf(e["gt_segments"], est)["f1"]
    # containment + dual-decode lyric (sung / mixed clips)
    if e.get("kind") in ("sung", "mixed") or e.get("lyric"):
        r["contained"] = int(not any(s.get("text") for s in mel))
        if e.get("lyric"):
            from .. import probe
            pr = probe.transcribe(audio, sr)  # whole-clip ASR with word timing
            txt = ""
            for s in segs:
                if s["kind"] == "melody" and s.get("notes"):
                    cand = dual_decode.candidate_from_probe(pr, s["t0"], s["t1"])
                    if cand and dual_decode.judge_sung_lyric(cand, len(s["notes"]), key=key):
                        txt = cand
                        break
            r["lyric_recall"] = dual_decode.word_recall(txt, e["lyric"])
            r["lyric_wer"] = dual_decode.wer(txt, e["lyric"])
            r["lyric_baseline_recall"] = 0.0  # containment drops text by construction
    return r


def _agg_rate(rows: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    k, n = int(sum(vals)), len(vals)
    mean, lo, hi = metrics.wilson_ci(k, n)
    return {"acc": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": n}


def _agg_mean(rows: list[dict], key: str) -> dict | None:
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    mean, lo, hi = metrics.bootstrap_ci(vals)
    return {"mean": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(vals)}


RATE_METRICS = ["note_count_ok", "contained"]
MEAN_METRICS = ["pitch_seq_f1", "note_f1", "segment_f1", "lyric_recall"]


def _agg(rows: list[dict]) -> dict:
    out = {}
    for m in RATE_METRICS:
        a = _agg_rate(rows, m)
        if a:
            out[m] = a
    for m in MEAN_METRICS:
        a = _agg_mean(rows, m)
        if a:
            out[m] = a
    return out


def main() -> int:
    import os
    from .. import secrets
    secrets.load_env()
    key = os.environ.get("OPENAI_API_KEY")

    speakers = []
    for src in SOURCES:
        if src.exists():
            for d in sorted(p for p in src.iterdir() if p.is_dir()):
                ents = _entries(d)
                if ents:
                    speakers.append((d.name, src.name, [score_clip(d / e["wav"], e, key) for e in ents]))

    if not speakers:
        print("no real-voice clips found under fixtures/human/{voices,sung}/")
        return 0

    held_out_rows = [r for name, _, rows in speakers if name not in OPERATOR for r in rows]
    per_speaker = {f"{name} ({src})": _agg(rows) for name, src, rows in speakers}
    pooled = _agg(held_out_rows)
    n_voices = len({name for name, _, _ in speakers if name not in OPERATOR})

    all_metrics = RATE_METRICS + MEAN_METRICS
    present = {m for agg in per_speaker.values() for m in agg}
    pending = [m for m in all_metrics if m not in present]

    print(f"\n=== Held-out real-voice benchmark — {n_voices} held-out voice(s) ===")
    print(f"{'metric':16}{'held-out pooled (95% CI)':>34}{'n':>5}")
    for m in all_metrics:
        a = pooled.get(m)
        if not a:
            continue
        v = a.get("acc", a.get("mean"))
        print(f"{m:16}{f'{v:.3f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]':>34}{a['n']:>5}")
    if pending:
        print(f"\nPENDING ground truth (add to manifests to light up): {', '.join(pending)}")
    print("\nper speaker:")
    for sp, agg in per_speaker.items():
        keys = ", ".join(f"{m}={agg[m].get('acc', agg[m].get('mean'))}" for m in all_metrics if m in agg)
        print(f"  {sp:18} {keys}")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rd = RUNS / f"{now}-voicebench"
    rd.mkdir(parents=True, exist_ok=True)
    out = {"phase": "real-voice-benchmark", "n_held_out_voices": n_voices,
           "held_out_pooled": pooled, "per_speaker": per_speaker, "pending_metrics": pending,
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()),
           "per_clip": {f"{name}": rows for name, _, rows in speakers}}
    (rd / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {rd.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
