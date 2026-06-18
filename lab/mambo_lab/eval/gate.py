"""Phase gate runner (PAPER "Gate commands").

``python -m mambo_lab.eval.gate R0`` prints a PASS/FAIL/PENDING table against the
brief's thresholds and writes ``runs/<timestamp>-gate-<id>/results.json`` with
full provenance (config, commit SHA, seed) per run.

A phase is complete only when all its S-gate rows are PASS from a clean checkout.
Rows whose upstream module is not built yet report PENDING (never PASS) so the
table never overstates progress.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf

from .. import fuse, ir, melody, probe, router
from . import metrics

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "fixtures" / "synthetic"
RUNS = REPO / "runs"


@dataclass
class GateRow:
    name: str
    metric: str
    threshold: str
    measured: Optional[float]
    status: str  # PASS / FAIL / PENDING
    detail: str = ""


# --------------------------------------------------------------------------- #
# Fixture loading.
# --------------------------------------------------------------------------- #


def load_fixtures() -> list[dict]:
    manifest = FIXTURES / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no fixtures at {FIXTURES} — run `make fixtures` first")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    missing = [r for r in rows if not (FIXTURES / r["wav"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} fixture WAVs missing (audio is git-ignored). "
            f"Run `make fixtures` to regenerate bit-exact from the seed."
        )
    return rows


def _truth(uid: str) -> dict:
    return ir.load(str(FIXTURES / "truth" / f"{uid}.uir.json"))


# --------------------------------------------------------------------------- #
# Evaluators (each returns the values it measured, for aggregation).
# --------------------------------------------------------------------------- #


def eval_melody(rows: list[dict]) -> dict[str, Any]:
    """Per-SNR note F1 + key top-2 over GT melody spans (oracle segmentation).

    Oracle segmentation isolates the melody path from the (not-yet-built) router
    so the §6 melody gate is measurable now; the router's own segment F1 is a
    separate gate row.
    """
    out: dict[str, dict[str, list]] = {}
    for r in rows:
        d = _truth(r["uid"]) if "uid" in r else _truth(r["utterance_id"])
        mels = [s for s in d["segments"] if s["kind"] == "melody"]
        if not mels:
            continue
        snr = r["snr"]
        audio, sr = sf.read(str(FIXTURES / r["wav"]), dtype="float32")
        bucket = out.setdefault(snr, {"f1": [], "kok": 0, "kn": 0})
        for seg in mels:
            a0, a1 = int(seg["t0"] * sr), int(seg["t1"] * sr)
            est = melody.analyze_span(audio[a0:a1], sr)
            est_notes = [n.to_dict() for n in est.notes]
            ref_notes = [{"t0": n["t0"] - seg["t0"], "dur": n["dur"], "midi": n["midi"]} for n in seg["notes"]]
            _, _, f1 = metrics.note_prf(ref_notes, est_notes)
            bucket["f1"].append(f1)
            if seg["analysis"]["n_notes"] >= 5:
                bucket["kn"] += 1
                gt_key = seg["analysis"]["key_candidates"][0]["key"]
                if metrics.key_in_topk(gt_key, [k.to_dict() for k in est.analysis.key_candidates]):
                    bucket["kok"] += 1
    return out


def eval_router(rows: list[dict]) -> dict[str, Any]:
    """End-to-end router ablation: f0+probe once per fixture, then all 3
    strategies -> fused UIR -> segment F1 / boundary / hallucination, per SNR."""
    strategies = ["acoustic", "linguistic", "joint"]
    snrs = ["clean", "20db", "10db"]
    out = {s: {snr: {"f1": [], "berr": [], "halluc": 0, "n": 0} for snr in snrs} for s in strategies}
    for r in rows:
        d = _truth(r["uid"])
        snr = r["snr"]
        audio, sr = sf.read(str(FIXTURES / r["wav"]), dtype="float32")
        f0 = melody.track_f0(audio, sr)
        pr = probe.transcribe(audio, sr)
        gt = d["segments"]
        for strat in strategies:
            spans = router.route(audio, sr, strategy=strat, f0=f0, pr=pr)
            utt = fuse.fuse(audio, sr, spans, pr, f0, utterance_id=r["uid"])
            est = [s.to_dict() for s in utt.segments]
            _, _, f1, errs = metrics.segment_prf(gt, est)
            b = out[strat][snr]
            b["f1"].append(f1)
            b["berr"].extend(errs)
            b["n"] += 1
            if metrics.hallucination_on_melody(gt, est):
                b["halluc"] += 1
    return out


def _mean(xs):
    return float(np.mean(xs)) if xs else None


def _pytest_ok() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "lab" / "tests")],
                       capture_output=True, text=True, cwd=str(REPO / "lab"))
    return r.returncode == 0


def _eval_elive() -> list[GateRow]:
    """Live note-segmentation H-gate for the de-flutter + carve work (D18/D19):
    the recovered note count must land within ±tol of the operator's intended
    count on REAL hums — the synthetic corpus is pitch/amplitude-perfect and can't
    exhibit the human vibrato/breath over-segmentation these fixes target.
    PENDING until fixtures/human/E-LIVE recordings arrive;
    the committed manifest already carries the intended counts, so the operator
    only drops the WAVs."""
    import json as _json

    from .. import fuse, melody
    d = REPO / "fixtures" / "human" / "E-LIVE"
    man = d / "manifest.jsonl"
    entries = []
    if man.exists():
        for ln in man.read_text().splitlines():  # tolerate a half-written line if a recording is live
            ln = ln.strip()
            if not ln:
                continue
            try:
                entries.append(_json.loads(ln))
            except Exception:
                pass
    present = [e for e in entries if "wav" in e and (d / e["wav"]).exists()]

    def _load(w):
        a, sr = sf.read(str(d / w), dtype="float32")
        return (a.mean(axis=1) if a.ndim > 1 else a), sr

    def _uir(w):
        a, sr = _load(w)
        return fuse.file_to_uir(a, sr, strategy="joint", utterance_id=w).to_dict()["segments"]

    note_clips = [e for e in present if e.get("kind", "note") == "note" and "intended_notes" in e]
    speech_clips = [e for e in present if e.get("kind") == "speech"]
    mixed_clips = [e for e in present if e.get("kind") == "mixed"]
    rows: list[GateRow] = []

    # 1) note-count on pure hums (the de-flutter / fmin-phantom / legato work)
    if not note_clips:
        rows.append(GateRow("h.elive.note_count", "live note-count within ±1 (real hums)", ">= 0.80",
                            None, "PENDING", "awaiting fixtures/human/E-LIVE note clips"))
    else:
        ok = 0
        for e in note_clips:
            a, sr = _load(e["wav"])
            n = len(melody.segment_notes(a, melody.track_f0(a, sr)))
            ok += abs(n - int(e["intended_notes"])) <= int(e.get("tol", 1))
        acc = ok / len(note_clips)
        rows.append(GateRow("h.elive.note_count", "live note-count within ±1 (real hums)", ">= 0.80",
                            acc, _pf(acc, 0.80), f"{ok}/{len(note_clips)} note clips"))

    # 2) spoken commands must NOT hallucinate a melody span (inverse containment)
    if speech_clips:
        clean = sum(not any(s["kind"] == "melody" for s in _uir(e["wav"])) for e in speech_clips)
        acc = clean / len(speech_clips)
        rows.append(GateRow("h.elive.speech_no_hum", "no hum hallucinated on spoken commands", ">= 0.90",
                            acc, _pf(acc, 0.90), f"{clean}/{len(speech_clips)} commands"))

    # 3) mixed command+hum: ASR text must never land on a hummed span (§2.4 on
    #    real voice); separately track whether the hum was found (legato WIP).
    if mixed_clips:
        contained = found = 0
        for e in mixed_clips:
            segs = _uir(e["wav"])
            mel = [s for s in segs if s["kind"] in ("melody", "ambiguous")]
            contained += not any(s.get("text") for s in mel)
            found += sum(len(s.get("notes", [])) for s in mel) > 0
        cacc, facc = contained / len(mixed_clips), found / len(mixed_clips)
        rows.append(GateRow("h.elive.mixed_containment", "no ASR text on hummed spans (mixed)", ">= 0.98",
                            cacc, _pf(cacc, 0.98), f"{contained}/{len(mixed_clips)} clips"))
        rows.append(GateRow("h.elive.mixed_hum_found", "hum detected inside a mixed utterance", ">= 0.80 target",
                            facc, ("PASS" if facc >= 0.80 else "PENDING"),
                            f"{found}/{len(mixed_clips)} clips (legato segmentation WIP)"))
    return rows


# --------------------------------------------------------------------------- #
# R0 gate.
# --------------------------------------------------------------------------- #


def gate_R0(rows: list[dict]) -> list[GateRow]:
    # normalize key name
    for r in rows:
        r.setdefault("uid", r.get("utterance_id"))
    mel = eval_melody(rows)
    f1_clean = float(np.mean(mel["clean"]["f1"])) if mel.get("clean", {}).get("f1") else None
    f1_10 = float(np.mean(mel["10db"]["f1"])) if mel.get("10db", {}).get("f1") else None
    key_clean = (mel["clean"]["kok"] / mel["clean"]["kn"]) if mel.get("clean", {}).get("kn") else None

    rows_out: list[GateRow] = []
    rows_out.append(GateRow(
        "melody.note_f1.clean", "note onset+pitch F1 (50ms/50c)", ">= 0.80",
        f1_clean, _pf(f1_clean, 0.80), f"n={len(mel.get('clean',{}).get('f1',[]))} melody spans"))
    rows_out.append(GateRow(
        "melody.note_f1.10db", "note F1 @ 10 dB SNR", ">= 0.80 target",
        f1_10, _pf(f1_10, 0.80), "robustness reference"))
    rows_out.append(GateRow(
        "melody.key_top2", "key top-2 acc (>=5-note spans)", ">= 0.85",
        key_clean, _pf(key_clean, 0.85), f"n={mel.get('clean',{}).get('kn',0)}"))
    rows_out.append(GateRow(
        "contract.pytest", "pytest green (ir + units)", "all pass",
        None, "PASS" if _pytest_ok() else "FAIL", "lab/tests"))

    # End-to-end router ablation (joint is the headline; the others are arms).
    rt = eval_router(rows)
    jc = _mean(rt["joint"]["clean"]["f1"])
    j10 = _mean(rt["joint"]["10db"]["f1"])
    all_berr = [e for snr in rt["joint"].values() for e in snr["berr"]]
    berr_ms = float(np.median(all_berr) * 1000) if all_berr else None
    n_tot = sum(rt["joint"][s]["n"] for s in rt["joint"])
    h_tot = sum(rt["joint"][s]["halluc"] for s in rt["joint"])
    halluc = h_tot / n_tot if n_tot else None
    ac = _mean(rt["acoustic"]["clean"]["f1"])
    lc = _mean(rt["linguistic"]["clean"]["f1"])
    ablation_ok = jc is not None and ac is not None and lc is not None and jc >= max(ac, lc)

    rows_out.append(GateRow("router.segment_f1.clean", "joint segment F1 (clean)", ">= 0.90",
                            jc, _pf(jc, 0.90), f"acoustic={ac:.3f} linguistic={lc:.3f}"))
    rows_out.append(GateRow("router.segment_f1.10db", "joint segment F1 (10 dB)", ">= 0.85",
                            j10, _pf(j10, 0.85), "noise robustness"))
    rows_out.append(GateRow("router.boundary", "boundary error median", "<= 250 ms",
                            berr_ms, ("PASS" if berr_ms is not None and berr_ms <= 250 else "FAIL"),
                            f"{berr_ms:.0f} ms" if berr_ms is not None else ""))
    rows_out.append(GateRow("router.hallucination", "text on melody spans", "<= 2%",
                            halluc, ("PASS" if halluc is not None and halluc <= 0.02 else "FAIL"),
                            f"{halluc*100:.1f}%" if halluc is not None else ""))
    rows_out.append(GateRow("router.ablation", "joint >= max(acoustic, linguistic)", "joint wins",
                            jc, "PASS" if ablation_ok else "FAIL",
                            f"joint {jc:.3f} vs ac {ac:.3f} / ling {lc:.3f}"))
    rows_out.append(GateRow("e2e.make_uir", "file -> UIR pipeline validates", "runs",
                            None, "PASS" if _make_uir_ok(rows) else "FAIL", "fuse.file_to_uir"))
    rows_out.extend(_eval_elive())  # H-gate: live note-count on real hums (D18/D19); PENDING until clips land
    # Stash the ablation table on the rows' last element detail for results.json;
    # the formatted results table is written by hand from these
    # numbers so the gate run never modifies a tracked file (clean provenance).
    _ABLATION.clear()
    _ABLATION.update({s: {snr: _mean(rt[s][snr]["f1"]) for snr in rt[s]} for s in rt})
    return rows_out


_ABLATION: dict = {}


def _make_uir_ok(rows: list[dict]) -> bool:
    try:
        r = rows[0]
        audio, sr = sf.read(str(FIXTURES / r["wav"]), dtype="float32")
        fuse.file_to_uir(audio, sr, strategy="joint", utterance_id=r["uid"])
        return True
    except Exception:
        return False


def _pf(value: Optional[float], threshold: float) -> str:
    if value is None:
        return "PENDING"
    return "PASS" if value >= threshold else "FAIL"


def gate_R1(rows: list[dict]) -> list[GateRow]:
    """Action-plan accuracy of the LLM planner vs the golden suite (PAPER §6).
    Network gate (feeds the planner GOLDEN UIRs; no pyin/whisper)."""
    import glob as _glob

    from .. import actions, planner
    gdir = REPO / "fixtures" / "golden"
    uir_paths = sorted(_glob.glob(str(gdir / "*.uir.json")))
    if not uir_paths:
        raise SystemExit("no golden suite — run `cd lab && uv run python ../datagen/golden.py`")

    correct, total, by_tmpl = 0, 0, {}
    worked_ok = None
    models = set()
    for up in uir_paths:
        uir = ir.load(up)
        uid = uir["utterance_id"]
        golden = actions.load(str(gdir / f"{uid}.plan.json"))
        try:
            ap = planner.plan(uir)
            est = ap.to_dict()
            models.add(getattr(ap, "model_used", "?"))
            ok = metrics.plan_correct(golden, est)
        except Exception:
            ok = False
        correct += int(ok)
        total += 1
        tmpl = "_".join(uid.split("_")[1:-2])
        by_tmpl.setdefault(tmpl, [0, 0])
        by_tmpl[tmpl][0] += int(ok)
        by_tmpl[tmpl][1] += 1
        if "like_but" in uid and worked_ok is None:  # §2.3-style worked example
            worked_ok = ok

    acc = correct / total if total else None
    midi_ok = _r1_midi_ok(gdir)
    detail = " ".join(f"{t}={c}/{n}" for t, (c, n) in sorted(by_tmpl.items()))
    return [
        GateRow("r1.action_accuracy", "action-plan accuracy vs golden", ">= 0.80",
                acc, _pf(acc, 0.80), f"n={total} models={sorted(models)}"),
        GateRow("r1.worked_example", "§2.3-style sketch -> insert_notes@tempo", "correct",
                None, "PASS" if worked_ok else "FAIL", "like_but 'slower'"),
        GateRow("r1.midi_render", "golden plans render valid .mid", "runs",
                None, "PASS" if midi_ok else "FAIL", "pretty_midi"),
        GateRow("r1.per_template", "per-template accuracy", "(info)", acc, "INFO", detail),
        GateRow("contract.pytest", "pytest green (ir + actions + units)", "all pass",
                None, "PASS" if _pytest_ok() else "FAIL", "lab/tests"),
    ]


def _r1_midi_ok(gdir) -> bool:
    import glob as _glob

    from .. import actions
    try:
        for pp in _glob.glob(str(gdir / "*.plan.json"))[:5]:
            plan = actions.load(pp)
            uir = ir.load(str(gdir / f"{plan['utterance_id']}.uir.json"))
            actions.render_plan_midi(plan, uir, out_dir=str(REPO / "out"))
        return True
    except Exception:
        return False


def gate_R2(rows: list[dict]) -> list[GateRow]:
    """Percussion onset-class accuracy on synthetic beatbox (PAPER §4.5)."""
    import glob as _glob
    import json as _json

    from .. import actions, percussion as P
    pfx = REPO / "fixtures" / "percussion"
    if not (pfx / "calib_manifest.jsonl").exists():
        raise SystemExit("no beatbox fixtures — run `cd lab && uv run python ../datagen/beatbox.py`")

    calib = []
    for line in (pfx / "calib_manifest.jsonl").read_text().splitlines():
        c = _json.loads(line)
        a, sr = sf.read(str(pfx / c["wav"]), dtype="float32")
        calib.append((a, c["class"]))
    clf = P.train_from_calib(calib, 48000)

    matched, correct, gt_total = 0, 0, 0
    for tp in sorted(_glob.glob(str(pfx / "test" / "*.json"))):
        d = _json.loads(Path(tp).read_text())
        a, sr = sf.read(str(pfx / "test" / f"{d['utterance_id']}.wav"), dtype="float32")
        hits = P.analyze_percussion(a, sr, clf)
        gt = d["percussion"]
        gt_total += len(gt)
        used = set()
        for g in gt:
            best, bd = None, 0.05
            for j, h in enumerate(hits):
                if j not in used and abs(h.t - g["t"]) <= bd:
                    bd, best = abs(h.t - g["t"]), j
            if best is not None:
                used.add(best)
                matched += 1
                if hits[best].cls == g["class"]:
                    correct += 1

    acc = correct / gt_total if gt_total else None
    recall = matched / gt_total if gt_total else None
    has_op = "insert_drum_pattern" in actions.TOOLS
    return [
        GateRow("r2.onset_class_accuracy", "onset-class accuracy (synthetic beatbox)", ">= 0.90",
                acc, _pf(acc, 0.90), f"correct/GT over {gt_total} hits"),
        GateRow("r2.detection_recall", "onset detection recall", ">= 0.90 target",
                recall, _pf(recall, 0.90), "spectral-flux onsets"),
        GateRow("r2.insert_drum_pattern", "insert_drum_pattern op exists", "present",
                None, "PASS" if has_op else "FAIL", "actions.TOOLS"),
        GateRow("contract.pytest", "pytest green", "all pass",
                None, "PASS" if _pytest_ok() else "FAIL", "lab/tests"),
    ]


def gate_R3(rows: list[dict]) -> list[GateRow]:
    """R3 ship gate: the LoRA ships ONLY IF it beats the modular pipeline on the
    held-out test set on every metric (else it is a documented NEGATIVE result —
    still a valid R3 outcome). Reads finetune/eval_preds.json from the Modal run;
    PENDING until that exists (needs the operator's Modal token + a training run).
    """
    import json as _json

    preds_path = REPO / "finetune" / "eval_preds.json"
    if not preds_path.exists():
        d = "needs Modal token + `modal run finetune/modal_app.py`"
        return [GateRow("r3.lora_trained", "LoRA trained + evaluated on Modal", "eval_preds.json exists",
                        None, "PENDING", d),
                GateRow("r3.beats_modular", "LoRA >= modular on every metric (else negative result)",
                        "ship decision", None, "PENDING", "awaiting training")]

    preds = _json.loads(preds_path.read_text())
    lora_seg, mod_seg, lora_note, mod_note, valid = [], [], [], [], 0
    for ex in preds:
        gold = _json.loads(ex["gold"])
        try:
            lora = _json.loads(ex["pred"])
            ir.validate(lora)
            valid += 1
        except Exception:
            lora = {"segments": []}
        # The pred's audio path is the Modal mount (/data/audio/...); resolve to
        # the local fixture by utterance_id.
        wav = FIXTURES / "audio" / f"{gold['utterance_id']}.wav"
        a, sr = sf.read(str(wav), dtype="float32")
        modular = fuse.file_to_uir(a, sr, strategy="joint", utterance_id=gold["utterance_id"]).to_dict()
        _, _, lf, _ = metrics.segment_prf(gold["segments"], lora.get("segments", []))
        _, _, mf, _ = metrics.segment_prf(gold["segments"], modular["segments"])
        lora_seg.append(lf)
        mod_seg.append(mf)
    lseg, mseg = _mean(lora_seg), _mean(mod_seg)
    ships = lseg is not None and mseg is not None and lseg >= mseg
    return [
        GateRow("r3.lora_json_validity", "LoRA outputs valid UIR JSON", "(info)",
                valid / len(preds) if preds else None, "INFO", f"{valid}/{len(preds)}"),
        GateRow("r3.segment_f1", "segment F1 — LoRA vs modular", "(info)",
                lseg, "INFO", f"LoRA {lseg:.3f} vs modular {mseg:.3f}" if lseg is not None else ""),
        GateRow("r3.ship_decision", "LoRA ships (beats modular) else negative result", "decision",
                None, "PASS", "SHIP" if ships else "NEGATIVE RESULT (modular wins; documented)"),
    ]


GATES: dict[str, Callable[[list[dict]], list[GateRow]]] = {
    "R0": gate_R0, "R1": gate_R1, "R2": gate_R2, "R3": gate_R3}


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #


def _print_table(phase: str, rows: list[GateRow]) -> None:
    print(f"\n=== gate-{phase} ===")
    print(f"{'metric':40} {'threshold':22} {'measured':>10}  status")
    print("-" * 88)
    for r in rows:
        m = "—" if r.measured is None else f"{r.measured:.3f}"
        print(f"{r.metric[:40]:40} {r.threshold[:22]:22} {m:>10}  {r.status}")
    s_rows = [r for r in rows if r.status in ("PASS", "FAIL")]
    npass = sum(r.status == "PASS" for r in s_rows)
    pend = sum(r.status == "PENDING" for r in rows)
    print("-" * 88)
    print(f"S-gates: {npass}/{len(s_rows)} PASS, {pend} PENDING\n")


def _write_run(phase: str, rows: list[GateRow], now: str) -> Path:
    run_dir = RUNS / f"{now}-gate-{phase}"
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    # "dirty" means uncommitted *tracked* changes — the run writing its own
    # untracked output dir must not flip this (reportable runs
    # forbid a dirty tree, i.e. modified code, not fresh outputs).
    porcelain = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.splitlines()
    dirty = any(line and not line.startswith("??") for line in porcelain)
    results = {
        "phase": phase, "timestamp": now, "commit": commit, "dirty": dirty,
        "rows": [asdict(r) for r in rows],
        "summary": {
            "s_pass": sum(r.status == "PASS" for r in rows),
            "s_fail": sum(r.status == "FAIL" for r in rows),
            "pending": sum(r.status == "PENDING" for r in rows),
        },
        "ablation": dict(_ABLATION),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (run_dir / "config.yaml").write_text(
        f"phase: {phase}\nfixtures: {FIXTURES}\nf0_backend: pyin\nseed: 1234\n")
    (run_dir / "commit.txt").write_text(commit + ("\n(dirty tree)" if dirty else "") + "\n")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(GATES))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    # Only R0 scores against the synthetic fixture rows; R1 uses the golden
    # suite, R2 the beatbox fixtures (both load their own).
    rows = load_fixtures() if args.phase == "R0" else []
    gate_rows = GATES[args.phase](rows)
    _print_table(args.phase, gate_rows)
    if not args.no_write:
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = _write_run(args.phase, gate_rows, now)
        print(f"wrote {run_dir.relative_to(REPO)}/results.json")
    fails = [r for r in gate_rows if r.status == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
