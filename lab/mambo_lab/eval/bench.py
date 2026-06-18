"""MamboBench — the one-command benchmark.

Make the eval runnable by strangers:
  mambobench eval --predictions DIR    # score any system's UIR JSONs vs ground truth
  mambobench baselines [--limit N]     # run the B1/B3/B4/B5 baselines on the fixtures

A "system" is anything that, for each fixture WAV, emits a `mambo.utterance.v1`
JSON named ``<utterance_id>.uir.json`` into a predictions directory. MamboBench
scores it on the PAPER §6 metric suite and prints a leaderboard-style table.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from .. import fuse, ir, probe
from . import metrics

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "fixtures" / "synthetic"


# --------------------------------------------------------------------------- #
# Scoring a UIR prediction against ground truth (the §6 suite).
# --------------------------------------------------------------------------- #


def _hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, float) - 69.0) / 12.0)


def score_one(gold: dict, pred: dict) -> dict:
    gsegs, psegs = gold.get("segments", []), pred.get("segments", [])
    _, _, seg_f1, berr = metrics.segment_prf(gsegs, psegs)
    # note F1 over all GT melody notes vs all predicted melody notes (global time)
    gnotes = [n for s in gsegs if s["kind"] == "melody" for n in s.get("notes", [])]
    pnotes = [n for s in psegs if s["kind"] in ("melody", "ambiguous") for n in s.get("notes", [])]
    _, _, note_f1 = metrics.note_prf(gnotes, pnotes) if (gnotes or pnotes) else (1.0, 1.0, 1.0)
    halluc = metrics.hallucination_on_melody(gsegs, psegs)
    # key top-2 over matched melody segments (by index)
    key_ok = key_n = 0
    for gs in gsegs:
        if gs["kind"] == "melody" and gs.get("analysis", {}).get("n_notes", 0) >= 5:
            key_n += 1
            gk = (gs["analysis"].get("key_candidates") or [{}])[0].get("key")
            # nearest predicted melody segment in time
            best = _nearest_melody(psegs, gs)
            cand = (best or {}).get("analysis", {}).get("key_candidates", []) if best else []
            if gk and metrics.key_in_topk(gk, cand):
                key_ok += 1
    return {"seg_f1": seg_f1, "note_f1": note_f1, "halluc": int(halluc),
            "boundary_ms": float(np.median(berr) * 1000) if berr else 0.0,
            "key_ok": key_ok, "key_n": key_n}


def _nearest_melody(segs, gseg) -> Optional[dict]:
    mels = [s for s in segs if s["kind"] in ("melody", "ambiguous")]
    if not mels:
        return None
    return min(mels, key=lambda s: abs(s["t0"] - gseg["t0"]))


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    kn = sum(s["key_n"] for s in scores)
    return {
        "n": len(scores),
        "segment_f1": float(np.mean([s["seg_f1"] for s in scores])),
        "note_f1": float(np.mean([s["note_f1"] for s in scores])),
        "boundary_ms": float(np.median([s["boundary_ms"] for s in scores])),
        "hallucination": float(np.mean([s["halluc"] for s in scores])),
        "key_top2": (sum(s["key_ok"] for s in scores) / kn) if kn else None,
    }


# --------------------------------------------------------------------------- #
# Loading fixtures + predictions.
# --------------------------------------------------------------------------- #


def _truth(uid: str, fixtures: Path) -> dict:
    return json.load(open(fixtures / "truth" / f"{uid}.uir.json"))


def eval_predictions(pred_dir: str, fixtures: Path) -> dict:
    scores = []
    for pp in sorted(glob.glob(str(Path(pred_dir) / "*.uir.json"))):
        uid = Path(pp).name[: -len(".uir.json")]
        tp = fixtures / "truth" / f"{uid}.uir.json"
        if not tp.exists():
            continue
        try:
            pred = json.load(open(pp))
        except Exception:
            pred = {"segments": []}
        scores.append(score_one(json.load(open(tp)), pred))
    return aggregate(scores)


# --------------------------------------------------------------------------- #
# Baselines (the PAPER §6 B-table).
# --------------------------------------------------------------------------- #


def baseline_whisper_only(audio, sr) -> dict:
    """B1: transcribe the whole utterance, treat it ALL as one speech span
    (the status quo). Quantifies hallucination damage + missing melody."""
    pr = probe.transcribe(audio, sr)
    seg = {"kind": "speech", "t0": 0.0, "t1": round(len(audio) / sr, 4),
           "confidence": 1.0, "text": pr.text}
    return {"schema": "mambo.utterance.v1", "utterance_id": "b1",
            "audio": {"sample_rate": sr, "duration_s": len(audio) / sr}, "segments": [seg]}


def run_baselines(fixtures: Path, limit: int = 0) -> dict[str, dict]:
    from collections import defaultdict

    rows = [json.loads(l) for l in (fixtures / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        # Stratify across templates so the table is representative, not the first
        # N (which are all one template and trivially scored).
        by_t = defaultdict(list)
        for r in rows:
            by_t["_".join(r["utterance_id"].split("_")[1:-2])].append(r)
        per = max(1, limit // max(1, len(by_t)))
        rows = [r for t in sorted(by_t) for r in by_t[t][:per]]
    systems = {"B1 whisper-only": [], "B3 acoustic": [], "B4 linguistic": [], "B5 joint (ours)": []}
    for r in rows:
        uid = r["utterance_id"]
        gold = _truth(uid, fixtures)
        audio, sr = sf.read(str(fixtures / r["wav"]), dtype="float32")
        f0 = None  # pipeline computes its own
        systems["B1 whisper-only"].append(score_one(gold, baseline_whisper_only(audio, sr)))
        for name, strat in [("B3 acoustic", "acoustic"), ("B4 linguistic", "linguistic"), ("B5 joint (ours)", "joint")]:
            pred = fuse.file_to_uir(audio, sr, strategy=strat, utterance_id=uid).to_dict()
            systems[name].append(score_one(gold, pred))
    return {name: aggregate(s) for name, s in systems.items()}


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def _print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n### {title}\n")
    print(f"{'system':22} {'seg F1':>7} {'note F1':>8} {'bound(ms)':>10} {'halluc':>7} {'key@2':>6}")
    print("-" * 66)
    for name, a in rows.items():
        if not a:
            continue
        k = "—" if a.get("key_top2") is None else f"{a['key_top2']:.2f}"
        print(f"{name:22} {a['segment_f1']:7.3f} {a['note_f1']:8.3f} "
              f"{a['boundary_ms']:10.0f} {a['hallucination']*100:6.1f}% {k:>6}")


def _write_baselines_run(rows: dict) -> Path:
    """Persist the B-table to a committed run (O1: the headline baselines must be
    reproducible from a `runs/` artifact, not a print-only function)."""
    import datetime as _dt
    import subprocess as _sp

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "runs" / f"{now}-baselines"
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = _sp.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    porcelain = _sp.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.splitlines()
    dirty = any(line and not line.startswith("??") for line in porcelain)
    (run_dir / "results.json").write_text(json.dumps(
        {"bench": "MamboBench baselines (synthetic)", "timestamp": now, "commit": commit,
         "dirty": dirty, "systems": rows}, indent=2) + "\n")
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mambobench", description="MamboBench — mixed-vocal-utterance parsing benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("eval", help="score a system's UIR predictions vs ground truth")
    pe.add_argument("--predictions", required=True, help="dir of <utterance_id>.uir.json")
    pe.add_argument("--fixtures", default=str(FIXTURES))
    pb = sub.add_parser("baselines", help="run the B1/B3/B4/B5 baselines on the fixtures")
    pb.add_argument("--limit", type=int, default=0)
    pb.add_argument("--fixtures", default=str(FIXTURES))
    args = ap.parse_args(argv)

    if args.cmd == "eval":
        agg = eval_predictions(args.predictions, Path(args.fixtures))
        _print_table(f"MamboBench eval — {args.predictions}", {"your system": agg})
        return 0
    if args.cmd == "baselines":
        rows = run_baselines(Path(args.fixtures), limit=args.limit)
        _print_table("MamboBench baselines (synthetic)", rows)
        run_dir = _write_baselines_run(rows)
        print(f"\nwrote {run_dir.relative_to(REPO)}/results.json")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
