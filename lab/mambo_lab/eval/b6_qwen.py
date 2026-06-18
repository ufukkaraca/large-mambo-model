"""B6 (valid route) — score base Qwen2-Audio's hum→notes against Mambo + GT.

Companion to `b6_omni.py`. The OpenRouter free omni route was *inconclusive* — its
audio never ingested (audio_tokens=0; see verifications.md). This scores a
**confirmed-audio** route instead: base Qwen2-Audio-7B run on GPU via the R3 Modal
app (`finetune/modal_app.py::b6_transcribe`), which definitionally ingests the
audio. So its number is an interpretable on-task answer to "can a current omni
model hear pitch?" — scored against the *same* ground truth and with the *same*
transposition-invariant metric the modular pipeline (Mambo) is scored on.

Two steps:
    cd lab && uv run modal run ../finetune/modal_app.py --action b6   # GPU transcribe
    cd lab && uv run python -m mambo_lab.eval.b6_qwen                 # score (this file)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from . import gate
from .b6_omni import parse_notes, pitch_seq_acc
from .. import melody

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "finetune" / "b6_qwen_raw.json"
MODEL = "Qwen/Qwen2-Audio-7B-Instruct (base, no LoRA)"


def run() -> dict:
    if not RAW.exists():
        raise SystemExit(f"no transcriptions at {RAW} — run "
                         "`uv run modal run ../finetune/modal_app.py --action b6` first")
    by_uid = {r["uid"]: r["text"] for r in json.loads(RAW.read_text())}
    rows = gate.load_fixtures()
    for r in rows:
        r.setdefault("uid", r.get("utterance_id"))
    rows_by_uid = {r["uid"]: r for r in rows}

    per, omni_acc, mambo_acc, pairs = [], [], [], []
    for uid, text in by_uid.items():
        r = rows_by_uid.get(uid)
        if r is None:
            continue
        d = gate._truth(uid)
        gt = [n["midi"] for s in d["segments"] if s["kind"] == "melody" for n in s.get("notes", [])]
        a, sr = sf.read(str(gate.FIXTURES / r["wav"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        mambo = [n.midi for n in melody.segment_notes(a, melody.track_f0(a, sr))]
        onotes = parse_notes(text)
        m_acc, o_acc = pitch_seq_acc(mambo, gt), pitch_seq_acc(onotes, gt)
        mambo_acc.append(m_acc)
        omni_acc.append(o_acc)
        if onotes and gt:
            pairs.append((float(np.median(gt)), float(np.median(onotes))))
        per.append({"uid": uid, "gt_n": len(gt), "omni_n": len(onotes),
                    "omni_pitch_acc": round(o_acc, 3), "mambo_pitch_acc": round(m_acc, 3),
                    "omni_median": float(np.median(onotes)) if onotes else None,
                    "gt_median": float(np.median(gt)) if gt else None,
                    "omni_raw": text.replace("\n", " ")[:160]})

    # pitch-tracking r is a QUALITY signal here, not a validity gate: a real audio
    # model ingests the audio by construction, so the number is interpretable
    # regardless — unlike the OpenRouter route where r was the only ingestion proxy.
    r_corr = None
    if len(pairs) >= 3:
        g, p = zip(*pairs)
        if np.std(g) > 0 and np.std(p) > 0:
            r_corr = round(float(np.corrcoef(g, p)[0, 1]), 2)
    return {
        "model": MODEL, "route": "base Qwen2-Audio via R3 Modal app (confirmed-audio)",
        "n_clips": len(per), "per_clip": per,
        "mambo_pitch_acc_mean": round(float(np.mean(mambo_acc)), 3) if mambo_acc else None,
        "omni_pitch_acc_mean": round(float(np.mean(omni_acc)), 3) if omni_acc else None,
        "pitch_corr_r": r_corr, "pitch_corr_n": len(pairs),
        "audio_ingestion": "confirmed by construction (local GPU inference on a true audio LLM)",
    }


def main() -> int:
    r = run()
    print(f"\n=== B6 (valid route) · {r['model']} vs Mambo · {r['n_clips']} pure hums ===")
    print(f"  Mambo pitch-seq accuracy: {r['mambo_pitch_acc_mean']}")
    print(f"  Qwen2-Audio  pitch-seq accuracy: {r['omni_pitch_acc_mean']}  "
          f"(pitch-tracking r={r['pitch_corr_r']}, n={r['pitch_corr_n']})")
    print(f"  audio ingestion: {r['audio_ingestion']}")
    delta = (r["mambo_pitch_acc_mean"] or 0) - (r["omni_pitch_acc_mean"] or 0)
    print(f"  → Mambo beats base Qwen2-Audio by {delta:+.3f} on its own task"
          if delta > 0 else "  → see per-clip detail")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "runs" / f"{now}-b6-qwen"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"\n  wrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
