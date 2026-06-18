"""B6, the honest version — does a FRONTIER omni model hear pitch on our task?

Fixes the methodology flaw an independent audit found in the original B6: the
recall-only metric conflated "mis-pitched" with "emitted too few notes", and one
under-prompted open 7B (Qwen2-Audio) was over-read as "LLMs can't hear pitch".

This run:
  * tests a FRONTIER hosted audio model (gpt-audio-1.5) with a best-guess prompt
    that defeats the refusal the default prompt triggers (audio ingestion confirmed
    by non-zero audio_tokens);
  * scores with `pitch_prf` (precision / recall / F1 + note-count error) so the two
    failure modes are SEPARATED — under-generation vs wrong pitch;
  * re-scores the committed open Qwen2-Audio transcriptions with the same honest
    metric, and the modular pipeline (Mambo) for reference.

Cost-bounded (12 short clips ≈ $0.05). Needs OPENAI_API_KEY in .env.

    cd lab && uv run python -m mambo_lab.eval.b6_frontier
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from . import gate
from .b6_omni import parse_notes, pitch_prf
from .. import melody, secrets

REPO = Path(__file__).resolve().parents[3]
MODEL = "gpt-audio-1.5"
PROMPT = ("You are a music transcription tool. The audio is a person humming one monophonic "
          "melody. Output ONLY your best-guess note sequence with octaves as "
          "'NOTES: C4, E4, ...'. Approximate is fine; never refuse.")


def _ask(b64: str, key: str) -> tuple[str, dict]:
    body = {"model": MODEL, "modalities": ["text"], "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return (r["choices"][0]["message"]["content"] or ""), r.get("usage", {})


def _agg(reports: list[dict]) -> dict:
    if not reports:
        return {}
    return {k: round(float(np.mean([r[k] for r in reports])), 3)
            for k in ("precision", "recall", "f1", "note_count_err")}


def main() -> int:
    secrets.load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (.env)")
    rows = gate.load_fixtures()
    for r in rows:
        r.setdefault("uid", r.get("utterance_id"))
    clips = [r for r in rows if r["snr"] == "clean" and "pure_hum" in r["uid"]][:12]
    qwen_raw = {}
    qf = REPO / "finetune" / "b6_qwen_raw.json"
    if qf.exists():
        qwen_raw = {x["uid"]: x["text"] for x in json.loads(qf.read_text())}

    mambo, front, qwen, per, cost, audio_tok = [], [], [], [], 0.0, 0
    for r in clips:
        gt = [n["midi"] for s in gate._truth(r["uid"])["segments"] if s["kind"] == "melody" for n in s.get("notes", [])]
        a, sr = sf.read(str(gate.FIXTURES / r["wav"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        m_notes = [n.midi for n in melody.segment_notes(a, melody.track_f0(a, sr))]
        mambo.append(pitch_prf(m_notes, gt))
        txt, u = _ask(base64.b64encode((gate.FIXTURES / r["wav"]).read_bytes()).decode(), key)
        f_notes = parse_notes(txt)
        front.append(pitch_prf(f_notes, gt))
        ai = (u.get("prompt_tokens_details", {}) or {}).get("audio_tokens", 0)
        audio_tok += ai
        cost += (u.get("prompt_tokens", 0) - ai) / 1e6 * 2.5 + ai / 1e6 * 40 + u.get("completion_tokens", 0) / 1e6 * 10
        row = {"uid": r["uid"], "gt_n": len(gt), "frontier_n": len(f_notes),
               "frontier": pitch_prf(f_notes, gt), "mambo": pitch_prf(m_notes, gt), "frontier_raw": txt[:100]}
        if r["uid"] in qwen_raw:
            qn = parse_notes(qwen_raw[r["uid"]])
            qwen.append(pitch_prf(qn, gt))
            row["qwen"] = pitch_prf(qn, gt)
        per.append(row)

    out = {"task": "B6 honest — frontier omni vs Mambo on hum→notes (precision/recall/F1/note-count)",
           "frontier_model": MODEL, "n_clips": len(clips), "audio_ingested": audio_tok > 0,
           "est_cost_usd": round(cost, 4),
           "mambo": _agg(mambo), "frontier": _agg(front), "qwen_open": _agg(qwen), "per_clip": per}
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = REPO / "runs" / f"{now}-b6-frontier"
    run.mkdir(parents=True, exist_ok=True)
    import subprocess
    out["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["dirty"] = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    (run / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n=== B6 honest (precision / recall / F1 · note-count error) — {len(clips)} pure hums ===")
    print(f"{'system':28s}{'prec':>7s}{'recall':>8s}{'F1':>7s}{'noteΔ':>8s}")
    print(f"{'Mambo (modular)':28s}{out['mambo']['precision']:>7.3f}{out['mambo']['recall']:>8.3f}{out['mambo']['f1']:>7.3f}{out['mambo']['note_count_err']:>8.2f}")
    print(f"{MODEL+' (frontier)':28s}{out['frontier']['precision']:>7.3f}{out['frontier']['recall']:>8.3f}{out['frontier']['f1']:>7.3f}{out['frontier']['note_count_err']:>8.2f}")
    if out["qwen_open"]:
        q = out["qwen_open"]
        print(f"{'Qwen2-Audio-7B (open)':28s}{q['precision']:>7.3f}{q['recall']:>8.3f}{q['f1']:>7.3f}{q['note_count_err']:>8.2f}")
    print(f"\naudio ingested: {out['audio_ingested']} · est cost: ${out['est_cost_usd']}")
    print(f"wrote {run.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
