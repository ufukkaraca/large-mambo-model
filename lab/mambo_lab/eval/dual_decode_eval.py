"""§6 dual-decode — does reasoning recover a SUNG lyric the containment baseline
discards, while keeping the melody? (D23 P2, the measured §3.5 reasoning-novelty.)

Runs on real sung-demonstration clips in `fixtures/human/sung/<name>/` (see
the repository for the recording protocol + manifest format). For each clip:

  notes      = pitch tracker over the clip            -> note-count vs intended
  candidate  = ASR transcript minus the spoken frame  -> the words actually sung
  baseline   = containment: drops all text on a hum   -> lyric word-recall = 0 BY RULE
  dual-decode= reasoning keeps candidate iff it judges a real sung lyric -> word-recall

Headline = lyric word-recall (dual-decode vs baseline 0) on sung demos, AND a
false-lyric check on wordless `sung_control_*` clips (reasoning must NOT invent a
lyric — containment must still hold there). Needs Whisper locally + OPENAI_API_KEY.

    cd lab && uv run python -m mambo_lab.eval.dual_decode_eval
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import dual_decode, melody, probe, secrets

REPO = Path(__file__).resolve().parents[3]
SUNG = REPO / "fixtures" / "human" / "sung"
RUNS = REPO / "runs"

# spoken frames to strip so the residual is the sung lyric (kept loose; the GT
# word-recall is order-free so a stray frame word costs little).
_FRAMES = [r"give me something like", r"make (?:it|the \w+) go", r"something like",
           r"how about", r"what about", r"but (?:slower|faster|softer|warmer|higher|lower)",
           r"a bit \w+", r"and \w+", r"instead( of)?"]
_FRAME_RE = re.compile("|".join(_FRAMES), re.I)


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


def _residual_lyric(transcript: str) -> str:
    return _FRAME_RE.sub(" ", transcript or "").strip()


def score_clip(path: Path, e: dict, key: str) -> dict:
    a, sr = _load(path)
    n_notes = len(melody.segment_notes(a, melody.track_f0(a, sr)))
    transcript = (probe.transcribe(a, sr).text or "").strip()
    candidate = _residual_lyric(transcript)
    is_lyric = dual_decode.judge_sung_lyric(candidate, n_notes, key=key)
    dd_lyric = candidate if is_lyric else ""
    gt_lyric = e.get("lyric", "")
    is_control = e.get("kind") == "sung_control" or not gt_lyric
    return {
        "wav": e["wav"], "is_control": is_control,
        "transcript": transcript, "candidate": candidate,
        "judged_sung_lyric": is_lyric, "dual_decode_lyric": dd_lyric,
        "n_notes": n_notes, "intended_notes": e.get("intended_notes"),
        "note_ok": (abs(n_notes - e["intended_notes"]) <= int(e.get("tol", 1))) if e.get("intended_notes") else None,
        # the measured contrast:
        "lyric_recall_baseline": 0.0 if gt_lyric else 1.0,            # containment drops it
        "lyric_recall_dualdecode": dual_decode.word_recall(dd_lyric, gt_lyric),
        "wer_dualdecode": dual_decode.wer(dd_lyric, gt_lyric) if gt_lyric else None,
    }


def main() -> int:
    secrets.load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not SUNG.exists() or not any(SUNG.iterdir()):
        print(f"no sung clips yet at {SUNG}/ — record per the protocol in the repository (eval is wired + waiting).")
        return 0
    rows = []
    for d in sorted(p for p in SUNG.iterdir() if p.is_dir()):
        for e in _entries(d):
            rows.append(score_clip(d / e["wav"], e, key))
    if not rows:
        print(f"{SUNG}/ exists but no manifested clips found.")
        return 0

    demos = [r for r in rows if not r["is_control"]]
    ctrls = [r for r in rows if r["is_control"]]
    dd_recall = round(float(np.mean([r["lyric_recall_dualdecode"] for r in demos])), 3) if demos else None
    note_ok = [r["note_ok"] for r in demos if r["note_ok"] is not None]
    note_acc = round(float(np.mean(note_ok)), 3) if note_ok else None
    false_lyric = sum(1 for r in ctrls if r["judged_sung_lyric"])

    print(f"\n=== Dual-decode — sung demonstrations (n={len(demos)} demos, {len(ctrls)} controls) ===")
    print(f"lyric word-recall: baseline (containment) = 0.000  |  dual-decode = {dd_recall}")
    print(f"melody note-count ±1 (preserved): {note_acc}")
    print(f"control false-lyric (must be 0): {false_lyric}/{len(ctrls)}")
    print("\nper sung demo:")
    for r in demos:
        print(f"  {r['wav']:22} notes={r['n_notes']}/{r['intended_notes']} "
              f"recall={r['lyric_recall_dualdecode']:.2f}  lyric={r['dual_decode_lyric'][:40]!r}")

    out = {"task": "dual-decode: melody + captured lyric on sung demonstrations",
           "n_demos": len(demos), "n_controls": len(ctrls),
           "lyric_recall_baseline": 0.0, "lyric_recall_dualdecode": dd_recall,
           "note_count_acc": note_acc, "control_false_lyric": f"{false_lyric}/{len(ctrls)}",
           "per_clip": rows}
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rd = RUNS / f"{now}-dual-decode"
    rd.mkdir(parents=True, exist_ok=True)
    out["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out["dirty"] = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    (rd / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {rd.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
