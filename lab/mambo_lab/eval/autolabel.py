"""First-pass auto-labeler for the real-voice benchmark (BENCHMARK.md §Labeling).

Bootstraps ground truth so humans VERIFY rather than transcribe from scratch: for
each clip it runs pyin (a candidate pitch sequence + note count) and Whisper (a
candidate transcript), and writes a `labels_draft.jsonl` the human then corrects.

CRITICAL: auto labels are marked `gt_source:"auto"` and the benchmark
(`voicebench`) IGNORES them — using the system's own pyin output as ground truth
would be circular. A human must listen, fix the pitches/lyric, and flip
`gt_source` to `"verified"` before the label counts. For "known-tune" hums
(a scale, a nursery rhyme) verification is a quick sanity check, not transcription.

    cd lab && uv run python -m mambo_lab.eval.autolabel ../fixtures/human/voices/<name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

from .. import melody, probe


def _load(p: Path):
    a, sr = sf.read(str(p), dtype="float32")
    return (a.mean(axis=1) if a.ndim > 1 else a), sr


def draft_clip(path: Path, e: dict) -> dict:
    audio, sr = _load(path)
    notes = melody.segment_notes(audio, melody.track_f0(audio, sr))
    d = {"wav": e["wav"], "kind": e.get("kind", "note"), "gt_source": "auto"}
    if notes:
        d["gt_pitches"] = [n.midi for n in notes]   # CANDIDATE — human must verify
        d["intended_notes"] = len(notes)
    if e.get("kind") in ("speech", "sung", "mixed") or "command" in e["wav"]:
        txt = (probe.transcribe(audio, sr).text or "").strip()
        if txt:
            d["transcript_auto"] = txt              # CANDIDATE lyric/command text
    # carry any GT the human already set
    for k in ("verified_notes", "lyric", "text", "tol"):
        if k in e:
            d[k] = e[k]
    return d


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: autolabel <speaker_dir>")
        return 2
    d = Path(argv[0])
    man = d / "manifest.jsonl"
    if not man.exists():
        print(f"no manifest at {man}")
        return 1
    ents = [json.loads(ln) for ln in man.read_text().splitlines() if ln.strip()]
    ents = [e for e in ents if "wav" in e and (d / e["wav"]).exists()]
    drafts = [draft_clip(d / e["wav"], e) for e in ents]
    out = d / "labels_draft.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in drafts) + "\n")

    print(f"\nauto-labeled {len(drafts)} clips → {out.relative_to(d.parents[2])}")
    print("REVIEW each line: fix gt_pitches / lyric, then set gt_source=\"verified\" and merge into manifest.jsonl.\n")
    for x in drafts:
        pit = x.get("gt_pitches", [])
        print(f"  {x['wav']:24} notes={len(pit):>2} pitches={pit[:10]}"
              + (f"  text={x['transcript_auto'][:40]!r}" if x.get("transcript_auto") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
