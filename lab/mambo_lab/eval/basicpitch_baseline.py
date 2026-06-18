"""BasicPitch (Spotify) vs Mambo note detection on the E-LIVE hums.

BasicPitch is DAWZY's hum->MIDI transcriber (PAPER §3.5). It is a
general-purpose POLYPHONIC instrument transcriber, not humming-specialised, so on
monophonic hums it over-segments (vibrato / overtones / pitch bends become extra
notes). Mambo's `melody.segment_notes` (pYIN + amplitude-onset UNION pitch-plateau,
with de-flutter + fmin-phantom drop) is tuned for humming.

basic-pitch 0.3.0 needs legacy Keras + old scipy, so it is NOT a core dependency;
run this baseline in an ephemeral env (keeps the lockfile clean):

    cd lab && TF_USE_LEGACY_KERAS=1 uv run \
        --with basic-pitch --with tf-keras --with "scipy<1.13" \
        python -m mambo_lab.eval.basicpitch_baseline
"""

from __future__ import annotations

import datetime as dt
import json
import os
import warnings
from pathlib import Path

import soundfile as sf

from .. import melody

REPO = Path(__file__).resolve().parents[3]
ELIVE = REPO / "fixtures" / "human" / "E-LIVE"
RUNS = REPO / "runs"


def run() -> dict:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from basic_pitch import ICASSP_2022_MODEL_PATH  # ephemeral dep — import lazily
    from basic_pitch.inference import predict

    man = {json.loads(l)["wav"]: json.loads(l)
           for l in (ELIVE / "manifest.jsonl").read_text().splitlines() if l.strip()}
    note_clips = [w for w in sorted(man)
                  if man[w].get("kind", "note") == "note" and "intended_notes" in man[w]]
    rows, mok, bok = [], 0, 0
    for w in note_clips:
        e = man[w]
        want, tol = e["intended_notes"], e.get("tol", 1)
        a, sr = sf.read(str(ELIVE / w), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        mambo = len(melody.segment_notes(a, melody.track_f0(a, sr)))
        _, _, notes = predict(str(ELIVE / w), ICASSP_2022_MODEL_PATH)
        bp = len(notes)
        mok += abs(mambo - want) <= tol
        bok += abs(bp - want) <= tol
        rows.append({"wav": w, "intended": want, "mambo": mambo, "basicpitch": bp})
    n = len(note_clips)
    return {"n": n, "mambo_acc": mok / n, "basicpitch_acc": bok / n, "rows": rows}


def main() -> int:
    r = run()
    print(f"\n{'clip':<22} {'want':>4} {'mambo':>6} {'basicpitch':>11}")
    for row in r["rows"]:
        print(f"{row['wav']:<22} {row['intended']:>4} {row['mambo']:>6} {row['basicpitch']:>11}")
    print(f"\nnote-count within ±1 on {r['n']} real hums:")
    print(f"  Mambo (pYIN+onset∪pitch-plateau)   : {r['mambo_acc']:.2f}")
    print(f"  BasicPitch (Spotify, DAWZY's tool) : {r['basicpitch_acc']:.2f}")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{now}-basicpitch-baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"\nwrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
