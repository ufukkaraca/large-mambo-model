"""Real-voice baselines on the operator's E-LIVE set (PAPER §3.5 §Baselines).

Produces the paper's headline real-voice comparison, reproducibly:

  * NAIVE single-path (Whisper-over-everything, no containment) vs Mambo
    (joint + note-arbiter containment): content words confabulated onto the
    hummed span. This is the §2.4 claim on human voice.
  * §2.4 ablation on real voice: acoustic / linguistic / joint hum-detection on
    the mixed command+hum utterances, and no-hum on the spoken commands.

Run:  cd lab && uv run python -m mambo_lab.eval.elive_baselines
Writes runs/<ts>-elive-baselines/results.json (provenance like the gates).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import fuse, melody, probe, router

REPO = Path(__file__).resolve().parents[3]
ELIVE = REPO / "fixtures" / "human" / "E-LIVE"
RUNS = REPO / "runs"


def _load(w: Path):
    a, sr = sf.read(str(w), dtype="float32")
    return (a.mean(axis=1) if a.ndim > 1 else a), sr


def _manifest() -> dict:
    man = ELIVE / "manifest.jsonl"
    out = {}
    if man.exists():
        for ln in man.read_text().splitlines():
            if ln.strip():
                e = json.loads(ln)
                out[e["wav"]] = e
    return out


def _hum_spans(segs):
    return [(s["t0"], s["t1"]) for s in segs if s["kind"] in ("melody", "ambiguous")]


def run() -> dict:
    man = _manifest()
    mixed = [w for w in sorted(man) if man[w].get("kind") == "mixed"]
    speech = [w for w in sorted(man) if man[w].get("kind") == "speech"]

    naive_leak = mambo_leak = 0
    per_clip = []
    arms = {"acoustic": 0, "linguistic": 0, "joint": 0}  # hum-detected on mixed
    for w in mixed:
        a, sr = _load(ELIVE / w)
        f0 = melody.track_f0(a, sr)
        pr = probe.transcribe(a, sr)
        joint = fuse.fuse(a, sr, router.route(a, sr, strategy="joint", f0=f0, pr=pr), pr, f0,
                          utterance_id=w).to_dict()["segments"]
        hums = _hum_spans(joint)
        # NAIVE: raw Whisper words overlapping the hummed span = confabulation
        naive_words = [wd.w for wd in pr.words
                       if any(wd.t0 < t1 and wd.t1 > t0 for (t0, t1) in hums)]
        mambo_text = [s.get("text", "") for s in joint
                      if s["kind"] in ("melody", "ambiguous") and s.get("text")]
        naive_leak += len(naive_words) > 0
        mambo_leak += len(mambo_text) > 0
        # §2.4 arms: did the arm find any notes (hum) at all?
        for arm in arms:
            segs = fuse.fuse(a, sr, router.route(a, sr, strategy=arm, f0=f0, pr=pr), pr, f0,
                             utterance_id=w).to_dict()["segments"]
            arms[arm] += sum(len(s.get("notes", [])) for s in segs
                             if s["kind"] in ("melody", "ambiguous")) > 0
        per_clip.append({"wav": w, "naive_words_on_hum": naive_words, "mambo_text_on_hum": mambo_text})

    # spoken commands: no melody hallucinated, per arm
    speech_no_hum = {"acoustic": 0, "linguistic": 0, "joint": 0}
    for w in speech:
        a, sr = _load(ELIVE / w)
        f0 = melody.track_f0(a, sr)
        pr = probe.transcribe(a, sr)
        for arm in speech_no_hum:
            segs = fuse.fuse(a, sr, router.route(a, sr, strategy=arm, f0=f0, pr=pr), pr, f0,
                             utterance_id=w).to_dict()["segments"]
            speech_no_hum[arm] += (not any(s["kind"] == "melody" for s in segs))

    nm, ns = len(mixed), len(speech)
    return {
        "n_mixed": nm, "n_speech": ns,
        "containment": {"naive_halluc_clips": naive_leak, "mambo_halluc_clips": mambo_leak},
        "hum_detected_by_arm": arms,
        "speech_no_hum_by_arm": speech_no_hum,
        "per_clip": per_clip,
    }


def main() -> int:
    r = run()
    nm, ns = r["n_mixed"], r["n_speech"]
    print("\n=== E-LIVE real-voice baselines ===")
    print(f"\nContainment on the {nm} mixed command+hum utterances (words confabulated onto the hum):")
    print(f"  NAIVE  Whisper-over-everything : {r['containment']['naive_halluc_clips']}/{nm} clips leak words onto the hum")
    print(f"  MAMBO  joint + containment     : {r['containment']['mambo_halluc_clips']}/{nm} clips  (§2.4 on real voice)")
    print(f"\n§2.4 arms — hum detected inside the {nm} mixed utterances:")
    for arm, v in r["hum_detected_by_arm"].items():
        print(f"  {arm:<11} {v}/{nm}")
    print(f"\n§2.4 arms — {ns} spoken commands correctly kept hum-free (no melody hallucinated):")
    for arm, v in r["speech_no_hum_by_arm"].items():
        print(f"  {arm:<11} {v}/{ns}")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{now}-elive-baselines"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"\nwrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
