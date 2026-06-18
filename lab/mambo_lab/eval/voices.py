"""Multi-speaker generalization: the E-LIVE H-gate metrics per HELD-OUT voice.

Every threshold in the note/segmentation path was tuned on the operator's voice.
This scores each additional speaker in fixtures/human/voices/<name>/ on the same
four metrics — note-count (±1), speech-no-hum, mixed-containment, mixed-hum-found
— so we can see whether the result generalizes beyond N=1.

    cd lab && uv run python -m mambo_lab.eval.voices
Writes runs/<ts>-voices/results.json.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import soundfile as sf

from . import metrics
from .. import fuse, melody, probe  # noqa: F401  (probe imported via fuse)

REPO = Path(__file__).resolve().parents[3]
VOICES = REPO / "fixtures" / "human" / "voices"
META = REPO / "fixtures" / "human" / "voice_meta.json"
RUNS = REPO / "runs"


def _meta(name: str) -> dict:
    """Committed per-speaker metadata (musician flag, etc.); {} if unknown."""
    if not META.exists():
        return {}
    try:
        return json.loads(META.read_text()).get(name, {}) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _entries(speaker_dir: Path) -> list[dict]:
    man = speaker_dir / "manifest.jsonl"
    if not man.exists():
        return []
    out = []
    for ln in man.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return [e for e in out if "wav" in e and (speaker_dir / e["wav"]).exists()]


def _load(p: Path):
    a, sr = sf.read(str(p), dtype="float32")
    return (a.mean(axis=1) if a.ndim > 1 else a), sr


def score_speaker(speaker_dir: Path) -> dict | None:
    ents = _entries(speaker_dir)
    if not ents:
        return None
    note = [e for e in ents if e.get("kind", "note") == "note" and "intended_notes" in e]
    speech = [e for e in ents if e.get("kind") == "speech"]
    mixed = [e for e in ents if e.get("kind") == "mixed"]

    def uir(w):
        a, sr = _load(speaker_dir / w)
        return fuse.file_to_uir(a, sr, strategy="joint", utterance_id=w).to_dict()["segments"]

    note_hits, note_detail = [], []
    for e in note:
        a, sr = _load(speaker_dir / e["wav"])
        n = len(melody.segment_notes(a, melody.track_f0(a, sr)))
        # prefer a hand-verified count if present; else the
        # speaker's self-reported intent (noisy for non-musicians).
        gt = int(e.get("verified_notes", e["intended_notes"]))
        tol = int(e.get("tol", 1))
        note_hits.append(int(abs(n - gt) <= tol))
        note_detail.append({"wav": e["wav"], "gt": gt, "detected": n,
                            "gt_source": "verified" if "verified_notes" in e else "self-report",
                            "hit": int(abs(n - gt) <= tol)})
    sp_hits = [int(not any(s["kind"] == "melody" for s in uir(e["wav"]))) for e in speech]
    contain_hits, found_hits = [], []
    for e in mixed:
        segs = uir(e["wav"])
        mel = [s for s in segs if s["kind"] in ("melody", "ambiguous")]
        contain_hits.append(int(not any(s.get("text") for s in mel)))
        found_hits.append(int(sum(len(s.get("notes", [])) for s in mel) > 0))

    def m(hits):  # Wilson CI — correct at the 0/n and n/n boundaries
        k, nn = int(sum(hits)), len(hits)
        mean, lo, hi = metrics.wilson_ci(k, nn)
        return {"ok": k, "n": nn,
                "acc": round(mean, 3) if mean is not None else None,
                "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None}
    meta = _meta(speaker_dir.name)
    return {
        "speaker": speaker_dir.name,
        "musician": meta.get("musician"),  # True / False / None(unknown)
        "note_count": m(note_hits), "speech_no_hum": m(sp_hits),
        "mixed_containment": m(contain_hits), "mixed_hum_found": m(found_hits),
        "note_detail": note_detail,  # per-clip intended(self-report) vs detected
        "note_count_gt": "self-reported (intended_notes); hand-count not yet verified",
    }


def main() -> int:
    if not VOICES.exists():
        print(f"no voices at {VOICES} — unzip friend recordings into voices/<name>/")
        return 0
    rows = [r for r in (score_speaker(d) for d in sorted(VOICES.iterdir()) if d.is_dir()) if r]
    if not rows:
        print("no scorable speakers found")
        return 0
    print("\nWilson 95% CIs per speaker — intervals are wide at this n, honestly.")
    print("note-count GT is SELF-REPORTED (intended_notes); for a non-musician it may "
          "conflate model error with label error — hand-count review queued.\n")
    print(f"{'speaker':<10} {'musician':<9} {'metric':<20} {'acc':>6}  {'95% CI':>14}  n")
    METS = [("note_count", "note-count ±1"), ("speech_no_hum", "speech-no-hum"),
            ("mixed_containment", "mixed-containment"), ("mixed_hum_found", "mixed-hum-found")]
    review = []
    for r in rows:
        mus = {True: "yes", False: "no", None: "?"}[r.get("musician")]
        for key, name in METS:
            mm = r[key]
            ci = f"[{mm['ci95'][0]:.2f}, {mm['ci95'][1]:.2f}]" if mm["ci95"] else "—"
            acc = f"{mm['acc']:.2f}" if mm["acc"] is not None else "—"
            print(f"{r['speaker']:<10} {mus:<9} {name:<20} {acc:>6}  {ci:>14}  {mm['n']}")
        # flag big GT-vs-detected divergences (|Δ| >= 3) as hand-count targets
        for d in r.get("note_detail", []):
            if d["gt_source"] == "self-report" and abs(d["detected"] - d["gt"]) >= 3:
                review.append(f"{r['speaker']}/{d['wav']}: self-report {d['gt']} vs detected {d['detected']}")
        print()
    if review:
        print(f"⚑ {len(review)} clip(s) with |self-report − detected| ≥ 3 — hand-count review candidates:")
        for x in review[:12]:
            print("   ", x)
        print()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{now}-voices"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
