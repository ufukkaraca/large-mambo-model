"""§6.1(f) — does per-voice calibration ("voiceprint") recover note-count on the
held-out voices, with zero regression on a normal one?

For each held-out speaker we derive a Voiceprint from their own held + speech
clips (vibrato depth → split threshold, via the deadband in `voiceprint.py`), then
re-score note-count ±1 BASELINE (shipped pitch_step=1.0) vs CALIBRATED (the
speaker's pitch_step). This is the committed artifact behind the PAPER §6.1(f)
table — pyin-only, no network, ~2 min.

    cd lab && uv run python -m mambo_lab.eval.voiceprint_eval
Writes runs/<ts>-voiceprint/results.json.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import soundfile as sf

from . import metrics
from .. import melody, voiceprint

REPO = Path(__file__).resolve().parents[3]
VOICES = REPO / "fixtures" / "human" / "voices"
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


def _ncount(hits: list[int]) -> dict:
    k, n = int(sum(hits)), len(hits)
    mean, lo, hi = metrics.wilson_ci(k, n)
    return {"ok": k, "n": n, "acc": round(mean, 3) if mean is not None else None,
            "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None}


def score_speaker(d: Path) -> dict | None:
    ents = _entries(d)
    note = [e for e in ents if e.get("kind", "note") == "note" and "intended_notes" in e]
    if not note:
        return None
    speech = [e for e in ents if e.get("kind") == "speech"]
    # calibration inputs: the 1-note "hold_*" clips (vibrato) + spoken commands (voicing)
    held = [e for e in note if "hold" in e["wav"]] or [e for e in note if int(e.get("verified_notes", e["intended_notes"])) == 1]
    vp = voiceprint.derive([_load(d / e["wav"]) for e in held],
                           [_load(d / e["wav"]) for e in speech], label=d.name)

    base_hits, cal_hits, detail = [], [], []
    for e in note:
        a, sr = _load(d / e["wav"])
        track = melody.track_f0(a, sr)
        nb = len(melody.segment_notes(a, track, pitch_step=1.0))
        nc = len(melody.segment_notes(a, track, pitch_step=vp.pitch_step))
        gt = int(e.get("verified_notes", e["intended_notes"]))
        tol = int(e.get("tol", 1))
        base_hits.append(int(abs(nb - gt) <= tol))
        cal_hits.append(int(abs(nc - gt) <= tol))
        detail.append({"wav": e["wav"], "gt": gt, "baseline_n": nb, "calibrated_n": nc})

    base, cal = _ncount(base_hits), _ncount(cal_hits)
    return {"speaker": d.name,
            "vibrato_semitones": round(vp.vibrato_semitones, 3),
            "pitch_step": round(vp.pitch_step, 3),
            "deadband_untouched": vp.pitch_step == 1.0,
            "note_count_baseline": base, "note_count_calibrated": cal,
            "delta": round((cal["acc"] or 0) - (base["acc"] or 0), 3),
            "note_detail": detail}


def main() -> int:
    if not VOICES.exists():
        print(f"no voices at {VOICES}")
        return 0
    rows = [r for r in (score_speaker(p) for p in sorted(VOICES.iterdir()) if p.is_dir()) if r]
    if not rows:
        print("no scorable speakers")
        return 0
    print("\n§6.1(f) voiceprint calibration — note-count ±1 (Wilson 95% CI), baseline vs calibrated\n")
    print(f"{'speaker':<8}{'vibrato':>9}{'step':>7}{'baseline':>22}{'calibrated':>22}{'Δ':>8}")
    print("-" * 78)
    for r in rows:
        b, c = r["note_count_baseline"], r["note_count_calibrated"]
        bs = f"{b['acc']:.2f} [{b['ci95'][0]:.2f},{b['ci95'][1]:.2f}]"
        cs = f"{c['acc']:.2f} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]"
        tag = r["speaker"] + ("*" if r["deadband_untouched"] else "")
        print(f"{tag:<8}{r['vibrato_semitones']:>9.2f}{r['pitch_step']:>7.2f}{bs:>22}{cs:>22}{r['delta']:>+8.2f}")
    print("\n* = deadband left this voice at the shipped default (no regression by construction).")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rd = RUNS / f"{now}-voiceprint"
    rd.mkdir(parents=True, exist_ok=True)
    out = {"phase": "R0-voiceprint-calibration", "timestamp": now,
           "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
           "metric": "note-count within ±1 of self-reported intended_notes; voiceprint from own held+speech clips",
           "speakers": rows}
    (rd / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {rd.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
