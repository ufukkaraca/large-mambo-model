"""Component ablations — are the robustness mechanisms load-bearing or decorative?

Toggles each mechanism OFF (by monkeypatching it to a passthrough, no core signature
change) and measures the metric delta vs the full pipeline, per SNR:
  - carve (D18, `router._carve_embedded_hum`)  → joint segment F1 (its job: recover
    the noisy-hum 10 dB collapse)
  - de-flutter (D19, `melody._deflutter`)       → note F1 (merges vibrato-split notes)
  - pitch-plateau onsets (D20, `melody._pitch_onsets`) → note F1 (legato splitting)

Synthetic hums are detache (D9), so the de-flutter/plateau deltas are expected ≈0 here
— that is the *no-regression* check; their real benefit is on real voice (E-LIVE:
note-count 0.52→0.857, D20). The carve delta should be real at 10 dB.

    cd lab && uv run python -m mambo_lab.eval.ablation   # heavy (whisper+pyin); daytime
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

from . import gate
from .. import melody, router

REPO = Path(__file__).resolve().parents[3]


@contextlib.contextmanager
def _off(*patches):
    """Temporarily replace (module, attr, fn) triples; restore on exit."""
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    for m, a, fn in patches:
        setattr(m, a, fn)
    try:
        yield
    finally:
        for m, a, fn in saved:
            setattr(m, a, fn)


# passthroughs (signature-agnostic): carve/deflutter return their input unchanged;
# pitch-onsets returns no extra boundaries (amplitude-only segmentation).
_no_carve = (router, "_carve_embedded_hum", lambda *a, **k: a[0])
_no_deflutter = (melody, "_deflutter", lambda *a, **k: a[0])
_no_plateau = (melody, "_pitch_onsets", lambda *a, **k: [])


def _joint_f1(rows) -> dict:
    rt = gate.eval_router(rows)
    return {snr: round(float(np.mean(rt["joint"][snr]["f1"])), 4) for snr in ("clean", "20db", "10db")}


def _note_f1(rows) -> dict:
    mel = gate.eval_melody(rows)
    return {snr: round(float(np.mean(mel[snr]["f1"])), 4) for snr in ("clean", "20db", "10db") if mel.get(snr, {}).get("f1")}


def run() -> dict:
    rows = gate.load_fixtures()
    for r in rows:
        r.setdefault("uid", r.get("utterance_id"))

    full_seg = _joint_f1(rows)
    full_note = _note_f1(rows)
    with _off(_no_carve):
        nocarve_seg = _joint_f1(rows)
    with _off(_no_deflutter):
        nodefl_note = _note_f1(rows)
    with _off(_no_plateau):
        noplat_note = _note_f1(rows)

    def delta(full, abl):
        return {snr: round(full[snr] - abl[snr], 4) for snr in full if snr in abl}

    return {
        "metric_carve": "joint segment F1", "metric_note": "note F1",
        "full": {"joint_seg_f1": full_seg, "note_f1": full_note},
        "no_carve": {"joint_seg_f1": nocarve_seg, "delta": delta(full_seg, nocarve_seg)},
        "no_deflutter": {"note_f1": nodefl_note, "delta": delta(full_note, nodefl_note)},
        "no_pitch_plateau": {"note_f1": noplat_note, "delta": delta(full_note, noplat_note)},
    }


def main() -> int:
    r = run()
    print("\n=== Component ablation (synthetic; Δ = full − ablated, positive = mechanism helps) ===\n")
    print(f"{'mechanism':22s}{'metric':>18s}{'clean':>9s}{'20 dB':>9s}{'10 dB':>9s}")
    print("-" * 67)
    fs = r["full"]["joint_seg_f1"]
    print(f"{'FULL pipeline':22s}{'joint seg F1':>18s}{fs['clean']:>9.3f}{fs['20db']:>9.3f}{fs['10db']:>9.3f}")
    d = r["no_carve"]["delta"]
    print(f"{'  − carve (D18)':22s}{'Δ seg F1':>18s}{d.get('clean',0):>+9.3f}{d.get('20db',0):>+9.3f}{d.get('10db',0):>+9.3f}")
    fn = r["full"]["note_f1"]
    print(f"{'FULL pipeline':22s}{'note F1':>18s}" + "".join(f"{fn.get(s,float('nan')):>9.3f}" for s in ('clean','20db','10db')))
    for key, lbl in [("no_deflutter", "  − de-flutter (D19)"), ("no_pitch_plateau", "  − plateau (D20)")]:
        d = r[key]["delta"]
        print(f"{lbl:22s}{'Δ note F1':>18s}" + "".join(f"{d.get(s,0):>+9.3f}" for s in ('clean','20db','10db')))
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "runs" / f"{now}-ablation"
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (run_dir / "results.json").write_text(json.dumps({"commit": commit, **r}, indent=2) + "\n")
    print(f"\nwrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
