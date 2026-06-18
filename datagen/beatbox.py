"""Synthetic beatbox proxies for the percussion path (PAPER §4.5).

Layered noise-burst drum sounds (kick / snare / hat) with EXACT labeled onsets,
arranged into patterns ("boots and cats"). Plus a per-user *calibration* set
(4-12 isolated hits per class) — the few-shot examples the classifier trains on,
mirroring Dubler/Ramires per-user calibration. Real beatbox arrives via
operator recordings for the H-gate.

Run: cd lab && uv run python ../datagen/beatbox.py
Deterministic (fixed seed) — regenerable bit-exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
CLASSES = ("kick", "snare", "hat")


def _bandpass(x, lo, hi, sr):
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, x)


def _highpass(x, lo, sr):
    sos = butter(4, lo, btype="high", fs=sr, output="sos")
    return sosfilt(sos, x)


def synth_hit(cls: str, rng: np.random.Generator) -> np.ndarray:
    """One drum hit as a layered noise burst, with per-hit variation."""
    dur = {"kick": 0.20, "snare": 0.16, "hat": 0.07}[cls]
    n = int(dur * SR)
    t = np.arange(n) / SR
    if cls == "kick":
        f0 = rng.uniform(55, 70)
        freq = f0 * np.exp(-t * 28) + 40
        sig = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * rng.uniform(16, 20))
        sig += 0.3 * rng.normal(0, 1, n) * np.exp(-t * 70)  # beater click
    elif cls == "snare":
        noise = _bandpass(rng.normal(0, 1, n), 180, rng.uniform(2800, 3600), SR)
        sig = noise * np.exp(-t * rng.uniform(20, 26))
        sig += 0.5 * np.sin(2 * np.pi * rng.uniform(170, 200) * t) * np.exp(-t * 30)  # body
    else:  # hat
        sig = _highpass(rng.normal(0, 1, n), rng.uniform(5000, 6500), SR) * np.exp(-t * rng.uniform(70, 95))
    m = np.abs(sig).max()
    return (sig / (m + 1e-9) * rng.uniform(0.7, 0.95)).astype(np.float32)


# A few "boots and cats" style patterns: per-step list of classes (or None).
PATTERNS = [
    ["kick", "hat", "snare", "hat"] * 2,                       # boots-and-cats
    ["kick", "hat", "snare", "hat", "kick", "kick", "snare", "hat"],
    ["kick", "snare", "hat", "snare"] * 2,
    ["kick", "hat", "hat", "snare", "kick", "hat", "snare", "hat"],
]


def render_pattern(pattern, tempo, rng) -> tuple[np.ndarray, list[dict]]:
    step = 60.0 / tempo / 2.0  # eighth-note grid
    total = len(pattern) * step + 0.3
    buf = np.zeros(int(total * SR) + SR // 5, dtype=np.float32)
    gt = []
    for i, cls in enumerate(pattern):
        if cls is None:
            continue
        onset = max(0.0, i * step + rng.uniform(-0.008, 0.008))  # slight human timing
        hit = synth_hit(cls, rng)
        a = int(onset * SR)
        buf[a:a + len(hit)] += hit
        gt.append({"t": round(onset, 4), "class": cls})
    buf += rng.normal(0, 0.002, size=buf.shape).astype(np.float32)  # room floor
    m = np.abs(buf).max()
    buf = (buf / (m + 1e-9) * 0.9).astype(np.float32)
    gt.sort(key=lambda h: h["t"])
    return buf, gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../fixtures/percussion")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calib-per-class", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    (out / "calib").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Calibration: isolated hits per class.
    calib = []
    for cls in CLASSES:
        for k in range(args.calib_per_class):
            hit = synth_hit(cls, rng)
            pad = np.zeros(int(0.05 * SR), dtype=np.float32)
            audio = np.concatenate([pad, hit, pad])
            name = f"calib/{cls}_{k:02d}.wav"
            sf.write(str(out / name), audio, SR, subtype="PCM_16")
            calib.append({"wav": name, "class": cls})
    (out / "calib_manifest.jsonl").write_text("\n".join(json.dumps(c) for c in calib) + "\n")

    # Test patterns.
    manifest = []
    for i in range(args.n_test):
        pat = PATTERNS[i % len(PATTERNS)]
        tempo = float(rng.integers(80, 140))
        audio, gt = render_pattern(pat, tempo, rng)
        uid = f"bb_{i:03d}"
        sf.write(str(out / "test" / f"{uid}.wav"), audio, SR, subtype="PCM_16")
        (out / "test" / f"{uid}.json").write_text(json.dumps(
            {"utterance_id": uid, "tempo": tempo, "percussion": gt}) + "\n")
        manifest.append({"utterance_id": uid, "n_hits": len(gt), "tempo": tempo})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"wrote {len(calib)} calib hits + {len(manifest)} test patterns "
          f"({sum(m['n_hits'] for m in manifest)} labeled onsets) to {out}")


if __name__ == "__main__":
    main()
