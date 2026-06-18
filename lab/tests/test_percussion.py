"""Unit tests for the percussion path (synthesize hits, calibrate, classify)."""

import sys
from pathlib import Path

import numpy as np

from mambo_lab import percussion as P

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "datagen"))
import beatbox  # noqa: E402

SR = beatbox.SR


def _calib(rng, per_class=6):
    out = []
    for cls in P.CLASSES:
        for _ in range(per_class):
            hit = beatbox.synth_hit(cls, rng)
            pad = np.zeros(int(0.05 * SR), dtype=np.float32)
            out.append((np.concatenate([pad, hit, pad]), cls))
    return out


def test_classifier_separates_classes():
    rng = np.random.default_rng(1)
    clf = P.train_from_calib(_calib(rng), SR)
    rng2 = np.random.default_rng(99)
    correct = 0
    for cls in P.CLASSES:
        for _ in range(8):
            hit = beatbox.synth_hit(cls, rng2)
            pad = np.zeros(int(0.05 * SR), dtype=np.float32)
            audio = np.concatenate([pad, hit, pad])
            pred, _ = clf.predict(P.hit_features(audio, SR, 0.05))
            correct += pred == cls
    assert correct / 24 >= 0.9


def test_detect_onsets_on_pattern():
    rng = np.random.default_rng(2)
    audio, gt = beatbox.render_pattern(["kick", "hat", "snare", "hat"], 100, rng)
    onsets = P.detect_onsets(audio, SR)
    assert abs(len(onsets) - len(gt)) <= 1  # ~one per hit


def test_to_uir_percussion_format():
    hits = [P.PercussionHit(t=0.5, cls="kick", confidence=0.9)]
    out = P.to_uir_percussion(hits)
    assert out == [{"t": 0.5, "class": "kick", "confidence": 0.9}]
