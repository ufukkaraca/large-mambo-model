"""Unit tests for the melody path (synthesize a known phrase, recover it)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from mambo_lab import melody
from mambo_lab.eval import metrics

# Pull in the datagen synth primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "datagen"))
import synth  # noqa: E402


def _render(notes, seed=3):
    rng = np.random.default_rng(seed)
    audio, _ = synth.render_hum(notes, synth.SR, rng)
    return audio, synth.SR


def test_recovers_known_phrase_notes():
    # C4 E4 G4 E4 D4 C4 at ~100 bpm, detache.
    notes = [(60, 0.0, 0.5), (64, 0.6, 0.5), (67, 1.2, 0.5),
             (64, 1.8, 0.5), (62, 2.4, 0.5), (60, 3.0, 0.7)]
    audio, sr = _render(notes)
    seg = melody.analyze_span(audio, sr)
    est = [n.to_dict() for n in seg.notes]
    ref = [{"t0": t0, "dur": d, "midi": m} for m, t0, d in notes]
    _, _, f1 = metrics.note_prf(ref, est)
    assert f1 >= 0.80, f"note F1 {f1:.2f} below 0.80 on a clean known phrase"


def test_key_topk_on_tonal_phrase():
    # C major phrase framed by the tonic.
    notes = [(60, 0.0, 0.9), (64, 1.0, 0.5), (67, 1.6, 0.5),
             (64, 2.2, 0.5), (62, 2.8, 0.5), (60, 3.4, 0.9)]
    audio, sr = _render(notes)
    seg = melody.analyze_span(audio, sr)
    cands = [c.to_dict() for c in seg.analysis.key_candidates]
    assert metrics.key_in_topk("C major", cands, k=2), f"C major not in top-2: {cands}"


def test_silence_yields_no_notes():
    audio = np.zeros(synth.SR, dtype=np.float32)
    seg = melody.analyze_span(audio, synth.SR)
    assert seg.notes == []


def _wobbly_tone(hz, dur, sr, vib_cents=40, vib_hz=6, trem=0.4, trem_hz=5):
    """A single SUSTAINED pitch with human-like vibrato + tremolo — the live
    failure mode where amplitude ripple spuriously splits one held note."""
    t = np.arange(int(dur * sr)) / sr
    fmod = hz * 2 ** ((vib_cents / 100.0) * np.sin(2 * np.pi * vib_hz * t) / 12.0)
    phase = 2 * np.pi * np.cumsum(fmod) / sr
    env = np.minimum(t / 0.02, 1) * np.minimum((dur - t) / 0.05, 1)
    amp = 1.0 - trem * (0.5 + 0.5 * np.sin(2 * np.pi * trem_hz * t))
    return (0.6 * env * amp * np.sin(phase)).astype(np.float32)


def test_deflutter_collapses_wobbly_held_note():
    # One held A4 with vibrato+tremolo must NOT explode into many notes (the
    # 'da-da-da -> 12 notes' live complaint); de-flutter folds it to one pitch.
    audio = _wobbly_tone(440.0, 2.0, synth.SR)
    notes = melody.segment_notes(audio, melody.track_f0(audio, synth.SR))
    assert len(notes) <= 2, f"held note over-segmented into {len(notes)}: {[n.midi for n in notes]}"
    assert all(n.midi == 69 for n in notes), [n.midi for n in notes]


def _legato_tone(midis, dur_each, sr):
    """One continuous amplitude envelope; pitch steps between notes with NO
    re-attack — the legato case onset-only segmentation misses entirely."""
    fps = np.concatenate([np.full(int(dur_each * sr), 440 * 2 ** ((m - 69) / 12)) for m in midis])
    nfr = len(fps)
    ph = 2 * np.pi * np.cumsum(fps) / sr
    t = np.arange(nfr)
    env = np.minimum(t / (0.02 * sr), 1) * np.minimum((nfr - t) / (0.05 * sr), 1)
    return (0.6 * env * np.sin(ph)).astype(np.float32)


def test_segments_legato_melody_by_pitch():
    # 4 connected notes, no amplitude attacks between them -> pitch plateaus must
    # recover them (amplitude onsets alone would see one note).
    audio = _legato_tone([60, 62, 64, 67], 0.45, synth.SR)
    notes = melody.segment_notes(audio, melody.track_f0(audio, synth.SR))
    assert 3 <= len(notes) <= 5, f"legato melody -> {len(notes)} notes: {[n.midi for n in notes]}"


def test_drop_floor_phantoms():
    # pYIN pins to FMIN in silent lead-in -> a phantom low note at the velocity
    # floor. Drop it; keep a genuinely LOUD low note and normal notes.
    fmin_midi = int(round(69 + 12 * np.log2(melody.FMIN / 440.0)))  # ~39
    phantom = melody.TrackedNote(midi=fmin_midi, t0=0.0, dur=0.5, cents_dev=49, vel=40)
    loud_low = melody.TrackedNote(midi=fmin_midi, t0=0.6, dur=0.5, cents_dev=0, vel=72)
    real = melody.TrackedNote(midi=60, t0=1.2, dur=0.5, cents_dev=0, vel=80)
    kept, _ = melody._drop_floor_phantoms([phantom, real, loud_low], [(0, 1), (2, 3), (4, 5)])
    assert phantom not in kept, "silent FMIN phantom should be dropped"
    assert real in kept and loud_low in kept, "real notes (incl. a loud low one) must survive"


def test_deflutter_preserves_distinct_pitches():
    # De-flutter is equal-pitch only: a real phrase keeps every note.
    notes = [(60, 0.0, 0.5), (64, 0.6, 0.5), (67, 1.2, 0.5), (60, 1.8, 0.5)]
    audio, sr = _render(notes)
    est = melody.segment_notes(audio, melody.track_f0(audio, sr))
    assert len(est) >= 4, f"distinct-pitch phrase lost notes: {[n.midi for n in est]}"


@pytest.mark.parametrize("hz,midi", [(261.63, 60), (440.0, 69)])
def test_f0_stats_voicing(hz, midi):
    t = np.arange(int(0.8 * synth.SR)) / synth.SR
    tone = 0.5 * np.sin(2 * np.pi * hz * t).astype(np.float32)
    track = melody.track_f0(tone, synth.SR)
    stats = melody.f0_statistics(track)
    assert stats["voicing_ratio"] > 0.7
    assert abs(69 + 12 * np.log2(stats["median_hz"] / 440.0) - midi) < 0.3
