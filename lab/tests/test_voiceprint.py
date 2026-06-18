"""Per-user voiceprint — the calibration mapping (no audio needed for the core)."""

from mambo_lab import melody
from mambo_lab.voiceprint import DEFAULT, Voiceprint


def test_default_is_neutral():
    # the shipped default must be a no-op so existing gates are unchanged
    assert DEFAULT.pitch_step == 1.0
    assert DEFAULT.f0_min == melody.FMIN and DEFAULT.f0_max == melody.FMAX


def test_pitch_step_deadband():
    # normal-vibrato voices (the held-out voices measure 0.60/0.68 st) stay at the
    # 1.0 default, so calibration never over-merges a clean voice. Deadband edge is V0=1.0 st.
    assert Voiceprint(vibrato_semitones=0.40).pitch_step == 1.0
    assert Voiceprint(vibrato_semitones=0.68).pitch_step == 1.0
    assert Voiceprint(vibrato_semitones=1.00).pitch_step == 1.0  # edge of the deadband
    # only an abnormally wide-vibrato voice (Rim ≈ 3.2 st, ~5× normal) gets a looser
    # split threshold so its wobble stops splitting held notes
    assert Voiceprint(vibrato_semitones=3.24).pitch_step > 2.0
    # monotonic above the deadband + capped at 2.5
    assert Voiceprint(vibrato_semitones=1.5).pitch_step < Voiceprint(vibrato_semitones=2.0).pitch_step
    assert Voiceprint(vibrato_semitones=5.0).pitch_step == 2.5


def test_round_trip_and_from_none():
    vp = Voiceprint(f0_min=120.0, f0_max=500.0, vibrato_semitones=0.7, speech_voicing=0.8, label="x")
    back = Voiceprint.from_dict(vp.to_dict())
    assert back.f0_max == 500.0 and back.vibrato_semitones == 0.7
    assert Voiceprint.from_dict(None) is DEFAULT       # missing → shipped default
    assert Voiceprint.from_dict({}).pitch_step == 1.0  # empty → neutral
