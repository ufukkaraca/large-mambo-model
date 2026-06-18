"""Golden + invariant tests for mambo.utterance.v1 (ir.py)."""

import copy

import pytest

from mambo_lab import ir
from mambo_lab.ir import (
    F0Stats,
    KeyCandidate,
    MelodyAnalysis,
    Note,
    Segment,
    SessionContext,
    Track,
    UIRValidationError,
    Utterance,
    Word,
)


def _worked_example() -> Utterance:
    """The PAPER §2.3 / §4.6 worked example, built via the typed layer."""
    return Utterance(
        utterance_id="utt_20260610T172103Z_001",
        sample_rate=48000,
        duration_s=9.4,
        source="builtin_mic",
        segments=[
            Segment(
                kind="speech",
                t0=0.0,
                t1=2.10,
                confidence=0.97,
                role="instruction",
                text="give me something like",
                words=[Word("give", 0.0, 0.21), Word("me", 0.21, 0.34)],
                asr_engine="SpeechTranscriber",
                asr_lang="en-US",
            ),
            Segment(
                kind="melody",
                t0=2.10,
                t1=6.30,
                confidence=0.91,
                role="exemplar",
                notes=[Note(midi=61, t0=2.16, dur=0.42, vel=92, cents_dev=-14)],
                analysis=MelodyAnalysis(
                    n_notes=7,
                    key_candidates=[KeyCandidate("B minor", 0.71), KeyCandidate("D major", 0.66)],
                    tempo_bpm=96,
                    tempo_confidence=0.55,
                    contour="u u d u d d",
                ),
                f0=F0Stats(engine="pyin", voicing_ratio=0.94, median_hz=277.2),
            ),
            Segment(
                kind="speech",
                t0=6.30,
                t1=9.40,
                confidence=0.96,
                text="but slower maybe on something warmer",
            ),
        ],
        session_context=SessionContext(
            daw="garageband",
            selected_track=Track(2, "Drums", "drummer"),
            tracks=[Track(1, "Keys", "software_instrument")],
            project_tempo_bpm=120,
            project_key="C major",
            transport="stopped",
        ),
        history_refs=["utt_x_000"],
    )


def test_worked_example_validates():
    _worked_example().validate()


def test_schema_const_and_required():
    d = _worked_example().to_dict()
    assert d["schema"] == "mambo.utterance.v1"
    bad = copy.deepcopy(d)
    bad["schema"] = "mambo.utterance.v2"
    with pytest.raises(UIRValidationError):
        ir.validate(bad)


def test_midi_to_name():
    assert ir.midi_to_name(60) == "C4"
    assert ir.midi_to_name(61) == "C#4"
    assert ir.midi_to_name(69) == "A4"
    assert ir.midi_to_name(57) == "A3"


@pytest.mark.parametrize("hz,midi", [(440.0, 69.0), (261.63, 60.0), (880.0, 81.0)])
def test_hz_midi_roundtrip(hz, midi):
    assert ir.hz_to_midi_float(hz) == pytest.approx(midi, abs=0.02)


def test_containment_melody_with_text_is_invalid():
    """The hallucination-containment rule, enforced structurally."""
    d = _worked_example().to_dict()
    d["segments"][1]["text"] = "the moon is rising tonight"  # confabulation leak
    with pytest.raises(UIRValidationError):
        ir.validate(d)


def test_speech_with_notes_is_invalid():
    d = _worked_example().to_dict()
    d["segments"][0]["notes"] = [{"midi": 60, "name": "C4", "t0": 0.1, "dur": 0.2, "vel": 90}]
    with pytest.raises(UIRValidationError):
        ir.validate(d)


def test_note_name_must_match_midi():
    d = _worked_example().to_dict()
    d["segments"][1]["notes"][0]["name"] = "D4"  # wrong spelling for midi 61
    with pytest.raises(UIRValidationError):
        ir.validate(d)


def test_overlapping_segments_invalid():
    d = _worked_example().to_dict()
    d["segments"][1]["t0"] = 1.0  # overlaps the preceding speech span (ends 2.10)
    with pytest.raises(UIRValidationError):
        ir.validate(d)


def test_note_outside_segment_invalid():
    d = _worked_example().to_dict()
    d["segments"][1]["notes"][0]["t0"] = 0.0  # before the segment starts
    with pytest.raises(UIRValidationError):
        ir.validate(d)


def test_ambiguous_segment_carries_both_decodings():
    seg = Segment(
        kind="ambiguous",
        t0=0.0,
        t1=1.0,
        confidence=0.5,
        text="la la la",
        notes=[Note(midi=60, t0=0.0, dur=0.5)],
        analysis=MelodyAnalysis(n_notes=1),
    )
    utt = Utterance("utt_amb", 48000, 1.0, segments=[seg])
    utt.validate()


def test_additional_properties_rejected():
    d = _worked_example().to_dict()
    d["unexpected_field"] = True
    with pytest.raises(UIRValidationError):
        ir.validate(d)
