"""Tests for mambo.action.v1 (actions.py) — schema, refs, MIDI rendering."""

import pytest

from mambo_lab import actions
from mambo_lab.actions import Action, ActionPlan, ActionValidationError


def _uir():
    return {
        "schema": "mambo.utterance.v1",
        "utterance_id": "utt_x",
        "audio": {"sample_rate": 48000, "duration_s": 4.0},
        "segments": [
            {"kind": "speech", "t0": 0.0, "t1": 1.0, "confidence": 1.0, "text": "give me something like"},
            {"kind": "melody", "t0": 1.2, "t1": 3.6, "confidence": 0.9,
             "notes": [{"midi": 61, "name": "C#4", "t0": 1.2, "dur": 0.4, "vel": 92},
                       {"midi": 64, "name": "E4", "t0": 1.7, "dur": 0.4, "vel": 88}],
             "analysis": {"n_notes": 2, "tempo_bpm": {"value": 96, "confidence": 0.8}}},
        ],
    }


def test_worked_example_plan_validates():
    plan = ActionPlan(
        utterance_id="utt_20260610T172103Z_001",
        intent_summary="insert the hummed phrase, slower, on a warmer patch",
        actions=[
            Action("play_preview", {"notes_ref": "seg:1", "tempo_bpm": 80, "patch": "mellow_piano"}),
            Action("insert_notes", {"notes_ref": "seg:1", "tempo_bpm": 80, "track": {"by": "selected"}},
                   artifacts={"midi_file": "out/utt.mid"}),
            Action("ask_user", {"question": "Tempo 80 feel right, or slower?"},
                   when="analysis.tempo_bpm.confidence < 0.6"),
        ],
    )
    plan.validate()
    assert plan.needs_confirmation is True  # insert_notes is a confirm-op


def test_needs_confirmation_enforced():
    d = ActionPlan("u", "s", [Action("insert_notes", {"notes_ref": "seg:1", "tempo_bpm": 80})]).to_dict()
    d["needs_confirmation"] = False
    with pytest.raises(ActionValidationError):
        actions.validate(d)


def test_mixer_nudge_no_confirmation():
    plan = ActionPlan("u", "kick drums up", [Action("change_track_volume", {"delta_db": 2})])
    plan.validate()
    assert plan.needs_confirmation is False


def test_unknown_op_rejected():
    d = ActionPlan("u", "s", []).to_dict()
    d["actions"] = [{"op": "frobnicate", "args": {}}]
    with pytest.raises(ActionValidationError):
        actions.validate(d)


def test_bad_op_args_rejected():
    d = ActionPlan("u", "s", []).to_dict()
    d["actions"] = [{"op": "set_project_tempo", "args": {"bpm": "fast"}}]  # bpm must be number
    d["needs_confirmation"] = True
    with pytest.raises(ActionValidationError):
        actions.validate(d)


def test_resolve_notes_ref():
    notes = actions.resolve_notes_ref(_uir(), "seg:1")
    assert len(notes) == 2 and notes[0]["midi"] == 61
    with pytest.raises(ValueError):
        actions.resolve_notes_ref(_uir(), "seg:9")


def test_render_midi_roundtrip(tmp_path):
    import pretty_midi

    notes = actions.resolve_notes_ref(_uir(), "seg:1")
    path = str(tmp_path / "out.mid")
    actions.render_midi(notes, 96, path)
    pm = pretty_midi.PrettyMIDI(path)
    pitches = sorted(n.pitch for n in pm.instruments[0].notes)
    assert pitches == [61, 64]
    # first note shifted to t=0
    assert min(n.start for n in pm.instruments[0].notes) == pytest.approx(0.0, abs=1e-3)


def test_planner_tools_format():
    tools = actions.planner_tools()
    names = {t["function"]["name"] for t in tools}
    assert {"play_preview", "insert_notes", "ask_user"} <= names
    assert all(t["type"] == "function" for t in tools)
