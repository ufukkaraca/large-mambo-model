"""Tests for the REAPER producer (no REAPER needed: plan -> describe + inbox)."""

import json

from mambo_lab import actions, oracle
from mambo_lab.daw import reaper


def _uir_hum():
    return {
        "schema": "mambo.utterance.v1", "utterance_id": "t_hum",
        "audio": {"sample_rate": 48000, "duration_s": 2.0},
        "segments": [{"kind": "melody", "t0": 0.0, "t1": 2.0, "confidence": 0.9, "role": "exemplar",
                      "notes": [{"midi": 60, "name": "C4", "t0": 0.0, "dur": 0.5, "vel": 90},
                                {"midi": 64, "name": "E4", "t0": 0.6, "dur": 0.5, "vel": 90}],
                      "analysis": {"n_notes": 2, "tempo_bpm": {"value": 100, "confidence": 0.8}}}],
        "session_context": oracle.default_session_context(),
    }


def test_describe_hum_plan():
    plan = oracle.oracle_plan(_uir_hum()).to_dict()
    text = reaper.describe(plan)
    assert "insert" in text.lower()


def test_submit_writes_plan_and_midi(tmp_path):
    uir = _uir_hum()
    plan = oracle.oracle_plan(uir).to_dict()
    inbox = tmp_path / "inbox"
    path = reaper.submit(plan, uir, inbox=inbox)
    assert path.exists() and path.name.endswith(".plan.json")
    written = json.loads(path.read_text())
    mids = [a["artifacts"]["midi_file"] for a in written["actions"] if a.get("artifacts")]
    assert mids and all((inbox / __import__("pathlib").Path(m).name).exists() for m in mids)


def test_finalize_live_is_decisive():
    # the live failure shape: preview + ask, no insert
    uir = {"utterance_id": "live", "segments": [
        {"kind": "speech", "t0": 0, "t1": 1, "confidence": 1, "text": "give me something like that"},
        {"kind": "melody", "t0": 1.2, "t1": 4, "confidence": 0.9,
         "notes": [{"midi": 60, "name": "C4", "t0": 1.2, "dur": 0.4, "vel": 90}],
         "analysis": {"n_notes": 5, "tempo_bpm": {"value": 92, "confidence": 0.4}}}]}
    plan = {"schema": "mambo.action.v1", "utterance_id": "live", "intent_summary": "x",
            "needs_confirmation": True, "actions": [
                {"op": "play_preview", "args": {"notes_ref": "seg:1", "tempo_bpm": 92}},
                {"op": "ask_user", "args": {"question": "tempo?"}}]}
    out = reaper.finalize_live(plan, uir)
    ops = [a["op"] for a in out["actions"]]
    assert "ask_user" not in ops          # took initiative
    assert "insert_notes" in ops           # committed the hum
    ins = next(a for a in out["actions"] if a["op"] == "insert_notes")
    assert ins["args"]["notes_ref"] == "seg:1" and ins["args"]["tempo_bpm"] == 92.0


def test_finalize_live_keeps_commands():
    # a pure command (no melody): just drop any ask, keep the action
    uir = {"utterance_id": "c", "segments": [{"kind": "speech", "t0": 0, "t1": 1, "confidence": 1, "text": "mute"}]}
    plan = {"schema": "mambo.action.v1", "utterance_id": "c", "intent_summary": "x",
            "needs_confirmation": False, "actions": [{"op": "mute_track", "args": {}}]}
    out = reaper.finalize_live(plan, uir)
    assert [a["op"] for a in out["actions"]] == ["mute_track"]


def test_text_command_plans():
    for text, op in [("make it electric", "set_track_instrument"),
                     ("loop that", "transport"),
                     ("kick the drums up", "change_track_volume")]:
        uir = {"schema": "mambo.utterance.v1", "utterance_id": "c",
               "audio": {"sample_rate": 16000, "duration_s": 0.0},
               "segments": [{"kind": "speech", "t0": 0.0, "t1": 1.0, "confidence": 1.0, "text": text}],
               "session_context": oracle.default_session_context()}
        ops = [a.op for a in oracle.oracle_plan(uir).actions]
        assert op in ops, f"{text!r} -> {ops}, expected {op}"


def _plan(text):
    uir = {"schema": "mambo.utterance.v1", "utterance_id": "c",
           "audio": {"sample_rate": 16000, "duration_s": 0.0},
           "segments": [{"kind": "speech", "t0": 0.0, "t1": 1.0, "confidence": 1.0, "text": text}],
           "session_context": oracle.default_session_context()}
    return oracle.oracle_plan(uir)


def test_new_capabilities_pan_undo_intensity():
    for text, op in [("pan the bass left", "pan_track"), ("undo that", "undo"),
                     ("never mind", "undo"), ("turn the drums way up", "change_track_volume")]:
        assert op in [a.op for a in _plan(text).actions], f"{text!r}"
    # volume intensity: "way up" moves more than "up a bit"
    bit = next(a for a in _plan("turn the bass up a bit").actions if a.op == "change_track_volume")
    lot = next(a for a in _plan("turn the bass way up").actions if a.op == "change_track_volume")
    assert abs(lot.args["delta_db"]) > abs(bit.args["delta_db"])
    # pan direction + plans validate against the (extended) action schema
    left = next(a for a in _plan("pan the bass left").actions if a.op == "pan_track")
    assert left.args["delta_pan"] < 0
    actions.validate(_plan("pan the bass left").to_dict())
    actions.validate(_plan("undo that").to_dict())


def test_transport_and_explicit_tempo():
    stop = next(a for a in _plan("stop the playback").actions if a.op == "transport")
    assert stop.args["action"] == "stop"
    play = next(a for a in _plan("play").actions if a.op == "transport")
    assert play.args["action"] == "play"
    t = next(a for a in _plan("set the tempo to 128").actions if a.op == "set_project_tempo")
    assert t.args["bpm"] == 128.0


def test_command_targets_named_track():
    # multitrack: a command noun resolves to its track (by name) so the bridge
    # routes the op to the right track, not always the Mambo track.
    cases = [("kick the drums up a bit", "change_track_volume", "Drums"),
             ("make the bass a little louder", "change_track_volume", "Bass"),
             ("mute the keys", "mute_track", "Keys"),
             ("make the bass electric", "set_track_instrument", "Bass")]
    for text, op, name in cases:
        uir = {"schema": "mambo.utterance.v1", "utterance_id": "c",
               "audio": {"sample_rate": 16000, "duration_s": 0.0},
               "segments": [{"kind": "speech", "t0": 0.0, "t1": 1.0, "confidence": 1.0, "text": text}],
               "session_context": oracle.default_session_context()}
        act = next(a for a in oracle.oracle_plan(uir).actions if a.op == op)
        tgt = act.args.get("track", act.args)  # volume/instrument nest under .track; mute/solo direct
        assert tgt.get("by") == name, f"{text!r} -> {act.args}"


def test_record_take_command():
    p = _plan("record me a take on the bass")
    act = next(a for a in p.actions if a.op == "record_take")
    assert act.args["track"]["by"] == "Bass"
    actions.validate(p.to_dict())
    assert any(a.op == "record_take" for a in _plan("give me a take").actions)
    # "start recording …" stays plain transport record, not a logged take
    p3 = _plan("start recording on the selected track")
    assert any(a.op == "transport" and a.args.get("action") == "record" for a in p3.actions)
    assert not any(a.op == "record_take" for a in p3.actions)
