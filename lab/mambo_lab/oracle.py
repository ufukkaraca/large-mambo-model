"""Deterministic reference planner — produces the GOLDEN action plans the R1
gate scores the LLM planner against.

This is the "right answer" encoder: a rule-based UIR+context -> mambo.action.v1
mapping over the known template intents. It is NOT the product (the LLM planner
is); it is the eval oracle and a sanity baseline. Keeping it deterministic means
the golden suite is regenerable bit-exact (seed traceability).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .actions import Action, ActionPlan

# Command-noun -> track name (resolved to an index via session_context.tracks).
TRACK_NOUNS = {
    "drums": "Drums", "kick": "Drums", "hi-hats": "Hats", "hi-hat": "Hats",
    "hats": "Hats", "hat": "Hats", "bass": "Bass", "keys": "Keys", "key": "Keys",
    "lead": "Lead", "vocal": "Vocal", "reverb": "Vocal", "guitar": "Guitar",
}

DEFAULT_TRACKS = [
    {"index": 0, "name": "Drums", "kind": "drummer"},
    {"index": 1, "name": "Bass", "kind": "software_instrument"},
    {"index": 2, "name": "Keys", "kind": "software_instrument"},
    {"index": 3, "name": "Lead", "kind": "software_instrument"},
    {"index": 4, "name": "Vocal", "kind": "audio"},
    {"index": 5, "name": "Guitar", "kind": "audio"},
    {"index": 6, "name": "Hats", "kind": "drummer"},
]


def default_session_context() -> dict[str, Any]:
    return {"daw": "garageband", "selected_track": DEFAULT_TRACKS[2],
            "tracks": DEFAULT_TRACKS, "project_tempo_bpm": 120,
            "project_key": "C major", "transport": "stopped"}


def _track_index(noun_text: str, ctx: dict) -> Optional[int]:
    tracks = ctx.get("tracks", [])
    by_name = {t["name"]: t["index"] for t in tracks}
    for noun, name in TRACK_NOUNS.items():
        if noun in noun_text and name in by_name:
            return by_name[name]
    return None


def _track_target(noun_text: str, ctx: dict) -> Optional[dict]:
    """Resolve a command noun ("drums", "bass", …) to a track target carrying the
    track NAME (`by`) so the REAPER bridge can find it by name regardless of track
    order; `index` is the session-context fallback. None = no track named."""
    tracks = ctx.get("tracks", [])
    by_name = {t["name"]: t["index"] for t in tracks}
    for noun, name in TRACK_NOUNS.items():
        if noun in noun_text and name in by_name:
            return {"by": name, "index": by_name[name]}
    return None


def oracle_plan(uir: dict[str, Any], ctx: Optional[dict] = None) -> ActionPlan:
    ctx = ctx or uir.get("session_context") or default_session_context()
    uid = uir.get("utterance_id", "utt")
    segs = uir.get("segments", [])
    text = " ".join(s.get("text", "") for s in segs if s["kind"] == "speech").lower()

    melody_idx = [i for i, s in enumerate(segs)
                  if s["kind"] in ("melody", "ambiguous") and s.get("role") in (None, "exemplar")]
    if not melody_idx:
        melody_idx = [i for i, s in enumerate(segs) if s["kind"] in ("melody", "ambiguous")]

    if melody_idx:
        return _sketch_plan(uid, segs, melody_idx[0], text)
    return _command_plan(uid, text, ctx)


def _sketch_plan(uid: str, segs: list, i: int, text: str) -> ActionPlan:
    seg = segs[i]
    an = seg.get("analysis", {})
    tempo = (an.get("tempo_bpm") or {}).get("value") or 100
    conf = (an.get("tempo_bpm") or {}).get("confidence", 1.0)
    if "slower" in text:
        tempo = round(tempo * 0.8)
    elif "faster" in text:
        tempo = round(tempo * 1.2)
    ref = f"seg:{i}"
    acts = [
        Action("play_preview", {"notes_ref": ref, "tempo_bpm": float(tempo)}),
        Action("insert_notes", {"notes_ref": ref, "tempo_bpm": float(tempo), "track": {"by": "selected"}},
               artifacts={"midi_file": f"out/{uid}.mid"}),
    ]
    intent = f"insert the hummed phrase at {tempo} bpm"
    keys = an.get("key_candidates", [])
    if len(keys) >= 2 and abs(keys[0]["score"] - keys[1]["score"]) < 0.1:
        intent += f" ({keys[0]['key']} or {keys[1]['key']})"
    if conf < 0.6:
        acts.append(Action("ask_user", {"question": f"Tempo around {tempo} BPM?"}))
    return ActionPlan(uid, intent, acts)


def _command_plan(uid: str, text: str, ctx: dict) -> ActionPlan:
    tgt = _track_target(text, ctx)       # {"by": name, "index": i} or None
    track = tgt or {}

    if any(p in text for p in ("undo", "never mind", "scratch that", "take that back")):
        return ActionPlan(uid, "undo the last change", [Action("undo", {})])

    # intensity: "a bit" -> 2 dB, plain -> 3 dB, "a lot/way/much" -> 6 dB
    mag = 6.0 if any(w in text for w in ("a lot", "way ", "much", "lots", "loads")) \
        else (2.0 if any(w in text for w in ("a bit", "a little", "slightly", "touch", "tad")) else 3.0)

    def vol(sign):
        a = {"delta_db": sign * mag}
        if tgt:
            a["track"] = tgt
        return Action("change_track_volume", a)

    instruments = {"electric": "electric_piano", "strings": "strings", "synth": "synth",
                   "pad": "warm_pad", "warmer": "warm_pad", "warm": "warm_pad",
                   "bright": "bright_piano", "grand": "piano"}
    for w, patch in instruments.items():
        if w in text:
            args = {"patch": patch}
            if tgt:
                args["track"] = tgt
            where = f" on {tgt['by']}" if tgt else ""
            return ActionPlan(uid, f"change the instrument to {patch}{where}",
                              [Action("set_track_instrument", args)])
    if "loop" in text:
        return ActionPlan(uid, "loop the selection", [Action("transport", {"action": "cycle"})])
    if "mute" in text:
        return ActionPlan(uid, "mute a track", [Action("mute_track", track)])
    if "solo" in text:
        return ActionPlan(uid, "solo a track", [Action("solo_track", track)])
    if any(p in text for p in ("record a take", "record me", "record a few", "lay down a take",
                               "one more", "another take", "give me a take", "roll it", "let's roll")):
        a = {"track": tgt} if tgt else {}
        return ActionPlan(uid, f"record a take{' on ' + tgt['by'] if tgt else ''}",
                          [Action("record_take", a)])
    if "record" in text:  # "start recording on the selected track"
        return ActionPlan(uid, "arm and record", [Action("transport", {"action": "record"})])
    if "back to the start" in text or "go to the start" in text:
        return ActionPlan(uid, "return to start and play",
                          [Action("transport", {"action": "to_start"}), Action("transport", {"action": "play"})])
    if "stop" in text or "pause" in text:
        return ActionPlan(uid, "stop", [Action("transport", {"action": "stop"})])
    if "add" in text and "track" in text:
        return ActionPlan(uid, "create a track", [Action("create_track", {"kind": "software_instrument"})])
    if "loop" in text:
        return ActionPlan(uid, "toggle cycle", [Action("transport", {"action": "cycle"})])
    if "tempo" in text or "bpm" in text:
        m = re.search(r"(?:to|at|of)\s+(\d{2,3})|(\d{2,3})\s*bpm", text)
        if m:
            bpm = float(m.group(1) or m.group(2))
        else:  # relative: "double", "faster"/"up", else nudge down
            cur = ctx.get("project_tempo_bpm", 120)
            bpm = cur * 2 if "double" in text else (round(cur * 1.1) if any(w in text for w in ("faster", "up", "quicker")) else round(cur * 0.9))
        return ActionPlan(uid, f"set tempo to {bpm:g}", [Action("set_project_tempo", {"bpm": float(bpm)})])
    if "pan" in text:
        side = -1.0 if "left" in text else (1.0 if "right" in text else 0.0)
        if side == 0.0:
            return ActionPlan(uid, "pan which way?", [Action("ask_user", {"question": "Pan left or right?"})])
        amt = 0.5 if any(w in text for w in ("a lot", "way", "hard", "full")) else 0.3
        a = {"delta_pan": side * amt}
        if tgt:
            a["track"] = tgt
        return ActionPlan(uid, f"pan {'left' if side < 0 else 'right'}", [Action("pan_track", a)])
    if any(w in text for w in ("up", "louder")):
        return ActionPlan(uid, "raise a fader", [vol(1.0)])
    if any(w in text for w in ("down", "quieter", "lower")):
        return ActionPlan(uid, "lower a fader", [vol(-1.0)])
    if "play" in text:  # late: specific commands ("play louder") win first
        return ActionPlan(uid, "play", [Action("transport", {"action": "play"})])
    return ActionPlan(uid, "unrecognized; ask", [Action("ask_user", {"question": "Could you rephrase that?"})])
