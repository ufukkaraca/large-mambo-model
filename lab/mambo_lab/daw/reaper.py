"""REAPER backend (producer side): drop a mambo.action.v1 plan + rendered MIDI
into an inbox folder that the REAPER Lua bridge (gb-bridge/reaper/mambo_bridge.lua)
watches and applies live.

This keeps the actuation decoupled (the brief's contract): perception+planning
emit the plan; the DAW layer just consumes it. No REAPER needed to test the
producer — `describe()` prints the actions, and `submit()` writes the inbox.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .. import actions

REPO = Path(__file__).resolve().parents[3]
INBOX = REPO / "out" / "reaper_inbox"


def submit(plan: dict[str, Any], uir: dict[str, Any], *, inbox: Path = INBOX) -> Path:
    """Render the plan's MIDI into the inbox and write the plan JSON beside it.
    Unique filename per submission so REAPER re-processes repeated commands.
    Returns the plan-file path."""
    inbox.mkdir(parents=True, exist_ok=True)
    sub_id = f"{plan.get('utterance_id', 'utt')}-{int(time.time() * 1000)}"
    # Point artifact MIDI into the inbox BEFORE rendering so render writes there.
    for a in plan.get("actions", []):
        if a.get("artifacts", {}).get("midi_file"):
            a["artifacts"]["midi_file"] = str(inbox / f"{sub_id}.mid")
    actions.render_plan_midi(plan, uir, out_dir=str(inbox))
    plan_path = inbox / f"{sub_id}.plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan_path


def finalize_live(plan: dict[str, Any], uir: dict[str, Any]) -> dict[str, Any]:
    """Make a plan decisive for LIVE use (take initiative — don't dead-end):
      * drop `ask_user` actions (pick a default and act);
      * if the utterance contains a hum but the plan never commits it, add the
        `insert_notes` (a preview alone leaves the track empty).
    The careful preview→confirm→ask flow stays the default elsewhere (R1 gate);
    live performance wants action, and the user corrects by voice.
    """
    uid = uir.get("utterance_id", "utt")
    segs = uir.get("segments", [])
    melody_ref = next((f"seg:{i}" for i, s in enumerate(segs)
                       if s["kind"] in ("melody", "ambiguous") and s.get("notes")), None)
    acts = [a for a in plan.get("actions", []) if a["op"] != "ask_user"]
    if melody_ref and not any(a["op"] == "insert_notes" for a in acts):
        tempo = next((a["args"].get("tempo_bpm") for a in acts
                      if a["op"] == "play_preview" and a["args"].get("tempo_bpm")), None)
        tempo = tempo or _melody_tempo(uir, melody_ref) or 100.0
        acts.append({"op": "insert_notes",
                     "args": {"notes_ref": melody_ref, "tempo_bpm": float(tempo), "track": {"by": "selected"}},
                     "artifacts": {"midi_file": f"out/{uid}.mid"}})
    plan["actions"] = acts
    plan["needs_confirmation"] = any(a["op"] in actions.CONFIRM_OPS for a in acts)
    # P3 dual-decode: if the hummed span was a SUNG demonstration, carry its captured
    # lyric as a live-only plan annotation (NOT part of mambo.action.v1) so the bridge
    # can name the inserted-MIDI region/take with it. Bridge-side naming is a thin Lua
    # follow-up; the lyric reaches the inbox here.
    if melody_ref:
        mi = int(melody_ref.split(":")[1])
        if mi < len(segs) and segs[mi].get("kind") == "ambiguous" and segs[mi].get("text"):
            plan["lyric"] = segs[mi]["text"]
    return plan


def _melody_tempo(uir: dict[str, Any], ref: str) -> Optional[float]:
    i = int(ref.split(":")[1])
    segs = uir.get("segments", [])
    if i < len(segs):
        t = (segs[i].get("analysis", {}).get("tempo_bpm") or {}).get("value")
        return float(t) if t else None
    return None


def _tname(a: dict[str, Any]) -> Optional[str]:
    """Target track name for an action (volume/instrument nest under .track;
    mute/solo carry `by` directly), or None for the selected/Mambo track."""
    t = a.get("track", a)
    by = t.get("by") if isinstance(t, dict) else None
    return by if (by and by != "selected") else None


_VERB = {
    "play_preview": lambda a: f"preview the notes at {a.get('tempo_bpm','?')} bpm (co-pilot synth)",
    "insert_notes": lambda a: f"insert the hummed notes onto the Mambo track at {a.get('tempo_bpm','?')} bpm",
    "insert_drum_pattern": lambda a: "insert the detected drum pattern",
    "set_track_instrument": lambda a: f"change the {_tname(a)+' ' if _tname(a) else ''}instrument to {a.get('patch','?')}",
    "change_track_volume": lambda a: f"nudge the {_tname(a) or 'selected'} fader {a.get('delta_db','?'):+g} dB",
    "transport": lambda a: f"transport: {a.get('action','?')}",
    "mute_track": lambda a: f"mute the {_tname(a) or 'selected'} track",
    "solo_track": lambda a: f"solo the {_tname(a) or 'selected'} track",
    "pan_track": lambda a: f"pan the {_tname(a) or 'selected'} track {'left' if a.get('delta_pan', 0) < 0 else 'right'}",
    "undo": lambda a: "undo the last change",
    "record_take": lambda a: f"record a take on the {_tname(a) or 'selected'} track",
    "create_track": lambda a: f"create a {a.get('kind','?')} track",
    "set_project_tempo": lambda a: f"set tempo to {a.get('bpm','?')} bpm",
    "ask_user": lambda a: f"ASK: {a.get('question','?')}",
}


def describe(plan: dict[str, Any]) -> str:
    """Human-readable preview of what REAPER will do (dry-run)."""
    lines = [f"intent: {plan.get('intent_summary','')}"]
    for a in plan.get("actions", []):
        fn = _VERB.get(a["op"], lambda x: a["op"])
        try:
            lines.append(f"  • {fn(a.get('args', {}))}")
        except Exception:
            lines.append(f"  • {a['op']} {a.get('args', {})}")
    if plan.get("needs_confirmation"):
        lines.append("  (needs confirmation before audible-in-project actions)")
    return "\n".join(lines)
