"""mambo.action.v1 — the action plan (PAPER §4.7), the core deliverable of R1.

The planner's tool calls are recorded as a structured, replayable plan,
deliberately decoupled from any DAW (build priority: understanding → execution →
application). Note-bearing ops are additionally rendered to standard MIDI files,
so the core output is testable and immediately useful (drag the `.mid` into
GarageBand) before any DAW adapter exists.

This module owns: the tool catalog (shared with the planner's tool definitions),
the action-plan JSON schema + validation, `notes_ref` resolution into a UIR, and
`.mid` rendering via `pretty_midi`. The two schemas split the system cleanly:
`mambo.utterance.v1` is *what was meant*, `mambo.action.v1` is *what to do*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import jsonschema

from . import ir

SCHEMA_VERSION = "mambo.action.v1"

# --------------------------------------------------------------------------- #
# Tool catalog (PAPER §4.7). One source of truth -> validation schema AND the
# planner's tool definitions. `confirm` marks audible-in-project actions that
# carry needs_confirmation (ground rule #3): insert/record/tempo confirm; mixer
# nudges do not.
# --------------------------------------------------------------------------- #

TOOLS: dict[str, dict[str, Any]] = {
    "play_preview": {
        "description": "Audition notes through the co-pilot's own synth (NOT the DAW) for confirmation.",
        "confirm": False,
        "params": {
            "type": "object",
            "properties": {
                "notes_ref": {"type": "string", "description": "e.g. 'seg:1' — a melody segment in the UIR"},
                "tempo_bpm": {"type": "number"},
                "patch": {"type": "string"},
            },
            "required": ["notes_ref", "tempo_bpm"],
            "additionalProperties": False,
        },
    },
    "insert_notes": {
        "description": "Insert the referenced notes onto a track (renders a .mid). Audible in project.",
        "confirm": True,
        "params": {
            "type": "object",
            "properties": {
                "notes_ref": {"type": "string"},
                "tempo_bpm": {"type": "number"},
                "track": {"type": "object", "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                          "additionalProperties": False},
            },
            "required": ["notes_ref", "tempo_bpm"],
            "additionalProperties": False,
        },
    },
    "transport": {
        "description": "Transport control.",
        "confirm": False,  # record is confirmed via the action's own needs_confirmation
        "params": {
            "type": "object",
            "properties": {"action": {"enum": ["play", "stop", "record", "to_start", "cycle"]}},
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "select_track": {
        "description": "Select a track by index.",
        "confirm": False,
        "params": {"type": "object", "properties": {"index": {"type": "integer"}},
                   "required": ["index"], "additionalProperties": False},
    },
    "mute_track": {
        "description": "Mute a track (selected if no index).",
        "confirm": False,
        "params": {"type": "object", "properties": {"index": {"type": "integer"}}, "additionalProperties": False},
    },
    "solo_track": {
        "description": "Solo a track (selected if no index).",
        "confirm": False,
        "params": {"type": "object", "properties": {"index": {"type": "integer"}}, "additionalProperties": False},
    },
    "change_track_volume": {
        "description": "Relative fader move in dB (e.g. +2). Cheap to reverse — executes immediately.",
        "confirm": False,
        "params": {
            "type": "object",
            "properties": {"delta_db": {"type": "number"},
                           "track": {"type": "object", "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                                     "additionalProperties": False}},
            "required": ["delta_db"], "additionalProperties": False,
        },
    },
    "create_track": {
        "description": "Create a new track of the given kind.",
        "confirm": True,
        "params": {"type": "object", "properties": {"kind": {"type": "string"}},
                   "required": ["kind"], "additionalProperties": False},
    },
    "set_project_tempo": {
        "description": "Set the project tempo. Audible in project.",
        "confirm": True,
        "params": {"type": "object", "properties": {"bpm": {"type": "number"}},
                   "required": ["bpm"], "additionalProperties": False},
    },
    "ask_user": {
        "description": "Ask the user to disambiguate (low confidence). Use when tempo conf < 0.6 or keys tie.",
        "confirm": False,
        "params": {
            "type": "object",
            "properties": {"question": {"type": "string"},
                           "options": {"type": "array", "items": {"type": "string"}}},
            "required": ["question"], "additionalProperties": False,
        },
    },
    "set_track_instrument": {
        "description": "Change a track's instrument/patch (e.g. 'electric_piano', 'strings', 'synth'). Audible in project. REAPER-capable (GarageBand cannot script this).",
        "confirm": True,
        "params": {
            "type": "object",
            "properties": {"patch": {"type": "string"},
                           "track": {"type": "object",
                                     "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                                     "additionalProperties": False}},
            "required": ["patch"], "additionalProperties": False,
        },
    },
    "insert_drum_pattern": {
        "description": "Insert the detected beatbox drum pattern (the UIR percussion[]) onto a drum track. Audible in project.",
        "confirm": True,
        "params": {
            "type": "object",
            "properties": {"track": {"type": "object",
                                     "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                                     "additionalProperties": False}},
            "additionalProperties": False,
        },
    },
    "search_local_loops": {
        "description": "Search the local Apple Loops index (audition only; insertion stays manual).",
        "confirm": False,
        "params": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "melody_ref": {"type": "string"}},
            "required": ["query"], "additionalProperties": False,
        },
    },
    "pan_track": {
        "description": "Pan a track. delta_pan in [-1, 1] added to the current pan (negative = left). Cheap to reverse.",
        "confirm": False,
        "params": {
            "type": "object",
            "properties": {"delta_pan": {"type": "number"},
                           "track": {"type": "object", "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                                     "additionalProperties": False}},
            "required": ["delta_pan"], "additionalProperties": False,
        },
    },
    "undo": {
        "description": "Undo the last applied change in the DAW (e.g. 'undo that', 'never mind').",
        "confirm": False,
        "params": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "record_take": {
        "description": "Arm + record a new take of a track; logs it to the session take history.",
        "confirm": True,
        "params": {
            "type": "object",
            "properties": {"track": {"type": "object",
                                     "properties": {"by": {"type": "string"}, "index": {"type": "integer"}},
                                     "additionalProperties": False},
                           "label": {"type": "string"},
                           "section": {"type": "string"}},
            "additionalProperties": False,
        },
    },
}

CONFIRM_OPS = {name for name, t in TOOLS.items() if t["confirm"]}


def planner_tools() -> list[dict[str, Any]]:
    """The tool catalog in OpenAI/OpenRouter function-calling format."""
    return [
        {"type": "function",
         "function": {"name": name, "description": t["description"], "parameters": t["params"]}}
        for name, t in TOOLS.items()
    ]


# --------------------------------------------------------------------------- #
# Action-plan schema.
# --------------------------------------------------------------------------- #

ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": SCHEMA_VERSION,
    "type": "object",
    "required": ["schema", "utterance_id", "intent_summary", "needs_confirmation", "actions"],
    "properties": {
        "schema": {"const": SCHEMA_VERSION},
        "utterance_id": {"type": "string", "minLength": 1},
        "intent_summary": {"type": "string"},
        "needs_confirmation": {"type": "boolean"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op", "args"],
                "properties": {
                    "op": {"enum": list(TOOLS)},
                    "args": {"type": "object"},
                    "artifacts": {"type": "object",
                                  "properties": {"midi_file": {"type": "string"}},
                                  "additionalProperties": False},
                    "when": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

_VALIDATOR = jsonschema.Draft202012Validator(ACTION_SCHEMA)


class ActionValidationError(ValueError):
    pass


def validate(doc: dict[str, Any]) -> None:
    errors = sorted(_VALIDATOR.iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.path) or "<root>"
        raise ActionValidationError(f"{loc}: {e.message}")
    # per-op arg validation against the tool catalog
    for i, a in enumerate(doc["actions"]):
        op = a["op"]
        sub = jsonschema.Draft202012Validator(TOOLS[op]["params"])
        suberr = sorted(sub.iter_errors(a["args"]), key=lambda e: list(e.path))
        if suberr:
            raise ActionValidationError(f"actions/{i} ({op}): {suberr[0].message}")
    # needs_confirmation must be true if any action is a confirm-op
    if any(a["op"] in CONFIRM_OPS for a in doc["actions"]) and not doc["needs_confirmation"]:
        raise ActionValidationError("needs_confirmation must be true when the plan contains a confirm-op")


# --------------------------------------------------------------------------- #
# Builder + notes_ref resolution + MIDI rendering.
# --------------------------------------------------------------------------- #


@dataclass
class Action:
    op: str
    args: dict[str, Any]
    artifacts: Optional[dict[str, str]] = None
    when: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "args": self.args}
        if self.artifacts:
            d["artifacts"] = self.artifacts
        if self.when:
            d["when"] = self.when
        return d


@dataclass
class ActionPlan:
    utterance_id: str
    intent_summary: str
    actions: list[Action] = field(default_factory=list)

    @property
    def needs_confirmation(self) -> bool:
        return any(a.op in CONFIRM_OPS for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "utterance_id": self.utterance_id,
            "intent_summary": self.intent_summary,
            "needs_confirmation": self.needs_confirmation,
            "actions": [a.to_dict() for a in self.actions],
        }

    def validate(self) -> "ActionPlan":
        validate(self.to_dict())
        return self

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


_SEG_REF = re.compile(r"^seg:(\d+)$")


def resolve_notes_ref(uir: dict[str, Any], ref: str) -> list[dict[str, Any]]:
    """Resolve 'seg:N' -> the notes of the N-th segment in the UIR."""
    m = _SEG_REF.match(ref or "")
    if not m:
        raise ValueError(f"bad notes_ref {ref!r} (expected 'seg:<index>')")
    idx = int(m.group(1))
    segs = uir.get("segments", [])
    if idx >= len(segs):
        raise ValueError(f"notes_ref {ref} out of range ({len(segs)} segments)")
    return segs[idx].get("notes", [])


def render_midi(notes: list[dict[str, Any]], tempo_bpm: float, path: str) -> str:
    """Render notes to a standard MIDI file at ``tempo_bpm`` (velocities from the
    UIR). Note times are taken as written (global utterance time); the first note
    is shifted to t=0 so the region starts at the bar."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    inst = pretty_midi.Instrument(program=0)
    t0 = min((n["t0"] for n in notes), default=0.0)
    for n in notes:
        start = n["t0"] - t0
        inst.notes.append(pretty_midi.Note(
            velocity=int(n.get("vel", 90)), pitch=int(n["midi"]),
            start=round(start, 4), end=round(start + n["dur"], 4)))
    pm.instruments.append(inst)
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    pm.write(path)
    return path


def render_plan_midi(plan: dict[str, Any], uir: dict[str, Any], out_dir: str = "out") -> dict[str, str]:
    """Render the .mid for every note-bearing action that declares an artifact."""
    written: dict[str, str] = {}
    for a in plan["actions"]:
        mf = a.get("artifacts", {}).get("midi_file")
        if not mf or "notes_ref" not in a["args"]:
            continue
        notes = resolve_notes_ref(uir, a["args"]["notes_ref"])
        tempo = a["args"].get("tempo_bpm") or uir_tempo(uir) or 120
        render_midi(notes, tempo, mf)
        written[a["args"]["notes_ref"]] = mf
    return written


def uir_tempo(uir: dict[str, Any]) -> Optional[float]:
    for s in uir.get("segments", []):
        t = s.get("analysis", {}).get("tempo_bpm", {})
        if isinstance(t, dict) and t.get("value"):
            return t["value"]
    return None


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    validate(doc)
    return doc


def dump(doc: dict[str, Any], path: str, *, indent: int = 2) -> None:
    validate(doc)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=indent, ensure_ascii=False)
        f.write("\n")
