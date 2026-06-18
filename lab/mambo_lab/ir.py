"""mambo.utterance.v1 — the Utterance Intermediate Representation (UIR).

The keystone artifact (PAPER §4.6). It is, simultaneously:
  (a) the planner's input,
  (b) the evaluation target,
  (c) the fine-tune's output format, and
  (d) the log/replay format.

This module owns the schema. Two layers are provided:
  * A JSON Schema (`UTTERANCE_SCHEMA`) + ``validate()`` — the contract enforcer.
  * Typed dataclasses + ``Utterance.to_dict()/from_dict()`` — the ergonomic
    builder the pipeline and fixtures use.

Schema rules that matter (PAPER §4.6):
  * Timestamps are GLOBAL to the utterance (deixis resolution).
  * Ambiguity is first-class: key candidates and tempo carry confidences;
    a segment may be ``kind="ambiguous"`` carrying BOTH a speech and a melody
    decoding.
  * ``session_context`` is captured at utterance time.

Ground rule (the ASR-is-evidence containment rule): ASR output is evidence, never truth. A melody or
ambiguous segment's *committed* ``text`` must come only from a span that passed
the speech-verification gate; ``melody`` segments therefore carry no ``text``.
``validate()`` enforces this structurally (a ``melody`` segment with a ``text``
field is invalid).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import jsonschema

SCHEMA_VERSION = "mambo.utterance.v1"

SegmentKind = Literal["speech", "melody", "ambiguous"]
SegmentRole = Literal["instruction", "exemplar", "contrast", "filler"]

# Drum classes used across the percussion path (PAPER §4.5 / Phase R2).
DRUM_CLASSES = ("kick", "snare", "hat")

# --------------------------------------------------------------------------- #
# JSON Schema — the contract enforcer.
# --------------------------------------------------------------------------- #

_WORD = {
    "type": "object",
    "required": ["w", "t0", "t1"],
    "properties": {
        "w": {"type": "string"},
        "t0": {"type": "number", "minimum": 0},
        "t1": {"type": "number", "minimum": 0},
        # probe-time evidence features (PAPER §4.3) — optional, never truth
        "logprob": {"type": "number"},
    },
    "additionalProperties": False,
}

_NOTE = {
    "type": "object",
    "required": ["midi", "name", "t0", "dur", "vel"],
    "properties": {
        "midi": {"type": "integer", "minimum": 0, "maximum": 127},
        "name": {"type": "string"},
        "t0": {"type": "number", "minimum": 0},
        "dur": {"type": "number", "exclusiveMinimum": 0},
        "vel": {"type": "integer", "minimum": 1, "maximum": 127},
        "cents_dev": {"type": "number"},
    },
    "additionalProperties": False,
}

_KEY_CANDIDATE = {
    "type": "object",
    "required": ["key", "score"],
    "properties": {
        "key": {"type": "string"},
        "score": {"type": "number"},
    },
    "additionalProperties": False,
}

_ANALYSIS = {
    "type": "object",
    "required": ["n_notes"],
    "properties": {
        "key_candidates": {"type": "array", "items": _KEY_CANDIDATE, "maxItems": 8},
        "tempo_bpm": {
            "type": "object",
            "required": ["value", "confidence"],
            "properties": {
                "value": {"type": ["number", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
        "contour": {"type": "string"},
        "n_notes": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

_F0 = {
    "type": "object",
    "required": ["engine", "voicing_ratio"],
    "properties": {
        "engine": {"type": "string"},
        "voicing_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "median_hz": {"type": ["number", "null"]},
        # f0-stability feature used by the acoustic gate (PAPER §4.3)
        "f0_stability": {"type": "number"},
    },
    "additionalProperties": False,
}

# A segment is one of three shapes, keyed by ``kind``. We enforce per-kind
# required/forbidden fields with allOf+if/then so the hallucination-containment
# rule is structural, not advisory.
_SEGMENT_COMMON_PROPS = {
    "kind": {"enum": ["speech", "melody", "ambiguous"]},
    "t0": {"type": "number", "minimum": 0},
    "t1": {"type": "number", "minimum": 0},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "role": {"enum": ["instruction", "exemplar", "contrast", "filler"]},
    # speech-ish fields
    "text": {"type": "string"},
    "words": {"type": "array", "items": _WORD},
    "asr": {
        "type": "object",
        "required": ["engine"],
        "properties": {
            "engine": {"type": "string"},
            "lang": {"type": "string"},
        },
        "additionalProperties": False,
    },
    # melody-ish fields
    "notes": {"type": "array", "items": _NOTE},
    "analysis": _ANALYSIS,
    "f0": _F0,
}

_SEGMENT = {
    "type": "object",
    "required": ["kind", "t0", "t1", "confidence"],
    "properties": _SEGMENT_COMMON_PROPS,
    "additionalProperties": False,
    "allOf": [
        {
            # speech: must carry text; must NOT carry notes/analysis.
            "if": {"properties": {"kind": {"const": "speech"}}},
            "then": {
                "required": ["text"],
                "not": {
                    "anyOf": [
                        {"required": ["notes"]},
                        {"required": ["analysis"]},
                    ]
                },
            },
        },
        {
            # melody: must carry notes; must NOT carry text/words (containment).
            "if": {"properties": {"kind": {"const": "melody"}}},
            "then": {
                "required": ["notes"],
                "not": {
                    "anyOf": [
                        {"required": ["text"]},
                        {"required": ["words"]},
                    ]
                },
            },
        },
        {
            # ambiguous: carries BOTH decodings for the planner to disambiguate.
            "if": {"properties": {"kind": {"const": "ambiguous"}}},
            "then": {"required": ["text", "notes"]},
        },
    ],
}

_PERCUSSION_HIT = {
    "type": "object",
    "required": ["t", "class", "confidence"],
    "properties": {
        "t": {"type": "number", "minimum": 0},
        "class": {"enum": list(DRUM_CLASSES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}

_TRACK = {
    "type": "object",
    "required": ["index", "name", "kind"],
    "properties": {
        "index": {"type": "integer"},
        "name": {"type": "string"},
        "kind": {"type": "string"},
    },
    "additionalProperties": False,
}

UTTERANCE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "mambo.utterance.v1",
    "type": "object",
    "required": ["schema", "utterance_id", "audio", "segments"],
    "properties": {
        "schema": {"const": SCHEMA_VERSION},
        "utterance_id": {"type": "string", "minLength": 1},
        "audio": {
            "type": "object",
            "required": ["sample_rate", "duration_s"],
            "properties": {
                "sample_rate": {"type": "integer", "exclusiveMinimum": 0},
                "duration_s": {"type": "number", "minimum": 0},
                "source": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "segments": {"type": "array", "items": _SEGMENT},
        "percussion": {"type": "array", "items": _PERCUSSION_HIT},
        "session_context": {
            "type": "object",
            "properties": {
                "daw": {"type": "string"},
                "selected_track": _TRACK,
                "tracks": {"type": "array", "items": _TRACK},
                "project_tempo_bpm": {"type": ["number", "null"]},
                "project_key": {"type": ["string", "null"]},
                "transport": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "history_refs": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_VALIDATOR = jsonschema.Draft202012Validator(UTTERANCE_SCHEMA)


class UIRValidationError(ValueError):
    """Raised when a UIR document violates mambo.utterance.v1."""


def validate(doc: dict[str, Any]) -> None:
    """Validate a UIR dict against mambo.utterance.v1. Raises on the first error.

    Beyond JSON-Schema structural checks, enforces semantic invariants:
      * segment ``t1 >= t0`` and segments are non-overlapping & time-ordered;
      * note times fall within their segment;
      * note ``name`` matches ``midi`` (spelling is canonical-by-construction).
    """
    errors = sorted(_VALIDATOR.iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        loc = "/".join(str(p) for p in e.path) or "<root>"
        raise UIRValidationError(f"{loc}: {e.message}")

    segs = doc.get("segments", [])
    prev_t1 = -math.inf
    for i, seg in enumerate(segs):
        if seg["t1"] < seg["t0"]:
            raise UIRValidationError(f"segments/{i}: t1 ({seg['t1']}) < t0 ({seg['t0']})")
        if seg["t0"] < prev_t1 - 1e-6:
            raise UIRValidationError(
                f"segments/{i}: starts at {seg['t0']} before previous segment ends ({prev_t1})"
            )
        prev_t1 = seg["t1"]
        for j, note in enumerate(seg.get("notes", [])):
            n_t1 = note["t0"] + note["dur"]
            if note["t0"] < seg["t0"] - 1e-3 or n_t1 > seg["t1"] + 1e-3:
                raise UIRValidationError(
                    f"segments/{i}/notes/{j}: note [{note['t0']:.3f},{n_t1:.3f}] "
                    f"outside segment [{seg['t0']:.3f},{seg['t1']:.3f}]"
                )
            expected = midi_to_name(note["midi"])
            if note["name"] != expected:
                raise UIRValidationError(
                    f"segments/{i}/notes/{j}: name {note['name']!r} != {expected!r} for midi {note['midi']}"
                )


# --------------------------------------------------------------------------- #
# Pitch-name helpers (canonical sharp spelling; cents relative to A4=440).
# --------------------------------------------------------------------------- #

_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_name(midi: int) -> str:
    """MIDI number -> canonical name, e.g. 61 -> 'C#4' (C4 = middle C = 60)."""
    return f"{_PITCH_CLASSES[midi % 12]}{midi // 12 - 1}"


def hz_to_midi_float(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# --------------------------------------------------------------------------- #
# Typed builder layer.
# --------------------------------------------------------------------------- #


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None so optional fields stay absent."""
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class Word:
    w: str
    t0: float
    t1: float
    logprob: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Note:
    midi: int
    t0: float
    dur: float
    vel: int = 90
    cents_dev: Optional[float] = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = midi_to_name(self.midi)

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "midi": self.midi,
                "name": self.name,
                "t0": round(self.t0, 4),
                "dur": round(self.dur, 4),
                "vel": self.vel,
                "cents_dev": None if self.cents_dev is None else round(self.cents_dev, 1),
            }
        )


@dataclass
class KeyCandidate:
    key: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "score": round(self.score, 4)}


@dataclass
class MelodyAnalysis:
    n_notes: int
    key_candidates: list[KeyCandidate] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    tempo_confidence: float = 0.0
    contour: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"n_notes": self.n_notes}
        if self.key_candidates:
            d["key_candidates"] = [k.to_dict() for k in self.key_candidates]
        if self.tempo_bpm is not None or self.tempo_confidence:
            d["tempo_bpm"] = {
                "value": None if self.tempo_bpm is None else round(self.tempo_bpm, 1),
                "confidence": round(self.tempo_confidence, 3),
            }
        if self.contour:
            d["contour"] = self.contour
        return d


@dataclass
class F0Stats:
    engine: str
    voicing_ratio: float
    median_hz: Optional[float] = None
    f0_stability: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "engine": self.engine,
                "voicing_ratio": round(self.voicing_ratio, 4),
                "median_hz": None if self.median_hz is None else round(self.median_hz, 2),
                "f0_stability": None if self.f0_stability is None else round(self.f0_stability, 4),
            }
        )


@dataclass
class Segment:
    kind: SegmentKind
    t0: float
    t1: float
    confidence: float = 1.0
    role: Optional[SegmentRole] = None
    # speech
    text: Optional[str] = None
    words: Optional[list[Word]] = None
    asr_engine: Optional[str] = None
    asr_lang: Optional[str] = None
    # melody
    notes: Optional[list[Note]] = None
    analysis: Optional[MelodyAnalysis] = None
    f0: Optional[F0Stats] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "t0": round(self.t0, 4),
            "t1": round(self.t1, 4),
            "confidence": round(self.confidence, 4),
        }
        if self.role is not None:
            d["role"] = self.role
        if self.text is not None:
            d["text"] = self.text
        if self.words is not None:
            d["words"] = [w.to_dict() for w in self.words]
        if self.asr_engine is not None:
            d["asr"] = _clean({"engine": self.asr_engine, "lang": self.asr_lang})
        if self.notes is not None:
            d["notes"] = [n.to_dict() for n in self.notes]
        if self.analysis is not None:
            d["analysis"] = self.analysis.to_dict()
        if self.f0 is not None:
            d["f0"] = self.f0.to_dict()
        return d


@dataclass
class Track:
    index: int
    name: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "kind": self.kind}


@dataclass
class SessionContext:
    daw: Optional[str] = None
    selected_track: Optional[Track] = None
    tracks: Optional[list[Track]] = None
    project_tempo_bpm: Optional[float] = None
    project_key: Optional[str] = None
    transport: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = _clean(
            {
                "daw": self.daw,
                "project_tempo_bpm": self.project_tempo_bpm,
                "project_key": self.project_key,
                "transport": self.transport,
            }
        )
        if self.selected_track is not None:
            d["selected_track"] = self.selected_track.to_dict()
        if self.tracks is not None:
            d["tracks"] = [t.to_dict() for t in self.tracks]
        return d


@dataclass
class Utterance:
    utterance_id: str
    sample_rate: int
    duration_s: float
    source: str = "synthetic"
    segments: list[Segment] = field(default_factory=list)
    percussion: list[dict[str, Any]] = field(default_factory=list)
    session_context: Optional[SessionContext] = None
    history_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "utterance_id": self.utterance_id,
            "audio": _clean(
                {
                    "sample_rate": self.sample_rate,
                    "duration_s": round(self.duration_s, 4),
                    "source": self.source,
                }
            ),
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.percussion:
            d["percussion"] = self.percussion
        if self.session_context is not None:
            d["session_context"] = self.session_context.to_dict()
        if self.history_refs:
            d["history_refs"] = list(self.history_refs)
        return d

    def validate(self) -> "Utterance":
        validate(self.to_dict())
        return self

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


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
