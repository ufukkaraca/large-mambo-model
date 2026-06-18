"""Speech path (PAPER §4.4 fusion): final text from VERIFIED speech spans only.

The probe (Pass 1) already ran ASR over the whole utterance; for a span the
router committed as speech, we keep the probe's words inside that span — the
probe engine is the product engine here, so no re-decode is needed (PAPER §4.3:
"Speech spans keep their transcription … or are re-run if the probe engine
differs"). A span committed as melody never reaches this function, so no ASR
text can leak onto a hum (the ASR-is-evidence containment rule).

An ElevenLabs Scribe engine is available as an alternative (DECISIONS D11) for a
higher-quality second pass; off by default to keep R0 offline + deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import ir, probe


@dataclass
class SpeechText:
    text: str
    words: list[ir.Word]
    engine: str


def from_probe(pr: probe.ProbeResult, t0: float, t1: float) -> SpeechText:
    """Extract the probe's words within [t0, t1] -> verified speech text."""
    words = [w for w in pr.words if w.t1 > t0 + 1e-3 and w.t0 < t1 - 1e-3]
    ir_words = [ir.Word(w=_clean_word(w.w), t0=round(max(w.t0, t0), 4),
                        t1=round(min(w.t1, t1), 4), logprob=round(w.logprob, 3)) for w in words]
    text = _normalize(" ".join(w.w for w in words))
    return SpeechText(text=text, words=ir_words, engine=pr.engine)


def from_scribe(audio: np.ndarray, sr: int, t0: float, t1: float) -> SpeechText:
    """Re-decode a verified speech span with ElevenLabs Scribe (optional)."""
    from . import eleven

    a, b = int(t0 * sr), int(t1 * sr)
    res = eleven.stt(audio[a:b], sr)
    words = [ir.Word(w=_clean_word(w.w), t0=round(t0 + w.t0, 4), t1=round(t0 + w.t1, 4))
             for w in res.words]
    return SpeechText(text=_normalize(res.text), words=words, engine="elevenlabs:scribe_v1")


def _clean_word(w: str) -> str:
    return w.strip()


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,!?").lower()
