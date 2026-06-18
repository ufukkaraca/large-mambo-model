"""ASR probe (PAPER §4.3, Pass 1): transcribe the WHOLE utterance, keep the
*confidence geometry*, trust none of the text.

The top-down router reads the signal-level footprint of confabulation — per-word
log-probability collapse, compression-ratio spikes, no-speech / music tokens,
timestamp instability — to find where the transcript degenerates into a hum
(PAPER §2.4). This module produces that evidence; it never decides a span is
speech, and nothing here may be copied into a UIR ``text`` field without passing
the router's speech-verification gate (the ASR-is-evidence containment rule). A span committed as
melody has its probe text discarded.

Backend: faster-whisper (CTranslate2), CPU int8. Model id via
``MAMBO_WHISPER_MODEL`` (default ``base.en``); downloaded on first use.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

PROBE_SR = 16000  # faster-whisper expects 16 kHz mono float32


@dataclass
class ProbeWord:
    w: str
    t0: float
    t1: float
    logprob: float  # ln P(word); collapses on non-speech


@dataclass
class ProbeSegment:
    t0: float
    t1: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[ProbeWord] = field(default_factory=list)


@dataclass
class ProbeResult:
    """Whole-utterance ASR evidence. ``text`` is evidence, NEVER truth."""

    engine: str
    text: str
    segments: list[ProbeSegment]
    duration_s: float

    @property
    def words(self) -> list[ProbeWord]:
        return [w for s in self.segments for w in s.words]

    def confidence_at(self, t0: float, t1: float) -> dict:
        """Aggregate confidence features over a time window — the router's
        evidence that a span is (not) speech. Lower ``mean_word_logprob`` and
        higher ``compression_ratio`` / ``no_speech_prob`` => likelier non-speech.
        """
        words = [w for w in self.words if w.t1 > t0 and w.t0 < t1]
        seg_overlap = [s for s in self.segments if s.t1 > t0 and s.t0 < t1]
        lp = [w.logprob for w in words]
        return {
            "n_words": len(words),
            "mean_word_logprob": float(np.mean(lp)) if lp else None,
            "min_word_logprob": float(np.min(lp)) if lp else None,
            "no_speech_prob": float(np.mean([s.no_speech_prob for s in seg_overlap])) if seg_overlap else None,
            "compression_ratio": float(np.mean([s.compression_ratio for s in seg_overlap])) if seg_overlap else None,
            "words_per_sec": len(words) / (t1 - t0) if t1 > t0 else 0.0,
            "has_music_token": any(_music_token(w.w) for w in words),
        }


_MUSIC_TOKENS = ("♪", "[music]", "[Music]", "(music)", "music playing")


def _music_token(w: str) -> bool:
    wl = w.lower()
    return any(tok.lower() in wl for tok in _MUSIC_TOKENS)


@lru_cache(maxsize=2)
def _model(model_id: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_id, device="cpu", compute_type="int8")


def _to_probe_sr(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != PROBE_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=PROBE_SR)
    return audio.astype(np.float32)


def transcribe(audio: np.ndarray, sr: int, *, model_id: Optional[str] = None,
               lang: str = "en") -> ProbeResult:
    model_id = model_id or os.environ.get("MAMBO_WHISPER_MODEL", "base.en")
    wav = _to_probe_sr(audio, sr)
    model = _model(model_id)
    segments, _info = model.transcribe(
        wav, language=lang, word_timestamps=True, beam_size=1,
        condition_on_previous_text=False,
    )
    out_segs: list[ProbeSegment] = []
    texts: list[str] = []
    for s in segments:
        words = [
            ProbeWord(w=w.word.strip(), t0=float(w.start), t1=float(w.end),
                      logprob=float(math.log(max(w.probability, 1e-6))))
            for w in (s.words or [])
        ]
        out_segs.append(ProbeSegment(
            t0=float(s.start), t1=float(s.end), text=s.text.strip(),
            avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob),
            compression_ratio=float(s.compression_ratio), words=words,
        ))
        texts.append(s.text.strip())
    return ProbeResult(engine=f"faster-whisper:{model_id}", text=" ".join(texts).strip(),
                       segments=out_segs, duration_s=len(wav) / PROBE_SR)
