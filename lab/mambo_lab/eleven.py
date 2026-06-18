"""ElevenLabs client — TTS and STT for the lab.

Uses (operator grant, scoped key — see DECISIONS D11):
  * `tts()` — realistic multi-voice speech for richer fixtures than macOS `say`.
  * `stt()` — Scribe speech-to-text, a high-quality final-ASR option for
    speech.py (NOTE: Scribe gives word timestamps but not the per-word logprob /
    compression-ratio features the *probe* router needs — those stay on Whisper).

The key is read via ``mambo_lab.secrets`` from the git-ignored ``.env``. Audio is
returned as float32 mono at a requested sample rate (resampled if needed).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

from . import secrets

BASE = "https://api.elevenlabs.io/v1"

# A handful of distinct English voices (verified available on the grant), spanning
# timbre/gender/accent so fixtures don't overfit one speaker.
VOICES = {
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Roger": "CwhRBWXzGAHq8TQ4Fs17",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "River": "SAz9YHcvj6GT2YYXdXww",
}
DEFAULT_VOICE = "George"


def _key() -> str:
    return secrets.get("ELEVENLABS_API_KEY", required=True)


def tts(text: str, *, voice: str = DEFAULT_VOICE, model: str = "eleven_flash_v2_5",
        sr: int = 48000) -> np.ndarray:
    """Synthesize ``text`` -> float32 mono at ``sr``. Returns the waveform."""
    import requests

    voice_id = VOICES.get(voice, voice)
    # Prefer raw PCM (no decode dependency); fall back to MP3 if the tier blocks pcm.
    for out_fmt, decode in (("pcm_44100", _decode_pcm), ("mp3_44100_128", _decode_compressed)):
        r = requests.post(
            f"{BASE}/text-to-speech/{voice_id}",
            headers={"xi-api-key": _key(), "Content-Type": "application/json"},
            params={"output_format": out_fmt},
            json={"text": text, "model_id": model},
            timeout=60,
        )
        if r.status_code == 200:
            audio, native_sr = decode(r.content)
            if native_sr != sr:
                import librosa

                audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr)
            return audio.astype(np.float32)
    r.raise_for_status()
    raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")


def _decode_pcm(content: bytes) -> tuple[np.ndarray, int]:
    return np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0, 44100


def _decode_compressed(content: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, native_sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    return audio.mean(axis=1), native_sr


@dataclass
class SttWord:
    w: str
    t0: float
    t1: float


@dataclass
class SttResult:
    text: str
    language_code: str = ""
    words: list[SttWord] = field(default_factory=list)


def stt(audio: np.ndarray, sr: int, *, model: str = "scribe_v1") -> SttResult:
    """Speech-to-text via ElevenLabs Scribe. ASR evidence — not UIR truth."""
    import requests
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    r = requests.post(
        f"{BASE}/speech-to-text",
        headers={"xi-api-key": _key()},
        data={"model_id": model},
        files={"file": ("audio.wav", buf, "audio/wav")},
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    words = [
        SttWord(w=w.get("text", ""), t0=float(w.get("start", 0.0)), t1=float(w.get("end", 0.0)))
        for w in d.get("words", []) if w.get("type", "word") == "word"
    ]
    return SttResult(text=d.get("text", ""), language_code=d.get("language_code", ""), words=words)
