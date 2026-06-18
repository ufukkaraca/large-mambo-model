"""Audio-synthesis primitives for the synthetic fixture bootstrap (PAPER §5.2).

Voice-like hum rendering with the deliberate stressors the note-HMM must absorb:
5–6 Hz vibrato (±30 cents), 60–120 ms portamento glides into onsets, formant
shaping. Plus speech via macOS ``say`` and procedural colored noise for SNR
sweeps. Ground truth is exact by construction — the caller knows every note and
boundary it asked for.

All randomness flows through an explicit ``numpy.random.Generator`` so a fixed
seed reproduces the corpus bit-exact (seed traceability).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000  # project sample rate (PAPER §4.2)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# --------------------------------------------------------------------------- #
# Hum synthesis.
# --------------------------------------------------------------------------- #


@dataclass
class RenderedNote:
    """What was actually rendered for one note — the ground truth for that note."""

    midi: int
    t0: float  # onset in seconds, relative to the start of the rendered hum
    dur: float  # sounding duration in seconds
    vel: int


def _adsr(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n - a if n - a > 0 else 0)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r > 0:
        env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


def _formant_weights(freqs: np.ndarray, formants=(700.0, 1220.0, 2600.0), bw=900.0) -> np.ndarray:
    """Voice-like spectral envelope: sum of Gaussians at formant frequencies."""
    w = np.zeros_like(freqs)
    for fc in formants:
        w += np.exp(-0.5 * ((freqs - fc) / bw) ** 2)
    return w / (w.max() + 1e-9)


def render_note(
    midi: int,
    dur: float,
    sr: int,
    rng: np.random.Generator,
    prev_midi: int | None = None,
    *,
    vel: int = 90,
) -> np.ndarray:
    """Render one voice-like hummed note with vibrato + (optional) portamento."""
    n = int(round(dur * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(n) / sr

    # Pitch trajectory in MIDI-semitone space.
    pitch = np.full(n, float(midi))
    # Vibrato: 5–6 Hz, ±30 cents (= ±0.30 semitone), random phase.
    vib_rate = rng.uniform(5.0, 6.0)
    vib_phase = rng.uniform(0, 2 * np.pi)
    pitch += 0.30 * np.sin(2 * np.pi * vib_rate * t + vib_phase)
    # Portamento: glide from prev pitch into this onset over 60–120 ms.
    if prev_midi is not None and prev_midi != midi:
        porta = rng.uniform(0.06, 0.12)
        pn = min(int(porta * sr), n)
        if pn > 1:
            glide = np.linspace(prev_midi - midi, 0.0, pn)
            pitch[:pn] += glide

    f0 = midi_to_hz(pitch)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Additive voice-like timbre: harmonics weighted by a formant envelope.
    sig = np.zeros(n, dtype=np.float64)
    base_hz = midi_to_hz(midi)
    n_harm = max(4, int(min(8, (sr / 2) / base_hz)))
    harm_idx = np.arange(1, n_harm + 1)
    weights = _formant_weights(harm_idx * base_hz) * (1.0 / harm_idx)  # +natural rolloff
    for k, wgt in zip(harm_idx, weights):
        sig += wgt * np.sin(k * phase)
    sig /= np.abs(sig).max() + 1e-9

    sig *= _adsr(n, sr, attack=0.018, release=min(0.06, dur * 0.4))
    sig *= vel / 127.0
    return sig


def render_hum(
    notes: list[tuple[int, float, float]],
    sr: int,
    rng: np.random.Generator,
    *,
    gap_jitter: float = 0.01,
) -> tuple[np.ndarray, list[RenderedNote]]:
    """Render a melody as a continuous hum.

    ``notes`` is a list of ``(midi, onset_s, dur_s)`` with onsets relative to the
    hum start. Returns the audio and the exact ``RenderedNote`` ground truth.
    """
    total = max((on + dur for _, on, dur in notes), default=0.0) + 0.08
    buf = np.zeros(int(round(total * sr)) + sr // 10, dtype=np.float64)
    truth: list[RenderedNote] = []
    prev_midi: int | None = None
    for midi, onset, dur in notes:
        vel = int(rng.uniform(78, 104))
        wave = render_note(midi, dur, sr, rng, prev_midi=prev_midi, vel=vel)
        start = int(round(onset * sr))
        buf[start : start + len(wave)] += wave
        truth.append(RenderedNote(midi=midi, t0=onset, dur=dur, vel=vel))
        prev_midi = midi
    # Trim to the true end of audio content.
    end = int(round((notes[-1][1] + notes[-1][2]) * sr)) + int(0.06 * sr) if notes else 0
    buf = buf[:end]
    # Gentle breath noise so voicing_ratio isn't a perfect 1.0.
    buf += rng.normal(0, 0.002, size=buf.shape)
    return buf.astype(np.float32), truth


# --------------------------------------------------------------------------- #
# Speech synthesis via macOS `say`.
# --------------------------------------------------------------------------- #

_SAY = shutil.which("say")


def say_available() -> bool:
    return _SAY is not None


def render_speech(text: str, voice: str, sr: int, rate_wpm: int = 180) -> np.ndarray:
    """Synthesize speech with macOS ``say`` -> mono float32 at ``sr``."""
    if _SAY is None:
        raise RuntimeError("macOS `say` not available; cannot synthesize speech spans")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "speech.wav"
        subprocess.run(
            [
                _SAY, "-v", voice, "-r", str(rate_wpm),
                "-o", str(out), "--file-format=WAVE", f"--data-format=LEI16@{sr}",
                text,
            ],
            check=True,
            capture_output=True,
        )
        audio, file_sr = sf.read(str(out), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # to mono
    if file_sr != sr:  # `say` honors the requested rate, but be defensive
        import librosa

        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    # Trim leading/trailing near-silence so segment boundaries are tight.
    return _trim_silence(audio.astype(np.float32))


def _trim_silence(x: np.ndarray, thresh: float = 3e-3, pad: int = 240) -> np.ndarray:
    energy = np.abs(x)
    nz = np.where(energy > thresh)[0]
    if len(nz) == 0:
        return x
    lo = max(0, nz[0] - pad)
    hi = min(len(x), nz[-1] + pad)
    return x[lo:hi]


# --------------------------------------------------------------------------- #
# Noise + mixing.
# --------------------------------------------------------------------------- #


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping of white noise."""
    white = rng.normal(0, 1, n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:] * n)
    shaped = np.fft.irfft(spec * scale, n=n)
    shaped /= np.abs(shaped).max() + 1e-9
    return shaped.astype(np.float64)


def _active_rms(x: np.ndarray, thresh: float = 5e-3) -> float:
    a = np.abs(x)
    active = x[a > thresh]
    if len(active) < 16:
        active = x
    return float(np.sqrt(np.mean(active**2)) + 1e-12)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``signal`` scaled to the requested SNR (over active RMS)."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(signal)]
    sig_rms = _active_rms(signal)
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-12
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    return (signal + noise * (target_noise_rms / noise_rms)).astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.abs(x).max()
    if m < 1e-9:
        return x
    return (x * (peak / m)).astype(np.float32)
