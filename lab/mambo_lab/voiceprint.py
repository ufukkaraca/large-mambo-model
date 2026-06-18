"""Per-user voice calibration — the "voiceprint" (PAPER §6.1(e) limitation fix).

The melody detector's thresholds were tuned on the operator's low-male voice; N=4
showed they don't generalize (a female voice over-segments and trips false melody
on speech — Rim, §E-LIVE-3). The fix is *parameterization, not one-size-fits-all*:
a ~20 s onboarding (hum a steady note, hum low→high, say a command) yields a
`Voiceprint` that scales the detector's f0 range, vibrato/onset thresholds, and the
speech-vs-hum voicing boundary to THIS voice.

A `Voiceprint` is derived from calibration audio (here, or from any held-note +
speech clips); `melody.track_f0` / `melody.segment_notes` take it as an optional
arg and fall back to the shipped constants (`DEFAULT`) when absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import melody


@dataclass
class Voiceprint:
    f0_min: float = melody.FMIN          # pyin floor (Hz)
    f0_max: float = melody.FMAX          # pyin ceiling (Hz)
    vibrato_semitones: float = 0.4       # held-note wobble (semitone std) — sets the split threshold
    speech_voicing: float = 0.5          # voicing ratio of this speaker's speech (router boundary)
    label: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Voiceprint":
        if not d:
            return DEFAULT
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    # how big a pitch step counts as a new note for THIS voice. DEADBAND design:
    # a normal-vibrato voice (held-note wobble std ≤ V0) stays at exactly the 1.0
    # default — so calibration NEVER over-merges a clean voice — and only a voice
    # with abnormally wide vibrato gets a looser threshold (so its wobble stops
    # splitting held notes). Capped so it can't run away.
    #
    # V0 is set a-priori: normal held-note wobble is well under 1 semitone, so we
    # only loosen above 1.0 st. The N=3 held-out voices VALIDATE this rather than
    # tune it — the two normal voices measure 0.60/0.68 st (inside the deadband,
    # untouched, zero regression) and only the genuine outlier (3.2 st, ~5×) is
    # loosened. An earlier V0=0.55 mis-fit regressed a normal voice (Eddy −0.06);
    # see DECISIONS D22 and runs/*-voiceprint.
    @property
    def pitch_step(self) -> float:
        V0 = 1.0  # held-note wobble below 1 semitone = normal → shipped default
        return float(min(2.5, 1.0 + 1.0 * max(0.0, self.vibrato_semitones - V0)))


DEFAULT = Voiceprint()


def _voiced_midi(audio: np.ndarray, sr: int, *, fmin: float = melody.FMIN, fmax: float = melody.FMAX) -> np.ndarray:
    t = melody.track_f0(audio, sr, fmin=fmin, fmax=fmax)
    f0 = t.f0_hz[t.voiced]
    f0 = f0[f0 > 0]
    return 69.0 + 12.0 * np.log2(f0 / 440.0) if len(f0) else np.array([])


def _wobble_semitones(audio: np.ndarray, sr: int) -> float | None:
    """Vibrato/wobble depth on a (nominally single) held note: the std of MIDI
    after removing slow drift — robust to a little portamento."""
    m = _voiced_midi(audio, sr)
    if len(m) < 8:
        return None
    # detrend with a moving median to strip any intended glide, keep the wobble
    k = max(3, len(m) // 8)
    trend = np.array([np.median(m[max(0, i - k):i + k + 1]) for i in range(len(m))])
    return float(np.std(m - trend))


def derive(held: list[tuple[np.ndarray, int]], speech: list[tuple[np.ndarray, int]],
           *, label: str = "user") -> Voiceprint:
    """Build a Voiceprint from calibration clips: `held` = steady/low-high hums (f0
    range + vibrato), `speech` = spoken command(s) (voicing). Missing inputs fall
    back to the shipped defaults so a partial calibration still works."""
    f0s, wobbles = [], []
    for a, sr in held:
        t = melody.track_f0(a, sr)
        v = t.f0_hz[t.voiced]
        f0s.extend(v[v > 0].tolist())
        w = _wobble_semitones(a, sr)
        if w is not None:
            wobbles.append(w)
    voicings = [float(np.mean(melody.track_f0(a, sr).voiced)) for a, sr in speech] or [DEFAULT.speech_voicing]

    vp = Voiceprint(label=label)
    if f0s:
        lo, hi = np.percentile(f0s, 5), np.percentile(f0s, 95)
        vp.f0_min = float(max(60.0, lo / 1.5))   # a margin below/above their range
        vp.f0_max = float(min(1400.0, hi * 1.5))
    if wobbles:
        vp.vibrato_semitones = float(np.median(wobbles))
    vp.speech_voicing = float(np.mean(voicings))
    return vp
