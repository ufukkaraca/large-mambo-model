"""Melody path (PAPER §4.4): f0 -> notes -> key / tempo / contour.

Pipeline:
  1. f0 tracking — librosa pYIN backend (pluggable; PESTO/torchcrepe later, see
     DECISIONS D4). Gold-standard offline tracker, ideal for the synthetic
     S-gate and as the eval baseline.
  2. Note segmentation — amplitude-onset boundaries + per-region median pitch.
     A Tony/pYIN-style pitch HMM (Viterbi over semitone states) was built first
     (git history, DECISIONS D9) but over-segmented the deliberately
     portamento-stressed synthetic hums: the stickiness that absorbs vibrato
     turns each glide into a staircase of fake plateaus (note F1 ~0.72). Each
     rendered note has a crisp amplitude attack regardless of pitch glide, so
     onset detection gives far cleaner boundaries (note F1 ~0.86); the pitch is
     the median over the post-attack stable region, which the glide no longer
     corrupts. The f0-stability feature the pitch HMM motivated still feeds the
     router (``f0_statistics``).
  3. Key — Krumhansl-Schmuckler on the duration-weighted pitch-class profile;
     top-2 candidates with scores (PAPER §3.2: always emit top-2).
  4. Tempo — grid search over IOIs against a half-beat grid; confidence from
     fit quality (hums are rubato — report it, don't pretend).
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from . import ir

# --------------------------------------------------------------------------- #
# f0 tracking.
# --------------------------------------------------------------------------- #

HOP = 256
FRAME = 2048
FMIN = 80.0
FMAX = 1000.0


@dataclass
class F0Track:
    times: np.ndarray  # frame center times (s)
    f0_hz: np.ndarray  # Hz, NaN where unvoiced
    voiced: np.ndarray  # bool
    voiced_prob: np.ndarray
    sr: int
    hop: int


def track_f0(audio: np.ndarray, sr: int, *, backend: str = "pyin",
             fmin: float = FMIN, fmax: float = FMAX) -> F0Track:
    audio = np.asarray(audio, dtype=np.float32)
    if backend != "pyin":
        raise ValueError(f"unknown f0 backend {backend!r} (only 'pyin' in R0)")
    f0, vflag, vprob = librosa.pyin(  # fmin/fmax tightenable per-voice (voiceprint)
        audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=FRAME, hop_length=HOP
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    voiced = np.nan_to_num(vflag, nan=False).astype(bool)
    return F0Track(times=times, f0_hz=f0, voiced=voiced,
                   voiced_prob=np.nan_to_num(vprob), sr=sr, hop=HOP)


def f0_statistics(track: F0Track) -> dict:
    """Features the router needs (voicing ratio, median Hz, f0 stability)."""
    voiced = track.voiced
    voicing_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    vf = track.f0_hz[voiced]
    vf = vf[~np.isnan(vf)]
    median_hz = float(np.median(vf)) if len(vf) else None
    return {"voicing_ratio": voicing_ratio, "median_hz": median_hz,
            "f0_stability": _f0_stability(track)}


def _f0_stability(track: F0Track) -> float:
    """Mean within-250ms std of semitone f0 — low for humming, high for speech.

    This is the load-bearing acoustic feature the joint router leans on when the
    ASR confidence footprint is weak (PAPER §2.4).
    """
    midi = 69.0 + 12.0 * np.log2(np.where(track.voiced, track.f0_hz, np.nan) / 440.0)
    win = max(1, int(0.25 / (track.hop / track.sr)))
    stds = []
    for i in range(0, len(midi), win):
        seg = midi[i : i + win]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= 3:
            stds.append(np.std(seg))
    return float(np.mean(stds)) if stds else 0.0


# --------------------------------------------------------------------------- #
# Note segmentation.
# --------------------------------------------------------------------------- #


@dataclass
class TrackedNote:
    midi: int
    t0: float
    dur: float
    cents_dev: float
    vel: int


def segment_notes(
    audio: np.ndarray,
    track: F0Track,
    *,
    min_note: float = 0.08,
    attack_skip: float = 0.05,
    onset_delta: float = 0.12,
    onset_wait: int = 3,
    pitch_step: float = 1.0,
) -> list[TrackedNote]:
    """Decode discrete notes: amplitude onsets -> regions -> median pitch."""
    f0 = track.f0_hz
    midi = 69.0 + 12.0 * np.log2(np.where(track.voiced & ~np.isnan(f0), f0, np.nan) / 440.0)
    T = len(midi)
    if T == 0:
        return []
    dt = track.hop / track.sr
    skip = int(attack_skip / dt)
    min_frames = max(1, int(min_note / dt))

    # Note boundaries = amplitude attacks (catch same-pitch re-articulation, e.g.
    # "da-da-da") plus pitch plateaus, but a plateau is added ONLY where amplitude
    # missed the boundary (LEGATO — smooth humming changes pitch without
    # re-attacking). A plateau that merely follows an attack within ``settle`` is
    # the same note's pitch settling after its onset, so it is suppressed — that
    # keeps detache notes amplitude-driven (no doubled boundary / over-segmentation).
    amp_onsets = sorted(int(o) for o in _onset_frames(audio, track.sr, delta=onset_delta, wait=onset_wait))
    settle = max(1, int(0.16 / dt))
    pitch_onsets = [p for p in _pitch_onsets(midi, dt, min_note=min_note, step=pitch_step)
                    if not any(0 <= p - a <= settle for a in amp_onsets)]
    onsets = sorted(set(amp_onsets) | set(pitch_onsets))
    onsets = [o for o in onsets if 0 <= o < T]
    voiced_idx = np.where(~np.isnan(midi))[0]
    if len(voiced_idx) == 0:
        return []
    # Ensure the first note isn't lost if onset detection starts late.
    if not onsets or onsets[0] > voiced_idx[0] + min_frames:
        onsets = [int(voiced_idx[0])] + onsets
    bounds = sorted(set(onsets + [T]))

    notes: list[TrackedNote] = []
    spans: list[tuple[int, int]] = []  # (start_frame, end_frame) per note, for de-flutter
    for a, b in zip(bounds, bounds[1:]):
        reg = midi[a:b]
        v = np.where(~np.isnan(reg))[0]
        if len(v) < min_frames:
            continue
        rs = a + int(v[0])
        re = a + int(v[-1]) + 1  # offset = last voiced frame (trim trailing silence)
        t0 = a * dt  # amplitude onset = note slot start
        t1 = re * dt
        if t1 - t0 < min_note:
            continue
        stable = midi[max(rs, a) + skip : re]
        stable = stable[~np.isnan(stable)]
        if len(stable) < 2:
            stable = reg[~np.isnan(reg)]
        if len(stable) == 0:
            continue
        median_midi = float(np.median(stable))
        midi_n = int(round(median_midi))
        cents = (median_midi - midi_n) * 100.0
        vel = _velocity(audio, track.sr, t0, t1 - t0)
        notes.append(TrackedNote(midi=midi_n, t0=round(t0, 4), dur=round(t1 - t0, 4),
                                 cents_dev=round(cents, 1), vel=vel))
        spans.append((rs, re))
    notes, spans = _drop_floor_phantoms(notes, spans)
    return _deflutter(notes, spans, midi, dt, audio, track.sr)


SILENCE_VEL = 40   # _velocity floor (40 = rms≈0); a real note clears it (vel >= 42 observed)


def _drop_floor_phantoms(notes: list[TrackedNote],
                         spans: list[tuple[int, int]]) -> tuple[list, list]:
    """Drop notes sounded at SILENCE energy. pYIN reports *some* pitch in near-silent
    regions (breath lead-in, tail, gaps between notes), and the onset at the
    silence→sound edge spawns a phantom note there — the operator's reported
    "beginnings/endings" artifact. The pitch pYIN lands on varies by voice and
    recording (FMIN ≈80 Hz → midi 39 for one speaker, ~200 Hz → midi 55/56 for
    another), so a pitch-based rule overfits; the **silence** (velocity floor) is
    the voice-independent tell — a real hummed note always clears it. Generalised
    from the FMIN-specific rule after a held-out voice exposed phantoms off FMIN."""
    keep = [(n, sp) for n, sp in zip(notes, spans) if n.vel > SILENCE_VEL]
    if not keep:
        return notes, spans
    out_n, out_s = zip(*keep)
    return list(out_n), list(out_s)


DEFLUTTER_TROUGH = 0.30   # merge same-pitch notes only if energy between stays above this * note level
_TROUGH_WIN_S = 0.025     # fine RMS window (s) — must resolve a short consonant gap, not smear it


def _deflutter(notes: list[TrackedNote], spans: list[tuple[int, int]], midi: np.ndarray,
               dt: float, audio: np.ndarray, sr: int) -> list[TrackedNote]:
    """Merge a note into its predecessor when they are the SAME pitch and the
    energy between them never drops near silence — one sustained note that a
    spurious amplitude onset split (human vibrato / breath pulse / tremolo, whose
    amplitude only dips partway). A deliberate re-articulation (`da`-`da`) closes
    the mouth for the consonant, so energy returns near silence between syllables,
    and is preserved; a real pitch step is a different midi and is preserved
    (legato glides included). Equal-midi only, so synthetic melodies — which
    change pitch between notes — are untouched (note F1 unaffected). The RMS is
    measured on a fine 25 ms window so a short consonant gap is resolved, not
    averaged away by a coarse frame."""
    if len(notes) < 2:
        return notes
    win = max(HOP, int(_TROUGH_WIN_S * sr))
    rms = librosa.feature.rms(y=np.asarray(audio, dtype=np.float32),
                              frame_length=win, hop_length=HOP, center=True)[0]
    rms = _match_len_local(rms, len(midi))
    out = [notes[0]]
    out_spans = [spans[0]]
    for n, (s, e) in zip(notes[1:], spans[1:]):
        prev, (ps, pe) = out[-1], out_spans[-1]
        lo, hi = min(pe, s), max(pe, s)
        between = rms[lo:hi] if hi > lo else rms[max(0, s - 1):s + 2]
        core = rms[ps:e][rms[ps:e] > 0]
        note_level = float(np.median(core)) if len(core) else 1.0
        trough = float(np.min(between)) / (note_level + 1e-9) if len(between) else 0.0
        if n.midi == prev.midi and trough > DEFLUTTER_TROUGH:  # no near-silence between -> one held note
            t0 = prev.t0
            t1 = round(n.t0 + n.dur, 4)
            out[-1] = TrackedNote(midi=prev.midi, t0=t0, dur=round(t1 - t0, 4),
                                  cents_dev=prev.cents_dev, vel=_velocity(audio, sr, t0, t1 - t0))
            out_spans[-1] = (ps, e)
        else:
            out.append(n)
            out_spans.append((s, e))
    return out


def _match_len_local(a: np.ndarray, T: int) -> np.ndarray:
    if len(a) == T:
        return a
    return a[:T] if len(a) > T else np.pad(a, (0, T - len(a)), mode="edge")


def _smooth_pitch(midi: np.ndarray, win: int) -> np.ndarray:
    """Median-smooth the (NaN-gapped) semitone contour — kills vibrato wobble so a
    held note reads as one level, without smearing a real step across the gap."""
    if win < 3:
        return midi
    half, out = win // 2, midi.copy()
    for i in range(len(midi)):
        seg = midi[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = np.median(seg) if len(seg) else np.nan
    return out


def _pitch_onsets(midi: np.ndarray, dt: float, *, min_note: float,
                  step: float = 1.0, smooth_s: float = 0.08) -> list[int]:
    """Boundaries where pitch SETTLES on a new level >= ``step`` from the current
    note and HOLDS for ~min_note. A legato note change is a sustained step →
    boundary; a portamento glide doesn't hold a new level → no boundary (no
    staircasing — the failure that retired the pitch-HMM). The level is tracked
    CONTINUOUSLY (no semitone rounding), so a note at a fractional pitch doesn't
    flicker across a boundary, and smoothing + the hold requirement filter vibrato
    (which returns) — a steady or wobbly held note stays one note."""
    sm = _smooth_pitch(midi, max(1, int(smooth_s / dt)))
    hold = max(2, int(min_note / dt))
    n, onsets, level, i = len(sm), [], None, 0
    while i < n:
        if np.isnan(sm[i]):
            i += 1
            continue
        if level is None:                       # establish the first note's level
            level = float(sm[i])
            i += 1
            continue
        if abs(sm[i] - level) >= step:          # candidate departure — must it hold?
            seg = sm[i: i + hold]
            seg = seg[~np.isnan(seg)]
            if len(seg) >= max(2, int(hold * 0.6)) and abs(float(np.median(seg)) - level) >= step * 0.7:
                onsets.append(i)
                level = float(np.median(seg))   # new note's level
                i += len(seg)
                continue
        i += 1
    return onsets


def _onset_frames(audio: np.ndarray, sr: int, *, delta: float, wait: int) -> list[int]:
    try:
        on = librosa.onset.onset_detect(
            y=np.asarray(audio, dtype=np.float32), sr=sr, hop_length=HOP,
            units="frames", backtrack=True, delta=delta, wait=wait,
        )
        return [int(o) for o in on]
    except Exception:
        return []


def _velocity(audio: np.ndarray, sr: int, t0: float, dur: float) -> int:
    a, b = int(t0 * sr), int((t0 + dur) * sr)
    seg = np.asarray(audio[a:b], dtype=np.float64)
    if len(seg) == 0:
        return 90
    rms = float(np.sqrt(np.mean(seg**2)))
    return int(np.clip(40 + 600 * rms, 1, 127))


# --------------------------------------------------------------------------- #
# Key (Krumhansl-Schmuckler) + tempo + contour.
# --------------------------------------------------------------------------- #

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PC = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def estimate_key(notes: list[TrackedNote], top_k: int = 2) -> list[ir.KeyCandidate]:
    if not notes:
        return []
    profile = np.zeros(12)
    for n in notes:
        profile[n.midi % 12] += n.dur
    if profile.sum() == 0:
        return []
    profile = profile - profile.mean()
    cands = []
    for tonic in range(12):
        maj = np.roll(_KS_MAJOR - _KS_MAJOR.mean(), tonic)
        mino = np.roll(_KS_MINOR - _KS_MINOR.mean(), tonic)
        cands.append((f"{_PC[tonic]} major", _corr(profile, maj)))
        cands.append((f"{_PC[tonic]} minor", _corr(profile, mino)))
    cands.sort(key=lambda x: x[1], reverse=True)
    return [ir.KeyCandidate(k, float(s)) for k, s in cands[:top_k]]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da < 1e-9 or db < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def estimate_tempo(notes: list[TrackedNote]) -> tuple[float | None, float]:
    """Grid-search BPM against a half-beat IOI grid; confidence from fit."""
    if len(notes) < 3:
        return None, 0.0
    iois = np.diff([n.t0 for n in notes])
    iois = iois[iois > 0.05]
    if len(iois) < 2:
        return None, 0.0
    best_bpm, best_res = None, np.inf
    for bpm in np.arange(50, 180.5, 0.5):
        half_beat = 30.0 / bpm  # seconds per half beat
        ratios = iois / half_beat
        res = float(np.mean(np.abs(ratios - np.round(ratios))))
        if res < best_res:
            best_res, best_bpm = res, float(bpm)
    confidence = float(np.clip(1.0 - best_res * 3.0, 0.0, 1.0))
    return best_bpm, round(confidence, 3)


def contour(notes: list[TrackedNote]) -> str:
    syms = []
    for a, b in zip(notes, notes[1:]):
        syms.append("u" if b.midi > a.midi else "d" if b.midi < a.midi else "=")
    return " ".join(syms)


# --------------------------------------------------------------------------- #
# Top-level: audio span -> melody Segment.
# --------------------------------------------------------------------------- #


def analyze_span(audio: np.ndarray, sr: int, *, t_offset: float = 0.0,
                 role: str | None = None, confidence: float = 1.0,
                 pitch_step: float = 1.0) -> ir.Segment:
    """Decode a melody span (audio already cropped to the span) -> ir.Segment.

    ``t_offset`` shifts note/segment times into global utterance time.
    ``pitch_step`` is the per-voice note-split threshold (voiceprint calibration).
    """
    track = track_f0(audio, sr)
    tracked = segment_notes(audio, track, pitch_step=pitch_step)
    notes = [
        ir.Note(midi=n.midi, t0=round(t_offset + n.t0, 4), dur=n.dur, vel=n.vel, cents_dev=n.cents_dev)
        for n in tracked
    ]
    keys = estimate_key(tracked)
    bpm, tconf = estimate_tempo(tracked)
    analysis = ir.MelodyAnalysis(
        n_notes=len(notes), key_candidates=keys,
        tempo_bpm=bpm, tempo_confidence=tconf, contour=contour(tracked),
    )
    stats = f0_statistics(track)
    f0 = ir.F0Stats(engine="pyin", voicing_ratio=stats["voicing_ratio"],
                    median_hz=stats["median_hz"], f0_stability=stats["f0_stability"])
    t1 = t_offset + (notes[-1].t0 - t_offset + notes[-1].dur if notes else len(audio) / sr)
    return ir.Segment(kind="melody", t0=round(t_offset, 4), t1=round(t1, 4),
                      confidence=confidence, role=role, notes=notes, analysis=analysis, f0=f0)
