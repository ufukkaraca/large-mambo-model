"""Segmentation router (PAPER §2.4, §4.3): acoustic | linguistic | joint.

Three strategies behind one interface, the ablation arms B3/B4/B5:

  * **acoustic** (bottom-up) — classify each frame by how it *sounds*: local
    voicing ratio ⊕ f0 stability ⊕ hysteresis, min segment 400 ms. No language,
    so no role assignment. (R0 uses f0+spectral stats, not YAMNet — DECISIONS
    D13; YAMNet is the documented upgrade.)
  * **linguistic** (top-down) — frame rules over the ASR probe transcript +
    confidence-collapse detection, **with the f0 verification gate** before any
    melody commitment (PAPER §2.4's load-bearing rule). Roles come free from the
    sentence frame ("like ___" → exemplar; "instead of ___" → contrast).
  * **joint** — propose–verify: acoustic + linguistic evidence fused; a span is
    melody only if the pitch is stable AND the transcript has no confident words
    there; language assigns the role.

The router emits ``SpanProposal``s (boundaries + kind + role); the decoders
(melody / speech) run later in ``fuse.py``. ASR text never crosses into a melody
commitment (the ASR-is-evidence containment rule) — that is enforced here (f0 gate) and structurally
in ``ir.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import melody, probe

Strategy = Literal["acoustic", "linguistic", "joint"]

# Thresholds (tunable; defaults set against the synthetic S-gate).
VOICING_HI = 0.65          # local voicing fraction above which a frame can be melody (acoustic arm)
MELODY_VOICING = 0.80      # higher bar for a "hole" to be melody — voiced SPEECH (~0.5-0.78)
                           # must not qualify; sustained hums (~0.9+) do
F0_STABLE_MAX = 1.1        # local semitone std below which pitch counts as "stable"
WORD_LP_CONF = -1.1        # ln P(word) above which an ASR word is "confident speech"
MIN_SEG = 0.40             # minimum committed segment (s)
HYSTERESIS = 0.30          # label median-filter window (s)
GATE_VOICING = 0.55        # f0 verification gate: min mean voicing over a melody span
GATE_F0_STD = 1.3          # f0 verification gate: max mean f0 std over a melody span


@dataclass
class SpanProposal:
    t0: float
    t1: float
    kind: Literal["speech", "melody", "ambiguous"]
    role: Optional[str] = None
    confidence: float = 1.0
    suppress_text: bool = False  # containment: span is hum-like; ASR text not trusted


@dataclass
class Frames:
    """Per-frame features on the f0 time grid (hop = melody.HOP)."""

    times: np.ndarray
    dt: float
    voiced: np.ndarray          # bool
    local_voicing: np.ndarray   # fraction voiced in a ~300 ms window
    f0_std: np.ndarray          # local semitone std (NaN-safe; large where unstable)
    energy: np.ndarray          # per-frame RMS (normalized 0..1)
    word_lp: np.ndarray         # max ln P(word) covering the frame (-inf if none)
    sound: np.ndarray           # bool: energy above the silence floor


def compute_frames(audio: np.ndarray, sr: int, f0: melody.F0Track,
                   pr: probe.ProbeResult) -> Frames:
    midi = 69.0 + 12.0 * np.log2(np.where(f0.voiced, f0.f0_hz, np.nan) / 440.0)
    T = len(midi)
    dt = f0.hop / sr
    win = max(1, int(0.30 / dt))

    valid = (~np.isnan(midi)).astype(float)
    x = np.nan_to_num(midi, nan=0.0)
    cnt = uniform_filter1d(valid, win, mode="nearest") * win
    cnt = np.maximum(cnt, 1e-6)
    mean = uniform_filter1d(x, win, mode="nearest") * win / cnt
    mean_sq = uniform_filter1d(x * x, win, mode="nearest") * win / cnt
    f0_std = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    f0_std = np.where(uniform_filter1d(valid, win, mode="nearest") > 0.15, f0_std, 99.0)
    local_voicing = uniform_filter1d(valid, win, mode="nearest")

    rms = librosa.feature.rms(y=np.asarray(audio, dtype=np.float32),
                              frame_length=melody.FRAME, hop_length=melody.HOP)[0]
    rms = _match_len(rms, T)
    energy = rms / (rms.max() + 1e-9)
    sound = energy > 0.04  # quiet trailing TTS ("but slower") sits well below a loud hum

    word_lp = np.full(T, -np.inf)
    for w in pr.words:
        a, b = int(w.t0 / dt), int(np.ceil(w.t1 / dt))
        a, b = max(0, a), min(T, b)
        if b > a:
            word_lp[a:b] = np.maximum(word_lp[a:b], w.logprob)

    return Frames(times=f0.times, dt=dt, voiced=f0.voiced, local_voicing=local_voicing,
                  f0_std=f0_std, energy=energy, word_lp=word_lp, sound=sound)


def _match_len(a: np.ndarray, T: int) -> np.ndarray:
    if len(a) == T:
        return a
    if len(a) > T:
        return a[:T]
    return np.pad(a, (0, T - len(a)), mode="edge")


# --------------------------------------------------------------------------- #
# Frame labelling per strategy.  0 = gap/silence, 1 = speech, 2 = melody.
# --------------------------------------------------------------------------- #

GAP, SPEECH, MELODY = 0, 1, 2


def _label_acoustic(f: Frames) -> np.ndarray:
    lab = np.full(len(f.times), GAP)
    melodyish = (f.local_voicing > VOICING_HI) & (f.f0_std < F0_STABLE_MAX) & f.sound
    lab[f.sound] = SPEECH
    lab[melodyish] = MELODY
    return lab


def _label_linguistic(f: Frames) -> np.ndarray:
    lab = np.full(len(f.times), GAP)
    confident_word = f.word_lp > WORD_LP_CONF
    lab[f.sound] = SPEECH
    # A SUSTAINED highly-voiced "hole" in the confident transcript is a melody
    # candidate. Per-frame f0 stability is NOT required (portamento glides are
    # still melody) — stability is checked at the SPAN level (_f0_gate). The high
    # voicing bar keeps voiced speech vowels out of melody.
    hole = f.sound & ~confident_word & (f.local_voicing > MELODY_VOICING)
    lab[hole] = MELODY
    lab[confident_word] = SPEECH
    return lab


def _label_joint(f: Frames) -> np.ndarray:
    # Joint shares the linguistic frame labels (robust speech detection); its
    # acoustic veto runs at the SPAN level (_reclassify_hum_spans), not per
    # frame — a per-frame veto wrongly flips sustained speech vowels to melody.
    return _label_linguistic(f)


def _reclassify_hum_spans(spans: list[SpanProposal], f: Frames) -> list[SpanProposal]:
    """Joint propose-verify (PAPER §2.4): a SPEECH span that is acoustically a
    clear, sustained hum (very high voicing + mostly stable pitch) is reclassified
    as melody — catching Whisper words that were confidently but wrongly time-
    stamped into the middle of a hum, which frame labels can't fix."""
    for sp in spans:
        if sp.kind != "speech" or sp.t1 - sp.t0 < 0.5:
            continue
        i, j = int(sp.t0 / f.dt), int(sp.t1 / f.dt)
        if j <= i:
            continue
        voicing = float(np.mean(f.local_voicing[i:j]))
        frac_stable = float(np.mean(f.f0_std[i:j] < F0_STABLE_MAX))
        # A clear sustained hum sits at ~0.9+ voicing; very-voiced speech phrases
        # like "give me something like" peak ~0.86, so the 0.88 bar excludes them
        # while still recovering hums that Whisper mis-labeled as speech.
        if voicing > 0.88 and frac_stable > 0.60:
            sp.kind = "melody"
    return _merge_adjacent(spans)


# A hum embedded in a swallowed speech span must be SUSTAINED to carve out (a
# spoken vowel never holds stable pitch this long); only a genuinely COLLAPSED
# speech span (the whole utterance merged into one — many seconds) is a
# candidate, so normal intros/outros/commands (a real spoken command is far
# shorter) and clean fixtures are left untouched. This recovers the 10-20 dB
# collapse where Whisper mis-times words over the hum and the frame labeller
# merges the whole utterance into speech.
MIN_CARVE_SPAN = 2.5       # a speech span at least this long is a carve candidate (mixed utterances ~3s)
MIN_MELODY_CARVE = 0.6     # an interior hum must be at least this long to split out (short embedded hums)
CARVE_MERGE_GAP = 0.35     # bridge sub-this humish gaps (noise fragments one hum) before splitting


def _runs(mask: np.ndarray, dt: float, bridge: float = 0.0) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask as [start, end) index pairs; runs
    separated by a gap shorter than ``bridge`` seconds are merged (noise punches
    brief holes in one sustained hum — they are not real boundaries)."""
    raw: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1
    if bridge <= 0 or not raw:
        return raw
    merged = [raw[0]]
    for a, b in raw[1:]:
        if (a - merged[-1][1]) * dt < bridge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _carve_embedded_hum(spans: list[SpanProposal], f: Frames) -> list[SpanProposal]:
    """Joint propose–verify, span-internal (PAPER §2.4): a COLLAPSED SPEECH span
    whose interior is acoustically a sustained, stable-pitch hum — but which the
    frame labeller swallowed into speech (Whisper confidently mis-timed words over
    the hum, or noise dropped the per-frame voicing) — is SPLIT into
    speech | melody | speech. Acoustic-only on purpose: the f0 evidence survives
    10-20 dB noise (note F1 ~0.95) where the transcript does not. Without this the
    whole [S M S] utterance collapses to one speech span (segment F1 -> 0)."""
    out: list[SpanProposal] = []
    for sp in spans:
        if sp.kind != "speech" or sp.t1 - sp.t0 < MIN_CARVE_SPAN:
            out.append(sp)
            continue
        i, j = int(sp.t0 / f.dt), min(int(sp.t1 / f.dt), len(f.times))
        humish = ((f.local_voicing[i:j] > MELODY_VOICING)
                  & (f.f0_std[i:j] < F0_STABLE_MAX) & f.sound[i:j])
        runs = [(i + a, i + b) for (a, b) in _runs(humish, f.dt, CARVE_MERGE_GAP)
                if (b - a) * f.dt >= MIN_MELODY_CARVE and _f0_gate(f, i + a, i + b)]
        if not runs:
            out.append(sp)
            continue
        cursor = i
        for a, b in runs:
            if (a - cursor) * f.dt >= MIN_SEG:
                out.append(SpanProposal(round(f.times[cursor], 4), round(f.times[a], 4),
                                        "speech", confidence=sp.confidence))
            tb = round(f.times[min(b, len(f.times) - 1)] + f.dt, 4)
            out.append(SpanProposal(round(f.times[a], 4), tb, "melody", confidence=sp.confidence))
            cursor = b
        if cursor < len(f.times) and (sp.t1 - f.times[min(cursor, len(f.times) - 1)]) >= MIN_SEG:
            out.append(SpanProposal(round(f.times[min(cursor, len(f.times) - 1)], 4), sp.t1,
                                    "speech", confidence=sp.confidence))
    return _merge_adjacent(out)


_LABELLERS = {"acoustic": _label_acoustic, "linguistic": _label_linguistic, "joint": _label_joint}


# --------------------------------------------------------------------------- #
# Labels -> spans (hysteresis, snap to silence, min segment, f0 gate).
# --------------------------------------------------------------------------- #


def _smooth(lab: np.ndarray, win: int) -> np.ndarray:
    """Median-filter the label stream for hysteresis (~0.3 s)."""
    if win < 3:
        return lab
    half = win // 2
    out = lab.copy()
    for i in range(len(lab)):
        a, b = max(0, i - half), min(len(lab), i + half + 1)
        vals, counts = np.unique(lab[a:b], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _spans_from_labels(lab: np.ndarray, f: Frames, strategy: str) -> list[SpanProposal]:
    spans: list[SpanProposal] = []
    i, n = 0, len(lab)
    while i < n:
        if lab[i] == GAP:
            i += 1
            continue
        kind_lab = lab[i]
        j = i
        while j < n and lab[j] == kind_lab:
            j += 1
        t0, t1 = f.times[i], f.times[j - 1] + f.dt
        if t1 - t0 >= MIN_SEG:
            kind = "melody" if kind_lab == MELODY else "speech"
            if kind == "melody" and not _f0_gate(f, i, j):
                kind = "speech"  # gate failed -> not melody
            spans.append(SpanProposal(t0=round(t0, 4), t1=round(t1, 4), kind=kind))
        i = j
    return _merge_adjacent(spans)


def _mark_text_containment(spans: list[SpanProposal], audio: np.ndarray, sr: int,
                           f0: "melody.F0Track", *, min_notes: int = 3,
                           pitch_step: float = 1.0) -> None:
    """Speech-verification gate (the ASR-is-evidence containment rule): the melody detector is the
    arbiter. A SPEECH span that actually contains a clean note sequence
    (>= min_notes stable notes) is a hum — Whisper's mis-timed/flooded words over
    it are not trusted, so its text is discarded. This holds even at 10-20 dB
    where the per-frame router under-detects the melody but the note path (note
    F1 ~0.95) still recovers the notes. Real speech yields few/no stable notes
    and keeps its text. Eliminates the hallucination leak (0/225)."""
    for sp in spans:
        if sp.kind != "speech" or sp.t1 - sp.t0 < 0.8:
            continue
        a, b = int(sp.t0 * sr), int(sp.t1 * sr)
        sub_track = _slice_track(f0, sp.t0, sp.t1)
        try:
            notes = melody.segment_notes(audio[a:b], sub_track, pitch_step=pitch_step)
        except Exception:
            notes = []
        # A real hum yields >= min_notes SUSTAINED notes (mean dur ~0.5-1.1 s); a
        # spoken command's voiced vowels yield short note-like blips (~0.1-0.17 s)
        # and MUST keep their text (else real commands lose their words — the
        # live-mic failure mode). Sustained-note duration is the gate-safe
        # discriminator (a hum's notes stay long even in noisy mixed clips).
        if len(notes) >= min_notes and float(np.mean([nt.dur for nt in notes])) > 0.30:
            sp.suppress_text = True


def _slice_track(tr: "melody.F0Track", t0: float, t1: float) -> "melody.F0Track":
    m = (tr.times >= t0) & (tr.times < t1)
    return melody.F0Track(times=tr.times[m] - t0, f0_hz=tr.f0_hz[m], voiced=tr.voiced[m],
                          voiced_prob=tr.voiced_prob[m], sr=tr.sr, hop=tr.hop)


def _safety_net(f: Frames) -> list[SpanProposal]:
    """Fragmentation can drop every sub-MIN_SEG run, leaving nothing. For a
    sounded utterance, fall back to one span over the sounded extent, typed by
    its mean voicing (a clear hum -> melody, else speech)."""
    idx = np.where(f.sound)[0]
    if len(idx) == 0:
        return []
    t0, t1 = f.times[idx[0]], f.times[idx[-1]] + f.dt
    voicing = float(np.mean(f.local_voicing[idx[0]:idx[-1] + 1]))
    frac_stable = float(np.mean(f.f0_std[idx[0]:idx[-1] + 1] < F0_STABLE_MAX))
    kind = "melody" if (voicing > 0.85 and frac_stable > 0.6) else "speech"
    return [SpanProposal(t0=round(t0, 4), t1=round(t1, 4), kind=kind)]


def _f0_gate(f: Frames, i: int, j: int) -> bool:
    """A span commits as melody only if voicing + f0 stability corroborate."""
    voicing = float(np.mean(f.local_voicing[i:j]))
    std = float(np.median(f.f0_std[i:j]))
    return voicing >= GATE_VOICING and std <= GATE_F0_STD


def _sanitize_spans(spans: list[SpanProposal]) -> list[SpanProposal]:
    """Sort and de-overlap: rounding in the carve / merge steps can leave a span
    starting a few ms before the previous one ends, which ir.validate() rejects.
    Clamp each start to the previous end (adjacent is fine) and drop empties."""
    out: list[SpanProposal] = []
    for s in sorted(spans, key=lambda x: x.t0):
        if out and s.t0 < out[-1].t1:
            s.t0 = out[-1].t1
        if s.t1 - s.t0 >= MIN_SEG * 0.5:
            out.append(s)
    return out


def _merge_adjacent(spans: list[SpanProposal], gap: float = 0.30) -> list[SpanProposal]:
    """Merge same-kind spans separated by < ``gap`` (a brief glide / breath that
    fragmented one hum or one phrase — not a real span boundary)."""
    out: list[SpanProposal] = []
    for s in spans:
        if out and out[-1].kind == s.kind and s.t0 - out[-1].t1 < gap:
            out[-1].t1 = s.t1
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Role assignment (linguistic / joint) from the probe transcript.
# --------------------------------------------------------------------------- #

_EXEMPLAR_CUES = ("like", "goes", "go", "something")
_CONTRAST_CUES = ("instead", "not", "rather")
_COMMAND_VERBS = ("kick", "make", "mute", "solo", "start", "go", "add", "bring",
                  "turn", "pan", "loop", "drop", "play", "record")


def _assign_roles(spans: list[SpanProposal], pr: probe.ProbeResult) -> None:
    words = pr.words
    for k, sp in enumerate(spans):
        if sp.kind == "melody":
            before = _words_before(words, sp.t0, 1.5)
            after = _words_after(words, sp.t1, 1.0)
            ctx = " ".join(w.w.lower() for w in before + after)
            if any(c in ctx for c in _CONTRAST_CUES):
                sp.role = "contrast"
            else:
                sp.role = "exemplar"
        elif sp.kind == "speech":
            first = _words_after(words, sp.t0 - 0.05, sp.t1 - sp.t0)
            head = (first[0].w.lower().strip(".,") if first else "")
            sp.role = "instruction" if head in _COMMAND_VERBS else None
    # "make it go X instead of Y": the second melody is the contrast, first exemplar
    melodies = [s for s in spans if s.kind == "melody"]
    if len(melodies) == 2 and melodies[1].role == "exemplar":
        # if an "instead"/"not" sits between them, mark the later one contrast
        mid = " ".join(w.w.lower() for w in _words_between(pr.words, melodies[0].t1, melodies[1].t0))
        if any(c in mid for c in _CONTRAST_CUES):
            melodies[1].role = "contrast"


def _words_before(words, t, span):
    return [w for w in words if t - span <= w.t1 <= t + 0.05]


def _words_after(words, t, span):
    return [w for w in words if t - 0.05 <= w.t0 <= t + span]


def _words_between(words, t0, t1):
    return [w for w in words if w.t0 >= t0 - 0.05 and w.t1 <= t1 + 0.05]


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #


def route(audio: np.ndarray, sr: int, *, strategy: Strategy = "joint",
          f0: Optional[melody.F0Track] = None,
          pr: Optional[probe.ProbeResult] = None,
          pitch_step: float = 1.0) -> list[SpanProposal]:
    """Segment an utterance into speech/melody span proposals.
    ``pitch_step`` = per-voice note-split threshold for the embedded-hum carve."""
    if f0 is None:
        f0 = melody.track_f0(audio, sr)
    if pr is None:
        pr = probe.transcribe(audio, sr)
    frames = compute_frames(audio, sr, f0, pr)
    lab = _LABELLERS[strategy](frames)
    lab = _smooth(lab, int(HYSTERESIS / frames.dt))
    spans = _spans_from_labels(lab, frames, strategy)
    if strategy == "joint":
        spans = _reclassify_hum_spans(spans, frames)
        spans = _carve_embedded_hum(spans, frames)
    if not spans:
        spans = _safety_net(frames)  # never return empty for a sounded utterance
    spans = _sanitize_spans(spans)   # no overlapping segments (ir.validate)
    if strategy in ("linguistic", "joint"):
        _mark_text_containment(spans, audio, sr, f0, pitch_step=pitch_step)  # the ASR-is-evidence containment rule: text only off non-melodic spans
        _assign_roles(spans, pr)
    elif strategy == "acoustic":
        # acoustic can't read roles; default melody->exemplar by position
        for s in spans:
            if s.kind == "melody":
                s.role = "exemplar"
    return spans
