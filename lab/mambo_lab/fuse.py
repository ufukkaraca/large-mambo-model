"""Fusion (PAPER §4.4 step 4): span proposals -> decoders -> mambo.utterance.v1.

This is where the router's boundaries become a percept: melody spans go to the
melody path, verified speech spans keep their probe text, and the result is the
single coupling artifact between perception and cognition — the UIR. Also exposes
``file_to_uir``, the full file -> UIR pipeline that ``make uir`` and the R0 gate
call.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import ir, melody, probe, router, speech


def fuse(audio: np.ndarray, sr: int, spans: list[router.SpanProposal],
         pr: probe.ProbeResult, f0: melody.F0Track, *,
         utterance_id: str = "utt", source: str = "synthetic",
         pitch_step: float = 1.0,
         session_context: Optional[ir.SessionContext] = None) -> ir.Utterance:
    segments: list[ir.Segment] = []
    for sp in spans:
        a, b = int(sp.t0 * sr), int(sp.t1 * sr)
        if sp.kind == "melody":
            seg = melody.analyze_span(audio[a:b], sr, t_offset=sp.t0,
                                      role=sp.role, confidence=sp.confidence, pitch_step=pitch_step)
            seg.t0, seg.t1 = round(sp.t0, 4), round(sp.t1, 4)
            _clamp_notes(seg)
            segments.append(seg)
        elif sp.kind == "speech":
            if sp.suppress_text:
                # Containment: hum-like span, ASR text not trusted -> no text leak.
                segments.append(ir.Segment(
                    kind="speech", t0=round(sp.t0, 4), t1=round(sp.t1, 4),
                    confidence=sp.confidence, role=sp.role, text=""))
            else:
                st = speech.from_probe(pr, sp.t0, sp.t1)
                segments.append(ir.Segment(
                    kind="speech", t0=round(sp.t0, 4), t1=round(sp.t1, 4),
                    confidence=sp.confidence, role=sp.role, text=st.text,
                    words=st.words or None, asr_engine=st.engine, asr_lang="en"))
        else:  # ambiguous: carry both decodings
            mseg = melody.analyze_span(audio[a:b], sr, t_offset=sp.t0, role=sp.role, pitch_step=pitch_step)
            st = speech.from_probe(pr, sp.t0, sp.t1)
            segments.append(ir.Segment(
                kind="ambiguous", t0=round(sp.t0, 4), t1=round(sp.t1, 4),
                confidence=sp.confidence, role=sp.role, text=st.text,
                words=st.words or None, asr_engine=st.engine,
                notes=mseg.notes, analysis=mseg.analysis, f0=mseg.f0))

    utt = ir.Utterance(utterance_id=utterance_id, sample_rate=sr,
                       duration_s=len(audio) / sr, source=source,
                       segments=segments, session_context=session_context)
    return utt


def _clamp_notes(seg: ir.Segment) -> None:
    """Keep notes inside the committed segment bounds (validation invariant)."""
    if not seg.notes:
        return
    kept = []
    for n in seg.notes:
        if n.t0 < seg.t0 - 1e-3:
            continue
        if n.t0 + n.dur > seg.t1:
            n.dur = round(max(0.0, seg.t1 - n.t0), 4)
        if n.dur > 0.01:
            kept.append(n)
    seg.notes = kept
    if seg.analysis:
        seg.analysis.n_notes = len(kept)


def file_to_uir(audio: np.ndarray, sr: int, *, strategy: router.Strategy = "joint",
                utterance_id: str = "utt", pitch_step: float = 1.0,
                session_context: Optional[ir.SessionContext] = None) -> ir.Utterance:
    """Full pipeline: WAV samples -> mambo.utterance.v1 (validated).
    ``pitch_step`` = the active voiceprint's per-voice note-split threshold (1.0 = default)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    f0 = melody.track_f0(audio, sr)
    pr = probe.transcribe(audio, sr)
    spans = router.route(audio, sr, strategy=strategy, f0=f0, pr=pr, pitch_step=pitch_step)
    utt = fuse(audio, sr, spans, pr, f0, utterance_id=utterance_id, pitch_step=pitch_step,
               session_context=session_context)
    utt.validate()
    return utt
