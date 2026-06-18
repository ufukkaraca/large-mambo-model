"""Synthetic fixture bootstrap — PAPER "Phase R0, step 1".

Generates ``fixtures/synthetic/`` with EXACT ground truth by construction so
every R0 gate is runnable with zero human audio:

  * speech spans via macOS ``say`` (several voices, ~25 instruction templates);
  * melody spans = diatonic phrases rendered as voice-like hums (vibrato,
    portamento, formants) — the note-HMM stressors;
  * mixing: speech–melody–speech, hum-first, and two-span "X instead of Y",
    with 80–300 ms gaps and a {clean, 20 dB, 10 dB} SNR sweep;
  * per file: WAV + ground-truth ``mambo.utterance.v1`` + a manifest line with
    full provenance.

Run (from repo root): ``cd lab && uv run python ../datagen/bootstrap.py``
Deterministic: same ``--seed`` reproduces the corpus bit-exact.

Limitation to restate in every report: synthetic hums are easier than real ones.
An S-gate pass is not an H-gate pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import synth  # local module (datagen/synth.py)
from mambo_lab import ir

SR = synth.SR

# --------------------------------------------------------------------------- #
# Vocabulary.
# --------------------------------------------------------------------------- #

# macOS voices to draw speech from (filtered to those installed at runtime).
CANDIDATE_VOICES = ["Alex", "Samantha", "Daniel", "Karen", "Fred", "Victoria", "Tom"]

COMMANDS = [
    "kick the drums up a bit",
    "make the bass a little louder",
    "mute the keys",
    "solo the lead track",
    "start recording on the selected track",
    "go back to the start and play",
    "add a new software instrument track",
    "bring the reverb down on the vocal",
    "turn the hi-hats down",
    "pan the guitar left",
    "loop the chorus",
    "drop the tempo a little",
]

MODS = ["but slower", "but faster", "but a bit higher", "but softer", "and warmer"]

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)


# --------------------------------------------------------------------------- #
# Melody generation.
# --------------------------------------------------------------------------- #


def build_melody(rng: np.random.Generator) -> dict:
    """Return a melody spec: notes [(midi,onset,dur)], key string, tempo bpm."""
    tonic_pc = int(rng.integers(0, 12))
    mode_name, scale = ("major", MAJOR) if rng.random() < 0.5 else ("minor", MINOR)
    key = f"{PITCH_CLASSES[tonic_pc]} {mode_name}"

    # Place the tonic in a comfortable hum register (~C3–B3, lifting low
    # tonics up an octave) so root_midi % 12 == tonic_pc — the key LABEL must
    # match the actual tonic pitch class of the rendered notes.
    root_midi = 48 + tonic_pc  # C3..B3
    if root_midi < 55:
        root_midi += 12

    tempo = float(int(rng.integers(70, 132)))
    beat = 60.0 / tempo
    n_notes = int(rng.integers(5, 9))

    # Tonal random walk: start AND end on the tonic so the labeled key is the
    # melody's actual key (K-S is a perceptual model — it should agree). A note
    # held longer on the tonic at the cadence reinforces it.
    degrees = [0]
    for _ in range(n_notes - 1):
        step = int(rng.choice([-2, -1, -1, 1, 1, 2]))
        degrees.append(degrees[-1] + step)
    degrees[-1] = int(round(degrees[-2] / 7.0)) * 7  # cadence on nearest tonic

    dur_choices = np.array([0.5, 1.0, 1.0, 1.5, 2.0])  # in beats
    notes: list[tuple[int, float, float]] = []
    onset = 0.0
    for i, degree in enumerate(degrees):
        oct_shift, d = 0, degree
        while d < 0:
            d += 7
            oct_shift -= 1
        while d >= 7:
            d -= 7
            oct_shift += 1
        midi = root_midi + scale[d] + 12 * oct_shift
        is_cadence = i in (0, len(degrees) - 1)
        dur_beats = 2.0 if is_cadence else float(rng.choice(dur_choices))
        dur = dur_beats * beat
        notes.append((int(midi), round(onset, 4), round(dur * 0.92, 4)))  # slight detache
        onset += dur

    contour = _contour(notes)
    return {"notes": notes, "key": key, "tempo": tempo, "contour": contour}


def _contour(notes: list[tuple[int, float, float]]) -> str:
    syms = []
    for (a, _, _), (b, _, _) in zip(notes, notes[1:]):
        syms.append("u" if b > a else "d" if b < a else "=")
    return " ".join(syms)


# --------------------------------------------------------------------------- #
# Span assembly.
# --------------------------------------------------------------------------- #


def _melody_segment(notes_truth: list[synth.RenderedNote], spec: dict, t_off: float, role: str) -> ir.Segment:
    notes = [
        ir.Note(midi=n.midi, t0=round(t_off + n.t0, 4), dur=n.dur, vel=n.vel)
        for n in notes_truth
    ]
    analysis = ir.MelodyAnalysis(
        n_notes=len(notes),
        key_candidates=[ir.KeyCandidate(spec["key"], 1.0)],
        tempo_bpm=spec["tempo"],
        tempo_confidence=1.0,
        contour=spec["contour"],
    )
    f0 = ir.F0Stats(engine="ground_truth", voicing_ratio=1.0,
                    median_hz=round(synth.midi_to_hz(float(np.median([n.midi for n in notes_truth]))), 2))
    return ir.Segment(kind="melody", t0=round(t_off, 4), t1=round(t_off + notes_truth[-1].t0 + notes_truth[-1].dur, 4),
                      confidence=1.0, role=role, notes=notes, analysis=analysis, f0=f0)


def _render_speech(text: str, voice: str, backend: str) -> np.ndarray:
    """Render a speech span via the chosen backend (default offline `say`)."""
    if backend == "eleven":
        from mambo_lab import eleven

        return synth._trim_silence(eleven.tts(text, voice=voice, sr=SR))
    return synth.render_speech(text, voice, SR)


def assemble(spans: list[dict], rng: np.random.Generator, backend: str = "say"
             ) -> tuple[np.ndarray, list[ir.Segment]]:
    """Concatenate spans with 80–300 ms gaps; return audio + GT segments (global time)."""
    chunks: list[np.ndarray] = []
    segments: list[ir.Segment] = []
    t = 0.0
    for i, span in enumerate(spans):
        if i > 0:
            gap = rng.uniform(0.08, 0.30)
            chunks.append(np.zeros(int(gap * SR), dtype=np.float32))
            t += gap
        if span["type"] == "speech":
            audio = _render_speech(span["text"], span["voice"], backend)
            dur = len(audio) / SR
            seg = ir.Segment(kind="speech", t0=round(t, 4), t1=round(t + dur, 4), confidence=1.0,
                             role=span.get("role"), text=span["text"])
            chunks.append(audio)
            segments.append(seg)
            t += dur
        else:  # melody
            audio, truth = synth.render_hum(span["spec"]["notes"], SR, rng)
            chunks.append(audio)
            segments.append(_melody_segment(truth, span["spec"], t, span.get("role", "exemplar")))
            t += len(audio) / SR
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32), segments


# --------------------------------------------------------------------------- #
# Templates -> span lists.
# --------------------------------------------------------------------------- #


def make_base(kind: str, idx: int, voices: list[str], rng: np.random.Generator) -> tuple[list[dict], str]:
    """Return (spans, template_label) for one base utterance of the given kind."""
    v = lambda: voices[int(rng.integers(0, len(voices)))]  # noqa: E731
    if kind == "speech_command":
        return [{"type": "speech", "text": COMMANDS[idx % len(COMMANDS)], "voice": v(), "role": "instruction"}], "speech_command"
    if kind == "pure_hum":
        return [{"type": "melody", "spec": build_melody(rng), "role": "exemplar"}], "pure_hum"
    if kind == "like_but":
        return ([
            {"type": "speech", "text": "give me something like", "voice": v(), "role": "instruction"},
            {"type": "melody", "spec": build_melody(rng), "role": "exemplar"},
            {"type": "speech", "text": MODS[idx % len(MODS)], "voice": v()},
        ], "like_but")
    if kind == "can_bass":
        return ([
            {"type": "speech", "text": "can the bass do this", "voice": v(), "role": "instruction"},
            {"type": "melody", "spec": build_melody(rng), "role": "exemplar"},
        ], "can_bass")
    if kind == "hum_first":
        return ([
            {"type": "melody", "spec": build_melody(rng), "role": "exemplar"},
            {"type": "speech", "text": "that, but on strings", "voice": v()},
        ], "hum_first")
    if kind == "contrast":
        return ([
            {"type": "speech", "text": "make it go", "voice": v(), "role": "instruction"},
            {"type": "melody", "spec": build_melody(rng), "role": "exemplar"},
            {"type": "speech", "text": "instead of", "voice": v()},
            {"type": "melody", "spec": build_melody(rng), "role": "contrast"},
        ], "contrast")
    if kind == "warmer":
        return ([
            {"type": "speech", "text": "something like", "voice": v(), "role": "instruction"},
            {"type": "melody", "spec": build_melody(rng), "role": "exemplar"},
            {"type": "speech", "text": "on something warmer", "voice": v()},
        ], "warmer")
    raise ValueError(kind)


# Plan of base utterances: (kind, count). Guarantees coverage of special cases.
PLAN = [
    ("speech_command", 14),
    ("pure_hum", 12),
    ("like_but", 14),
    ("can_bass", 8),
    ("hum_first", 8),
    ("contrast", 9),
    ("warmer", 10),
]  # total 75 base utterances


def snr_label(snr) -> str:
    return "clean" if snr is None else f"{int(snr)}db"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../fixtures/synthetic")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--snrs", default="clean,20,10")
    ap.add_argument("--limit", type=int, default=0, help="cap base utterances (smoke runs)")
    ap.add_argument("--voice-backend", choices=["say", "eleven"], default="say",
                    help="speech source: offline macOS `say` (default, no key) or ElevenLabs TTS")
    ap.add_argument("--scale", type=int, default=1, help="multiply per-template counts (MamboMix)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)

    if args.voice_backend == "eleven":
        from mambo_lab import eleven

        voices = list(eleven.VOICES)
    else:
        if not synth.say_available():
            raise SystemExit("macOS `say` not available — cannot synthesize speech spans.")
        import subprocess

        installed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        voices = [vn for vn in CANDIDATE_VOICES if vn in installed] or ["Alex"]

    snrs = [None if s == "clean" else float(s) for s in args.snrs.split(",")]
    rng = np.random.default_rng(args.seed)

    base_specs: list[tuple[list[dict], str]] = []
    for kind, count in PLAN:
        for i in range(count * args.scale):
            base_specs.append(make_base(kind, i, voices, rng))
    if args.limit:
        base_specs = base_specs[: args.limit]

    manifest = open(out / "manifest.jsonl", "w", encoding="utf-8")
    n_written = 0
    for bi, (spans, label) in enumerate(base_specs):
        # Render the clean utterance once (deterministic), then add noise per SNR.
        render_rng = np.random.default_rng(args.seed * 100003 + bi)
        clean, segments = assemble(spans, render_rng, backend=args.voice_backend)
        clean = synth.normalize_peak(clean, 0.9)
        for snr in snrs:
            noise_rng = np.random.default_rng(args.seed * 7 + bi * 13 + (0 if snr is None else int(snr)))
            audio = clean if snr is None else synth.mix_at_snr(clean, synth.pink_noise(len(clean), noise_rng), snr)
            uid = f"syn_{label}_{bi:03d}_{snr_label(snr)}"
            wav_path = out / "audio" / f"{uid}.wav"
            sf.write(str(wav_path), audio, SR, subtype="PCM_16")

            utt = ir.Utterance(
                utterance_id=uid, sample_rate=SR, duration_s=len(audio) / SR,
                source="synthetic", segments=segments,
            )
            ir.dump(utt.to_dict(), str(out / "truth" / f"{uid}.uir.json"))

            manifest.write(json.dumps({
                "utterance_id": uid, "template": label, "snr": snr_label(snr),
                "n_segments": len(segments), "duration_s": round(len(audio) / SR, 3),
                "voices": sorted({s["voice"] for s in spans if s["type"] == "speech"}),
                "voice_backend": args.voice_backend,
                "seed": args.seed, "wav": f"audio/{uid}.wav", "truth": f"truth/{uid}.uir.json",
            }) + "\n")
            n_written += 1
    manifest.close()
    print(f"wrote {n_written} utterances ({len(base_specs)} base × {len(snrs)} SNR) to {out}")


if __name__ == "__main__":
    main()
