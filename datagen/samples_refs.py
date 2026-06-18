"""Render the human-humming reference MIDIs requested for the reference set.

Writes 4 short, hummable reference melodies as standard MIDI (and a WAV preview)
to fixtures/human/R0-A/refs/. The operator plays each, then hums it a cappella;
because the reference is known MIDI, the recording gets aligned ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

import synth

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "human" / "R0-A" / "refs"

# (name, key, tempo, [(midi, beats)]) — simple, singable, tonic-framed phrases.
REFS = [
    ("hum_ref1", "C major", 96, [(60, 1), (62, 1), (64, 1), (60, 1), (64, 1), (67, 2)]),
    ("hum_ref2", "A minor", 84, [(69, 1), (72, 1), (71, 1), (69, 1), (67, 1), (69, 2)]),
    ("hum_ref3", "G major", 112, [(67, 1), (69, 1), (71, 1), (74, 1), (71, 1), (67, 2)]),
    ("hum_ref4", "D minor", 76, [(62, 1), (65, 1), (69, 1), (65, 1), (64, 1), (62, 2)]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, key, tempo, seq in REFS:
        beat = 60.0 / tempo
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0)  # piano preview
        notes_spec = []
        t = 0.0
        for midi, beats in seq:
            dur = beats * beat
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=midi, start=t, end=t + dur * 0.95))
            notes_spec.append((midi, round(t, 4), round(dur * 0.92, 4)))
            t += dur
        pm.instruments.append(inst)
        pm.write(str(OUT / f"{name}.mid"))
        audio, _ = synth.render_hum(notes_spec, synth.SR, rng)
        import soundfile as sf
        sf.write(str(OUT / f"{name}.wav"), synth.normalize_peak(audio, 0.9), synth.SR, subtype="PCM_16")
        print(f"wrote {name}: {key} @ {tempo} bpm, {len(seq)} notes")
    print(f"refs -> {OUT}")


if __name__ == "__main__":
    main()
