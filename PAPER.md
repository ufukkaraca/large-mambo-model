# Mambo: A Mixed-Modality Voice Interface for a DAW Producer Co-Pilot

**Understanding interleaved speech and hummed melody (with mouth percussion in design) in a single utterance, and acting on it inside a DAW (REAPER)**

**Ufuk Karaca** · Rodyr, Inc.

*Preprint v1, June 2026. Implementation, experiments, and drafting were assisted extensively by AI coding agents (Claude Opus 4.8 and Fable 5) under the author's direction; the author takes full responsibility for the work (see the disclosure in §7.1). Code, fixtures, and every committed run: [github.com/ufukkaraca/large-mambo-model](https://github.com/ufukkaraca/large-mambo-model).*

---

## Abstract

A music producer and an artist talk in a *mixed vocal modality*: spoken instructions interleaved with hummed melody in a single breath, such as *"give me something like ♪hmm-hmm-hmmm♪, but slower."* No mainstream voice interface parses this. Speech recognizers hallucinate words over a hum; voice-to-MIDI tools treat everything as music and have no instruction channel; and even a frontier audio LLM (`gpt-audio-1.5`) reconstructs our synthetic hums at only **F1 ≈ 0.48**, and inconsistently. We present **Mambo**, an open, reproducible producer co-pilot that parses one mixed utterance. It fuses bottom-up acoustic with top-down linguistic evidence to segment the utterance into speech, melody, and percussion spans, routes each to a specialist decoder, and emits a structured representation plus a replayable action plan that drives a DAW (REAPER), built from off-the-shelf parts at near-zero training cost. A structural rule forbids ASR text on a hummed span, giving **0% lyric hallucination**; a reasoning layer inverts it to **dual-decode a *sung* demonstration into both melody and a captured lyric**, a representation acoustic routing cannot express (word-recall ≈0.6 end-to-end versus a structural 0). The task itself is not new (we cite Nokia US6476306B2 and the DAWZY assistant as prior art), so our contribution is the open system, a benchmark with multi-seed confidence intervals, and **three negative results**: an end-to-end fine-tune that *loses* to the modular pipeline, a cross-voice generalization gap (partially mitigated by per-voice calibration), and a reasoning layer that only *ties* a heuristic on role disambiguation. Every reported number reproduces from a committed run.

---

## 1. Introduction and motivation

### 1.1 The producer metaphor

The target interaction is a *sounding board*: the artist stays in flow and talks to the system the way they would talk to a producer sitting at the desk:

- *"Kick the drums up a bit."* This is a *mixing instruction*, resolvable to a relative fader move on a known track.
- *"Give me something like ♪ da-da-da-daa ♪."* This is a *musical sketch*: the payload is not words but a pitch sequence with rhythm.
- *"...but slower, and warmer."* This is a *modifier* that refers back to the sketch, in the same breath.
- *"♪ boots-and-cats ♪ on the kick and hat."* This is *mouth percussion* mapped to drum voices.
- (later) imitating a trombone with the mouth: *timbre intent*, where the melody and the instrument identity arrive in the same sound.

The defining property is that **these modes interleave inside a single utterance**, and the meaning of the whole depends on parsing each part in its own vocabulary: words → text; hums → notes, key, tempo, contour; beatbox → drum onsets; imitation → instrument class.

### 1.2 Why nothing does this today

Three families of technology each cover one corner:

| Family | What it does | Why it fails here |
|---|---|---|
| ASR (Whisper, Apple Speech, Siri) | speech → text | Hums are out-of-distribution. Whisper emits `[Music]` markers or, worse, hallucinated words unrelated to the audio on non-speech inputs [[1]](#references). The melody is destroyed, and garbage leaks into the instruction. |
| Voice-to-MIDI (Dubler 2, imitone, basic-pitch) | voice → notes/drums | Assumes *everything* is musical input. Spoken words become spurious note salad. No instruction channel at all. |
| Audio LLMs (Qwen-Audio/Omni, Gemini audio, gpt-audio) | audio → semantic answers | Strong at "what genre/mood is this", demonstrably weak at *which notes are these* (§3.1, the empirical finding this paper builds on). |

The gap is not one missing model but a missing *architecture*: something that knows **which vocabulary each moment of audio belongs to** and routes accordingly, or a single model trained on exactly this mixed distribution, which (per our survey, §3.5) nobody has published.

### 1.3 Contributions

**In one sentence:** *Mambo parses **command + hummed melody in a single spoken utterance** (via a joint acoustic⊕linguistic segmentation router with a structural ASR-on-hum containment rule) from off-the-shelf parts at ≈zero training cost, handling inline (no mode-switch) what the nearest open system (DAWZY) does behind a separate record-hum button.* Parsing speech+hum from one utterance is **not new**: it appears in expired prior art (Nokia US6476306B2, priority 2000) and, as a DAW assistant, in DAWZY [[40]](#references); we cite both and position against them in §3.5. We name the underlying task **Mixed Vocal Utterance Parsing**: map one mic utterance interleaving spoken instructions, hummed melody, and beatboxed rhythm to a structured `(speech | melody | percussion)` segmentation with per-span decodings and roles. We contribute a metric suite, baselines, an open reproducible co-pilot, a measured reasoning capability, and three negative results (§5.3, §6). What we claim as *new* is narrow and specific: the **structural ASR-on-hum containment rule** (no `text` field may survive on a melody span, enforced in the schema validator, giving 0% lyric hallucination), the **inline single-utterance** treatment, and, its reasoned inverse, a **dual-decode** that promotes a *sung* span to carry both melody and lyric (measured: word-recall ≈0.6 end-to-end vs a containment baseline's structural 0, §6.1g), a representation a strict segment-and-route does not produce; not the speech+hum task itself. Concretely:

1. A first-principles decomposition of mixed-utterance understanding and a survey of prior art, including the closest published systems (§2–3).
2. The **segment-and-route system** (a joint acoustic⊕linguistic router, a versioned intermediate-representation schema carrying the ASR-on-hum containment invariant, specialist decoders, an LLM planner, and a REAPER actuation layer) built and demoed end-to-end (§4).
3. **MamboBench**, a benchmark and metric suite for the task, with baselines and multi-seed confidence intervals (§6).
4. **Three negative results** (an end-to-end fine-tune that loses to the modular pipeline (§5.3), a cross-voice generalization gap and its calibration fix (§6.1f), and a reasoning layer that ties a heuristic on role disambiguation (§6.1g)), with a limitations section (§6.2) and a path to a larger real-voice benchmark (§7).

---

## 2. First-principles analysis

### 2.1 What is actually in the signal

A single co-pilot utterance is a time-ordered sequence of segments, each drawn from a different channel of vocal communication, each demanding a different decoder and a different output alphabet:

| Channel | Carrier | Output alphabet | Decoder family | Maturity |
|---|---|---|---|---|
| Lexical | phonemes | text tokens | ASR | excellent |
| Melodic | f0 contour + voicing | note events (pitch, onset, duration), key, tempo | pitch tracking + note segmentation | good (monophonic) |
| Percussive | broadband transients | onset times + drum class | onset detection + few-shot classification | good *with per-user calibration* |
| Timbral | spectral envelope of imitation | instrument identity | imitation embedding match | research-grade |
| Prosodic | energy/speed/emphasis | modifiers ("bigger", "softer") | mostly redundant with the lexical channel early on | defer |

Two structural facts follow:

**Fact 1: the channels are mutually destructive under the wrong decoder.** ASR on a hum produces hallucinated words (verified, [[1]](#references)); a pitch tracker on speech produces meaningless note salad (speech is voiced only part of the time, with a constantly gliding f0). Therefore *segmentation is not an optimization; it is correctness-critical*.

**Fact 2: the fusion product is naturally symbolic.** Once each segment is decoded in its own alphabet, the utterance becomes a small structured document: text spans + note lists + onset lists, with timestamps. Everything downstream (reasoning, DAW actions, evaluation, logging, training data) gets simpler if this document is an explicit, versioned schema. We call it the **UIR** (§4.6). Notably, this also decouples the *cognition* layer from audio entirely, which matters because the strongest reasoning models (Claude) take no audio input (verified against Anthropic's API reference: supported input modalities are text, images, and documents only).

### 2.2 Three candidate architectures

**A. End-to-end audio LLM.** One model hears the whole utterance and emits the parsed intent. *Verdict: blocked today.* Every direct benchmark of pitch perception in audio LLMs (§3.1) shows they cannot transcribe melody reliably; CMI-Bench's authors state instruction-following audio LLMs "fall significantly short of task-specific supervised MIR models," failing hardest exactly on "structured, time-based outputs like melody extraction" [[3]](#references). An end-to-end model would have to be *fine-tuned into existence*, possible (VocalParse proved a LALM can emit interleaved text+note streams for singing [[5]](#references)) but not a starting point.

**B. Segment-and-route (parallel system).** A classifier routes each window of audio to specialist decoders; outputs are fused into the UIR; a text LLM reasons over it. *Verdict: buildable now, entirely from verified parts.* Apple's built-in SoundAnalysis classifier already distinguishes `speech` / `singing` / `humming` / `whistling` among 300+ on-device classes [[6]](#references); PESTO does streaming pitch at <10 ms latency with ~30k parameters [[7]](#references); the pYIN/Tony note-HMM solves scoop/vibrato-robust note segmentation [[8]](#references); Apple's SpeechTranscriber and WhisperKit both deliver on-device streaming ASR with timestamps [[9]](#references)[[10]](#references).

**C. Hybrid: B now, A later, with B as the teacher.** Build the modular pipeline first. Its by-products (a synthetic mixed-utterance corpus with free labels (§5.2), an evaluation harness (§6), and a working product to compare against) are exactly the prerequisites for the fine-tune. The fine-tuned model only ships if it beats the pipeline on the eval set.

**This paper adopts C.** The parallel system is the floor, the trained model is the ceiling, and the same IR serves both.

### 2.3 Worked example, end to end

> *"Give me something like* ♪ (seven hummed notes) ♪ *but slower, maybe on something warmer."*

1. **Router**: frames 0.0–2.1 s classified `speech`, 2.1–6.3 s `humming` (voicing ratio 0.94, piecewise-stable f0, no 4–8 Hz syllabic modulation), 6.3–9.2 s `speech`. Top-down corroboration (§2.4): the linguistic frame *"something like ___ but"* independently brackets 2.1–6.3 s as a demonstrative span and assigns its role (exemplar).
2. **Speech path**: spans → `"give me something like"` / `"but slower, maybe on something warmer"`.
3. **Melody path**: f0 contour → note HMM → 7 note events; first note C♯4 (−14 cents); Krumhansl–Schmuckler key candidates: B minor 0.71, D major 0.66; tempo estimate 96 BPM (low confidence, since hums are rubato; report it as such).
4. **Fusion** → UIR document (full JSON in §4.6).
5. **Planner** (Claude, with the DAW session state in context): interprets "slower" as tempo below the sketch's ~96, "warmer" as an instrument-palette constraint; emits `play_preview(notes, tempo=80, patch="mellow piano")` *for confirmation*, then on approval `create_midi_region(track, notes, tempo=80)`.
6. **Actuation**: preview through the co-pilot's own synth; on "yes", the notes are inserted into the DAW (REAPER) on the target track via the action plan + MIDI written to its watched inbox.

Note what the IR did to the artist's own loose phrasing ("a seven-note melody starting in C♯ and ending in B minor"): the system does not need the artist to be technically precise. It *derives* the technical description and can echo it back ("a 7-note phrase in B minor, starts on C♯. Want it at 80 BPM?"). That echo-back is the sounding-board UX.

### 2.4 Where segment boundaries come from: bottom-up, top-down, or joint

§2.2 assumed *a* segmenter; there are in fact two distinct sources of evidence for where the segments are, and they fail in complementary ways.

**Bottom-up (acoustic).** Classify each window of audio by how it *sounds* (the §3.3 toolbox: SoundAnalysis/YAMNet class scores, voicing and f0 statistics). Needs no understanding of the words; ceiling is ~90–95% with known hard cases (sung words, sustained spoken vowels) and inherent boundary jitter.

**Top-down (linguistic).** Reason over the *sentence itself*. An utterance like *"can you do something like ___, but slower"* is largely self-describing: the frame *"do something like ___"* predicts that a demonstrative section follows, and brackets it: the span begins where the lexical stream stops making sense after "like" and ends where it resumes ("but slower"). The procedure: probe the **whole** utterance with ASR (word timestamps + confidence retained), let a reasoner detect (a) the linguistic frame and (b) the span where transcription degenerates, then split the audio there and send the unknown span to the **musical mapper** (melody path) for re-decoding. Implemented as cheap rules for the common frames, with an LLM pass as fallback for unusual phrasing. Its strengths are exactly where acoustics are weak:

- Linguistic frames are extremely strong priors. Humans use the same cue: a listener who hears *"something like…"* is already expecting a demonstration before it starts.
- **Role assignment comes free.** The span is known to be an *exemplar* (vs. a contrast, a filler, background noise) because the sentence says so. With multiple musical spans (*"make it go ♪X♪ instead of ♪Y♪"*), only language can label which is the target and which is the reference.
- Deixis (*"…that, but on strings"*) resolves in the same step.
- The minimal version needs **zero additional acoustic models**.

**The trap: why top-down cannot stand alone.** The "hole" in the transcript is not reliably a hole. Whisper *can* fill non-speech with fluent, grammatical confabulated words ([[1]](#references); we harvested 47 such confabulations from real hums for §6.1g), and because humming is voiced, `no_speech_prob`-style flags often stay quiet. (How often it confabulates vs stays silent is voice- and noise-dependent: on our *clean synthetic* hums it usually stays silent, while real and noisy hums confabulate; so the risk is real but not a fixed rate.) A purely textual reader may see *"can you do something like the moon is rising tonight but slower"*, with no visible gap at all. The unknown-span detector therefore must read the **signal-level footprint of confabulation** (per-word log-probability collapse, compression-ratio spikes, `[Music]`/♪ tokens, timestamp instability) and demand **acoustic corroboration** (voicing ratio, f0 stability from the always-running pitch tracker) before committing a span as music. The transcript text alone is insufficient evidence; the transcript's *confidence geometry* plus the f0 contour is.

**Conclusion: joint segmentation.** Acoustics propose without understanding; language disambiguates what acoustics cannot and labels every span's role. Mambo's router (§4.3) fuses both in a propose–verify design. A useful corollary for the build order: the *first* prototype can be **top-down-only with f0 verification** (full-utterance ASR probe + frame rules + pitch tracker, no acoustic classifier at all), and the acoustic channel added as a robustness upgrade. The ablation (acoustic-only vs language-only vs joint) is reported in §6.1a.

**What the ablation found** (qualitatively here; the full 8-seed CIs and paired tests are in §6.1a). The three arms separate as predicted. The acoustic-only arm trails badly everywhere. The interesting comparison is language-only vs joint: on *clean* audio they are a statistical tie, but under noise the joint arm wins decisively and on every seed, because Whisper word-flooding on noisy hums degrades the language-only reader while the acoustic veto and embedded-hum carve hold the joint arm steady. So the acoustic channel earns its keep through **noise-robustness**. Hallucination is a separate matter: it is contained *structurally*, by the containment rule, rather than by the choice of routing arm, and the decisive contrast is against the transcribe-everything baseline (§6.1b).

---

## 3. Prior art and evidence

All claims below were gathered and cross-checked June 2026; primary-source verifications are logged in the repository.

### 3.1 Audio LLMs do not reliably hear pitch (yet): the current evidence, scoped

- **PitchBench** (May 2026) tests absolute/relative pitch, chords, and melodic-line tracking directly across 28 experiments; its abstract conclusion (verified) is *"Current ALMs do not yet possess stable pitch perception, even for controlled synthetic and instrumental stimuli,"* with "pitch hearing remains highly unreliable" across the **frontier** models it evaluates [[2]](#references). Multiple-choice formats inflate scores massively (one reported example is Gemini 3.1 Pro ~7% open-ended note naming versus ~46% multiple-choice), which explains any "near-ceiling" snippet as an MC-format artifact rather than genuine open-ended pitch hearing. **This frontier-level conclusion substantially weakens the "newest omni model might already hear pitch" counterexample.**
- **CMI-Bench** (ISMIR 2025) reframes real MIR tasks as instructions with standard metrics: audio LLMs "fall significantly short of task-specific supervised MIR models" everywhere except captioning, and "all models struggle with tasks requiring structured, time-based outputs like melody extraction and beat tracking" [[3]](#references).
- **MuChoMusic** (ISMIR 2024): the best model reaches only ~51% (Qwen-Audio, 51.4%) and most models score below 50% on human-validated music MCQs, with heavy "language shortcut" reliance: answering from text priors, not audio [[4]](#references). **RUListening** (2025) sharpened this: text-only LLMs scored up to 56.4% on the original benchmark *without hearing anything* [[11]](#references).
- **MMAU music subset**: Qwen2-Audio-7B-Instruct ≈56%, Qwen2.5-Omni ≈69–71%, but the questions are mostly semantic (genre/mood/instrument), not note-level [[12]](#references).

**Scope of this claim.** The borrowed-benchmark evidence above scopes the thesis to: *as of June 2026, on standard pitch/MIR benchmarks, audio LLMs do not reliably transcribe melody*, not the stronger "they cannot, on Mambo." The decisive on-task experiment is **B6** (§6.1c): score current omni models' note reconstruction on Mambo's own clips against the same ground truth the modular pipeline is scored on, using a transposition-invariant precision/recall/F1 decomposition with note-count error. A **frontier** model (`gpt-audio-1.5`, audio ingestion confirmed) emits roughly the *right note count* but reaches only **F1 ≈ 0.48**, and *inconsistently* (0.42–0.52 across 4 temperature-0 runs); the modular pipeline reconstructs the exact sequence far more faithfully and *stably*. So the defensible on-task claim is **not** "omni models can't hear pitch" but: *a specialized tracker substantially out-reconstructs even a frontier general audio LLM at exact note sequences*, and routes pitch to a path that provably works, also winning on latency, cost, and offline use. Caveats: the modular score is an in-distribution ceiling on synthetic hums (§6.1c); the only free *hosted* audio route (`nvidia/nemotron-…-omni:free`) never ingested audio and is reported as a null, not a flattering 0.0.

**Design consequence:** pitch must come from a dedicated tracker, not from an LLM's ears. The LLM's job is semantics over symbols.

Catalog of open audio LLMs relevant to the (optional) fine-tune track, the commercially-usable, music-aware, fine-tunable short list:

| Model | Size / license | Why it matters here |
|---|---|---|
| **Qwen2-Audio-7B-Instruct** | 8.2B, Apache 2.0 | Best-documented LoRA path (LLaMA-Factory, ms-swift); fits one A100; the default fine-tune target [[13]](#references) |
| **Qwen2.5-Omni-7B** | 7B, Apache 2.0 | Stronger music understanding; LoRA confirmed in LLaMA-Factory [[14]](#references) |
| **Qwen3-Omni-30B-A3B** | 30B MoE (3B active), Apache 2.0 | Strongest open music scores (vendor-reported); ~10× training footprint [[15]](#references) |
| Music Flamingo (NVIDIA) | ~8B, **noncommercial** | Best dedicated music brain; license blocks product use [[16]](#references) |
| Voxtral / Phi-4-MM / Ultravox / Kimi-Audio | 3–24B, permissive | Speech-first; little or no music evidence |

Also confirmed: **Claude accepts no audio input** (Anthropic API reference; the Claude apps' voice mode is dictation+TTS around a text model). **Gemini** audio input is cheap (≈$0.002/min on Flash-class models) but shows no evidence of reliable note transcription; **OpenAI realtime audio** is ~30–60× pricier per minute and conversation-optimized. No general-purpose API does note-level hum transcription reliably (absence-of-evidence after targeted search, medium-high confidence).

### 3.2 Voice-to-MIDI: solved as a dedicated task, with known failure modes

**Pitch trackers** (for the streaming front-end):

| Tracker | Real-time | Size | License | Note |
|---|---|---|---|---|
| **PESTO** (Sony CSL, ISMIR 2023) | **yes, native streaming, <10 ms** | ~30k params | **LGPL-3.0** | accuracy near CREPE (CREPE has "800× more parameters"); trained on vocal data; ~0.7 ms/frame ONNX on CPU [[7]](#references) |
| **SwiftF0** (Aug 2025) | yes (ONNX) | ~96k params | — | ~42× faster than CREPE on CPU, more noise-robust; Python/ONNX despite the name [[17]](#references) |
| torchcrepe (tiny) | usable, 10 ms hops | small | MIT | fallback if LGPL is unacceptable |
| pYIN (librosa) | **offline** (full-sequence Viterbi) | DSP | ISC | gold-standard baseline for eval, not streaming |
| basic-pitch (Spotify) | **offline**, file-based | tiny; ships **CoreML** serialization | Apache 2.0 | best *note-level* model; use in batch "transcribe that" mode [[18]](#references) |
| RMVPE | no streaming mode | — | Apache 2.0 | built for vocals over polyphonic mixes; overkill |

**f0 → discrete notes:** the canonical solution is the **Tony/pYIN note-HMM** (Mauch et al. 2015): per-note attack/stable/silent states, where attack states carry σ=5 semitones of pitch tolerance vs σ=0.9 for stable states. This is precisely how vocal scoops and onset glides are absorbed without spawning ghost notes, and vibrato stays inside the stable state's tolerance [[8]](#references). The note's pitch is the median over its stable region.

**Key estimation:** Krumhansl–Schmuckler on the *symbolic* note sequence (duration-weighted pitch-class profile correlated against 24 key profiles). Known to degrade on short fragments, so always emit **top-2 candidates with scores** and let the planner (or the user) disambiguate [[19]](#references).

**Commercial prior art and its lessons.** Vochlea **Dubler 2** (~$91–193) does real-time voice→MIDI (pitched mode) and beatbox→drums (trigger mode, **trained per-user from up to 12 repetitions per sound**). Review consensus: trigger mode is first-class; pitch mode "is most potent when used by a skilled, accurate vocalist." MusicTech's verdict is literally "still better for beats than pitch" [[20]](#references). **imitone** ($25–60) claims <30 ms response and suffers the same ghost-note family of complaints. The transferable design lesson, consistent across products and the Tony paper: **vocal pitch instability is solved with musical priors + hysteresis** (scale lock, a stickiness/latency slider, octave assist, per-user calibration), *not* with a better raw tracker. None of these products has any instruction channel; they assume all input is musical. That assumption is exactly the gap Mambo fills.

Also relevant: Google's hum-to-search is a melody *embedding retrieval* system, not transcription [[21]](#references), a useful pattern for a future "find a loop like this hum" feature (§4.8).

### 3.3 Speech vs. melody routing: hard, but tractable with ~1 s windows

- Humans need ~1 s to distinguish speaking from singing at ~95% accuracy (Ohishi et al., Interspeech 2005: >80% machine accuracy with 2 s; humans ~70% at 300 ms, ~95% at 1 s) [[22]](#references). **Budget 0.5–1 s of decision latency; do not attempt instant frame-level routing.**
- Singing-voice-detection CNNs reach ~90–95% frame accuracy on benchmarks, a realistic ceiling, to be cleaned up with hysteresis/median filtering [[23]](#references).
- **Plain VADs are the wrong tool**: Silero VAD is speech-specialized and frequently does *not* fire on singing/humming [[24]](#references); it cannot gate the melody path.
- Off-the-shelf classifiers with the right label sets exist on both platforms: **Apple SoundAnalysis** (`SNClassifySoundRequest`, macOS 12+) ships 300+ on-device classes including `speech`, `singing`, `humming`, `whistling`, `rapping`, with tunable window/overlap [[6]](#references); **YAMNet** (AudioSet, 521 classes) includes Speech/Singing/Humming at 960 ms windows / 480 ms hops, CPU-light [[25]](#references), the natural router for the Python lab.
- The robust discriminator is a **fusion**: classifier scores ⊕ f0 statistics from the already-running pitch tracker (humming = high voicing ratio, piecewise-stable f0, no 4–8 Hz syllabic modulation; speech = 4–8 Hz syllable rhythm, f0 declination, frequent voicing breaks) ⊕ ~0.5–1 s hysteresis. Realistic target: ≥90–95% utterance-level routing; the known-hard cases are *sung words* and spoken prosody with sustained vowels.
- **No public dataset of interleaved speech+humming utterances exists** (verified absence), so the corpus must be synthesized (§5.2). Closest parallel resources: NUS-48E (sung *and* spoken versions of 48 songs), MUSAN (speech/music/noise, the standard discrimination corpus), HumTrans, MIR-QBSH.

### 3.4 Controlling a macOS DAW: the GarageBand constraints

The shipped target, **REAPER**, is fully scriptable (ReaScript/OSC), so actuation there is routine and is where the system is demoed. **GarageBand** (the original target) was the highest-uncertainty area, and its constraints are what ground the action vocabulary and motivate the capability-honest adapter (§4.8). Findings (current as of GarageBand 10.4.13/10.4.14, mid-2026):

1. **No AppleScript dictionary.** GarageBand 10 dropped the limited scripting that GarageBand '11 had; only the Standard Suite (open/quit) responds. No official automation API of any kind, and no API announcement in any 2025–2026 release. (High confidence, multiple sources [[26]](#references).)
2. **No Mackie Control / HUI**, and the hidden OSC control-surface driver that third-party remotes once used was *removed* in GarageBand 10.2 (2017) [[27]](#references).
3. **Logic Remote's protocol is not a practical channel**: it runs over Apple's MultipeerConnectivity framework; the transport layer was reverse-engineered (evilsocket's `mpcfw`, 2022) but the application-layer command schema has never been published [[28]](#references).
4. **Lua MIDI Device Scripts (MDS).** Since ~10.4.4, GarageBand supports the same Lua controller-scripting system as Logic Pro 10.5+; Apple's release notes state it "provides LUA script support for developers of 3rd-party MIDI controllers." A working community example (M-Audio Oxygen Pro Mini) implements **transport (play/stop/record/cycle), track select, mute/solo, and 8 assignable knobs/faders** [[29]](#references). Strategy: publish a **virtual CoreMIDI device** from the co-pilot app and ship a custom Lua MDS that binds it, making the co-pilot look like a hardware control surface. Caveats (must be verified in Phase 2): the Lua API is undocumented (learn from Apple's bundled scripts); the community install path is *inside the app bundle* (code-signing/update fragility); whether GarageBand reads the user-level MDS folders that Logic documents (`~/Music/Audio Music Apps/MIDI Device Scripts`) is unverified.
5. **Virtual CoreMIDI source for notes works**: GarageBand accepts input from any CoreMIDI source, but plays it on the **currently-selected software-instrument track only** (omni-channel; no per-track channel routing; Program Change does nothing; no MIDI-learn). MIDI is good for *performance*, useless for *app control*. That is what the MDS layer is for [[30]](#references).
6. **Keystroke simulation (CGEvent)** covers the discrete actions: Space play/stop, R record, Ctrl-R arm, Cmd-Opt-N new track, M/S mute/solo selected, Return go-to-start. Requires the Accessibility permission. There is *no menu item* for Record; keystroke is the practical route. Continuous values (fader to −3 dB) have **no** shortcut; those need the MDS fader or AX manipulation [[31]](#references).
7. **Accessibility (AX) scripting** is possible (GarageBand is largely VoiceOver-usable) but brittle, with unnamed elements; the mix-automation UI is reported *not* accessible. Use as last-resort fallback, expect breakage across 10.4.x updates [[32]](#references).
8. **Fallback hosts**, in order of API quality: **REAPER** (ReaScript: full documented Lua/Python API, the right *development harness* to validate the planner before fighting GarageBand), **Ableton Live** (AbletonOSC, NIME 2023), **Logic Pro** (full MCU + Scripter MIDI-FX + same Lua MDS; still no project-state API, so community "Logic MCP" servers resort to AppleScript+AX) [[33]](#references). And the nuclear option: **be your own host** with AudioKit (MIT, Swift; PitchTap, MIDI, Sampler/Synth), full control, but you abandon GarageBand's instruments, Drummer, and timeline [[34]](#references).

**Assessment:** GarageBand can be driven well enough for the co-pilot's core verbs (play notes in, transport, track select, mute/solo, relative volume via MDS fader) through a *composite* adapter. Anything deeper (inserting Apple Loops programmatically, editing regions, plugin parameters) is out of reach without AX fragility, and the design should not promise it. The architecture therefore treats the DAW adapter as a pluggable interface with GarageBand as the first, deliberately-scoped backend (§4.8).

### 3.5 The specific combination is open territory

The closest published work, **VocalParse** (May 2026), fine-tunes a large audio-language model to emit an **interleaved lyric+note sequence** (lyrics, pitch, note values, BPM) in one autoregressive stream, SOTA on singing transcription [[5]](#references). It proves the core premise of the end-to-end track: a LALM *can* be trained to produce joint text+note structured output. But it transcribes *singing*; no evidence it handles spoken instructions interleaved with melody. SongTrans (2024) and STARS (2025) are likewise singing-only. Suno accepts a hum *clip* plus *typed* text, never mixed in one utterance. The nearest *system* is **DAWZY** (NeurIPS 2025 workshop [[40]](#references)): natural-language/voice control of REAPER via Whisper + BasicPitch + an LLM emitting MCP/ReaPy tool calls, but its hum path is **behind a separate "record hum" button** (BasicPitch on an isolated clip), disjoint from the Whisper command path, with no inline segmentation and no ASR-on-hum containment. Mambo is *DAWZY without the mode-switch*: one utterance, segmented and contained, with quantitative note/segment/containment metrics and confidence intervals (§6). The **patent** prior art, **Nokia US6476306B2** (priority 2000), is a **query-by-humming retrieval** system (a spoken keyword plus a hum matched against a *stored melody library*), so it is genuine prior art for the **combined speech+hum input modality and the front-end segmentation**, but it neither transcribes to MIDI nor controls a DAW; we cite it for that input/segmentation lineage, not as humming-to-MIDI prior art. Beyond both, Mambo adds a capability neither has, a **reasoned dual-decode**: where DAWZY's BasicPitch path discards any words and VocalParse transcribes singing only in isolation, Mambo's reasoning layer recognizes a *sung demonstration inside a command utterance* and emits **both** its melody and its captured lyric (measured word-recall ≈0.6 end-to-end on N=8, §6.1g), a representation a strict segment-and-route does not produce, since it sends each span to a single decoder.

**Where Mambo sits (the novelty grid).** Columns are the capabilities a producer co-pilot needs; ✓ = supported, ◑ = partial/separate path.

| System | command + hum in **one utterance** | pitch → notes | spoken instructions | DAW control | ASR-on-hum **containment** | training cost |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Whisper / Siri (ASR) | ✗ | ✗ | ✓ | ✗ | ✗ (hallucinates on non-speech [[1]](#references)) | — |
| Vochlea Dubler 2, imitone | ✗ | ✓ (real-time) | ✗ | ◑ (MIDI) | n/a | — |
| Suno / Loop Copilot | ◑ (clip + *typed* text) | ✓ | ◑ (typed) | ✗ | n/a | high |
| Audio-LLMs (Qwen2-Audio, Gemini) | ◑ | ✗ (unreliable, §3.1) | ✓ | ✗ | ✗ | high |
| VocalParse (2026) | ✗ (singing only) | ✓ | ✗ | ✗ | n/a | high (LALM FT) |
| **DAWZY** (2025) | ✗ (hum = mode-switch) | ✓ (BasicPitch) | ✓ (Whisper) | ✓ (REAPER) | ✗ (separate paths) | low |
| **Mambo (ours)** | **✓** | **✓ (tracker)** | **✓** | **✓ (REAPER)** | **✓ (0%, §6.1)** | **≈$0** |

Mambo is the only row with ✓ across {one-utterance, pitch, instructions, containment}, the wedge. Repeated targeted searches found nothing else that jointly parses instruction-speech + hummed melody in a single breath.

### 3.6 The Apple on-device stack (verified building blocks)

| Need | Component | Status |
|---|---|---|
| Mic capture | `AVAudioEngine` input tap (~100 ms buffers; smaller via AVAudioSinkNode if ever needed) | stable, sufficient |
| Router classifier | SoundAnalysis built-in (`speech`/`singing`/`humming`/`whistling` among 300+ classes, tunable window/overlap); custom **CreateML sound classifier** if it under-performs (16 kHz mono, ~0.975 s windows, 75% overlap) | on-device, free [[6]](#references) |
| Speech→text | **SpeechAnalyzer/SpeechTranscriber** (macOS 26 "Tahoe", WWDC25): on-device, streaming, `AttributedString` results carrying `audioTimeRange` (word/run-level timing); benchmarked ~2.2× faster than Whisper Large V3 Turbo with comparable quality (speed verified; WER parity anecdotal) [[9]](#references). Alternative: **WhisperKit** (argmax), native Swift ANE streaming, 0.45 s hypothesis latency, 2.2% WER reported, word timestamps [[10]](#references) |
| Pitch | PESTO/SwiftF0 via ONNX Runtime (CoreML execution provider) or converted to CoreML; basic-pitch ships CoreML for batch mode | viable; AudioKit PitchTap is the weakest option (classical tracker) |
| Notes out | **CoreMIDI virtual source** (`MIDISourceCreateWithProtocol`); **MIDIKit** Swift package | stable since macOS 11 |
| Trivial intents on-device | **Apple Foundation Models framework** (macOS 26): ~3B on-device LLM, `@Generable` guided generation straight into Swift enums + `Tool` protocol; **4,096-token context** | exactly the shape of a free, offline intent parser; small-model limits apply [[35]](#references) |
| Complex planning | **Claude API** via URLSession (no official Swift SDK; community: SwiftAnthropic): Messages API, tool use, structured outputs (`output_config.format`), prompt caching | §4.7 |

---

## 4. System architecture

### 4.1 Overview

![Mambo system pipeline](docs/figures/mambo_system.png)

***Figure 1.** The segment-and-route pipeline. One utterance is segmented and routed by joint acoustic⊕linguistic evidence (§2.4); each span is decoded by a specialist; the results fuse into the UIR (`mambo.utterance.v1`), where the structural containment rule forbids ASR text on a hummed span (0% lyric hallucination). A reasoning layer then verifies span roles and, for a sung demonstration, **dual-decodes** it into notes + captured lyric (§6.1g); an LLM planner emits the action plan (`mambo.action.v1`) + MIDI that drive REAPER.*

The detailed control/data flow (GitHub-rendered):

```mermaid
flowchart TD
    MIC[Mic capture\nAVAudioEngine, 48kHz\npush-to-talk hotkey] --> RING[Ring buffer + always-on\nf0 tracker side-channel]
    RING --> ROUTER{Segmentation router\nSoundAnalysis/YAMNet scores\n⊕ f0 statistics\n⊕ 0.5–1s hysteresis}
    ROUTER -- speech --> ASR[Speech path\nSpeechTranscriber / WhisperKit\nstreaming, word timestamps]
    ROUTER -- melody --> MEL[Melody path\nPESTO streaming f0\n→ note-HMM Tony-style\n→ key (K-S top-2), tempo, contour]
    ROUTER -- percussion --> PERC[Percussion path (Phase 4)\nonset + few-shot per-user classifier]
    ASR --> UIR[UIR fusion\nmambo.utterance.v1 JSON]
    MEL --> UIR
    PERC --> UIR
    SESSION[DAW session state\nselected track, tempo, key,\ntrack list cache] --> PLANNER
    UIR --> PLANNER{Planner\nClaude tool-use loop\n(claude-opus-4-8)\nor on-device FM for trivial intents}
    PLANNER -- "preview first" --> SYNTH[Co-pilot's own synth\nplay back interpretation\nfor confirmation]
    SYNTH -- user approves --> ACT
    PLANNER --> ACT[DAW adapter\nREAPER]
    ACT --> MIDI[Virtual CoreMIDI source\nnotes → selected track]
    ACT --> MDS[Lua MIDI Device Script\ntransport, track select,\nmute/solo, fader moves]
    ACT --> KEYS[CGEvent keystrokes\nrecord, new track, navigation]
    ACT -.last resort.-> AX[AX UI scripting]
```

**Interaction loop (the sounding-board contract):** capture → parse → *echo the interpretation back* (text + audible preview) → confirm → act. Given §3.3's routing ceiling (~90–95%), silent execution would erode trust; a one-keypress confirm ("hear it? y/n") keeps errors cheap and doubles as labeled feedback data.

### 4.2 Capture

- `AVAudioEngine` input tap, 48 kHz mono, ~100 ms buffers into a ring buffer.
- **Push-to-talk hotkey first, wake-word never (initially).** Two studio-specific reasons: (a) open-mic in a room playing music guarantees router stress; (b) the Dubler reviews document the *monitoring problem*: it is hard to hum accurately while hearing the DAW's output [[20]](#references). PTT sidesteps both and removes a whole class of segmentation errors at the cost of one key.
- The pitch tracker runs **always-on within an utterance** (not only on melody-classified segments) so its f0 statistics are available *to* the router as features, and so no melody onset is lost to router latency.

### 4.3 Segmentation router (joint acoustic + linguistic, propose–verify)

Implements §2.4's joint design. Within a push-to-talk utterance, segmentation runs post-utterance in three passes (the sounding-board UX is post-utterance anyway; the §4.9 latency budget holds):

**Pass 1: probe everything, trust nothing.**
- Acoustic class scores: SoundAnalysis built-in classifier (Swift) / YAMNet (Python lab), ~1 s windows, 0.5 s hop.
- f0 statistics from the always-on tracker: voicing ratio, f0 stability (variance of semitone-quantized f0 within 250 ms), 4–8 Hz syllabic-band modulation energy.
- **ASR probe over the whole utterance**, retaining word timestamps, per-word confidence/avg-logprob, compression ratio, and music/no-speech tokens. *Hallucination containment rule:* the probe transcript is evidence, never truth. No UIR `text` field is ever populated from a span that has not been verified as speech.

**Pass 2: propose.** One proposer combines: (a) **linguistic frame detection** over the probe transcript: demonstrative cues (*"like ___"*, *"goes ___"*, *"make it ___"*, a trailing comparative such as *"but slower"* immediately after a span), rules first, an LLM pass as fallback for unusual framing; (b) ASR-confidence-collapse features (the signal-level confabulation footprint); (c) acoustic class scores and f0 statistics. Output: candidate segments with `(t0, t1, kind, role, confidence)`, roles per §2.4 (exemplar / contrast / filler). Hysteresis over ~0.5–1 s windows, minimum segment 400 ms, boundaries snapped to local energy minima.

**Pass 3: verify, then decode.** A span committed as melody must pass the acoustic gate (voicing ratio, f0 stability) regardless of how confident the linguistic frame was; its probe-ASR text is then discarded and the span re-decoded by the melody path (§4.4). Speech spans keep their transcription (or are re-run through the product ASR if the probe engine differs). Irreconcilable spans are committed as `"kind": "ambiguous"` carrying *both* decodings; the planner is instructed to ask.

**Algorithm 1: Joint propose–verify segmentation** (as implemented, `mambo_lab/router.py`):

```
input  : audio x, sample rate sr
f0  ← pYIN(x)                          # always-on pitch: voicing, f0-std
pr  ← ASR_probe(x)                     # words + per-word logprob, no_speech, music tokens
F   ← frames(x, f0, pr)                # per frame: local_voicing, f0_std, sound, confident_word

# Pass 2 — propose (the ablation arms; "joint" is shipped)
label each frame GAP | SPEECH | MELODY by strategy:
   acoustic   : MELODY  if local_voicing > VOICING_HI ∧ f0_std < F0_STABLE ∧ sound
   linguistic : MELODY  if  sound ∧ ¬confident_word ∧ local_voicing > MELODY_VOICING   # a "hole"
   joint      : the two, fused — linguistic frame proposes, acoustic gate vetoes
L ← hysteresis_smooth(L);  S ← spans(L)         # min 400 ms, boundaries → local energy minima

# Pass 3 — verify + contain (joint only)
S ← carve_embedded_hum(S)              # carve: split a collapsed speech span whose interior is a
                                       #      sustained hum into  speech | melody | speech
for sp in S:                           # speech-verification gate (the containment rule):
   if count_stable_notes(sp) ≥ 3:      #   ≥3 stable notes ⇒ it is a hum ⇒
      sp.kind ← melody;  sp.suppress_text = true     #   ASR text is never trusted onto it
return fuse(S):  melody → pitch tracker · speech → verified text · ambiguous → both decodings
```

The single structural invariant, *a verified-melody span may not carry ASR text*, is what makes hallucination containment a property of the representation (`ir.py.validate()`), not a heuristic; it is the load-bearing line of the whole system.

Notes:
- The no-frame case (utterance *starts* with the hum: *"♪…♪ — that, on strings"*) is covered: acoustics propose the span, the following deixis assigns its role.
- Upgrade path if the built-in acoustic labels under-perform on *your* voice: CreateML sound classifier fine-tuned on ~30 min of self-recorded data (§5.1).
- The §2.4 ablation (acoustic-only vs language-only vs joint) is reported in §6.1a. One open experiment remains: macOS 26's new Audio Mix API claims speech/ambient separation, so test whether it treats humming as "speech" or "ambient" (undocumented).

### 4.4 Melody path

1. **f0**: PESTO streaming (10 ms hops). LGPL-3.0 note: fine for a personal project and for dynamically-linked use; torchcrepe-tiny (MIT) is the swap-in if licensing ever matters.
2. **Notes**: Tony-style HMM (attack σ≈5 st, stable σ≈0.9 st, silent), median pitch over stable region, minimum note 80 ms. Tunables exposed Dubler-style: *stickiness* (transition penalty), optional *scale lock* once a key is established.
3. **Post-analysis**: Krumhansl–Schmuckler key (top-2 + scores); tempo from inter-onset intervals (report confidence, since hums are rubato; the planner should prefer asking over guessing when confidence < ~0.6); contour string (e.g. `"u u d u d d"`), cheap, and useful for "like the one before" matching later.
4. **Batch fallback**: on "transcribe that again properly," re-run the segment through basic-pitch (CoreML) offline for a higher-quality second pass.

### 4.5 Percussion path (Phase 4)

Onset detector (spectral flux) + per-user few-shot classifier over onset-synchronized embeddings. The literature (Ramires' LVT; Delgado 2022) and Dubler both converge on **per-user calibration with ~5–12 examples per drum sound** as the thing that makes this work; generic models generalize poorly across people. Stowell & Plumbley's delayed-decision result sets the latency budget: classify ~50–100 ms after onset, not at onset [[36]](#references). Calibration UX: "give me four kicks… four snares… four hats" (60 seconds, once).

### 4.6 The Utterance IR: `mambo.utterance.v1`

The keystone artifact. Versioned, JSON, designed to be (a) the planner's input, (b) the evaluation target, (c) the fine-tune's output format, and (d) the log/replay format.

```jsonc
{
  "schema": "mambo.utterance.v1",
  "utterance_id": "utt_20260610T172103Z_001",
  "audio": { "sample_rate": 48000, "duration_s": 9.4, "source": "builtin_mic" },
  "segments": [
    {
      "kind": "speech",
      "t0": 0.00, "t1": 2.10, "confidence": 0.97,
      "text": "give me something like",
      "words": [ { "w": "give", "t0": 0.00, "t1": 0.21 } /* … */ ],
      "asr": { "engine": "SpeechTranscriber", "lang": "en-US" }
    },
    {
      "kind": "melody",
      "t0": 2.10, "t1": 6.30, "confidence": 0.91,
      "notes": [
        { "midi": 61, "name": "C#4", "t0": 2.16, "dur": 0.42, "vel": 92, "cents_dev": -14 }
        /* … 6 more … */
      ],
      "analysis": {
        "key_candidates": [
          { "key": "B minor", "score": 0.71 },
          { "key": "D major", "score": 0.66 }
        ],
        "tempo_bpm": { "value": 96, "confidence": 0.55 },
        "contour": "u u d u d d",
        "n_notes": 7
      },
      "f0": { "engine": "pesto", "voicing_ratio": 0.94, "median_hz": 277.2 }
    },
    {
      "kind": "speech",
      "t0": 6.30, "t1": 9.40, "confidence": 0.96,
      "text": "but slower maybe on something warmer"
    }
  ],
  "percussion": [],          // Phase 4: [{ "t": 3.10, "class": "kick", "confidence": 0.88 }]
  "session_context": {
    "daw": "reaper",
    "selected_track": { "index": 2, "name": "Drums", "kind": "instrument" },
    "tracks": [ { "index": 1, "name": "Keys", "kind": "instrument" } /* … */ ],
    "project_tempo_bpm": 120,
    "project_key": "C major",
    "transport": "stopped"
  },
  "history_refs": ["utt_…_000"]   // for "like the one before"
}
```

Schema rules that matter:

- **Timestamps are global** to the utterance so the planner can resolve deixis ("…like *that* but slower", the melody segment temporally adjacent to "that").
- **Ambiguity is first-class**: key candidates and tempo confidences are part of the contract; segments may carry `"kind": "ambiguous"` with both decodings. The planner is instructed to ask rather than guess below confidence thresholds.
- **`session_context` is captured at utterance time** (selected track, tempo, track list) because a DAW may not be queryable later (GarageBand cannot be); the adapter maintains this as a shadow state when needed (§4.8).
- The schema is the **single point of coupling** between perception and cognition. The Phase-5 fine-tuned model's job is defined as: audio in → this JSON out. Same evaluator for both implementations.

### 4.7 Planner and the action plan

A text-domain LLM with tool use, receiving (1) a frozen system prompt defining the producer persona, the tool catalog, and the confirmation policy; (2) the session context; (3) the UIR; (4) recent utterance history.

- **Model**: `claude-opus-4-8` (Messages API, tool use, adaptive thinking) as the default planner. Pricing reality check (verified June 2026): Opus 4.8 $5/$25 per MTok in/out; Sonnet 4.6 $3/$15; Haiku 4.5 $1/$5. A command turn is ~3–5k input tokens (mostly cacheable system prompt + tools) + <1k output; with prompt caching this is **roughly $0.01–0.05 per command on Opus**, negligible for a personal tool. Tiering down for cost is a user decision, not a default.
- **Latency tier**: Apple Foundation Models (on-device, free, ~3B, guided generation into Swift enums) for the closed-vocabulary fast path (transport verbs, mute/solo, "louder/quieter"), with deterministic fallback to Claude when the on-device parse is low-confidence or the intent is open-ended. Its 4,096-token context fits a trimmed UIR but not the full history; design the fast path stateless.
- **Tool catalog (initial)**:

| Tool | Backing mechanism |
|---|---|
| `play_preview(notes, tempo, patch)` | co-pilot's own synth (AVAudioEngine/AudioKit sampler), *not* the DAW |
| `insert_notes(notes, tempo)` | arm record (keystroke) + stream via virtual MIDI to selected track |
| `transport(play\|stop\|record\|to_start\|cycle)` | Lua MDS, keystroke fallback |
| `select_track(index)` / `mute_track` / `solo_track` | Lua MDS |
| `change_track_volume(delta_db)` | Lua MDS fader (relative moves only) |
| `create_track(kind)` | keystroke (Cmd-Opt-N) + AX for the type dialog |
| `set_project_tempo(bpm)` | AX (brittle; may start as "ask the user to do it") |
| `ask_user(question, options)` | UI |
| `search_local_loops(query, melody_ref?)` | own index of Apple Loops metadata on disk; audition via own player; insertion stays manual (drag), honest scope |
| `generate_sample(text_prompt, melody_ref)` | Phase 6: MusicGen-Melody (chromagram conditioning: "vibe of the hum", not exact notes; weights CC-BY-NC, fine for personal use) [[37]](#references) |

- **Policy**: destructive or audible-in-project actions (insert, record, tempo change) require either a preceding `play_preview` confirmation or an explicit user "just do it". Mixer moves (volume nudge, mute) execute immediately; they are cheap to reverse and the artist said them imperatively.
- The planner *never* sees audio; it sees the UIR. This is forced (Claude has no audio input) and desirable (auditable, replayable, model-agnostic).

**The action plan: `mambo.action.v1`.** The planner's tool calls are recorded as a structured, replayable action plan: **the core system's primary output artifact**, deliberately decoupled from any DAW (the build priority is understanding first, execution second, application third). Note-bearing operations are additionally rendered to **standard MIDI files**, so the core deliverable is testable, auditable, and immediately useful (drag the `.mid` into any DAW by hand) before any DAW adapter exists. An example, "insert these notes at this placement":

```jsonc
{
  "schema": "mambo.action.v1",
  "utterance_id": "utt_20260610T172103Z_001",
  "intent_summary": "insert the hummed 7-note phrase, slower, on a warmer patch",
  "needs_confirmation": true,
  "actions": [
    { "op": "play_preview", "args": { "notes_ref": "seg:1", "tempo_bpm": 80, "patch": "mellow_piano" } },
    { "op": "insert_notes", "args": { "notes_ref": "seg:1", "tempo_bpm": 80, "track": { "by": "selected" } },
      "artifacts": { "midi_file": "out/utt_20260610T172103Z_001.mid" } },
    { "op": "ask_user", "args": { "question": "Tempo 80 feel right, or slower?" },
      "when": "analysis.tempo_bpm.confidence < 0.6" }
  ]
}
```

`notes_ref` pointers resolve into the UIR (`seg:1` = the melody segment), so plan and percept stay joined for replay and eval. Actuation layers (§4.8) are pure *consumers* of this schema; the evaluation harness asserts on it directly with no DAW attached (§6). Together the two schemas split the system cleanly: `mambo.utterance.v1` is *what was meant*, `mambo.action.v1` is *what should be done about it*.

### 4.8 Actuation: the DAW adapter (REAPER implemented; GarageBand analyzed)

Nothing above this section depends on it: the core system's outputs are the action plan and MIDI artifacts (§4.7), and the whole understanding stack is developed and evaluated DAW-free. Actuation sits behind a `DAWAdapter` protocol, and **the shipped backend is REAPER**: the plan and MIDI are written to a watched inbox and applied via ReaScript/OSC, with scriptable assertions in tests ("did track 2 gain +2 dB?"). REAPER's full scripting API makes it the natural development and demo target, and it is where every real-voice and live result in this paper was produced.

We also analyzed **GarageBand** as the harder macOS target (the original goal); its tight constraints are what ground the action vocabulary in reality, and they motivate the protocol's capability-honest design. Findings, behind the same `DAWAdapter`:

1. **Notes in**: a virtual CoreMIDI source (MIDIKit). Plays on the *selected* track only, omni-channel ⇒ `select_track` must precede `insert_notes`; timing comes from the co-pilot's own scheduler (timestamped packets), recording armed via keystroke.
2. **Control surface**: publish a second virtual MIDI device ("Mambo Control") and ship a **Lua MIDI Device Script** binding it to transport/track-select/mute/solo/volume, the channel for "kick the drums up a bit" → `change_track_volume(+2 dB)` as a relative encoder move. (Capability ceiling + whether user-level script folders work need a spike before relying on it.)
3. **Keystrokes**: CGEvent for discrete verbs with shortcuts; requires Accessibility permission; guard with a frontmost-app check.
4. **AX**: last resort only, feature-flagged, expected to break across updates.
5. **Shadow session state**: the adapter cannot *query* GarageBand, so it maintains its own model (selected track, tempo, track list) from its own actions plus periodic read-only AX scans. Drift is the main failure mode; mitigation is a cheap "resync" scan plus the confirmation UX.

A GarageBand backend therefore only has to match the proven REAPER interface; Logic Pro (via MCU + the same Lua MDS system) is a natural third backend.

### 4.9 Latency budget

| Stage | Budget |
|---|---|
| Router decision after segment boundary | ≤ 1.0 s |
| ASR finalization after speech ends | ≤ 0.7 s (streaming partials throughout) |
| Note finalization after melody ends | ≤ 0.3 s (f0 is real-time; HMM finalizes at boundary) |
| UIR fusion | ≤ 0.1 s |
| Planner (Claude, cached prompt) | 1–3 s (on-device fast path: ≤ 0.3 s) |
| Actuation | ≤ 0.5 s |
| **Utterance end → preview/action** | **≈ 2–5 s** (fast path ≈ 1.5 s) |

Acceptable for a sounding board (a human producer takes longer); not a live-performance instrument (that is Dubler's job, not Mambo's).

---

## 5. Optional training track

### 5.0 Stage 0: no training

Everything in §4 ships with zero trained-by-us models. This is the floor, and Phases 0–3 of the build plan live here.

### 5.1 Stage 1: custom router (only if needed)

If SoundAnalysis/YAMNet under-perform on the user's actual voice/mic/room: CreateML sound classifier (3 classes + background), trained on ~30–60 min of data: self-recorded speech/hums/beatbox + MUSAN speech + HumTrans hums. Hours of effort, zero cloud cost, on-device inference.

### 5.2 Stage 2: the synthetic mixed-utterance corpus ("MamboMix")

The dataset that does not exist anywhere (verified, §3.3), and the single highest-leverage artifact of the whole project, because **labels are free by construction**:

1. **Templates**: ~50 instruction frames mined from real producer language: `"give me something like {MELODY} but {MOD}"`, `"can the bass do {MELODY}?"`, `"{MELODY} — that, on strings"`, `"make it go {MELODY} instead"`…
2. **Speech spans**: TTS (several voices) *plus* self-recorded readings (a few hundred, since domain prosody matters), *plus* LibriSpeech/Common Voice fillers for router robustness.
3. **Melody spans**: (a) HumTrans segments: 56 h of hummed melodies with MIDI ground truth; **CC BY-NC 4.0**, fine for this personal/research project, but it contaminates any *commercial* redistribution of trained weights; use the **Dynamic HumTrans corrected annotations** (the original onsets are misaligned) [[38]](#references); (b) MIR-QBSH (4,431 hummed/sung queries); (c) self-recorded hums of known MIDI (an afternoon yields hundreds); (d) synthetic "vocal-ish" renditions of random MIDI phrases (sine+formant or DDSP) for unlimited volume.
4. **Splicing**: concatenate speech+melody spans with 80–300 ms gaps, convolve with room IRs, add noise (MUSAN) at varied SNR, vary loudness. Emit the exact `mambo.utterance.v1` ground truth alongside each WAV.
5. **Scale**: 10–50k utterances ≈ 30–140 h audio, generated locally in hours.

This corpus serves three customers: router training/eval (Stage 1), end-to-end pipeline eval (§6), and fine-tune training (Stage 3). The recorded-not-synthetic slice (a few hundred real utterances, hand-checked) is the *test* set; synthetic data never tests itself.

### 5.3 Stage 3: end-to-end LoRA fine-tune

- **Target**: Qwen2-Audio-7B-Instruct (Apache 2.0), best-documented adapter path (LLaMA-Factory lists it; ms-swift recipes exist). Qwen2.5-Omni-7B is the alternative if its stronger music prior shows through.
- **Task format**: VocalParse-style, audio in, `mambo.utterance.v1` JSON out (the same schema the pipeline emits; the pipeline is the teacher and the baseline).
- **Compute (verified pricing)**: LoRA on one A100 80 GB: RunPod ~$1.19–1.39/h, Vast from ~$0.67/h, H100 ~$1.99–2.69/h. 30k clips × 3 epochs ≈ 5–15 GPU-h ≈ **$10–30/run; $50–300 all-in** with sweeps and failures. QLoRA on a 24 GB consumer card is viable for short clips. (On-Mac MLX LoRA of an *audio* model is unverified, so assume a Linux GPU box for training; inference of the result on-Mac is a later problem, and the pipeline remains the shipped default.)
- **Gate**: the fine-tune ships only if it beats the modular pipeline on the held-out *recorded* test set on every §6 metric. Expected realistic outcome: it wins on segmentation-boundary cases and loses on exact pitch until the melody-path tools are distilled in, which is fine; this stage is a later optimization, not the product.

### 5.4 Stage 4+: timbre intent (the trombone)

Vocal-imitation → instrument identity as embedding match: encode the imitation segment (CLAP or MuQ-MuLan embeddings), nearest-neighbor against an instrument-label/text anchor set; VocalSketch/VimSketch are the datasets [[39]](#references). This slots into the UIR as `"timbre_hint": {"label": "trombone", "score": …}` on a melody segment, with no architectural change. Defer until the core loop is loved.

---

## 6. Evaluation and results

All metrics computed by one harness over (a) MamboMix-synthetic (dev) and (b) the recorded, hand-verified set (test). The same harness scores the modular pipeline and any fine-tuned model. That is the point of the UIR.

| Layer | Metric | Tooling | Gate (test set) |
|---|---|---|---|
| Router | segment-level precision/recall/F1 per class; boundary error (ms) | own | F1 ≥ 0.90; boundary ≤ 250 ms |
| Melody | note onset+pitch F1 (50 ms / 50-cent tolerance); RPA on f0 | `mir_eval.transcription` | F1 ≥ 0.75 on clean hums (HumTrans baselines are poor, §3.2, so beat them) |
| Key | top-1 / top-2 accuracy on segments ≥ 5 notes | own (K-S) | top-2 ≥ 0.85 |
| Speech | WER on speech spans only | jiwer | ≤ 1.5× the ASR's clean-speech WER |
| Hallucination | non-empty ASR output on melody spans (the Whisper failure mode) | own | ≤ 2% of melody spans |
| End-to-end (core, DAW-free) | **action-plan accuracy**: ~50-utterance scripted suite, asserting the emitted `mambo.action.v1` against golden plans; compare rendered `.mid` note-for-note (`mir_eval`) | own harness, no DAW | ≥ 80% for the core system; ≥ 90% before the DAW-application stage |
| End-to-end (application) | **execution accuracy**: the same suite replayed through a live `DAWAdapter`. Did the asserted DAW state change happen? | ReaScript assertions (REAPER first), then GarageBand smoke suite | ≥ 90% |
| UX | utterance → preview latency; confirmation-rejection rate | logs | ≤ 5 s; rejections trending down |

### 6.1 Measured results

Everything below is reproducible from committed `runs/` (`make gate-R0`, `make gate-R1`, `mambobench baselines`, and `mambo_lab.eval.{multiseed,voices,b6_omni,b6_frontier,b6_qwen,voiceprint_eval,semantic_reason_eval,dual_decode_eval,ablation}`). Synthetic numbers are dev-set; real-voice numbers are the held-out real-voice gates.

**(a) Segmentation router: the §2.4 ablation (synthetic, 8-seed mean ± 95% t-CI; run `20260617T111138Z-multiseed`, `dirty:false`).**

| arm | seg F1 clean | seg F1 10 dB | hallucination |
|---|---|---|---|
| acoustic (bottom-up) | 0.749 [0.732, 0.766] | 0.784 [0.765, 0.804] | 0% (0/1800) |
| linguistic (top-down) | 0.895 [0.874, 0.915] | 0.865 [0.847, 0.883] | 0.3% (5/1800) |
| **joint (ours)** | **0.919 [0.908, 0.929]** | **0.913 [0.903, 0.923]** | **0.1% (1/1800)** |

Joint is **marginally ahead on clean audio and wins decisively under noise**, confirmed by a **paired** test across 8 seeds (the correct comparison: marginal CIs can overlap while every per-seed difference is one-signed). For joint−linguistic: at **10 dB** the difference is +0.048, **sign-test p = 0.008** (8/8 seeds), bootstrap 95% CI **[+0.036, +0.059]**, d_z 2.73; at **20 dB** +0.044, **p = 0.008** (8/8), CI [+0.028, +0.063], d_z 1.59; on **clean** it is small (+0.024, 7/8 seeds, CI [+0.008, +0.042], **sign-test p = 0.070**, d_z 0.94), i.e. *no worse, marginally better*. Against the acoustic arm joint dominates at every SNR (p = 0.008, d_z 6–9). **On hallucination, it is now low across *all* arms** (joint 0.1%, linguistic 0.3%, acoustic 0%): the structural containment rule plus the de-flutter and embedded-hum-carve refinements largely eliminated the ASR-text-on-hum leakage that an earlier pipeline showed (a stale run reported ~10.7% for the language-only arm; it does not reproduce on current code). So the joint router earns its keep through **noise-robustness** (above), not through a hallucination gap between routing strategies, and joint's residual 0.1% is a *single* routing error (a hum mislabeled as speech) across 1800 spans, not a containment failure (the rule still forbids text on any span labeled melody). The decisive hallucination comparison is against the *transcribe-everything* baseline B1, which hallucinates on **70%** of utterances (§6.1b). That is the "why route at all" result. Melody note onset+pitch F1 **0.997 [0.994, 1.000]** clean / 0.965 [0.953, 0.976] at 10 dB; key top-2 **0.982 [0.966, 0.999]**.

*Within-pipeline component ablation* (single-seed Δ = full − ablated): each robustness mechanism is load-bearing where it should be: the embedded-hum **carve** recovers **+0.019–0.035** segment F1 (most at 20 dB); **de-flutter** recovers **+0.032–0.069** note F1; the **pitch-plateau onsets** measure **0.000** on synthetic (detached *by construction*), so their benefit is the real-legato case (on real-voice recordings, note-count rose 0.52→0.857 with them). The mechanisms earn their rows, and the synthetic gates are *not* merely "built to pass": removing the carve and de-flutter collapses them.

**(b) Baselines: the headline B-table (`mambobench baselines`, full 225-utterance set across all SNRs; committed run `runs/20260616T225527Z-baselines`).**

| # | system | seg F1 | note F1 | hallucination | key@2 |
|---|---|---|---|---|---|
| B1 | whisper-only (status quo) | 0.187 | 0.187 | **69.8%** | 0.00 |
| B3 | acoustic-only | 0.781 | 0.793 | 0% | 0.95 |
| B4 | linguistic-only | 0.829 | 0.789 | 0% | 0.87 |
| **B5** | **joint (ours)** | **0.909** | **0.871** | **0%** | 0.94 |
| B7 | end-to-end LoRA (Qwen2-Audio, §5.3) | 0.465 | — | — | — |

The transcribe-everything status quo (B1) hallucinates words over the hum on **70%** of utterances and destroys the melody (note F1 0.19); routing fixes both, and the modular pipeline (B5) beats the end-to-end LoRA (B7, 0.465, on its own held-out split) at ~zero training cost, a documented negative result for end-to-end at this data scale (§5.3). (The B-table is the *full* set across all SNRs, so the arms sit below their clean-only §6.1a numbers; the systems' *ordering*, and B1's collapse, is what the comparison establishes.)

**(c) B6: does an omni LLM hear pitch on *our* task?** We score note-sequence reconstruction on 12 clean pure hums against the same ground truth the modular pipeline is scored on. An earlier version of this experiment reported a single recall-only number (0.194 for Qwen). But a recall metric on a *length-mismatched* sequence conflates two distinct failure modes: emitting the **wrong pitches** versus emitting **too few notes**. We therefore report a transposition-invariant **precision / recall / F1** decomposition plus mean **note-count error** (|n_pred − n_gt|), which separates them, and we add a **frontier** hosted model (`gpt-audio-1.5`, audio ingestion confirmed via `audio_tokens > 0`). Because the frontier model is **not deterministic** even at temperature 0, we ran it **4 times** and report the mean (± range):

| system | prec | recall | F1 | note-count err |
|---|---|---|---|---|
| **Mambo (modular)** | **1.00** | **1.00** | **1.00** | **0.00** |
| `gpt-audio-1.5` (frontier, 4 runs) | 0.45 | 0.52 | **0.48** (range 0.42–0.52) | 1.71 |
| base Qwen2-Audio-7B (open) | 0.43 | 0.19 | 0.26 | 3.58 |
| free hosted omni (nemotron) | — | — | *inconclusive* | `audio_tokens=0` (null) |

The decomposition separates the two failure modes. The open 7B's low recall is **mostly under-generation**: it emits 2–3 notes regardless of a 5–8-note truth (note-count error 3.6), so the old 0.194 over-read a length artifact as pitch-deafness. A *frontier* omni model emits roughly the **right note count** (error 1.7; often exact: 6/6, 7/7, 5/5) and reaches **F1 ≈ 0.48** (it *partially* hears pitch), but it does not reconstruct the melody at the tracker's fidelity, and it does so **inconsistently** (F1 0.42–0.52 across 4 identical-prompt runs; sd 0.04). The defensible claim is therefore **not** "audio LLMs cannot hear pitch" but: *a specialized pitch tracker substantially and **stably** out-reconstructs even a frontier general audio LLM on exact note sequences*, which is what the modular routing is for (and it wins on latency, cost, and offline use regardless). **Caveat:** the modular 1.00 is an *in-distribution ceiling*: these are additive-synthesis hums whose f0 the pYIN tracker matches by construction; the cross-system *gap*, not the absolute 1.00, is the result (4 runs `runs/*-b6-frontier`; `runs/20260616T083412Z-b6-qwen`).

**(d) Planner (R1): 50-utterance golden suite, action-plan accuracy.**

| backend | accuracy | cost |
|---|---|---|
| OpenRouter free (`gpt-oss-120b`) | 0.880 | $0, hosted |
| **local Ollama `qwen2.5:7b`** | **0.860** | **$0, offline, private** |

A free, fully-offline 7B **clears the 0.80 bar** and matches the hosted free model: the budget user needs no API spend (the deterministic oracle remains the instant fallback).

**(e) Generalization to real voices (N=4, Wilson 95% CIs; nothing tuned on the held-out voices; held-out from `runs/20260616T161753Z-voices`, operator column from `runs/20260616T043431Z-gate-R0`).**

| metric | operator | Eddy | Omer | Rim |
|---|---|---|---|---|
| **containment** (no ASR text on hums) | 1.00 | **1.00** | **1.00** | **1.00** |
| speech-no-hum (no false melody on speech) | 1.00 | 0.88 | 0.62 | **0.12** |
| note-count ±1 | 0.91 | 0.81 | 0.38 | **0.12** |

**Containment, the central §2.4 claim, generalizes across all four voices (1.00).** But the melody detector's **precision** does **not**: on resonant/vibrato voices it both **over-fragments hums** (note-count 0.91→0.12; Omer detected 19 vs 6 hummed) and **fires on speech** (speech-no-hum 0.91→0.12; on Rim, 7 of 8 spoken commands trigger a spurious melody). This is a genuine model generalization gap, not label noise, and traces to a concrete cause: the note-split threshold was a fixed 1.0 semitone, tuned on the operator's voice, so a singer's wider vibrato (Rim's held-note wobble is **3.24 st** vs the operator's ~0.4) repeatedly trips it.

**(f) The fix: per-voice calibration ("voiceprint").** A ~20 s onboarding measures the speaker's held-note vibrato depth and f0 range and loosens the note-split threshold *only* for abnormally wide vibrato. A **deadband** leaves any voice with held-note wobble below **1.0 semitone** at the shipped default, so calibration cannot regress a normal voice. The threshold is set *a-priori* (normal held-note wobble is well under 1 st), and the held-out voices then **validate** it rather than tune it. Note-count ±1 (Wilson 95% CI), each speaker's voiceprint derived from their own held + speech clips (committed run `runs/20260617T110639Z-voiceprint`):

| speaker | measured wobble | per-voice step | baseline | calibrated | Δ |
|---|---|---|---|---|---|
| Eddy (normal) | 0.68 st | 1.00 (deadband, untouched) | 0.81 [0.57, 0.93] | 0.81 [0.57, 0.93] | +0.00 |
| Omer (normal) | 0.60 st | 1.00 (deadband, untouched) | 0.38 [0.18, 0.61] | 0.38 [0.18, 0.61] | +0.00 |
| **Rim** (wide-vibrato) | **3.24 st** (~5×) | 2.50 (loosened) | 0.12 [0.04, 0.36] | **0.31 [0.14, 0.56]** | **+0.19** |

Both normal voices fall inside the deadband (untouched, with **zero regression**, as the design promises), and only the one genuinely pathological voice (Rim's held-note wobble is 3.24 st, ~5× the others) is loosened, recovering note-count **0.12→0.31**. This converts the limitation into a *controlled, parameterized* result: the hum/speech boundary is inherently speaker-dependent, and a light per-user calibration adapts it. Two caveats: (i) at n≈16 per voice the CIs are wide and **+0.19 is suggestive, not certified**; (ii) calibration does **not** recover Omer, whose over-fragmentation has a different cause than wide vibrato (a remaining limitation, not a fixed one). The false-melody-on-speech half is addressed separately by the rules-only semantic-verify pass (§6.2).

**(g) The reasoning layer: where it does and doesn't help.** The reasoning pass (semantic-verify) makes two kinds of judgment, and they fare very differently.

*Role disambiguation (a near-tie).* Classifying a span as command / demonstration / confabulation over real material (14 commands + 16 demonstrations + **47 Whisper confabulations harvested from real hums**), an LLM (gpt-5-mini) scores **0.974** vs a careful rules baseline's **0.961** (run `runs/20260617T114830Z-semantic-reason`). That is a *tie*: real hum-confabulations are mostly degenerate babble ("da da da", "mm-hmm") a heuristic already rejects (confab 47/47 for rules). Reasoning's edge is qualitative: it recovers an out-of-vocabulary command the keyword list misses ("go back to the start and play") but also over-reads a stray "D." So role disambiguation **does not, on its own, justify the reasoning layer**, a third reportable negative result.

*Dual-decode (a measured capability acoustic routing cannot have).* The containment rule gives 0% lyric-hallucination by *dropping every ASR word on a hummed span*, correct for a wordless hum, but it also discards a real **sung** lyric. Acoustic segment-and-route (Nokia, DAWZY) cannot do otherwise: it sends a span to melody **or** speech, never both. Dual-decode is the reasoned exception: for a hummed span carrying suppressed ASR words, reasoning judges whether they are a genuine sung lyric and, if so, promotes the span to `ambiguous` (notes **+** lyric); babble stays melody-only, so containment holds. On **8 sung demonstrations + 3 wordless-hum controls** (operator voice; run `runs/20260617T141416Z-dual-decode`):

| | containment baseline | dual-decode (end-to-end) | dual-decode (transcript-level) |
|---|---|---|---|
| sung-lyric word-recall (n=8) | **0.000** *(drops text by construction)* | **0.61** | 0.83 *(upper bound)* |
| false-lyric on wordless controls (n=3) | 0/3 | **0/3** | 0/3 |

The two numbers differ because the dual-decode's recall depends on *how the candidate lyric is obtained*. The **end-to-end** number (**0.61**, the shipping pipeline; `runs/20260617T171005Z-voicebench`) takes the sung words from the whole-utterance ASR restricted to the *routed melody span's* time window, so tight span boundaries clip some words ("all of the lights tonight" → only "lights tonight" lands in the span). The **transcript-level** number (0.83; `runs/20260617T141416Z-dual-decode`) extracts from the whole utterance with the command frame stripped, an upper bound that assumes perfect lyric/frame separation. Reasoning made the right *structural* call **11/11**: promoted all 8 sung spans, refused all 3 wordless hums (one with real "mmm-mm" babble), so containment held throughout (0/3 false lyrics). What recall it loses is **ASR mishearing** ("then"→"than", "the dark"→"a dog", "float on"→"flow done") **plus melody-span tightness**, not a reasoning error. **Caveats:** small-n (N=8, single voice), no GT note timing here, a proof-of-capability, not a benchmark; the **0.61 → 0.83 gap is a concrete v2 target** (span-boundary widening / reasoning-based lyric extraction). But it is direct evidence for the one genuinely novel, LLM-only thing in the system: **melody and captured lyric from a single sung span**, which a strict segment-and-route does not produce. In sum: reasoning is *not* a better router (it ties heuristics on role), but it *unlocks a representation* (sung melody+lyric) that routing alone cannot reach.

**What is *not* yet measured.** The §6 protocol's named real-voice metrics (note onset+pitch F1, segment F1 *on human voice*) are **PENDING** a hand-labeled recorded GT set. The shipped real-voice real-voice gates above are lighter **proxies** (note-count-±1, no-melody-on-speech, zero-text-on-hums). All real-voice numbers are at small n (n≈6–16, N=4 speakers, the operator plus 3 held-out) with wide CIs: enough to show containment generalizes and note-segmentation does not, not to certify a rate. The frozen, hand-verified multi-voice GT set is the next data milestone, not a claimed result.

The confirmation UX doubles as a labeling flywheel: every accept/reject is a free end-to-end label on real usage, banked for Stage-1/3 training.

### 6.2 Limitations

- **Real-voice GT is a proxy, at small n.** The §6 named metrics (note onset+pitch F1, segment F1 on *human* voice) are PENDING a hand-labeled set; today's real-voice gates are proxies over N=4 speakers, n≈6–16 per metric, with wide Wilson CIs (§6.1e). Containment generalizes; we do **not** yet certify a note-F1 *rate* on real voice.
- **Melody-detector precision does not yet generalize *at the acoustic layer*.** On wide-vibrato voices the raw router over-segments hums and fires on speech (§6.1e). Two fixes sit above it: per-voice calibration recovers note-count on the genuine wide-vibrato outlier (§6.1f, Rim 0.12→0.31, zero regression on normal voices), and a **rules-only semantic-verify pass** (`semantic_verify.py`), reasoning that a *complete command with no demonstrative cue* should carry no hum, recovers the false-melody-on-speech case in the shipped pipeline (**Rim speech-no-hum 0.12→0.75**, no effect on framed demonstrations). The residual cases (sung-vs-confabulated disambiguation) need the LLM phase of that pass, which is designed but not built.
- **Synthetic gates are easy by construction.** Synthetic hums are detached and easier than real ones, so note F1≈1.0 and key top-2≈0.98 are near-ceiling partly *by construction* (GT onsets = amplitude attacks; melodies are tonic-framed). The synthetic numbers bound the *pipeline*, not human-voice difficulty; the §6.1b full-set arms and the real-voice gates are the difficulty signal.
- **Noise is procedural pink noise, tuned-against.** The carve was tuned to the pink-noise 10 dB failure; robustness to real noise (MUSAN, room reverb) is untested and is the experiment most likely to dent the carve claim (future work).
- **Scope.** English-only ASR; monophonic melody only (no chords/polyphony); push-to-talk, post-utterance segmentation (not streaming mid-utterance); evaluated on REAPER (GarageBand actuation is designed, §4.8, but deprioritized).
- **No human upper bound.** The 0.91 router F1 has no human/oracle ceiling on real voice yet; the oracle-segmentation melody path (§6.1) is the only headroom estimate.

---

## 7. Future work, ethics, and reproducibility

Open questions worth a future experiment: does macOS 26's Audio Mix API separate hum-as-speech or hum-as-ambient? B6 has now been run on a frontier model (`gpt-audio-1.5`, F1 ≈ 0.48 and unstable, §6.1c); the open follow-up is whether a frontier model *with a music-specialized decoding head or tool call* closes the remaining gap to the tracker, and whether the gap persists on *real* (non-synthetic) hums where the tracker's in-distribution ceiling no longer holds. The staged build plan and a detailed risk register are kept in the appendices below (repository version; omitted from the PDF).

<!-- PDF-STRIP-START -->
**Risk register.**

| Risk | Severity | Mitigation |
|---|---|---|
| **GarageBand Lua MDS capability ceiling unknown** (undocumented API; bundle-install fragility; user-folder support unverified) | High: it is the only path to fader control | Phase-2 spike is scheduled *early*; fallbacks: keystroke/AX subset, or REAPER/Logic backend; worst case the co-pilot still does notes+transport+navigation |
| Shadow session state drifts from GarageBand reality | Medium | read-only AX resync; confirmation UX catches mistakes; scope tools to relative moves (`+2 dB`, not `to −3 dB`) |
| Router fails on sung words / spoken sustained vowels | Medium (inherent, §3.3) | ambiguous-segment dual-decoding in the UIR; planner asks; PTT discipline; per-user Stage-1 tune |
| **Melody detector precision overfits vocal timbre**: on resonant/vibrato held-out voices it over-fragments hums (note-count 0.91→0.12 across N=4) *and* fires on speech (speech-no-hum 0.91→0.12; Rim 7/8 commands); §6.1(e) | **Medium**, *partially mitigated*: per-voice calibration (§6.1(f)) recovers note-count on the wide-vibrato outlier (Rim 0.12→0.31, wide CI) with zero regression on normal voices; containment held at 1.00 throughout | shipped a ~20 s voiceprint onboarding (deadband vibrato→split-threshold) that parameterizes the detector per voice; remaining work: a dedicated per-user voicing threshold for the false-melody-on-speech half + a frozen hand-verified multi-voice GT |
| macOS 26 dependency (SpeechTranscriber, Foundation Models) | Low-Medium | WhisperKit covers ASR pre-26; on-device planner is an optimization, not a dependency |
| Licenses: PESTO LGPL-3.0; HumTrans CC BY-NC; MusicGen weights CC-BY-NC | Low for a personal project; **blocking** for commercialization of trained weights | tracked per-component; swaps identified (torchcrepe; self-recorded hums) |
| GarageBand update breaks keystrokes/AX/MDS install | Medium, recurring | version-pin, smoke-test script per update, adapter feature flags |
| Apple ships a real GarageBand API or an Apple-Intelligence music feature | Happy risk | the IR/planner/eval survive; only the adapter shrinks |
| Unverified research items | — | consolidated re-verification checklist in the repository; none are release blockers |
<!-- PDF-STRIP-END -->

### 7.1 Ethics, consent, and AI disclosure

**Human data.** The real-voice evaluation (§6.1e–f) uses recordings from the author and three volunteers, collected through an in-browser collector that states the research purpose and obtains explicit consent (name + consent checkbox) before recording; the prompts are scripted commands and neutral hums with no sensitive content. A voice recording is identifying, so: participants consented to research use and may withdraw; **no raw human audio is committed or redistributed** (only derived per-clip metrics + the ground-truth manifest are; the WAVs, including the author's own, are git-ignored); participants are referred to by first name only and approved this use. This is a personal research project with no institutional review board; we therefore followed a deliberately conservative protocol (minimal, non-sensitive data; opt-out on request) and flag the absence of formal IRB oversight as a limitation for any larger study, which would seek ethics approval and a documented data-management plan (the planned N=12–20 collection).

**Reproducibility (artifact statement).** Every quantitative claim traces to a committed `runs/<timestamp>-<phase>/results.json` carrying its git commit and a `dirty` flag; reportable runs require a clean tree. The synthetic corpus is regenerable bit-exact from a fixed seed (`datagen/`), and the gates run from one command (`make gate-R0`/`gate-R1`/`mambobench baselines`); each reported result names the committed run it comes from. Code, the evaluation harness (`mambo_lab/eval/`), and the synthetic ground truth will be released; human audio will not (consent/identifiability), but its derived per-clip scores are committed.

**AI contribution and disclosure.** The author conceived and directed this work, including the architecture, the two-schema contract, the evaluation methodology, the honesty standard, and the decision of what to claim, and verified and takes full responsibility for every result. Generative-AI coding agents (Anthropic Claude, models Opus 4.8 and Fable 5) were used extensively under that direction to implement the system, run the experiments, and draft the manuscript. Per the authorship policies of arXiv, the ICMJE, and the major publishers, an AI system cannot be an author (authorship requires accountability and consent that only a person can hold), so this use is disclosed here rather than credited by authorship. Every reported number is produced by committed code on committed data and is independently reproducible from the artifacts above; none is generated or estimated by a language model.

---

<!-- PDF-STRIP-START -->
## Appendix: Roadmap and cost (staged build plan)

Staged by the project's stated priority: **research → execution → application**. Stages express priority, not a strict serial wall (stage E can begin once R1 is stable); the DAW does not gate anything in stage R.

| Stage | Phase | Deliverable | Cost |
|---|---|---|---|
| **R: Research** (the core: mixed audio → structured action) | R0 | Python lab: file → UIR via the joint router (incl. the language-guided proposer and the §2.4 ablation) | $0 |
| | R1 | Planner → `mambo.action.v1` + rendered `.mid` artifacts; golden-suite eval, no DAW | API pennies |
| | R2 | Percussion path + per-user calibration; router hardening (CreateML) if real logs demand it | $0 |
| | R3 | MamboMix corpus + (optional) Qwen2-Audio LoRA fine-tune | **$50–300** |
| **E: Execution** (real-time) | E1 | Swift capture + streaming router + live UIR + preview synth + confirmation UX | $0 |
| **A: Application** (DAW) | A1 | GarageBand adapter spike: virtual MIDI, keystrokes, Lua MDS capability report | $0 |
| | A2 | Closed loop in GarageBand (hum → confirm → notes in track; "drums up" → fader); REAPER/Logic backends as wanted | API pennies/command |
| **T: Texture** (anytime after R1) | T1 | Timbre hints; loop search; generation (MusicGen-Melody) | $0–GPU pennies |

Full per-phase acceptance criteria, commands, and the verification checklist are in the open-source repository.
<!-- PDF-STRIP-END -->

---

## 8. Conclusion

The studio problem is real and, as of June 2026, unsolved in both research and product: voice interfaces are monolingual in a setting where humans are fluently multilingual across words, melody, and rhythm. Moreover, the obvious unification candidate (audio LLMs) measurably lacks the one faculty the job requires most: pitch. The first-principles answer is to stop asking one model to hear everything: route each vocal register to the decoder that speaks its language, meet in a symbolic intermediate representation, and let a text-domain reasoner, which is excellent at exactly the "producer judgment" part, plan against an honest model of what the DAW can actually do. Every component on that path is verified buildable today at zero training cost, and the same architecture's by-products (corpus, eval harness, logged confirmations) pave the optional road to the end-to-end model.

---

## References

1. Whisper hallucination on non-speech: Barański, Jasiński, Bartolewska, Kacprzak, Witkowski & Kowalczyk (AGH University of Krakow), "Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio," 2025. https://arxiv.org/abs/2501.11378. *We cite only the qualitative phenomenon (Whisper hallucinating unrelated words on non-speech); both the specific 40.3% / 301,317-inference rate and the "no phonetic or semantic connection" phrasing are absent from the abstract and were dropped or paraphrased pending full-text page-verification. Our own harvested confabulations (§6.1g) are the primary on-task evidence we rely on.* For the related harm framing and the ~1% whole-phrase hallucination rate, see Koenecke et al., "Careless Whisper: Speech-to-Text Hallucination Harms," FAccT 2024. https://arxiv.org/abs/2402.08021
2. PitchBench (May 2026). https://arxiv.org/abs/2605.26176
3. CMI-Bench (ISMIR 2025). https://arxiv.org/abs/2506.12285 · https://github.com/nicolaus625/CMI-bench
4. MuChoMusic (ISMIR 2024). https://arxiv.org/abs/2408.01337
5. VocalParse (May 2026). https://arxiv.org/abs/2605.04613 · https://github.com/pymaster17/VocalParse
6. Apple SoundAnalysis built-in classifier (WWDC21). https://developer.apple.com/documentation/soundanalysis/snclassifysoundrequest · https://developer.apple.com/videos/play/wwdc2021/10036/
7. PESTO (ISMIR 2023; TISMIR 2025). https://github.com/SonyCSLParis/pesto · https://arxiv.org/abs/2508.01488
8. Tony / pYIN note HMM: Mauch et al., TENOR 2015. https://www.tenor-conference.org/proceedings/2015/04-Mauch-Tony.pdf
9. Apple SpeechAnalyzer/SpeechTranscriber (WWDC25). https://developer.apple.com/documentation/speech/speechanalyzer · speed comparison: https://www.macstories.net/stories/hands-on-how-apples-new-speech-apis-outpace-whisper-for-lightning-fast-transcription/
10. WhisperKit (Argmax, ICML 2025). https://arxiv.org/html/2507.10860v1 · https://github.com/argmaxinc/WhisperKit
11. RUListening / RUL-MuchoMusic (2025). https://arxiv.org/abs/2504.00369
12. MMAU (2024) and music-subset leaderboard. https://arxiv.org/abs/2410.19168 · https://llm-stats.com/benchmarks/mmau-music
13. Qwen2-Audio (2024). https://arxiv.org/abs/2407.10759 · fine-tuning: https://github.com/hiyouga/LlamaFactory
14. Qwen2.5-Omni-7B. https://huggingface.co/Qwen/Qwen2.5-Omni-7B
15. Qwen3-Omni (Sep 2025). https://github.com/QwenLM/Qwen3-Omni · https://arxiv.org/abs/2509.17765
16. Music Flamingo (NVIDIA, 2025). https://research.nvidia.com/labs/adlr/MF/
17. SwiftF0 (Aug 2025). https://arxiv.org/abs/2508.18440 · https://github.com/lars76/swift-f0
18. basic-pitch (Spotify, ICASSP 2022). https://github.com/spotify/basic-pitch
19. Krumhansl–Schmuckler key finding. http://rnhart.net/articles/key-finding/ · https://essentia.upf.edu/reference/std_Key.html
20. Vochlea Dubler 2 reviews. https://musictech.com/reviews/software-instruments/vochlea-dubler-2-review/ · https://www.soundonsound.com/reviews/vochlea-dubler-2 · https://www.musicradar.com/reviews/vochlea-dubler-2
21. Google Hum-to-Search (2020). https://blog.research.google/2020/11/the-machine-learning-behind-hum-to.html
22. Ohishi et al., Interspeech 2005 (speech/singing discrimination). https://www.isca-archive.org/interspeech_2005/ohishi05_interspeech.html
23. Singing voice detection survey (2022). https://www.mdpi.com/1099-4300/24/1/114
24. Silero VAD on singing. https://github.com/snakers4/silero-vad/discussions/546
25. YAMNet (AudioSet). https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
26. GarageBand 10 AppleScript loss. https://discussions.apple.com/thread/6794976 · https://www.macscripter.net/t/scripting-garagaband/56367
27. Control-surface/OSC removal. http://www.delora.com/advice/logic_cs_problems/
28. Logic Remote / MultipeerConnectivity reverse engineering (evilsocket, 2022). https://www.evilsocket.net/2022/10/20/Reverse-Engineering-the-Apple-MultiPeer-Connectivity-Framework/ · https://github.com/evilsocket/mpcfw
29. GarageBand Lua MIDI Device Scripts. Release notes: https://support.apple.com/en-us/109515 · community example: https://github.com/sverdianto/oxygenpromini_gb_lua · Logic MDS folders: https://support.apple.com/guide/logicpro/automatic-assignment-for-usb-midi-controllers-ctlsbfee6d57/mac
30. GarageBand MIDI input behavior. https://support.apple.com/guide/garageband/play-software-instruments-gbndd5278cd1/10.4.4/mac/11.0 · https://discussions.apple.com/thread/251749930
31. GarageBand keyboard shortcuts. https://support.apple.com/guide/garageband/gbnd58362a62/mac
32. AX/VoiceOver state of GarageBand. https://www.applevis.com/blog/garageband-part-1-basics · https://applevis.com/forum/macos-mac-apps/automation-garageband-mac
33. Alternatives: REAPER ReaScript https://www.reaper.fm/sdk/reascript/reascripthelp.html · AbletonOSC (NIME 2023) https://github.com/ideoforms/AbletonOSC · Logic Scripter https://support.apple.com/guide/logicpro/use-scripter-lgce728c68f6/mac
34. AudioKit. https://github.com/AudioKit/AudioKit · PitchTap: https://www.audiokit.io/SoundpipeAudioKit/documentation/soundpipeaudiokit/pitchtap
35. Apple Foundation Models framework (WWDC25); context window: https://developer.apple.com/documentation/technotes/tn3193-managing-the-on-device-foundation-model-s-context-window
36. Vocal percussion: Ramires LVT https://arxiv.org/abs/1811.02406 · AVP dataset https://arxiv.org/abs/2009.11737 · Delgado 2022 https://arxiv.org/abs/2204.04646 · Stowell & Plumbley 2010 (delayed decision) https://qmro.qmul.ac.uk/xmlui/bitstream/123456789/2581/2/STOWELLDelayedDecision2010POST.pdf
37. MusicGen melody conditioning. https://facebookresearch.github.io/audiocraft/docs/MUSICGEN.html · https://arxiv.org/abs/2306.05284
38. HumTrans https://arxiv.org/abs/2309.09623 (CC BY-NC 4.0: https://huggingface.co/datasets/dadinghh2/HumTrans) · Dynamic HumTrans (corrected annotations) https://arxiv.org/abs/2410.05455 · https://github.com/shubham-gupta-30/humming_transcription
39. VocalSketch https://zenodo.org/records/13862 · MuQ/MuQ-MuLan https://github.com/tencent-ailab/MuQ · NUS-48E https://ieeexplore.ieee.org/document/6694316/ · MUSAN https://arxiv.org/abs/1510.08484 · MIR-QBSH https://music-ir.org/mirex/wiki/2009:Query_by_Singing/Humming
40. DAWZY: natural-language/voice control of a DAW (REAPER) via Whisper + BasicPitch + an LLM emitting MCP/ReaPy tool calls; NeurIPS 2025 AI-for-Music workshop. https://arxiv.org/abs/2512.03289 · https://openreview.net/forum?id=GUgut5mO52. *The nearest system; hum is a separate "record-hum" mode, no inline segmentation or ASR-on-hum containment (the Mambo wedge, §3.5).*
