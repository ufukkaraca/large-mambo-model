# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-18

Initial public release. The Stage-R research build: the modular, reproducible
pipeline described in the paper.

### Added
- **Joint acoustic⊕linguistic router** that segments a single utterance into spoken
  and hummed spans and routes each to a specialist decoder (`router.py`).
- **The two-schema contract** — `mambo.utterance.v1` (UIR) and `mambo.action.v1` —
  with the structural containment rule: no ASR text on a hummed span (`ir.py`).
- **Pitch tracker** (`librosa.pyin`), **ASR probe** (faster-whisper), and the
  **reasoning layer**: semantic-verify + sung-lyric dual-decode (`dual_decode.py`).
- **LLM planner** (`planner.py`) driving REAPER via the Lua bridge (`gb-bridge/`).
- **Mambo Studio** — the live voice+hum cockpit (`make studio`).
- **MamboBench** and the R0–R3 gates, with committed `runs/` provenance for every
  number in the paper, including the negative results (LoRA < modular; cross-voice
  generalization gap; reasoning-vs-rules tie on role disambiguation).
- Synthetic fixture generation (`datagen/`) and the in-browser voice collector
  (`tools/voice_collect/`).

[0.1.0]: https://github.com/ufukkaraca/large-mambo-model/releases/tag/v0.1.0
