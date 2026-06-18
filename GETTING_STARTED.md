# Getting started with Mambo

Hum a melody and talk to your DAW — Mambo turns mixed *"make the bass louder,
something like ♪…♪ but slower"* speech-and-humming into actions in REAPER. This is
the 5-minute path from a fresh clone to talking to your session.

## 1 · Install (once)

```bash
cd lab
uv sync --extra asr   # core perception stack + the faster-whisper ASR probe (Python 3.11/3.12, fetched by uv)
uv run pytest -q      # optional sanity check — should be all green
```

Pitch tracking is core (`librosa.pyin`), so `asr` is the only heavy extra the live
pipeline needs. Optional: `--extra f0` (PESTO/torchcrepe pitch backends), `--extra
train` (the Modal LoRA track). **API keys are all optional** — Mambo runs offline;
`cp ../.env.example ../.env` to enable the hosted planner or the reasoning layer
(see the key table in the [README](README.md#configuration-api-keys-are-optional)).

## 2 · Connect REAPER (once, ~2 min)

REAPER is the live surface (it's fully scriptable). Install it (free trial at
reaper.fm), then in REAPER: **Actions → Load ReaScript… → `gb-bridge/reaper/mambo_bridge.lua`
→ Run**. You'll see `[mambo] watching …` in the ReaScript console — leave it
running. Full notes (instrument VSTs, custom repo paths): `gb-bridge/reaper/README.md`.

To have it auto-load every time, copy it into REAPER's `Scripts/` folder as
`__startup.lua`. Find that folder via REAPER → **Options → Show REAPER resource path**
(on macOS it's `~/Library/Application Support/REAPER/Scripts/`; Linux/Windows differ).

## 3 · Open Mambo Studio and talk

```bash
make studio      # opens http://localhost:8765
```

- The header shows **🟢 REAPER connected** once REAPER is open with the bridge
  (🟠 if not — Studio still shows what it *would* do).
- The browser will ask for **microphone access** — allow it.
- **Tap the mic** (or press `Space`), speak a command and/or hum a melody, **tap
  again to stop**. Studio shows what it heard and what REAPER was told to do.

### What you can say

| Say / do | Result |
|---|---|
| *hum a melody* | inserts the notes on the Mambo track |
| "make the bass louder" / "kick the drums up" | raises that track's fader |
| "mute the keys" · "solo the lead" | mutes/solos that track |
| "make the bass electric" | sets that track's instrument |
| "loop that" · "go back to the start and play" | transport |
| "something like ♪…♪ but slower" | inserts the hum, slower |

Track names (Drums, Bass, Keys, Lead, Vocal, Guitar, Hats) are find-or-created in
REAPER automatically — no template needed.

## Without REAPER / without a mic

- **Scripted demo (no mic):** `make reaper-demo` — a hummed melody appears, changes
  instrument, loops.
- **A file instead of the mic:** `cd lab && uv run python -m mambo_lab.cli reaper --file my_hum.wav`.
- **Reproduce the research numbers:** `make gate-R0` (synthetic + real-voice gates).
  This regenerates speech fixtures with macOS `say`; on Linux, set `ELEVENLABS_API_KEY`
  and run `make fixtures-eleven` first.

## Troubleshooting

- **"🟠 REAPER not detected"** — REAPER isn't open, or the bridge script isn't
  running (re-run the ReaScript; check for `[mambo] watching …`).
- **Mic does nothing / "barely heard you"** — grant the browser microphone access;
  speak a touch louder/closer. Studio turns off browser noise-suppression on
  purpose so your hum reaches the pitch tracker raw.
- **Commands land on the wrong track** — say the track noun explicitly ("the
  bass", "the drums"); un-named commands target the Mambo/selected track.
