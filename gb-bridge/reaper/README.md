# Mambo → REAPER (the live showcase)

Hum a melody and talk to it — it appears as notes on a track in REAPER, plays,
and reacts to your voice: *"make it electric," "loop that," "kick it up."* REAPER
is fully scriptable, so the whole loop works here (unlike GarageBand, which can't
script instrument changes).

How it works: the Mambo pipeline turns your audio into a `mambo.action.v1` plan +
a rendered `.mid`, and drops them into an **inbox folder**. A small Lua script you
load in REAPER watches that folder and applies each plan. No Python-in-REAPER, no
extra installs.

## Setup (once, ~2 min)
1. Install **REAPER** (free, fully-functional trial: reaper.fm) and open it.
2. Actions → **Load ReaScript…** → pick
   `gb-bridge/reaper/mambo_bridge.lua` → **Run**. You'll see
   `[mambo] watching …` in the ReaScript console. Leave it running.
   - *(If your repo path differs, edit `INBOX` at the top of the `.lua`.)*
3. *(Optional, for real sounds)* edit the `PATCHES` table in the `.lua` to point
   "electric_piano", "strings", etc. at instrument VSTs you have. Out of the box
   it uses stock **ReaSynth**, so you'll hear the notes immediately.

## Run the demo
**Canned showcase (always works, no mic needed):**
```bash
make reaper-demo
```
Watch REAPER: a **"Mambo"** track appears with the hummed notes and plays → the
instrument changes → it starts looping.

**Your own voice / commands:**
```bash
cd lab
uv run python -m mambo_lab.cli reaper --file my_hum.wav      # a hum -> notes
uv run python -m mambo_lab.cli reaper --file "make_it_electric.wav"  # spoken command
uv run python -m mambo_lab.cli reaper --text "okay now loop that"    # or just type it
```
- `--file` runs the real perception + the (free) LLM planner.
- `--text` skips listening — handy for trying commands.
- `--oracle` uses the deterministic planner (no network) — most reliable for demos.
- `--dry-run` prints what it *would* do without touching REAPER.

## What's wired
hum → `insert_notes` (notes on the Mambo track, at the hummed tempo) ·
`set_track_instrument` ("make it electric/strings/synth/warm") ·
`transport` ("play", "stop", "loop that" → cycle, "back to the start") ·
`change_track_volume` ("kick it up" → fader) · `mute`/`solo` · `set_project_tempo`.

## Honest limits (rough demo)
- Stock ReaSynth gives one synthy timbre by default; map `PATCHES` to real VSTs
  for convincing piano/EP/strings.
- The free LLM planner is ~88% accurate; `--oracle` is deterministic.
- Latency is a few seconds (sounding-board feel), not a live instrument.
- This is the REAPER backend; GarageBand (notes + transport + faders only) and a
  real-time Mac app (Stage E) come next.
