"""DAW actuation layer (PAPER §4.8) — pure consumers of mambo.action.v1.

REAPER is the first backend (fully scriptable: the brief's recommended dev
harness). The same action plan drives GarageBand/Logic later via their narrower
channels. The producer side (Python) writes plans+MIDI to a watched folder; the
consumer side (REAPER Lua, in gb-bridge/reaper/) applies them.
"""
