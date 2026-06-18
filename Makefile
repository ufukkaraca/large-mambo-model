# Mambo Stage-R lab. All Python runs through the lab's uv environment.
# Targets are phony; the lab venv is created on first `uv run` via uv sync.

LAB := lab
UV  := uv
SEED := 1234
SNRS := clean,20,10

.PHONY: voice-collect help test fixtures fixtures-force fixtures-eleven fixtures-beatbox mambomix gate-R0 gate-R1 gate-R2 gate-R3 uir demo reaper-demo reaper-listen samples-refs sung-recorder pdf clean

help:
	@echo "Mambo Stage-R targets:"
	@echo "  make test          - run pytest (contract + unit)"
	@echo "  make fixtures      - regenerate synthetic fixtures if audio is missing"
	@echo "  make fixtures-force- regenerate synthetic fixtures unconditionally"
	@echo "  make gate-R0       - run the R0 gate table (-> runs/<ts>-gate-R0/)"
	@echo "  make uir FILE=x.wav- file -> mambo.utterance.v1 (R0 pipeline)"
	@echo "  make demo FILE=x.wav - UIR -> action plan + .mid (R1)"
	@echo "  make samples-refs  - render human-humming reference MIDIs"
	@echo "  make studio        - Mambo Studio: live voice+hum cockpit -> REAPER (http://localhost:8765)"

test:
	cd $(LAB) && $(UV) run pytest

# Regenerate audio only when it is absent (it is git-ignored but bit-exact
# from the seed) so `make gate-R0` works from a clean checkout.
fixtures:
	@if [ -z "$$(ls fixtures/synthetic/audio/*.wav 2>/dev/null)" ]; then \
		echo "fixtures audio missing — regenerating (seed=$(SEED))"; \
		$(MAKE) fixtures-force; \
	else \
		echo "fixtures present ($$(ls fixtures/synthetic/audio/*.wav | wc -l | tr -d ' ') wavs)"; \
	fi

fixtures-force:
	cd $(LAB) && $(UV) run python ../datagen/bootstrap.py --out ../fixtures/synthetic --seed $(SEED) --snrs $(SNRS)

# Richer speech via ElevenLabs TTS (needs ELEVENLABS_API_KEY in .env; not for
# clean-checkout CI). Separate corpus so the offline `say` gate stays canonical.
fixtures-eleven:
	cd $(LAB) && $(UV) run python ../datagen/bootstrap.py --out ../fixtures/synthetic_eleven --seed $(SEED) --snrs $(SNRS) --voice-backend eleven

gate-R0: fixtures
	cd $(LAB) && $(UV) run python -m mambo_lab.eval.gate R0

gate-R1: fixtures
	cd $(LAB) && $(UV) run python -m mambo_lab.eval.gate R1

fixtures-beatbox:
	@if [ -z "$$(ls fixtures/percussion/test/*.wav 2>/dev/null)" ]; then \
		echo "beatbox fixtures missing — regenerating"; \
		cd $(LAB) && $(UV) run python ../datagen/beatbox.py; \
	else echo "beatbox fixtures present"; fi

gate-R2: fixtures-beatbox
	cd $(LAB) && $(UV) run python -m mambo_lab.eval.gate R2

# R3 LoRA (the Large Mambo Model). MamboMix corpus -> a separate, git-ignored dir
# so the committed R0-R2 fixtures stay fixed.
mambomix:
	cd $(LAB) && $(UV) run python ../datagen/bootstrap.py --out ../fixtures/mambomix --seed 2024 --scale 8 --snrs $(SNRS)

gate-R3:
	cd $(LAB) && $(UV) run python -m mambo_lab.eval.gate R3

uir:
	@test -n "$(FILE)" || (echo "usage: make uir FILE=path/to.wav" && exit 2)
	cd $(LAB) && $(UV) run python -m mambo_lab.cli uir --file $(FILE)

demo:
	@test -n "$(FILE)" || (echo "usage: make demo FILE=path/to.wav" && exit 2)
	cd $(LAB) && $(UV) run python -m mambo_lab.cli demo --file $(FILE)

# Live REAPER showcase: open REAPER with gb-bridge/reaper/mambo_bridge.lua running,
# then `make reaper-demo` — a hum appears as notes, turns electric, then loops.
reaper-demo: fixtures
	cd $(LAB) && $(UV) run python -m mambo_lab.cli reaper --oracle --file ../fixtures/synthetic/audio/syn_pure_hum_014_clean.wav
	cd $(LAB) && $(UV) run python -m mambo_lab.cli reaper --oracle --text "make it electric"
	cd $(LAB) && $(UV) run python -m mambo_lab.cli reaper --oracle --text "okay now loop that"

# Live push-to-talk: hum/speak into the mic -> REAPER (open REAPER first).
reaper-listen:
	cd $(LAB) && $(UV) run python -m mambo_lab.cli listen

studio:
	cd $(LAB) && $(UV) run python -m mambo_lab.studio

samples-refs:
	cd $(LAB) && $(UV) run python ../datagen/samples_refs.py

# Build the arXiv-ready PDF of PAPER.md (needs pandoc + a LaTeX engine; macOS fonts).
pdf:
	scripts/build_pdf.sh mambo.pdf

# Studio recorder for the sung-lyric demonstrations (dual-decode eval).
sung-recorder:
	python3 tools/sung_recorder/server.py

voice-collect:
	cd tools/voice_collect && python3 -m http.server 8770

clean:
	rm -rf fixtures/synthetic/audio/*.wav
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
