# Contributing to Mambo

Thanks for your interest. There are two high-value ways to help, and they need
different things from you.

## 1. Contribute a voice (no coding — the most useful thing right now)

Mambo's biggest open question is cross-voice generalization (the paper's real-voice
eval is small, N=4). More diverse voices is the single highest-value contribution.

- Open `tools/voice_collect/` — it's a self-contained HTML page (no install).
- Record ~30 short prompts in your browser. **Nothing uploads**; it builds one
  `.zip` locally.
- Send the `.zip` back (open an issue and we'll share a drop link, or attach it).

Diversity matters more than volume: a range of gender, f0 range, accent, and
musician / non-musician backgrounds is what we lack. By contributing audio you
confirm it's your own voice and you consent to its use for research and to derived
metrics being published (raw audio is never redistributed — see `SECURITY.md`).

## 2. Contribute code

### Dev setup

```bash
cd lab
uv sync --extra asr     # core + the faster-whisper ASR probe (the live pipeline)
uv run pytest           # contract + unit tests — should be all green
```

Add `--extra f0` for the optional PESTO/torchcrepe pitch backends, `--extra train`
for the Modal LoRA track. Python 3.11 or 3.12 (uv installs a matching interpreter).
See `GETTING_STARTED.md` for the full app + REAPER setup and the platform notes.

### The two rules that keep this project honest

This is a research codebase; two conventions are non-negotiable because the paper's
claims depend on them:

1. **The two schemas are the contract.** `mambo.utterance.v1` (`lab/mambo_lab/ir.py`,
   what was meant) and `mambo.action.v1` (`lab/mambo_lab/actions.py`, what to do).
   Any schema change is a version bump plus a migration of every fixture.
2. **A feature isn't done without a gate.** Every reported number must reproduce
   from committed code on committed data. If you add a capability, add or extend a
   gate (`lab/mambo_lab/eval/`) and commit the `runs/` provenance. Don't loosen a
   gate to make a change pass.

A structural invariant you'll see enforced in `ir.validate()`: **ASR text may never
survive on a hummed (melody) span** — it's what gives 0% lyric hallucination. Keep it.

### Workflow

1. Fork and branch from `main`.
2. `uv run pytest` stays green; add tests for new behavior.
3. Run the relevant gate (`make gate-R0` / `gate-R1` / `gate-R2`) if you touched the
   perception, planner, or percussion paths, and include the result in the PR.
4. Match the surrounding style (the code is heavily commented with the *why* — keep
   that density). Keep diffs focused.
5. Open a PR describing what changed and which gate proves it.

### Reporting bugs / ideas

Open an issue. For anything security- or privacy-sensitive (keys, recorded audio),
see `SECURITY.md` instead of filing a public issue.
