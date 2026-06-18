# R3 — LoRA fine-tune (the "Large Mambo Model")

Optional end-to-end model: a LoRA on **Qwen2-Audio-7B-Instruct** that emits
`mambo.utterance.v1` directly from audio. Trains on **Modal** (free $30/mo credit
≈ 8 A100-hours ≈ one 3-epoch LoRA). **Ships only if it beats the modular pipeline
on the held-out test set** (else it is a documented negative result — the brief's
"research dessert").

## One-time operator setup (Modal is OAuth-only)
1. Sign in at https://modal.com/signup → "Continue with Google"
   (`ufukkaraca19@gmail.com`) → $30/mo free credit.
2. `pip install modal && modal token new` (browser auth) **or** copy a token from
   Settings → Tokens, and put it in the git-ignored `.env`:
   ```
   MODAL_TOKEN_ID=ak-...
   MODAL_TOKEN_SECRET=as-...
   ```

## Run (all automatable once the token is set)
```bash
make mambomix                      # scale the synthetic corpus (local; fans)
cd lab && uv run python -m mambo_lab.finetune.dataset --audio-prefix /data/audio/
modal run finetune/modal_app.py --action prepare              # dataset + config -> volume
modal volume put mambo-data fixtures/synthetic/audio /audio   # bulk audio -> volume
modal run finetune/modal_app.py --action train                # LoRA on an A100 (~$3-4)
modal run finetune/modal_app.py --action evaluate             # adapter -> finetune/eval_preds.json
make gate-R3                       # LoRA vs modular on the test set -> ship / negative result
```

## Files
- `qwen2_audio_lora.yaml` — LLaMA-Factory LoRA config (rank 16, 3 epochs, bf16).
- `modal_app.py` — Modal app (prepare / train / evaluate on A100).
- `lab/mambo_lab/finetune/dataset.py` — (audio, GT UIR) → LLaMA-Factory ShareGPT-audio.
- `eval/gate.py::gate_R3` — the ship gate (LoRA ≥ modular on every metric).

## Cost guard
A100-80GB ≈ $3.5/hr; one 3-epoch LoRA ≈ a few hours ≈ within the $30 free credit.
Hard cap $300 across sweeps. The held-out/recorded slice
is **test-only**, never trained on.
