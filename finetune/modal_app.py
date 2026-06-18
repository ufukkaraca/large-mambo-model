"""Modal app — R3 LoRA training of Qwen2-Audio-7B on the free $30/mo credits.

End-to-end on Modal (the brief's "rented Linux GPU box"):
  modal run finetune/modal_app.py::prepare    # upload dataset + audio + config
  modal run finetune/modal_app.py::train       # LoRA fine-tune on an A100
  modal run finetune/modal_app.py::evaluate    # adapter -> UIRs on the test set
  modal volume get mambo-output mambo-qwen2audio-lora ./finetune/adapter

Auth: a Modal account ($30/mo free) + token are operator-provided (Modal is
OAuth-only). Once `modal token set` is run (or
MODAL_TOKEN_ID/SECRET are in the env), this runs unattended. The LoRA SHIPS ONLY
IF it beats the modular pipeline on the held-out test set (the R3 gate).
"""

from __future__ import annotations

import modal

app = modal.App("mambo-lora")

# Image: LLaMA-Factory + audio deps. Qwen2-Audio needs transformers + librosa.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch", "transformers>=4.45", "accelerate>=0.34", "datasets",
        "peft>=0.12", "librosa", "soundfile", "torchaudio", "torchcodec",
        # NOTE: no deepspeed — single-GPU LoRA doesn't need it, and accelerate
        # tries to import it (failing on CUDA_HOME) merely because it's installed.
        "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git",
    )
)

data_vol = modal.Volume.from_name("mambo-data", create_if_missing=True)
out_vol = modal.Volume.from_name("mambo-output", create_if_missing=True)
hf_cache = modal.Volume.from_name("mambo-hf-cache", create_if_missing=True)

GPU = "A100-80GB"  # ~$3.5/hr; $30 free credit ≈ 8 h ≈ enough for a 3-epoch LoRA
VOLS = {"/data": data_vol, "/output": out_vol, "/root/.cache/huggingface": hf_cache}


@app.function(image=image, volumes={"/data": data_vol}, timeout=1800)
def prepare(dataset_json: bytes, dataset_info: bytes, config: bytes) -> str:
    """Write the (small) dataset json + config into the volume. Bulk audio is
    uploaded separately via `modal volume put mambo-data ./audio /audio`."""
    import pathlib

    root = pathlib.Path("/data")
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "dataset" / "mambomix_train.json").write_bytes(dataset_json)
    (root / "dataset" / "dataset_info.json").write_bytes(dataset_info)
    (root / "config.yaml").write_bytes(config)
    data_vol.commit()
    return "wrote dataset json + config (upload audio via `modal volume put`)"


@app.function(image=image, gpu=GPU, volumes=VOLS, timeout=4 * 3600)
def train() -> str:
    import subprocess

    subprocess.run(["llamafactory-cli", "train", "/data/config.yaml"], check=True)
    out_vol.commit()
    return "training complete -> /output/mambo-qwen2audio-lora"


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=2 * 3600)
def evaluate(test_json: bytes) -> list[dict]:
    """Run the trained adapter on the held-out test set -> predicted UIR strings.
    Returns [{utterance_id, gold, pred}] for the local R3 gate to score against
    the modular pipeline."""
    import json

    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    from peft import PeftModel
    import librosa

    base = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(base, device_map="cuda", torch_dtype="bfloat16")
    model = PeftModel.from_pretrained(model, "/output/mambo-qwen2audio-lora")
    model.eval()

    out = []
    for ex in json.loads(test_json):
        wav = ex["audios"][0].replace("/data/audio/", "/data/audio/")
        audio, _ = librosa.load(wav, sr=16000)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": wav},
                                             {"type": "text", "text": ex["messages"][0]["content"].replace("<audio>", "")}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[audio], sampling_rate=16000, return_tensors="pt").to("cuda")
        gen = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        pred = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        gold = ex["messages"][1]["content"]
        out.append({"audio": wav, "gold": gold, "pred": pred})
    return out


@app.function(image=image, gpu="A10G",
              volumes={"/root/.cache/huggingface": hf_cache}, timeout=1800)
def b6_transcribe(clips: list[dict], prompt: str) -> list[dict]:
    """B6, the *valid* route: BASE Qwen2-Audio-7B (NO LoRA) transcribes hummed
    melodies to notes on Mambo's own pure-hum clips. Unlike the OpenRouter free
    omni route — where the audio never reached the model (audio_tokens=0) — a real
    audio LLM run locally on GPU *does* ingest the audio, so its score is an
    interpretable on-task answer to "can an omni model hear pitch?" (PAPER §3.1/§6).
    `clips=[{uid, wav: bytes}]`; returns `[{uid, text}]`."""
    import inspect
    import io

    import librosa
    import soundfile as sf
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    base = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base, device_map="cuda", torch_dtype="bfloat16")
    model.eval()
    # transformers renamed the audio kwarg across versions (`audios` vs `audio`);
    # earlier this silently dropped the audio. Detect it and verify ingestion.
    params = inspect.signature(proc.__call__).parameters
    audio_kw = "audios" if "audios" in params else ("audio" if "audio" in params else "audios")

    out = []
    for c in clips:
        audio, sr = sf.read(io.BytesIO(c["wav"]), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "clip.wav"},
            {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, return_tensors="pt", padding=True,
                      sampling_rate=16000, **{audio_kw: [audio]}).to("cuda")
        # audio truly ingested iff the processor produced acoustic features
        ingested = any("feature" in k or "input_values" in k for k in inputs.keys())
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        pred = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out.append({"uid": c["uid"], "text": pred, "audio_ingested": bool(ingested),
                    "audio_kw": audio_kw, "input_keys": list(inputs.keys())})
    return out


@app.local_entrypoint()
def main(action: str = "train"):
    """`modal run finetune/modal_app.py --action prepare|train|evaluate|b6`."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    data = repo / "finetune" / "data"
    if action == "b6":
        # Valid B6: base Qwen2-Audio on the 12 clean pure-hum clips. Reads fixtures
        # directly (no mambo_lab import here) and writes raw transcriptions; scoring
        # happens in the lab env via `python -m mambo_lab.eval.b6_qwen`.
        fx = repo / "fixtures" / "synthetic"
        rows = [json.loads(x) for x in (fx / "manifest.jsonl").read_text().splitlines() if x.strip()]
        ph = [r for r in rows if r["snr"] == "clean" and "pure_hum" in r["utterance_id"]]
        clips = [{"uid": r["utterance_id"], "wav": (fx / r["wav"]).read_bytes()} for r in ph]
        prompt = ("This audio is a person humming one monophonic melody. List the musical "
                  "notes you hear, in order, with octaves (e.g. C4, E4, G4). End your reply "
                  "with a line 'NOTES: <comma-separated list>'.")
        print(f"B6: sending {len(clips)} pure-hum clips to base Qwen2-Audio on Modal (A10G)…")
        res = b6_transcribe.remote(clips, prompt)
        (repo / "finetune" / "b6_qwen_raw.json").write_text(json.dumps(res, indent=2))
        print(f"wrote {len(res)} transcriptions -> finetune/b6_qwen_raw.json")
        print("score it:  cd lab && uv run python -m mambo_lab.eval.b6_qwen")
        return
    if action == "prepare":
        print(prepare.remote((data / "mambomix_train.json").read_bytes(),
                             (data / "dataset_info.json").read_bytes(),
                             (repo / "finetune" / "qwen2_audio_lora.yaml").read_bytes()))
        print("Now upload audio:  modal volume put mambo-data "
              "fixtures/synthetic/audio /audio")
    elif action == "train":
        print(train.remote())
    elif action == "evaluate":
        res = evaluate.remote((data / "mambomix_test.json").read_bytes())
        (repo / "finetune" / "eval_preds.json").write_text(json.dumps(res, indent=2))
        print(f"wrote {len(res)} predictions -> finetune/eval_preds.json")
