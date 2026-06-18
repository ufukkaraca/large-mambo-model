"""Build the LoRA training dataset (PAPER §5.3): (audio, GT UIR) pairs ->
LLaMA-Factory ShareGPT-audio format for Qwen2-Audio-7B-Instruct.

Task format mirrors the modular pipeline's output exactly (PAPER §5.3): audio in,
``mambo.utterance.v1`` JSON out — the same schema, the same evaluator. The
recorded/human slice is TEST-only and never trained on.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

INSTRUCTION = (
    "<audio>Parse this studio utterance into a mambo.utterance.v1 percept. "
    "Output ONLY the JSON: segments (speech spans with text; melody spans with "
    "notes, key, tempo, contour), and percussion[] if any. No prose."
)


def target_uir(uir: dict) -> str:
    """The assistant target: the UIR the model should emit (compact JSON).

    Drops ``session_context``/``history_refs`` (those are context the planner
    gets, not perception output) so the model learns audio -> percept only.
    """
    out = {k: v for k, v in uir.items() if k not in ("session_context", "history_refs")}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def to_example(audio_path: str, uir: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": target_uir(uir)},
        ],
        "audios": [audio_path],
    }


DATASET_INFO = {
    "mambomix": {
        "file_name": "mambomix_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "audios": "audios"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }
}


def build(fixtures_dir: str, out_dir: str, *, audio_prefix: str = "", test_frac: float = 0.1) -> dict:
    """Convert a fixtures dir (truth/*.uir.json + audio/) into the LF dataset.

    ``audio_prefix`` is prepended to each wav path (e.g. the Modal volume mount
    point at train time). Deterministic split (every 10th utterance -> test).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fx = Path(fixtures_dir)
    truth = sorted(glob.glob(str(fx / "truth" / "*.uir.json")))
    train, test = [], []
    for i, tp in enumerate(truth):
        uir = json.load(open(tp))
        uid = uir["utterance_id"]
        wav = f"{audio_prefix}{uid}.wav" if audio_prefix else str(fx / "audio" / f"{uid}.wav")
        ex = to_example(wav, uir)
        (test if i % int(1 / test_frac) == 0 else train).append(ex)

    (out / "mambomix_train.json").write_text(json.dumps(train, ensure_ascii=False))
    (out / "mambomix_test.json").write_text(json.dumps(test, ensure_ascii=False))
    (out / "dataset_info.json").write_text(json.dumps(DATASET_INFO, indent=2))
    stats = {"train": len(train), "test": len(test), "total": len(truth)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="../fixtures/synthetic")
    ap.add_argument("--out", default="../finetune/data")
    ap.add_argument("--audio-prefix", default="", help="prepended to wav paths (e.g. /data/audio/ on Modal)")
    args = ap.parse_args()
    stats = build(args.fixtures, args.out, audio_prefix=args.audio_prefix)
    print(f"dataset: {stats['train']} train + {stats['test']} test (held-out) -> {args.out}")


if __name__ == "__main__":
    main()
