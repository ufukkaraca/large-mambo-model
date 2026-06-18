"""LoRA fine-tune track (PAPER §5.3): audio -> mambo.utterance.v1 JSON.

The optional Large Mambo Model — a LoRA on Qwen2-Audio-7B-Instruct that emits the
UIR end-to-end. Ships ONLY if it beats the modular pipeline on the held-out test
set (the modular pipeline is the teacher and the baseline). Trains on a rented
GPU (Modal); inference + the gate run here.
"""
