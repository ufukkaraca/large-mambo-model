"""B6 — does a current omni audio-LLM actually hear pitch on Mambo's OWN task?

The decisive on-task test PAPER.md §3.1 needs (the thesis otherwise rests on
borrowed benchmarks + one omni outlier). Prompt a free audio-input omni model
(OpenRouter) to transcribe a pure hum into notes, and compare its pitch sequence
to the same ground truth the modular pipeline (`melody.segment_notes`) is scored on.

Budget note (2026-06-16): a free audio model exists
(nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free) but OpenRouter gates audio
behind a >=$0.50 balance and our key is capped at $0.10 -> HTTP 402. This harness
is READY; it runs the moment the cap is raised (see the repository). The
parsing/scoring is unit-tested offline so only the API call is gated.

    cd lab && uv run python -m mambo_lab.eval.b6_omni        # uses OPENROUTER_API_KEY from .env
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from . import gate, metrics
from .. import melody

REPO = Path(__file__).resolve().parents[3]

# B6 target is configurable so the SAME harness (+ ingestion guard) runs the free
# omni model OR a FRONTIER one, via any OpenAI-compatible audio endpoint:
#   free (default):  this model on OpenRouter
#   frontier (paid): MAMBO_B6_MODEL=openai/gpt-4o-audio-preview            (via OpenRouter)
#                    MAMBO_B6_MODEL=google/gemini-2.5-flash                (via OpenRouter)
#   direct OpenAI:   MAMBO_B6_MODEL=gpt-4o-audio-preview \
#                    MAMBO_B6_BASE_URL=https://api.openai.com/v1 MAMBO_B6_API_KEY=sk-…
# The guard (audio_tokens / pitch-tracking r / no-audio control) validates ingestion
# before any number is trusted, so a frontier run can't masquerade either.
OMNI_MODEL = os.environ.get("MAMBO_B6_MODEL") or "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
B6_BASE_URL = os.environ.get("MAMBO_B6_BASE_URL") or "https://openrouter.ai/api/v1"

_NOTE_RE = re.compile(r"\b([A-Ga-g])([#b♯♭]?)(-?\d)\b")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_notes(text: str) -> list[int]:
    """Pull a MIDI sequence out of a model's free-text reply (C4, F#3, Bb5, …)."""
    out = []
    for m in _NOTE_RE.finditer(text or ""):
        pc = _PC[m.group(1).upper()]
        if m.group(2) in ("#", "♯"):
            pc += 1
        elif m.group(2) in ("b", "♭"):
            pc -= 1
        out.append(12 * (int(m.group(3)) + 1) + pc)
    return out


def _lcs(pred: list[int], gt: list[int], *, transpose_invariant: bool = True) -> int:
    """Longest common subsequence length (transposition-invariant: shift pred so its
    first note matches gt's first, so a model that hears contour-but-not-absolute
    still scores)."""
    if not pred or not gt:
        return 0
    pr = [p + (gt[0] - pred[0]) for p in pred] if transpose_invariant else pred
    m, n = len(pr), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pr[i - 1] == gt[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def pitch_seq_acc(pred: list[int], gt: list[int], *, transpose_invariant: bool = True) -> float:
    """Transposition-invariant LCS RECALL = fraction of the GT sequence recovered.
    NOTE: recall alone conflates 'mis-pitched' with 'emitted too few notes' (an
    under-generating model is capped at n_pred/n_gt regardless of pitch accuracy);
    use `pitch_prf` to separate the two failure modes — see §6.1(c)."""
    if not gt:
        return 1.0 if not pred else 0.0
    return _lcs(pred, gt, transpose_invariant=transpose_invariant) / len(gt)


def pitch_prf(pred: list[int], gt: list[int], *, transpose_invariant: bool = True) -> dict:
    """Honest decomposition of a model's hum→notes transcription, so we can say
    *which* way it fails: (a) `n_pred` vs `n_gt` + `note_count_err` exposes
    under-/over-generation; (b) `precision` (LCS / n_pred) is how many of the notes
    it DID emit were right (transposition-invariant); (c) `recall` (LCS / n_gt) is
    how much of the GT it got; (d) `f1` balances them. A model that emits 3 correct
    notes for a 7-note hum scores precision 1.0, recall 0.43 — *accurate but
    under-generating*, not *pitch-deaf*."""
    np_, ng = len(pred), len(gt)
    lcs = _lcs(pred, gt, transpose_invariant=transpose_invariant)
    precision = lcs / np_ if np_ else (1.0 if ng == 0 else 0.0)
    recall = lcs / ng if ng else (1.0 if np_ == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"n_pred": np_, "n_gt": ng, "note_count_err": abs(np_ - ng),
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def _key() -> str | None:
    # explicit B6 key (for a frontier/direct provider) wins, else the OpenRouter key
    if os.environ.get("MAMBO_B6_API_KEY"):
        return os.environ["MAMBO_B6_API_KEY"]
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = REPO / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if ln.startswith("OPENROUTER_API_KEY="):
                return ln.split("=", 1)[1].strip()
    return None


_PROMPT = ("This audio is a person humming a single monophonic melody. List the musical "
           "notes you hear, in order, with octaves (e.g. C4, E4, G4). End your reply with "
           "a line 'NOTES: <comma-separated list>'.")


def transcribe_omni(wav_path: Path | None, *, model: str = OMNI_MODEL, key: str) -> dict:
    """One omni call. Returns {text, audio_tokens, prompt_tokens}. `wav_path=None`
    sends the prompt with NO audio — the control that exposes confabulation.

    Two robustness fixes learned on the free nvidia/nemotron omni route (2026-06-16):
      * it is a *reasoning* model — the answer lands in `message.reasoning`, and
        `message.content` is often null; we read both.
      * 300 tokens is all-reasoning-no-answer; we give it room (900)."""
    content: list[dict] = [{"type": "text", "text": _PROMPT}]
    if wav_path is not None:
        b64 = base64.b64encode(Path(wav_path).read_bytes()).decode()
        content.append({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}})
    body = {"model": model, "temperature": 0, "max_tokens": 900,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(
        B6_BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://mambo.local", "X-Title": "Mambo B6"})
    r = json.load(urllib.request.urlopen(req, timeout=200))
    m = r["choices"][0]["message"]
    u = r.get("usage", {}) or {}
    text = (m.get("content") or "") + " " + (m.get("reasoning") or "")
    return {"text": text.strip(),
            "audio_tokens": (u.get("prompt_tokens_details") or {}).get("audio_tokens"),
            "prompt_tokens": u.get("prompt_tokens")}


def _verdict(per: list[dict], control_notes: list[int], audio_toks: list) -> dict:
    """Decide whether this run is a VALID measurement of the model's pitch hearing,
    before quoting any accuracy. A negative omni score only means "can't hear pitch"
    if we can show the audio was actually ingested — otherwise we are scoring a
    broken pipe. Two guards: (1) does the predicted pitch track the GT pitch across
    clips (Pearson r, with n)?  (2) does the model behave differently with audio vs
    the no-audio control?  Plus the provider's own audio-token meter."""
    paired = [(p["gt_median"], p["omni_median"]) for p in per if p.get("omni_median") is not None]
    r = None
    if len(paired) >= 3:
        g, pr = zip(*paired)
        if np.std(g) > 0 and np.std(pr) > 0:
            r = round(float(np.corrcoef(g, pr)[0, 1]), 2)
    audio_ingested = any(bool(a) for a in audio_toks)  # any non-zero audio-token count
    n = len(paired)
    # significance floor for r at small n (|r| s.t. p<0.05, two-sided): n=4→.95 n=5→.88 n=6→.81 n=8→.71 n=12→.58
    crit = {3: 0.997, 4: 0.95, 5: 0.878, 6: 0.811, 7: 0.754, 8: 0.707, 10: 0.632, 12: 0.576}
    sig = r is not None and abs(r) >= crit.get(n, 0.6)
    valid = audio_ingested or sig
    return {"audio_tokens_seen": audio_ingested, "pitch_corr_r": r, "pitch_corr_n": n,
            "pitch_corr_significant": bool(sig), "control_emits_default_scale": control_notes[:8],
            "valid_measurement": bool(valid),
            "note": ("VALID: audio demonstrably ingested / pitch tracks GT — omni accuracy is interpretable."
                     if valid else
                     "INCONCLUSIVE: free route shows audio_tokens=0, pitch does not significantly track GT, "
                     "and the no-audio control confabulates the same kind of answer. Cannot distinguish "
                     "'model is pitch-deaf' from 'audio not decoded on this route'. A valid B6 needs a "
                     "confirmed-audio route (base Qwen2-Audio via the R3 Modal app, or paid Gemini/GPT-4o-audio).")}


def run(limit: int = 12) -> dict:
    key = _key()
    rows = gate.load_fixtures()  # needs `make fixtures` (synthetic gives exact pitch GT)
    for r in rows:
        r.setdefault("uid", r.get("utterance_id"))
    clips = [r for r in rows if r["snr"] == "clean" and "pure_hum" in str(r["uid"])][:limit]
    per, omni_acc, mambo_acc, audio_toks, blocked = [], [], [], [], None
    for r in clips:
        d = gate._truth(r["uid"])
        gt = [n["midi"] for s in d["segments"] if s["kind"] == "melody" for n in s.get("notes", [])]
        a, sr = sf.read(str(gate.FIXTURES / r["wav"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        mambo = [n.midi for n in melody.segment_notes(a, melody.track_f0(a, sr))]
        m_acc = pitch_seq_acc(mambo, gt)
        mambo_acc.append(m_acc)
        row = {"uid": r["uid"], "gt_n": len(gt), "gt_median": float(np.median(gt)) if gt else None,
               "mambo_n": len(mambo), "mambo_pitch_acc": round(m_acc, 3)}
        try:
            resp = transcribe_omni(gate.FIXTURES / r["wav"], key=key)
            onotes = parse_notes(resp["text"])
            o_acc = pitch_seq_acc(onotes, gt)
            omni_acc.append(o_acc)
            audio_toks.append(resp["audio_tokens"])
            row.update(omni_n=len(onotes), omni_pitch_acc=round(o_acc, 3),
                       omni_median=float(np.median(onotes)) if onotes else None,
                       audio_tokens=resp["audio_tokens"], omni_raw=resp["text"][:140])
        except urllib.error.HTTPError as e:
            blocked = f"HTTP {e.code}: {e.read().decode()[:120]}"
            row["omni"] = blocked
        per.append(row)
    # no-audio control: the same prompt with no audio exposes pure confabulation
    control_notes = []
    if not blocked:
        try:
            control_notes = parse_notes(transcribe_omni(None, key=key)["text"])
        except urllib.error.HTTPError:
            pass
    verdict = _verdict(per, control_notes, audio_toks)
    out = {"model": OMNI_MODEL, "n_clips": len(clips), "per_clip": per,
           "mambo_pitch_acc_mean": round(float(np.mean(mambo_acc)), 3) if mambo_acc else None,
           "omni_pitch_acc_mean": round(float(np.mean(omni_acc)), 3) if omni_acc else None,
           "no_audio_control_notes": control_notes, "ingestion_verdict": verdict, "blocked": blocked}
    return out


def main() -> int:
    r = run()
    print(f"\n=== B6 · omni ({OMNI_MODEL}) vs Mambo · pitch-sequence accuracy on {r['n_clips']} pure hums ===")
    if r["blocked"] and r["omni_pitch_acc_mean"] is None:
        print(f"\n  OMNI BLOCKED: {r['blocked']}")
        print("  (raise the OpenRouter key cap to >= $0.50 — see the repository, then re-run)\n")
    print(f"  Mambo pitch-seq accuracy: {r['mambo_pitch_acc_mean']}")
    print(f"  Omni  pitch-seq accuracy: {r['omni_pitch_acc_mean']}")
    v = r["ingestion_verdict"]
    print(f"\n  audio-ingestion check → audio_tokens seen: {v['audio_tokens_seen']} · "
          f"pitch-tracking r={v['pitch_corr_r']} (n={v['pitch_corr_n']}, sig={v['pitch_corr_significant']}) · "
          f"no-audio control notes: {r['no_audio_control_notes'][:8]}")
    print(f"  {'✅ VALID measurement' if v['valid_measurement'] else '⚠️  INCONCLUSIVE'} — {v['note']}")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "runs" / f"{now}-b6"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(r, indent=2) + "\n")
    print(f"\n  wrote {run_dir.relative_to(REPO)}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
