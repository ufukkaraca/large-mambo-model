"""Dual-decode: reasoning promotes a hummed span to BOTH melody + captured lyric.

The §3.5 reasoning-novelty, made concrete (D23 P2). The containment rule
(`ir.validate`) gives 0% lyric-hallucination by *dropping* every ASR word on a
melody span — correct for a wordless hum, but it also throws away a real **sung**
lyric ("give me something like ♪we were younger♪"). Acoustic/keyword routing cannot
recover that: it routes a span to melody OR speech, never both, and has no way to
tell sung words from babble.

Dual-decode is the *reasoned exception*: for a melody span that carried suppressed
ASR words, a reasoning step judges whether those words are a genuine sung lyric. If
yes, the span is promoted to `kind="ambiguous"` carrying BOTH the notes (tracker)
and the lyric (ASR) — the one representation acoustic routing cannot express. If no
(babble on a wordless hum), it stays melody-only and containment holds. So the
SAME reasoning decision yields 0% hallucination on hums AND lyric capture on sung
demonstrations.

This module is pure given a `reason_fn(text, n_notes) -> bool`; `judge_sung_lyric`
is the LLM implementation, `rules_sung_lyric` a no-network fallback. The eval that
runs it on real sung clips is `eval/dual_decode_eval.py`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import Counter
from typing import Any, Callable

MODEL = "gpt-5-mini"

_PROMPT = (
    "You are the reasoning layer of a studio voice assistant. A speech recognizer ran "
    "over a HUMMED/SUNG span and produced the transcript below. Decide whether the "
    "speaker SANG REAL WORDS (a lyric the producer is demonstrating) or whether the "
    "words are spurious ASR output over a WORDLESS hum (babble like 'da da da', "
    "'mm-hmm', 'do re mi', single fillers like 'oh'/'you', or pure repetition).\n"
    'Reply with ONLY a JSON object: {"sung_lyric": true|false}.'
)


def rules_sung_lyric(text: str, n_notes: int = 0) -> bool:
    """No-network fallback: a real lyric has lexical diversity and isn't pure
    onomatopoeia. Conservative — when unsure, FALSE (keep containment)."""
    toks = re.findall(r"[a-z']+", (text or "").lower())
    if len(toks) < 2:
        return False
    onomatopoeia = {"da", "la", "na", "ta", "duh", "doo", "do", "re", "mi", "fa", "so",
                    "la", "si", "ti", "mm", "hmm", "mmm", "ooh", "oh", "ah", "dada", "nana"}
    content = [t for t in toks if t not in onomatopoeia]
    if len(content) < 2:
        return False
    uniq = len(set(toks))
    if uniq / len(toks) < 0.5:  # heavy repetition
        return False
    return True


def judge_sung_lyric(text: str, n_notes: int = 0, *, key: str | None = None) -> bool:
    """LLM reasoning decision. Falls back to rules if no API key is available."""
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return rules_sung_lyric(text, n_notes)
    body = {"model": MODEL, "max_completion_tokens": 2000, "messages": [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": f"transcript: {text!r}\nnotes detected: {n_notes}"}]}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=60))
        msg = r["choices"][0]["message"]["content"] or ""
        m = re.search(r'"sung_lyric"\s*:\s*(true|false)', msg)
        return (m.group(1) == "true") if m else rules_sung_lyric(text, n_notes)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError):
        return rules_sung_lyric(text, n_notes)


def candidate_from_probe(pr: Any, t0: float, t1: float) -> str:
    """Candidate lyric for a span = words from the WHOLE-clip ASR (`pr.words`,
    with timing) that fall in [t0, t1]. This beats re-ASR'ing the isolated melody
    slice, which loses words/context — the slice method measured ~0.42 end-to-end
    vs ~0.83 here on the same clips (the dual-decode's accuracy is bottlenecked by
    how the candidate text is obtained, not by the reasoning judge)."""
    words = [w for w in getattr(pr, "words", []) if w.t1 > t0 and w.t0 < t1]
    return " ".join(w.w for w in words).strip()


def promote(uir: dict[str, Any], span_text: dict[int, str], *,
            reason_fn: Callable[[str, int], bool] = judge_sung_lyric) -> dict[str, Any]:
    """Promote each melody segment whose suppressed ASR text (`span_text[i]`) the
    reasoning judges a real sung lyric to `kind="ambiguous"` (notes + lyric). Pure
    given `reason_fn`. Other segments are untouched, so containment + gates hold."""
    segs = uir.get("segments", [])
    out_segs = []
    for i, s in enumerate(segs):
        cand = (span_text.get(i) or "").strip()
        if s.get("kind") == "melody" and s.get("notes") and cand and reason_fn(cand, len(s["notes"])):
            promoted = dict(s)
            promoted["kind"] = "ambiguous"
            promoted["text"] = cand
            out_segs.append(promoted)
        else:
            out_segs.append(s)
    out = dict(uir)
    out["segments"] = out_segs
    return out


# ----- scoring (pure) ----------------------------------------------------------

_WORD = re.compile(r"[a-z']+")


def _norm(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def word_recall(pred: str, gt: str) -> float:
    """Fraction of ground-truth lyric words recovered (multiset recall — order-free,
    robust to ASR word-splitting). 1.0 if gt empty."""
    g = Counter(_norm(gt))
    if not g:
        return 1.0
    p = Counter(_norm(pred))
    hit = sum(min(p[w], g[w]) for w in g)
    return round(hit / sum(g.values()), 3)


def wer(pred: str, gt: str) -> float:
    """Word error rate (Levenshtein over word tokens). 0.0 if gt empty."""
    a, b = _norm(gt), _norm(pred)
    if not a:
        return 0.0
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return round(d[len(b)] / len(a), 3)
