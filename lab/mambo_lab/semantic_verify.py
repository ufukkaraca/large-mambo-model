"""Semantic-verify pass — P1: rules-only spurious-melody suppression.

The first phase of the reasoning layer (design: PAPER §6.1g). NO
LLM. It drops a *phantom* melody segment — the short, trailing/between-word sustained
vowel that the acoustic router mislabels as a hum on a spoken command (the Rim
failure, §E-LIVE-4) — while preserving real demonstrations and bare hums.

It is a pure UIR→UIR refinement applied AFTER perception (studio `_process`),
BEFORE planning. The acoustic gates call the router directly and never see it, so
they are unchanged.

The rule (observed structure, `fixtures/human/voices/rim`):
  * a real demonstration's melody is preceded by a demonstrative cue ("…like that",
    "…go mmm", "can the bass do") and is several notes long;
  * a phantom is a SHORT melody (≤ PHANTOM_MAX_NOTES) that FOLLOWS speech which does
    NOT end with a demonstrative cue — i.e. a complete command with a trailing vowel.
So: drop a melody iff (it is short) ∧ (it has a preceding speech span) ∧ (that span
does not end with a demonstrative cue) ∧ (no contrast cue follows). Never touches a
hum-first / bare hum (no preceding speech) or a framed demonstration.
"""

from __future__ import annotations

from typing import Any

# demonstrative cues that BRACKET a hummed demonstration (checked at the tail of the
# preceding speech span); kept deliberately tight to avoid firing on generic commands.
_DEMO_CUES = ("like", "something", "goes", "go", "do", "sounds", "version", "kinda",
              "kind of", "how about", "what about", "give me")
# contrast cues that follow a demonstration ("…but slower", "…instead")
_CONTRAST_CUES = ("but", "instead", "rather", "not", "slower", "faster", "higher", "lower")
# a phantom is a 1-2 note fragment (a single sustained vowel the router over-split);
# a deliberate demonstration is longer. ≤2 fixes the severe case (Rim speech-no-hum
# 0.12→0.75) while leaving borderline 3-note spans alone (raise to 3 to also catch
# milder over-segmentation, at some cost to short real demos — see PAPER §6.1g).
PHANTOM_MAX_NOTES = 2


def _tail_has_cue(text: str, cues: tuple[str, ...], n_words: int = 4) -> bool:
    words = (text or "").lower().split()
    tail = " ".join(words[-n_words:])
    return any(c in tail for c in cues)


def _head_has_cue(text: str, cues: tuple[str, ...], n_words: int = 3) -> bool:
    words = (text or "").lower().split()
    head = " ".join(words[:n_words])
    return any(head.startswith(c) or f" {c}" in f" {head}" for c in cues)


def verify(uir: dict[str, Any], *, max_notes: int = PHANTOM_MAX_NOTES) -> dict[str, Any]:
    """P1: drop phantom melody segments. Returns the (possibly) refined UIR dict."""
    segs = uir.get("segments", [])
    if not segs:
        return uir
    drop = set()
    for i, s in enumerate(segs):
        if s.get("kind") != "melody" or len(s.get("notes", [])) > max_notes:
            continue
        prev = next((segs[j] for j in range(i - 1, -1, -1) if segs[j].get("kind") == "speech"), None)
        if prev is None:
            continue  # hum-first or bare hum → keep
        if _tail_has_cue(prev.get("text", ""), _DEMO_CUES):
            continue  # framed demonstration ("…like that") → keep
        nxt = next((segs[j] for j in range(i + 1, len(segs)) if segs[j].get("kind") == "speech"), None)
        if nxt is not None and _head_has_cue(nxt.get("text", ""), _CONTRAST_CUES):
            continue  # "…[hum] but slower / instead" → a demonstration → keep
        drop.add(i)  # short melody after a complete command, no frame → phantom
    if not drop:
        return uir
    out = dict(uir)
    out["segments"] = [s for i, s in enumerate(segs) if i not in drop]
    return out
