"""Semantic-verify P1 — phantom-melody suppression rule (no audio)."""

from mambo_lab.semantic_verify import verify


def _seg(kind, t0, t1, text="", n=0):
    s = {"kind": kind, "t0": t0, "t1": t1, "confidence": 1.0}
    if kind == "speech":
        s["text"] = text
    if kind in ("melody", "ambiguous"):
        s["notes"] = [{"midi": 60, "t0": t0, "dur": 0.2, "vel": 90} for _ in range(n)]
    return s


def _uir(*segs):
    return {"segments": list(segs)}


def _has_melody(u):
    return any(s["kind"] == "melody" for s in u["segments"])


def test_drops_phantom_after_command():
    # the Rim case: short melody trailing a complete command, no demonstrative cue
    u = _uir(_seg("speech", 0, 1.4, "mute the keys"), _seg("melody", 1.4, 1.9, n=1))
    assert not _has_melody(verify(u))


def test_drops_phantom_even_with_empty_asr():
    # Rim cmd_solo: ASR got nothing, but the trailing short melody is still a phantom
    u = _uir(_seg("speech", 0, 1.5, ""), _seg("melody", 1.5, 2.1, n=1))
    assert not _has_melody(verify(u))


def test_keeps_framed_demonstration():
    u = _uir(_seg("speech", 0, 1.7, "give me something like that"), _seg("melody", 2.2, 3.4, n=5))
    assert _has_melody(verify(u))


def test_keeps_hum_first():
    u = _uir(_seg("melody", 0, 1.2, n=2), _seg("speech", 1.3, 2.0, "that but on strings"))
    assert _has_melody(verify(u))


def test_keeps_bare_hum():
    u = _uir(_seg("melody", 0, 2.0, n=2))
    assert _has_melody(verify(u))


def test_keeps_long_melody_after_command():
    # >3 notes after unframed speech → conservative keep (could be a real unframed hum)
    u = _uir(_seg("speech", 0, 1.0, "play the thing"), _seg("melody", 1.0, 3.0, n=5))
    assert _has_melody(verify(u))


def test_keeps_when_contrast_follows():
    # "can the bass ♪…♪ but slower" — short melody saved by the trailing contrast cue
    u = _uir(_seg("speech", 0, 1.0, "can the bass"), _seg("melody", 1.2, 2.4, n=2),
             _seg("speech", 2.5, 3.2, "but slower"))
    assert _has_melody(verify(u))


def test_no_segments_is_safe():
    assert verify({"segments": []}) == {"segments": []}
