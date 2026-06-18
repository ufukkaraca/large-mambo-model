"""Studio app logic (no server / no audio)."""

from mambo_lab.studio import _maybe_lyric


def test_lyric_capture_strips_the_cue():
    assert _maybe_lyric("jot this down walking down the avenue") == "walking down the avenue"
    assert _maybe_lyric("lyric: she said goodbye") == "she said goodbye"
    assert _maybe_lyric("note that down chasing the sunrise") == "chasing the sunrise"
    assert _maybe_lyric("remember this line we were younger then") == "we were younger then"


def test_commands_are_not_lyrics():
    for cmd in ("make the bass louder", "mute the keys", "loop that", "pan the drums left"):
        assert _maybe_lyric(cmd) is None


def test_take_commands():
    from mambo_lab.studio import _maybe_take
    assert _maybe_take("keep that")["op"] == "keep"
    assert _maybe_take("that's a keeper")["op"] == "keep"
    assert _maybe_take("keep take 3") == {"op": "keep", "take": 3}
    assert _maybe_take("scratch that take")["op"] == "scratch"
    assert _maybe_take("go back to the keeper") == {"op": "recall", "ref": "keeper"}
    assert _maybe_take("play take 2") == {"op": "recall", "ref": 2}
    assert _maybe_take("make the bass louder") is None
