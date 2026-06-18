"""Unit tests for the dual-decode reasoning promotion + lyric scoring (no audio,
no network — the audio/LLM path is exercised by eval/dual_decode_eval.py on real
sung clips)."""

from mambo_lab import dual_decode as dd
from mambo_lab import ir


def test_word_recall():
    assert dd.word_recall("we were younger then", "we were younger then") == 1.0
    assert dd.word_recall("we were older", "we were younger") == round(2 / 3, 3)
    assert dd.word_recall("anything", "") == 1.0          # empty GT -> trivially satisfied
    assert dd.word_recall("", "two words here") == 0.0


def test_wer():
    assert dd.wer("we were younger", "we were younger") == 0.0
    assert dd.wer("we are younger", "we were younger") == round(1 / 3, 3)  # one substitution
    assert dd.wer("anything", "") == 0.0


def test_rules_sung_lyric_distinguishes_lyric_from_babble():
    assert dd.rules_sung_lyric("we were younger then", 5) is True
    assert dd.rules_sung_lyric("hold on to me now", 4) is True
    # babble / onomatopoeia / fillers -> not a lyric (containment must hold)
    assert dd.rules_sung_lyric("da da da da", 4) is False
    assert dd.rules_sung_lyric("do re mi fa", 4) is False
    assert dd.rules_sung_lyric("mm-hmm mm-hmm", 2) is False
    assert dd.rules_sung_lyric("oh", 3) is False
    assert dd.rules_sung_lyric("na na na na", 4) is False


def _melody_uir():
    return {
        "schema": "mambo.utterance.v1",
        "utterance_id": "t",
        "audio": {"sample_rate": 16000, "duration_s": 2.0},
        "segments": [{
            "kind": "melody", "t0": 0.0, "t1": 2.0, "confidence": 0.9, "role": "exemplar",
            "notes": [{"midi": 60, "name": "C4", "t0": 0.0, "dur": 0.4, "vel": 90},
                      {"midi": 62, "name": "D4", "t0": 0.4, "dur": 0.4, "vel": 90}],
            "analysis": {"n_notes": 2},
        }],
    }


def test_promote_sung_lyric_to_ambiguous():
    uir = _melody_uir()
    reason = lambda text, n: "younger" in text          # deterministic stub
    out = dd.promote(uir, {0: "we were younger then"}, reason_fn=reason)
    seg = out["segments"][0]
    assert seg["kind"] == "ambiguous"
    assert seg["text"] == "we were younger then"
    assert seg["notes"]                                  # notes preserved
    ir.validate(out)                                     # ambiguous(text+notes) is a valid UIR


def test_promote_keeps_babble_as_melody():
    uir = _melody_uir()
    reason = lambda text, n: dd.rules_sung_lyric(text, n)
    out = dd.promote(uir, {0: "da da da da"}, reason_fn=reason)
    seg = out["segments"][0]
    assert seg["kind"] == "melody"                       # NOT promoted
    assert not seg.get("text")                            # containment holds — no lyric leak
    ir.validate(out)


def test_promote_noop_without_candidate_text():
    uir = _melody_uir()
    out = dd.promote(uir, {}, reason_fn=lambda t, n: True)
    assert out["segments"][0]["kind"] == "melody"        # no candidate -> untouched
