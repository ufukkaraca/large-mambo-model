"""Unit tests for router helpers (pure functions over synthetic Frames — no
audio/ASR, so these stay fast)."""

import numpy as np

from mambo_lab import router
from mambo_lab.router import Frames, SpanProposal


def _frames(T=400, dt=0.005333):
    times = np.arange(T) * dt
    return Frames(times=times, dt=dt, voiced=np.zeros(T, bool),
                  local_voicing=np.zeros(T), f0_std=np.full(T, 99.0),
                  energy=np.zeros(T), word_lp=np.full(T, -np.inf), sound=np.zeros(T, bool))


def test_merge_adjacent_same_kind():
    spans = [SpanProposal(0, 1, "melody"), SpanProposal(1.1, 2, "melody"),
             SpanProposal(3, 4, "speech")]
    out = router._merge_adjacent(spans, gap=0.30)
    assert [s.kind for s in out] == ["melody", "speech"]
    assert out[0].t1 == 2


def test_merge_keeps_distinct_kinds():
    spans = [SpanProposal(0, 1, "speech"), SpanProposal(1.05, 2, "melody")]
    out = router._merge_adjacent(spans, gap=0.30)
    assert [s.kind for s in out] == ["speech", "melody"]


def test_label_joint_hum_region():
    f = _frames()
    # frames 100..300 are a sustained voiced stable-pitch hum, no words
    f.sound[50:350] = True
    f.local_voicing[100:300] = 0.95
    f.f0_std[100:300] = 0.3
    lab = router._label_joint(f)
    assert (lab[150:280] == router.MELODY).mean() > 0.9


def test_label_joint_speech_with_words_not_melody():
    f = _frames()
    f.sound[:] = True
    f.local_voicing[:] = 0.6           # speech-level voicing
    f.word_lp[:] = -0.2                # confident words throughout
    lab = router._label_joint(f)
    assert (lab == router.SPEECH).all()


def test_safety_net_speech():
    f = _frames()
    f.sound[100:300] = True
    f.local_voicing[100:300] = 0.6
    out = router._safety_net(f)
    assert len(out) == 1 and out[0].kind == "speech"


def test_reclassify_hum_to_melody():
    f = _frames()
    f.local_voicing[:] = 0.95
    f.f0_std[:] = 0.3
    spans = [SpanProposal(0.0, 4.0, "speech")]  # long, very voiced, stable -> hum
    out = router._reclassify_hum_spans(spans, f)
    assert out[0].kind == "melody"
