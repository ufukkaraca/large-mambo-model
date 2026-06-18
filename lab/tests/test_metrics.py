"""Unit tests for the eval metrics (segment matching, hallucination, WER)."""

from mambo_lab.eval import metrics


def _seg(kind, t0, t1, **kw):
    return {"kind": kind, "t0": t0, "t1": t1, **kw}


def test_segment_prf_perfect():
    ref = [_seg("speech", 0, 2), _seg("melody", 2.2, 6), _seg("speech", 6.2, 9)]
    p, r, f, errs = metrics.segment_prf(ref, [dict(s) for s in ref])
    assert (p, r, f) == (1.0, 1.0, 1.0)
    assert max(errs) == 0.0


def test_segment_prf_missed_melody():
    ref = [_seg("speech", 0, 2), _seg("melody", 2.2, 6)]
    est = [_seg("speech", 0, 2)]  # melody dropped
    p, r, f, _ = metrics.segment_prf(ref, est)
    assert r < 1.0 and f < 1.0


def test_ambiguous_matches_either():
    ref = [_seg("melody", 0, 4)]
    est = [_seg("ambiguous", 0, 4, text="la la", notes=[])]
    _, _, f, _ = metrics.segment_prf(ref, est)
    assert f == 1.0


def test_hallucination_detected_on_melody():
    ref = [_seg("melody", 2.0, 6.0)]
    leaked = [_seg("speech", 2.0, 6.0, text="the moon is rising")]
    clean = [_seg("melody", 2.0, 6.0, notes=[])]
    assert metrics.hallucination_on_melody(ref, leaked) is True
    assert metrics.hallucination_on_melody(ref, clean) is False


def test_wer():
    assert metrics.wer("kick the drums up", "kick the drums up") == 0.0
    assert metrics.wer("kick the drums up", "kick the drum up") > 0.0


def test_bootstrap_ci():
    mean, lo, hi = metrics.bootstrap_ci([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert abs(mean - 0.5) < 1e-9 and 0.0 <= lo < mean < hi <= 1.0
    m2, l2, h2 = metrics.bootstrap_ci([1, 1, 1, 1, 1, 1])  # all-success -> tight at 1
    assert m2 == 1.0 and l2 == 1.0 and h2 == 1.0
    assert metrics.bootstrap_ci([1, 0, 1, 0]) == metrics.bootstrap_ci([1, 0, 1, 0])  # deterministic
    assert metrics.bootstrap_ci([]) == (None, None, None)


def test_wilson_ci():
    p, lo, hi = metrics.wilson_ci(6, 6)  # perfect pass rate is NOT [1,1]
    assert p == 1.0 and hi > 0.99 and 0.5 < lo < 1.0
    p2, lo2, hi2 = metrics.wilson_ci(0, 6)  # 0/6
    assert p2 == 0.0 and lo2 < 0.01 and 0.0 < hi2 < 0.5
    p3, lo3, hi3 = metrics.wilson_ci(3, 6)
    assert p3 == 0.5 and lo3 < 0.5 < hi3
    assert metrics.wilson_ci(0, 0) == (None, None, None)
