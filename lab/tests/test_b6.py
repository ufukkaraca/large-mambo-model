"""B6 harness offline logic (parsing + scoring); the API call is budget-gated."""

from mambo_lab.eval.b6_omni import parse_notes, pitch_seq_acc


def test_parse_notes():
    assert parse_notes("C4, E4, G4") == [60, 64, 67]
    assert parse_notes("C#4, Db4") == [61, 61]          # enharmonic, both 61
    assert parse_notes("the melody is A3 B3 C4.") == [57, 59, 60]
    assert parse_notes("I cannot determine the notes") == []


def test_pitch_seq_acc():
    assert pitch_seq_acc([60, 64, 67], [60, 64, 67]) == 1.0
    assert pitch_seq_acc([], [60, 64]) == 0.0
    # transposition-invariant: a contour shifted by +5 still scores 1.0 (generous, pro-thesis)
    assert pitch_seq_acc([65, 69, 72], [60, 64, 67]) == 1.0
    # partial recovery
    assert abs(pitch_seq_acc([60, 67], [60, 64, 67]) - 2 / 3) < 1e-9
    assert pitch_seq_acc([], []) == 1.0
