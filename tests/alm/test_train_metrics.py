"""TDD Phase 2: CER/edit-distance metric is correct (host-runnable, no GPU)."""
from pranava.alm.train import _levenshtein


def test_identical_is_zero():
    assert _levenshtein([1, 2, 3], [1, 2, 3]) == 0


def test_all_different():
    assert _levenshtein([1, 2, 3], [4, 5, 6]) == 3


def test_insertions_deletions():
    assert _levenshtein([1, 2], [1, 2, 3, 4]) == 2
    assert _levenshtein([1, 2, 3, 4], [1, 4]) == 2


def test_empty():
    assert _levenshtein([], [1, 2, 3]) == 3
    assert _levenshtein([1, 2, 3], []) == 3
    assert _levenshtein([], []) == 0
