import numpy as np
import pytest

from wld.stats import (
    _relabelings,
    exact_permutation_test,
    hellinger,
    homogeneity_test,
    pseudo_f,
)


def test_homogeneity_identical_rows():
    """Groups with identical proportions give chi2 = 0, p = 1 and no effect."""
    r = homogeneity_test(np.array([[10, 20, 30], [20, 40, 60]]))
    assert r["chi2"] == pytest.approx(0)
    assert r["p"] == pytest.approx(1)
    assert r["cramers_v"] == pytest.approx(0)
    assert r["dof"] == 2 and r["n"] == 180


def test_homogeneity_perfect_separation():
    """Groups with no categories in common have Cramér's V = 1."""
    r = homogeneity_test(np.array([[50, 0], [0, 50]]))
    assert r["cramers_v"] == pytest.approx(1)
    assert r["min_expected"] == pytest.approx(25)


def test_homogeneity_hand_computed():
    """Matches a hand-computed 2x2 chi-square (no continuity correction)."""
    # Expected = 15 in every cell; chi2 = 4 * (5**2 / 15).
    r = homogeneity_test(np.array([[20, 10], [10, 20]]))
    assert r["chi2"] == pytest.approx(100 / 15)


def test_homogeneity_ignores_empty_bins():
    """A bin that no group uses is dropped instead of crashing the test."""
    r = homogeneity_test(np.array([[20, 0, 10], [10, 0, 20]]))
    assert r["chi2"] == pytest.approx(100 / 15)
    assert r["dof"] == 1


def test_hellinger_rows_are_unit_vectors():
    """Square-rooted proportions have unit Euclidean norm."""
    h = hellinger(np.array([[1, 3], [5, 5]]))
    assert np.linalg.norm(h, axis=1) == pytest.approx([1, 1])


def test_pseudo_f_hand_computed():
    """Two groups {0,2} and {10,12}: SSB = 100, SSW = 4, F = (100/1)/(4/2) = 50."""
    x = np.array([[0.0], [2.0], [10.0], [12.0]])
    assert pseudo_f(x, np.array(["a", "a", "b", "b"])) == pytest.approx(50)


def test_relabelings_enumerate_all_partitions():
    """2+2+2 items have 6!/(2!2!2!) = 90 labelings, all distinct, sizes preserved."""
    labels = np.array(list("aabbcc"))
    perms = [tuple(p) for p in _relabelings(labels)]
    assert len(perms) == len(set(perms)) == 90
    assert all(sorted(p) == sorted(labels) for p in perms)


def test_exact_test_detects_clear_groups():
    """Well-separated groups get the smallest attainable p (observed partition
    and its 2 label-swaps out of 6!/(3!3!) = 20 labelings)."""
    x = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    r = exact_permutation_test(x, np.array(list("aaabbb")))
    assert r["n_permutations"] == 20
    assert r["p"] == pytest.approx(2 / 20)


def test_exact_test_null_when_groups_interleaved():
    """When labels are unrelated to the values, p is large."""
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    r = exact_permutation_test(x, np.array(list("ababab")))
    assert r["p"] > 0.5
