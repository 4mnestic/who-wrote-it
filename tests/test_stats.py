import numpy as np
import pytest

from wld.stats import homogeneity_test


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
