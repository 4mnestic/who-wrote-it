"""Chi-square test with an effect size."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency


def homogeneity_test(table: np.ndarray) -> dict:
    """Chi-square test that every row of a groups x bins count table comes from
    the same distribution. Returns chi2, dof, p, n, Cramér's V and the smallest
    expected cell count (the test assumes it is at least ~5)."""
    table = np.asarray(table, dtype=float)
    table = table[:, table.sum(axis=0) > 0]  # a bin no group uses carries no information
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    n = table.sum()
    k = min(table.shape) - 1
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p": float(p),
        "n": int(n),
        "cramers_v": float(np.sqrt(chi2 / (n * k))),
        "min_expected": float(expected.min()),
    }
