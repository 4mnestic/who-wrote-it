"""Hypothesis tests on word-length distributions."""

from __future__ import annotations

from itertools import combinations

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


def hellinger(counts: np.ndarray) -> np.ndarray:
    """Row-normalize counts to proportions and take square roots, so Euclidean
    distance between rows is (proportional to) the Hellinger distance."""
    counts = np.asarray(counts, dtype=float)
    return np.sqrt(counts / counts.sum(axis=1, keepdims=True))


def pseudo_f(x: np.ndarray, labels: np.ndarray) -> float:
    """PERMANOVA pseudo-F: between-group over within-group sum of squares,
    each divided by its degrees of freedom."""
    groups = np.unique(labels)
    grand = x.mean(axis=0)
    ssb = ssw = 0.0
    for g in groups:
        xg = x[labels == g]
        ssb += len(xg) * np.sum((xg.mean(axis=0) - grand) ** 2)
        ssw += np.sum((xg - xg.mean(axis=0)) ** 2)
    n, k = len(x), len(groups)
    return float((ssb / (k - 1)) / (ssw / (n - k)))


def _relabelings(labels: np.ndarray):
    """Yield every assignment of the same group sizes to the items (exhaustive)."""
    groups, sizes = np.unique(labels, return_counts=True)

    def rec(remaining: tuple[int, ...], gi: int, out: np.ndarray):
        if gi == len(groups) - 1:
            out[list(remaining)] = groups[gi]
            yield out.copy()
            return
        for chosen in combinations(remaining, sizes[gi]):
            out[list(chosen)] = groups[gi]
            rest = tuple(i for i in remaining if i not in chosen)
            yield from rec(rest, gi + 1, out)

    yield from rec(tuple(range(len(labels))), 0, np.empty_like(labels))


def exact_permutation_test(x: np.ndarray, labels: np.ndarray) -> dict:
    """Exact permutation test of group differences with the *item* (here, a
    book) as the unit: compare the observed pseudo-F with its value under
    every possible relabeling that keeps group sizes fixed."""
    labels = np.asarray(labels)
    observed = pseudo_f(x, labels)
    null = np.array([pseudo_f(x, perm) for perm in _relabelings(labels)])
    return {
        "pseudo_f": observed,
        "p": float(np.mean(null >= observed - 1e-12)),
        "n_permutations": int(len(null)),
    }
