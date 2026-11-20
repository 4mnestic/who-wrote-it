import numpy as np
import pytest

from wld.corpus import Book


@pytest.fixture
def synthetic_books():
    """Six books, two per author, where each author has a clearly different
    word-length distribution (so a working classifier should be near-perfect)."""
    rng = np.random.default_rng(0)
    profiles = {
        "A": [0.1, 0.5, 0.2, 0.1, 0.1],
        "B": [0.3, 0.1, 0.1, 0.4, 0.1],
        "C": [0.1, 0.1, 0.5, 0.1, 0.2],
    }
    return [
        Book(f"{a}{i}", a, "", rng.choice([1, 2, 3, 5, 14], size=3000, p=p))
        for a, p in profiles.items()
        for i in range(2)
    ]
