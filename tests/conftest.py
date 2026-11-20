import numpy as np
import pytest

from wld.corpus import Book

# Small vocabulary mixing function words and content words of varied lengths.
VOCAB = np.array(["the", "upon", "whilst", "and", "of", "cat", "elephant",
                  "a", "remarkable", "to"])


def make_books(profiles: dict[str, list[float]], per_author: int, n_words: int, seed=0):
    """Books whose words are drawn from an author-specific distribution over VOCAB."""
    rng = np.random.default_rng(seed)
    return [Book(f"{a}{i}", a, "T" + a, rng.choice(VOCAB, size=n_words, p=p))
            for a, p in profiles.items() for i in range(per_author)]


@pytest.fixture
def synthetic_books():
    """Six books, two per author, with clearly different word habits, so a
    working classifier should be near-perfect with either feature set."""
    return make_books({
        "A": [0.3, 0.3, 0, 0.1, 0.1, 0.1, 0, 0.1, 0, 0],
        "B": [0.1, 0, 0.3, 0.1, 0.1, 0, 0.3, 0, 0.1, 0],
        "C": [0.1, 0, 0, 0.1, 0.3, 0, 0, 0.1, 0.1, 0.3],
    }, per_author=2, n_words=3000)
