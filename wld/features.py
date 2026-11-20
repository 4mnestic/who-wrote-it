"""Two ways to turn a passage into numbers: word lengths and function words."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .corpus import MAX_LEN, Book

# The 70 function words used by Mosteller & Wallace (1964) for the Federalist
# Papers. They are topic-neutral: a book about war and one about marriage both
# use "the", "upon" and "would" at rates set by the writer's habits.
FUNCTION_WORDS = (
    "a all also an and any are as at be been but by can do down even every for "
    "from had has have her his if in into is it its may more must my no not now "
    "of on one only or our shall should so some such than that the their then "
    "there things this to up upon was were what when which who will with would "
    "your"
).split()
_FW_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


@dataclass(frozen=True)
class FeatureSet:
    """Maps each word to a column (or -1 to ignore it); a passage's features
    are the share of its words that fall in each column."""

    name: str
    n_features: int
    encode: Callable[[np.ndarray], np.ndarray]


def _encode_length(words: np.ndarray) -> np.ndarray:
    lengths = Book("", "", "", words).lengths
    return np.minimum(lengths, MAX_LEN) - 1


def _encode_function_words(words: np.ndarray) -> np.ndarray:
    return np.array([_FW_INDEX.get(w, -1) for w in words], dtype=np.int64)


WORD_LENGTH = FeatureSet("Word length", MAX_LEN, _encode_length)
FUNCTION_WORD = FeatureSet("Function words", len(FUNCTION_WORDS), _encode_function_words)


def rates(codes: np.ndarray, n_features: int) -> np.ndarray:
    """Share of all words in each feature column (one passage)."""
    if len(codes) == 0:
        return np.zeros(n_features)
    kept = codes[codes >= 0]
    return np.bincount(kept, minlength=n_features) / len(codes)


def chunk_rates(codes: np.ndarray, n_features: int, chunk_size: int) -> np.ndarray:
    """Split encoded words into consecutive chunks of `chunk_size` (dropping the
    remainder) and return one row of feature rates per chunk."""
    n = len(codes) // chunk_size
    chunks = codes[: n * chunk_size].reshape(n, chunk_size)
    rows = np.repeat(np.arange(n), chunk_size)
    flat = chunks.ravel()
    keep = flat >= 0
    counts = np.bincount(
        rows[keep] * n_features + flat[keep], minlength=n * n_features
    ).reshape(n, n_features)
    return counts / chunk_size
