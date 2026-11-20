"""Can a passage's word-length distribution identify its author?"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .corpus import Book, bin_counts


def chunk_features(lengths: np.ndarray, chunk_size: int) -> np.ndarray:
    """Split a book into consecutive non-overlapping chunks of `chunk_size`
    words (the remainder is dropped); return each chunk's bin proportions."""
    n = len(lengths) // chunk_size
    if n == 0:
        return np.empty((0, len(bin_counts(lengths[:0]))))
    rows = [bin_counts(c) for c in lengths[: n * chunk_size].reshape(n, chunk_size)]
    return np.array(rows, dtype=float) / chunk_size


def leave_one_book_out(books: list[Book], chunk_size: int) -> dict:
    """Train on chunks from all books but one, predict the held-out book's
    chunks, repeat for every book. Holding out whole books keeps chunks of
    the test book out of training, so the model must generalize to new books
    rather than recognize a book it has already seen."""
    feats = [chunk_features(b.lengths, chunk_size) for b in books]
    y_true, y_pred, per_book = [], [], []
    for i, book in enumerate(books):
        x_train = np.vstack([f for j, f in enumerate(feats) if j != i])
        y_train = np.concatenate(
            [[b.author] * len(f) for j, (b, f) in enumerate(zip(books, feats)) if j != i]
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ).fit(x_train, y_train)
        pred = model.predict(feats[i])
        y_true += [book.author] * len(pred)
        y_pred += list(pred)
        values, counts = np.unique(pred, return_counts=True)
        per_book.append(
            {
                "title": book.title,
                "author": book.author,
                "chunks": len(pred),
                "chunk_accuracy": float(np.mean(pred == book.author)),
                "majority_vote": str(values[np.argmax(counts)]),
            }
        )
    return {
        "chunk_size": chunk_size,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "chance": 1 / len({b.author for b in books}),
        "books_correct_by_vote": sum(r["majority_vote"] == r["author"] for r in per_book),
        "per_book": per_book,
        "y_true": y_true,
        "y_pred": y_pred,
    }
