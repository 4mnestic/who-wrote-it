"""Author classifiers, always evaluated on texts the model never saw."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .corpus import Book
from .features import FeatureSet, chunk_rates, rates


def make_model():
    """Standardized multinomial logistic regression, classes weighted equally."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
    )


def _chunks(books: list[Book], fs: FeatureSet, chunk_size: int) -> list[np.ndarray]:
    return [chunk_rates(fs.encode(b.words), fs.n_features, chunk_size) for b in books]


def leave_one_book_out(books: list[Book], chunk_size: int, fs: FeatureSet) -> dict:
    """Train on chunks from every book but one, predict the held-out book's
    chunks, and repeat for every book. Holding out whole books means the model
    must generalize to a new book, not recognize one it has already seen."""
    feats = _chunks(books, fs, chunk_size)
    y_true, y_pred, per_book = [], [], []
    for i, book in enumerate(books):
        x_train = np.vstack([f for j, f in enumerate(feats) if j != i])
        y_train = np.concatenate(
            [[b.author] * len(f) for j, (b, f) in enumerate(zip(books, feats)) if j != i]
        )
        pred = make_model().fit(x_train, y_train).predict(feats[i])
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
        "features": fs.name,
        "chunk_size": chunk_size,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "books_correct_by_vote": sum(r["majority_vote"] == r["author"] for r in per_book),
        "per_book": per_book,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def paper_level(papers: list[Book], fs: FeatureSet) -> np.ndarray:
    """One row of feature rates per whole document."""
    return np.array([rates(fs.encode(p.words), fs.n_features) for p in papers])


def leave_one_out_documents(known: list[Book], fs: FeatureSet) -> dict:
    """Leave-one-document-out accuracy on documents of known authorship."""
    x, y = paper_level(known, fs), np.array([p.author for p in known])
    pred, per_document = [], []
    for i, doc in enumerate(known):
        model = make_model().fit(np.delete(x, i, 0), np.delete(y, i))
        proba = model.predict_proba(x[i : i + 1])[0]
        pred.append(model.classes_[np.argmax(proba)])
        per_document.append({"title": doc.title, "author": doc.author, "predicted": pred[-1]}
                            | {f"p_{c.lower()}": float(p) for c, p in zip(model.classes_, proba)})
    pred = np.array(pred)
    return {
        "per_document": per_document,
        "features": fs.name,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "correct": int(np.sum(pred == y)),
        "n": len(y),
        "misclassified": "; ".join(p.title for p, ok in zip(known, pred == y) if not ok),
    }


def predict_documents(known: list[Book], unknown: list[Book], fs: FeatureSet,
                      positive: str) -> np.ndarray:
    """Train on all known documents; return P(`positive`) for each unknown one."""
    model = make_model().fit(paper_level(known, fs), [p.author for p in known])
    col = list(model.classes_).index(positive)
    return model.predict_proba(paper_level(unknown, fs))[:, col]
