"""Run the full analysis: `python -m wld.analysis`. Writes results/ and figures/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .classify import leave_one_book_out  # noqa: E402
from .corpus import Book, bin_counts, bin_labels, load_books  # noqa: E402
from .stats import exact_permutation_test, hellinger, homogeneity_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZES = [100, 250, 500, 1000, 2500, 5000, 10000]
NULL_CHUNK_SIZE, NULL_DRAWS, SEED = 1000, 200, 0

# Validated categorical slots 1-3 (all-pairs safe), assigned to authors in sorted order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def colors(authors) -> dict[str, str]:
    """Fixed color per author: same author, same color in every figure."""
    return dict(zip(sorted(authors), PALETTE))


def author_table(books: list[Book]) -> tuple[list[str], np.ndarray]:
    """Authors (sorted) and their pooled word-length bin counts."""
    authors = sorted({b.author for b in books})
    table = np.array(
        [sum(bin_counts(b.lengths) for b in books if b.author == a) for a in authors]
    )
    return authors, table


def tests(books: list[Book]) -> pd.DataFrame:
    """Chi-square homogeneity: between authors, then among each author's books."""
    authors, table = author_table(books)
    rows = [{"comparison": "Between authors (all books pooled)", **homogeneity_test(table)}]
    for a in authors:
        t = np.array([bin_counts(b.lengths) for b in books if b.author == a])
        rows.append({"comparison": f"Among {a}'s four books", **homogeneity_test(t)})
    return pd.DataFrame(rows)


def classifier_null(books: list[Book], rng: np.random.Generator) -> np.ndarray:
    """Balanced accuracy when author labels are shuffled across whole books
    (group sizes kept), so chunks of a book always share a label. This is the
    accuracy you'd expect from book-to-book variation alone."""
    labels = np.array([b.author for b in books])
    out = []
    for _ in range(NULL_DRAWS):
        perm = rng.permutation(labels)
        fake = [Book(b.title, a, b.translator, b.lengths) for b, a in zip(books, perm)]
        out.append(leave_one_book_out(fake, NULL_CHUNK_SIZE)["balanced_accuracy"])
    return np.array(out)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def fig_distributions(authors, table, path):
    """Share of words at each length, one line per author."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    x = np.arange(table.shape[1])
    props = table / table.sum(axis=1, keepdims=True)
    for a, p in zip(authors, props):
        ax.plot(x, p * 100, color=colors(authors)[a], lw=2, marker="o", ms=4, label=a)
    ax.set_xticks(x, bin_labels())
    ax.set_xlabel("Word length (letters)", color=INK)
    ax.set_ylabel("Share of words (%)", color=INK)
    ax.set_title("The three authors' word-length curves almost overlap", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mean_length(books, path):
    """Mean word length of every book, grouped by author."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    order = sorted(books, key=lambda b: (b.author, b.lengths.mean()), reverse=True)
    for i, b in enumerate(order):
        ax.plot(b.lengths.mean(), i, "o", ms=9, color=colors({b.author for b in books})[b.author], mec=SURFACE,
                mew=2, label=b.author)
    ax.set_yticks(range(len(order)), [b.title for b in order], color=INK)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend([uniq[a] for a in sorted(uniq)], sorted(uniq), frameon=False,
              labelcolor=INK, loc="upper left")
    ax.set_xlabel("Mean word length (letters)", color=INK)
    ax.set_title("Books by the same author vary about as much as authors do",
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_accuracy(curve: pd.DataFrame, null: np.ndarray, path):
    """Held-out-book accuracy vs passage length, against chance and the shuffled-label null."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.plot(curve.chunk_size, curve.balanced_accuracy * 100, color=PALETTE[0],
            lw=2, marker="o", ms=6, label="Word-length classifier")
    ax.axhline(100 / 3, color=INK_2, lw=1, ls="--")
    ax.text(curve.chunk_size.max(), 100 / 3 + 1, "chance (33%)", ha="right",
            color=INK_2, fontsize=8)
    lo, hi = np.percentile(null * 100, [2.5, 97.5])
    ax.errorbar([NULL_CHUNK_SIZE], [np.median(null) * 100],
                yerr=[[np.median(null) * 100 - lo], [hi - np.median(null) * 100]],
                fmt="s", color=INK_2, ms=6, capsize=4,
                label="Shuffled authors (95% range)")
    for _, r in curve.iterrows():
        ax.annotate(f"{r.balanced_accuracy:.0%}", (r.chunk_size, r.balanced_accuracy * 100),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(curve.chunk_size, [f"{c:,}" for c in curve.chunk_size])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Passage length (words)", color=INK)
    ax.set_ylabel("Balanced accuracy on unseen books (%)", color=INK)
    ax.set_title("Guessing the author from word lengths alone", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(root: Path = ROOT) -> dict:
    res, figs = root / "results", root / "figures"
    res.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    books = load_books()

    # 1. Corpus summary and distributions.
    pd.DataFrame(
        [{"title": b.title, "author": b.author, "translator": b.translator or "",
          "words": len(b.lengths), "mean_length": round(b.lengths.mean(), 3)}
         for b in books]
    ).to_csv(res / "books.csv", index=False)
    authors, table = author_table(books)
    props = pd.DataFrame(table / table.sum(axis=1, keepdims=True),
                         index=authors, columns=bin_labels())
    props.round(5).to_csv(res / "author_distributions.csv", index_label="author")

    # 2. Chi-square homogeneity tests with effect sizes.
    tests(books).to_csv(res / "chi_square_tests.csv", index=False)

    # 3. Book-level exact permutation test.
    perm = exact_permutation_test(
        hellinger([bin_counts(b.lengths) for b in books]),
        np.array([b.author for b in books]),
    )

    # 4. Leave-one-book-out classification.
    runs = [leave_one_book_out(books, c) for c in CHUNK_SIZES]
    curve = pd.DataFrame(
        [{k: r[k] for k in ("chunk_size", "balanced_accuracy", "books_correct_by_vote")}
         for r in runs]
    )
    curve.round(4).to_csv(res / "classification_by_chunk_size.csv", index=False)
    ref = runs[CHUNK_SIZES.index(NULL_CHUNK_SIZE)]
    pd.DataFrame(ref["per_book"]).round(4).to_csv(
        res / f"classification_per_book_{NULL_CHUNK_SIZE}.csv", index=False)
    pd.crosstab(pd.Series(ref["y_true"], name="true"),
                pd.Series(ref["y_pred"], name="predicted")).to_csv(
        res / f"confusion_matrix_{NULL_CHUNK_SIZE}.csv")
    null = classifier_null(books, np.random.default_rng(SEED))

    summary = {
        "permutation_test": perm,
        "classifier_null": {
            "chunk_size": NULL_CHUNK_SIZE,
            "draws": NULL_DRAWS,
            "observed": ref["balanced_accuracy"],
            "null_median": float(np.median(null)),
            "null_95th_percentile": float(np.percentile(null, 95)),
            "p": float((1 + np.sum(null >= ref["balanced_accuracy"])) / (1 + len(null))),
        },
    }
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig_distributions(authors, table, figs / "word_length_by_author.png")
    fig_mean_length(books, figs / "mean_length_by_book.png")
    fig_accuracy(curve, null, figs / "classifier_accuracy.png")
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(main(), indent=2))
