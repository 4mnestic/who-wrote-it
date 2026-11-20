# Who wrote it? Identifying authors from their writing habits

[![tests](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml)

In 1887, T. C. Mendenhall claimed every writer has a telltale pattern of word lengths. I tested
that on 2.4 million words from Austen, Dickens and Tolstoy, found a signal that works far
better (function words), and used it on the disputed Federalist Papers.

## Results

- **Word length doesn't identify authors.** I cut each book into passages and trained a
  logistic regression (scikit-learn) to name the author from its word lengths. To stop it
  from cheating by recognizing a book's characters, it was always tested on a book it never
  saw: train on 11, test on the 12th, rotate. It was right 50% of the time on 1,000-word
  passages, against 33% for a random guess. A chi-square test (SciPy) calls the authors'
  differences significant, but the effect size (Cramér's V = 0.04) is no bigger than the
  gap between one author's own books. With 2.4 million words almost anything is
  "significant"; size is what matters.
- **Function words identify authors.** I swapped in how often each passage uses 70 common words (*the,
  upon, would, whilst*…). They appear whatever the topic, so they reflect habit rather than
  plot. With the same model and the same test, accuracy reaches 90% at 1,000 words and 99%
  at 5,000, and all 12 novels are attributed correctly. Keeping the model fixed means the
  gain comes from the signal, not a fancier algorithm.
- **The Federalist Papers.** Before trusting the model on the disputed papers, I checked it
  on the 65 known ones, holding out one at a time. Hamilton wrote 51 of them, so always
  guessing "Hamilton" would score 78%; I weighted the authors equally and scored with
  balanced accuracy to rule that out. It got 64 right, missing only the shortest paper by
  52% to 48%. On the disputed papers, 11 come out at least 93% Madison, matching the
  historical consensus. No. 55 comes out 41%, so I report it as too close to call rather
  than force a verdict.

![Accuracy by passage length](figures/accuracy_by_passage_length.png)

## The Federalist Papers

Hamilton and Madison both claimed authorship of 12 of these 1787–88 essays. Their word
lengths are nearly identical, so word length can't separate them (grey). Function words can
(blue):

![Disputed Federalist Papers](figures/federalist_disputed.png)

The clearest single clue is *upon*. Hamilton uses it 3.3 times per 1,000 words, Madison
0.18 times, and the disputed papers 0.13 times.

Full tables are in [METHODS.md](METHODS.md); code is in [`wld/`](wld/).

## Limitations

- Only three novelists, and a two-way choice for the Federalist Papers. Choosing among many
  authors is harder.
- The Tolstoy novels are English translations, so their "author" signal partly reflects the
  translators.

## Reproduce

One command rebuilds every table (pandas) and chart (matplotlib) from the raw texts,
including the cleaning: editors' prefaces and chapter headings are stripped by code rather
than by hand, so the results can be regenerated exactly.
Dependencies are locked with uv, and automated tests (pytest) check the cleaning, features
and statistics on every push via GitHub Actions.

```bash
uv sync
uv run python -m wld.analysis   # rewrites results/ and figures/
uv run pytest
```

## Background

This project began as a term paper for LING 388, which I later extended with
[Claude](https://claude.ai).

## References

- Mendenhall, T. C. (1887). The characteristic curves of composition. *Science*, 9(214), 237–249.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*.
  Addison-Wesley.
