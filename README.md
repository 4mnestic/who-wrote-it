# Who wrote it? Identifying authors from their writing habits

[![tests](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/4mnestic/word-length-distribution-goodness-of-fit/actions/workflows/ci.yml)

In 1887, T. C. Mendenhall claimed every writer has a telltale pattern of word lengths. I tested
that on 2.4 million words from Austen, Dickens and Tolstoy, found a signal that works far
better, and used it on the disputed Federalist Papers.

## Results

- **Word length doesn't identify authors.** On 1,000-word passages it picks the right author
  50% of the time, against 33% for a random guess. The differences between authors are
  statistically significant but no bigger than the differences among one author's own books.
- **Function words do.** Using how often a writer uses 70 common words (*the, upon,
  would, whilst*…), accuracy is 90% at 1,000 words and 99% at 5,000. All 12 novels are
  attributed correctly.
- **The Federalist Papers.** After correctly attributing 64 of 65 papers of known
  authorship, the model gives 11 of the 12 disputed papers to Madison with at least 93%
  confidence, matching the historical consensus. No. 55 is too close to call.

![Accuracy by passage length](figures/accuracy_by_passage_length.png)

## The Federalist Papers

Hamilton and Madison both claimed authorship of 12 of these 1787–88 essays. Their word
lengths are nearly identical, so word length can't separate them (grey). Function words can
(blue):

![Disputed Federalist Papers](figures/federalist_disputed.png)

The clearest single clue is *upon*. Hamilton uses it 3.3 times per 1,000 words, Madison
0.18 times, and the disputed papers 0.13 times.

## How it was tested

- **No peeking:** each model is scored only on books (or papers) it never saw in training.
- **Fair scoring:** accuracy is balanced across authors, so a model can't score well by
  always guessing the most prolific one.
- **Clean text:** prefaces, contents pages and chapter headings are stripped in code.

Full tables and test details are in [METHODS.md](METHODS.md).

## Limitations

- Only three novelists, and a two-way choice for the Federalist Papers. Choosing among many
  authors is harder.
- The Tolstoy novels are English translations, so their "author" signal partly reflects the
  translators.

## Reproduce

```bash
uv sync
uv run python -m wld.analysis   # rewrites results/ and figures/
uv run pytest
```

Code is in `wld/`; every number above comes from a file in `results/`.

## Background

This project began as a term paper for LING 388, which I later extended with
[Claude](https://claude.ai).

## References

- Mendenhall, T. C. (1887). The characteristic curves of composition. *Science*, 9(214), 237–249.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*.
  Addison-Wesley.
