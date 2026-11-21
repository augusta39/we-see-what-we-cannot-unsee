# We See What We Cannot Unsee

*Pursuits and Perception* · Roddsi

A life is a handful of pursuits sharing one finite clock, and what we have already lived changes what each new moment comes to mean, and so where our time goes next. In short, we see what we cannot unsee.

## Where it stands

- **Supported, modestly:** an experience's significance keeps shifting after it ends (37% of importance ratings moved when people retold a memory, Hippocorpus), people revise what life events mean for years (MainLife), feeling and meaning move together without being the same thing (experience sampling, 351 people), and significance and attention feed each other (three-month goal diary, 247 people).
- **Untested:** the central claim, H6, that changes in how much past experiences matter predict how later events are read and where finite time goes. One attempt was inconclusive because its similarity measure failed a validity check.
- All of the evidence is correlational and the effects are small.

## Watch

<!-- For inline players: click the pencil to edit this file on GitHub, then drag each .mp4 from your computer into the editor just below its line. GitHub uploads it and inserts a link that plays in place. Free accounts allow up to 10 MB per video. -->

- **The full explainer** (9 min): the idea and its assumptions, then the math and the pyramid geometry, then the evidence. [Watch](videos/we_see_what_we_cannot_unsee_full.mp4)
- **True stories** (4 min): Archimedes, Darwin and Wallace, Fleming and Jobs, tied back to the idea. [Watch](videos/we_see_what_we_cannot_unsee_true_stories.mp4)

## Read

- [The essay](essay/essay.md), with the formal model and proofs in its appendix

<!-- Add your Medium link here once it is published. -->

## What's here

```
essay/       the essay in Markdown, with its four figures
analysis/    reanalysis scripts and their results (JSON)
protocols/   the two H6 test plans, frozen and hashed before each run
video/       the Python that draws every frame of the videos
videos/      the rendered videos
data/        where to get the public datasets (no raw data stored)
```

## Reproduce

- **Analysis:** Python 3 with numpy, pandas, scipy, scikit-learn and openpyxl. Get the data from the sources in [`data/README.md`](data/README.md), then follow [`analysis/README.md`](analysis/README.md).
- **Videos:** Python 3 with numpy, Pillow and matplotlib, plus ffmpeg and the Latin Modern fonts. See [`video/README.md`](video/README.md).

## Credits

Idea, hypotheses and essay: Roddsi.
Data analysis, statistics, code and animation: Claude (Anthropic), at Roddsi's direction.
The datasets belong to their original authors; see [`data/README.md`](data/README.md).

## License

No license has been chosen yet, so by default all rights are reserved: people can read and fork this on GitHub but may not reuse it. To add one, use **Add file → Create new file**, name it `LICENSE`, and pick a template.
