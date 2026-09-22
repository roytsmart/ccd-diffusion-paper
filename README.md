# ccd-diffusion-paper

[![tests](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/tests.yml/badge.svg)](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/tests.yml)
[![Black](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/black.yml/badge.svg)](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/black.yml)
[![Ruff](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/ruff.yml/badge.svg)](https://github.com/roytsmart/ccd-diffusion-paper/actions/workflows/ruff.yml)

**Measuring the Charge-diffusion Kernel of a Back-illuminated CCD in Orbit from Glancing Particle Tracks**

Roy T. Smart and Charles C. Kankelborg
*Montana State University, Department of Physics*

An article for the *Journal of Astronomical Telescopes, Instruments, and
Systems* (JATIS).

📄 **[Read the current draft (pdf)](https://roytsmart.github.io/ccd-diffusion-paper/ccd-diffusion.pdf)**
🔍 **[Browse every track](https://roytsmart.github.io/ccd-diffusion-paper/docs/browser.html)** and **[the frames they came from](https://roytsmart.github.io/ccd-diffusion-paper/docs/frames.html)** in the [documentation](https://roytsmart.github.io/ccd-diffusion-paper/docs/)

## What this is

A charged particle crossing a back-illuminated CCD at a glancing angle samples
every depth of the silicon along its track, so the width of the charge it
leaves behind maps the lateral diffusion of charge against depth in a single
exposure. This article uses such tracks, found in level-1 images from the
Interface Region Imaging Spectrograph (IRIS), to measure the depth-dependent
charge-diffusion kernel of its four CCDs in orbit, with no laboratory access,
and compares it with the field-free diffusion model used to predict the noise
of ultraviolet images.

It grew out of an appendix of
[ccd-noise-paper](https://github.com/roytsmart/ccd-noise-paper), which needed
one number from the measurement, the probability that two electrons from a
photon absorbed at the back surface are collected in the same pixel.

## This repository *is* the article

Every figure, table, and number quoted in the prose is computed from the track
cutouts and fits in `ccd_diffusion/tracks/data/` when the article is built.
Calling

```python
import ccd_diffusion
ccd_diffusion.pdf()
```

renders the figures, typesets the LaTeX with SPIE's `spieman` class, and
produces `ccd-diffusion.pdf`. Numbers reach the prose as `aastex.Variable`
macros rather than literals.

The exploratory analysis behind the article is a notebook,
`docs/reports/tracks.ipynb`, which is executed and published with the
documentation. The documentation also holds a browser of every track in
the package, with its cutout, its fit, and the model evaluated beside it,
and a gallery of the three level-1 frames from each camera of every campaign
with the tracks outlined; `python -m ccd_diffusion.tracks browser
--output DIR` writes what those pages read.

## Layout

| path | contents |
|---|---|
| `_document.py` | assembles the article and renders the pdf |
| `spie.py` | the front and back matter of an SPIE journal article |
| `sections/` | the prose, as LaTeX strings, one module per section |
| `figures/`, `tables/` | one module per figure and table |
| `_variables.py` | the `aastex.Variable` macros quoted in the prose |
| `_acronyms.py` | acronym definitions used as `\ACRONYM` macros |
| `_ccd.py` | the sensor model the measurement is compared with |
| `tracks/` | the track cutouts, the fit of the diffusion model, and the same-pixel probabilities |
| `tracks/data/` | the cutouts, the frames searched, the fits of every track and of each CCD's depletion spread, and two example level-1 images |
| `sources.bib` | the bibliography |
| `spieman.cls`, `spiejour.bst` | the SPIE journal class and bibliography style |

The LaTeX is generated through [`aastex`](https://github.com/sun-data/aastex),
whose journal-independent parts (variables, acronyms, sections, figures, the
build) are used with SPIE's class.

## Building

Requires Python 3.12 or newer and a LaTeX installation with `latexmk` and the
`authblk` package. On Ubuntu:

```bash
sudo apt-get install latexmk texlive-publishers texlive-science texlive-latex-extra cm-super libcairo2-dev
```

```bash
pip install -e .[test]
pytest                       # builds and validates the pdf
python -c "import ccd_diffusion; ccd_diffusion.pdf()"
```

The track cutouts are extracted from about 3200 level-1 images, 3.8 GB in
all, which are fetched from the IRIS archive at LMSAL one block at a time
and, unless `--keep` is given, deleted once their block is done. Fitting
every track then takes a few minutes. Both results are stored under
`tracks/data/` and only recomputed by

```bash
python -m ccd_diffusion.tracks select    # ask the catalog which frames each campaign covers
python -m ccd_diffusion.tracks extract   # fetch the images and rewrite the cutouts and census
python -m ccd_diffusion.tracks fit       # refit every track and rewrite the fits
```

The campaigns themselves are a table in `tracks/_select.py`, giving the
time windows each one covers, the cameras to keep, and how heavily to
subsample the frames outside the anomaly that the background is estimated
from. Adding a campaign there and running the `data` workflow is how the
dataset grows. Candidates come from

```bash
python -m ccd_diffusion.tracks search --stop 2026-09 --output candidates.csv
```

which asks the catalog, month by month, for every frame taken inside the
anomaly, groups them by observing program and day, and keeps the ones
whose spectrograph window mostly looks off the limb (the pointing
keywords give the center of the field, and a program pointed beyond the
limb can still read out rows on the disk, so the window's own coordinates
decide), with exposures of 4 to 8 s and enough such frames, ranked by the
exposed seconds inside the anomaly on the part of the slit off the limb,
since a longer exposure crowds the frame with hits that spoil the tracks
around them. Which candidates to adopt is still a judgment.

The images are cached in `~/.cache/ccd_diffusion/iris`, or wherever
`CCD_DIFFUSION_CACHE` points.

Neither step needs to run on your own machine. The `data` workflow in
GitHub Actions (started from the Actions tab) rebuilds the frame list
from the catalog, or takes the one committed in the package when started
with `frames: package`, since the catalog is slow at times, then
works out which campaigns have changed, extracts each of those in its own
job while taking the rest from the data already in the repository, then
joins the campaigns, refits every track, and opens a pull request with the
new data, an article preview, and a documentation preview. Each campaign
carries a fingerprint in `tracks/data/iris_campaigns.csv` over the frames
it covers and the extraction code, so adding a campaign fetches only that
campaign, and changing the finder fetches everything. The workflow
approves the checks GitHub holds on the pull request it opened, so it
needs no personal token.

The `docs` workflow builds the documentation on every push and pull
request and publishes it beside the article on GitHub Pages, under
`docs/` for `main` and `pr/N/docs/` as a preview.

Formatting and linting, both enforced in CI:

```bash
black ccd_diffusion
ruff check
```
