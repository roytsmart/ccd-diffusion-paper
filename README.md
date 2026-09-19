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

## What this is

A charged particle crossing a back-illuminated CCD at a glancing angle samples
every depth of the silicon along its track, so the width of the charge it
leaves behind maps the lateral diffusion of charge against depth in a single
exposure. This article uses such tracks, found in level-1 images from the
Interface Region Imaging Spectrograph (IRIS), to measure the depth-dependent
charge-diffusion kernel of its three CCDs in orbit, with no laboratory access,
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
documentation.

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
dataset grows.

The images are cached in `~/.cache/ccd_diffusion/iris`, or wherever
`CCD_DIFFUSION_CACHE` points.

Neither step needs to run on your own machine. The `data` workflow in
GitHub Actions (started from the Actions tab) extracts every campaign in
its own job, with the images cached between runs, then joins the
campaigns, refits every track, and opens a pull request with the new data
and an article preview. A fine-grained personal access token stored as the
`DATA_TOKEN` secret lets that pull request run the tests like any other.

Formatting and linting, both enforced in CI:

```bash
black ccd_diffusion
ruff check
```
