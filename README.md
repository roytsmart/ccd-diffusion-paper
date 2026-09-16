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
| `tracks/data/` | the cutouts, the frames searched, and the fits of every track |
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

Fitting every track takes a few minutes, so the fits are stored in
`tracks/data/iris_fits.csv` and only recomputed by

```bash
python -m ccd_diffusion.tracks
```

Formatting and linting, both enforced in CI:

```bash
black ccd_diffusion
ruff check
```
