# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ccd-diffusion-paper` is a **reproducible scientific article**, not a conventional
software library. The `ccd_diffusion` Python package programmatically generates a
complete JATIS (SPIE) LaTeX article (text, figures, tables, and numeric values)
measuring the depth-dependent charge-diffusion kernel of the IRIS CCDs from glancing
particle tracks. Calling `ccd_diffusion.pdf()` produces `ccd-diffusion.pdf`.

This is one package within the larger Kankelborg-Group workspace (see the parent
`../CLAUDE.md`). It depends on the group's stack, `named-arrays`
(`import named_arrays as na`), `optika`, and `aastex`, plus `pylatex`, `scipy`, and
`astropy.units`. Use `named_arrays` rather than `numpy` for array work in new code.

The measurement began as an appendix of `../ccd-snr-paper` (`ccd_snr/tracks/`), and
`ccd_diffusion/tracks/` is a copy of that code and data. All analysis code stays in
this package for now; do not split it into a library without being asked.

## Commands

Run from this package directory:

```bash
pip install -e .[test]          # install for development
pytest                          # run tests; test_pdf compiles the LaTeX → PDF
pytest ccd_diffusion/_document_test.py::test_pdf   # build the PDF specifically
python -m ccd_diffusion.tracks fit      # refit every track and rewrite tracks/data/iris_fits.csv and iris_depleted.csv (minutes)
python -m ccd_diffusion.tracks select   # ask the JSOC catalog which frames each campaign covers and rewrite the frame list
python -m ccd_diffusion.tracks plan     # which campaigns the frame list and the current extractor would change (fresh/stale/new)
python -m ccd_diffusion.tracks extract  # fetch 3.8 GB of level-1 images block by block and rewrite the cutouts (an hour or more; ask first)
python -m ccd_diffusion.tracks extract --dataset sji --output out/sji   # one campaign, results elsewhere
python -m ccd_diffusion.tracks export --dataset sji --output out/sji    # the same layout, from the package's own data
python -m ccd_diffusion.tracks merge out/* --report report.md          # join campaigns extracted or exported separately
black ccd_diffusion             # format (CI enforces --check)
ruff check                      # lint (CI enforces)
```

Building the PDF requires LaTeX with `latexmk` and the `authblk` package
(`texlive-latex-extra` on Ubuntu), and matplotlib's usetex support, since
`_document.py` sets `text.usetex = True`. CI runs only on Linux.

## Architecture

The article is assembled in `ccd_diffusion/_document.py`: `document()` builds an
`aastex.Document` on SPIE's `spieman` class (passing the class and bibliography
style files shipped in this package), appends the front matter from `spie.py`, each
section, the back matter the journal requires (disclosures, availability,
acknowledgments), and the bibliography.

- **`spie.py`** holds the SPIE-specific front and back matter (`Title`, `Author`,
  `Affiliation`, `Abstract`, `Keywords`, `Corresponding`, `Disclosures`,
  `Availability`, `Acknowledgments`, `Biography`). Everything journal-independent
  (`Variable`, `Acronym`, `Section`, `Figure`, `Bibliography`) comes from `aastex`.
- **`sections/`**, **`figures/`**, **`tables/`** mirror the parts of the article, each
  a subpackage of public factory functions re-exported in its `__init__.py`.
- **`_variables.py`** defines `aastex.Variable` macros for every numeric value cited in
  the prose, computed from `tracks`. Reference `\variableName` in section strings
  rather than hardcoding a number.
- **`tracks/`** is the measurement: `_select.py` (the `campaigns` table and the JSOC
  catalog query that turns it into the frame list; expanding the dataset means adding
  campaigns there), `_provenance.py` (a fingerprint per campaign over its frame serial
  numbers and a token-level hash of the extractor, stored in `data/iris_campaigns.csv`;
  `plan`/`export`/`merge` make regeneration incremental), `_archive.py` (fetching level-1 images from LMSAL
  into `~/.cache/ccd_diffusion/iris`), `_extract.py` (backgrounds, masks, the track
  finder, and the azimuth census; the one module that works on plain numpy arrays),
  `_tracks.py` (the cutouts and metadata), `_fit.py` (the three-parameter width model;
  `scan` fits t_c, sigma_max, orientation, and the nuisance centerline to one track at
  every sigma_d on the grid, assembling misfits from a per-slice table), `_depleted.py`
  (`pooled` picks one sigma_d per CCD by summing the flat tracks' misfits, iterated with
  the flat selection; `fit_all` calls it), `_samepix.py`, `_stacked.py` (the analyses),
  `_images.py` (two example frames). Every data product under `tracks/data/`
  is committed because regenerating it needs the archive or minutes of fitting.
- **`_ccd.py`** is the `optika` sensor model the measurement is compared with.
- **`.github/workflows/data.yml`** regenerates everything under `tracks/data/` on
  GitHub Actions (`workflow_dispatch`): a `select` job builds the frame list and the
  plan, one job per campaign extracts it or exports it from the package if its
  fingerprint is current, and a `merge` job (which runs even if a campaign failed) joins,
  fits, and opens a pull request. Nothing is cached; the cutouts in the repo are the
  cache. Prefer it to running `extract` locally.
- **`docs/reports/tracks.ipynb`** is the exploratory notebook, executed by nbsphinx on
  every documentation build; its text cells are raw reStructuredText.

## Conventions

- Every module declares an explicit `__all__` and exposes small factory functions,
  re-exported up the package via `__init__.py`.
- Modules import the top-level package as `import ccd_diffusion` and reach back into
  it (`ccd_diffusion.tracks.fits()`) rather than using deep relative imports.
- Prose is LaTeX in raw strings; write such content with the Write or Edit tools, not
  shell heredocs, which mangle backslashes.
- No em dashes in prose.
