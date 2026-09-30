import pathlib
import pytest
import pymupdf
import ccd_diffusion


@pytest.fixture(scope="module")
def pdf() -> pathlib.Path:
    """
    The article, built once for the tests below: generating the figures is
    most of the cost, and :func:`ccd_diffusion.pdf` keeps the LaTeX source
    beside the pdf, so the source can be checked without a second build.
    """
    return ccd_diffusion.pdf()


def test_document(pdf: pathlib.Path):
    tex = pdf.with_suffix(".tex").read_text(encoding="utf-8")
    assert r"\documentclass[12pt]{spieman}" in tex
    assert r"\bibliographystyle{spiejour}" in tex


def test_pdf(pdf: pathlib.Path):
    assert isinstance(pdf, pathlib.Path)
    assert pdf.exists()

    with pymupdf.open(pdf) as document:
        text = "".join(page.get_text() for page in document)

    # The bibliography and the cross-references are only resolved if the
    # article is compiled repeatedly with bibtex in between, which `latexmk`
    # does and a bare `pdflatex` does not.
    assert "References" in text
    assert "(?)" not in text
    assert "??" not in text
