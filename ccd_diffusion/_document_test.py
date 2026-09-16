import pathlib
import pymupdf
import pylatex
import ccd_diffusion


def test_document():
    doc = ccd_diffusion.document()
    assert isinstance(doc, pylatex.Document)
    tex = doc.dumps()
    assert r"\documentclass[12pt]{spieman}" in tex
    assert r"\bibliographystyle{spiejour}" in tex


def test_pdf():
    pdf = ccd_diffusion.pdf()
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
