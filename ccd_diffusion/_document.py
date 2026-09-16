import pathlib
import matplotlib.pyplot as plt
import pylatex
import aastex
import ccd_diffusion

__all__ = [
    "document",
    "pdf",
]


def document(linenumbers: bool = True) -> aastex.Document:
    """
    An :mod:`aastex` representation of the article, built on the SPIE journal
    class.

    Parameters
    ----------
    linenumbers
        Whether to number the lines, as the journal likes for review.
    """

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["lines.linewidth"] = 1

    spie = ccd_diffusion.spie

    doc = aastex.Document(
        documentclass="spieman",
        class_files=[spie.path_class, spie.path_bibliography_style],
        bibliographystyle="spiejour",
        document_options=["12pt"],
        lmodern=False,
        textcomp=False,
        linenumbers=False,
    )

    doc.packages.append(aastex.Package("amsmath"))
    doc.packages.append(aastex.Package("graphicx"))
    doc.packages.append(aastex.Package("siunitx"))
    doc.packages.append(aastex.Package("hyperref"))
    if linenumbers:
        doc.packages.append(aastex.Package("lineno"))
        doc.preamble.append(pylatex.Command("linenumbers"))

    doc.preamble += ccd_diffusion.acronyms()
    doc.variables += ccd_diffusion.variables()

    doc.preamble.append(
        spie.Title(
            "Measuring the Charge-diffusion Kernel of a Back-illuminated CCD "
            "in Orbit from Glancing Particle Tracks"
        )
    )
    authors = ccd_diffusion.authors()
    affiliations = []
    for author in authors:
        doc.preamble.append(author)
        for affiliation in author.affiliations:
            if affiliation not in affiliations:
                affiliations.append(affiliation)
    doc.preamble += affiliations

    doc.append(pylatex.Command("maketitle"))
    doc.append(ccd_diffusion.sections.abstract())
    doc.append(ccd_diffusion.keywords())
    (corresponding,) = [a for a in authors if a.corresponding]
    doc.append(spie.Corresponding(corresponding))

    doc.append(ccd_diffusion.sections.introduction())
    doc.append(ccd_diffusion.sections.method())
    doc.append(ccd_diffusion.sections.results())
    doc.append(ccd_diffusion.sections.discussion())
    doc.append(ccd_diffusion.sections.conclusion())

    doc.append(ccd_diffusion.sections.disclosures())
    doc.append(ccd_diffusion.sections.availability())
    doc.append(ccd_diffusion.sections.acknowledgments())

    doc.append(aastex.Bibliography("sources"))

    return doc


def pdf(linenumbers: bool = True) -> pathlib.Path:
    """
    Build a pdf version of :func:`document` and return the path of the document.

    Parameters
    ----------
    linenumbers
        Whether to number the lines, as the journal likes for review.
    """

    doc = document(linenumbers=linenumbers)

    path = pathlib.Path(__file__).parent / "ccd-diffusion"
    doc.generate_pdf(
        filepath=path,
        clean_tex=False,
    )

    return path.with_suffix(".pdf")
