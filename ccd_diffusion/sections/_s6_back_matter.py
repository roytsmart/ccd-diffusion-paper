import ccd_diffusion

__all__ = [
    "disclosures",
    "availability",
    "acknowledgments",
]


def disclosures():
    """The conflict-of-interest and AI-tool disclosures the journal requires."""
    return ccd_diffusion.spie.Disclosures(r"""
The authors declare that there are no financial interests, commercial
affiliations, or other potential conflicts of interest that could have
influenced the objectivity of this research or the writing of this paper.
The analysis code and the text of this article were drafted with the
assistance of Claude, a large language model, under the direction and
review of the authors.""")


def availability():
    """The code, data, and materials availability statement the journal requires."""
    return ccd_diffusion.spie.Availability(r"""
This article is executable \cite{Lasser2020}: every figure, table, and
number quoted in the text is computed from the tracks when the article is
built.
The code, the track cutouts, and the fits are available at
\url{https://github.com/roytsmart/ccd-diffusion-paper}, and the level-1
\IRIS\ images they were extracted from are available from the \IRIS\ data
archive.""")


def acknowledgments():
    """The acknowledgments, before the references."""
    result = ccd_diffusion.spie.Acknowledgments()
    result.append(r"""
This work was supported by NASA through the \IRIS\ mission.""")
    return result
