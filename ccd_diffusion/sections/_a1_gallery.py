import aastex
import ccd_diffusion

__all__ = [
    "gallery",
]


def gallery() -> aastex.Section:
    result = aastex.Section(
        "The tracks as the finder cut them out", label="appendix:gallery"
    )
    result.escape = False
    result.append(r"""
The cutouts distributed with this article are seven pixels wide and follow
the fitted centerline, so they show the tracks after the finder has done
its work.
Figure~\ref{fig:gallery} shows the longest of the flat tracks and one
track that the Bragg cut rejects.
Sub-pixel structure is invisible at this scale, which is why the wedge
signature has to be extracted by fitting.""")
    result.append(ccd_diffusion.figures.gallery())
    return result
