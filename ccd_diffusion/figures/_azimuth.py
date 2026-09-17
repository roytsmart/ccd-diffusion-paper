import pathlib
import aastex

__all__ = [
    "azimuth",
]

_path = pathlib.Path(__file__).parent / "azimuth.png"
"""
The census of track azimuths, made from the full level-1 frames during the
exploratory analysis. The frames are not distributed with this article, so
the figure is kept as an image rather than regenerated.
"""


def azimuth() -> aastex.Figure:
    """The direction of every elongated particle hit in the spectrograph frames."""
    result = aastex.Figure("azimuth", position="htb!")
    result.add_image(_path, width=aastex.NoEscape(r"\textwidth"))
    result.add_caption(aastex.NoEscape(r"""
The azimuth in the detector frame of every elongated connected component
above $5\sigma$ in the spectrograph frames, with a charge-weighted width
under 3 pixels and an aspect ratio over 4, before any of the finder's cuts,
so that the whole circle is populated.
Tracks are undirected, so each is drawn at $\theta$ and $\theta + 180^\circ$,
and the grey wedges are the windows about the row and column axes that the
finder accepts.
Left: the frames taken inside \SAA\ at both spacecraft rolls.
Right: quiet frames at the same rolls.
The trapped protons arrive strongly aligned, mostly within the window
along the slit, while the cosmic rays are isotropic to within counting
noise; the preferred direction shifts and broadens when the spacecraft is
rolled, so it is a property of the arrival direction as seen by the
instrument rather than of the detector."""))
    return result
