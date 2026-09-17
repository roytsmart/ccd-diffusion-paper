import numpy as np
import matplotlib.pyplot as plt
import aastex
import ccd_diffusion

__all__ = [
    "gallery",
]

_num = 12
"""The number of cutouts shown."""


def gallery() -> aastex.Figure:
    """The longest flat tracks and one stopping track, as the finder cut them out."""
    tracks = ccd_diffusion.tracks
    fits = tracks.fits()
    longest = sorted([f for f in fits if f.flat], key=lambda f: -f.track.length)[
        : _num - 1
    ]
    stopping = max(fits, key=lambda f: f.bragg)

    fig, ax = plt.subplots(
        nrows=_num // 2,
        ncols=2,
        figsize=(6.5, 5.5),
        constrained_layout=True,
    )
    for a, f in zip(ax.flat, longest + [stopping]):
        charge = f.track.charge
        if f.orientation < 0:
            charge = charge[{tracks.axis_slice: slice(None, None, -1)}]
        image = charge.transpose((tracks.axis_pixel, tracks.axis_slice)).ndarray
        a.imshow(
            image,
            cmap="viridis",
            origin="lower",
            aspect="equal",
            interpolation="none",
            vmin=0,
            vmax=np.percentile(image, 98),
        )
        a.set_title(
            f"{f.track.name} ({f.track.chip}), {f.track.length} slices, Bragg ratio {f.bragg:.1f}",
            fontsize=6,
        )
        a.tick_params(labelsize=5)
        a.set_xlabel("along track (pixels)", fontsize=6)
        a.set_ylabel("transverse", fontsize=6)

    result = aastex.Figure("gallery", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The eleven longest flat tracks and, in the last panel, the track with the
strongest Bragg rise, as the finder cut them out: seven pixels wide,
re-centered on the integer part of the fitted centerline in every slice,
and oriented with the back surface at the left.
Each panel's color scale is clipped at its 98th percentile so that a single
bright pixel does not hide the track.
The stopping track's charge per slice climbs by a factor of
twenty toward one end, and the selection of Section~\ref{sec:method}
rejects it."""))
    return result
