import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import aastex
import ccd_diffusion

__all__ = [
    "image",
]

_stretch = 10
"""The scale of the arcsinh stretch, in data numbers above the background."""

_vmax = 400
"""The data number above the background shown as black."""

"""The track shown at full resolution."""

_half_zoom = 40
"""The half-width of the region shown around that track, in pixels."""


def _window(
    data: np.ndarray, columns: None | slice = None
) -> tuple[np.ndarray, slice, slice]:
    """
    The part of a level-1 image which was read out, since the rest of the
    frame is filled with zeros, restricted to the given columns.
    """
    valid = data > 0
    if columns is not None:
        mask = np.zeros_like(valid)
        mask[:, columns] = True
        valid = valid & mask
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    rows = slice(rows.min(), rows.max() + 1)
    cols = slice(cols.min(), cols.max() + 1)
    return data[rows, cols], rows, cols


def _show(ax, data: np.ndarray, rows: slice, cols: slice):
    """Display part of an image with the arcsinh stretch, unread pixels white."""
    background = np.median(data[data > 0])
    stretched = np.arcsinh((data - background) / _stretch)
    stretched[data <= 0] = np.nan
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad("white")
    ax.imshow(
        stretched,
        cmap=cmap,
        origin="lower",
        vmin=0,
        vmax=np.arcsinh(_vmax / _stretch),
        extent=[cols.start - 0.5, cols.stop - 0.5, rows.start - 0.5, rows.stop - 0.5],
        interpolation="none",
    )


def _boxes(ax, tracks, pad: float = 8, linewidth: float = 0.6):
    for track in tracks:
        x, y, w, h = track.extent
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x - pad, y - pad),
                w + 2 * pad,
                h + 2 * pad,
                fill=False,
                edgecolor="tab:red",
                linewidth=linewidth,
            )
        )


def image() -> aastex.Figure:
    """
    A level-1 image from each camera, taken during the same pass through the
    South Atlantic Anomaly, with the tracks the finder extracted outlined.
    """
    images = {im.image: im for im in ccd_diffusion.tracks.images()}
    sji = images["SJI_2796"]
    fuv = images["FUV"]

    data_sji, rows_sji, cols_sji = _window(sji.data.ndarray)
    data_fuv, rows_fuv, cols_fuv = _window(fuv.data.ndarray, columns=slice(3500, None))
    zoom = max(sji.tracks, key=lambda t: t.length)  # the longest track in the frame
    x, y, w, h = zoom.extent
    cx, cy = x + w / 2, y + h / 2
    rows_zoom = slice(int(cy - _half_zoom), int(cy + _half_zoom))
    cols_zoom = slice(int(cx - _half_zoom), int(cx + _half_zoom))
    data_zoom = sji.data.ndarray[rows_zoom, cols_zoom]

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.9),
        gridspec_kw=dict(width_ratios=[data_sji.shape[1], data_fuv.shape[1], 360]),
        constrained_layout=True,
    )
    ax_sji, ax_fuv, ax_zoom = ax

    _show(ax_sji, data_sji, rows_sji, cols_sji)
    _boxes(ax_sji, sji.tracks)
    _boxes(ax_sji, [zoom], pad=_half_zoom, linewidth=1)
    ax_sji.set_title(
        f"(a) slit-jaw image, 2796 \\AA, {len(sji.tracks)} tracks", fontsize=8
    )
    ax_sji.set_xlabel("column")
    ax_sji.set_ylabel("row")

    _show(ax_fuv, data_fuv, rows_fuv, cols_fuv)
    _boxes(ax_fuv, fuv.tracks)
    n_fuv2 = sum(t.chip == "FUV2" for t in fuv.tracks)
    ax_fuv.set_title(f"(b) spectrograph, FUV2 window, {n_fuv2} tracks", fontsize=8)
    ax_fuv.set_xlabel("column")

    _show(ax_zoom, data_zoom, rows_zoom, cols_zoom)
    _boxes(ax_zoom, [zoom], pad=2, linewidth=0.8)
    ax_zoom.set_title(f"(c) track {zoom.name}", fontsize=8)
    ax_zoom.set_xlabel("column")

    result = aastex.Figure("image", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
Level-1 images taken seconds apart during a pass through \SAA\ on
2018 May 4, with the spacecraft
rolled by $-90^\circ$ so that the limb runs horizontally across the field.
(a) The slit-jaw image at 2796 \AA, whose lower half is off the limb.
(b) The widest of the readout windows of the spectrograph image, on the
\FUV{}2 \CCD.
(c) One of the tracks in (a) at full resolution.
The images are shown with an inverse hyperbolic sine stretch above the
background, saturating at \imageVmax\ data numbers, and the red boxes
outline the tracks the finder kept from each frame.
Most of the particle hits are the short streaks and dots that pepper both
images; the tracks used here are the long, thin, nearly vertical ones."""))
    return result
