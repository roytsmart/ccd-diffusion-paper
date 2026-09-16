import matplotlib.pyplot as plt
import aastex
import named_arrays as na
import ccd_diffusion

__all__ = [
    "tracks",
]

_chips = {
    "SJI": dict(marker="o", color="black"),
    "FUV2": dict(marker="s", color="tab:blue"),
    "FUV1": dict(marker="^", color="tab:orange"),
}


def _example() -> "ccd_diffusion.tracks.Fit":
    """A well-constrained track on the SJI CCD to display as an example."""
    fits = [f for f in ccd_diffusion.tracks.fits() if f.track.chip == "SJI" and f.flat]
    fits = [f for f in fits if 18 <= f.track.length <= 24]
    return max(fits, key=lambda f: f.gain)


def tracks() -> aastex.Figure:
    """
    An example track and the same-column probability against depth for each
    CCD.
    """

    example = _example()
    track = example.track
    axis_slice = ccd_diffusion.tracks.axis_slice
    axis_pixel = ccd_diffusion.tracks.axis_pixel

    fig, ax = plt.subplots(
        nrows=2,
        figsize=(5, 4.5),
        gridspec_kw=dict(height_ratios=[1, 2.4]),
        constrained_layout=True,
    )
    ax_track, ax_profile = ax

    charge = track.charge
    if example.orientation < 0:
        charge = charge[{axis_slice: slice(None, None, -1)}]
    na.plt.imshow(
        charge / 1000,
        axis_x=axis_slice,
        axis_y=axis_pixel,
        ax=ax_track,
        cmap="gray_r",
        origin="lower",
        extent=[0, track.length, -3.5, 3.5],
        aspect="auto",
    )
    ax_track.set_xlabel("slice (back surface to front)")
    ax_track.set_ylabel("pixel")
    ax_track.set_yticks([-3, 0, 3])
    ax_track.set_title(
        f"track {track.name}, ${track.length}$ slices, "
        rf"$t_c={example.critical_depth:.2f}$, "
        rf"$\sigma_\mathrm{{max}}={example.width_max.value:.1f}$ $\mu$m",
        fontsize=8,
    )

    for chip, style in _chips.items():
        p = ccd_diffusion.tracks.profile(chip)
        ax_profile.errorbar(
            p.depth.ndarray,
            p.measured.ndarray,
            p.error.ndarray,
            linestyle="None",
            markersize=3,
            label=chip,
            **style,
        )
    p = ccd_diffusion.tracks.profile("SJI")
    na.plt.plot(
        p.depth,
        p.paper,
        ax=ax_profile,
        color="tab:red",
        label="field-free model",
    )
    na.plt.plot(
        p.depth,
        p.none,
        ax=ax_profile,
        color="gray",
        linestyle="--",
        label="no diffusion",
    )
    ax_profile.set_xlabel("fractional depth, $t = z / D$")
    ax_profile.set_ylabel(r"same-column probability, $\sum_j f_j^2$")
    ax_profile.set_xlim(0, 1)
    ax_profile.set_ylim(0.5, 1.02)
    ax_profile.legend(fontsize=7, loc="lower right")

    result = aastex.Figure("tracks", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
Top: a glancing particle track on the \IRIS\ \SJI\ \CCD, in thousands of
electrons per pixel, rotated so that it runs from the back surface (left)
to the front surface (right).
Bottom: the probability that two electrons deposited in the same slice are
collected in the same column, averaged over the flat tracks on each \CCD\
in bins of fractional depth, with the standard error of each mean.
The solid line is the prediction of the field-free model,
Equation~\ref{eq:width} with $t_c = \modelCriticalDepth$ and
$\sigma_\text{max} = \modelWidthMax$ $\mu$m, evaluated at the fitted
centerline of each \SJI\ track and averaged in the same bins, and the dashed
line is the same with no charge diffusion."""))
    return result
