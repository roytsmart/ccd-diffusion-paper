import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import astropy.units as u
import aastex
import ccd_diffusion

__all__ = [
    "core",
    "parameters",
]

_chips = {
    "FUV1": "tab:orange",
    "FUV2": "tab:blue",
    "NUV": "tab:green",
    "SJI": "black",
}

_enclosed = (0.5, 0.9)
"""The fractions of a CCD's core tracks the contours of the joint distribution enclose."""


def core() -> list["ccd_diffusion.tracks.Fit"]:
    """The flat tracks whose fitted :math:`t_c` lies between 0.25 and 0.6."""
    return [
        f
        for f in ccd_diffusion.tracks.fits()
        if f.flat and 0.25 < f.critical_depth < 0.6
    ]


def _levels(density: np.ndarray, fractions: tuple[float, ...]) -> list[float]:
    """The density levels whose contours enclose the given fractions of the total."""
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    return sorted(float(flat[np.searchsorted(cumulative, f)]) for f in fractions)


def parameters() -> aastex.Figure:
    """The distributions of the fitted parameters on each CCD and in each dataset."""
    tracks = ccd_diffusion.tracks

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.5),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = ax

    # the fits are exhaustive searches on grids, so every histogram is
    # binned on the grid cells, one step wide and centered on the grid
    # points, and the joint distribution is
    # a histogram on the grid cells, smoothed by a cell, and drawn as the
    # contours enclosing half and nine tenths of each CCD's core tracks
    grid_tc = tracks.critical_depth.ndarray
    grid_sm = tracks.width_max.ndarray.to_value(u.um)
    step_tc = float(np.diff(grid_tc)[0])
    step_sm = float(np.diff(grid_sm)[0])
    edges_tc = np.concatenate([grid_tc - step_tc / 2, [grid_tc[-1] + step_tc / 2]])
    edges_sm = np.concatenate([grid_sm - step_sm / 2, [grid_sm[-1] + step_sm / 2]])
    ax_a.axvspan(0.25, 0.6, color="0.9")
    for chip, color in _chips.items():
        tc = [f.critical_depth for f in tracks.flat(chip)]
        if not tc:
            continue
        num_core = sum(1 for f in core() if f.track.chip == chip)
        ax_a.hist(
            tc,
            bins=edges_tc,
            histtype="step",
            linewidth=1,
            color=color,
            label=f"{chip} ({len(tc)} flat, {num_core} core)",
        )
    ax_a.set_xlabel("$t_c$")
    ax_a.set_ylabel("tracks")
    ax_a.set_title("(a) flat tracks", fontsize=8)

    for chip, color in _chips.items():
        sm = [f.width_max.to_value(u.um) for f in core() if f.track.chip == chip]
        if not sm:
            continue
        ax_b.hist(
            sm,
            bins=edges_sm,
            histtype="step",
            linewidth=1,
            color=color,
        )
    ax_b.set_xlabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_b.set_ylabel("tracks")
    ax_b.set_title("(b) core tracks, $0.25 < t_c < 0.6$", fontsize=8)

    for chip, color in _chips.items():
        subset = [f for f in core() if f.track.chip == chip]
        if not subset:
            continue
        tc = np.array([f.critical_depth for f in subset])
        sm = np.array([f.width_max.to_value(u.um) for f in subset])
        density, _, _ = np.histogram2d(tc, sm, bins=[edges_tc, edges_sm])
        density = scipy.ndimage.gaussian_filter(density, 1)
        ax_c.contour(
            grid_tc,
            grid_sm,
            density.T,
            levels=_levels(density, _enclosed),
            colors=[color],
            linewidths=[0.6, 1.2],
        )
    campaigns = 0
    for dataset in tracks.datasets:
        subset = [f for f in core() if f.track.dataset == dataset]
        if not subset:
            continue
        campaigns += 1
        tc = np.array([f.critical_depth for f in subset])
        sm = np.array([f.width_max.to_value(u.um) for f in subset])
        ax_c.errorbar(
            tc.mean(),
            sm.mean(),
            xerr=tc.std() / np.sqrt(len(tc)),
            yerr=sm.std() / np.sqrt(len(sm)),
            fmt="o",
            color="black",
            markerfacecolor="white",
            markersize=3.5,
            linewidth=0.6,
            label="campaign means" if campaigns == 1 else None,
        )
    ax_c.set_xlim(0.2, 0.65)
    ax_c.set_ylim(0, 10.5)
    ax_c.set_xlabel("$t_c$")
    ax_c.set_ylabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_c.set_title("(c) core tracks, jointly", fontsize=8)

    handles, labels = ax_a.get_legend_handles_labels()
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    fig.legend(
        handles + handles_c,
        labels + labels_c,
        loc="outside lower center",
        ncol=3,
        fontsize=6,
    )

    result = aastex.Figure("parameters", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The fitted parameters.
(a) The critical depth of every flat track on each \CCD.
The tails below 0.25 and above 0.6 are tracks Equation~\ref{eq:width}
describes poorly, sharp or diffuse along most of their length, and they
are excluded from the shaded core used elsewhere.
(b) The back-surface width of the core tracks.
(c) The joint distribution of the core tracks on each \CCD, as the
contours enclosing half (thick) and nine tenths (thin) of them, on the
grid the fits are searched over (0.05 in $t_c$ and 0.5 $\mu$m in
$\sigma_\text{max}$) smoothed by one cell, with the mean of each campaign
of Table~\ref{tab:datasets} and its standard error as an open circle.
The campaigns, spanning two particle populations, four spacecraft rolls,
and the spectrograph and slit-jaw \CCD{}s, agree to within a few
hundredths in $t_c$, and the contours run wider in $\sigma_\text{max}$
than in $t_c$, since a single track constrains its back-surface width
less well than the depth at which it sharpens."""))
    return result
