import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import aastex
import ccd_diffusion

__all__ = [
    "models",
]

_chips = {
    "FUV1": "tab:orange",
    "FUV2": "tab:blue",
    "NUV": "tab:green",
    "SJI": "black",
}

_fractions_profile = (0, 0.5, 0.9)
"""The depths of the profiles of panel (a), as fractions of the field-free layer."""


def _density(
    y: na.AbstractScalarArray,
    width: na.AbstractScalarArray,
    weight: float | na.AbstractScalarArray = 1,
    axis: None | str = None,
) -> na.AbstractScalarArray:
    """
    The density of a Gaussian, or of a mixture of Gaussians along ``axis``,
    at the given positions.

    Parameters
    ----------
    y
        The positions.
    width
        The standard deviation of the Gaussian, or of each Gaussian of the
        mixture.
    weight
        The weight of each Gaussian of the mixture.
    axis
        The logical axis of the Gaussians of a mixture.
    """
    result = np.exp(-np.square(y / width) / 2) / (np.sqrt(2 * np.pi) * width)
    if axis is not None:
        result = (weight * result).sum(axis)
    return result


def models() -> aastex.Figure:
    """The slab model of the charge cloud against the Gaussian it replaces."""
    tracks = ccd_diffusion.tracks

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.5),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = ax

    # (a) the cloud at three depths for the median fit of the FUV2 tracks,
    # beside the Gaussian of the same variance
    typical = tracks.summary("FUV2")
    tc = typical.critical_depth[1]
    sm = typical.width_max[1]
    sd = typical.width_depleted
    y = na.linspace(-2.5, 2.5, axis="y", num=1001)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(_fractions_profile)))
    for x, color in zip(_fractions_profile, colors):
        depth = na.ScalarArray(x * tc)
        width, weight = tracks.cloud(depth, tc, sm, sd)
        slab = _density(y, width, weight, tracks.axis_mixture)
        gaussian = _density(y, tracks.width(depth, tc, sm, sd))
        ax_a.plot(y.ndarray, slab.ndarray, color=color, label=f"$t / t_c = {x:g}$")
        ax_a.plot(y.ndarray, gaussian.ndarray, color=color, linestyle="--")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-4, 30)
    ax_a.set_xlim(-2.5, 2.5)
    ax_a.set_xlabel("offset (pixels)")
    ax_a.set_ylabel("charge per pixel of offset")
    ax_a.set_title("(a) the cloud, slab and Gaussian", fontsize=8)
    ax_a.legend(fontsize=5, loc="upper right")

    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        c = tracks.comparison(chip)
        depth = c.depth.ndarray
        ax_b.plot(
            depth,
            c.misfit_depth.ndarray / c.num,
            color=color,
            linewidth=0.8,
            label=chip,
        )
        for model, linestyle in (("gaussian", "--"), ("slab", "-")):
            ax_c.plot(
                depth,
                c.residual[model].ndarray.to_value(u.um),
                color=color,
                linewidth=0.8,
                linestyle=linestyle,
            )
    ax_b.axhline(0, color="0.6", linewidth=0.5)
    ax_b.set_xlabel("$t = z / D$")
    ax_b.set_ylabel("misfit per track, slab minus Gaussian", fontsize=7)
    ax_b.set_title("(b) where the slab fits better", fontsize=8)
    ax_c.axhline(0, color="0.6", linewidth=0.5)
    ax_c.set_xlabel("$t = z / D$")
    ax_c.set_ylabel(r"model-free minus fitted width ($\mu$m)", fontsize=7)
    ax_c.set_title("(c) residual widths", fontsize=8)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6)

    result = aastex.Figure("models", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The slab model of Equation~\ref{eq:width} against the Gaussian model of
Equation~\ref{eq:gaussian}.
(a) The charge cloud along one axis at three depths in the field-free
layer, for the median fit of the \FUV{}2 tracks and its $\sigma_d$ (solid),
beside a Gaussian of the same variance (dashed).
At the back surface the slab's cloud falls off exponentially rather than
as a Gaussian, and near the depletion region it has a narrow core and
broad wings.
(b) The misfit of the slab model minus that of the Gaussian, per track,
summed over the slices of each depth bin of the tracks flat under both
models, each model at its own $\sigma_d$; below zero the slab model fits
better.
(c) The model-free width of Figure~\ref{fig:stacked}b minus that found the
same way in the fractions the fits predict, for the slab model (solid) and
the Gaussian (dashed)."""))
    return result
