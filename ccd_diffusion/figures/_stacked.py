import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import astropy.units as u
import aastex
import ccd_diffusion

__all__ = [
    "stacked",
]

_chips = {
    "FUV1": "tab:orange",
    "FUV2": "tab:blue",
    "NUV": "tab:green",
    "SJI": "black",
}


def stacked() -> aastex.Figure:
    """
    The flat tracks stacked on their fitted centerlines, and the diffusion
    width against depth measured without a parametric model beside the
    per-track fits.
    """
    tracks = ccd_diffusion.tracks

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.5),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = ax

    s = tracks.stack("FUV2")
    mappable = ax_a.imshow(
        s.image.ndarray.T,
        origin="lower",
        cmap="gray_r",
        # the wings carry a few percent of the charge, so a square-root
        # stretch shows them beside the core
        norm=matplotlib.colors.PowerNorm(gamma=0.5),
        aspect="auto",
        extent=[0, 1, s.offset.ndarray[0], s.offset.ndarray[-1]],
        interpolation="none",
    )
    fig.colorbar(mappable, ax=ax_a, label="charge fraction per pixel")
    ax_a.set_xlabel("$t = z / D$")
    ax_a.set_ylabel("offset from fitted line (pixels)")
    ax_a.set_title(f"(a) {len(tracks.flat('FUV2'))} FUV2 tracks stacked", fontsize=8)

    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        w = tracks.widths(chip)
        best = w.best.ndarray.to_value(u.um)
        yerr = np.stack(
            [
                best - w.lower.ndarray.to_value(u.um),
                w.upper.ndarray.to_value(u.um) - best,
            ]
        )
        ax_b.errorbar(
            w.depth.ndarray,
            best,
            yerr,
            fmt="o",
            color=color,
            markersize=2,
            linewidth=0.8,
            label=f"{chip}, {len(tracks.flat(chip))} tracks",
            zorder=3,
        )
        ax_b.plot(
            w.depth.ndarray,
            w.fitted.ndarray.to_value(u.um),
            color=color,
            linewidth=0.8,
        )
    ax_b.set_xlabel("$t = z / D$")
    ax_b.set_ylabel(r"diffusion width $\sigma$ ($\mu$m)")
    ax_b.set_ylim(0, 8)
    ax_b.set_title(r"(b) $\sigma(t)$ from the slices and the fits", fontsize=8)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6)

    k = tracks.kernel("FUV2")
    distance = k.distance.ndarray
    centers = (distance[:-1] + distance[1:]) / 2
    half = tracks.width_depth_kernel / 2
    colors = plt.cm.viridis(np.linspace(0, 0.85, k.depth.size))
    for i, color in enumerate(colors):
        index = {tracks.axis_depth: i}
        t = float(k.depth[index].ndarray)
        ax_c.stairs(
            k.measured[index].ndarray,
            distance,
            color=color,
            linewidth=0.8,
            label=f"$t = {t:g}$",
            zorder=3,
        )
        ax_c.plot(
            centers,
            k.model[index].ndarray,
            color=color,
            linewidth=0.8,
            linestyle="--",
        )
        # the depth bin marked along the top of the stacked image
        ax_a.axvspan(t - half, t + half, ymin=0.96, color=color, linewidth=0)
    ax_c.set_xlim(-2, 2)
    # headroom above the peaks for the legend
    ax_c.set_ylim(-0.03, 1.4)
    ax_c.set_yticks([0, 0.5, 1])
    ax_c.set_xlabel("offset (pixels)")
    ax_c.set_ylabel("charge fraction in pixel")
    ax_c.set_title("(c) FUV2 at four depths", fontsize=8)
    ax_c.legend(fontsize=6, loc="upper center", ncol=2)

    result = aastex.Figure("stacked", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
(a) Every slice of every flat track on the \FUV{}2 \CCD\ placed at its
fractional depth and aligned on its fitted centerline, with each pixel's
charge spread uniformly over its own width.
The wedge is subtle: the wings at $t < 0.4$ carry a few percent of the
charge.
(b) Points: the diffusion width in each depth bin on each \CCD, found by
summing the misfit of Equation~\ref{eq:misfit} over every slice in the
bin on a grid of trial widths and minimizing, with no parametric model of
the depth dependence; the error bars span the widths within two of the
minimum $M$.
Lines: the average of the per-track fits of Equation~\ref{eq:width} on the
same \CCD, which reproduces the model-free profile at every depth,
including the floor of a few tenths of a micron beyond $t_c$ that the
$\sigma_d$ term supplies.
(c) The kernel itself, integrated over a pixel.
Steps: the mean fraction of the charge of the \FUV{}2 slices in four depth
bins, each a tenth of the thickness wide and marked along the top of (a),
collected in a pixel whose center lies at the given offset from the fitted
centerline.
Dashed lines: the same mean of the fractions the per-track fits predict
for the same pixels.
The measured peak falls a few hundredths below the fits at every depth,
including the depletion region where the fits are pixel-sharp, and the
measured wings stand above them: this is the stray charge that separates
the mean from the median in Figure~\ref{fig:profile}, which the robust
misfit ignores and the mean includes."""))
    return result
