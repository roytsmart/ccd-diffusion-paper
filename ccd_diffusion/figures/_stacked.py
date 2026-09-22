import numpy as np
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
    tc_paper, sm_paper = tracks.paper_model()
    D = ccd_diffusion.ccd().thickness_substrate.to_value(u.um)

    fig, ax = plt.subplots(
        ncols=2,
        figsize=(5.5, 2.7),
        constrained_layout=True,
    )
    ax_a, ax_b = ax

    s = tracks.stack("FUV2")
    mappable = ax_a.imshow(
        s.image.ndarray.T,
        origin="lower",
        cmap="gray_r",
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
    reference = tracks.widths("SJI")
    ax_b.plot(
        reference.depth.ndarray,
        reference.model.ndarray.to_value(u.um),
        color="tab:red",
        linestyle="--",
        label=rf"model, $z_f={sm_paper.to_value(u.um):.2f}$ $\mu$m, $D={D:.0f}$ $\mu$m",
    )
    ax_b.axvline(tc_paper, color="gray", linestyle=":", linewidth=0.8)
    ax_b.set_xlabel("$t = z / D$")
    ax_b.set_ylabel(r"diffusion width $\sigma$ ($\mu$m)")
    ax_b.set_ylim(0, 8)
    ax_b.set_title(r"(b) $\sigma(t)$ from the slices and the fits", fontsize=8)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6)

    result = aastex.Figure("stacked", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
(a) Every slice of every flat track on the \FUV{}2 \CCD\ placed at its
fractional depth and aligned on its fitted centerline, with each pixel's
charge spread uniformly over its own width.
The wedge is subtle: the wings at $t < 0.4$ carry a few percent of the
charge.
(b) Points: the diffusion width in each depth bin on each \CCD, found by
summing the misfit of every slice in the bin over a grid of trial widths
and minimizing, with no parametric model of the depth dependence; the
error bars span the widths within two units of misfit of the minimum.
Lines: the average of the per-track fits of Equation~\ref{eq:width} on the
same \CCD, which reproduces the model-free profile at every depth,
including the floor of 0.5 to 1 $\mu$m beyond $t_c$ (dotted) that the
$\sigma_d$ term supplies and the dashed field-free model sets to zero."""))
    return result
