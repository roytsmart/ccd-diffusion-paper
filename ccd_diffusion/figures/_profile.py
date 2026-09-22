import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import aastex
import ccd_diffusion

__all__ = [
    "profile",
]

_chips = ["FUV1", "FUV2", "NUV", "SJI"]


def profile() -> aastex.Figure:
    """The same-column probability against depth on each CCD."""
    tracks = ccd_diffusion.tracks
    tc_paper, sm_paper = tracks.paper_model()

    chips = [c for c in _chips if tracks.flat(c)]
    fig, ax = plt.subplots(
        ncols=len(chips),
        figsize=(6.5, 2.3),
        sharey=True,
        constrained_layout=True,
    )
    for a, chip in zip(ax, chips):
        p = tracks.profile(chip)
        na.plt.plot(
            p.depth, p.none, ax=a, color="gray", linestyle="--", label="no diffusion"
        )
        na.plt.plot(
            p.depth,
            p.paper,
            ax=a,
            color="tab:red",
            label=rf"model, $z_f = {sm_paper.to_value(u.um):.2f}$ $\mu$m",
        )
        na.plt.plot(p.depth, p.fitted, ax=a, color="tab:blue", label="per-track fits")
        a.errorbar(
            p.depth.ndarray,
            p.measured.ndarray,
            p.error.ndarray,
            fmt="o",
            color="black",
            markersize=2.5,
            linewidth=0.8,
            label="tracks",
        )
        a.set_xlabel("$t = z / D$")
        a.set_title(f"{chip}, {tracks.summary(chip).num_flat} flat tracks", fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"same-column probability, $\sum_j f_j^2$")
    ax[0].set_ylim(0.4, 1.02)
    ax[0].legend(fontsize=5, loc="lower right")

    result = aastex.Figure("profile", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The probability that two electrons deposited in the same slice are
collected in the same column, averaged over the flat tracks on each \CCD\
in bins of fractional depth, with the standard error of each mean.
The red line is the field-free model, Equation~\ref{eq:width} with
$t_c = \modelCriticalDepth$ and $\sigma_\text{max} = \modelWidthMax$
$\mu$m, evaluated at the fitted centerline of each track and averaged in
the same bins; the blue line is the same for each track's own fit; and the
dashed line is the same with no charge diffusion, which is below one only
where a centerline runs near a pixel boundary.
The back surface agrees with the model on the \FUV{}2 and \SJI\ \CCD{}s,
while beyond $t_c$ the tracks settle below the no-diffusion curve on all
three."""))
    return result
