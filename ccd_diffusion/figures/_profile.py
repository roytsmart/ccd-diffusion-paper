import matplotlib.pyplot as plt
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
        na.plt.plot(p.depth, p.fitted, ax=a, color="tab:blue", label="per-track fits")
        a.errorbar(
            p.depth.ndarray,
            p.mean.ndarray,
            p.error_mean.ndarray,
            fmt="o",
            color="0.6",
            markerfacecolor="white",
            markersize=2.5,
            linewidth=0.8,
            label="tracks, mean",
        )
        a.errorbar(
            p.depth.ndarray,
            p.measured.ndarray,
            p.error.ndarray,
            fmt="o",
            color="black",
            markersize=2.5,
            linewidth=0.8,
            label="tracks, median",
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
collected in the same column, over the flat tracks on each \CCD\ in bins
of fractional depth.
The dashed line is Equation~\ref{eq:width} with no charge diffusion,
averaged over the slices of each bin, which is below one only where a
centerline runs near a pixel boundary, and the blue line is the same at
each track's own fit.
The black points are the dashed line minus the median, over the slices
of the bin, of how far each slice falls short of its own no-diffusion
value, with the standard error of the median; the open points are the
plain mean of the slices.
Charge from other hits in the frame that touched a track and was cut out
with it can only lower the sum, so the mean sits below the median
wherever such charge is common, beyond $t_c$ on all four \CCD{}s, while
at the back surface the two agree.
The back surface is lowest on the \FUV{}2 and \SJI\ \CCD{}s and highest
on \FUV{}1."""))
    return result
