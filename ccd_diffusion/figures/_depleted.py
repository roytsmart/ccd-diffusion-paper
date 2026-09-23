import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import aastex
import ccd_diffusion

__all__ = [
    "depleted",
]

_chips = {
    "FUV1": "tab:orange",
    "FUV2": "tab:blue",
    "NUV": "tab:green",
    "SJI": "black",
}


def depleted() -> aastex.Figure:
    """The fit of the spread inside the depletion region, pooled over each CCD."""
    tracks = ccd_diffusion.tracks

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.3),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = ax
    ax_tc = ax_b.twinx()

    preferred = {}
    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        d = tracks.depleted(chip)
        sd = d.width_depleted.ndarray.to_value(u.um)
        preferred[chip] = [
            f.width_depleted_preferred.to_value(u.um) for f in tracks.flat(chip)
        ]
        ax_a.plot(
            sd,
            d.misfit.ndarray,
            color=color,
            linewidth=0.8,
            label=f"{chip} ({d.num})",
        )
        ax_b.plot(
            sd,
            d.width_max.ndarray.to_value(u.um),
            color=color,
            linewidth=0.8,
        )
        ax_tc.plot(
            sd,
            d.critical_depth.ndarray,
            color=color,
            linestyle="--",
            linewidth=0.8,
        )
    ax_c.hist(
        list(preferred.values()),
        bins=np.arange(-0.05, 1.6, 0.1),
        color=[_chips[c] for c in preferred],
        label=list(preferred),
        weights=[np.ones(len(p)) / len(p) for p in preferred.values()],
    )

    ax_a.set_xlabel(r"$\sigma_d$ ($\mu$m)")
    ax_a.set_ylabel("pooled misfit above minimum")
    ax_a.set_xlim(-0.05, 1.55)
    ax_a.set_title(r"(a) pooled misfit against $\sigma_d$", fontsize=8)
    ax_a.legend(fontsize=5)

    ax_b.set_xlabel(r"$\sigma_d$ ($\mu$m)")
    ax_b.set_ylabel(r"median $\sigma_\mathrm{max}$ ($\mu$m)")
    ax_b.set_ylim(3, 8)
    ax_tc.set_ylabel("median $t_c$")
    ax_tc.set_ylim(0.2, 0.6)
    ax_b.set_title(r"(b) per-track fits at each $\sigma_d$", fontsize=8)
    ax_tc.legend(
        handles=[
            plt.Line2D([], [], color="0.4", linewidth=0.8),
            plt.Line2D([], [], color="0.4", linewidth=0.8, linestyle="--"),
        ],
        labels=[r"$\sigma_\mathrm{max}$", "$t_c$"],
        fontsize=5,
        loc="upper center",
        ncol=2,
    )

    ax_c.set_xlabel(r"per-track best $\sigma_d$ ($\mu$m)")
    ax_c.set_ylabel("fraction of tracks")
    ax_c.set_title("(c) per-track preference", fontsize=8)
    ax_c.legend(fontsize=5)

    result = aastex.Figure("depleted", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The spread inside the depletion region, $\sigma_d$ in
Equation~\ref{eq:width}, which is shared by every track on a \CCD.
(a) The misfit summed over the flat tracks on each \CCD\ at each
$\sigma_d$, with each track refit in $t_c$, $\sigma_\text{max}$,
orientation, and centerline, shown relative to its minimum, which lies at
$\sigma_d = \widthDepletedFuvOne$ $\mu$m on \FUV{}1, \widthDepletedFuvTwo\
$\mu$m on \FUV{}2, \widthDepletedNuv\ $\mu$m on \NUV, and
\widthDepletedSji\ $\mu$m on \SJI.
(b) The median $t_c$ and $\sigma_\text{max}$ of the same tracks at each
$\sigma_d$: the spread inside the depletion region trades against the
field-free wedge, stepping $t_c$ down while $\sigma_\text{max}$ holds.
(c) The $\sigma_d$ each flat track prefers on its own.
The preference of any one track is weak, spread over the whole grid, and
it is only in the sum that the minimum is sharp."""))
    return result
