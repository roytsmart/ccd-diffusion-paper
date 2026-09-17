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
    "SJI": "black",
}


def depleted() -> aastex.Figure:
    """The fit allowing an extra spread inside the depletion region."""
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
        d = tracks.depleted(chip)
        sd = d.width_depleted.ndarray.to_value(u.um)
        preferred[chip] = d.preferred.to_value(u.um)
        ax_a.plot(
            sd,
            d.misfit.ndarray,
            "o-",
            color=color,
            markersize=2.5,
            linewidth=0.8,
            label=f"{chip} ({len(d.preferred)})",
        )
        ax_b.plot(
            sd,
            d.width_max.ndarray.to_value(u.um),
            "s--",
            color=color,
            markersize=2.5,
            linewidth=0.8,
            label=rf"{chip} $\sigma_\mathrm{{max}}$",
        )
        ax_tc.plot(
            sd,
            d.critical_depth.ndarray,
            "o-",
            color=color,
            markersize=2.5,
            linewidth=0.8,
            label=f"{chip} $t_c$",
        )
    ax_c.hist(
        list(preferred.values()),
        bins=np.arange(-0.25, 3.3, 0.5),
        color=list(_chips.values()),
        label=list(_chips),
        weights=[np.ones(len(p)) / len(p) for p in preferred.values()],
    )

    ax_a.set_xlabel(r"$\sigma_d$ ($\mu$m)")
    ax_a.set_ylabel("pooled misfit above minimum")
    ax_a.set_title(r"(a) profile in $\sigma_d$", fontsize=8)
    ax_a.legend(fontsize=5)

    ax_b.set_xlabel(r"$\sigma_d$ ($\mu$m)")
    ax_b.set_ylabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_b.set_ylim(3, 8)
    ax_tc.set_ylabel("$t_c$")
    ax_tc.set_ylim(0.2, 0.6)
    ax_b.set_title(
        r"(b) best $t_c$, $\sigma_\mathrm{max}$ at each $\sigma_d$", fontsize=8
    )
    handles, labels = ax_b.get_legend_handles_labels()
    handles_tc, labels_tc = ax_tc.get_legend_handles_labels()
    ax_tc.legend(
        handles + handles_tc, labels + labels_tc, fontsize=5, loc="upper center", ncol=2
    )

    ax_c.set_xlabel(r"per-track best $\sigma_d$ ($\mu$m)")
    ax_c.set_ylabel("fraction of tracks")
    ax_c.set_title("(c) per-track preference", fontsize=8)
    ax_c.legend(fontsize=5)

    result = aastex.Figure("depleted", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
Allowing diffusion inside the depletion region, Equation~\ref{eq:depleted}.
(a) The misfit pooled over the flat tracks on each \CCD, minimized over
$t_c$ and $\sigma_\text{max}$ at each $\sigma_d$ and shown relative to its
minimum, which lies at $\sigma_d = \widthDepleted$ $\mu$m on every \CCD.
(b) The pooled best-fit $t_c$ and $\sigma_\text{max}$ at each $\sigma_d$:
adding the extra spread lowers $t_c$ and raises $\sigma_\text{max}$, since
some of what the two-parameter fit attributed to the field-free layer was
this floor.
(c) The $\sigma_d$ preferred by each track on its own, which is spread over
0 to 1 $\mu$m: the effect is modest for any one track and consistent across
them."""))
    return result
