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

    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        d = tracks.depleted(chip)
        sd = d.width_depleted.ndarray.to_value(u.um)
        ax_a.plot(
            sd,
            d.misfit.ndarray,
            color=color,
            linewidth=0.8,
            label=f"{chip} ({d.num})",
        )
        for a, m, e in (
            (
                ax_b,
                d.width_max_mean.ndarray.to_value(u.um),
                d.width_max_error.ndarray.to_value(u.um),
            ),
            (ax_c, d.critical_depth_mean.ndarray, d.critical_depth_error.ndarray),
        ):
            a.plot(sd, m, color=color, linewidth=0.8)
            a.fill_between(sd, m - e, m + e, color=color, alpha=0.2, linewidth=0)

    for a in ax:
        a.set_xlabel(r"$\sigma_d$ ($\mu$m)")
        a.set_xlim(-0.05, 1.55)
    ax_a.set_ylabel("pooled misfit above minimum")
    ax_a.set_title(r"(a) pooled misfit against $\sigma_d$", fontsize=8)
    ax_a.legend(fontsize=5)
    ax_b.set_ylabel(r"mean $\sigma_\mathrm{max}$ ($\mu$m)")
    ax_b.set_title(r"(b) back-surface width", fontsize=8)
    ax_c.set_ylabel("mean $t_c$")
    ax_c.set_title(r"(c) critical depth", fontsize=8)

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
(b) and (c) The mean $\sigma_\text{max}$ and $t_c$ of the same tracks at
each $\sigma_d$, with the standard error of the mean shaded: the spread
inside the depletion region trades against the field-free wedge, so
$t_c$ falls steadily as $\sigma_d$ rises, fastest on \FUV{}1, while
$\sigma_\text{max}$ rises by a few percent."""))
    return result
