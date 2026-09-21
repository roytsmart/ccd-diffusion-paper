import numpy as np
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
    "SJI": "black",
}

_markers = ["D", "s", "^", "v", "o"]
"""The marker for each dataset in the joint distribution."""


def core() -> list["ccd_diffusion.tracks.Fit"]:
    """The flat tracks whose fitted :math:`t_c` lies between 0.25 and 0.6."""
    return [
        f
        for f in ccd_diffusion.tracks.fits()
        if f.flat and 0.25 < f.critical_depth < 0.6
    ]


def parameters() -> aastex.Figure:
    """The distributions of the fitted parameters on each CCD and in each dataset."""
    tracks = ccd_diffusion.tracks
    tc_paper, sm_paper = tracks.paper_model()
    zf = sm_paper.to_value(u.um)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(
        ncols=3,
        figsize=(6.5, 2.3),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = ax

    ax_a.axvspan(0.25, 0.6, color="0.9")
    for chip, color in _chips.items():
        tc = [f.critical_depth for f in tracks.flat(chip)]
        ax_a.hist(
            tc,
            bins=np.arange(0, 1.06, 0.05),
            histtype="step",
            linewidth=1,
            color=color,
            label=f"{chip} ({len(tc)})",
        )
    ax_a.axvline(
        tc_paper, color="tab:red", linestyle="--", label=f"model, {tc_paper:.2f}"
    )
    ax_a.set_xlabel("$t_c$")
    ax_a.set_ylabel("tracks")
    ax_a.set_title("(a) flat tracks", fontsize=8)
    ax_a.legend(fontsize=5)

    for chip, color in _chips.items():
        sm = [f.width_max.to_value(u.um) for f in core() if f.track.chip == chip]
        ax_b.hist(
            sm,
            bins=np.arange(0, 10.1, 0.5),
            histtype="step",
            linewidth=1,
            color=color,
            label=f"{chip} ({len(sm)})",
        )
    ax_b.axvline(zf, color="tab:red", linestyle="--", label=f"model, {zf:.2f} $\\mu$m")
    ax_b.set_xlabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_b.set_ylabel("tracks")
    ax_b.set_title("(b) tracks with $0.25 < t_c < 0.6$", fontsize=8)
    ax_b.legend(fontsize=5)

    for chip, color in _chips.items():
        subset = [f for f in core() if f.track.chip == chip]
        tc = np.array([f.critical_depth for f in subset]) + rng.uniform(
            -0.02, 0.02, len(subset)
        )
        sm = np.array([f.width_max.to_value(u.um) for f in subset]) + rng.uniform(
            -0.2, 0.2, len(subset)
        )
        ax_c.scatter(tc, sm, s=3, color=color, alpha=0.4, linewidths=0)
    ax_c.plot(tc_paper, zf, "*", color="tab:red", markersize=9, label="model")
    import itertools

    for (dataset, info), marker in zip(
        tracks.datasets.items(), itertools.cycle(_markers)
    ):
        subset = [f for f in core() if f.track.dataset == dataset]
        if not subset:
            continue
        tc = np.array([f.critical_depth for f in subset])
        sm = np.array([f.width_max.to_value(u.um) for f in subset])
        ax_c.errorbar(
            tc.mean(),
            sm.mean(),
            xerr=tc.std() / np.sqrt(len(tc)),
            yerr=sm.std() / np.sqrt(len(sm)),
            fmt=marker,
            color="black",
            markerfacecolor="white",
            markersize=4,
            linewidth=0.8,
            label=f"{info['date']} ({info['image']})",
        )
    ax_c.set_xlim(0.2, 0.65)
    ax_c.set_ylim(0, 10.5)
    ax_c.set_xlabel("$t_c$")
    ax_c.set_ylabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_c.set_title("(c) joint distribution", fontsize=8)
    ax_c.legend(fontsize=5, loc="lower right")

    result = aastex.Figure("parameters", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The fitted parameters.
(a) The critical depth of every flat track on each \CCD, with the
field-free model dashed.
The tails below 0.25 and above 0.6 are tracks the model describes
poorly, sharp or diffuse along most of their length, and they are
excluded from the shaded core used elsewhere.
(b) The back-surface width of the core tracks.
(c) Each core track jittered off the fit grid, colored by \CCD\ as in (a),
with the mean of each dataset and its standard error as open symbols.
Two particle populations, two spacecraft rolls, and two cameras agree to
within a few hundredths in $t_c$, all within 0.04 of the model, and the
diagonal smear is the degeneracy between $t_c$ and $\sigma_\text{max}$ in
a single-track fit."""))
    return result
