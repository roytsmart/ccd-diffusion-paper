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
    "NUV": "tab:green",
    "SJI": "black",
}


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
        if not tc:
            continue
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
        if not sm:
            continue
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

    # the fits are exhaustive searches on grids, 0.05 in t_c and 0.5 um in
    # sigma_max, so each track is spread uniformly over its grid cell; a
    # narrower jitter leaves the grid showing through as gaps
    step_tc = float(np.diff(tracks.critical_depth.ndarray)[0])
    step_sm = float(np.diff(tracks.width_max.ndarray.to_value(u.um))[0])
    for chip, color in _chips.items():
        subset = [f for f in core() if f.track.chip == chip]
        tc = np.array([f.critical_depth for f in subset]) + rng.uniform(
            -step_tc / 2, step_tc / 2, len(subset)
        )
        sm = np.array([f.width_max.to_value(u.um) for f in subset]) + rng.uniform(
            -step_sm / 2, step_sm / 2, len(subset)
        )
        ax_c.scatter(tc, sm, s=3, color=color, alpha=0.4, linewidths=0)
    ax_c.plot(tc_paper, zf, "*", color="tab:red", markersize=9, label="model")
    # one open circle per campaign; with twenty of them a legend naming each
    # would cover the panel, and the caption says what they are
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
            markersize=4,
            linewidth=0.8,
            label=f"campaign means ({campaigns})" if campaigns == 1 else None,
        )
    handles, labels = ax_c.get_legend_handles_labels()
    labels = [f"campaign means ({campaigns})" if "campaign" in x else x for x in labels]
    ax_c.set_xlim(0.2, 0.65)
    ax_c.set_ylim(0, 10.5)
    ax_c.set_xlabel("$t_c$")
    ax_c.set_ylabel(r"$\sigma_\mathrm{max}$ ($\mu$m)")
    ax_c.set_title("(c) joint distribution", fontsize=8)
    ax_c.legend(handles, labels, fontsize=5, loc="lower right")

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
(c) Each core track, colored by \CCD\ as in (a) and spread uniformly over
its cell of the fit grid (0.05 in $t_c$ and 0.5 $\mu$m in
$\sigma_\text{max}$), with the mean of each campaign of
Table~\ref{tab:datasets} and its standard error as an open circle.
The campaigns, spanning two particle populations, four spacecraft rolls,
and three cameras, agree to within a few hundredths in $t_c$, all within
0.04 of the model, and the diagonal smear is the degeneracy between $t_c$
and $\sigma_\text{max}$ in a single-track fit."""))
    return result
