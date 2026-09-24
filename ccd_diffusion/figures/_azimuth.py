import numpy as np
import matplotlib.pyplot as plt
import aastex
import ccd_diffusion
from ccd_diffusion.tracks import slope_maximum, datasets

__all__ = [
    "azimuth",
]

_width_maximum = 3.0
"""The widest component drawn, four times its rms width across, in pixels."""

_aspect_minimum = 4.0
"""The least ratio of length to width of a component drawn."""

_panels = {
    "inside the anomaly": (("2014b", True), ("2018may", True)),
    "quiet frames": (("2018", False), ("2018may", False)),
}
"""The campaigns drawn in each panel, with whether their frames were inside the anomaly."""

_colors = {
    ("2014b", True): "tab:orange",
    ("2018may", True): "tab:blue",
    ("2018", False): "tab:gray",
    ("2018may", False): "tab:cyan",
}

_bin = 10.0
"""The width of the azimuth bins in degrees."""


def azimuth() -> aastex.Figure:
    """The direction of every elongated particle hit in the spectrograph frames."""
    census = ccd_diffusion.tracks.load_census()
    census = [
        c
        for c in census
        if c.width < _width_maximum and c.length > _aspect_minimum * c.width
    ]
    edges = np.arange(0, 360 + _bin, _bin)
    centers = np.radians((edges[:-1] + edges[1:]) / 2)
    window = np.degrees(np.arctan(slope_maximum))

    fig = plt.figure(figsize=(6.5, 3.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.17, wspace=0.2)
    ax = [fig.add_subplot(1, 2, i + 1, projection="polar") for i in range(2)]
    for a, (title, keys) in zip(ax, _panels.items()):
        for key in keys:
            dataset, saa = key
            mine = [c.azimuth for c in census if c.dataset == dataset and c.saa == saa]
            # the azimuth is measured from the slit axis, and a track has no
            # direction, so each is drawn at its azimuth and its opposite
            theta = np.concatenate([mine, np.add(mine, 180)]) % 360
            count, _ = np.histogram(theta, bins=edges)
            fraction = 100 * count / len(mine)
            info = datasets[dataset]
            kind = "SAA" if saa else "quiet"
            label = f"{kind}, {info['date'][:10]}, roll {info['roll']} (n={len(mine)})"
            a.bar(
                centers,
                fraction,
                width=np.radians(_bin),
                color=_colors[key],
                alpha=0.6,
                label=label,
            )
        for axis in (0, 90, 180, 270):
            a.bar(
                np.radians(axis),
                30,
                width=np.radians(2 * window),
                color="0.9",
                zorder=0,
            )
        a.set_theta_zero_location("N")
        a.set_theta_direction(-1)
        a.set_ylim(0, 25)
        a.set_yticks([10, 20])
        percent = r"\%" if plt.rcParams["text.usetex"] else "%"
        a.set_yticklabels([f"10{percent}", f"20{percent}"], fontsize=6)
        a.set_rlabel_position(45)
        a.set_xticks([])
        a.grid(alpha=0.4)
        a.set_title(title, fontsize=8, pad=16)
        a.text(0, 25.5, "along slit", ha="center", va="bottom", fontsize=7)
        a.text(
            np.radians(90),
            23.8,
            "along dispersion",
            ha="right",
            va="center",
            fontsize=7,
        )
        a.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.02))

    result = aastex.Figure("azimuth", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The azimuth in the detector frame of every elongated connected component
above $5\sigma$ in the spectrograph frames, with a charge-weighted width
under 3 pixels and an aspect ratio over 4, before any of the finder's cuts,
so that the whole circle is populated.
Tracks are undirected, so each is drawn at $\theta$ and $\theta + 180^\circ$,
and the grey wedges are the windows about the row and column axes that the
finder accepts.
Left: the frames taken inside \SAA\ at both spacecraft rolls.
Right: quiet frames at the same rolls.
The trapped protons arrive strongly aligned, mostly within the window
along the slit, while the cosmic rays are isotropic to within counting
noise; the preferred direction shifts and broadens when the spacecraft is
rolled, so it is a property of the arrival direction as seen by the
instrument rather than of the detector."""))
    return result
