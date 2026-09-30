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

_cameras = ("FUV", "NUV")
"""
The cameras whose frames are drawn: the spectrograph, whose rows and
columns run along the slit and along the dispersion.
"""

_rolls = {
    0: "tab:orange",
    -75: "tab:green",
    90: "tab:purple",
    -90: "tab:blue",
}
"""The spacecraft rolls of the campaigns, in degrees, and their colors."""

_bin = 10.0
"""The width of the azimuth bins in degrees."""


def _census() -> list["ccd_diffusion.tracks.Component"]:
    """The elongated components of the spectrograph frames drawn in the figure."""
    tracks = ccd_diffusion.tracks
    camera = {(f["dataset"], int(f["fsn"])): f["image"] for f in tracks.frames()}
    return [
        c
        for c in tracks.load_census()
        if c.width < _width_maximum
        and c.length > _aspect_minimum * c.width
        and camera.get((c.dataset, c.fsn)) in _cameras
    ]


def _accepted_isotropic() -> float:
    """The fraction of an isotropic population within the finder's windows."""
    return 4 * np.degrees(np.arctan(slope_maximum)) / 180


def _accepted(roll: int) -> float:
    """
    The fraction of the components inside the anomaly at the given roll
    whose azimuth lies within the finder's windows about either axis.
    """
    window = np.degrees(np.arctan(slope_maximum))
    a = np.array(
        [c.azimuth for c in _census() if c.saa and datasets[c.dataset]["roll"] == roll]
    )
    slit = np.minimum(a, 180 - a) < window
    dispersion = np.abs(a - 90) < window
    return float(np.mean(slit | dispersion))


def azimuth() -> aastex.Figure:
    """The direction of every elongated particle hit in the spectrograph frames."""
    census = _census()
    edges = np.arange(0, 360 + _bin, _bin)
    window = np.degrees(np.arctan(slope_maximum))

    fig = plt.figure(figsize=(6.5, 3.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.2, wspace=0.2)
    ax = [fig.add_subplot(1, 2, i + 1, projection="polar") for i in range(2)]
    for a, (title, saa) in zip(
        ax, (("inside the anomaly", True), ("quiet frames", False))
    ):
        for roll, color in _rolls.items():
            mine = [
                c
                for c in census
                if c.saa == saa and datasets[c.dataset]["roll"] == roll
            ]
            if not mine:
                continue
            num = len({c.dataset for c in mine})
            azimuths = [c.azimuth for c in mine]
            # the azimuth is measured from the slit axis, and a track has no
            # direction, so each is drawn at its azimuth and its opposite
            theta = np.concatenate([azimuths, np.add(azimuths, 180)]) % 360
            count, _ = np.histogram(theta, bins=edges)
            fraction = 100 * count / len(mine)
            # the histogram drawn as a closed outline, so the rolls overlap
            # without hiding one another
            a.plot(
                np.radians(np.repeat(edges, 2)[1:-1]),
                np.repeat(fraction, 2),
                color=color,
                linewidth=1,
                label=(
                    f"roll {roll}$^\\circ$, {num} campaign{'s' if num > 1 else ''} "
                    f"(n={len(mine)})"
                ),
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
above $5\sigma$ in the spectrograph frames of every campaign, with a
charge-weighted width under 3 pixels and an aspect ratio over 4, before any
of the finder's cuts, so that the whole circle is populated, grouped by the
roll of the spacecraft.
Tracks are undirected, so each is drawn at $\theta$ and $\theta + 180^\circ$,
and the grey wedges are the windows about the row and column axes that the
finder accepts.
Left: the frames taken inside \SAA.
Right: quiet frames.
The trapped protons arrive strongly aligned, along the slit at rolls of
$0^\circ$ and $-90^\circ$, along the dispersion at $90^\circ$, and between
the two at $-75^\circ$, so the preferred direction is a property of the
arrival direction as seen by the spacecraft rather than of the detector.
The cosmic rays of the quiet frames at roll $0^\circ$ are close to
isotropic, while the quiet frames of the one campaign at $-90^\circ$, taken
at the edge of the anomaly, keep some of its alignment."""))
    return result
