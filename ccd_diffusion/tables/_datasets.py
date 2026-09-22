import numpy as np
import astropy.units as u
import pylatex
import ccd_diffusion

__all__ = [
    "datasets",
]

_cameras = {"FUV": "F", "NUV": "N", "SJI": "S"}
"""One letter per camera, for the cameras column of the table."""


def datasets() -> pylatex.Table:
    """The observing campaigns searched for tracks, one row per campaign."""
    tracks = ccd_diffusion.tracks
    frames = tracks.frames()
    fits = tracks.fits()

    result = pylatex.Table(position="htb!")
    result.escape = False
    result.append(pylatex.Command("centering"))
    result.add_caption(pylatex.NoEscape(r"""
The observing campaigns searched for tracks, with the cameras kept from
each (F for the far-ultraviolet spectrograph, N for the near-ultraviolet
one, S for the slit-jaw imager), the spacecraft roll, the exposure time,
and the number of frames searched, with the number taken inside \SAA\ in
parentheses.
The tracks of every campaign but the two marked $\dagger$, whose frames
were taken outside \SAA, are left by trapped protons; those two hold
cosmic rays.
The core tracks are the flat tracks with $0.25 < t_c < 0.6$, over which the
mean $t_c$ with its standard error and the median $\sigma_\text{max}$ with
its interquartile range are taken."""))
    result.append(pylatex.Command("label", "tab:datasets"))
    result.append(pylatex.Command("footnotesize"))
    # ten columns at the default column padding run past the text width
    result.append(pylatex.NoEscape(r"\setlength{\tabcolsep}{3.5pt}"))

    tabular = pylatex.Tabular("llrrrrrrcc", booktabs=False)
    tabular.escape = False
    tabular.add_hline()
    tabular.add_row(
        [
            pylatex.NoEscape(s)
            for s in (
                "date",
                "cameras",
                "roll",
                "exp. (s)",
                "frames",
                "tracks",
                "flat",
                "core",
                "$t_c$",
                r"$\sigma_\text{max}$ ($\mu$m)",
            )
        ]
    )
    tabular.add_hline()
    for name, info in tracks.datasets.items():
        mine = [f for f in frames if f["dataset"] == name]
        subset = [f for f in fits if f.track.dataset == name]
        core = [f for f in subset if f.flat and 0.25 < f.critical_depth < 0.6]
        if not core:
            continue
        tc = np.array([f.critical_depth for f in core])
        sm = np.percentile([f.width_max.to_value(u.um) for f in core], [25, 50, 75])
        date = info["date"]
        if info["particles"] == "cosmic rays":
            date += r"$^\dagger$"
        cameras = "".join(
            _cameras[c.strip()] for c in info["image"].split(",") if c.strip()
        )
        row = [
            date,
            cameras,
            f"${info['roll']}^\\circ$",
            f"{info['exposure']}",
            f"{len(mine)} ({sum(f['saa'] == '1' for f in mine)})",
            f"{len(subset)}",
            f"{sum(f.flat for f in subset)}",
            f"{len(core)}",
            f"${tc.mean():.3f} \\pm {tc.std() / np.sqrt(len(tc)):.3f}$",
            f"{sm[1]:.1f} [{sm[0]:.1f}, {sm[2]:.1f}]",
        ]
        tabular.add_row([pylatex.NoEscape(c) for c in row])
    tabular.add_hline()

    result.append(tabular)
    return result
