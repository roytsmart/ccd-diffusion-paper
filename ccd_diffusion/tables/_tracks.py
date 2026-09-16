import astropy.units as u
import pylatex
import ccd_diffusion

__all__ = [
    "num_frames",
    "tracks",
]

_chips = ["FUV1", "FUV2", "SJI"]


def num_frames(chip: str) -> tuple[int, int]:
    """
    The number of frames searched for tracks on the given CCD, and the number
    of them taken inside the SAA.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2`` or ``SJI``.
    """
    image = "FUV" if chip.startswith("FUV") else "SJI"
    frames = [f for f in ccd_diffusion.tracks.frames() if f["image"].startswith(image)]
    return len(frames), sum(f["saa"] == "1" for f in frames)


def tracks() -> pylatex.Table:
    """
    The charge diffusion measured on each CCD, one row per CCD.
    """
    result = pylatex.Table(position="htb!")
    result.escape = False
    result.append(pylatex.Command("centering"))
    result.add_caption(pylatex.NoEscape(r"""
The charge diffusion measured from glancing particle tracks on the three
\IRIS\ \CCD{}s.
The number of frames is the number of level-1 images searched, with the
number taken inside \SAA\ in parentheses.
The flat tracks are those which constrain $t_c$ to within 0.15 and show no
Bragg rise along their length.
$t_c$ and $\sigma_\text{max}$ are the medians of the per-track fits, with the
interquartile range in brackets.
$p$ is the probability that two electrons deposited within $D / 10$ of the
back surface are collected in the same column, $\mathcal{P} = p^2$ is the
corresponding probability that they are collected in the same pixel, and the
last column is $\mathcal{P}$ predicted by the field-free model for the same
tracks."""))
    result.append(pylatex.Command("label", "tab:tracks"))

    tabular = pylatex.Tabular("lrrrccccc", booktabs=False)
    tabular.escape = False
    tabular.add_hline()
    tabular.add_row(
        [
            pylatex.NoEscape(s)
            for s in (
                r"\CCD",
                "frames",
                "tracks",
                "flat",
                "$t_c$",
                r"$\sigma_\text{max}$ ($\mu$m)",
                "$p$",
                r"$\mathcal{P}$",
                r"$\mathcal{P}_\text{model}$",
            )
        ]
    )
    tabular.add_hline()
    for chip in _chips:
        s = ccd_diffusion.tracks.summary(chip)
        n_frames, n_saa = num_frames(chip)
        tc = s.critical_depth
        sm = s.width_max.to_value(u.um)
        row = [
            chip,
            f"{n_frames} ({n_saa})",
            f"{s.num_tracks}",
            f"{s.num_flat}",
            f"{tc[1]:.2f} [{tc[0]:.2f}, {tc[2]:.2f}]",
            f"{sm[1]:.1f} [{sm[0]:.1f}, {sm[2]:.1f}]",
            f"${s.same_pixel_1d:.3f} \\pm {s.same_pixel_1d_error:.3f}$",
            f"${s.same_pixel:.3f} \\pm {s.same_pixel_error:.3f}$",
            f"{s.same_pixel_paper:.3f}",
        ]
        tabular.add_row([pylatex.NoEscape(c) for c in row])
    tabular.add_hline()

    result.append(tabular)
    return result
