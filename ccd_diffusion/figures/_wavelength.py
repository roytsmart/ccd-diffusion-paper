import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
import aastex
import ccd_diffusion

__all__ = [
    "wavelength",
]

_chips = {
    "FUV1": "tab:orange",
    "FUV2": "tab:blue",
    "NUV": "tab:green",
    "SJI": "black",
}

_bands = {
    "IRIS FUV": (133, 141) * u.nm,
    "IRIS NUV": (278, 284) * u.nm,
}
"""The passbands of IRIS, shaded for reference."""

_wavelength = na.geomspace(10, 1100, axis="wavelength", num=400) * u.nm
"""The wavelengths plotted, up to the band gap of silicon."""


def _interpolated(
    depth: np.ndarray,
    centers: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    A measured profile, given at the centers of its depth bins, at every
    depth, linear between the centers and held at the first and last values
    beyond them.
    """
    return np.interp(depth, centers, values)


def wavelength() -> aastex.Figure:
    """
    The diffusion width and the same-pixel probability of a photon against
    its wavelength, from the measured profiles of each CCD and the absorption
    length of silicon.
    """
    tracks = ccd_diffusion.tracks
    D = tracks.thickness.to_value(u.um)

    # the depth at which a photon is absorbed, on a grid fine enough for the
    # nanometer absorption lengths of the ultraviolet
    z = np.concatenate([[0], np.geomspace(1e-4, D, 3000)])
    length = 1 / optika.chemicals.Chemical("Si").absorption(_wavelength)
    length = length.to(u.um)
    alpha = 1 / length.ndarray.value[:, np.newaxis]
    # the distribution of absorption depth among the photons absorbed within
    # the sensor; the rest pass through and are not counted
    pdf = alpha * np.exp(-alpha * z) / (1 - np.exp(-alpha * D))

    def average(values: np.ndarray) -> np.ndarray:
        return scipy.integrate.trapezoid(pdf * values, z, axis=1)

    fig, ax = plt.subplots(
        ncols=2,
        figsize=(6.5, 2.7),
        constrained_layout=True,
    )
    ax_a, ax_b = ax
    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        w = tracks.widths(chip)
        sigma = _interpolated(z / D, w.depth.ndarray, w.best.ndarray.to_value(u.um))
        rms = np.sqrt(average(np.square(sigma)))
        p = tracks.profile(chip)
        same = _interpolated(z / D, p.depth.ndarray, np.square(p.measured.ndarray))
        same = average(same)
        ax_a.plot(_wavelength.ndarray.value, rms, color=color, label=chip)
        ax_b.plot(_wavelength.ndarray.value, same, color=color, label=chip)
    ax_length = ax_a.twinx()
    ax_length.plot(
        _wavelength.ndarray.value,
        length.ndarray.value,
        color="gray",
        linestyle=":",
        linewidth=0.8,
    )
    ax_length.set_yscale("log")
    ax_length.set_ylim(1e-3, 1e3)
    ax_length.set_ylabel(r"absorption length ($\mu$m), dotted", color="gray")
    ax_length.tick_params(axis="y", colors="gray")
    for name, (lo, hi) in _bands.items():
        for a in (ax_a, ax_b):
            a.axvspan(lo.value, hi.value, color="0.9", zorder=0)
        ax_a.text(
            np.sqrt(lo.value * hi.value),
            0.15,
            name,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6,
            color="0.4",
        )
    for a in (ax_a, ax_b):
        a.set_xscale("log")
        a.set_xlim(_wavelength.ndarray.value[0], _wavelength.ndarray.value[-1])
        a.set_xlabel("wavelength (nm)")
    ax_a.set_ylim(0, 6)
    ax_a.set_ylabel(r"rms diffusion width ($\mu$m)")
    ax_a.set_title("(a) the charge cloud of a photon", fontsize=8)
    ax_b.set_ylim(0, 1)
    ax_b.set_ylabel(r"same-pixel probability $\mathcal{P}$")
    ax_b.set_title("(b) two electrons of one photon", fontsize=8)
    ax_b.legend(fontsize=6, loc="upper left")

    result = aastex.Figure("wavelength", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The measurement as an instrument designer meets it, against the wavelength
of the photon.
A photon is absorbed at a depth drawn from the exponential with the
absorption length of silicon (dotted, from the tabulated optical constants
\cite{Palik1985,Henke1993}), among the photons absorbed within the
\thickness\ $\mu$m thickness, and the measured profiles are averaged over
that depth.
(a) The root mean square of the model-free width of
Figure~\ref{fig:stacked}b.
(b) The square of the same-column probability of Figure~\ref{fig:profile},
the probability that two electrons of the same photon are collected in the
same \pixelPitch\ $\mu$m pixel, which enters the variance of the image.
Each profile is held at the value of its first bin up to the back surface.
Below 350 nm the absorption length is under 10 nm, so the whole
ultraviolet sees the back-surface values of Table~\ref{tab:tracks};
through the visible the photons reach the depletion region and the cloud
narrows; and near the band gap the sensor is nearly transparent and the
few photons absorbed are spread through its thickness."""))
    return result
