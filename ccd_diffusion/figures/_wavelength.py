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

_wavelength = na.geomspace(0.15, 1000, axis="wavelength", num=1001) * u.nm
"""The wavelengths plotted, the grid of the companion noise article."""


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
    The diffusion width of a photon against its wavelength, from the measured
    width profile of each CCD and the absorption length of silicon.
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

    nm = _wavelength.ndarray.value
    energy = _wavelength.to(u.eV, equivalencies=u.spectral()).ndarray.value

    fig, ax = plt.subplots(
        figsize=(6.5, 3.2),
        constrained_layout=True,
    )
    for chip, color in _chips.items():
        if not tracks.flat(chip):
            continue
        w = tracks.widths(chip)
        sigma = _interpolated(z / D, w.depth.ndarray, w.best.ndarray.to_value(u.um))
        rms = np.sqrt(average(np.square(sigma)))
        ax.plot(nm, rms, color=color, label=chip)
    ax_length = ax.twinx()
    ax_length.plot(nm, length.ndarray.value, color="gray", linestyle=":", linewidth=0.8)
    ax_length.set_yscale("log")
    ax_length.set_ylim(1e-3, 1e3)
    ax_length.set_ylabel(r"absorption length ($\mu$m), dotted", color="gray")
    ax_length.tick_params(axis="y", colors="gray")
    # the photon energy along the top, as in the companion article
    ax_energy = ax.twiny()
    ax_energy.plot(energy, np.full_like(energy, np.nan))
    ax_energy.set_xscale("log")
    ax_energy.set_xlim(energy[0], energy[-1])
    ax_energy.set_xlabel("energy (eV)")
    for name, (lo, hi) in _bands.items():
        ax.axvspan(lo.value, hi.value, color="0.9", zorder=0)
        ax.text(
            np.sqrt(lo.value * hi.value),
            0.15,
            name,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6,
            color="0.4",
        )
    ax.set_xscale("log")
    ax.set_xlim(nm[0], nm[-1])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylim(0, 6)
    ax.set_ylabel(r"diffusion width $\sigma$ ($\mu$m)")
    ax.legend(fontsize=6, loc="lower left", ncol=2)

    result = aastex.Figure("wavelength", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
The width of the charge cloud a photon leaves in each \CCD, against the
wavelength of the photon.
The width is the standard deviation of the cloud.
A photon is absorbed at a random depth, exponentially distributed with the
absorption length of silicon (dotted, from the tabulated optical constants
\cite{Palik1985,Henke1993}), and the charge it liberates spreads by the
width measured at that depth in Figure~\ref{fig:stacked}b, held at the
value of its first bin up to the back surface.
The curve is the standard deviation of the cloud that results, which is the
square root of the squared width averaged over depth, weighted by the
fraction of photons absorbed at each depth; photons that pass through the
\thickness\ $\mu$m of silicon are not counted.
Between 30 and 350 nm every photon is absorbed within 10 nm of the back
surface, and the cloud has the back-surface width of
Table~\ref{tab:tracks}.
In the visible and in the soft X-rays the photons penetrate to the
depletion region and the cloud narrows.
Near the band gap silicon is nearly transparent, and the few photons
absorbed are spread through its thickness."""))
    return result
