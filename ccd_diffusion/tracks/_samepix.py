import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
import ccd_diffusion
from ._tracks import axis_pixel, half_width
from ._fit import Fit, fits, width, fractions

__all__ = [
    "axis_depth",
    "depth_bins",
    "depth_back",
    "same_pixel",
    "same_pixel_model",
    "Profile",
    "profile",
    "Summary",
    "summary",
]

axis_depth = "depth"
"""The logical axis of the binned depth profile."""

depth_bins = na.linspace(0, 1, axis=axis_depth, num=11)
"""The edges of the depth bins used by :func:`profile`."""

depth_back = 0.1
"""Slices shallower than this fractional depth are considered to be at the back surface."""


def same_pixel(fit: Fit) -> na.AbstractScalarArray:
    """
    The measured probability that two charges from the same slice
    are collected in the same pixel column, :math:`\\sum_j f_j^2`,
    corrected for the read-noise bias :math:`\\sum_j \\epsilon_j^2`, and
    NaN for a slice too faint to be :attr:`Track.usable`.

    Parameters
    ----------
    fit
        The fit whose track to evaluate.
    """
    track = fit.track
    f = np.square(track.fraction).sum(axis_pixel)
    e = np.square(track.error) * (2 * half_width + 1)
    return np.where(track.usable, f - e, np.nan)


def same_pixel_model(
    fit: Fit,
    critical_depth: float,
    width_max: u.Quantity,
    width_depleted: u.Quantity = 0 * u.um,
) -> na.AbstractScalarArray:
    """
    The probability that two charges from the same slice are collected in the
    same pixel column predicted by the diffusion model for the best-fit
    centerline of the given track.

    Parameters
    ----------
    fit
        The fit whose best-fit centerline to use.
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface.
    width_depleted
        The spread acquired crossing the depletion region.
    """
    w = width(fit.depth, critical_depth, width_max, width_depleted)
    q = fractions(fit.position, w, fit.track.slope)
    return np.square(q).sum(axis_pixel)


@dataclasses.dataclass(eq=False)
class Profile:
    """The same-pixel probability as a function of depth for one CCD."""

    chip: str
    """The CCD the profile was measured on."""

    depth: na.AbstractScalarArray
    """The center of each depth bin."""

    num: na.AbstractScalarArray
    """The number of slices in each depth bin."""

    measured: na.AbstractScalarArray
    """The mean measured same-pixel probability in each bin."""

    error: na.AbstractScalarArray
    """The standard error of :attr:`measured`."""

    fitted: na.AbstractScalarArray
    """The mean same-pixel probability predicted by the per-track fits, including :math:`\\sigma_d`."""

    none: na.AbstractScalarArray
    """The mean same-pixel probability predicted with no charge diffusion."""


def _binned(depth: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
    keep = np.isfinite(values)
    depth, values = depth[keep], values[keep]
    edges = depth_bins.ndarray
    index = np.clip(np.digitize(depth, edges) - 1, 0, len(edges) - 2)
    num = np.bincount(index, minlength=len(edges) - 1)
    total = np.bincount(index, weights=values, minlength=len(edges) - 1)
    square = np.bincount(index, weights=np.square(values), minlength=len(edges) - 1)
    mean = total / num
    variance = square / num - np.square(mean)
    error = np.sqrt(variance / num)
    return num, mean, error


@functools.cache
def profile(chip: str) -> Profile:
    """
    Bin the measured and modeled same-pixel probabilities of every flat track
    on the given CCD by depth.

    Parameters
    ----------
    chip
        The CCD to summarize, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    depth = []
    measured = []
    fitted = []
    none = []
    for f in fits():
        if f.track.chip != chip or not f.flat:
            continue
        depth.append(f.depth.ndarray)
        measured.append(same_pixel(f).ndarray)
        fitted.append(
            same_pixel_model(f, f.critical_depth, f.width_max, f.width_depleted).ndarray
        )
        none.append(same_pixel_model(f, 0, 0 * u.um).ndarray)
    depth = np.concatenate(depth)

    def bin(values):
        return na.ScalarArray(
            _binned(depth, np.concatenate(values))[1], axes=axis_depth
        )

    num, mean, error = _binned(depth, np.concatenate(measured))
    edges = depth_bins.ndarray
    return Profile(
        chip=chip,
        depth=na.ScalarArray((edges[:-1] + edges[1:]) / 2, axes=axis_depth),
        num=na.ScalarArray(num, axes=axis_depth),
        measured=na.ScalarArray(mean, axes=axis_depth),
        error=na.ScalarArray(error, axes=axis_depth),
        fitted=bin(fitted),
        none=bin(none),
    )


@dataclasses.dataclass(eq=False)
class Summary:
    """The diffusion measurement for one CCD."""

    chip: str
    """The CCD that was measured."""

    num_tracks: int
    """The number of tracks found on this CCD."""

    num_flat: int
    """The number of tracks that pass the :attr:`Fit.flat` cut."""

    width_depleted: u.Quantity
    """The spread inside the depletion region fit to the CCD as a whole."""

    critical_depth: tuple[float, float, float]
    """The 25th, 50th and 75th percentiles of the fitted :math:`t_c`."""

    critical_depth_error: float
    """The standard error of the mean fitted :math:`t_c`."""

    width_max: u.Quantity
    """The 25th, 50th and 75th percentiles of the fitted :math:`\\sigma_\\text{max}`."""

    same_pixel_1d: float
    """The measured same-pixel probability at the back surface along one axis."""

    same_pixel_1d_error: float
    """The standard error of :attr:`same_pixel_1d`."""

    @property
    def same_pixel(self) -> float:
        """The measured same-pixel probability at the back surface in two dimensions."""
        return self.same_pixel_1d**2

    @property
    def same_pixel_error(self) -> float:
        """The standard error of :attr:`same_pixel`."""
        return 2 * self.same_pixel_1d * self.same_pixel_1d_error


@functools.cache
def summary(chip: str) -> Summary:
    """
    Summarize the diffusion measurement on the given CCD.

    Parameters
    ----------
    chip
        The CCD to summarize, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    all_fits = [f for f in fits() if f.track.chip == chip]
    flat = [f for f in all_fits if f.flat]

    depth = np.concatenate([f.depth.ndarray for f in flat])
    measured = np.concatenate([same_pixel(f).ndarray for f in flat])
    back = (depth < depth_back) & np.isfinite(measured)

    tc = np.array([f.critical_depth for f in flat])
    sm = u.Quantity([f.width_max for f in flat])

    return Summary(
        chip=chip,
        num_tracks=len(all_fits),
        num_flat=len(flat),
        width_depleted=ccd_diffusion.tracks.depleted(chip).best,
        critical_depth=tuple(np.percentile(tc, [25, 50, 75])),
        critical_depth_error=float(tc.std() / np.sqrt(len(tc))),
        width_max=np.percentile(sm, [25, 50, 75]),
        same_pixel_1d=float(measured[back].mean()),
        same_pixel_1d_error=float(measured[back].std() / np.sqrt(back.sum())),
    )
