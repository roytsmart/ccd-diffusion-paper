import dataclasses
import functools
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import ccd_diffusion
from ._tracks import axis_pixel, half_width
from ._slab import axis_mixture
from ._fit import Fit, fits, cloud, fractions

__all__ = [
    "axis_depth",
    "depth_bins",
    "depth_back",
    "same_pixel",
    "same_pixel_model",
    "same_pixel_ratio",
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
    same pixel column predicted by the model of the given fit for its
    best-fit centerline.

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
    w, weight = cloud(fit.depth, critical_depth, width_max, width_depleted, fit.model)
    q = fractions(fit.position, w, fit.track.slope, weight, axis_mixture)
    return np.square(q).sum(axis_pixel)


_landing = np.linspace(-0.5, 0.5, 101)
"""Where in its pixel a photon lands, in pixels from the center of the pixel."""

_edges_wide = np.arange(-half_width - 3, half_width + 4) + 0.5
"""The boundaries of the pixels around the one a photon lands in, wide enough to hold its cloud."""


def same_pixel_ratio(
    critical_depth: float,
    width_max: u.Quantity,
    width_depleted: u.Quantity = 0 * u.um,
    model: str = "slab",
    depth: float = depth_back / 2,
) -> float:
    """
    The probability that two electrons of a photon absorbed at the given
    depth are collected in the same pixel, divided by the square of the
    probability that they are collected in the same column, for a photon
    landing anywhere in its pixel.

    The cloud of a Gaussian model is the product of its profiles along the
    two axes, so the ratio is one.
    The cloud of the slab model is not: given the time an electron takes to
    reach the depletion region its offsets along the two axes are
    independent Gaussians, but a slow electron spreads along both, so the
    cloud is a mixture of such products, from which each electron is drawn
    independently.
    For Gaussians :math:`k` and :math:`l` of the mixture with weights
    :math:`w_k` and :math:`w_l`, let :math:`S_{kl}` be the probability that
    two electrons drawn from them are collected in the same column; then
    the probability for the same column is :math:`p = \\sum_{kl} w_k w_l
    S_{kl}`, and that for the same pixel is :math:`\\sum_{kl} w_k w_l
    S_{kl}^2`, which is never less than :math:`p^2`.

    Parameters
    ----------
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface.
    width_depleted
        The spread acquired crossing the depletion region.
    model
        One of :data:`ccd_diffusion.tracks.models`.
    depth
        The fractional depth at which the photon is absorbed.
    """
    width, weight = cloud(
        na.ScalarArray(depth),
        critical_depth,
        width_max,
        width_depleted,
        model,
    )
    w = np.maximum(width.ndarray, 1e-3)
    weight = np.broadcast_to(weight.ndarray, w.shape)
    x = _edges_wide[np.newaxis, np.newaxis, :] - _landing[np.newaxis, :, np.newaxis]
    cdf = scipy.special.erf(x / (np.sqrt(2) * w[:, np.newaxis, np.newaxis])) / 2
    q = np.diff(cdf, axis=-1)
    overlap = np.einsum("kpj,lpj->kl", q, q) / _landing.size
    column = weight @ overlap @ weight
    pixel = weight @ np.square(overlap) @ weight
    return float(pixel / np.square(column))


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
    """
    The measured same-pixel probability in each bin, :attr:`none` minus the
    median over the slices of the bin of the shortfall of each slice from
    its own no-diffusion prediction.

    Charge from other hits in the frame that touched a track and was cut
    out with it can only land off the peak of the slice, so it can only
    lower :math:`\\sum_j f_j^2`, while the read noise scatters it both
    ways. The median of the shortfall is unmoved by such one-sided
    contamination until it reaches half the slices, where the mean is
    pulled down by every contaminated slice in proportion to its charge.
    """

    error: na.AbstractScalarArray
    """
    The standard error of :attr:`measured`, that of a median from the
    median absolute deviation of the shortfall.
    """

    mean: na.AbstractScalarArray
    """The plain mean measured same-pixel probability in each bin."""

    error_mean: na.AbstractScalarArray
    """The standard error of :attr:`mean`."""

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


def _binned_median(depth: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    The median of the values in each depth bin and its standard error,
    :math:`\\sqrt{\\pi / 2}` times that of the mean with the scatter taken
    from the median absolute deviation, which the contaminated slices do not
    inflate.
    """
    keep = np.isfinite(values)
    depth, values = depth[keep], values[keep]
    edges = depth_bins.ndarray
    index = np.clip(np.digitize(depth, edges) - 1, 0, len(edges) - 2)
    median = np.empty(len(edges) - 1)
    error = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        v = values[index == i]
        median[i] = np.median(v)
        scatter = 1.4826 * np.median(np.abs(v - median[i]))
        error[i] = np.sqrt(np.pi / 2) * scatter / np.sqrt(v.size)
    return median, error


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
    measured = np.concatenate(measured)
    fitted = np.concatenate(fitted)
    none = np.concatenate(none)

    def bin(values):
        return na.ScalarArray(_binned(depth, values)[1], axes=axis_depth)

    num, mean, error_mean = _binned(depth, measured)
    # the shortfall of each slice from its own no-diffusion prediction, so
    # that the median is not spread by the geometry of the centerline
    shortfall, error = _binned_median(depth, none - measured)
    edges = depth_bins.ndarray
    return Profile(
        chip=chip,
        depth=na.ScalarArray((edges[:-1] + edges[1:]) / 2, axes=axis_depth),
        num=na.ScalarArray(num, axes=axis_depth),
        measured=na.ScalarArray(_binned(depth, none)[1] - shortfall, axes=axis_depth),
        error=na.ScalarArray(error, axes=axis_depth),
        mean=na.ScalarArray(mean, axes=axis_depth),
        error_mean=na.ScalarArray(error_mean, axes=axis_depth),
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

    thickness: u.Quantity
    """
    The thickness of the sensor implied by the fits of the slab model, whose
    cloud at the back surface is as wide as the field-free layer is thick:
    the mean :math:`\\sigma_\\text{max}` of the flat tracks divided by their
    mean :math:`t_c`.

    The ratio of each track alone lies on the lattice of the ratios of the
    grid points, so the ratio of the means is the steadier estimate.
    """

    thickness_error: u.Quantity
    """The standard error of :attr:`thickness`, from those of the two means."""

    same_pixel_1d: float
    """The measured same-pixel probability at the back surface along one axis."""

    same_pixel_1d_error: float
    """The standard error of :attr:`same_pixel_1d`."""

    same_pixel_ratio: float
    """
    The same-pixel probability in two dimensions over the square of that
    along one axis, from :func:`same_pixel_ratio` for the median fit of the
    slab model at the back surface.
    """

    @property
    def same_pixel(self) -> float:
        """
        The measured same-pixel probability at the back surface in two
        dimensions, the square of :attr:`same_pixel_1d` times
        :attr:`same_pixel_ratio`.
        """
        return self.same_pixel_1d**2 * self.same_pixel_ratio

    @property
    def same_pixel_error(self) -> float:
        """The standard error of :attr:`same_pixel`."""
        return 2 * self.same_pixel_1d * self.same_pixel_1d_error * self.same_pixel_ratio


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
    thickness = sm.mean() / tc.mean()
    thickness_error = (
        thickness
        * np.hypot(
            sm.std() / sm.mean(),
            tc.std() / tc.mean(),
        )
        / np.sqrt(len(flat))
    )

    width_depleted = ccd_diffusion.tracks.depleted(chip).best
    critical_depth = tuple(np.percentile(tc, [25, 50, 75]))
    width_max = np.percentile(sm, [25, 50, 75])

    return Summary(
        chip=chip,
        num_tracks=len(all_fits),
        num_flat=len(flat),
        width_depleted=width_depleted,
        critical_depth=critical_depth,
        critical_depth_error=float(tc.std() / np.sqrt(len(tc))),
        width_max=width_max,
        thickness=thickness,
        thickness_error=thickness_error,
        same_pixel_1d=float(measured[back].mean()),
        same_pixel_1d_error=float(measured[back].std() / np.sqrt(back.sum())),
        same_pixel_ratio=same_pixel_ratio(
            critical_depth[1], width_max[1], width_depleted
        ),
    )
