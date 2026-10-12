import csv
import dataclasses
import functools
import pathlib
import concurrent.futures
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import ccd_diffusion
from ._tracks import (
    axis_slice,
    axis_pixel,
    half_width,
    width_pixel,
    thickness,
    Track,
    load,
    _directory_data,
)
from ._slab import axis_mixture, transit, _misfit, _erf_table

__all__ = [
    "models",
    "axis_critical_depth",
    "axis_width_max",
    "axis_width_depleted",
    "axis_offset",
    "axis_tilt",
    "critical_depth",
    "critical_depth_maximum",
    "width_max",
    "width_depleted",
    "offset",
    "tilt",
    "width",
    "cloud",
    "fractions",
    "loss",
    "table",
    "Fit",
    "Scan",
    "scan",
    "fit",
    "fit_all",
    "fits",
    "save",
    "join_fits",
]

models = ("slab", "gaussian", "gaussian_quadratic", "slab_tied")
"""
The models of the charge cloud fit to every track.

In each, the cloud of charge created in the field-free layer has a
variance which falls from :math:`\\sigma_\\text{max}^2` at the back surface
to zero at :math:`t_c`, plus the spread :math:`\\sigma_d^2 \\, g(t)` of the
depletion region.

``slab``
    The model of the article: the cloud of charge that diffuses until it
    reaches the depletion region and is reflected at the back surface, a
    mixture of Gaussians whose variance falls as :math:`1 - (t / t_c)^2`.
``gaussian``
    The model the article compares it with: a single Gaussian whose
    variance falls as :math:`1 - t / t_c`.
``gaussian_quadratic``
    A single Gaussian with the variance of the slab model, which separates
    the shape of the cloud from the way its width falls with depth.
``slab_tied``
    The slab model with :math:`\\sigma_\\text{max}` tied to the thickness
    of the field-free layer, :math:`t_c D`, as the physics of the slab has
    it for a sensor of the nominal thickness :math:`D`, which leaves
    :math:`t_c` the only parameter of each track.

The last two are compared with the others in the documentation.
"""

_shape = dict(
    slab="slab",
    gaussian="gaussian",
    gaussian_quadratic="gaussian",
    slab_tied="slab",
)
"""The shape of the cloud of each model, a mixture of Gaussians or a single one."""

_quadratic = dict(
    slab=True,
    gaussian=False,
    gaussian_quadratic=True,
    slab_tied=True,
)
"""Whether the variance of each model falls as :math:`1 - (t / t_c)^2` rather than :math:`1 - t / t_c`."""

_tied = dict(
    slab=False,
    gaussian=False,
    gaussian_quadratic=False,
    slab_tied=True,
)
"""Whether each model ties :math:`\\sigma_\\text{max}` to :math:`t_c D`."""


def _check(model: str) -> None:
    """
    Raise an error unless the given model is one of :data:`models`.

    Parameters
    ----------
    model
        The name of the model.
    """
    if model not in models:
        raise ValueError(f"unknown model {model!r}, expected one of {models}")


axis_critical_depth = "critical_depth"
"""The logical axis of the grid of fractional field-free thicknesses."""

axis_width_max = "width_max"
"""The logical axis of the grid of back-surface diffusion widths."""

axis_width_depleted = "width_depleted"
"""The logical axis of the grid of spreads acquired inside the depletion region."""

axis_offset = "offset"
"""The logical axis of the grid of centerline offsets."""

axis_tilt = "tilt"
"""The logical axis of the grid of centerline tilt corrections."""

_axis_orientation = "orientation"
"""The logical axis of the two orientations of a track, entering at its first slice or its last."""

critical_depth = na.linspace(0, 1, axis=axis_critical_depth, num=21)
"""
The grid of fractional field-free thicknesses, :math:`t_c = z_f / D`,
searched by :func:`scan`.
"""

width_max = na.linspace(0, 10, axis=axis_width_max, num=21) * u.um
"""The grid of back-surface diffusion widths, :math:`\\sigma_\\text{max}`, searched by :func:`scan`."""

width_depleted = (
    np.round(na.linspace(0, 1.5, axis=axis_width_depleted, num=16), 3) * u.um
)
"""
The grid of spreads acquired crossing the full depletion region,
:math:`\\sigma_d`, searched by :func:`scan`: 0 to 1.5 micrometers in steps
of 0.1, rounded so that a grid value read back from the data files
compares equal. The pooled misfit of every CCD has its minimum near 0.75
and climbs steeply on both sides, so the grid covers the answer with room
to spare, and the step sets the resolution of the pooled value.

Unlike :math:`t_c` and :math:`\\sigma_\\text{max}`, which are fit to each
track, :math:`\\sigma_d` is a property of the drift field and is shared by
every track on a CCD, so :func:`fit_all` chooses one value per CCD by
pooling the misfit of its tracks.
"""

_grid_width_depleted = width_depleted
"""An alias of :data:`width_depleted` for methods whose parameters shadow it."""

critical_depth_maximum = 0.7
"""The largest fitted :math:`t_c` for which a track counts as having crossed the sensor."""

offset = na.linspace(-0.6, 0.6, axis=axis_offset, num=25)
"""The grid of centerline offsets (in pixels) marginalized over by :func:`scan`."""

tilt = na.linspace(-0.03, 0.03, axis=axis_tilt, num=5)
"""The grid of centerline tilt corrections (in pixels per slice) marginalized over by :func:`scan`."""

_interval = 2
"""
How far above the least misfit a grid point may lie and still count toward
the range of :math:`t_c` and :math:`\\sigma_\\text{max}` a track allows.
"""

_edges = na.arange(-half_width - 1, half_width + 1, axis=axis_pixel) + 0.5
"""The pixel boundaries across the track, in pixels from the central pixel."""


def width(
    depth: na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
    width_max: u.Quantity | na.AbstractScalarArray,
    width_depleted: u.Quantity | na.AbstractScalarArray = 0 * u.um,
    model: str = "slab",
) -> na.AbstractScalarArray:
    """
    The standard deviation of the charge cloud, in pixels, for charge
    deposited at the given fractional depth.

    The variance is

    .. math::

        \\sigma^2(t) = \\sigma_\\text{max}^2 \\, w(t / t_c)
        + \\sigma_d^2 \\, g(t),

    where the first term applies only in the field-free layer, :math:`t <
    t_c`, with :math:`w(x) = 1 - x^2` in the slab model and :math:`1 - x`
    in the Gaussian model (see :data:`models`), and :math:`g = 1` there,
    since that charge drifts across the full depleted thickness, falling
    linearly to zero at the gates, :math:`g = \\min[(1 - t) / (1 - t_c), 1]`.

    Parameters
    ----------
    depth
        The fractional depth below the back surface, :math:`t = z / D`.
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface,
        :math:`\\sigma_\\text{max}`.
    width_depleted
        The spread acquired crossing the full depletion region,
        :math:`\\sigma_d`, zero in the field-free model.
    model
        One of :data:`models`.
    """
    _check(model)
    tc = np.maximum(critical_depth, 1e-6)
    x = depth / tc
    shape = 1 - np.square(x) if _quadratic[model] else 1 - x
    wedge = np.where(depth < critical_depth, np.maximum(shape, 0), 0)
    field_free = np.square(width_max) * wedge
    g = _depletion(depth, critical_depth)
    result = np.sqrt(field_free + np.square(width_depleted) * g)
    return (result / width_pixel).to(u.dimensionless_unscaled).value


def _depletion(
    depth: float | na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
) -> na.AbstractScalarArray:
    """
    The fraction of the depletion region crossed by charge deposited at the
    given fractional depth, :math:`g(t)`.

    Parameters
    ----------
    depth
        The fractional depth below the back surface.
    critical_depth
        The fractional thickness of the field-free region.
    """
    return np.minimum((1 - depth) / np.maximum(1 - critical_depth, 1e-6), 1)


def cloud(
    depth: na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
    width_max: u.Quantity | na.AbstractScalarArray,
    width_depleted: u.Quantity | na.AbstractScalarArray = 0 * u.um,
    model: str = "slab",
    axis: str = axis_mixture,
) -> tuple[na.AbstractScalarArray, na.AbstractScalarArray]:
    """
    The charge cloud of charge deposited at the given fractional depth as a
    mixture of Gaussians: the standard deviation of each Gaussian, in
    pixels, and its weight.

    A Gaussian model is a mixture of one Gaussian, whose standard deviation
    is :func:`width`.
    A slab model is the mixture of
    :class:`optika.sensors.diffusion.SlabDiffusionModel` for a field-free
    layer of thickness :math:`\\sigma_\\text{max}`, a fraction
    :math:`t / t_c` of the way across it, with the spread of the depletion
    region added to every Gaussian.
    Its variance is that of :func:`width`.

    Parameters
    ----------
    depth
        The fractional depth below the back surface, :math:`t = z / D`.
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface,
        :math:`\\sigma_\\text{max}`.
    width_depleted
        The spread acquired crossing the full depletion region,
        :math:`\\sigma_d`.
    model
        One of :data:`models`.
    axis
        The logical axis of the Gaussians.
    """
    _check(model)
    if _shape[model] == "gaussian":
        w = width(depth, critical_depth, width_max, width_depleted, model)
        return w.add_axes(axis), na.ScalarArray(np.ones(1), axes=axis)
    tc = np.maximum(critical_depth, 1e-6)
    fraction = np.where(depth < critical_depth, depth / tc, 1)
    variance, weight = transit(fraction, axis)
    variance = variance * np.square(width_max)
    variance = variance + np.square(width_depleted) * _depletion(depth, critical_depth)
    result = np.sqrt(variance) / width_pixel
    return result.to(u.dimensionless_unscaled).value, weight


def fractions(
    position: na.AbstractScalarArray,
    width: na.AbstractScalarArray,
    slope: float,
    weight: float | na.AbstractScalarArray = 1,
    axis: None | str = None,
) -> na.AbstractScalarArray:
    """
    The fraction of each slice's charge expected in each of the
    :math:`2 h + 1` pixels across the track.

    The charge deposited in a slice is spread uniformly along the tilted
    centerline and diffuses as a Gaussian of the given width, or as a
    mixture of such Gaussians, each of which is approximated by a Gaussian
    of variance :math:`\\sigma^2 + m^2 / 12`, where :math:`m` is the slope
    of the centerline.

    Parameters
    ----------
    position
        The centerline position of each slice in pixels from the central pixel.
    width
        The standard deviation of the charge cloud in pixels, or of each
        Gaussian of the mixture along ``axis``.
    slope
        The tilt of the centerline in pixels per slice.
    weight
        The weight of each Gaussian of the mixture along ``axis``.
    axis
        The logical axis of the Gaussians of a mixture, :obj:`None` if the
        cloud is a single Gaussian.
    """
    s = np.sqrt(np.square(np.maximum(width, 1e-3)) + np.square(slope) / 12)
    x = (_edges - position) / (s * np.sqrt(2))
    x = x.astype(np.float32)
    cdf = scipy.special.erf(x) / 2
    result = np.diff(cdf, axis=axis_pixel)
    if axis is not None:
        result = (weight * result).sum(axis)
    return result / np.maximum(result.sum(axis_pixel), 1e-9)


def loss(
    track: Track,
    position: na.AbstractScalarArray,
    width: na.AbstractScalarArray,
    weight: float | na.AbstractScalarArray = 1,
    axis: None | str = None,
) -> na.AbstractScalarArray:
    """
    A robust misfit between the observed and modeled charge fractions,
    :math:`\\sum \\ln(1 + r^2 / 2)`, where :math:`r` is the residual in
    units of the read noise.

    Parameters
    ----------
    track
        The track to compare against.
    position
        The centerline position of each slice in pixels from the central pixel.
    width
        The standard deviation of the charge cloud in pixels, or of each
        Gaussian of the mixture along ``axis``.
    weight
        The weight of each Gaussian of the mixture along ``axis``.
    axis
        The logical axis of the Gaussians of a mixture, :obj:`None` if the
        cloud is a single Gaussian.
    """
    q = fractions(position, width, track.slope, weight, axis)
    residual = (track.fraction - q) / track.error
    return np.log1p(np.square(residual) / 2).sum((axis_slice, axis_pixel))


@dataclasses.dataclass(eq=False)
class Fit:
    """The result of fitting a model of the charge cloud to a single track."""

    track: Track
    """The track that was fit."""

    model: str
    """The model of the charge cloud, one of :data:`models`."""

    width_depleted: u.Quantity
    """The spread inside the depletion region, :math:`\\sigma_d`, shared by every track on the CCD."""

    orientation: int
    """``+1`` if the track enters the back surface at its first slice, ``-1`` if at its last."""

    critical_depth: float
    """The best-fit fractional thickness of the field-free region, :math:`t_c`."""

    width_max: u.Quantity
    """The best-fit width of the charge cloud at the back surface."""

    offset: float
    """The best-fit centerline offset in pixels."""

    tilt: float
    """The best-fit centerline tilt correction in pixels per slice."""

    misfit: float
    """The :func:`loss` of the best fit."""

    gain: float
    """The decrease in :func:`loss` relative to a model with no diffusion at all."""

    critical_depth_min: float
    """The smallest :math:`t_c` within two units of :func:`loss` of the best fit."""

    critical_depth_max: float
    """The largest :math:`t_c` within two units of :func:`loss` of the best fit."""

    width_max_min: u.Quantity
    """The smallest :math:`\\sigma_\\text{max}` within two units of :func:`loss` of the best fit."""

    width_max_max: u.Quantity
    """The largest :math:`\\sigma_\\text{max}` within two units of :func:`loss` of the best fit."""

    width_depleted_preferred: u.Quantity
    """The :math:`\\sigma_d` this track prefers on its own, ignoring the other tracks on the CCD."""

    @property
    def depth(self) -> na.AbstractScalarArray:
        """The fractional depth of each slice, accounting for :attr:`orientation`."""
        t = self.track.depth
        return t if self.orientation > 0 else 1 - t

    @property
    def position(self) -> na.AbstractScalarArray:
        """The best-fit centerline position of each slice."""
        return self.track.position + self.offset + self.tilt * self.track.index

    @property
    def width(self) -> na.AbstractScalarArray:
        """The best-fit standard deviation of the charge cloud of each slice in pixels."""
        return width(
            self.depth,
            self.critical_depth,
            self.width_max,
            self.width_depleted,
            self.model,
        )

    @property
    def predicted(self) -> na.AbstractScalarArray:
        """The fraction of each slice's charge the best fit puts in each pixel."""
        w, weight = cloud(
            self.depth,
            self.critical_depth,
            self.width_max,
            self.width_depleted,
            self.model,
        )
        return fractions(self.position, w, self.track.slope, weight, axis_mixture)

    @property
    def tight(self) -> bool:
        """Whether the track constrains :math:`t_c` to within 0.15."""
        gain = self.gain > 10
        interval = (self.critical_depth_max - self.critical_depth_min) <= 0.15 + 1e-6
        return bool(gain and interval)

    @property
    def bragg(self) -> float:
        """The ratio of the median charge per slice in the last third to that in the first third."""
        signal = self.track.signal.ndarray
        third = max(len(signal) // 3, 3)
        a = np.median(signal[:third])
        b = np.median(signal[-third:])
        return float(max(a, b) / min(a, b))

    @property
    def crossing(self) -> bool:
        """
        Whether the fit found a depleted end, :math:`t_c \\le` :data:`critical_depth_maximum`.

        A particle that crossed the full thickness of the sensor is
        pixel-sharp where it left through the gates. A feature that is wide
        from end to end, which the fit describes with :math:`t_c` near one,
        is not such a track: on the slit-jaw imager it is usually a spicule
        or other structure at the limb that the mask let through.
        """
        # the grid point at 0.7 is a hair above 0.7 in binary, while a saved
        # fit holds it rounded, so the boundary needs a tolerance to agree
        return self.critical_depth <= critical_depth_maximum + 1e-6

    @property
    def flat(self) -> bool:
        """
        Whether the track is :attr:`tight`, has no Bragg rise along its
        length, and is :attr:`crossing`.
        """
        return self.tight and (self.bragg < 1.5) and self.crossing


@dataclasses.dataclass(eq=False)
class Scan:
    """
    The fit of a single track at every :math:`\\sigma_d` on the grid
    :data:`width_depleted`.

    Every attribute below :attr:`model` is an array over
    :data:`axis_width_depleted`, holding the corresponding attribute of the
    :class:`Fit` at that :math:`\\sigma_d`.
    """

    track: Track
    """The track that was fit."""

    model: str
    """The model of the charge cloud, one of :data:`models`."""

    misfit: na.AbstractScalarArray
    """The smallest :func:`loss` at each :math:`\\sigma_d`."""

    orientation: na.AbstractScalarArray
    critical_depth: na.AbstractScalarArray
    width_max: na.AbstractScalarArray
    offset: na.AbstractScalarArray
    tilt: na.AbstractScalarArray
    gain: na.AbstractScalarArray
    critical_depth_min: na.AbstractScalarArray
    critical_depth_max: na.AbstractScalarArray
    width_max_min: na.AbstractScalarArray
    width_max_max: na.AbstractScalarArray

    @property
    def preferred(self) -> u.Quantity:
        """The :math:`\\sigma_d` which minimizes :attr:`misfit`."""
        grid = _grid_width_depleted
        return grid[np.argmin(self.misfit, axis=axis_width_depleted)].ndarray

    def at(self, width_depleted: u.Quantity) -> Fit:
        """
        The fit at the grid point nearest the given :math:`\\sigma_d`.

        Parameters
        ----------
        width_depleted
            The spread inside the depletion region.
        """
        grid = _grid_width_depleted
        nearest = np.argmin(np.abs(grid - width_depleted), axis=axis_width_depleted)
        i = {axis_width_depleted: int(nearest[axis_width_depleted].ndarray)}
        return Fit(
            track=self.track,
            model=self.model,
            width_depleted=grid[i].ndarray,
            orientation=int(self.orientation[i].ndarray),
            critical_depth=float(self.critical_depth[i].ndarray),
            width_max=self.width_max[i].ndarray,
            offset=float(self.offset[i].ndarray),
            tilt=float(self.tilt[i].ndarray),
            misfit=float(self.misfit[i].ndarray),
            gain=float(self.gain[i].ndarray),
            critical_depth_min=float(self.critical_depth_min[i].ndarray),
            critical_depth_max=float(self.critical_depth_max[i].ndarray),
            width_max_min=self.width_max_min[i].ndarray,
            width_max_max=self.width_max_max[i].ndarray,
            width_depleted_preferred=self.preferred,
        )


_width_step = 0.00125
"""The spacing, in pixels, of the widths tabulated by :func:`table`."""

_width_table = na.linspace(0, 0.85, axis="table", num=round(0.85 / _width_step) + 1)
"""
The widths, in pixels, tabulated by :func:`table`, which reach past the
widest model on the grids, :math:`\\sqrt{\\sigma_\\text{max}^2 + \\sigma_d^2}`
at 13 :math:`\\mu`m pixels.
"""


def table(track: Track) -> na.AbstractScalarArray:
    """
    The misfit of each slice of a track, on the nuisance grid of
    :data:`offset` and :data:`tilt` and a fine grid of widths.

    The misfit of the Gaussian model is the sum over slices of this table,
    since each slice contributes through its centerline position and its
    width alone, so :func:`scan` assembles every Gaussian model on its grids
    from one table instead of evaluating :func:`loss` afresh.

    Parameters
    ----------
    track
        The track to tabulate.
    """
    position = track.position + offset + tilt * track.index
    q = fractions(position, _width_table, track.slope)
    residual = (track.fraction - q) / track.error
    return np.log1p(np.square(residual) / 2).sum(axis_pixel)


def _grid_width_max(model: str) -> na.AbstractScalarArray:
    """
    The values of :math:`\\sigma_\\text{max}` searched for the given model,
    with axes ``(critical_depth, width_max)``: the grid :data:`width_max` at
    every :math:`t_c`, or :math:`t_c D` alone if the model ties the two.

    Parameters
    ----------
    model
        One of :data:`models`.
    """
    _check(model)
    if _tied[model]:
        return (critical_depth * thickness).add_axes(axis_width_max)
    return width_max + 0 * critical_depth


def _misfit_gaussian(track: Track, model: str = "gaussian") -> na.AbstractScalarArray:
    """
    The misfit of a Gaussian model to a track at every point of its grids
    and of the nuisance grid, assembled from the :func:`table` of the track,
    interpolated linearly in width.

    Parameters
    ----------
    track
        The track to fit.
    model
        One of the models of :data:`models` whose cloud is a single
        Gaussian.
    """
    if _shape[model] != "gaussian":
        raise ValueError(f"the cloud of the {model!r} model is not a Gaussian")
    grid = _grid_width_max(model)
    misfit_slice = table(track)
    depth = track.depth
    index_slice = na.arange(0, track.length, axis=axis_slice)
    last = _width_table.size - 2

    result = []
    for orientation in (+1, -1):
        t = depth if orientation > 0 else 1 - depth
        rows = []
        for a, tc in enumerate(critical_depth.ndarray):
            sm = grid[{axis_critical_depth: a}]
            w = width(t, tc, sm, width_depleted, model) / _width_step
            k = np.minimum(np.floor(w), last).astype(int)
            weight = w - k
            lo = misfit_slice[{axis_slice: index_slice, "table": k}]
            hi = misfit_slice[{axis_slice: index_slice, "table": k + 1}]
            rows.append((lo + weight * (hi - lo)).sum(axis_slice))
        result.append(na.stack(rows, axis=axis_critical_depth))
    return na.stack(result, axis=_axis_orientation)


_axes_model = (
    _axis_orientation,
    axis_critical_depth,
    axis_width_max,
    axis_width_depleted,
)
"""The axes of the grid of models searched by :func:`scan`, in order."""


def _misfit_slab(
    track: Track,
    center: dict[str, na.AbstractScalarArray],
    model: str = "slab",
    prune: bool = True,
) -> tuple[na.ScalarArray, dict[str, na.ScalarArray]]:
    """
    The misfit of a slab model to a track at every point of its grids,
    minimized over every offset and tilt of the nuisance grid, and the index
    of the best offset and tilt.

    Parameters
    ----------
    track
        The track to fit.
    center
        The index of the offset and of the tilt the search starts from at
        each grid point, with the axes of :data:`_axes_model`.
    model
        One of the models of :data:`models` whose cloud is that of the slab.
    prune
        Whether to abandon a centerline once its partial sum over the slices
        exceeds the least misfit found so far, which changes nothing but the
        cost.
    """
    if _shape[model] != "slab":
        raise ValueError(f"the cloud of the {model!r} model is not the slab's")
    tc = critical_depth.ndarray
    t = np.stack([track.depth.ndarray, 1 - track.depth.ndarray])
    t = t[:, np.newaxis, :]
    tc = tc[np.newaxis, :, np.newaxis]
    field_free = t < tc
    fraction = np.where(field_free, t / np.maximum(tc, 1e-6), 1)
    axes = (_axis_orientation, axis_critical_depth, axis_slice)
    variance, weight = transit(na.ScalarArray(fraction, axes=axes))
    variance = na.broadcast_to(
        variance,
        {
            _axis_orientation: 2,
            axis_critical_depth: critical_depth.size,
            axis_slice: track.length,
            axis_mixture: weight.size,
        },
    )
    grid = _grid_width_max(model)

    def plain(a: na.AbstractScalarArray, axes: tuple[str, ...]) -> np.ndarray:
        return np.ascontiguousarray(a.transpose(axes).ndarray, dtype=float)

    def pixels(a: na.AbstractScalarArray) -> na.AbstractScalarArray:
        return (a / width_pixel).to(u.dimensionless_unscaled).value

    def index(a: na.AbstractScalarArray) -> np.ndarray:
        return np.ascontiguousarray(a.transpose(_axes_model).ndarray, dtype=np.int64)

    misfit, best_offset, best_tilt = _misfit(
        fraction=plain(track.fraction, (axis_slice, axis_pixel)),
        error=np.ascontiguousarray(track.error.ndarray, dtype=float),
        position=np.ascontiguousarray(track.position.ndarray, dtype=float),
        index=np.ascontiguousarray(track.index.ndarray, dtype=float),
        blur=float(track.slope) ** 2 / 12,
        field_free=np.ascontiguousarray(field_free),
        variance=plain(variance, axes + (axis_mixture,)),
        weight=np.ascontiguousarray(weight.ndarray, dtype=float),
        depletion=np.ascontiguousarray(_depletion(t, tc)),
        width_max=plain(pixels(grid), (axis_critical_depth, axis_width_max)),
        width_depleted=plain(pixels(width_depleted), (axis_width_depleted,)),
        offset=np.ascontiguousarray(offset.ndarray, dtype=float),
        tilt=np.ascontiguousarray(tilt.ndarray, dtype=float),
        center_offset=index(center[axis_offset]),
        center_tilt=index(center[axis_tilt]),
        margin=_interval,
        prune=prune,
        edges=np.ascontiguousarray(_edges.ndarray, dtype=float),
        table=_erf_table,
    )
    nuisance = {
        axis_offset: na.ScalarArray(best_offset, axes=_axes_model),
        axis_tilt: na.ScalarArray(best_tilt, axes=_axes_model),
    }
    return na.ScalarArray(misfit, axes=_axes_model), nuisance


def _summarize(
    track: Track,
    model: str,
    misfit: na.AbstractScalarArray,
    nuisance: dict[str, na.AbstractScalarArray],
) -> Scan:
    """
    The fit at every :math:`\\sigma_d` from the misfit at every point of the
    grids, minimized over the centerline.

    Parameters
    ----------
    track
        The track that was fit.
    model
        The model of the charge cloud.
    misfit
        The misfit at every point of the grids, with the axes of
        :data:`_axes_model`.
    nuisance
        The index of the offset and of the tilt which minimize the misfit at
        every point of the grids.
    """
    grid = _grid_width_max(model)
    axes_model = (axis_critical_depth, axis_width_max, _axis_orientation)
    v_best = misfit.min(axes_model)
    index = np.argmin(misfit, axis=axes_model)

    v_none = misfit[
        {
            _axis_orientation: 0,
            axis_critical_depth: 0,
            axis_width_max: 0,
            axis_width_depleted: 0,
        }
    ]

    # the range of each parameter over the grid points within the interval
    # of the best fit
    ok = misfit < v_best + _interval
    inf = np.inf * u.um

    return Scan(
        track=track,
        model=model,
        misfit=v_best,
        orientation=np.where(index[_axis_orientation] == 0, +1, -1),
        critical_depth=critical_depth[index],
        width_max=grid[index],
        offset=offset[{axis_offset: nuisance[axis_offset][index]}],
        tilt=tilt[{axis_tilt: nuisance[axis_tilt][index]}],
        gain=v_none - v_best,
        critical_depth_min=np.where(ok, critical_depth, np.inf).min(axes_model),
        critical_depth_max=np.where(ok, critical_depth, -np.inf).max(axes_model),
        width_max_min=np.where(ok, grid, inf).min(axes_model),
        width_max_max=np.where(ok, grid, -inf).max(axes_model),
    )


def _scan_gaussian(track: Track, model: str = "gaussian") -> tuple[
    Scan,
    dict[str, na.AbstractScalarArray],
]:
    """
    Fit a Gaussian model to a single track at every :math:`\\sigma_d`, and
    give the index of the offset and of the tilt which minimize its misfit at
    every grid point.

    Parameters
    ----------
    track
        The track to fit.
    model
        One of the models of :data:`models` whose cloud is a single
        Gaussian.
    """
    misfit = _misfit_gaussian(track, model)
    nuisance = np.argmin(misfit, axis=(axis_offset, axis_tilt))
    nuisance = {a: nuisance[a] for a in (axis_offset, axis_tilt)}
    result = _summarize(
        track=track,
        model=model,
        misfit=misfit.min((axis_offset, axis_tilt)),
        nuisance=nuisance,
    )
    return result, nuisance


def _scan_slab(
    track: Track,
    center: dict[str, na.AbstractScalarArray],
    model: str = "slab",
) -> Scan:
    """
    Fit a slab model to a single track at every :math:`\\sigma_d`, starting
    the search for the centerline at each grid point from that of the
    Gaussian model.

    Parameters
    ----------
    track
        The track to fit.
    center
        The index of the offset and of the tilt which minimize the misfit of
        the Gaussian model at every grid point.
    model
        One of the models of :data:`models` whose cloud is that of the slab.
    """
    if _tied[model]:
        # start from the Gaussian's centerline at the grid point of
        # sigma_max nearest the tied value
        step = width_max[{axis_width_max: 1}] - width_max[{axis_width_max: 0}]
        tied = _grid_width_max(model)[{axis_width_max: 0}]
        nearest = (tied / step).to(u.dimensionless_unscaled).value
        nearest = np.clip(np.rint(nearest).astype(int), 0, width_max.size - 1)
        shape = {a: n for a, n in center[axis_offset].shape.items()}
        del shape[axis_width_max]
        index = {**na.indices(shape), axis_width_max: nearest}
        center = {a: c[index].add_axes(axis_width_max) for a, c in center.items()}
    misfit, nuisance = _misfit_slab(track, center, model)
    return _summarize(track, model, misfit, nuisance)


def _scan_models(track: Track) -> dict[str, Scan]:
    """
    Fit every model of :data:`models` to a single track at every
    :math:`\\sigma_d`.

    Parameters
    ----------
    track
        The track to fit.
    """
    gaussian, center = _scan_gaussian(track, "gaussian")
    result = dict(gaussian=gaussian)
    for model in models:
        if model in result:
            continue
        if _shape[model] == "gaussian":
            result[model], _ = _scan_gaussian(track, model)
        else:
            result[model] = _scan_slab(track, center, model)
    return {model: result[model] for model in models}


def scan(track: Track, model: str = "slab") -> Scan:
    """
    Fit a model of the charge cloud to a single track at every
    :math:`\\sigma_d`.

    The grid of :data:`critical_depth`, :data:`width_max` (or
    :math:`\\sigma_\\text{max} = t_c D` for a model that ties the two),
    :data:`width_depleted`, and both track orientations is searched
    exhaustively, and at each grid point the :func:`loss` is minimized over
    the nuisance grid of :data:`offset` and :data:`tilt`.
    The misfit of a Gaussian model is assembled from the :func:`table` of
    the track, interpolated linearly in width.
    That of a slab model is computed afresh, starting from the centerline
    the Gaussian model prefers at the same grid point and abandoning a
    centerline as soon as its partial sum over the slices exceeds a bound,
    which leaves the least misfit, and every misfit within two of it, as
    they would be if every centerline were tried in full.

    Parameters
    ----------
    track
        The track to fit.
    model
        One of :data:`models`.
    """
    _check(model)
    if _shape[model] == "gaussian":
        result, _ = _scan_gaussian(track, model)
        return result
    _, center = _scan_gaussian(track, "gaussian")
    return _scan_slab(track, center, model)


def fit(
    track: Track,
    width_depleted: u.Quantity = 0 * u.um,
    model: str = "slab",
) -> Fit:
    """
    Fit a model of the charge cloud to a single track at a given
    :math:`\\sigma_d`.

    Parameters
    ----------
    track
        The track to fit.
    width_depleted
        The spread inside the depletion region, which is shared by every
        track on a CCD and is chosen by :func:`fit_all`.
    model
        One of :data:`models`.
    """
    return scan(track, model).at(width_depleted)


_path_fits = {
    model: _directory_data / f"iris_fits{'' if model == 'slab' else '_' + model}.csv"
    for model in models
}
"""
The files holding the result of :func:`fit_all` for every track, one per
model, ``iris_fits.csv`` for the slab model of the article.
"""

_fields_fits = [
    "name",
    "width_depleted",
    "orientation",
    "critical_depth",
    "width_max",
    "offset",
    "tilt",
    "misfit",
    "gain",
    "critical_depth_min",
    "critical_depth_max",
    "width_max_min",
    "width_max_max",
    "width_depleted_preferred",
]


def fit_all(
    tracks: tuple[Track, ...],
) -> dict[str, tuple[tuple[Fit, ...], tuple["ccd_diffusion.tracks.Depleted", ...]]]:
    """
    Fit every model of :data:`models` to every given track, using one
    process per CPU, choosing one :math:`\\sigma_d` per CCD and model by
    pooling the misfits of its flat tracks (see
    :func:`ccd_diffusion.tracks.pooled`).

    Parameters
    ----------
    tracks
        The tracks to fit.

    Returns
    -------
    For each model, the fit of every track, in the given order, and the
    pooled fit of :math:`\\sigma_d` on every CCD.
    """
    from ._depleted import pooled

    # processes rather than threads, since the search of the Gaussian model
    # holds the interpreter for much of its time
    with concurrent.futures.ProcessPoolExecutor() as pool:
        scans = list(pool.map(_scan_models, tracks, chunksize=8))

    result = {}
    for model in models:
        fitted = {}
        depleted = []
        for chip in dict.fromkeys(t.chip for t in tracks):
            mine = [s[model] for s in scans if s[model].track.chip == chip]
            d = pooled(chip, mine)
            depleted.append(d)
            for s in mine:
                fitted[s.track.name] = s.at(d.best)
        result[model] = (tuple(fitted[t.name] for t in tracks), tuple(depleted))
    return result


def save(
    fits: tuple[Fit, ...],
    depleted: tuple["ccd_diffusion.tracks.Depleted", ...],
    model: str = "slab",
    directory: None | pathlib.Path = None,
) -> None:
    """
    Write the given fits to ``data/iris_fits.csv`` (or
    ``data/iris_fits_gaussian.csv`` for the Gaussian model), where
    :func:`fits` will find them, and the pooled fits of :math:`\\sigma_d` to
    ``data/iris_depleted.csv`` (or ``data/iris_depleted_gaussian.csv``),
    where :func:`ccd_diffusion.tracks.depleted` will.

    Fitting every track takes a while, so the fits are stored in the
    repository alongside the tracks and only recomputed by running
    ``python -m ccd_diffusion.tracks fit``.

    Parameters
    ----------
    fits
        The fits to save, one per track returned by :func:`load`.
    depleted
        The pooled fit of :math:`\\sigma_d` on every CCD.
    model
        The model the fits are of, one of :data:`models`.
    directory
        Where to write the files instead, for fits of some of the CCDs that
        :func:`join_fits` will join with the rest.
    """
    from ._depleted import save_depleted

    path = _path_fits[model]
    if directory is not None:
        path = pathlib.Path(directory) / path.name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields_fits)
        writer.writeheader()
        for r in fits:
            writer.writerow(
                dict(
                    name=r.track.name,
                    width_depleted=f"{r.width_depleted.to_value(u.um):.2f}",
                    orientation=r.orientation,
                    critical_depth=f"{r.critical_depth:.3f}",
                    width_max=f"{r.width_max.to_value(u.um):.2f}",
                    offset=f"{r.offset:.3f}",
                    tilt=f"{r.tilt:.4f}",
                    misfit=f"{r.misfit:.3f}",
                    gain=f"{r.gain:.3f}",
                    critical_depth_min=f"{r.critical_depth_min:.3f}",
                    critical_depth_max=f"{r.critical_depth_max:.3f}",
                    width_max_min=f"{r.width_max_min.to_value(u.um):.2f}",
                    width_max_max=f"{r.width_max_max.to_value(u.um):.2f}",
                    width_depleted_preferred=(
                        f"{r.width_depleted_preferred.to_value(u.um):.2f}"
                    ),
                )
            )
    save_depleted(depleted, model, directory)


def join_fits(directories: list[pathlib.Path]) -> None:
    """
    Join the fits of separately fitted CCDs, each written by :func:`save`
    to a directory of its own, into the package's data files, in the order
    of :func:`load`.

    Fitting the slab model to every track takes hours on a few cores, so the
    ``data`` workflow fits each CCD in a job of its own, which it can do
    since :math:`\\sigma_d` is pooled over the tracks of one CCD.

    Parameters
    ----------
    directories
        The directories written by :func:`save`, together holding every
        track once.
    """
    from ._depleted import _path_depleted

    tracks = load()
    chips = list(dict.fromkeys(t.chip for t in tracks))
    for model in models:
        rows = {}
        rows_depleted = {}
        for directory in directories:
            directory = pathlib.Path(directory)
            with open(directory / _path_fits[model].name, newline="") as f:
                for r in csv.DictReader(f):
                    rows[r["name"]] = r
            with open(directory / _path_depleted[model].name, newline="") as f:
                reader = csv.DictReader(f)
                fields_depleted = reader.fieldnames
                for r in reader:
                    rows_depleted.setdefault(r["chip"], []).append(r)
        missing = [t.name for t in tracks if t.name not in rows]
        if missing:
            raise ValueError(
                f"no {model} fit of {len(missing)} tracks, the first {missing[0]!r}"
            )
        with open(_path_fits[model], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_fields_fits)
            writer.writeheader()
            for t in tracks:
                writer.writerow(rows[t.name])
        with open(_path_depleted[model], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_depleted)
            writer.writeheader()
            for chip in chips:
                for r in rows_depleted[chip]:
                    writer.writerow(r)


@functools.cache
def fits(model: str = "slab") -> tuple[Fit, ...]:
    """
    Load the result of :func:`fit_all` for every track returned by
    :func:`load` from ``data/iris_fits.csv`` (or
    ``data/iris_fits_gaussian.csv`` for the Gaussian model).

    Parameters
    ----------
    model
        One of :data:`models`.
    """
    tracks = {track.name: track for track in load()}
    with open(_path_fits[model], newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Fit(
            track=tracks[r["name"]],
            model=model,
            width_depleted=float(r["width_depleted"]) * u.um,
            orientation=int(r["orientation"]),
            critical_depth=float(r["critical_depth"]),
            width_max=float(r["width_max"]) * u.um,
            offset=float(r["offset"]),
            tilt=float(r["tilt"]),
            misfit=float(r["misfit"]),
            gain=float(r["gain"]),
            critical_depth_min=float(r["critical_depth_min"]),
            critical_depth_max=float(r["critical_depth_max"]),
            width_max_min=float(r["width_max_min"]) * u.um,
            width_max_max=float(r["width_max_max"]) * u.um,
            width_depleted_preferred=float(r["width_depleted_preferred"]) * u.um,
        )
        for r in rows
    )
