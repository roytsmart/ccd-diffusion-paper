"""
The charge cloud of the slab model as a mixture of Gaussians, and the
compiled search that fits it to a track.

The cloud of charge created in the field-free layer is that of
:class:`optika.sensors.diffusion.SlabDiffusionModel`: the charge diffuses
until it reaches the depletion region and is reflected at the back surface.
That model gives it as a mixture of Gaussians, whose weights are the same at
every depth and whose variances, at a given fraction of the way across the
field-free layer, scale with the square of its thickness.
So :func:`transit` evaluates the mixture once for a layer of unit thickness,
and the fit scales its variances by :math:`\\sigma_\\text{max}^2`.
"""

import math
import numpy as np
import numba
import scipy.special
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "axis_mixture",
    "transit",
]

axis_mixture = "mixture"
"""The logical axis of the Gaussians of the charge cloud."""

_model = optika.sensors.diffusion.SlabDiffusionModel(thickness_depletion=0 * u.um)
"""A field-free layer of unit thickness with no depletion region beneath it."""


def transit(
    fraction: float | na.AbstractScalarArray,
    axis: str = axis_mixture,
) -> tuple[na.AbstractScalarArray, na.AbstractScalarArray]:
    """
    The charge cloud of charge created a given fraction of the way across
    the field-free layer, from the back surface, as a mixture of Gaussians:
    the variance of each Gaussian along one axis, in units of the square of
    the thickness of the layer, and its weight.

    The variance of the mixture is :math:`1 - x^2` at a fraction :math:`x`
    of the way across, and it vanishes at the depletion region, :math:`x = 1`.

    Parameters
    ----------
    fraction
        The fraction of the way across the field-free layer, from zero at
        the back surface to one at the depletion region.
    axis
        The logical axis of the Gaussians.
    """
    variance, weight = _model.mixture(
        depth=na.as_named_array(fraction) * u.um,
        thickness_substrate=1 * u.um,
        axis=axis,
    )
    return variance.to(u.um**2).value, weight


_erf_density = 2048
"""The number of values of the error function tabulated per unit argument."""

_erf_table = scipy.special.erf(np.arange(6 * _erf_density + 2) / _erf_density)
"""
The error function from 0 to 6, beyond which it is one to double precision,
interpolated linearly by :func:`_erf` to within :math:`3 \\times 10^{-8}`.
"""


@numba.njit(cache=True, inline="always")
def _erf(x: float, table: np.ndarray) -> float:
    """
    The error function, interpolated from :data:`_erf_table`, which is three
    times faster than :func:`math.erf` in the search.

    Parameters
    ----------
    x
        The argument.
    table
        :data:`_erf_table`.
    """
    a = abs(x) * _erf_density
    if a >= table.size - 2:
        result = 1.0
    else:
        i = int(a)
        result = table[i] + (a - i) * (table[i + 1] - table[i])
    return result if x >= 0 else -result


@numba.njit(cache=True, inline="always")
def _misfit_slice(
    fraction: np.ndarray,
    error: float,
    position: float,
    scale: np.ndarray,
    weight: np.ndarray,
    num: int,
    edges: np.ndarray,
    table: np.ndarray,
    model: np.ndarray,
) -> float:
    """
    The misfit of one slice, :math:`\\sum_j \\ln(1 + r_j^2 / 2)`, for a
    charge cloud that is a mixture of Gaussians.

    Parameters
    ----------
    fraction
        The measured fraction of the charge of the slice in each pixel.
    error
        The uncertainty of a fraction from read noise.
    position
        The centerline position of the slice in pixels from the central
        pixel.
    scale
        The reciprocal of :math:`\\sqrt{2}` times the standard deviation of
        each Gaussian, in reciprocal pixels.
    weight
        The weight of each Gaussian.
    num
        The number of Gaussians.
    edges
        The pixel boundaries across the track.
    table
        :data:`_erf_table`.
    model
        Scratch space for the modeled fractions, one per pixel.
    """
    num_pixel = edges.size - 1
    for j in range(num_pixel):
        model[j] = 0.0
    for k in range(num):
        s = scale[k]
        w = weight[k]
        previous = _erf((edges[0] - position) * s, table)
        for j in range(num_pixel):
            current = _erf((edges[j + 1] - position) * s, table)
            model[j] += w * (current - previous)
            previous = current
    total = 0.0
    for j in range(num_pixel):
        total += model[j]
    # the differences of the error function are twice the fractions, so the
    # floor on the total is twice that of `fractions`
    total = max(total, 2e-9)
    result = 0.0
    for j in range(num_pixel):
        r = (fraction[j] - model[j] / total) / error
        result += math.log1p(r * r / 2)
    return result


@numba.njit(cache=True, inline="always")
def _misfit_slices(
    start: float,
    bound: float,
    slices: np.ndarray,
    fraction: np.ndarray,
    error: np.ndarray,
    position: np.ndarray,
    scale: np.ndarray,
    weight: np.ndarray,
    num: np.ndarray,
    edges: np.ndarray,
    table: np.ndarray,
    model: np.ndarray,
) -> float:
    """
    A given misfit plus that of the given slices of a track for one
    centerline, or, once the sum so far reaches ``bound``, that partial sum.

    Every slice adds to the misfit, so a centerline whose partial sum
    reaches the least misfit found so far cannot have a smaller one, and the
    rest of its slices need not be evaluated.

    Parameters
    ----------
    start
        The misfit of the other slices of the track.
    bound
        The sum at which to stop.
    slices
        The indices of the slices to add, in the order to add them.
    fraction
        The measured fractions, with axes ``(slice, pixel)``.
    error
        The uncertainty of a fraction of each slice from read noise.
    position
        The centerline position of each slice.
    scale
        The reciprocal of :math:`\\sqrt{2}` times the standard deviation of
        each Gaussian of the cloud of each slice, with axes
        ``(slice, mixture)``.
    weight
        The weight of each Gaussian of the cloud of each slice, with axes
        ``(slice, mixture)``.
    num
        The number of Gaussians of the cloud of each slice.
    edges
        The pixel boundaries across the track.
    table
        :data:`_erf_table`.
    model
        Scratch space for the modeled fractions, one per pixel.
    """
    total = start
    for i in slices:
        if total >= bound:
            break
        total += _misfit_slice(
            fraction[i],
            error[i],
            position[i],
            scale[i],
            weight[i],
            num[i],
            edges,
            table,
            model,
        )
    return total


@numba.njit(cache=True, inline="always")
def _clouds(
    o: int,
    a: int,
    b: int,
    d: int,
    slices: np.ndarray,
    blur: float,
    field_free: np.ndarray,
    variance: np.ndarray,
    weight: np.ndarray,
    depletion: np.ndarray,
    width_max: np.ndarray,
    width_depleted: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    num: np.ndarray,
) -> None:
    """
    Write the charge cloud of the given slices of a track at one point of
    the grid, a mixture of Gaussians in the field-free layer and a single
    Gaussian beyond it, into ``scale``, ``weights``, and ``num``.

    Parameters
    ----------
    o
        The index of the orientation.
    a
        The index of :math:`t_c`.
    b
        The index of :math:`\\sigma_\\text{max}`.
    d
        The index of :math:`\\sigma_d`.
    slices
        The indices of the slices.
    blur
        The variance, in square pixels, of the uniform spread of the charge
        along the tilted centerline within a slice.
    field_free
        Whether each slice lies in the field-free layer.
    variance
        The variance of each Gaussian of the cloud of each slice for a
        field-free layer of unit thickness.
    weight
        The weight of each Gaussian.
    depletion
        The fraction of the depletion region crossed by the charge of each
        slice.
    width_max
        The grid of :math:`\\sigma_\\text{max}` in pixels, with axes
        ``(critical_depth, width_max)``, since a model may tie it to :math:`t_c`.
    width_depleted
        The grid of :math:`\\sigma_d` in pixels.
    scale
        Written with the reciprocal of :math:`\\sqrt{2}` times the standard
        deviation of each Gaussian of each slice.
    weights
        Written with the weight of each Gaussian of each slice.
    num
        Written with the number of Gaussians of each slice.
    """
    spread = width_depleted[d] ** 2
    square = width_max[a, b] ** 2
    for i in slices:
        floor = spread * depletion[o, a, i]
        if field_free[o, a, i] and square > 0:
            num[i] = weight.size
            for k in range(weight.size):
                v = variance[o, a, i, k] * square + floor
                scale[i, k] = 1 / math.sqrt(2 * (max(v, 1e-6) + blur))
                weights[i, k] = weight[k]
        else:
            num[i] = 1
            scale[i, 0] = 1 / math.sqrt(2 * (max(floor, 1e-6) + blur))
            weights[i, 0] = 1.0


@numba.njit(cache=True, nogil=True)
def _misfit(
    fraction: np.ndarray,
    error: np.ndarray,
    position: np.ndarray,
    index: np.ndarray,
    blur: float,
    field_free: np.ndarray,
    variance: np.ndarray,
    weight: np.ndarray,
    depletion: np.ndarray,
    width_max: np.ndarray,
    width_depleted: np.ndarray,
    offset: np.ndarray,
    tilt: np.ndarray,
    center_offset: np.ndarray,
    center_tilt: np.ndarray,
    margin: float,
    prune: bool,
    edges: np.ndarray,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The misfit of the slab model to one track at every point of the grid of
    orientation, :math:`t_c`, :math:`\\sigma_\\text{max}`, and
    :math:`\\sigma_d`, minimized over every offset and tilt of the
    centerline.

    Returns the misfit, and the index of the offset and of the tilt which
    minimize it, each with axes ``(orientation, critical_depth, width_max,
    width_depleted)``.

    The cloud of a slice beyond :math:`t_c` does not depend on
    :math:`\\sigma_\\text{max}`, so the misfit of those slices is summed once
    for each centerline and shared by every :math:`\\sigma_\\text{max}`.

    A fit needs the misfit exactly only where it lies within ``margin`` of
    the least misfit at the same :math:`\\sigma_d`, and at the point with no
    diffusion at all, so the search runs in two passes.
    The first starts at each grid point from a given centerline, the one
    the Gaussian model prefers there, and steps to the best of its
    neighbors until none is better.
    This finds a small misfit quickly, so the least of them at each
    :math:`\\sigma_d` bounds the least misfit there from above.
    The second tries every other centerline, adding the slices of the
    field-free layer brightest first and abandoning the centerline as soon
    as the sum exceeds either the least misfit found so far at that grid
    point or the bound plus ``margin``.
    Every misfit under that sum is then exact, the least among them, and
    every grid point within ``margin`` of it, while a grid point found to lie
    beyond it keeps the misfit of the first pass, which is beyond it too.

    Parameters
    ----------
    fraction
        The measured fractions, with axes ``(slice, pixel)``.
    error
        The uncertainty of a fraction of each slice from read noise.
    position
        The centerline position of each slice before the offset and tilt.
    index
        The index of each slice measured from the middle of the track.
    blur
        The variance, in square pixels, of the uniform spread of the charge
        along the tilted centerline within a slice.
    field_free
        Whether each slice lies in the field-free layer, with axes
        ``(orientation, critical_depth, slice)``.
    variance
        The variance of each Gaussian of the cloud of each slice for a
        field-free layer of unit thickness, from :func:`transit`, with axes
        ``(orientation, critical_depth, slice, mixture)``.
    weight
        The weight of each Gaussian.
    depletion
        The fraction of the depletion region crossed by the charge of each
        slice, :math:`g(t)`, with axes ``(orientation, critical_depth, slice)``.
    width_max
        The grid of :math:`\\sigma_\\text{max}` in pixels, with axes
        ``(critical_depth, width_max)``, since a model may tie it to :math:`t_c`.
    width_depleted
        The grid of :math:`\\sigma_d` in pixels.
    offset
        The grid of centerline offsets in pixels.
    tilt
        The grid of centerline tilts in pixels per slice.
    center_offset
        The index of the offset the search starts from at each grid point.
    center_tilt
        The index of the tilt the search starts from at each grid point.
    margin
        How far above the least misfit at each :math:`\\sigma_d` the misfit
        must be exact.
    prune
        Whether to abandon centerlines, which changes nothing the fit uses
        but the cost; without it every misfit is exact.
    edges
        The pixel boundaries across the track.
    table
        :data:`_erf_table`.
    """
    num_orientation, num_critical, num_slice, num_mixture = variance.shape
    num_max = width_max.shape[1]
    num_depleted = width_depleted.size
    num_offset = offset.size
    num_tilt = tilt.size
    shape = (num_orientation, num_critical, num_max, num_depleted)
    result = np.full(shape, np.inf)
    best_offset = np.zeros(shape, dtype=np.int64)
    best_tilt = np.zeros(shape, dtype=np.int64)

    brightest = np.argsort(error)
    scale = np.empty((num_slice, num_mixture))
    weights = np.empty((num_slice, num_mixture))
    num = np.empty(num_slice, dtype=np.int64)
    model = np.empty(edges.size - 1)
    tried = np.zeros(shape + (num_offset, num_tilt), dtype=np.bool_)
    shifted = np.empty((num_offset, num_tilt, num_slice))
    for c in range(num_offset):
        for e in range(num_tilt):
            for i in range(num_slice):
                shifted[c, e, i] = position[i] + offset[c] + tilt[e] * index[i]

    # the misfit of the slices beyond t_c, for every centerline
    deep = np.zeros((num_orientation, num_critical, num_depleted, num_offset, num_tilt))
    # the slices of the field-free layer, brightest first, and the number
    # of them, for each orientation and t_c
    shallow = np.empty((num_orientation, num_critical, num_slice), dtype=np.int64)
    num_shallow = np.zeros((num_orientation, num_critical), dtype=np.int64)
    for o in range(num_orientation):
        for a in range(num_critical):
            beyond = np.empty(num_slice, dtype=np.int64)
            num_beyond = 0
            for i in brightest:
                if field_free[o, a, i]:
                    shallow[o, a, num_shallow[o, a]] = i
                    num_shallow[o, a] += 1
                else:
                    beyond[num_beyond] = i
                    num_beyond += 1
            beyond = beyond[:num_beyond]
            for d in range(num_depleted):
                _clouds(
                    o,
                    a,
                    0,
                    d,
                    beyond,
                    blur,
                    field_free,
                    variance,
                    weight,
                    depletion,
                    width_max,
                    width_depleted,
                    scale,
                    weights,
                    num,
                )
                for c in range(num_offset):
                    for e in range(num_tilt):
                        deep[o, a, d, c, e] = _misfit_slices(
                            0.0,
                            np.inf,
                            beyond,
                            fraction,
                            error,
                            shifted[c, e],
                            scale,
                            weights,
                            num,
                            edges,
                            table,
                            model,
                        )

    # the first pass, stepping from the given centerline to the best of its
    # neighbors until none is better
    for o in range(num_orientation):
        for a in range(num_critical):
            slices = shallow[o, a, : num_shallow[o, a]]
            for d in range(num_depleted):
                for b in range(num_max):
                    _clouds(
                        o,
                        a,
                        b,
                        d,
                        slices,
                        blur,
                        field_free,
                        variance,
                        weight,
                        depletion,
                        width_max,
                        width_depleted,
                        scale,
                        weights,
                        num,
                    )
                    best = np.inf
                    c0 = center_offset[o, a, b, d]
                    e0 = center_tilt[o, a, b, d]
                    c1 = c0
                    e1 = e0
                    while True:
                        for c in range(max(c0 - 1, 0), min(c0 + 2, num_offset)):
                            for e in range(max(e0 - 1, 0), min(e0 + 2, num_tilt)):
                                if tried[o, a, b, d, c, e]:
                                    continue
                                tried[o, a, b, d, c, e] = True
                                m = _misfit_slices(
                                    deep[o, a, d, c, e],
                                    best if prune else np.inf,
                                    slices,
                                    fraction,
                                    error,
                                    shifted[c, e],
                                    scale,
                                    weights,
                                    num,
                                    edges,
                                    table,
                                    model,
                                )
                                if m < best:
                                    best = m
                                    c1 = c
                                    e1 = e
                        if c1 == c0 and e1 == e0:
                            break
                        c0 = c1
                        e0 = e1
                    result[o, a, b, d] = best
                    best_offset[o, a, b, d] = c1
                    best_tilt[o, a, b, d] = e1

    # the least misfit of the first pass at each sigma_d bounds the least
    # misfit there from above
    bound = np.full(num_depleted, np.inf)
    if prune:
        for d in range(num_depleted):
            for o in range(num_orientation):
                for a in range(num_critical):
                    for b in range(num_max):
                        bound[d] = min(bound[d], result[o, a, b, d] + margin)

    # the second pass, trying every other centerline
    for o in range(num_orientation):
        for a in range(num_critical):
            slices = shallow[o, a, : num_shallow[o, a]]
            for d in range(num_depleted):
                for b in range(num_max):
                    _clouds(
                        o,
                        a,
                        b,
                        d,
                        slices,
                        blur,
                        field_free,
                        variance,
                        weight,
                        depletion,
                        width_max,
                        width_depleted,
                        scale,
                        weights,
                        num,
                    )
                    # the point with no diffusion at all sets the gain of
                    # every fit, so it is found exactly however poor it is
                    none = o == 0 and a == 0 and b == 0 and d == 0
                    limit = np.inf if none else bound[d]
                    best = result[o, a, b, d]
                    c1 = best_offset[o, a, b, d]
                    e1 = best_tilt[o, a, b, d]
                    for c in range(num_offset):
                        for e in range(num_tilt):
                            if tried[o, a, b, d, c, e]:
                                continue
                            least = min(best, limit) if prune else np.inf
                            m = _misfit_slices(
                                deep[o, a, d, c, e],
                                least,
                                slices,
                                fraction,
                                error,
                                shifted[c, e],
                                scale,
                                weights,
                                num,
                                edges,
                                table,
                                model,
                            )
                            if m < min(best, least):
                                best = m
                                c1 = c
                                e1 = e
                    result[o, a, b, d] = best
                    best_offset[o, a, b, d] = c1
                    best_tilt[o, a, b, d] = e1

    return result, best_offset, best_tilt
