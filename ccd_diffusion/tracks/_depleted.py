import dataclasses
import functools
import concurrent.futures
import numpy as np
import astropy.units as u
import named_arrays as na
from ._tracks import width_pixel
from ._fit import (
    axis_critical_depth,
    axis_width_max,
    axis_offset,
    axis_tilt,
    offset,
    tilt,
    loss,
    Fit,
)
from ._stacked import flat

__all__ = [
    "axis_width_depleted",
    "critical_depth_depleted",
    "width_max_depleted",
    "width_depleted_grid",
    "width_depleted",
    "Depleted",
    "depleted",
]

axis_width_depleted = "width_depleted"
"""The logical axis of the grid of spreads inside the depletion region."""

critical_depth_depleted = na.linspace(0.25, 0.6, axis=axis_critical_depth, num=8)
"""The grid of :math:`t_c` searched by :func:`depleted`."""

width_max_depleted = na.linspace(2, 8, axis=axis_width_max, num=13) * u.um
"""The grid of :math:`\\sigma_\\text{max}` searched by :func:`depleted`."""

width_depleted_grid = na.linspace(0, 3, axis=axis_width_depleted, num=7) * u.um
"""The grid of :math:`\\sigma_d` searched by :func:`depleted`."""


def width_depleted(
    depth: na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
    width_max: u.Quantity | na.AbstractScalarArray,
    width_depleted: u.Quantity | na.AbstractScalarArray,
) -> na.AbstractScalarArray:
    """
    The field-free diffusion model plus an extra spread inside the depletion
    region, in pixels.

    The variance is :math:`\\sigma_\\text{max}^2 (1 - t / t_c)` for
    :math:`t < t_c` plus :math:`\\sigma_d^2 g(t)`, where :math:`g = 1` in the
    field-free layer, since that charge drifts across the full depleted
    thickness, and falls linearly to zero at the gates.

    Parameters
    ----------
    depth
        The fractional depth below the back surface, :math:`t = z / D`.
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface.
    width_depleted
        The extra spread acquired crossing the full depletion region.
    """
    field_free = np.square(width_max) * np.maximum(1 - depth / critical_depth, 0)
    g = np.minimum((1 - depth) / (1 - critical_depth), 1)
    result = np.sqrt(field_free + np.square(width_depleted) * g)
    return (result / width_pixel).to(u.dimensionless_unscaled).value


def _misfit(fit: Fit) -> na.AbstractScalarArray:
    """The misfit of one track on the three-parameter grid, minimized over the nuisance grid."""
    track = fit.track
    position = track.position + offset + tilt * track.index
    w = width_depleted(
        fit.depth,
        critical_depth_depleted,
        width_max_depleted,
        width_depleted_grid,
    )
    return loss(track, position, w).min((axis_offset, axis_tilt))


@dataclasses.dataclass(eq=False)
class Depleted:
    """The fit allowing diffusion inside the depletion region, pooled over one CCD."""

    chip: str
    """The CCD."""

    width_depleted: na.AbstractScalarArray
    """The grid of :math:`\\sigma_d`."""

    misfit: na.AbstractScalarArray
    """The pooled misfit minimized over :math:`t_c` and :math:`\\sigma_\\text{max}`, relative to its minimum."""

    critical_depth: na.AbstractScalarArray
    """The pooled best-fit :math:`t_c` at each :math:`\\sigma_d`."""

    width_max: na.AbstractScalarArray
    """The pooled best-fit :math:`\\sigma_\\text{max}` at each :math:`\\sigma_d`."""

    preferred: u.Quantity
    """The :math:`\\sigma_d` preferred by each flat track on its own."""

    @property
    def best(self) -> u.Quantity:
        """The :math:`\\sigma_d` which minimizes the pooled misfit."""
        index = np.argmin(self.misfit, axis=axis_width_depleted)
        return self.width_depleted[index].ndarray


@functools.cache
def depleted(chip: str) -> Depleted:
    """
    Fit every flat track on the given CCD with the three-parameter model of
    :func:`width_depleted`, with the same nuisance grid as :func:`fit`, and
    pool the misfits.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2`` or ``SJI``.
    """
    tracks = flat(chip)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        misfits = list(pool.map(_misfit, tracks))

    pooled = misfits[0]
    for v in misfits[1:]:
        pooled = pooled + v
    profile = pooled.min((axis_critical_depth, axis_width_max))
    index = np.argmin(pooled, axis=(axis_critical_depth, axis_width_max))
    preferred = u.Quantity(
        [width_depleted_grid[np.argmin(v, axis=v.axes)].ndarray for v in misfits]
    )

    return Depleted(
        chip=chip,
        width_depleted=width_depleted_grid,
        misfit=profile - profile.min(),
        critical_depth=critical_depth_depleted[index],
        width_max=width_max_depleted[index],
        preferred=preferred,
    )
