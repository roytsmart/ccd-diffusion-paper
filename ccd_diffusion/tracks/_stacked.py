import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
from ._tracks import axis_slice, axis_pixel, half_width, width_pixel
from ._fit import Fit, fits, width, fractions
from ._samepix import axis_depth

__all__ = [
    "depth_bins_fine",
    "widths_trial",
    "flat",
    "Stack",
    "stack",
    "Widths",
    "widths",
]

depth_bins_fine = na.linspace(0, 1, axis=axis_depth, num=21)
"""The edges of the depth bins used by :func:`stack` and :func:`widths`."""

widths_trial = na.linspace(0, 10, axis="width", num=41) * u.um
"""The trial widths searched by :func:`widths` in each depth bin."""

_offsets = na.arange(-half_width, half_width + 1, axis=axis_pixel)
"""The offset of each pixel of a cutout from its central pixel."""


def flat(chip: str) -> list[Fit]:
    """
    The fits of the flat tracks on the given CCD.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    return [f for f in fits() if f.track.chip == chip and f.flat]


@dataclasses.dataclass(eq=False)
class Stack:
    """Every slice of every flat track on one CCD, aligned on its fitted centerline."""

    chip: str
    """The CCD."""

    depth: na.AbstractScalarArray
    """The edges of the depth bins."""

    offset: na.AbstractScalarArray
    """The edges of the transverse bins, in pixels from the fitted centerline."""

    image: na.AbstractScalarArray
    """The mean charge fraction per pixel in each bin, with axes ``(depth, offset)``."""


@functools.cache
def stack(chip: str, num_offset: int = 28) -> Stack:
    """
    Stack the flat tracks on the given CCD, placing every slice at its
    fractional depth and aligning it on its fitted centerline, with each
    pixel's charge spread uniformly over its own width.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    num_offset
        The number of transverse bins across the cutout.
    """
    t, x, w = [], [], []
    for f in flat(chip):
        distance = _offsets - f.position
        fraction = np.where(f.track.usable, f.track.fraction, np.nan)
        shape = na.shape_broadcasted(distance, fraction)
        t.append(na.broadcast_to(f.depth, shape).ndarray.ravel())
        x.append(na.broadcast_to(distance, shape).ndarray.ravel())
        w.append(na.broadcast_to(fraction, shape).ndarray.ravel())
    t, x, w = np.concatenate(t), np.concatenate(x), np.concatenate(w)
    keep = np.isfinite(w)
    t, x, w = t[keep], x[keep], w[keep]

    edges_depth = depth_bins_fine.ndarray
    edges_offset = np.linspace(-half_width - 0.5, half_width + 0.5, num_offset + 1)
    lower = np.maximum(x[:, np.newaxis] - 0.5, edges_offset[np.newaxis, :-1])
    upper = np.minimum(x[:, np.newaxis] + 0.5, edges_offset[np.newaxis, 1:])
    overlap = np.clip(upper - lower, 0, None) / np.diff(edges_offset)
    index = np.clip(np.digitize(t, edges_depth) - 1, 0, edges_depth.size - 2)
    total = np.zeros((edges_depth.size - 1, num_offset))
    np.add.at(total, index, w[:, np.newaxis] * overlap)
    num = np.histogram(
        np.concatenate([f.depth.ndarray[f.track.usable.ndarray] for f in flat(chip)]),
        bins=edges_depth,
    )[0]

    return Stack(
        chip=chip,
        depth=depth_bins_fine,
        offset=na.ScalarArray(edges_offset, axes="offset"),
        image=na.ScalarArray(total / num[:, np.newaxis], axes=(axis_depth, "offset")),
    )


@dataclasses.dataclass(eq=False)
class Widths:
    """The diffusion width as a function of depth on one CCD."""

    chip: str
    """The CCD."""

    depth: na.AbstractScalarArray
    """The center of each depth bin."""

    best: na.AbstractScalarArray
    """The width which minimizes the summed misfit of the slices in each bin."""

    lower: na.AbstractScalarArray
    """The smallest width within two units of misfit of :attr:`best`."""

    upper: na.AbstractScalarArray
    """The largest width within two units of misfit of :attr:`best`."""

    fitted: na.AbstractScalarArray
    """The mean of the per-track fitted width curves, including :math:`\\sigma_d`, in each bin."""


@functools.cache
def widths(chip: str) -> Widths:
    """
    Measure the diffusion width in each depth bin without a parametric model,
    by summing the misfit of every slice of every flat track on the given CCD
    over a grid of trial widths and minimizing in each bin.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    edges = depth_bins_fine.ndarray
    w = (widths_trial / width_pixel).to(u.dimensionless_unscaled).value
    nll = np.zeros((edges.size - 1, widths_trial.size))
    for f in flat(chip):
        q = fractions(f.position, w, f.track.slope)
        r = (f.track.fraction - q) / f.track.error
        v = np.log1p(np.square(r) / 2).sum(axis_pixel)
        v = v * f.track.usable
        index = np.clip(np.digitize(f.depth.ndarray, edges) - 1, 0, nll.shape[0] - 1)
        np.add.at(nll, index, v.transpose((axis_slice, "width")).ndarray)

    trial = widths_trial.ndarray
    best = trial[np.argmin(nll, axis=1)]
    ok = nll < nll.min(axis=1, keepdims=True) + 2
    lower = u.Quantity([trial[row].min() for row in ok])
    upper = u.Quantity([trial[row].max() for row in ok])

    centers = na.ScalarArray((edges[:-1] + edges[1:]) / 2, axes=axis_depth)
    curves = na.stack(
        [
            width(centers, f.critical_depth, f.width_max, f.width_depleted)
            for f in flat(chip)
        ],
        axis="track",
    )

    return Widths(
        chip=chip,
        depth=centers,
        best=na.ScalarArray(best, axes=axis_depth),
        lower=na.ScalarArray(lower, axes=axis_depth),
        upper=na.ScalarArray(upper, axes=axis_depth),
        fitted=curves.mean("track") * width_pixel,
    )
