"""
The slab model of the article against the Gaussian it replaces, on the
tracks both call flat.
"""

import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
from ._tracks import axis_pixel
from ._fit import models, Fit, fits
from ._samepix import axis_depth
from ._stacked import depth_bins_fine, widths

__all__ = [
    "Comparison",
    "comparison",
]


@dataclasses.dataclass(eq=False)
class Comparison:
    """The slab and Gaussian models fit to the flat tracks of one CCD."""

    chip: str
    """The CCD."""

    num: int
    """The number of tracks flat under both models."""

    misfit: dict[str, float]
    """
    For each model of :data:`ccd_diffusion.tracks.models`, its misfit minus
    that of the Gaussian model, summed over the tracks flat under the slab
    and Gaussian models, each model at its own :math:`\\sigma_d`.
    """

    better: int
    """The number of those tracks the slab model fits better by more than two."""

    worse: int
    """The number of those tracks the slab model fits worse by more than two."""

    depth: na.AbstractScalarArray
    """The center of each depth bin."""

    misfit_depth: na.AbstractScalarArray
    """
    The misfit of the slab model minus that of the Gaussian model, summed
    over the slices of each depth bin of the same tracks.
    """

    residual: dict[str, na.AbstractScalarArray]
    """
    For each model, the model-free width in each depth bin minus the width
    found the same way in the fractions its fits predict.
    """

    @property
    def residual_rms(self) -> dict[str, u.Quantity]:
        """For each model, the root mean square of :attr:`residual` over the depth bins."""
        return {
            model: np.sqrt(np.mean(np.square(r.ndarray)))
            for model, r in self.residual.items()
        }


def _misfit_slices(fit: Fit) -> np.ndarray:
    """
    The misfit of each slice of a track at its best fit.

    Parameters
    ----------
    fit
        The fit.
    """
    track = fit.track
    residual = (track.fraction - fit.predicted) / track.error
    return np.log1p(np.square(residual) / 2).sum(axis_pixel).ndarray


@functools.cache
def comparison(chip: str) -> Comparison:
    """
    Compare the slab and Gaussian models on the flat tracks of the given CCD.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    pairs = [
        (s, g)
        for s, g in zip(fits("slab"), fits("gaussian"))
        if s.track.chip == chip and s.flat and g.flat
    ]
    difference = np.array([s.misfit - g.misfit for s, g in pairs])
    names = {s.track.name for s, _ in pairs}
    misfit = {}
    for model in models:
        mine = {f.track.name: f.misfit for f in fits(model) if f.track.name in names}
        misfit[model] = float(sum(mine[g.track.name] - g.misfit for _, g in pairs))

    # the depth of a slice depends on the orientation each model chose, so
    # the slab model's places the difference of every slice
    edges = depth_bins_fine.ndarray
    misfit_depth = np.zeros(edges.size - 1)
    for s, g in pairs:
        index = np.clip(np.digitize(s.depth.ndarray, edges) - 1, 0, edges.size - 2)
        np.add.at(misfit_depth, index, _misfit_slices(s) - _misfit_slices(g))

    residual = {}
    for model in ("slab", "gaussian"):
        w = widths(chip, model)
        residual[model] = w.best - w.fitted

    return Comparison(
        chip=chip,
        num=len(pairs),
        misfit=misfit,
        better=int(np.sum(difference < -2)),
        worse=int(np.sum(difference > 2)),
        depth=na.ScalarArray((edges[:-1] + edges[1:]) / 2, axes=axis_depth),
        misfit_depth=na.ScalarArray(misfit_depth, axes=axis_depth),
        residual=residual,
    )
