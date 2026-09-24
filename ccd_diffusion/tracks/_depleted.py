"""
The spread inside the depletion region, :math:`\\sigma_d`, is a property of
the drift field rather than of any one track, so it is fit once per CCD by
pooling the misfits of the flat tracks on that CCD over the grid
:data:`ccd_diffusion.tracks.width_depleted`.
"""

import csv
import dataclasses
import functools
import numpy as np
import astropy.units as u
import named_arrays as na
from ._tracks import _directory_data
from ._fit import axis_width_depleted, width_depleted, Scan

__all__ = [
    "Depleted",
    "pooled",
    "save_depleted",
    "depleted",
]


@dataclasses.dataclass(eq=False)
class Depleted:
    """The fit of :math:`\\sigma_d` pooled over the flat tracks on one CCD."""

    chip: str
    """The CCD."""

    width_depleted: na.AbstractScalarArray
    """The grid of :math:`\\sigma_d`."""

    misfit: na.AbstractScalarArray
    """The misfit summed over the flat tracks at each :math:`\\sigma_d`, relative to its minimum."""

    critical_depth: na.AbstractScalarArray
    """The median :math:`t_c` of the flat tracks at each :math:`\\sigma_d`."""

    width_max: na.AbstractScalarArray
    """The median :math:`\\sigma_\\text{max}` of the flat tracks at each :math:`\\sigma_d`."""

    critical_depth_mean: na.AbstractScalarArray
    """
    The mean :math:`t_c` of the flat tracks at each :math:`\\sigma_d`.

    The fits are searched on a grid, so the median can only sit on a grid
    point and only move by a whole step, while the mean moves continuously
    as :math:`\\sigma_d` trades against the field-free wedge.
    """

    critical_depth_error: na.AbstractScalarArray
    """The standard error of :attr:`critical_depth_mean`."""

    width_max_mean: na.AbstractScalarArray
    """The mean :math:`\\sigma_\\text{max}` of the flat tracks at each :math:`\\sigma_d`."""

    width_max_error: na.AbstractScalarArray
    """The standard error of :attr:`width_max_mean`."""

    num: int
    """The number of flat tracks pooled."""

    @property
    def best(self) -> u.Quantity:
        """The :math:`\\sigma_d` which minimizes the pooled misfit."""
        index = np.argmin(self.misfit, axis=axis_width_depleted)
        return self.width_depleted[index].ndarray


def pooled(chip: str, scans: list[Scan], iterations: int = 10) -> Depleted:
    """
    Choose the :math:`\\sigma_d` of a CCD by summing the misfits of its flat
    tracks over the grid of :math:`\\sigma_d` and taking the minimum.

    Which tracks are flat depends on their fits, which depend on
    :math:`\\sigma_d`, so the two are found together: starting from the
    field-free model, the flat tracks are selected at the current
    :math:`\\sigma_d`, their misfits pooled, and :math:`\\sigma_d` updated,
    until it stops changing.

    Parameters
    ----------
    chip
        The CCD.
    scans
        The scans of the tracks on that CCD.
    iterations
        The most rounds of selection to try.
    """
    best = 0 * u.um
    for _ in range(iterations):
        selected = [s for s in scans if s.at(best).flat]
        if not selected:
            raise ValueError(f"no flat tracks on {chip}")
        misfit = selected[0].misfit
        for s in selected[1:]:
            misfit = misfit + s.misfit
        new = width_depleted[np.argmin(misfit, axis=axis_width_depleted)].ndarray
        if new == best:
            break
        best = new

    def values(name):
        return u.Quantity([getattr(s, name).ndarray for s in selected])

    def median(name):
        return na.ScalarArray(np.median(values(name), axis=0), axes=axis_width_depleted)

    def mean(name):
        return na.ScalarArray(values(name).mean(axis=0), axes=axis_width_depleted)

    def error(name):
        v = values(name)
        return na.ScalarArray(
            v.std(axis=0) / np.sqrt(v.shape[0]), axes=axis_width_depleted
        )

    return Depleted(
        chip=chip,
        width_depleted=width_depleted,
        misfit=misfit - misfit.min(),
        critical_depth=median("critical_depth"),
        width_max=median("width_max"),
        critical_depth_mean=mean("critical_depth"),
        critical_depth_error=error("critical_depth"),
        width_max_mean=mean("width_max"),
        width_max_error=error("width_max"),
        num=len(selected),
    )


_path_depleted = _directory_data / "iris_depleted.csv"
"""The file holding the result of :func:`pooled` for every CCD."""


def save_depleted(depleted: tuple[Depleted, ...]) -> None:
    """
    Write the pooled fits to ``data/iris_depleted.csv``, where
    :func:`depleted` will find them.

    Parameters
    ----------
    depleted
        The pooled fit of every CCD.
    """
    fields = [
        "chip",
        "num",
        "width_depleted",
        "misfit",
        "critical_depth",
        "width_max",
        "critical_depth_mean",
        "critical_depth_error",
        "width_max_mean",
        "width_max_error",
    ]
    with open(_path_depleted, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in depleted:
            for i in range(d.width_depleted.size):
                index = {axis_width_depleted: i}

                def um(array, digits):
                    return f"{array[index].ndarray.to_value(u.um):.{digits}f}"

                writer.writerow(
                    dict(
                        chip=d.chip,
                        num=d.num,
                        width_depleted=um(d.width_depleted, 2),
                        misfit=f"{float(d.misfit[index].ndarray):.3f}",
                        critical_depth=f"{float(d.critical_depth[index].ndarray):.3f}",
                        width_max=um(d.width_max, 2),
                        critical_depth_mean=f"{float(d.critical_depth_mean[index].ndarray):.4f}",
                        critical_depth_error=f"{float(d.critical_depth_error[index].ndarray):.4f}",
                        width_max_mean=um(d.width_max_mean, 3),
                        width_max_error=um(d.width_max_error, 3),
                    )
                )


@functools.cache
def depleted(chip: str) -> Depleted:
    """
    Load the pooled fit of :math:`\\sigma_d` on the given CCD from
    ``data/iris_depleted.csv``.

    Parameters
    ----------
    chip
        The CCD, ``FUV1``, ``FUV2``, ``NUV``, or ``SJI``.
    """
    with open(_path_depleted, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["chip"] == chip]
    if not rows:
        raise ValueError(f"no pooled fit for {chip!r}")

    def column(name, unit=1):
        return na.ScalarArray(
            np.array([float(r[name]) for r in rows]) * unit,
            axes=axis_width_depleted,
        )

    return Depleted(
        chip=chip,
        width_depleted=column("width_depleted", u.um),
        misfit=column("misfit"),
        critical_depth=column("critical_depth"),
        width_max=column("width_max", u.um),
        critical_depth_mean=column("critical_depth_mean"),
        critical_depth_error=column("critical_depth_error"),
        width_max_mean=column("width_max_mean", u.um),
        width_max_error=column("width_max_error", u.um),
        num=int(rows[0]["num"]),
    )
