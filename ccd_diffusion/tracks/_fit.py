import csv
import dataclasses
import functools
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
    Track,
    load,
    _directory_data,
)

__all__ = [
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
]

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

critical_depth = na.linspace(0, 1, axis=axis_critical_depth, num=21)
"""
The grid of fractional field-free thicknesses, :math:`t_c = z_f / D`,
searched by :func:`scan`.
"""

width_max = na.linspace(0, 10, axis=axis_width_max, num=21) * u.um
"""The grid of back-surface diffusion widths, :math:`\\sigma_\\text{max}`, searched by :func:`scan`."""

width_depleted = na.linspace(0, 3, axis=axis_width_depleted, num=13) * u.um
"""
The grid of spreads acquired crossing the full depletion region,
:math:`\\sigma_d`, searched by :func:`scan`.

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

_edges = na.arange(-half_width - 1, half_width + 1, axis=axis_pixel) + 0.5
"""The pixel boundaries across the track, in pixels from the central pixel."""


def width(
    depth: na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
    width_max: u.Quantity | na.AbstractScalarArray,
    width_depleted: u.Quantity | na.AbstractScalarArray = 0 * u.um,
) -> na.AbstractScalarArray:
    """
    The standard deviation of the charge cloud, in pixels, for charge
    deposited at the given fractional depth.

    This is the diffusion model of the article,

    .. math::

        \\sigma^2(t) = \\sigma_\\text{max}^2 \\left( 1 - \\frac{t}{t_c} \\right)
        + \\sigma_d^2 \\, g(t),

    where the first term applies only in the field-free layer, :math:`t <
    t_c`, and :math:`g = 1` there, since that charge drifts across the full
    depleted thickness, falling linearly to zero at the gates,
    :math:`g = \\min[(1 - t) / (1 - t_c), 1]`.

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
    """
    tc = np.maximum(critical_depth, 1e-6)
    wedge = np.where(depth < critical_depth, np.maximum(1 - depth / tc, 0), 0)
    field_free = np.square(width_max) * wedge
    g = np.minimum((1 - depth) / np.maximum(1 - critical_depth, 1e-6), 1)
    result = np.sqrt(field_free + np.square(width_depleted) * g)
    return (result / width_pixel).to(u.dimensionless_unscaled).value


def fractions(
    position: na.AbstractScalarArray,
    width: na.AbstractScalarArray,
    slope: float,
) -> na.AbstractScalarArray:
    """
    The fraction of each slice's charge expected in each of the
    :math:`2 h + 1` pixels across the track.

    The charge deposited in a slice is spread uniformly along the tilted
    centerline and diffuses as a Gaussian of the given width, which is
    approximated by a Gaussian of variance :math:`\\sigma^2 + m^2 / 12`,
    where :math:`m` is the slope of the centerline.

    Parameters
    ----------
    position
        The centerline position of each slice in pixels from the central pixel.
    width
        The standard deviation of the charge cloud in pixels.
    slope
        The tilt of the centerline in pixels per slice.
    """
    s = np.sqrt(np.square(np.maximum(width, 1e-3)) + np.square(slope) / 12)
    x = (_edges - position) / (s * np.sqrt(2))
    x = x.astype(np.float32)
    cdf = scipy.special.erf(x) / 2
    result = np.diff(cdf, axis=axis_pixel)
    return result / np.maximum(result.sum(axis_pixel), 1e-9)


def loss(
    track: Track,
    position: na.AbstractScalarArray,
    width: na.AbstractScalarArray,
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
        The standard deviation of the charge cloud in pixels.
    """
    residual = (track.fraction - fractions(position, width, track.slope)) / track.error
    return np.log1p(np.square(residual) / 2).sum((axis_slice, axis_pixel))


@dataclasses.dataclass(eq=False)
class Fit:
    """The result of fitting the diffusion model to a single track."""

    track: Track
    """The track that was fit."""

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
        """The best-fit charge cloud width of each slice in pixels."""
        return width(
            self.depth, self.critical_depth, self.width_max, self.width_depleted
        )

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

    Every attribute below :attr:`track` is an array over
    :data:`axis_width_depleted`, holding the corresponding attribute of the
    :class:`Fit` at that :math:`\\sigma_d`.
    """

    track: Track
    """The track that was fit."""

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
            width_depleted=grid[i].ndarray,
            orientation=int(self.orientation[i].ndarray),
            critical_depth=float(self.critical_depth[i].ndarray),
            width_max=self.width_max[i].ndarray,
            offset=float(self.offset[i].ndarray),
            tilt=float(self.tilt[i].ndarray),
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

    The misfit of a model is the sum over slices of this table, since each
    slice contributes through its centerline position and its width alone,
    so :func:`scan` assembles every model on its grids from one table
    instead of evaluating :func:`loss` afresh.

    Parameters
    ----------
    track
        The track to tabulate.
    """
    position = track.position + offset + tilt * track.index
    q = fractions(position, _width_table, track.slope)
    residual = (track.fraction - q) / track.error
    return np.log1p(np.square(residual) / 2).sum(axis_pixel)


def scan(track: Track) -> Scan:
    """
    Fit the diffusion model to a single track at every :math:`\\sigma_d`.

    The grid of :data:`critical_depth`, :data:`width_max`,
    :data:`width_depleted`, and both track orientations is searched
    exhaustively, and at each grid point the :func:`loss` is minimized over
    the nuisance grid of :data:`offset` and :data:`tilt`.
    The misfit of each model is assembled from the :func:`table` of the
    track, interpolated linearly in width.

    Parameters
    ----------
    track
        The track to fit.
    """
    misfit_slice = table(track)
    depth = track.depth
    index_slice = na.arange(0, track.length, axis=axis_slice)
    last = _width_table.size - 2

    result = []
    for orientation in (+1, -1):
        t = depth if orientation > 0 else 1 - depth
        rows = []
        for tc in critical_depth.ndarray:
            w = width(t, tc, width_max, width_depleted) / _width_step
            k = np.minimum(np.floor(w), last).astype(int)
            weight = w - k
            lo = misfit_slice[{axis_slice: index_slice, "table": k}]
            hi = misfit_slice[{axis_slice: index_slice, "table": k + 1}]
            rows.append((lo + weight * (hi - lo)).sum(axis_slice))
        result.append(na.stack(rows, axis=axis_critical_depth))
    v = na.stack(result, axis="orientation")

    axes_nuisance = (axis_offset, axis_tilt)
    axes_model = (axis_critical_depth, axis_width_max, "orientation")
    v_model = v.min(axes_nuisance)
    v_best = v_model.min(axes_model)
    index = np.argmin(v_model, axis=axes_model)
    index_nuisance = np.argmin(v[index], axis=axes_nuisance)

    v_none = v_model[
        dict(orientation=0, critical_depth=0, width_max=0, width_depleted=0)
    ]

    ok = v_model < v_best + 2
    tc_ok = ok.any((axis_width_max, "orientation"))
    sm_ok = ok.any((axis_critical_depth, "orientation"))
    inf = np.inf * u.um

    return Scan(
        track=track,
        misfit=v_best,
        orientation=np.where(index["orientation"] == 0, +1, -1),
        critical_depth=critical_depth[index],
        width_max=width_max[index],
        offset=offset[index_nuisance],
        tilt=tilt[index_nuisance],
        gain=v_none - v_best,
        critical_depth_min=np.where(tc_ok, critical_depth, np.inf).min(
            axis_critical_depth
        ),
        critical_depth_max=np.where(tc_ok, critical_depth, -np.inf).max(
            axis_critical_depth
        ),
        width_max_min=np.where(sm_ok, width_max, inf).min(axis_width_max),
        width_max_max=np.where(sm_ok, width_max, -inf).max(axis_width_max),
    )


def fit(track: Track, width_depleted: u.Quantity = 0 * u.um) -> Fit:
    """
    Fit the diffusion model to a single track at a given :math:`\\sigma_d`.

    Parameters
    ----------
    track
        The track to fit.
    width_depleted
        The spread inside the depletion region, which is shared by every
        track on a CCD and is chosen by :func:`fit_all`.
    """
    return scan(track).at(width_depleted)


_path_fits = _directory_data / "iris_fits.csv"
"""The file holding the result of :func:`fit_all` for every track."""

_fields_fits = [
    "name",
    "width_depleted",
    "orientation",
    "critical_depth",
    "width_max",
    "offset",
    "tilt",
    "gain",
    "critical_depth_min",
    "critical_depth_max",
    "width_max_min",
    "width_max_max",
    "width_depleted_preferred",
]


def fit_all(
    tracks: tuple[Track, ...],
) -> tuple[tuple[Fit, ...], tuple["ccd_diffusion.tracks.Depleted", ...]]:
    """
    Fit the diffusion model to every given track, using one thread per CPU,
    choosing one :math:`\\sigma_d` per CCD by pooling the misfits of its
    flat tracks (see :func:`ccd_diffusion.tracks.pooled`).

    Parameters
    ----------
    tracks
        The tracks to fit.

    Returns
    -------
    The fit of every track, in the given order, and the pooled fit of
    :math:`\\sigma_d` on every CCD.
    """
    from ._depleted import pooled

    with concurrent.futures.ThreadPoolExecutor() as pool:
        scans = list(pool.map(scan, tracks))

    result = {}
    depleted = []
    for chip in dict.fromkeys(t.chip for t in tracks):
        mine = [s for s in scans if s.track.chip == chip]
        d = pooled(chip, mine)
        depleted.append(d)
        for s in mine:
            result[s.track.name] = s.at(d.best)
    return tuple(result[t.name] for t in tracks), tuple(depleted)


def save(
    fits: tuple[Fit, ...],
    depleted: tuple["ccd_diffusion.tracks.Depleted", ...],
) -> None:
    """
    Write the given fits to ``data/iris_fits.csv``, where :func:`fits` will
    find them, and the pooled fits of :math:`\\sigma_d` to
    ``data/iris_depleted.csv``, where :func:`ccd_diffusion.tracks.depleted`
    will.

    Fitting every track takes a while, so the fits are stored in the
    repository alongside the tracks and only recomputed by running
    ``python -m ccd_diffusion.tracks fit``.

    Parameters
    ----------
    fits
        The fits to save, one per track returned by :func:`load`.
    depleted
        The pooled fit of :math:`\\sigma_d` on every CCD.
    """
    from ._depleted import save_depleted

    with open(_path_fits, "w", newline="") as f:
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
    save_depleted(depleted)


@functools.cache
def fits() -> tuple[Fit, ...]:
    """
    Load the result of :func:`fit_all` for every track returned by
    :func:`load` from ``data/iris_fits.csv``.
    """
    tracks = {track.name: track for track in load()}
    with open(_path_fits, newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Fit(
            track=tracks[r["name"]],
            width_depleted=float(r["width_depleted"]) * u.um,
            orientation=int(r["orientation"]),
            critical_depth=float(r["critical_depth"]),
            width_max=float(r["width_max"]) * u.um,
            offset=float(r["offset"]),
            tilt=float(r["tilt"]),
            gain=float(r["gain"]),
            critical_depth_min=float(r["critical_depth_min"]),
            critical_depth_max=float(r["critical_depth_max"]),
            width_max_min=float(r["width_max_min"]) * u.um,
            width_max_max=float(r["width_max_max"]) * u.um,
            width_depleted_preferred=float(r["width_depleted_preferred"]) * u.um,
        )
        for r in rows
    )
