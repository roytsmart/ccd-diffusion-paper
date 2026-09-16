import csv
import dataclasses
import functools
import concurrent.futures
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
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
    "axis_offset",
    "axis_tilt",
    "critical_depth",
    "width_max",
    "offset",
    "tilt",
    "width",
    "fractions",
    "loss",
    "Fit",
    "fit",
    "fit_all",
    "fits",
    "save",
]

axis_critical_depth = "critical_depth"
"""The logical axis of the grid of fractional field-free thicknesses."""

axis_width_max = "width_max"
"""The logical axis of the grid of back-surface diffusion widths."""

axis_offset = "offset"
"""The logical axis of the grid of centerline offsets."""

axis_tilt = "tilt"
"""The logical axis of the grid of centerline tilt corrections."""

critical_depth = na.linspace(0, 1, axis=axis_critical_depth, num=21)
"""
The grid of fractional field-free thicknesses, :math:`t_c = z_f / D`,
searched by :func:`fit`.
"""

width_max = na.linspace(0, 10, axis=axis_width_max, num=21) * u.um
"""The grid of back-surface diffusion widths, :math:`\\sigma_\\text{max}`, searched by :func:`fit`."""

offset = na.linspace(-0.6, 0.6, axis=axis_offset, num=25)
"""The grid of centerline offsets (in pixels) marginalized over by :func:`fit`."""

tilt = na.linspace(-0.03, 0.03, axis=axis_tilt, num=5)
"""The grid of centerline tilt corrections (in pixels per slice) marginalized over by :func:`fit`."""

_edges = na.arange(-half_width - 1, half_width + 1, axis=axis_pixel) + 0.5
"""The pixel boundaries across the track, in pixels from the central pixel."""


def width(
    depth: na.AbstractScalarArray,
    critical_depth: float | na.AbstractScalarArray,
    width_max: u.Quantity | na.AbstractScalarArray,
) -> na.AbstractScalarArray:
    """
    The standard deviation of the charge cloud, in pixels, for charge
    deposited at the given fractional depth.

    This is the field-free diffusion model of the article,
    :math:`\\sigma(t) = \\sigma_\\text{max} \\sqrt{1 - t / t_c}` for
    :math:`t < t_c` and zero otherwise.

    Parameters
    ----------
    depth
        The fractional depth below the back surface, :math:`t = z / D`.
    critical_depth
        The fractional thickness of the field-free region, :math:`t_c`.
    width_max
        The width of the charge cloud at the back surface.
    """
    tc = np.maximum(critical_depth, 1e-6)
    result = width_max * np.sqrt(np.maximum(1 - depth / tc, 0))
    result = np.where(depth < critical_depth, result, 0 * width_max)
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
    """The decrease in :func:`loss` relative to a model with no diffusion."""

    critical_depth_min: float
    """The smallest :math:`t_c` within two units of :func:`loss` of the best fit."""

    critical_depth_max: float
    """The largest :math:`t_c` within two units of :func:`loss` of the best fit."""

    width_max_min: u.Quantity
    """The smallest :math:`\\sigma_\\text{max}` within two units of :func:`loss` of the best fit."""

    width_max_max: u.Quantity
    """The largest :math:`\\sigma_\\text{max}` within two units of :func:`loss` of the best fit."""

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
        return width(self.depth, self.critical_depth, self.width_max)

    @property
    def tight(self) -> bool:
        """Whether the track constrains :math:`t_c` to within 0.15."""
        gain = self.gain > 10
        interval = (self.critical_depth_max - self.critical_depth_min) <= 0.15
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
    def flat(self) -> bool:
        """Whether the track is :attr:`tight` and has no Bragg rise along its length."""
        return self.tight and (self.bragg < 1.5)


def fit(track: Track) -> Fit:
    """
    Fit the diffusion model to a single track.

    The grid of :data:`critical_depth`, :data:`width_max`, and both track
    orientations is searched exhaustively, and at each grid point the
    :func:`loss` is minimized over the nuisance grid of :data:`offset` and
    :data:`tilt`.

    Parameters
    ----------
    track
        The track to fit.
    """
    position = track.position + offset + tilt * track.index
    depth = track.depth

    result = []
    for orientation in (+1, -1):
        t = depth if orientation > 0 else 1 - depth
        rows = []
        for tc in critical_depth.ndarray:
            w = width(t, tc, width_max)
            v = loss(track, position, w)
            rows.append(v)
        result.append(na.stack(rows, axis=axis_critical_depth))
    v = na.stack(result, axis="orientation")

    axes_nuisance = (axis_offset, axis_tilt)
    v_model = v.min(axes_nuisance)
    v_best = v_model.min()
    index = np.argmin(
        v_model, axis=(axis_critical_depth, axis_width_max, "orientation")
    )
    index_nuisance = np.argmin(v[index], axis=axes_nuisance)

    v_none = v_model[dict(orientation=0, critical_depth=0, width_max=0)]

    ok = v_model < v_best + 2
    tc_ok = critical_depth[ok.any((axis_width_max, "orientation"))]
    sm_ok = width_max[ok.any((axis_critical_depth, "orientation"))]

    return Fit(
        track=track,
        orientation=+1 if index["orientation"].ndarray == 0 else -1,
        critical_depth=float(critical_depth[index].ndarray),
        width_max=width_max[index].ndarray,
        offset=float(offset[index_nuisance].ndarray),
        tilt=float(tilt[index_nuisance].ndarray),
        gain=float((v_none - v_best).ndarray),
        critical_depth_min=float(tc_ok.min().ndarray),
        critical_depth_max=float(tc_ok.max().ndarray),
        width_max_min=sm_ok.min().ndarray,
        width_max_max=sm_ok.max().ndarray,
    )


_path_fits = _directory_data / "iris_fits.csv"
"""The file holding the result of :func:`fit` for every track."""

_fields_fits = [
    "name",
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
]


def fit_all(tracks: tuple[Track, ...]) -> tuple[Fit, ...]:
    """
    Fit the diffusion model to every given track, using one thread per CPU.

    Parameters
    ----------
    tracks
        The tracks to fit.
    """
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return tuple(pool.map(fit, tracks))


def save(fits: tuple[Fit, ...]) -> None:
    """
    Write the given fits to ``data/iris_fits.csv``, where :func:`fits` will
    find them.

    Fitting every track takes several minutes, so the fits are stored in the
    repository alongside the tracks and only recomputed by running
    ``python -m ccd_diffusion.tracks``.

    Parameters
    ----------
    fits
        The fits to save, one per track returned by :func:`load`.
    """
    with open(_path_fits, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields_fits)
        writer.writeheader()
        for r in fits:
            writer.writerow(
                dict(
                    name=r.track.name,
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
                )
            )


@functools.cache
def fits() -> tuple[Fit, ...]:
    """
    Load the result of :func:`fit` for every track returned by :func:`load`
    from ``data/iris_fits.csv``.
    """
    tracks = {track.name: track for track in load()}
    with open(_path_fits, newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Fit(
            track=tracks[r["name"]],
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
        )
        for r in rows
    )
