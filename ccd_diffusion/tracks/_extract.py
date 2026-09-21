"""
Extract the glancing particle tracks from the IRIS level-1 images.

The images are processed one block at a time, where a block is a set of
frames which share a background: a stretch of a hundred frames of a
sit-and-stare, a day of a limb campaign, or the frames of one raster repeat.
The quiet frames of a block give a trimmed-mean background and a read-noise
map, and the frames searched for tracks are compared against it.

This module works on plain :mod:`numpy` arrays rather than
:mod:`named_arrays`, since it is image labeling and connected-component
bookkeeping with :mod:`scipy.ndimage`, and it is the only part of the
package which does.
"""

import csv
import dataclasses
import pathlib
import warnings
import numpy as np
import scipy.stats
import scipy.ndimage
from ._tracks import (
    axis_slice,
    axis_pixel,
    half_width,
    Track,
    frames,
    _directory_data,
)
from ._archive import directory_default, download, read, path

__all__ = [
    "off_limb",
    "length_minimum",
    "slope_maximum",
    "charge_minimum",
    "threshold",
    "Block",
    "blocks",
    "background",
    "Component",
    "find",
    "census",
    "extract",
    "save_tracks",
    "save_census",
    "load_census",
]

length_minimum = 12
"""The shortest track kept, in rows."""

slope_maximum = 0.35
"""The steepest track kept, in pixels per row."""

charge_minimum = 240.0
"""The least charge in a row, in electrons, for the row to constrain the centerline."""

threshold = 5.0
"""Pixels this many times the read noise above the background are labeled."""

_width_maximum = 8
"""The widest connected group, in pixels, considered as a track."""

_saturation = 15000.0 * 6
"""Charge in electrons above which a pixel is saturated."""

_touching = np.ones((3, 3), dtype=bool)
"""
Pixels which touch at a corner belong to the same group.

A glancing track steps sideways as it goes, and where the charge is sharp
the step shares no edge with the row before, so grouping by shared edges
alone breaks such a track into pieces.
"""

_gain = {"FUV": 6.0, "NUV": 18.0, "SJI": 18.0}
"""The camera gain in electrons per data number."""


def _camera(image: str) -> str:
    """The camera behind an image type: ``FUV``, ``NUV``, or ``SJI`` for any slit-jaw channel."""
    return image if image in ("FUV", "NUV") else "SJI"


_config = {
    "2014": dict(
        block="hundred",
        quiet="all",
        search="all",
        noise_maximum=6,
        mask="columns",
        camera="FUV",
    ),
    "2014b": dict(
        block="halves",
        quiet="quiet",
        search="saa",
        noise_maximum=6,
        mask="rows",
        camera="FUV",
    ),
    "2018": dict(
        block="repeat",
        quiet="quiet",
        search="limb",
        noise_maximum=None,
        mask=None,
        camera="FUV",
    ),
    "2018may": dict(
        block="day",
        quiet="quiet",
        search="saa",
        noise_maximum=6,
        mask="lines",
        camera="FUV",
    ),
    "sji": dict(
        block="channel",
        quiet="quiet",
        search="saa",
        noise_maximum=3,
        mask="limb",
        camera="SJI",
    ),
}
"""
How each of the original campaigns is processed: how its frames are grouped into blocks,
which frames form the background (``all`` of them, or only the ``quiet``
ones outside the SAA), which frames are searched for tracks (``all``,
``saa``, or those pointed off the ``limb``), the read noise in data numbers
above which a pixel is ignored, and how the mask of pixels with solar signal
is built: ``columns`` and ``rows`` cut those with many raised pixels,
``median`` also removes hot pixels, ``lines`` removes hot pixels and
raised columns but keeps every row, and ``limb`` keeps the part of a
slit-jaw field off the limb.

The read-noise limit and the mask were tuned on the ``camera`` named, and
a block of another camera in the same campaign takes the defaults for its
camera instead.
"""

_config_default = dict(
    block="channel", quiet="quiet", search="saa", noise_maximum="camera", mask="camera"
)
"""
How a campaign not listed in :data:`_config` is processed: one block per
day and camera, the background from the frames outside the anomaly, the
frames inside it searched, and the read-noise limit and the mask set by
the camera.
"""

_noise_maximum = {"FUV": 6.0, "NUV": 3.0, "SJI": 3.0}
"""The read noise in data numbers above which a pixel is ignored, by camera."""

_mask_camera = {"FUV": "lines", "NUV": "lines", "SJI": "limb"}
"""
The mask by camera: each spectrograph loses its emission-line columns and
hot pixels, every quadrant of its CCDs levelled on its own pedestal, and
the slit-jaw imager keeps only the part of its field off the limb.
"""

_limb_margin = 15.0
"""
How far above the photospheric limb, in arcseconds, a pixel must look for
it to be searched for tracks.

The chromosphere seen in the emission lines and the slit-jaw passbands
stands a few arcseconds above the limb the headers give, and spicules
reach higher still, so the margin keeps solar structure out of the search
on every camera.
"""


def _configuration(dataset: str, camera: None | str = None) -> dict:
    """
    How a campaign is processed, :data:`_config_default` if it is not
    listed, with the read-noise limit and the mask of a hand-tuned campaign
    replaced by the camera defaults for a camera other than the one it was
    tuned on.
    """
    result = dict(_config.get(dataset, _config_default))
    tuned = result.pop("camera", None)
    if camera is not None and tuned is not None and camera != tuned:
        result["noise_maximum"] = "camera"
        result["mask"] = "camera"
    return result


@dataclasses.dataclass(eq=False)
class Block:
    """A set of frames which share a background."""

    dataset: str
    """The campaign."""

    name: str
    """A label for the block within the campaign."""

    quiet: list[dict[str, str]]
    """The frames the background is estimated from."""

    search: list[dict[str, str]]
    """The frames searched for tracks."""

    @property
    def frames(self) -> list[dict[str, str]]:
        """Every frame of the block, each once."""
        seen = set()
        result = []
        for f in self.quiet + self.search:
            if f["fsn"] not in seen:
                seen.add(f["fsn"])
                result.append(f)
        return result


def _header_values(rows, directory, keys):
    """Read keywords from the headers of the given frames, downloading them first."""
    download(rows, directory)
    result = []
    for f in rows:
        _, header = read(f, directory)
        result.append({k: header.get(k) for k in keys})
    return result


def blocks(dataset: str, directory: None | pathlib.Path = None) -> list[Block]:
    """
    Group the frames of a campaign into blocks which share a background.

    Parameters
    ----------
    dataset
        The campaign, processed as :data:`_config` says if it is listed there
        and as :data:`_config_default` says otherwise.
    directory
        The cache directory, :data:`ccd_diffusion.tracks.directory_default`
        if :obj:`None`.
    """
    if directory is None:
        directory = directory_default
    config = _configuration(dataset)
    rows = sorted(
        (f for f in frames() if f["dataset"] == dataset), key=lambda f: f["time"]
    )

    def quiet(fs):
        return [f for f in fs if config["quiet"] == "all" or f["saa"] == "0"]

    # every camera has its own background, so the frames are split by image
    # type first and the grouping rule applies within each
    by_image: dict[str, list] = {}
    for f in rows:
        by_image.setdefault(f["image"], []).append(f)
    groups = {}
    for image, mine in by_image.items():
        prefix = "" if len(by_image) == 1 else f"{image}_"
        if config["block"] == "hundred":
            for i in range(0, len(mine), 100):
                groups[f"{prefix}{i // 100:02d}"] = mine[i : i + 100]
        elif config["block"] == "halves":
            half = len(mine) // 2
            groups[f"{prefix}0"] = mine[:half]
            groups[f"{prefix}1"] = mine[half:]
        elif config["block"] == "day":
            for f in mine:
                groups.setdefault(f"{prefix}{f['time'][:10]}", []).append(f)
        elif config["block"] == "channel":
            for f in mine:
                groups.setdefault(f"{f['time'][:10]}_{f['image'][-4:]}", []).append(f)
        elif config["block"] == "repeat":
            # the raster repeat is only in the header, so the frames are fetched here
            keys = _header_values(mine, directory, ["IIOLRPT"])
            for f, k in zip(mine, keys):
                groups.setdefault(f"{prefix}{k['IIOLRPT']}", []).append(f)
        else:
            raise ValueError(config["block"])

    result = []
    for name, fs in groups.items():
        if config["search"] == "all":
            search = list(fs)
        elif config["search"] == "saa":
            search = [f for f in fs if f["saa"] == "1"]
        elif config["search"] == "limb":
            keys = _header_values(fs, directory, ["XCEN"])
            search = [
                f for f, k in zip(fs, keys) if k["XCEN"] is not None and k["XCEN"] > 995
            ]
        else:
            raise ValueError(config["search"])
        result.append(Block(dataset=dataset, name=name, quiet=quiet(fs), search=search))
    return result


_rows_per_band = 64
"""How many rows :func:`background` works on at once."""

_quiet_maximum = 120
"""
The most frames a block's background is estimated from.

More do not improve it, and each full-resolution frame is 18 MB, so a
long block's quiet frames are thinned evenly to this many.
"""


def background(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The trimmed-mean background of a stack of frames and the read-noise map
    from the median absolute deviation of the residuals, both in data
    numbers, with unread pixels set to NaN.

    Parameters
    ----------
    stack
        The frames, with the frame index first.
    """
    # a stack of a hundred frames is gigabytes, and the sort behind the
    # trimmed mean and the deviations behind the noise would each copy it,
    # so both are taken a band of rows at a time
    result = np.empty(stack.shape[1:], dtype=np.float32)
    noise = np.empty(stack.shape[1:], dtype=np.float32)
    for r0 in range(0, stack.shape[1], _rows_per_band):
        band = stack[:, r0 : r0 + _rows_per_band]
        finite = np.isfinite(band).all(0) & (band > 0).all(0)
        mean = scipy.stats.trim_mean(np.nan_to_num(band, nan=0.0), 0.2, axis=0)
        residual = band - mean
        deviation = (
            np.nanmedian(np.abs(residual - np.nanmedian(residual, axis=0)), axis=0)
            * 1.4826
        )
        mean[~finite] = np.nan
        deviation[~finite] = np.nan
        result[r0 : r0 + _rows_per_band] = mean
        noise[r0 : r0 + _rows_per_band] = deviation
    return result, noise


_taps = 4
"""
The number of readout amplifiers of each CCD, one per quadrant, each of
which reads its quadrant at its own pedestal.
"""


def _pedestal(values: np.ndarray) -> float:
    """
    The pedestal of a set of read pixels: the peak of their histogram,
    refined as the mean of the pixels within the peak's full width at
    half maximum.

    The dark pixels of a quadrant pile into a sharp peak at the pedestal
    while the solar signal spreads over hundreds of data numbers, so the
    peak stands wherever the disk is in the quadrant, unlike a trimmed
    mean, which a slit-jaw quadrant half covered by the disk pulls tens of
    data numbers above the pedestal.

    Parameters
    ----------
    values
        The read pixels of one quadrant, in data numbers.
    """
    lo, hi = np.percentile(values, [0.5, 60])
    edges = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5)
    counts = np.histogram(values, edges)[0].astype(float)
    counts = scipy.ndimage.uniform_filter1d(counts, 3)
    i = int(np.argmax(counts))
    peak = edges[i] + 0.5
    half = counts[i] / 2
    lower = i
    while lower > 0 and counts[lower - 1] > half:
        lower -= 1
    upper = i
    while upper < len(counts) - 1 and counts[upper + 1] > half:
        upper += 1
    width = max(edges[upper + 1] - edges[lower], 1.0)
    near = values[np.abs(values - peak) <= width]
    return float(near.mean()) if near.size else float(peak)


def _level(bg: np.ndarray, camera: str) -> np.ndarray:
    """
    The background above its pedestal, in data numbers.

    The spectrograph image holds two CCDs side by side, and every CCD is
    read through :data:`_taps` amplifiers, one per quadrant, whose
    pedestals differ by up to a few data numbers, so each quadrant is
    levelled on its own by the :func:`_pedestal` of its read pixels; a
    single median for the whole image would flag the whole of the higher
    chip as elevated. A quadrant with no read pixels is left alone.
    """
    chips = 2 if camera == "FUV" else 1
    rows, columns = bg.shape
    width = columns // chips
    result = bg.copy()
    for chip in range(chips):
        left = chip * width
        right = left + width if chip < chips - 1 else columns
        middle = left + width // 2
        for r in (slice(0, rows // 2), slice(rows // 2, rows)):
            for c in (slice(left, middle), slice(middle, right)):
                quadrant = result[r, c]
                values = quadrant[np.isfinite(quadrant)]
                if values.size:
                    quadrant -= _pedestal(values)
    return np.nan_to_num(result, nan=0)


def off_limb(header, shape: tuple[int, int], camera: str) -> np.ndarray:
    """
    Which pixels of a level-1 image look at least :data:`_limb_margin`
    above the photospheric limb, from the pointing in its header.

    The header gives the position on the Sun of a reference pixel and how
    it changes along the slit (for a spectrograph) or across the field (for
    the slit-jaw imager), and the apparent solar radius. A frame whose
    header lacks them is searched everywhere.

    Parameters
    ----------
    header
        The header of the level-1 image.
    shape
        The shape of the image, ``(rows, columns)``.
    camera
        ``FUV``, ``NUV``, or ``SJI``.
    """
    rows, columns = shape
    try:
        limit = float(header["RSUN_OBS"]) + _limb_margin
        if camera == "SJI":
            x = np.arange(columns) + 1 - float(header["CRPIX1"])
            y = np.arange(rows)[:, np.newaxis] + 1 - float(header["CRPIX2"])
            lon = float(header["CRVAL1"]) + float(header["CDELT1"]) * (
                float(header["PC1_1"]) * x + float(header["PC1_2"]) * y
            )
            lat = float(header["CRVAL2"]) + float(header["CDELT2"]) * (
                float(header["PC2_1"]) * x + float(header["PC2_2"]) * y
            )
            return np.hypot(lon, lat) > limit
        # a spectrograph: axis 2 of the header runs along the slit, which is
        # the row axis of the image, and axis 3 is the other solar coordinate
        s = np.arange(rows) + 1 - float(header["CRPIX2"])
        lat = (
            float(header["CRVAL2"])
            + float(header["CDELT2"]) * float(header["PC2_2"]) * s
        )
        lon = (
            float(header["CRVAL3"])
            + float(header["CDELT3"]) * float(header["PC3_2"]) * s
        )
        along = np.hypot(lon, lat) > limit
        return np.broadcast_to(along[:, np.newaxis], shape)
    except (KeyError, TypeError, ValueError):
        return np.ones(shape, dtype=bool)


def _mask(
    kind: None | str,
    bg: np.ndarray,
    noise: np.ndarray,
    camera: str = "FUV",
) -> np.ndarray:
    """The pixels free of solar signal, hot pixels, and bad rows or columns."""
    finite = np.isfinite(bg)
    if kind is None:
        return finite
    level = _level(bg, camera)
    elevated = level > 2.5
    if kind == "columns":
        result = finite & ~elevated
        result[:, elevated.sum(0) > 30] = False
        return result
    if kind == "rows":
        result = finite & ~elevated
        result[elevated.sum(1) > 30, :] = False
        result[:, elevated.sum(0) > 30] = False
        return result
    if kind == "median":
        hot = scipy.ndimage.binary_dilation((level > 5) & finite)
        result = finite & ~hot
        result[elevated.sum(1) > 30, :] = False
        result[:, elevated.sum(0) > 30] = False
        return result
    if kind == "lines":
        # scattered light in the emission lines raises every row of a limb
        # pointing a little, so cutting rows as ``median`` does can remove
        # the whole frame, and the disk half of such a pointing raises a
        # few pixels of every column, so counting them as ``columns`` does
        # removes every column; the lines themselves are columns raised
        # along most of their length, so cut those by their typical level
        hot = scipy.ndimage.binary_dilation((level > 5) & finite)
        result = finite & ~hot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # unread columns
            typical = np.nanmedian(np.where(finite, level, np.nan), axis=0)
        result[:, np.nan_to_num(typical, nan=0) > 2.5] = False
        return result
    if kind == "limb":
        smooth = scipy.ndimage.median_filter(
            np.nan_to_num(bg, nan=np.nanmedian(bg)), 15
        )
        pedestal = np.nanpercentile(bg[finite], 5)
        disk = np.nanpercentile(bg[finite], 95)
        off = (smooth < pedestal + 0.15 * (disk - pedestal)) & finite
        off = scipy.ndimage.binary_erosion(off, iterations=25)
        hot = scipy.ndimage.binary_dilation(((bg - smooth) > 5) & finite)
        return off & ~hot & (noise < 3)
    raise ValueError(kind)


@dataclasses.dataclass(eq=False)
class Component:
    """A connected group of labeled pixels in a frame, with its shape moments."""

    dataset: str
    fsn: int
    saa: bool
    azimuth: float
    """The angle of the long axis from the row axis toward the column axis, in degrees from 0 to 180."""
    length: float
    """Four times the rms extent along the long axis, in pixels."""
    width: float
    """Four times the rms extent across it, in pixels."""
    num_pixels: int
    charge: float
    """The total charge in electrons."""


def _moments(sub: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    yy, xx = np.nonzero(mask)
    weight = sub[mask]
    total = weight.sum()
    y0 = (weight * yy).sum() / total
    x0 = (weight * xx).sum() / total
    cyy = (weight * (yy - y0) ** 2).sum() / total
    cxx = (weight * (xx - x0) ** 2).sum() / total
    cxy = (weight * (yy - y0) * (xx - x0)).sum() / total
    azimuth = 0.5 * np.degrees(np.arctan2(2 * cxy, cyy - cxx)) % 180
    eigenvalues = np.linalg.eigvalsh([[cyy, cxy], [cxy, cxx]])
    length = 4 * np.sqrt(max(eigenvalues[1], 0))
    width = 4 * np.sqrt(max(eigenvalues[0], 0))
    return float(azimuth), float(length), float(width)


def census(
    frame: dict[str, str],
    residual: np.ndarray,
    valid: np.ndarray,
    noise: float,
    gain: float,
) -> list[Component]:
    """
    Every elongated connected group of pixels in a frame, before any of the
    cuts of :func:`find`, for the census of track directions.

    Parameters
    ----------
    frame
        The row of ``data/iris_frames.csv`` describing the frame.
    residual
        The frame minus its background, in data numbers.
    valid
        The pixels to consider.
    noise
        The read noise in data numbers.
    gain
        The camera gain in electrons per data number.
    """
    r = np.where(valid, residual, 0)
    labels, _ = scipy.ndimage.label(r > threshold * noise, structure=_touching)
    result = []
    for k, sl in enumerate(scipy.ndimage.find_objects(labels), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < length_minimum:
            continue
        sub = r[sl]
        mask = (labels[sl] == k) & (sub > 0)
        if sub[mask].max() * gain > _saturation:
            continue
        azimuth, length, width = _moments(sub, mask)
        result.append(
            Component(
                dataset=frame["dataset"],
                fsn=int(frame["fsn"]),
                saa=frame["saa"] == "1",
                azimuth=azimuth,
                length=length,
                width=width,
                num_pixels=int(mask.sum()),
                charge=float(sub[mask].sum() * gain),
            )
        )
    return result


def find(
    frame: dict[str, str],
    residual: np.ndarray,
    valid: np.ndarray,
    noise: float,
    gain: float,
) -> list[Track]:
    """
    The glancing tracks in a frame.

    Pixels above :data:`threshold` times the read noise are labeled, grouping
    pixels which touch even at a corner, and each connected group at least
    :data:`length_minimum` rows (or columns) long
    and no more than a few pixels wide is fit with a straight line through
    the charge-weighted centroid of each row.
    Groups with a slope below :data:`slope_maximum`, a clear seven-pixel
    window around the line, and no other group inside that window are cut
    out as tracks.

    Parameters
    ----------
    frame
        The row of ``data/iris_frames.csv`` describing the frame.
    residual
        The frame minus its background, in data numbers.
    valid
        The pixels which may be part of a cutout.
    noise
        The read noise in data numbers.
    gain
        The camera gain in electrons per data number.
    """
    h_ = half_width
    labels, _ = scipy.ndimage.label(residual > threshold * noise, structure=_touching)
    result = []
    for k, sl in enumerate(scipy.ndimage.find_objects(labels), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < length_minimum or min(h, w) > _width_maximum:
            continue
        vertical = h >= w
        img = residual if vertical else residual.T
        ok_pixels = valid if vertical else valid.T
        lab = labels if vertical else labels.T
        s0, s1 = sl if vertical else (sl[1], sl[0])
        if img[s0, s1].max() * gain > _saturation:
            continue
        i = np.arange(s0.start, s0.stop)
        profile = img[s0, s1]
        signal = profile.sum(1)
        centroid = (profile * np.arange(s1.start, s1.stop)).sum(1) / np.where(
            signal > 0, signal, 1
        )
        ok = signal * gain > charge_minimum
        if ok.sum() < length_minimum or ok.mean() < 0.8:
            continue
        slope, intercept = np.polyfit(i[ok], centroid[ok], 1)
        if abs(slope) > slope_maximum:
            continue
        line = slope * i + intercept
        center = np.round(line).astype(int)
        lo, hi = center - h_, center + h_ + 1
        if lo.min() < 0 or hi.max() > img.shape[1]:
            continue
        if not all(ok_pixels[ii, a:b].all() for ii, a, b in zip(i, lo, hi)):
            continue
        others = np.stack([lab[ii, a:b] for ii, a, b in zip(i, lo, hi)])
        if np.any((others != 0) & (others != k)):
            continue
        charge = np.stack([img[ii, a:b] for ii, a, b in zip(i, lo, hi)]) * gain
        result.append(
            Track(
                name="",
                dataset=frame["dataset"],
                chip="",
                fsn=int(frame["fsn"]),
                slope=float(slope),
                noise=float(noise * gain),
                gain=float(gain),
                vertical=vertical,
                row=int(s0.start),
                column=int(s1.start),
                charge=_array(charge.astype(np.float32), (axis_slice, axis_pixel)),
                position=_array((line - center).astype(np.float32), (axis_slice,)),
            )
        )
    return result


def _array(ndarray, axes):
    import named_arrays as na

    return na.ScalarArray(ndarray.astype(float), axes=axes)


def _chip(track: Track, image: str, width: int) -> str:
    """Which CCD a track lies on: the FUV spectrograph image holds FUV1 and FUV2 side by side."""
    if image != "FUV":
        return _camera(image)
    column = track.column if track.vertical else track.row
    return "FUV1" if column < width // 2 else "FUV2"


def _extract_block(
    block: Block,
    directory: pathlib.Path,
    keep: bool,
    verbose: bool,
) -> tuple[list[Track], list[Component]]:
    """Fetch one block, estimate its background, and search its frames."""
    camera = _camera(block.frames[0]["image"])
    config = _configuration(block.dataset, camera)
    tracks = []
    components = []
    if config["quiet"] == "quiet" and len(block.quiet) < (
        60 if block.dataset == "sji" else 100
    ):
        if verbose:
            print(
                f"{block.dataset} block {block.name}: only {len(block.quiet)} quiet frames, skipped"
            )
        return tracks, components
    failed = download(block.frames, directory)
    if failed and verbose:
        print(
            f"{block.dataset} block {block.name}: {len(failed)} frames could not be fetched"
        )
    quiet = [f for f in block.quiet if f not in failed]
    if len(quiet) > _quiet_maximum:
        step = len(quiet) / _quiet_maximum
        quiet = [quiet[int(i * step)] for i in range(_quiet_maximum)]
    stack = np.stack([read(f, directory)[0] for f in quiet])
    bg, noise_map = background(stack)
    del stack
    kind = config["mask"]
    if kind == "camera":
        kind = _mask_camera[camera]
    mask = _mask(kind, bg, noise_map, camera)
    noise_maximum = config["noise_maximum"]
    if noise_maximum == "camera":
        noise_maximum = _noise_maximum[camera]
    if noise_maximum is not None:
        mask &= noise_map < noise_maximum
    if not mask.any():
        if verbose:
            print(
                f"{block.dataset} block {block.name}: no pixel passes the mask "
                f"(noise {np.nanmedian(noise_map):.2f} DN), skipped"
            )
        return tracks, components
    noise = float(np.nanmedian(noise_map[mask]))
    gain = _gain[camera]
    # the quiet frames of the roll -90 campaign enter the census as its cosmic-ray sample
    surveyed = list(block.search)
    if block.dataset == "2018may":
        surveyed += [f for f in block.quiet if f not in block.search]
    for f in surveyed:
        if f in failed:
            continue
        data, header = read(f, directory)
        residual = np.nan_to_num(data - bg)
        valid = np.isfinite(data) & (data > 0) & mask
        valid &= off_limb(header, data.shape, camera)
        if f in block.search:
            for track in find(f, residual, valid, noise, gain):
                track.chip = _chip(track, f["image"], data.shape[1])
                tracks.append(track)
        components += census(f, residual, valid, noise, gain)
    if verbose:
        print(
            f"{block.dataset} block {block.name}: {len(quiet)} quiet frames, "
            f"{len(block.search)} searched, noise {noise:.2f} DN, {len(tracks)} tracks"
        )
    if not keep:
        for f in block.frames:
            path(f, directory).unlink(missing_ok=True)
    return tracks, components


def extract(
    dataset: str,
    directory: None | pathlib.Path = None,
    keep: bool = True,
    verbose: bool = True,
) -> tuple[list[Track], list[Component]]:
    """
    Fetch the frames of a campaign block by block and extract its tracks and
    the census of its elongated components.

    Parameters
    ----------
    dataset
        The campaign, a key of :data:`_config`.
    directory
        The cache directory, :data:`ccd_diffusion.tracks.directory_default`
        if :obj:`None`.
    keep
        Whether to leave the frames in the cache once a block is done.
        The five campaigns are several gigabytes together.
    verbose
        Whether to report each block as it is processed.
    """
    if directory is None:
        directory = directory_default
    tracks = []
    components = []
    for block in blocks(dataset, directory):
        t, c = _extract_block(block, directory, keep, verbose)
        tracks += t
        components += c
    for index, track in enumerate(tracks, start=1):
        track.name = f"{dataset}-{index}"
    return tracks, components


def save_tracks(
    tracks: list[Track],
    directory: None | pathlib.Path = None,
    frames: None | list[dict[str, str]] = None,
) -> None:
    """
    Write the tracks to ``data/iris_tracks.npz`` and ``data/iris_tracks.csv``,
    and the number of tracks per frame to ``data/iris_frames.csv``, where
    :func:`load` and :func:`frames` will find them.

    Parameters
    ----------
    tracks
        The tracks of every campaign, in the order they were found.
    directory
        The data directory of the package if :obj:`None`.
    frames
        The frames whose track counts to write, every frame of the package
        if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    directory = pathlib.Path(directory)
    if frames is None:
        frames = list(globals()["frames"]())
    width = 2 * half_width + 1
    charge = np.concatenate(
        [t.charge.ndarray.astype(np.float32) for t in tracks]
        or [np.zeros((0, width), np.float32)]
    )
    position = np.concatenate(
        [t.position.ndarray.astype(np.float32) for t in tracks]
        or [np.zeros(0, np.float32)]
    )
    np.savez_compressed(directory / "iris_tracks.npz", charge=charge, position=position)
    fields = [
        "name",
        "dataset",
        "chip",
        "fsn",
        "length",
        "start",
        "slope",
        "noise",
        "gain",
        "vertical",
        "row",
        "column",
    ]
    start = 0
    with open(directory / "iris_tracks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in tracks:
            writer.writerow(
                dict(
                    name=t.name,
                    dataset=t.dataset,
                    chip=t.chip,
                    fsn=t.fsn,
                    length=t.length,
                    start=start,
                    slope=t.slope,
                    noise=t.noise,
                    gain=t.gain,
                    vertical=t.vertical,
                    row=t.row,
                    column=t.column,
                )
            )
            start += t.length
    counts = {}
    for t in tracks:
        counts[(t.dataset, str(t.fsn))] = counts.get((t.dataset, str(t.fsn)), 0) + 1
    rows = [dict(f) for f in frames]
    for r in rows:
        r["tracks"] = counts.get((r["dataset"], r["fsn"]), 0)
    fields = ["dataset", "fsn", "time", "image", "saa", "tracks"]
    with open(directory / "iris_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_census(
    components: list[Component], directory: None | pathlib.Path = None
) -> None:
    """
    Write the census of elongated components to ``data/iris_azimuth.csv``.

    Parameters
    ----------
    components
        The components of every campaign.
    directory
        The data directory of the package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    fields = [f.name for f in dataclasses.fields(Component)]
    with open(pathlib.Path(directory) / "iris_azimuth.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in components:
            row = dataclasses.asdict(c)
            for k in ("azimuth", "length", "width", "charge"):
                row[k] = f"{row[k]:.3f}"
            writer.writerow(row)


def load_census(directory: None | pathlib.Path = None) -> list[Component]:
    """
    Read a census written by :func:`save_census`.

    Parameters
    ----------
    directory
        The directory holding ``iris_azimuth.csv``, the data directory of the
        package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    with open(pathlib.Path(directory) / "iris_azimuth.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Component(
            dataset=r["dataset"],
            fsn=int(r["fsn"]),
            saa=r["saa"] == "True",
            azimuth=float(r["azimuth"]),
            length=float(r["length"]),
            width=float(r["width"]),
            num_pixels=int(r["num_pixels"]),
            charge=float(r["charge"]),
        )
        for r in rows
    ]
