"""
The export behind the track browser of the documentation: every track with
its cutout and its fit in one JSON file, and one level-1 frame from each
camera of every campaign rendered with its tracks outlined.
"""

import json
import pathlib
import numpy as np
import PIL.Image
import PIL.ImageDraw
import astropy.units as u
from ._tracks import (
    half_width,
    width_pixel,
    charge_minimum,
    Track,
    load,
    frames,
)
from ._archive import download, read
from ._extract import _camera, _level
from ._fit import Fit, fits, critical_depth_maximum
from ._depleted import depleted

__all__ = [
    "core_range",
    "export_tracks",
    "frame_choices",
    "render_frame",
    "export_frames",
]

core_range = (0.25, 0.6)
"""The range of :math:`t_c` within which a flat track counts as a core track."""

_stretch = 10
"""The scale of the arcsinh stretch of a rendered frame, in data numbers above the background."""

_vmax = 400
"""The data number above the background rendered as black."""

_pad = 8
"""The margin around a track's cutout in the box drawn on a rendered frame, in pixels."""

_outline = (214, 39, 40)
"""The color of the boxes drawn on a rendered frame."""


def _record(fit: Fit, time: str) -> dict:
    """One track, its cutout, and its fit, as plain Python values."""
    track = fit.track
    tc = fit.critical_depth
    return dict(
        name=track.name,
        dataset=track.dataset,
        chip=track.chip,
        fsn=track.fsn,
        time=time,
        vertical=track.vertical,
        row=track.row,
        column=track.column,
        length=track.length,
        slope=round(track.slope, 4),
        noise=round(track.noise, 2),
        gain=round(track.gain, 3),
        orientation=fit.orientation,
        tc=round(tc, 3),
        tc_min=round(fit.critical_depth_min, 3),
        tc_max=round(fit.critical_depth_max, 3),
        sm=round(fit.width_max.to_value(u.um), 3),
        sm_min=round(fit.width_max_min.to_value(u.um), 3),
        sm_max=round(fit.width_max_max.to_value(u.um), 3),
        sd=round(fit.width_depleted.to_value(u.um), 3),
        sd_preferred=round(fit.width_depleted_preferred.to_value(u.um), 3),
        offset=round(fit.offset, 3),
        tilt=round(fit.tilt, 4),
        improvement=round(fit.gain, 2),
        bragg=round(fit.bragg, 3),
        tight=fit.tight,
        crossing=fit.crossing,
        flat=fit.flat,
        core=fit.flat and core_range[0] < tc < core_range[1],
        charge=np.rint(track.charge.ndarray).astype(int).tolist(),
        position=np.round(track.position.ndarray, 3).tolist(),
    )


def export_tracks(path: pathlib.Path) -> int:
    """
    Write every track, its cutout, and its fit to one JSON file for the
    track browser of the documentation, and return how many were written.

    Parameters
    ----------
    path
        The file to write.
    """
    time = {(f["dataset"], int(f["fsn"])): f["time"] for f in frames()}
    records = [_record(f, time[(f.track.dataset, f.track.fsn)]) for f in fits()]
    chips = sorted({r["chip"] for r in records})
    # one list per field rather than one object per track, since the keys
    # would otherwise be a third of the file
    columns = {k: [r[k] for r in records] for k in (records[0] if records else {})}
    result = dict(
        half_width=half_width,
        width_pixel=width_pixel.to_value(u.um),
        charge_minimum=charge_minimum,
        critical_depth_maximum=critical_depth_maximum,
        core_range=list(core_range),
        width_depleted={c: depleted(c).best.to_value(u.um) for c in chips},
        campaigns=list(dict.fromkeys(r["dataset"] for r in records)),
        num=len(records),
        tracks=columns,
    )
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    return len(records)


frames_per_camera = 3
"""How many frames of each camera of each campaign the documentation shows."""


def frame_choices(num: int = frames_per_camera) -> list[dict[str, str]]:
    """
    The frames holding the most tracks from each camera (the slit-jaw
    imager counting as one, whichever of its channels) in every campaign,
    as rows of the frame list, skipping frames that yielded none.

    Parameters
    ----------
    num
        How many frames to keep per camera and campaign.
    """
    by_camera: dict[tuple[str, str], list[dict[str, str]]] = {}
    for f in frames():
        if int(f["tracks"]) > 0:
            by_camera.setdefault((f["dataset"], _camera(f["image"])), []).append(f)
    result = []
    for mine in by_camera.values():
        mine.sort(key=lambda f: (-int(f["tracks"]), int(f["fsn"])))
        result += mine[:num]
    return result


def render_frame(
    data: np.ndarray,
    tracks: "list[Track] | tuple[Track, ...]",
    path: pathlib.Path,
    camera: str = "FUV",
) -> dict:
    """
    Render a level-1 image as an 8-bit PNG with the stretch of the article's
    example images above the pedestal of each quadrant, as the finder's
    mask sees it, cropped to the part of the frame that was read out, with
    unread pixels white, every track outlined, and the first row at the
    bottom, and return the crop as ``rows`` and ``columns`` of the original.

    Parameters
    ----------
    data
        The level-1 image in data numbers, unread pixels zero.
    tracks
        The tracks the finder extracted from this image.
    path
        The PNG to write.
    camera
        ``FUV``, ``NUV``, or ``SJI``, which sets the quadrants.
    """
    valid = data > 0
    rows = np.flatnonzero(valid.any(axis=1))
    columns = np.flatnonzero(valid.any(axis=0))
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(columns.min()), int(columns.max()) + 1
    level = _level(np.where(valid, data, np.nan).astype(np.float32), camera)
    crop = level[r0:r1, c0:c1]
    read_out = valid[r0:r1, c0:c1]
    stretched = np.arcsinh(crop / _stretch) / np.arcsinh(_vmax / _stretch)
    stretched = np.nan_to_num(np.clip(stretched, 0, 1))  # a frame may hold NaN
    gray = 255 - np.rint(255 * stretched).astype(np.uint8)
    gray[~read_out] = 255
    image = PIL.Image.fromarray(gray).convert("RGB")
    draw = PIL.ImageDraw.Draw(image)
    for track in tracks:
        x, y, w, h = track.extent
        draw.rectangle(
            [x - _pad - c0, y - _pad - r0, x + w + _pad - c0, y + h + _pad - r0],
            outline=_outline,
        )
    image = image.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return dict(rows=[r0, r1], columns=[c0, c1])


def export_frames(
    directory: pathlib.Path,
    cache: None | pathlib.Path = None,
) -> list[dict]:
    """
    Fetch the frames chosen by :func:`frame_choices`, render each with
    :func:`render_frame` into the given directory, and describe them all in
    ``frames.json`` beside the images, for the documentation.

    Parameters
    ----------
    directory
        Where to write the images and their index.
    cache
        Where the level-1 images are kept,
        :data:`ccd_diffusion.tracks.directory_default` if :obj:`None`.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = frame_choices()
    failed = download(chosen, cache)
    by_frame: dict[tuple[str, int], list[Track]] = {}
    for track in load():
        by_frame.setdefault((track.dataset, track.fsn), []).append(track)
    records = []
    for f in chosen:
        if f in failed:
            continue
        data, header = read(f, cache)
        tracks = by_frame.get((f["dataset"], int(f["fsn"])), [])
        name = f"{f['dataset']}-{f['fsn']}.png"
        crop = render_frame(data, tracks, directory / name, _camera(f["image"]))
        records.append(
            dict(
                dataset=f["dataset"],
                fsn=int(f["fsn"]),
                time=f["time"],
                image=f["image"],
                saa=f["saa"] == "1",
                exposure=float(header.get("EXPTIME", 0)),
                file=name,
                tracks=[t.name for t in tracks],
                **crop,
            )
        )
    with open(directory / "frames.json", "w", encoding="utf-8") as f:
        json.dump(records, f, separators=(",", ":"))
    return records
