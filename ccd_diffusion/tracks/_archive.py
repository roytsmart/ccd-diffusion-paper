"""
Fetch the IRIS level-1 images the tracks were extracted from.

The frames are listed in ``data/iris_frames.csv`` and served by the IRIS
level-1 archive at LMSAL, one compressed FITS file per exposure, named after
the time of the exposure.
"""

import os
import pathlib
import concurrent.futures
import urllib.request
import numpy as np
from astropy.io import fits

__all__ = [
    "directory_default",
    "url",
    "path",
    "download",
    "read",
]

directory_default = pathlib.Path(
    os.environ.get(
        "CCD_DIFFUSION_CACHE",
        pathlib.Path.home() / ".cache" / "ccd_diffusion" / "iris",
    )
)
"""
Where the level-1 images are kept, ``~/.cache/ccd_diffusion/iris`` unless the
``CCD_DIFFUSION_CACHE`` environment variable says otherwise.
"""

_size_minimum = 100_000
"""A level-1 file smaller than this many bytes is a failed download."""


def url(time: str, image: str) -> str:
    """
    The address of a level-1 image in the LMSAL archive.

    Parameters
    ----------
    time
        The time of the exposure, ``T_OBS``, as ``YYYY-MM-DDThh:mm:ss.ccZ``.
    image
        The camera, ``FUV``, ``NUV``, or one of the ``SJI_*`` channels.
    """
    camera = image.lower() if image in ("FUV", "NUV") else "sji"
    return (
        "https://www.lmsal.com/solarsoft/irisa/data/level1/"
        f"{time[:4]}/{time[5:7]}/{time[8:10]}/H{time[11:13]}00/"
        f"iris{time[:4]}{time[5:7]}{time[8:10]}_"
        f"{time[11:13]}{time[14:16]}{time[17:19]}{time[20:22]}_{camera}.fits"
    )


def path(frame: dict[str, str], directory: None | pathlib.Path = None) -> pathlib.Path:
    """
    Where a frame is kept locally.

    Parameters
    ----------
    frame
        A row of ``data/iris_frames.csv``.
    directory
        The cache directory, :data:`directory_default` if :obj:`None`.
    """
    if directory is None:
        directory = directory_default
    return pathlib.Path(directory) / frame["dataset"] / f"{int(frame['fsn'])}.fits"


def _fetch(frame: dict[str, str], directory: pathlib.Path) -> bool:
    destination = path(frame, directory)
    if destination.exists() and destination.stat().st_size > _size_minimum:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url(frame["time"], frame["image"]), destination)
    except Exception:
        destination.unlink(missing_ok=True)
        return False
    if destination.stat().st_size <= _size_minimum:
        destination.unlink(missing_ok=True)
        return False
    return True


def download(
    frames: "list[dict[str, str]] | tuple[dict[str, str], ...]",
    directory: None | pathlib.Path = None,
    workers: int = 6,
) -> list[dict[str, str]]:
    """
    Fetch the given frames from the archive, skipping any already present,
    and return the frames which could not be fetched.

    Parameters
    ----------
    frames
        Rows of ``data/iris_frames.csv``.
    directory
        The cache directory, :data:`directory_default` if :obj:`None`.
    workers
        The number of simultaneous downloads.
    """
    if directory is None:
        directory = directory_default
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(lambda f: _fetch(f, directory), frames))
    return [f for f, good in zip(frames, ok) if not good]


def read(
    frame: dict[str, str],
    directory: None | pathlib.Path = None,
) -> tuple[np.ndarray, fits.Header]:
    """
    Read a level-1 image and its header.

    Parameters
    ----------
    frame
        A row of ``data/iris_frames.csv``.
    directory
        The cache directory, :data:`directory_default` if :obj:`None`.
    """
    with fits.open(path(frame, directory)) as hdul:
        return hdul[1].data.astype(np.float32), hdul[1].header
