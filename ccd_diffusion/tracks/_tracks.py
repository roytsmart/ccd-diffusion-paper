import csv
import dataclasses
import functools
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "axis_slice",
    "axis_pixel",
    "half_width",
    "width_pixel",
    "Track",
    "load",
    "frames",
]

axis_slice = "slice"
"""The logical axis along the track, one element per CCD row (or column)."""

axis_pixel = "pixel"
"""The logical axis across the track."""

half_width = 3
"""The number of pixels saved on each side of the track centerline."""

width_pixel = 13 * u.um
"""The pixel pitch of the IRIS CCDs."""

_directory_data = pathlib.Path(__file__).parent / "data"


@dataclasses.dataclass(eq=False)
class Track:
    """
    A glancing particle track cut out of an IRIS level-1 image.

    The track has been rotated so that it runs along :attr:`axis_slice`,
    and each slice contains the :math:`2 h + 1` pixels centered on the
    integer part of the least-squares centerline, where :math:`h` is
    :data:`half_width`.
    """

    name: str
    """A unique identifier for this track, ``<dataset>-<index>``."""

    dataset: str
    """The observing campaign this track was found in."""

    chip: str
    """The CCD this track was recorded on, ``FUV1``, ``FUV2`` or ``SJI``."""

    fsn: int
    """The IRIS frame serial number of the parent image."""

    slope: float
    """The tilt of the centerline in pixels per slice."""

    noise: float
    """The read noise of the parent image in electrons."""

    gain: float
    """The camera gain of the parent image in electrons per DN."""

    vertical: bool
    """Whether the track runs along the columns of the parent image."""

    row: int
    """The row of the parent image where the track starts."""

    column: int
    """The column of the parent image where the track starts."""

    charge: na.ScalarArray
    """The charge collected in each pixel of the cutout in electrons."""

    position: na.ScalarArray
    """The fractional centerline offset of each slice from the central pixel."""

    @property
    def length(self) -> int:
        """The number of slices in this track."""
        return self.charge.shape[axis_slice]

    @property
    def signal(self) -> na.AbstractScalarArray:
        """The total charge collected in each slice."""
        return self.charge.sum(axis_pixel)

    @property
    def fraction(self) -> na.AbstractScalarArray:
        """The fraction of each slice's charge collected in each pixel."""
        return self.charge / self.signal

    @property
    def error(self) -> na.AbstractScalarArray:
        """The read-noise uncertainty of :attr:`fraction`."""
        return self.noise / self.signal

    @property
    def depth(self) -> na.AbstractScalarArray:
        """
        The fractional depth of each slice below the back surface,
        for a track that enters the back surface at its first slice
        and exits through the front surface at its last slice.
        """
        return (na.arange(0, self.length, axis=axis_slice) + 0.5) / self.length

    @property
    def index(self) -> na.AbstractScalarArray:
        """The index of each slice measured from the middle of the track."""
        return na.arange(0, self.length, axis=axis_slice) - self.length / 2


@functools.cache
def load() -> tuple[Track, ...]:
    """
    Load the particle tracks extracted from the IRIS level-1 images.

    The tracks are stored in ``data/iris_tracks.csv`` (one row of metadata
    per track) and ``data/iris_tracks.npz`` (the cutouts of every track
    concatenated along :data:`axis_slice`).
    """
    arrays = np.load(_directory_data / "iris_tracks.npz")
    charge = arrays["charge"]
    position = arrays["position"]

    with open(_directory_data / "iris_tracks.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    result = []
    for r in rows:
        start = int(r["start"])
        stop = start + int(r["length"])
        result.append(
            Track(
                name=r["name"],
                dataset=r["dataset"],
                chip=r["chip"],
                fsn=int(r["fsn"]),
                slope=float(r["slope"]),
                noise=float(r["noise"]),
                gain=float(r["gain"]),
                vertical=r["vertical"] == "True",
                row=int(r["row"]),
                column=int(r["column"]),
                charge=na.ScalarArray(
                    ndarray=charge[start:stop].astype(float),
                    axes=(axis_slice, axis_pixel),
                ),
                position=na.ScalarArray(
                    ndarray=position[start:stop].astype(float),
                    axes=(axis_slice,),
                ),
            )
        )
    return tuple(result)


@functools.cache
def frames() -> tuple[dict[str, str], ...]:
    """
    Load the list of IRIS level-1 frames searched for tracks.

    Each element is a row of ``data/iris_frames.csv`` with the columns
    ``dataset``, ``fsn``, ``time``, ``image``, ``saa`` (whether the frame was
    taken inside the South Atlantic Anomaly) and ``tracks`` (the number of
    tracks found in the frame).
    """
    with open(_directory_data / "iris_frames.csv", newline="") as f:
        return tuple(csv.DictReader(f))
