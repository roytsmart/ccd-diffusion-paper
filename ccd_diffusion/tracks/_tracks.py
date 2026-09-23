import csv
import dataclasses
import functools
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "charge_minimum",
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

thickness = 14 * u.um
"""The thickness of the silicon of the IRIS CCDs."""

charge_minimum = 240.0
"""
The least charge in a slice, in electrons, for it to enter the statistics
of the pixel fractions.

This mirrors the finder's floor for a slice to constrain the centerline,
:data:`ccd_diffusion.tracks._extract.charge_minimum`, which is kept
separate so that the extraction code, whose text the campaign
fingerprints hash, need not change.
"""


_directory_data = pathlib.Path(__file__).parent / "data"

datasets = {
    "2014b": dict(
        date="2014-03-12",
        particles="SAA protons",
        roll=0,
        exposure=15,
        image="FUV, NUV",
    ),
    "2014": dict(
        date="2014-04-07", particles="cosmic rays", roll=0, exposure=4, image="FUV, NUV"
    ),
    "2018": dict(
        date="2018-03-16",
        particles="cosmic rays",
        roll=0,
        exposure=15,
        image="FUV, NUV",
    ),
    "2018may": dict(
        date="2018-05-04",
        particles="SAA protons",
        roll=-90,
        exposure=15,
        image="FUV, NUV",
    ),
    "sji": dict(
        date="2018-05-04/06",
        particles="SAA protons",
        roll=-90,
        exposure=15,
        image="SJI",
    ),
    "2014-08-20": dict(
        date="2014-08-20",
        particles="SAA protons",
        roll=-75,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-08-21": dict(
        date="2014-08-21",
        particles="SAA protons",
        roll=-75,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-08-23": dict(
        date="2014-08-23",
        particles="SAA protons",
        roll=90,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-08-24": dict(
        date="2014-08-24",
        particles="SAA protons",
        roll=90,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-08-25": dict(
        date="2014-08-25",
        particles="SAA protons",
        roll=90,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-11-02": dict(
        date="2014-11-02",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2014-11-27": dict(
        date="2014-11-27",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2015-03-26": dict(
        date="2015-03-26",
        particles="SAA protons",
        roll=-90,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2015-09-20": dict(
        date="2015-09-20",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2016-01-02": dict(
        date="2016-01-02",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2016-01-06": dict(
        date="2016-01-06",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV",
    ),
    "2016-05-16": dict(
        date="2016-05-16",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2016-10-28": dict(
        date="2016-10-28", particles="SAA protons", roll=0, exposure=8, image="FUV, NUV"
    ),
    "2016-12-13": dict(
        date="2016-12-13",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2016-12-17": dict(
        date="2016-12-17",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
    "2016-12-25": dict(
        date="2016-12-25",
        particles="SAA protons",
        roll=0,
        exposure=8,
        image="FUV, NUV, SJI",
    ),
}
"""
The observing campaigns the tracks were found in, in chronological order,
with the date, the dominant particle population (inferred from the energy
loss and from whether the frames were taken inside the South Atlantic
Anomaly), the roll angle of the spacecraft in degrees, the exposure time
in seconds, and the camera.
"""


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
    def extent(self) -> tuple[float, float, float, float]:
        """
        The bounding box of the cutout in the parent image, as ``(x, y,
        width, height)`` in pixels, where ``x`` runs along the columns.
        """
        h = half_width
        if self.vertical:
            return self.column - h - 0.5, self.row - 0.5, 2 * h + 1, self.length
        return self.row - 0.5, self.column - h - 0.5, self.length, 2 * h + 1

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

    @property
    def usable(self) -> na.AbstractScalarArray:
        """
        Whether each slice holds at least :data:`charge_minimum` of charge.

        The finder locates the centerline from such slices alone, and the
        statistics of the pixel fractions are taken from them alone too,
        since a slice with almost no charge has fractions of almost
        anything. The fit uses every slice, weighting each by its charge.
        """
        return self.signal >= charge_minimum


@functools.cache
def load(directory: None | pathlib.Path = None) -> tuple[Track, ...]:
    """
    Load the particle tracks extracted from the IRIS level-1 images.

    The tracks are stored in ``data/iris_tracks.csv`` (one row of metadata
    per track) and ``data/iris_tracks.npz`` (the cutouts of every track
    concatenated along :data:`axis_slice`).

    Parameters
    ----------
    directory
        The directory holding the two files, the data directory of the
        package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    directory = pathlib.Path(directory)
    arrays = np.load(directory / "iris_tracks.npz")
    charge = arrays["charge"]
    position = arrays["position"]

    with open(directory / "iris_tracks.csv", newline="") as f:
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
def frames(directory: None | pathlib.Path = None) -> tuple[dict[str, str], ...]:
    """
    Load the list of IRIS level-1 frames searched for tracks.

    Each element is a row of ``data/iris_frames.csv`` with the columns
    ``dataset``, ``fsn``, ``time``, ``image``, ``saa`` (whether the frame was
    taken inside the South Atlantic Anomaly) and ``tracks`` (the number of
    tracks found in the frame).

    Parameters
    ----------
    directory
        The directory holding the file, the data directory of the package
        if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    with open(pathlib.Path(directory) / "iris_frames.csv", newline="") as f:
        return tuple(csv.DictReader(f))
