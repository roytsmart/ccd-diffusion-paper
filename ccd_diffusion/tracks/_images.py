import csv
import dataclasses
import functools
import numpy as np
import named_arrays as na
from ._tracks import Track, load, _directory_data

__all__ = [
    "axis_row",
    "axis_column",
    "Image",
    "images",
]

axis_row = "row"
"""The logical axis along the columns of a level-1 image."""

axis_column = "column"
"""The logical axis along the rows of a level-1 image."""


@dataclasses.dataclass(eq=False)
class Image:
    """A level-1 IRIS image, kept as an example of the frames searched for tracks."""

    dataset: str
    """The observing campaign this image belongs to."""

    fsn: int
    """The IRIS frame serial number."""

    time: str
    """The time of the observation."""

    image: str
    """The camera, ``FUV`` or ``SJI_2796``."""

    saa: bool
    """Whether the image was taken inside the South Atlantic Anomaly."""

    roll: float
    """The roll angle of the spacecraft in degrees."""

    exptime: float
    """The exposure time in seconds."""

    data: na.ScalarArray
    """The image in data numbers, with axes ``(row, column)``."""

    @property
    def tracks(self) -> list[Track]:
        """The tracks the finder extracted from this image."""
        return [t for t in load() if t.dataset == self.dataset and t.fsn == self.fsn]


@functools.cache
def images() -> tuple[Image, ...]:
    """
    Load the example level-1 images distributed with this article, one from
    the spectrograph and one from the slit-jaw imager, taken seconds apart
    during the same pass through the South Atlantic Anomaly.
    """
    arrays = np.load(_directory_data / "iris_images.npz")
    with open(_directory_data / "iris_images.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(
        Image(
            dataset=r["dataset"],
            fsn=int(r["fsn"]),
            time=r["time"],
            image=r["image"],
            saa=r["saa"] == "1",
            roll=float(r["roll"]),
            exptime=float(r["exptime"]),
            data=na.ScalarArray(
                arrays[r["image"]].astype(float), axes=(axis_row, axis_column)
            ),
        )
        for r in rows
    )
