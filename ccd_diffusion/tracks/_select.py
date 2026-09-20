"""
Choose the IRIS level-1 frames to search for tracks.

The frames of each campaign are drawn from the level-1 catalog at the Joint
Science Operations Center, which answers a query over a time range with one
record per exposure carrying the time, the frame serial number, the camera,
and whether the exposure was taken inside the South Atlantic Anomaly.

A campaign keeps every frame taken inside the anomaly, since those are the
ones searched for tracks, and every :attr:`Campaign.stride`-th frame of the
sequence besides, which is what the background of a block is estimated
from. Expanding the dataset means adding campaigns to :data:`campaigns` and
running ``python -m ccd_diffusion.tracks select``.
"""

import dataclasses
import urllib.parse
import urllib.request
import json
import csv
import pathlib
from ._tracks import _directory_data

__all__ = [
    "url_catalog",
    "Campaign",
    "campaigns",
    "records",
    "select",
    "save_frames",
]

url_catalog = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"
"""The address of the catalog service at the Joint Science Operations Center."""

_series = "iris.lev1"
"""The data series the frames are drawn from."""

_keys = ("T_OBS", "FSN", "IMG_PATH", "SAA", "ISQOLTID")
"""The keywords fetched for every frame."""


@dataclasses.dataclass(eq=False)
class Campaign:
    """One observing campaign searched for tracks."""

    windows: tuple[tuple[str, str], ...]
    """
    The time ranges the campaign covers, each a pair of
    ``YYYY.MM.DD_hh:mm:ssZ`` strings, one per continuous stretch of the
    observation.
    """

    images: tuple[str, ...]
    """The cameras to keep, ``FUV`` or the ``SJI_*`` channels."""

    stride: int = 1
    """
    Keep every ``stride``-th frame of the sequence outside the anomaly.

    The quiet frames are only used to estimate the background of a block,
    so a long observation needs no more than a few hundred of them.
    """

    obsid: None | str = None
    """
    Keep only the frames of this observing program, the catalog's
    ``ISQOLTID``, if given.

    A program that runs all day has gaps in which other programs run,
    pointed anywhere, and those must not enter a campaign whose frames are
    meant to be dark.
    """


campaigns = {
    "2014b": Campaign(
        windows=(("2014.03.12_20:03:00Z", "2014.03.12_22:07:10Z"),),
        images=("FUV",),
    ),
    "2014": Campaign(
        windows=(("2014.04.07_11:18:30Z", "2014.04.07_12:49:20Z"),),
        images=("FUV",),
    ),
    "2018": Campaign(
        windows=(
            ("2018.03.16_11:20:55Z", "2018.03.16_12:11:00Z"),
            ("2018.03.16_12:49:38Z", "2018.03.16_13:25:30Z"),
        ),
        images=("FUV",),
    ),
    "2018may": Campaign(
        windows=(("2018.05.04_07:12:20Z", "2018.05.04_11:58:45Z"),),
        images=("FUV",),
        stride=4,
    ),
    "sji": Campaign(
        windows=(
            ("2018.05.04_07:11:30Z", "2018.05.04_11:58:00Z"),
            ("2018.05.05_06:58:30Z", "2018.05.05_12:03:10Z"),
            ("2018.05.06_07:07:00Z", "2018.05.06_12:05:05Z"),
        ),
        images=("SJI_1400", "SJI_2796"),
        stride=4,
    ),
    "2016-12-10": Campaign(
        windows=(("2016.12.10_05:48:13Z", "2016.12.10_07:55:52Z"),),
        images=("FUV", "SJI_1330", "SJI_2796"),
        stride=1,
        obsid="3620259342",
    ),
    "2016-12-12": Campaign(
        windows=(("2016.12.12_17:26:24Z", "2016.12.12_21:23:28Z"),),
        images=("FUV", "SJI_1400", "SJI_2796"),
        stride=1,
        obsid="3620110467",
    ),
    "2019-01-08": Campaign(
        windows=(("2019.01.08_00:00:05Z", "2019.01.08_09:53:06Z"),),
        images=("FUV", "SJI_1330"),
        stride=2,
        obsid="3660109523",
    ),
    "2020-01-05": Campaign(
        windows=(("2020.01.05_06:58:08Z", "2020.01.05_09:10:43Z"),),
        images=("FUV", "SJI_1400", "SJI_2796"),
        stride=1,
        obsid="3680109414",
    ),
    "2020-11-20": Campaign(
        windows=(("2020.11.20_05:23:16Z", "2020.11.20_07:28:13Z"),),
        images=("FUV", "SJI_2796"),
        stride=3,
        obsid="3610609752",
    ),
    "2021-12-31": Campaign(
        windows=(("2021.12.31_04:51:40Z", "2021.12.31_11:25:17Z"),),
        images=("FUV", "SJI_2796"),
        stride=5,
        obsid="3610611752",
    ),
    "2022-12-13": Campaign(
        windows=(("2022.12.13_07:59:13Z", "2022.12.13_10:48:58Z"),),
        images=("FUV", "SJI_1330"),
        stride=2,
        obsid="3660259533",
    ),
    "2024-12-28": Campaign(
        windows=(("2024.12.28_05:18:18Z", "2024.12.28_11:01:32Z"),),
        images=("FUV", "SJI_2796"),
        stride=4,
        obsid="3610611752",
    ),
    "2025-12-31": Campaign(
        windows=(("2025.12.31_07:08:13Z", "2025.12.31_12:51:06Z"),),
        images=("FUV", "SJI_2796"),
        stride=4,
        obsid="3610611752",
    ),
    "2026-01-10": Campaign(
        windows=(("2026.01.10_06:24:14Z", "2026.01.10_12:06:40Z"),),
        images=("FUV", "SJI_2796"),
        stride=4,
        obsid="3610611752",
    ),
}
"""
The observing campaigns searched for tracks, keyed as
:data:`ccd_diffusion.tracks.datasets`, which holds what the article says
about each of them.

The first five are the campaigns of the original measurement. The rest
were chosen by ``python -m ccd_diffusion.tracks search`` on 2026-09-20 for
the good tracks they should yield per frame fetched, at most two
program-days from any one year and exposures of 8 to 15 s, since a longer
exposure crowds the frame with hits that spoil the tracks around them,
with the frames outside the anomaly strided to leave at least about 150
per camera for the background.
"""


def records(window: tuple[str, str], timeout: float = 900) -> list[dict[str, str]]:
    """
    Every level-1 exposure in a time range, as the catalog describes it.

    Parameters
    ----------
    window
        The time range, a pair of ``YYYY.MM.DD_hh:mm:ssZ`` strings.
    timeout
        How long to wait on the catalog, in seconds.
    """
    start, stop = window
    query = urllib.parse.urlencode(
        dict(
            op="rs_list",
            ds=f"{_series}[{start}-{stop}]",
            key=",".join(_keys),
            R=0,
        )
    )
    with urllib.request.urlopen(f"{url_catalog}?{query}", timeout=timeout) as response:
        answer = json.load(response)
    if answer.get("status"):
        raise RuntimeError(f"the catalog refused the query for {window}: {answer}")
    values = {k["name"]: k["values"] for k in answer["keywords"]}
    return [
        {k: values[k][i] for k in _keys} for i in range(int(answer.get("count", 0)))
    ]


def select(dataset: str, verbose: bool = True) -> list[dict[str, str]]:
    """
    The frames of a campaign to search for tracks, as rows of
    ``data/iris_frames.csv``.

    Parameters
    ----------
    dataset
        The campaign, a key of :data:`campaigns`.
    verbose
        Whether to report each window as it is fetched.
    """
    campaign = campaigns[dataset]
    result = []
    for window in campaign.windows:
        rows = records(window)
        if campaign.obsid is not None:
            rows = [r for r in rows if r["ISQOLTID"] == campaign.obsid]
        for image in campaign.images:
            mine = sorted(
                (r for r in rows if r["IMG_PATH"] == image),
                key=lambda r: int(r["FSN"]),
            )
            keep = {int(r["FSN"]) for r in mine[:: campaign.stride]}
            keep |= {int(r["FSN"]) for r in mine if r["SAA"] == "1"}
            result += [
                dict(
                    dataset=dataset,
                    fsn=r["FSN"],
                    time=r["T_OBS"],
                    image=image,
                    saa=r["SAA"],
                    tracks=0,
                )
                for r in mine
                if int(r["FSN"]) in keep
            ]
        if verbose:
            print(f"{dataset}: {window[0]} to {window[1]}, {len(result)} frames so far")
    result.sort(key=lambda r: int(r["fsn"]))
    return result


def save_frames(
    frames: list[dict[str, str]],
    directory: None | pathlib.Path = None,
) -> None:
    """
    Write the frames to ``data/iris_frames.csv``, where
    :func:`ccd_diffusion.tracks.frames` will find them.

    The number of tracks in each frame is left at zero until
    :func:`ccd_diffusion.tracks.save_tracks` fills it in.

    Parameters
    ----------
    frames
        The frames of every campaign, as :func:`select` returns them.
    directory
        The data directory of the package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    fields = ["dataset", "fsn", "time", "image", "saa", "tracks"]
    with open(pathlib.Path(directory) / "iris_frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frames)
