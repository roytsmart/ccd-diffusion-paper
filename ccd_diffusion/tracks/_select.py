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
    """The cameras to keep, ``FUV``, ``NUV``, or the ``SJI_*`` channels."""

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
        images=(
            "FUV",
            "NUV",
        ),
    ),
    "2014": Campaign(
        windows=(("2014.04.07_11:18:30Z", "2014.04.07_12:49:20Z"),),
        images=(
            "FUV",
            "NUV",
        ),
    ),
    "2018": Campaign(
        windows=(
            ("2018.03.16_11:20:55Z", "2018.03.16_12:11:00Z"),
            ("2018.03.16_12:49:38Z", "2018.03.16_13:25:30Z"),
        ),
        images=(
            "FUV",
            "NUV",
        ),
    ),
    "2018may": Campaign(
        windows=(("2018.05.04_07:12:20Z", "2018.05.04_11:58:45Z"),),
        images=(
            "FUV",
            "NUV",
        ),
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
    "2013-12-23": Campaign(
        windows=(("2013.12.23_17:39:42Z", "2013.12.23_23:13:34Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=6,
        obsid="3820009492",
    ),
    "2014-08-23": Campaign(
        windows=(("2014.08.23_07:54:33Z", "2014.08.23_10:03:52Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=2,
        obsid="3820009453",
    ),
    "2014-12-26": Campaign(
        windows=(("2014.12.26_00:33:53Z", "2014.12.26_16:08:14Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_2796"),
        stride=14,
        obsid="3880009380",
    ),
    "2015-04-05": Campaign(
        windows=(("2015.04.05_17:51:15Z", "2015.04.05_23:50:57Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400"),
        stride=15,
        obsid="3860009280",
    ),
    "2015-09-20": Campaign(
        windows=(("2015.09.20_07:39:15Z", "2015.09.20_10:48:51Z"),),
        images=("FUV", "NUV", "SJI_2796"),
        stride=6,
        obsid="3623008713",
    ),
    "2015-10-30": Campaign(
        windows=(("2015.10.30_00:00:01Z", "2015.10.30_19:01:19Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=15,
        obsid="3660008003",
    ),
    "2016-01-02": Campaign(
        windows=(("2016.01.02_05:38:25Z", "2016.01.02_11:21:13Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=1,
        obsid="3620008076",
    ),
    "2016-05-17": Campaign(
        windows=(("2016.05.17_07:39:15Z", "2016.05.17_12:32:37Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=11,
        obsid="3640008423",
    ),
    "2016-10-28": Campaign(
        windows=(("2016.10.28_07:15:38Z", "2016.10.28_08:47:24Z"),),
        images=(
            "FUV",
            "NUV",
        ),
        stride=3,
        obsid="3640009123",
    ),
    "2017-10-19": Campaign(
        windows=(("2017.10.19_19:33:50Z", "2017.10.19_22:43:46Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=3,
        obsid="3640008059",
    ),
    "2018-02-25": Campaign(
        windows=(("2018.02.25_05:12:12Z", "2018.02.25_09:18:00Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=5,
        obsid="3640008059",
    ),
    "2020-07-01": Campaign(
        windows=(("2020.07.01_06:52:33Z", "2020.07.01_13:00:52Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=12,
        obsid="3600008435",
    ),
}
"""
The observing campaigns searched for tracks, keyed as
:data:`ccd_diffusion.tracks.datasets`, which holds what the article says
about each of them.

The first five are the campaigns of the original measurement. The rest
were chosen by ``python -m ccd_diffusion.tracks search`` on 2026-09-21 for
the good tracks they should yield per frame fetched, among programs read
at full resolution with 8 s exposures, at most three program-days from any
one year, with the frames outside the anomaly
strided to leave about 130 per spectrograph camera and at least 60 per
slit-jaw channel for the background. Exposures of 15 s, which the search
first allowed, crowd the frame with hits that spoil the tracks around them:
per anomaly frame they yielded half the flat tracks of the 8 s campaigns,
and two thirds of their tracks failed the cuts of the article against one
half. A slit-jaw channel with too few frames outside the anomaly for its
background at the campaign's stride is left out.
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
