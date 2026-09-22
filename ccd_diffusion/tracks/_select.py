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
import datetime
import http.client
import time
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
    "2014-08-20": Campaign(
        windows=(("2014.08.20_07:54:50Z", "2014.08.20_10:30:52Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=5,
        obsid="3820009453",
    ),
    "2014-08-21": Campaign(
        windows=(("2014.08.21_07:44:33Z", "2014.08.21_10:54:52Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400"),
        stride=6,
        obsid="3820009253",
    ),
    "2014-08-23": Campaign(
        windows=(("2014.08.23_07:54:33Z", "2014.08.23_10:03:52Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=2,
        obsid="3820009453",
    ),
    "2014-08-24": Campaign(
        windows=(("2014.08.24_08:08:08Z", "2014.08.24_10:27:52Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400"),
        stride=4,
        obsid="3820009253",
    ),
    "2014-08-25": Campaign(
        windows=(("2014.08.25_07:46:44Z", "2014.08.25_10:30:00Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=6,
        obsid="3820009453",
    ),
    "2014-11-02": Campaign(
        windows=(("2014.11.02_09:03:04Z", "2014.11.02_12:16:39Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_2796"),
        stride=4,
        obsid="3820009377",
    ),
    "2014-11-27": Campaign(
        windows=(("2014.11.27_13:59:30Z", "2014.11.27_19:05:22Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796"),
        stride=4,
        obsid="3860009154",
    ),
    "2015-03-26": Campaign(
        windows=(("2015.03.26_10:04:13Z", "2015.03.26_13:29:09Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_2796"),
        stride=9,
        obsid="3820009359",
    ),
    "2015-09-20": Campaign(
        windows=(("2015.09.20_07:39:15Z", "2015.09.20_10:48:51Z"),),
        images=("FUV", "NUV", "SJI_2796"),
        stride=6,
        obsid="3623008713",
    ),
    "2016-01-02": Campaign(
        windows=(("2016.01.02_05:38:25Z", "2016.01.02_11:21:13Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=1,
        obsid="3620008076",
    ),
    "2016-01-06": Campaign(
        windows=(("2016.01.06_18:29:15Z", "2016.01.06_19:20:03Z"),),
        images=("FUV", "NUV"),
        stride=1,
        obsid="3620008076",
    ),
    "2016-05-16": Campaign(
        windows=(("2016.05.16_07:31:33Z", "2016.05.16_12:01:05Z"),),
        images=("FUV", "NUV", "SJI_1400", "SJI_2796"),
        stride=10,
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
    "2016-12-13": Campaign(
        windows=(("2016.12.13_12:56:13Z", "2016.12.13_20:17:11Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=6,
        obsid="3620008063",
    ),
    "2016-12-17": Campaign(
        windows=(("2016.12.17_12:47:13Z", "2016.12.17_16:07:05Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=2,
        obsid="3620008063",
    ),
    "2016-12-25": Campaign(
        windows=(("2016.12.25_12:52:15Z", "2016.12.25_19:48:40Z"),),
        images=("FUV", "NUV", "SJI_1330", "SJI_1400", "SJI_2796", "SJI_2832"),
        stride=6,
        obsid="3630008064",
    ),
}
"""
The observing campaigns searched for tracks, keyed as
:data:`ccd_diffusion.tracks.datasets`, which holds what the article says
about each of them.

The first five are the campaigns of the original measurement. The rest
were chosen by ``python -m ccd_diffusion.tracks search`` on 2026-09-22
among programs read at full resolution with 8 s exposures, since a longer
exposure crowds the frame with hits that spoil the tracks around them,
whose spectrograph window looks at least half off the limb, where tracks
are sought (the pointing keywords give the center of the field, and a
program pointed beyond the limb can still read out a window of the slit
that lies on the disk), and with at least sixty frames inside the anomaly.
The mission holds only fourteen such program-days, so all but the
shortest are taken, and four limb pointings chosen before the window was
checked stay for the tracks they yield off the limb. The frames outside
the anomaly are strided to leave about 130 per spectrograph camera and at
least 60 per slit-jaw channel for the background; a slit-jaw channel with
too few is left out.
"""


_retries = (10, 30, 60)
"""
How long to wait, in seconds, before asking the catalog again after a
query times out or breaks off.
"""

_depth_maximum = 4
"""How many times a window is halved before a failing query is given up."""


def _parse_window_time(t: str) -> datetime.datetime:
    """A record-set time, ``YYYY.MM.DD_hh:mm:ssZ``, as a datetime."""
    return datetime.datetime.strptime(t, "%Y.%m.%d_%H:%M:%SZ")


def _records(window: tuple[str, str], timeout: float) -> list[dict[str, str]]:
    """One query to the catalog for every level-1 exposure in a time range."""
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


def records(
    window: tuple[str, str], timeout: float = 900, depth: int = 0
) -> list[dict[str, str]]:
    """
    Every level-1 exposure in a time range, as the catalog describes it.

    The catalog is slow at times and a long window can take longer than
    the timeout to answer, so a query that times out or breaks off is asked
    again after each pause in :data:`_retries`, and when it keeps failing
    the window is halved and each half asked for, :data:`_depth_maximum`
    times over. A refusal is raised at once, since asking again would not
    change it.

    Parameters
    ----------
    window
        The time range, a pair of ``YYYY.MM.DD_hh:mm:ssZ`` strings.
    timeout
        How long to wait on the catalog, in seconds.
    depth
        How many times the window has been halved already.
    """
    start, stop = window
    for wait in (*_retries, None):
        try:
            return _records(window, timeout)
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            if wait is None:
                if depth >= _depth_maximum:
                    raise
                t0, t1 = _parse_window_time(start), _parse_window_time(stop)
                mid = (t0 + (t1 - t0) / 2).strftime("%Y.%m.%d_%H:%M:%SZ")
                return records((start, mid), timeout, depth + 1) + records(
                    (mid, stop), timeout, depth + 1
                )
            time.sleep(wait)
    raise AssertionError("unreachable")


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
