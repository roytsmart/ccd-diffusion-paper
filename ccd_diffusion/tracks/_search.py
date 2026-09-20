"""
Find observations worth adding to the campaign table.

A good campaign points the slit at or beyond the limb, so the frames are
dark, exposes for several seconds, so a track outshines the read noise,
and holds many frames taken inside the South Atlantic Anomaly, which is
what its yield of tracks scales with. The level-1 catalog at the Joint
Science Operations Center can list, for a whole month, only the
far-ultraviolet frames taken inside the anomaly, each with its pointing,
exposure, roll, and observing program, so a sweep of the mission is one
query per month. The frames are grouped by program and day into candidate
campaigns, cut, and the survivors asked for their full span, which is the
window the campaign table needs.
"""

import os
import csv
import json
import time
import dataclasses
import datetime
import pathlib
import statistics
import urllib.parse
import urllib.request
import concurrent.futures
from ._select import url_catalog, _series

__all__ = [
    "Observation",
    "anomaly_month",
    "runs",
    "span",
    "search",
    "save_search",
    "load_search",
]

_keys = "T_OBS,ISQOLTID,XCEN,YCEN,EXPTIME,SAT_ROT"
"""The keywords fetched for every anomaly frame."""

_cache_default = (
    pathlib.Path(
        os.environ.get(
            "CCD_DIFFUSION_CACHE",
            pathlib.Path.home() / ".cache" / "ccd_diffusion" / "iris",
        )
    ).parent
    / "anomaly"
)
"""Where each month's anomaly frames are kept, beside the image cache."""


@dataclasses.dataclass(eq=False)
class Observation:
    """One observing program on one day, as a candidate campaign."""

    obsid: str
    """The observing program, the catalog's ``ISQOLTID``."""

    day: str
    """The day, ``YYYY-MM-DD``."""

    x: float
    """The median pointing in helioprojective x, arcseconds from disk center."""

    y: float
    """The median pointing in helioprojective y."""

    radius: float
    """The distance of the pointing from disk center, in arcseconds."""

    roll: float
    """The spacecraft roll in degrees."""

    exposure: float
    """The median far-ultraviolet exposure time in seconds."""

    anomaly: int
    """The number of far-ultraviolet frames taken inside the anomaly."""

    start: str = ""
    """The time of the program's first far-ultraviolet frame that day, once :func:`span` has run."""

    stop: str = ""
    """The time of its last."""

    frames: int = 0
    """The number of far-ultraviolet frames it took that day."""

    quiet: int = 0
    """The number of those taken outside the anomaly."""

    @property
    def seconds(self) -> float:
        """
        The exposed seconds inside the anomaly, the anomaly frames times the
        exposure, which the number of tracks scales with: the original
        campaigns gave about one track for every five such seconds.
        """
        return self.anomaly * self.exposure

    @property
    def hours(self) -> float:
        """The span of the program that day, in hours."""
        if not self.start or not self.stop:
            return 0.0
        return (_parse(self.stop) - _parse(self.start)).total_seconds() / 3600

    @property
    def window(self) -> tuple[str, str]:
        """The span as :data:`ccd_diffusion.tracks.campaigns` writes it."""
        return _catalog_time(self.start), _catalog_time(self.stop)


def _parse(t_obs: str) -> datetime.datetime:
    """A catalog ``T_OBS``, ``YYYY-MM-DDThh:mm:ss.ssZ``, as a datetime."""
    return datetime.datetime.strptime(t_obs[:19], "%Y-%m-%dT%H:%M:%S")


def _catalog_time(t_obs: str) -> str:
    """A catalog ``T_OBS`` in the form a record-set query takes, ``YYYY.MM.DD_hh:mm:ssZ``."""
    return _parse(t_obs).strftime("%Y.%m.%d_%H:%M:%SZ")


def _query(ds: str, key: str, timeout: float = 900) -> dict[str, list[str]]:
    """The keyword columns of a record-set query, empty if it matched nothing."""
    query = urllib.parse.urlencode(dict(op="rs_list", ds=ds, key=key, R=0))
    with urllib.request.urlopen(f"{url_catalog}?{query}", timeout=timeout) as response:
        answer = json.load(response)
    if answer.get("status"):
        raise RuntimeError(f"the catalog refused {ds}: {answer}")
    if not answer.get("count"):
        return {k: [] for k in key.split(",")}
    return {k["name"]: k["values"] for k in answer["keywords"]}


def _months(start: str, stop: str) -> list[str]:
    """The months ``YYYY-MM`` from one to another, inclusive."""
    y, m = (int(v) for v in start.split("-")[:2])
    y1, m1 = (int(v) for v in stop.split("-")[:2])
    result = []
    while (y, m) <= (y1, m1):
        result.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return result


def _bounds(month: str) -> tuple[str, str]:
    """The record-set time range of a month."""
    y, m = (int(v) for v in month.split("-"))
    y1, m1 = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y}.{m:02d}.01_00:00:00Z", f"{y1}.{m1:02d}.01_00:00:00Z"


def anomaly_month(
    month: str, cache: None | pathlib.Path = _cache_default
) -> list[dict[str, str]]:
    """
    Every far-ultraviolet frame taken inside the anomaly in a month, with
    its time, program, pointing, exposure, and roll.

    Parameters
    ----------
    month
        The month, ``YYYY-MM``.
    cache
        Where to keep the answer so the month is only ever asked for once,
        :data:`_cache_default` unless told otherwise, nowhere if :obj:`None`.
    """
    path = None if cache is None else pathlib.Path(cache) / f"{month}.json"
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = _anomaly_frames(*_bounds(month))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


_retries = (10, 30, 60)
"""How long to wait, in seconds, before asking the catalog again after it refuses a query."""


def _anomaly_frames(start: str, stop: str, depth: int = 0) -> list[dict[str, str]]:
    """
    The anomaly frames in a time range, asking again after a pause when the
    catalog refuses, and asking for each half of the range when it keeps
    refusing, since a refusal can mean the answer was too large.
    """
    ds = f"{_series}[{start}-{stop}][? SAA=1 and IMG_PATH='FUV' ?]"
    for wait in (*_retries, None):
        try:
            columns = _query(ds, _keys)
            break
        except RuntimeError:
            if wait is None:
                if depth >= 4:
                    raise
                t0, t1 = _parse_query_time(start), _parse_query_time(stop)
                mid = (t0 + (t1 - t0) / 2).strftime("%Y.%m.%d_%H:%M:%SZ")
                return _anomaly_frames(start, mid, depth + 1) + _anomaly_frames(
                    mid, stop, depth + 1
                )
            time.sleep(wait)
    names = _keys.split(",")
    return [{k: columns[k][i] for k in names} for i in range(len(columns["T_OBS"]))]


def _parse_query_time(t: str) -> datetime.datetime:
    """A record-set time, ``YYYY.MM.DD_hh:mm:ssZ``, as a datetime."""
    return datetime.datetime.strptime(t, "%Y.%m.%d_%H:%M:%SZ")


def runs(rows: list[dict[str, str]]) -> list[Observation]:
    """
    Group anomaly frames into candidate campaigns, one per observing
    program and day.

    Parameters
    ----------
    rows
        Frames as :func:`anomaly_month` returns them.
    """
    groups = {}
    for r in rows:
        groups.setdefault((r["ISQOLTID"], r["T_OBS"][:10]), []).append(r)

    def number(rs, key):
        values = []
        for r in rs:
            try:
                values.append(float(r[key]))
            except (TypeError, ValueError):
                pass
        return statistics.median(values) if values else float("nan")

    result = []
    for (obsid, day), rs in sorted(groups.items()):
        x, y = number(rs, "XCEN"), number(rs, "YCEN")
        result.append(
            Observation(
                obsid=obsid,
                day=day,
                x=x,
                y=y,
                radius=(x**2 + y**2) ** 0.5,
                roll=number(rs, "SAT_ROT"),
                exposure=number(rs, "EXPTIME"),
                anomaly=len(rs),
            )
        )
    return result


def span(observation: Observation) -> Observation:
    """
    Fill in when a candidate's program started and stopped that day and
    how many frames it took, from the catalog.

    Parameters
    ----------
    observation
        The candidate; it is returned with :attr:`Observation.start`,
        :attr:`Observation.stop`, :attr:`Observation.frames`, and
        :attr:`Observation.quiet` set.
    """
    day = datetime.datetime.strptime(observation.day, "%Y-%m-%d")
    start = day.strftime("%Y.%m.%d_00:00:00Z")
    stop = (day + datetime.timedelta(days=1)).strftime("%Y.%m.%d_00:00:00Z")
    columns = _query(
        f"{_series}[{start}-{stop}][? ISQOLTID={observation.obsid} and IMG_PATH='FUV' ?]",
        "T_OBS,SAA",
    )
    times = columns["T_OBS"]
    if times:
        observation.start, observation.stop = min(times), max(times)
        observation.frames = len(times)
        observation.quiet = sum(s != "1" for s in columns["SAA"])
    return observation


def search(
    start: str,
    stop: str,
    radius: float = 940,
    exposure: float = 4,
    anomaly: int = 30,
    workers: int = 4,
    verbose: bool = True,
    cache: None | pathlib.Path = _cache_default,
) -> list[Observation]:
    """
    The candidate campaigns between two months, the most exposed seconds
    inside the anomaly first.

    Parameters
    ----------
    start
        The first month, ``YYYY-MM``.
    stop
        The last month, inclusive.
    radius
        The least distance of the pointing from disk center, in arcseconds;
        the limb is near 960.
    exposure
        The least exposure time in seconds.
    anomaly
        The fewest frames inside the anomaly.
    workers
        How many catalog queries to run at once.
    verbose
        Whether to report progress.
    cache
        Where each month's anomaly frames are kept between runs.
    """
    months = _months(start, stop)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        batches = list(pool.map(lambda m: anomaly_month(m, cache), months))
    found = []
    for month, rows in zip(months, batches):
        found += runs(rows)
        if verbose:
            print(f"{month}: {len(rows)} anomaly frames")
    kept = [
        o
        for o in found
        if o.radius >= radius and o.exposure >= exposure and o.anomaly >= anomaly
    ]
    if verbose:
        print(
            f"{len(found)} program-days, {len(kept)} pass the cuts; asking their spans"
        )
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        kept = list(pool.map(span, kept))
    kept.sort(key=lambda o: (-o.seconds, o.day))
    return kept


_fields = [f.name for f in dataclasses.fields(Observation)]


def save_search(rows: list[Observation], path: pathlib.Path) -> None:
    """
    Write the result of :func:`search` as a table.

    Parameters
    ----------
    rows
        The candidates.
    path
        The file to write, a ``.csv``.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields)
        writer.writeheader()
        for o in rows:
            writer.writerow(dataclasses.asdict(o))


def load_search(path: pathlib.Path) -> list[Observation]:
    """
    Read a table written by :func:`save_search`.

    Parameters
    ----------
    path
        The file to read.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        kwargs = dict(r)
        for k in ("x", "y", "radius", "roll", "exposure"):
            kwargs[k] = float(kwargs[k])
        for k in ("anomaly", "frames", "quiet"):
            kwargs[k] = int(kwargs[k])
        result.append(Observation(**kwargs))
    return result
