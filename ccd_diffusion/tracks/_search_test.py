import io
import json
import pathlib
import ccd_diffusion
from . import _search, _extract


def test_months():
    assert _search._months("2018-11", "2019-02") == [
        "2018-11",
        "2018-12",
        "2019-01",
        "2019-02",
    ]
    assert _search._months("2020-06", "2020-06") == ["2020-06"]
    assert _search._bounds("2018-12") == (
        "2018.12.01_00:00:00Z",
        "2019.01.01_00:00:00Z",
    )


def test_catalog_time():
    assert _search._catalog_time("2018-05-04T07:12:17.29Z") == "2018.05.04_07:12:17Z"


def _frame(
    t,
    obsid="3620011417",
    x="954.1",
    y="7.2",
    exp="14.999",
    rot="-90.0",
    spat="1",
    sptrl="1",
):
    return dict(
        T_OBS=t,
        ISQOLTID=obsid,
        XCEN=x,
        YCEN=y,
        EXPTIME=exp,
        SAT_ROT=rot,
        SUMSPAT=spat,
        SUMSPTRL=sptrl,
    )


def _answer(rows):
    names = list(rows[0]) if rows else _search._keys.split(",")
    return dict(
        status=0,
        count=len(rows),
        keywords=[dict(name=k, values=[r[k] for r in rows]) for k in names],
    )


def test_anomaly_month_caches(monkeypatch, tmp_path: pathlib.Path):
    rows = [_frame("2018-05-04T07:12:25.29Z"), _frame("2018-05-04T07:12:42.06Z")]
    calls = []

    def urlopen(url, timeout=0):
        calls.append(url)
        return io.BytesIO(json.dumps(_answer(rows)).encode())

    monkeypatch.setattr(_search.urllib.request, "urlopen", urlopen)
    got = ccd_diffusion.tracks.anomaly_month("2018-05", tmp_path)
    assert got == rows
    assert len(calls) == 1
    assert "SAA%3D1" in calls[0] and "2018.05.01_00%3A00%3A00Z" in calls[0]
    assert (tmp_path / "2018-05.json").exists()
    # the cache answers the second time
    assert ccd_diffusion.tracks.anomaly_month("2018-05", tmp_path) == rows
    assert len(calls) == 1


def test_anomaly_month_empty(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(
        _search.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(dict(status=0, count=0)).encode()),
    )
    assert ccd_diffusion.tracks.anomaly_month("2013-07", tmp_path) == []


def test_runs():
    rows = [
        _frame("2018-05-04T07:00:00.00Z"),
        _frame("2018-05-04T08:00:00.00Z", x="960.0"),
        _frame("2018-05-04T09:00:00.00Z", x="948.0", exp="bad"),
        _frame("2018-05-05T07:00:00.00Z"),
        _frame("2018-05-04T07:30:00.00Z", obsid="1", x="0", y="0", exp="4.0", rot="0"),
    ]
    result = ccd_diffusion.tracks.runs(rows)
    assert [(o.obsid, o.day, o.anomaly) for o in result] == [
        ("1", "2018-05-04", 1),
        ("3620011417", "2018-05-04", 3),
        ("3620011417", "2018-05-05", 1),
    ]
    o = result[1]
    assert o.x == 954.1 and o.exposure == 14.999 and o.roll == -90
    assert abs(o.radius - (954.1**2 + 7.2**2) ** 0.5) < 1e-9
    assert o.seconds == 3 * 14.999
    assert o.hours == 0.0 and o.start == ""


def test_span(monkeypatch):
    answer = dict(
        status=0,
        count=4,
        keywords=[
            dict(
                name="T_OBS",
                values=[
                    "2018-05-04T07:12:25.29Z",
                    "2018-05-04T11:58:42.36Z",
                    "2018-05-04T09:00:00.00Z",
                    "2018-05-04T10:00:00.00Z",
                ],
            ),
            dict(name="SAA", values=["1", "0", "0", "1"]),
        ],
    )
    calls = []

    def urlopen(url, timeout=0):
        calls.append(url)
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(_search.urllib.request, "urlopen", urlopen)
    o = _search.Observation("3620011417", "2018-05-04", 954, 7, 954, -90, 15, 2)
    ccd_diffusion.tracks.span(o)
    assert "ISQOLTID%3D3620011417" in calls[0]
    assert o.start == "2018-05-04T07:12:25.29Z" and o.stop == "2018-05-04T11:58:42.36Z"
    assert o.frames == 4 and o.quiet == 2
    assert abs(o.hours - 4.77) < 0.01
    assert o.window == ("2018.05.04_07:12:25Z", "2018.05.04_11:58:42Z")


def test_search(monkeypatch, tmp_path: pathlib.Path):
    by_month = {
        "2018-05": [_frame("2018-05-04T07:00:00.00Z")] * 50
        + [_frame("2018-05-06T07:00:00.00Z", obsid="disk", x="10", y="10")] * 50
        + [_frame("2018-05-07T07:00:00.00Z", obsid="fast", exp="1.0")] * 50
        + [_frame("2018-05-08T07:00:00.00Z", obsid="few")] * 5
        + [_frame("2018-05-09T07:00:00.00Z", obsid="long", exp="4.0")] * 500,
        "2018-06": [],
    }
    monkeypatch.setattr(_search, "anomaly_month", lambda m, cache=None: by_month[m])
    spans = []
    monkeypatch.setattr(_search, "span", lambda o: (spans.append(o.obsid), o)[1])
    result = ccd_diffusion.tracks.search(
        "2018-05", "2018-06", verbose=False, cache=None
    )
    # the disk pointing, the fast cadence, and the few frames are cut
    assert [o.obsid for o in result] == ["long", "3620011417"]
    assert sorted(spans) == ["3620011417", "long"]
    assert result[0].seconds == 2000 and result[1].anomaly == 50


def test_search_roundtrip(tmp_path: pathlib.Path):
    rows = [
        _search.Observation(
            "3620011417",
            "2018-05-04",
            954.1,
            7.2,
            954.13,
            -90,
            15,
            137,
            "2018-05-04T07:12:25.29Z",
            "2018-05-04T11:58:42.36Z",
            1040,
            903,
        ),
        _search.Observation("1", "2018-05-05", 0, 0, 0, 0, 4, 30),
    ]
    ccd_diffusion.tracks.save_search(rows, tmp_path / "search.csv")
    loaded = ccd_diffusion.tracks.load_search(tmp_path / "search.csv")
    assert [o.obsid for o in loaded] == ["3620011417", "1"]
    a = loaded[0]
    assert a.anomaly == 137 and a.quiet == 903 and a.exposure == 15 and a.roll == -90
    assert a.seconds == 137 * 15 and abs(a.hours - 4.77) < 0.01
    assert a.window == ("2018.05.04_07:12:25Z", "2018.05.04_11:58:42Z")
    assert loaded[1].start == "" and loaded[1].hours == 0.0


def test_configuration_default():
    assert _extract._configuration("sji") is _extract._config["sji"]
    default = _extract._configuration("2027-something-new")
    assert default is _extract._config_default
    assert default["block"] == "channel" and default["search"] == "saa"
    assert default["noise_maximum"] == "camera" and default["mask"] == "camera"
    assert _extract._mask_camera == {"FUV": "lines", "SJI": "limb"}


def test_mask_lines_keeps_rows():
    import numpy as np

    # a spectrograph frame with a broad emission line raising forty pixels of
    # every row, a tenth of the width of the quadrant it lies in
    bg = np.full((100, 1600), 100.0)
    bg[:, 50:90] += 20
    bg[10, 10] += 30  # a hot pixel
    noise = np.full_like(bg, 3.0)
    lines = _extract._mask("lines", bg, noise)
    median = _extract._mask("median", bg, noise)
    assert lines.mean() > 0.7
    assert not lines[:, 50:90].any() and not lines[10, 10]
    # the median mask cuts every row here, since each holds the line's pixels
    assert not median.any()


def test_anomaly_frames_retries_then_splits(monkeypatch):
    calls, naps = [], []
    monkeypatch.setattr(_search.time, "sleep", lambda t: naps.append(t))

    def query(ds, key, timeout=0):
        calls.append(ds)
        # the whole month is refused every time; each half answers at once
        if "2019.09.01_00:00:00Z-2019.10.01_00:00:00Z" in ds:
            raise RuntimeError("refused")
        return {
            k: (["2019-09-05T00:00:00.00Z"] if k == "T_OBS" else ["v"])
            for k in key.split(",")
        }

    monkeypatch.setattr(_search, "_query", query)
    rows = _search._anomaly_frames("2019.09.01_00:00:00Z", "2019.10.01_00:00:00Z")
    assert len(rows) == 2
    assert naps == list(_search._retries)
    assert len(calls) == len(_search._retries) + 1 + 2
    assert "2019.09.16_00:00:00Z" in calls[-1]


def test_anomaly_frames_gives_up(monkeypatch):
    monkeypatch.setattr(_search.time, "sleep", lambda t: None)
    monkeypatch.setattr(
        _search,
        "_query",
        lambda ds, key, timeout=0: (_ for _ in ()).throw(RuntimeError("no")),
    )
    try:
        _search._anomaly_frames("2019.09.01_00:00:00Z", "2019.10.01_00:00:00Z")
    except RuntimeError:
        pass
    else:
        raise AssertionError("a range the catalog keeps refusing was accepted")


def test_runs_drops_binned_frames():
    rows = [
        _frame("2018-05-04T07:00:00.00Z"),
        _frame("2018-05-04T08:00:00.00Z", spat="2", sptrl="2"),
        _frame("2018-05-04T09:00:00.00Z", spat="1", sptrl="4"),
        _frame("2018-05-05T07:00:00.00Z", obsid="binned", spat="2", sptrl="4"),
    ]
    result = ccd_diffusion.tracks.runs(rows)
    assert [(o.obsid, o.anomaly) for o in result] == [("3620011417", 1)]
