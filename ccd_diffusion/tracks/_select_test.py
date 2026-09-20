import pathlib
import csv
import io
import json
import pytest
import ccd_diffusion
from . import _select


def test_campaigns_match_datasets():
    assert sorted(ccd_diffusion.tracks.campaigns) == sorted(
        ccd_diffusion.tracks.datasets
    )
    for campaign in ccd_diffusion.tracks.campaigns.values():
        assert campaign.windows
        assert campaign.images
        assert campaign.stride >= 1
        for start, stop in campaign.windows:
            assert start < stop


def _catalog(num: int = 40, saa: "None | range" = None):
    """A stand-in for the level-1 catalog: one FUV and one NUV frame per step."""
    if saa is None:
        saa = range(10, 20)
    rows = []
    for i in range(num):
        for offset, image in enumerate(("FUV", "NUV")):
            rows.append(
                dict(
                    T_OBS=f"2018-05-04T07:{i // 60:02d}:{i % 60:02d}.00Z",
                    FSN=str(1000 + 2 * i + offset),
                    IMG_PATH=image,
                    SAA="1" if i in saa else "0",
                    ISQOLTID="3620011417" if i < 30 else "9999999999",
                )
            )
    return rows


def test_select(monkeypatch):
    monkeypatch.setattr(_select, "records", lambda window, **kw: _catalog())
    monkeypatch.setitem(
        _select.campaigns,
        "test",
        _select.Campaign(windows=(("a", "b"),), images=("FUV",), stride=4),
    )
    result = _select.select("test", verbose=False)
    assert all(r["image"] == "FUV" for r in result)
    assert all(r["dataset"] == "test" for r in result)
    assert [int(r["fsn"]) for r in result] == sorted(int(r["fsn"]) for r in result)
    # every frame inside the anomaly is kept
    assert sum(r["saa"] == "1" for r in result) == 10
    # and the quiet frames are subsampled
    quiet = [r for r in result if r["saa"] == "0"]
    assert 0 < len(quiet) < 30
    assert all(r["tracks"] == 0 for r in result)


def test_select_stride_one(monkeypatch):
    monkeypatch.setattr(_select, "records", lambda window, **kw: _catalog())
    monkeypatch.setitem(
        _select.campaigns,
        "test",
        _select.Campaign(windows=(("a", "b"),), images=("FUV",)),
    )
    assert len(_select.select("test", verbose=False)) == 40


def test_select_many_windows(monkeypatch):
    monkeypatch.setattr(
        _select, "records", lambda window, **kw: _catalog(num=10, saa=range(0))
    )
    monkeypatch.setitem(
        _select.campaigns,
        "test",
        _select.Campaign(windows=(("a", "b"), ("c", "d")), images=("FUV", "NUV")),
    )
    result = _select.select("test", verbose=False)
    # both windows return the same stand-in catalog, so each frame appears twice
    assert len(result) == 40
    assert {r["image"] for r in result} == {"FUV", "NUV"}


def test_records_rejects_a_refusal(monkeypatch):
    monkeypatch.setattr(
        _select.urllib.request,
        "urlopen",
        lambda *a, **k: io.StringIO(json.dumps(dict(status=1, error="bad query"))),
    )
    with pytest.raises(RuntimeError, match="refused"):
        _select.records(("a", "b"))


def test_save_frames(tmp_path: pathlib.Path):
    frames = [dict(f) for f in ccd_diffusion.tracks.frames()][:20]
    for f in frames:
        f["tracks"] = 0
    ccd_diffusion.tracks.save_frames(frames, tmp_path)
    with open(tmp_path / "iris_frames.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(frames)
    assert list(rows[0]) == ["dataset", "fsn", "time", "image", "saa", "tracks"]
    assert [r["fsn"] for r in rows] == [f["fsn"] for f in frames]


def test_select_keeps_one_program(monkeypatch):
    monkeypatch.setattr(_select, "records", lambda window, **kw: _catalog())
    monkeypatch.setitem(
        _select.campaigns,
        "test",
        _select.Campaign(windows=(("a", "b"),), images=("FUV",), obsid="3620011417"),
    )
    result = _select.select("test", verbose=False)
    # the last ten steps belong to another program and are dropped
    assert len(result) == 30
    assert max(int(r["fsn"]) for r in result) < 1000 + 2 * 30
