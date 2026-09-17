import pathlib
import dataclasses
import numpy as np
import named_arrays as na
import ccd_diffusion
from . import _extract


def test_url():
    result = ccd_diffusion.tracks.url("2018-05-04T10:18:40.34Z", "FUV")
    assert result == (
        "https://www.lmsal.com/solarsoft/irisa/data/level1/2018/05/04/H1000/"
        "iris20180504_10184034_fuv.fits"
    )
    assert ccd_diffusion.tracks.url("2018-05-04T10:18:23.80Z", "SJI_2796").endswith(
        "iris20180504_10182380_sji.fits"
    )


def test_path(tmp_path: pathlib.Path):
    frame = dict(dataset="sji", fsn="39498499")
    assert (
        ccd_diffusion.tracks.path(frame, tmp_path) == tmp_path / "sji" / "39498499.fits"
    )


def test_blocks():
    # only the campaigns whose blocks need no header keywords, which would fetch frames
    for dataset in ("2014", "2014b", "2018may", "sji"):
        blocks = ccd_diffusion.tracks.blocks(dataset)
        assert blocks
        frames = [f for b in blocks for f in b.frames]
        assert len(frames) == sum(
            1 for f in ccd_diffusion.tracks.frames() if f["dataset"] == dataset
        )
        for b in blocks:
            assert all(f["saa"] == "1" for f in b.search) or dataset == "2014"


def test_background():
    rng = np.random.default_rng(0)
    stack = 100 + 3 * rng.standard_normal((40, 50, 60))
    stack[:, :, :5] = 0  # unread columns
    stack[3, 10, 10] = 5000  # a particle hit in one frame
    bg, noise = ccd_diffusion.tracks.background(stack)
    assert bg.shape == (50, 60)
    assert np.isnan(bg[:, :5]).all()
    assert abs(bg[10, 10] - 100) < 3  # the trimmed mean rejects the hit
    assert abs(np.nanmedian(noise) - 3) < 0.5


def _frame_with_track(gain: float, noise: float, slope: float = 0.1):
    """A synthetic background-subtracted frame holding one glancing track."""
    rng = np.random.default_rng(1)
    residual = noise * rng.standard_normal((120, 140)).astype(np.float32)
    rows = np.arange(30, 60)
    line = 70 + slope * (rows - rows[0])
    width = np.linspace(0.4, 0.05, rows.size)  # the wedge, in pixels
    columns = np.arange(140)
    for r, c, w in zip(rows, line, width):
        edges = columns[:, None] + np.array([[-0.5, 0.5]])
        from scipy.special import erf

        cdf = erf((edges - c) / (np.sqrt(2) * max(w, 0.05))) / 2
        residual[r] += (2000 / gain) * (cdf[:, 1] - cdf[:, 0])
    return residual, rows, line


def test_find_synthetic():
    gain, noise = 6.0, 3.0
    residual, rows, line = _frame_with_track(gain, noise)
    frame = dict(dataset="test", fsn="1", saa="1")
    valid = np.ones_like(residual, dtype=bool)
    found = ccd_diffusion.tracks.find(frame, residual, valid, noise, gain)
    assert len(found) == 1
    track = found[0]
    assert track.vertical
    assert track.length == rows.size
    assert track.row == rows[0]
    assert abs(track.slope - 0.1) < 0.02
    assert track.noise == noise * gain
    assert track.gain == gain
    assert np.allclose(track.signal.ndarray, 2000, rtol=0.1)
    assert np.all(np.abs(track.position.ndarray) <= 0.5)


def test_find_rejects_steep():
    gain, noise = 6.0, 3.0
    residual, rows, line = _frame_with_track(gain, noise, slope=0.6)
    frame = dict(dataset="test", fsn="1", saa="1")
    valid = np.ones_like(residual, dtype=bool)
    assert ccd_diffusion.tracks.find(frame, residual, valid, noise, gain) == []


def test_census_synthetic():
    gain, noise = 6.0, 3.0
    residual, rows, line = _frame_with_track(gain, noise)
    frame = dict(dataset="test", fsn="1", saa="1")
    valid = np.ones_like(residual, dtype=bool)
    components = ccd_diffusion.tracks.census(frame, residual, valid, noise, gain)
    assert len(components) == 1
    c = components[0]
    assert isinstance(c, ccd_diffusion.tracks.Component)
    assert abs(c.azimuth - np.degrees(np.arctan(0.1))) < 3
    assert c.length > 4 * c.width
    assert c.charge > 0.8 * 2000 * rows.size


def test_save_tracks(tmp_path: pathlib.Path):
    tracks = ccd_diffusion.tracks.load()[:5]
    ccd_diffusion.tracks.save_tracks(list(tracks), tmp_path)
    assert (tmp_path / "iris_tracks.npz").exists()
    assert (tmp_path / "iris_tracks.csv").exists()
    frames = (tmp_path / "iris_frames.csv").read_text().splitlines()
    assert frames[0] == "dataset,fsn,time,image,saa,tracks"
    assert len(frames) == len(ccd_diffusion.tracks.frames()) + 1


def test_save_census(tmp_path: pathlib.Path):
    components = [
        ccd_diffusion.tracks.Component("2014b", 1, True, 10.0, 20.0, 2.0, 30, 5000.0)
    ]
    ccd_diffusion.tracks.save_census(components, tmp_path)
    text = (tmp_path / "iris_azimuth.csv").read_text().splitlines()
    assert text[0].startswith("dataset,fsn,saa,azimuth")
    assert text[1].startswith("2014b,1,True,10.000")


def test_chip():
    track = ccd_diffusion.tracks.Track(
        name="",
        dataset="2018may",
        chip="",
        fsn=0,
        slope=0,
        noise=1,
        gain=6,
        vertical=True,
        row=0,
        column=3000,
        charge=na.ScalarArray(np.ones((12, 7)), axes=("slice", "pixel")),
        position=na.ScalarArray(np.zeros(12), axes="slice"),
    )
    assert _extract._chip(track, "FUV", 4144) == "FUV2"
    track.column = 500
    assert _extract._chip(track, "FUV", 4144) == "FUV1"
    assert _extract._chip(track, "SJI_2796", 2072) == "SJI"


def test_load_roundtrip(tmp_path: pathlib.Path):
    tracks = ccd_diffusion.tracks.load()[:7]
    ccd_diffusion.tracks.save_tracks(list(tracks), tmp_path)
    loaded = ccd_diffusion.tracks.load(tmp_path)
    assert len(loaded) == len(tracks)
    for a, b in zip(tracks, loaded):
        assert a.name == b.name and a.chip == b.chip and a.fsn == b.fsn
        assert np.allclose(a.charge.ndarray, b.charge.ndarray)
        assert np.allclose(a.position.ndarray, b.position.ndarray)


def test_load_census(tmp_path: pathlib.Path):
    components = [
        ccd_diffusion.tracks.Component("2014b", 1, True, 10.0, 20.0, 2.0, 30, 5000.0),
        ccd_diffusion.tracks.Component("sji", 2, False, -80.0, 15.0, 1.5, 20, 900.0),
    ]
    ccd_diffusion.tracks.save_census(components, tmp_path)
    loaded = ccd_diffusion.tracks.load_census(tmp_path)
    assert [dataclasses.asdict(c) for c in loaded] == [
        dataclasses.asdict(c) for c in components
    ]


def test_main_merge(tmp_path: pathlib.Path, monkeypatch):
    from ccd_diffusion.tracks import __main__ as main
    from ccd_diffusion.tracks import _extract

    tracks = ccd_diffusion.tracks.load()
    by_dataset = {}
    for t in tracks:
        by_dataset.setdefault(t.dataset, []).append(t)
    parts = []
    for dataset, ts in by_dataset.items():
        part = tmp_path / dataset
        part.mkdir()
        ccd_diffusion.tracks.save_tracks(ts, part)
        ccd_diffusion.tracks.save_census(
            [ccd_diffusion.tracks.Component(dataset, 1, True, 0.0, 9.0, 1.0, 9, 1.0)],
            part,
        )
        parts.append(part)
    merged = tmp_path / "merged"
    merged.mkdir()
    monkeypatch.setattr(_extract, "_directory_data", merged)
    main.main(["merge", *map(str, reversed(parts))])
    loaded = ccd_diffusion.tracks.load(merged)
    assert [t.name for t in loaded] == [t.name for t in tracks]
    assert len(ccd_diffusion.tracks.load_census(merged)) == len(parts)
