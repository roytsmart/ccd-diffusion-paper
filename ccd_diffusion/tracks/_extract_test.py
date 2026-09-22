import pathlib
import dataclasses
import json
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


def test_main_plan_and_export(tmp_path: pathlib.Path, capsys):
    from ccd_diffusion.tracks import __main__ as main

    main.main(["plan", "--json", str(tmp_path / "plan.json")])
    out = capsys.readouterr().out
    assert all(d in out for d in ccd_diffusion.tracks.campaigns)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert [p["dataset"] for p in plan] == list(ccd_diffusion.tracks.campaigns)
    assert all(p["status"] in ("fresh", "stale", "new") for p in plan)

    main.main(["export", "--dataset", "2018", "--output", str(tmp_path / "2018")])
    assert "2018:" in capsys.readouterr().out
    exported = ccd_diffusion.tracks.load(tmp_path / "2018")
    assert all(t.dataset == "2018" for t in exported)
    assert len(exported) == sum(
        t.dataset == "2018" for t in ccd_diffusion.tracks.load()
    )
    assert (tmp_path / "2018" / "iris_campaigns.csv").exists()


def test_mask_levels_each_quadrant():
    # a spectrograph frame whose second CCD sits a few data numbers above the
    # first, one quadrant of it higher still, with an emission line on each
    # CCD and the disk lighting the lower rows of a limb pointing
    rng = np.random.default_rng(2)
    bg = 100 + 0.3 * rng.standard_normal((200, 400)).astype(np.float32)
    bg[:, 200:] += 4
    bg[:100, 300:] += 2  # the upper right quadrant of the second CCD
    bg[130:, :] += 3  # the disk, across the lower quadrants of both CCDs
    bg[:, 50:53] += 30  # an emission line on each CCD
    bg[:, 300:303] += 30
    bg[:, :10] = np.nan  # unread columns
    noise = np.full_like(bg, 3.0)
    mask = _extract._mask("lines", bg, noise, "FUV")
    assert not mask[:, :10].any()
    assert not mask[:, 49:54].any()
    assert not mask[:, 299:304].any()
    assert mask[:, 10:49].mean() > 0.95
    assert mask[:, 200:299].mean() > 0.95
    assert mask[:, 304:].mean() > 0.95
    # every quadrant sits at its own pedestal once levelled, so the dark rows
    # of the four upper quadrants agree, as do those of the four lower ones
    level = _extract._level(bg, "FUV")
    upper = [np.nanmedian(level[:100, 100 * i : 100 * (i + 1)]) for i in range(4)]
    lower = [np.nanmedian(level[100:130, 100 * i : 100 * (i + 1)]) for i in range(4)]
    assert np.ptp(upper) < 0.3, upper
    assert np.ptp(lower) < 0.3, lower
    assert not np.isfinite(bg[:, :10]).any() and (level[:, :10] == 0).all()


def test_level_stands_on_the_pedestal_under_the_disk():
    # a slit-jaw frame in the left half of its CCD, the disk covering most
    # of the upper quadrant, whose pedestal sits one data number above the
    # lower one: levelling must find the dark peak, not the disk
    rng = np.random.default_rng(3)
    bg = 104 + 1.5 * rng.standard_normal((200, 400)).astype(np.float32)
    bg[:100] += 1
    bg[:80, :200] += 200 + 50 * rng.standard_normal((80, 200)).astype(np.float32)
    bg[:, 200:] = np.nan  # the unread half of the CCD
    level = _extract._level(bg, "SJI")
    upper = np.median(level[80:100, :200])
    lower = np.median(level[100:, :200])
    assert abs(upper) < 0.3 and abs(lower) < 0.3, (upper, lower)
    assert 150 < np.median(level[:80, :200]) < 250
    assert (level[:, 200:] == 0).all()


def test_background_in_bands_matches_whole():
    rng = np.random.default_rng(3)
    stack = 100 + 3 * rng.standard_normal((30, 200, 50)).astype(np.float32)
    stack[:, :7, :] = 0
    stack[5, 100, 10] = 5000
    bg, noise = ccd_diffusion.tracks.background(stack)
    # the same, computed on the whole stack at once
    import scipy.stats as st

    finite = (stack > 0).all(0)
    whole = st.trim_mean(stack, 0.2, axis=0)
    dev = (
        np.nanmedian(np.abs((stack - whole) - np.nanmedian(stack - whole, 0)), 0)
        * 1.4826
    )
    assert np.allclose(bg[finite], whole[finite], atol=1e-3)
    assert np.allclose(noise[finite], dev[finite], atol=1e-3)
    assert np.isnan(bg[:7]).all() and np.isnan(noise[:7]).all()
    assert bg.dtype == np.float32


def test_save_tracks_none(tmp_path: pathlib.Path):
    frames = [dict(f) for f in ccd_diffusion.tracks.frames()][:3]
    ccd_diffusion.tracks.save_tracks([], tmp_path, frames=frames)
    assert ccd_diffusion.tracks.load(tmp_path) == ()
    rows = ccd_diffusion.tracks.frames(tmp_path)
    assert len(rows) == 3 and all(r["tracks"] == "0" for r in rows)


def _header(**values):
    from astropy.io import fits

    header = fits.Header()
    for k, v in values.items():
        header[k] = v
    return header


def test_off_limb_spectrograph():
    # a slit lying along the west radial: the reference row looks at the
    # limb, rows above it look farther out, rows below it look onto the disk
    header = _header(
        RSUN_OBS=951.2,
        CRPIX2=548.0,
        CRVAL2=0.0,
        CRVAL3=951.2,
        CDELT2=0.1663,
        CDELT3=0.1663,
        PC2_2=0.0,
        PC3_2=1.0,
    )
    result = ccd_diffusion.tracks.off_limb(header, (1096, 4144), "FUV")
    assert result.shape == (1096, 4144)
    assert (result == result[:, :1]).all()  # the same for every column
    # the margin of 15 arcsec is 90 rows above the reference row
    assert not result[548 + 80, 0] and result[548 + 100, 0]
    assert not result[548 - 100, 0]
    # a slit that looks at the disk centre is on the disk everywhere
    header["CRVAL3"] = 0.0
    assert not ccd_diffusion.tracks.off_limb(header, (1096, 2072), "NUV").any()


def test_off_limb_slit_jaw():
    # roll -90: the columns run along the solar north, rows toward the disk
    header = _header(
        RSUN_OBS=951.2,
        CRPIX1=504.7,
        CRPIX2=502.9,
        CRVAL1=974.4,
        CRVAL2=-0.7,
        CDELT1=0.1679,
        CDELT2=0.1679,
        PC1_1=0.0,
        PC1_2=-1.0,
        PC2_1=1.0,
        PC2_2=0.0,
    )
    result = ccd_diffusion.tracks.off_limb(header, (1096, 2072), "SJI")
    # the reference pixel looks 23 arcsec above the limb, past the margin
    assert result[502, 504]
    assert result[300, 504] and not result[800, 504]
    assert result[:, 0].sum() < result.shape[0]


def test_window_off_limb():
    # a slit along solar y through x = 954 at row 488; the window read out
    # is rows 189 to 908, and rows above about 693 and below 194 of it look
    # more than 15 arcseconds beyond a 940 arcsecond limb
    header = dict(
        RSUN_OBS="940.0",
        CRPIX2="487.92",
        CRVAL2="7.2",
        CRVAL3="954.1",
        CDELT2="0.16632",
        PC2_2="1.0",
        PC3_2="0.0",
        TSR1="189",
        TER1="908",
    )
    fraction = ccd_diffusion.tracks.window_off_limb(header)
    assert abs(fraction - 222 / 720) < 0.01
    # the same slit read out only on the disk side of the pointing
    assert ccd_diffusion.tracks.window_off_limb({**header, "TER1": "500"}) < 0.05
    # a window entirely beyond the limb
    assert ccd_diffusion.tracks.window_off_limb({**header, "CRVAL3": "1100"}) == 1.0


def test_off_limb_without_pointing():
    header = _header(EXPTIME=4.0)
    assert ccd_diffusion.tracks.off_limb(header, (10, 20), "FUV").all()


def test_blocks_split_by_camera():
    frames = [
        dict(
            dataset="x",
            fsn=str(i),
            time=f"2020-01-01T00:{i:02d}:00Z",
            image=image,
            saa="1",
            tracks="0",
        )
        for i, image in enumerate(["FUV", "NUV", "FUV", "NUV", "SJI_2796"])
    ]
    import unittest.mock

    with unittest.mock.patch.object(_extract, "frames", lambda: frames):
        blocks = ccd_diffusion.tracks.blocks("x", pathlib.Path("."))
    names = {b.name: [f["image"] for f in b.frames] for b in blocks}
    assert names == {
        "2020-01-01_FUV": ["FUV", "FUV"],
        "2020-01-01_NUV": ["NUV", "NUV"],
        "2020-01-01_2796": ["SJI_2796"],
    }
