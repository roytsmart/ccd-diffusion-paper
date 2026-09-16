import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ccd_diffusion
from . import _fit


def test_load():
    tracks = ccd_diffusion.tracks.load()
    assert len(tracks) > 1000
    assert len({t.name for t in tracks}) == len(tracks)
    for track in tracks:
        assert isinstance(track, ccd_diffusion.tracks.Track)
        assert track.chip in ("FUV1", "FUV2", "SJI")
        assert track.length >= 12
        assert track.charge.shape == {
            ccd_diffusion.tracks.axis_slice: track.length,
            ccd_diffusion.tracks.axis_pixel: 2 * ccd_diffusion.tracks.half_width + 1,
        }
        assert track.position.shape == {ccd_diffusion.tracks.axis_slice: track.length}
        assert np.all(track.signal > 0)
        assert np.allclose(track.fraction.sum(ccd_diffusion.tracks.axis_pixel), 1)
        assert np.all((track.depth > 0) & (track.depth < 1))


def test_frames():
    frames = ccd_diffusion.tracks.frames()
    assert len(frames) > 1000
    for frame in frames:
        assert frame["dataset"]
        assert frame["image"].startswith(("FUV", "SJI"))
        assert frame["saa"] in ("0", "1")


def test_width():
    depth = na.linspace(0, 1, axis="t", num=11)
    result = ccd_diffusion.tracks.width(depth, 0.4, 5 * u.um)
    assert result.shape == depth.shape
    assert result.unit is None
    assert result[dict(t=0)] == 5 * u.um / ccd_diffusion.tracks.width_pixel
    assert np.all(result[depth >= 0.4] == 0)
    assert np.all(np.diff(result, axis="t") <= 0)


@pytest.mark.parametrize("slope", [0, 0.3])
def test_fractions(slope: float):
    position = 0.1
    width = na.linspace(0, 1, axis="slice", num=5)
    result = ccd_diffusion.tracks.fractions(position, width, slope)
    assert np.allclose(result.sum(ccd_diffusion.tracks.axis_pixel), 1)
    assert np.all(result >= 0)
    center = result[{ccd_diffusion.tracks.axis_pixel: ccd_diffusion.tracks.half_width}]
    assert np.all(np.diff(center, axis="slice") <= 0)


def _synthetic(
    critical_depth: float, width_max: u.Quantity
) -> ccd_diffusion.tracks.Track:
    length = 24
    signal = 20000
    noise = 5
    axis_slice = ccd_diffusion.tracks.axis_slice
    depth = (na.arange(0, length, axis=axis_slice) + 0.5) / length
    position = 0.2 + 0 * depth
    width = ccd_diffusion.tracks.width(depth, critical_depth, width_max)
    charge = signal * ccd_diffusion.tracks.fractions(position, width, 0)
    return ccd_diffusion.tracks.Track(
        name="synthetic",
        dataset="synthetic",
        chip="SJI",
        fsn=0,
        slope=0,
        noise=noise,
        gain=1,
        vertical=True,
        row=0,
        column=0,
        charge=charge,
        position=0 * depth,
    )


def test_fit_synthetic():
    critical_depth = 0.4
    width_max = 5 * u.um
    track = _synthetic(critical_depth, width_max)
    result = ccd_diffusion.tracks.fit(track)
    assert isinstance(result, ccd_diffusion.tracks.Fit)
    assert result.orientation == 1
    assert result.critical_depth == pytest.approx(critical_depth, abs=0.05)
    assert u.isclose(result.width_max, width_max, atol=0.5 * u.um)
    assert result.offset == pytest.approx(0.2, abs=0.05)
    assert result.gain > 10
    assert result.tight
    assert result.flat
    assert (
        result.critical_depth_min <= result.critical_depth <= result.critical_depth_max
    )
    assert result.width_max_min <= result.width_max <= result.width_max_max


def test_fits():
    fits = ccd_diffusion.tracks.fits()
    tracks = ccd_diffusion.tracks.load()
    assert len(fits) == len(tracks)
    assert [f.track.name for f in fits] == [t.name for t in tracks]
    assert sum(f.flat for f in fits) > 300


def test_fit_matches_stored():
    fits = {f.track.name: f for f in ccd_diffusion.tracks.fits()}
    tracks = sorted(ccd_diffusion.tracks.load(), key=lambda t: t.length)[:3]
    results = ccd_diffusion.tracks.fit_all(tuple(tracks))
    for track, result in zip(tracks, results):
        stored = fits[track.name]
        assert result.orientation == stored.orientation
        assert result.critical_depth == pytest.approx(stored.critical_depth, abs=1e-3)
        assert u.isclose(result.width_max, stored.width_max, atol=0.01 * u.um)
        assert result.gain == pytest.approx(stored.gain, abs=1e-2)


def test_save(monkeypatch, tmp_path):
    path = tmp_path / "fits.csv"
    monkeypatch.setattr(_fit, "_path_fits", path)
    stored = ccd_diffusion.tracks.fits()
    ccd_diffusion.tracks.save(stored)
    _fit.fits.cache_clear()
    try:
        reloaded = ccd_diffusion.tracks.fits()
    finally:
        _fit.fits.cache_clear()
    assert len(reloaded) == len(stored)
    for a, b in zip(reloaded, stored):
        assert a.track is b.track
        assert a.critical_depth == b.critical_depth
        assert a.width_max == b.width_max


def test_paper_model():
    critical_depth, width_max = ccd_diffusion.tracks.paper_model()
    assert 0 < critical_depth < 1
    assert 0 * u.um < width_max < ccd_diffusion.ccd().thickness_substrate


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "SJI"])
def test_profile(chip: str):
    result = ccd_diffusion.tracks.profile(chip)
    assert isinstance(result, ccd_diffusion.tracks.Profile)
    assert result.chip == chip
    for array in (
        result.measured,
        result.error,
        result.paper,
        result.fitted,
        result.none,
    ):
        assert array.shape == result.depth.shape
        assert np.all(np.isfinite(array))
    assert np.all(result.num > 0)
    assert np.all((result.measured > 0) & (result.measured < 1))
    assert np.all(result.none >= result.paper)


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "SJI"])
def test_summary(chip: str):
    result = ccd_diffusion.tracks.summary(chip)
    assert isinstance(result, ccd_diffusion.tracks.Summary)
    assert result.num_flat < result.num_tracks
    assert 0 < result.same_pixel < 1
    assert 0 < result.same_pixel_error < 0.1
    assert 0 < result.same_pixel_paper < 1
    assert (
        result.critical_depth[0] <= result.critical_depth[1] <= result.critical_depth[2]
    )
    assert result.width_max[0] <= result.width_max[1] <= result.width_max[2]


def test_summary_sji_matches_model():
    result = ccd_diffusion.tracks.summary("SJI")
    assert result.same_pixel == pytest.approx(
        result.same_pixel_paper,
        abs=3 * result.same_pixel_error,
    )
