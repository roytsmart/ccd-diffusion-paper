import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import ccd_diffusion
from . import _fit, _depleted


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
        # the finder asks four slices in five to carry charge; the rest may not
        assert np.mean(track.signal > 0) >= 0.8
        assert np.all(track.signal[track.usable] > 0)
        assert track.usable.shape == track.position.shape
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


def test_width_depleted():
    depth = na.linspace(0, 1, axis="t", num=11)
    pixel = ccd_diffusion.tracks.width_pixel
    result = ccd_diffusion.tracks.width(depth, 0.4, 5 * u.um, 1 * u.um)
    assert result.shape == depth.shape
    # the two spreads add in quadrature at the back surface
    assert np.isclose(result[dict(t=0)], np.sqrt(26) * u.um / pixel)
    # the field-free layer keeps the full depletion spread
    assert np.isclose(result[dict(t=3)], np.sqrt(25 * 0.25 + 1) * u.um / pixel)
    # which then falls linearly to zero at the gates
    assert result[dict(t=5)] > result[dict(t=8)] > 0
    assert result[dict(t=-1)] == 0
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


def test_scan_synthetic():
    track = _synthetic(0.4, 5 * u.um)
    result = ccd_diffusion.tracks.scan(track)
    assert isinstance(result, ccd_diffusion.tracks.Scan)
    grid = ccd_diffusion.tracks.width_depleted
    assert result.misfit.shape == grid.shape
    # the track was made without any depletion spread, so it prefers none
    assert result.preferred == 0 * u.um
    assert np.all(np.diff(result.misfit, axis=result.misfit.axes[0]) >= 0)


def test_scan_matches_loss():
    # the misfit assembled from the table agrees with the direct misfit of
    # the best fit, on a real track whose read noise sets the tolerance
    track = ccd_diffusion.tracks.flat("SJI")[0].track
    result = ccd_diffusion.tracks.scan(track)
    for sd in (0 * u.um, 1 * u.um):
        best = result.at(sd)
        direct = ccd_diffusion.tracks.loss(track, best.position, best.width)
        assert float(direct.ndarray) == pytest.approx(
            float(result.misfit[dict(width_depleted=int(sd.value * 4))].ndarray),
            abs=0.5,
        )


def test_fit_synthetic():
    critical_depth = 0.4
    width_max = 5 * u.um
    track = _synthetic(critical_depth, width_max)
    result = ccd_diffusion.tracks.fit(track)
    assert isinstance(result, ccd_diffusion.tracks.Fit)
    assert result.width_depleted == 0 * u.um
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
    for track in tracks:
        stored = fits[track.name]
        result = ccd_diffusion.tracks.fit(track, stored.width_depleted)
        assert result.width_depleted == stored.width_depleted
        assert result.orientation == stored.orientation
        assert result.critical_depth == pytest.approx(stored.critical_depth, abs=1e-3)
        assert u.isclose(result.width_max, stored.width_max, atol=0.01 * u.um)
        assert result.gain == pytest.approx(stored.gain, abs=1e-2)


def test_save(monkeypatch, tmp_path):
    stored = ccd_diffusion.tracks.fits()
    chips = ("FUV1", "FUV2", "SJI")
    pooled = tuple(ccd_diffusion.tracks.depleted(chip) for chip in chips)
    monkeypatch.setattr(_fit, "_path_fits", tmp_path / "fits.csv")
    monkeypatch.setattr(_depleted, "_path_depleted", tmp_path / "depleted.csv")
    ccd_diffusion.tracks.save(stored, pooled)
    _fit.fits.cache_clear()
    _depleted.depleted.cache_clear()
    try:
        reloaded = ccd_diffusion.tracks.fits()
        reloaded_pooled = tuple(ccd_diffusion.tracks.depleted(chip) for chip in chips)
    finally:
        _fit.fits.cache_clear()
        _depleted.depleted.cache_clear()
    assert len(reloaded) == len(stored)
    for a, b in zip(reloaded, stored):
        assert a.track is b.track
        assert a.width_depleted == b.width_depleted
        assert a.critical_depth == b.critical_depth
        assert a.width_max == b.width_max
        assert a.width_depleted_preferred == b.width_depleted_preferred
    for a, b in zip(reloaded_pooled, pooled):
        assert a.chip == b.chip
        assert a.num == b.num
        assert np.allclose(a.misfit, b.misfit)
        assert a.best == b.best


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
    # a guard on the article's claim rather than a proof of it: with thousands
    # of tracks the statistical error is far below the model's own uncertainty
    result = ccd_diffusion.tracks.summary("SJI")
    assert result.same_pixel == pytest.approx(result.same_pixel_paper, abs=0.03)


def test_flat_requires_a_depleted_end():
    fits = ccd_diffusion.tracks.fits()
    wide = [f for f in fits if f.tight and f.bragg < 1.5 and not f.crossing]
    assert wide, "the data hold features the fit calls wide from end to end"
    assert not any(f.flat for f in wide)
    assert all(
        f.critical_depth <= ccd_diffusion.tracks.critical_depth_maximum
        for f in fits
        if f.flat
    )


def test_images():
    images = ccd_diffusion.tracks.images()
    assert {im.image for im in images} == {"FUV", "SJI_2796"}
    for im in images:
        assert im.saa
        assert im.data.shape[ccd_diffusion.tracks.axis_row] == 1096
        assert len(im.tracks) > 5
        for track in im.tracks:
            assert track.fsn == im.fsn


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "SJI"])
def test_stack(chip: str):
    result = ccd_diffusion.tracks.stack(chip)
    assert isinstance(result, ccd_diffusion.tracks.Stack)
    assert result.image.shape == {
        ccd_diffusion.tracks.axis_depth: result.depth.size - 1,
        "offset": result.offset.size - 1,
    }
    assert np.all(np.isfinite(result.image))
    # the image is a density per pixel, so it integrates to one over the cutout
    width = np.diff(result.offset, axis="offset")
    assert np.allclose((result.image * width).sum("offset"), 1, atol=0.05)


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "SJI"])
def test_widths(chip: str):
    result = ccd_diffusion.tracks.widths(chip)
    assert isinstance(result, ccd_diffusion.tracks.Widths)
    assert np.all(result.lower <= result.best)
    assert np.all(result.best <= result.upper)
    assert result.fitted.shape == result.depth.shape
    assert result.model.shape == result.depth.shape
    # the back surface is wider than the front
    assert result.best[dict(depth=0)] > result.best[dict(depth=-1)]


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "SJI"])
def test_depleted(chip: str):
    result = ccd_diffusion.tracks.depleted(chip)
    assert isinstance(result, ccd_diffusion.tracks.Depleted)
    assert result.misfit.min() == 0
    assert 0 * u.um < result.best < 3 * u.um
    assert result.num == len(ccd_diffusion.tracks.flat(chip))
    # every flat track on the chip was fit at the pooled value
    for f in ccd_diffusion.tracks.flat(chip):
        assert f.width_depleted == result.best


def test_pooled():
    tracks = [f.track for f in ccd_diffusion.tracks.flat("SJI")][:12]
    scans = [ccd_diffusion.tracks.scan(t) for t in tracks]
    result = ccd_diffusion.tracks.pooled("SJI", scans)
    assert result.chip == "SJI"
    assert 0 < result.num <= len(tracks)
    assert result.misfit.shape == ccd_diffusion.tracks.width_depleted.shape
    assert result.critical_depth.shape == result.misfit.shape


def test_charge_minimum_mirrors_the_finder():
    from . import _extract, _tracks

    assert _tracks.charge_minimum == _extract.charge_minimum
