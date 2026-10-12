import concurrent.futures
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import ccd_diffusion
from . import _fit, _depleted


def test_load():
    tracks = ccd_diffusion.tracks.load()
    assert len(tracks) > 1000
    assert len({t.name for t in tracks}) == len(tracks)
    for track in tracks:
        assert isinstance(track, ccd_diffusion.tracks.Track)
        assert track.chip in ("FUV1", "FUV2", "NUV", "SJI")
        assert track.length >= 12
        assert track.charge.shape == {
            ccd_diffusion.tracks.axis_slice: track.length,
            ccd_diffusion.tracks.axis_pixel: 2 * ccd_diffusion.tracks.half_width + 1,
        }
        assert track.position.shape == {ccd_diffusion.tracks.axis_slice: track.length}
        assert np.all(track.signal[track.usable] > 0)
        assert track.usable.shape == track.position.shape
        assert np.allclose(track.fraction.sum(ccd_diffusion.tracks.axis_pixel), 1)
        assert np.all((track.depth > 0) & (track.depth < 1))
    # the finder asks four slices in five to carry charge on the pixels it
    # labeled; the seven-pixel cutout can still sum below zero where the
    # background was over-subtracted beside a bright neighbor, so all but a
    # track in a thousand carry charge in four slices of five
    positive = [np.mean(t.signal > 0).ndarray >= 0.8 for t in tracks]
    assert np.mean(positive) > 0.999


def test_frames():
    frames = ccd_diffusion.tracks.frames()
    assert len(frames) > 1000
    for frame in frames:
        assert frame["dataset"]
        assert frame["image"].startswith(("FUV", "NUV", "SJI"))
        assert frame["saa"] in ("0", "1")


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_width(model: str):
    depth = na.linspace(0, 1, axis="t", num=11)
    result = ccd_diffusion.tracks.width(depth, 0.4, 5 * u.um, model=model)
    assert result.shape == depth.shape
    assert result.unit is None
    assert result[dict(t=0)] == 5 * u.um / ccd_diffusion.tracks.width_pixel
    assert np.all(result[depth >= 0.4] == 0)
    assert np.all(np.diff(result, axis="t") <= 0)


@pytest.mark.parametrize(
    "model,wedge",
    [("slab", 1 - 0.75**2), ("gaussian", 1 - 0.75)],
)
def test_width_depleted(model: str, wedge: float):
    depth = na.linspace(0, 1, axis="t", num=11)
    pixel = ccd_diffusion.tracks.width_pixel
    result = ccd_diffusion.tracks.width(depth, 0.4, 5 * u.um, 1 * u.um, model)
    assert result.shape == depth.shape
    # the two spreads add in quadrature at the back surface
    assert np.isclose(result[dict(t=0)], np.sqrt(26) * u.um / pixel)
    # the field-free layer keeps the full depletion spread
    assert np.isclose(result[dict(t=3)], np.sqrt(25 * wedge + 1) * u.um / pixel)
    # which then falls linearly to zero at the gates
    assert result[dict(t=5)] > result[dict(t=8)] > 0
    assert result[dict(t=-1)] == 0
    assert np.all(np.diff(result, axis="t") <= 0)


def test_width_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        ccd_diffusion.tracks.width(0.1, 0.4, 5 * u.um, model="lorentzian")


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_cloud(model: str):
    tracks = ccd_diffusion.tracks
    depth = na.linspace(0, 1, axis="t", num=21)
    args = (depth, 0.4, 5 * u.um, 0.8 * u.um)
    width, weight = tracks.cloud(*args, model=model)
    assert np.allclose(weight.sum(tracks.axis_mixture), 1)
    # the variance of the mixture is that of the model, except within a
    # tenth of the field-free layer of the depletion region, where the slab's
    # mixture misses the rare electrons that wander back toward the surface
    variance = (weight * np.square(width)).sum(tracks.axis_mixture)
    expected = np.square(tracks.width(*args, model=model))
    away = (depth < 0.36) | (depth >= 0.4)
    assert np.allclose(variance[away], expected[away], rtol=0.02)


@pytest.mark.parametrize("depth", [0.05, 0.2, 0.35, 0.39, 0.6])
def test_cloud_slab_is_optika(depth: float):
    # the slab model with a free sigma_max is optika's model of a sensor
    # sigma_max / t_c thick, so its pixel fractions are differences of
    # optika's profile at the pixel edges
    tracks = ccd_diffusion.tracks
    tc = 0.4
    sm = 5 * u.um
    sd = 0.8 * u.um
    thickness = sm / tc
    model = optika.sensors.diffusion.SlabDiffusionModel(
        thickness_depletion=(1 - tc) * thickness,
        width_depletion=sd,
    )
    position = 0.17
    t = na.ScalarArray(np.array([depth]), axes=tracks.axis_slice)
    width, weight = tracks.cloud(t, tc, sm, sd)
    result = tracks.fractions(position, width, 0, weight, tracks.axis_mixture)
    edges = (tracks._fit._edges - position) * tracks.width_pixel
    cdf = model.cdf(edges, depth * thickness, thickness)
    expected = np.diff(cdf, axis=tracks.axis_pixel)
    expected = expected / expected.sum(tracks.axis_pixel)
    assert np.allclose(result, expected, atol=1e-6)


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
    critical_depth: float,
    width_max: u.Quantity,
    model: str = "slab",
) -> ccd_diffusion.tracks.Track:
    tracks = ccd_diffusion.tracks
    length = 24
    signal = 20000
    noise = 5
    axis_slice = tracks.axis_slice
    depth = (na.arange(0, length, axis=axis_slice) + 0.5) / length
    position = 0.2 + 0 * depth
    width, weight = tracks.cloud(depth, critical_depth, width_max, model=model)
    charge = signal * tracks.fractions(position, width, 0, weight, tracks.axis_mixture)
    return tracks.Track(
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


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_scan_synthetic(model: str):
    track = _synthetic(0.4, 5 * u.um, model)
    result = ccd_diffusion.tracks.scan(track, model)
    assert isinstance(result, ccd_diffusion.tracks.Scan)
    assert result.model == model
    grid = ccd_diffusion.tracks.width_depleted
    assert result.misfit.shape == grid.shape
    # the track was made without any depletion spread, so it prefers none,
    # and the misfit climbs away from it; the other parameters are refit on
    # their own grids at every step, which lets it wobble by a few units
    assert result.preferred == 0 * u.um
    misfit = result.misfit.ndarray
    assert np.all(np.diff(misfit[::5]) > 0)
    assert misfit[-1] > 5 * misfit[0]


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_scan_matches_loss(model: str):
    # the misfit of the search agrees with the direct misfit of the best
    # fit, on a real track whose read noise sets the tolerance
    tracks = ccd_diffusion.tracks
    track = tracks.flat("SJI")[0].track
    result = tracks.scan(track, model)
    grid = tracks.width_depleted.ndarray
    for sd in (0 * u.um, 1 * u.um):
        best = result.at(sd)
        width, weight = tracks.cloud(
            best.depth, best.critical_depth, best.width_max, sd, model
        )
        direct = tracks.loss(track, best.position, width, weight, tracks.axis_mixture)
        i = int(np.argmin(np.abs(grid - sd)))
        assert float(direct.ndarray) == pytest.approx(
            float(result.misfit[dict(width_depleted=i)].ndarray), abs=0.5
        )
        assert best.misfit == pytest.approx(float(direct.ndarray), abs=0.5)


def test_scan_slab_prunes_exactly():
    # abandoning centerlines leaves the least misfit, and every misfit
    # within two of it, as they are when every centerline is tried in full
    tracks = ccd_diffusion.tracks
    track = min(tracks.flat("FUV2"), key=lambda f: f.track.length).track
    misfit = _fit._misfit_gaussian(track)
    center = np.argmin(misfit, axis=(tracks.axis_offset, tracks.axis_tilt))
    center = {a: center[a] for a in (tracks.axis_offset, tracks.axis_tilt)}
    pruned, _ = _fit._misfit_slab(track, center, prune=True)
    full, _ = _fit._misfit_slab(track, center, prune=False)
    pruned, full = pruned.ndarray, full.ndarray
    least = full.min(axis=(0, 1, 2), keepdims=True)
    near = full < least + _fit._interval
    near[0, 0, 0, 0] = True
    assert np.array_equal(pruned[near], full[near])
    assert np.all(pruned >= full)


def test_scan_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        ccd_diffusion.tracks.scan(_synthetic(0.4, 5 * u.um), "lorentzian")


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_fit_synthetic(model: str):
    critical_depth = 0.4
    width_max = 5 * u.um
    track = _synthetic(critical_depth, width_max, model)
    result = ccd_diffusion.tracks.fit(track, model=model)
    assert isinstance(result, ccd_diffusion.tracks.Fit)
    assert result.model == model
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


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_fits(model: str):
    fits = ccd_diffusion.tracks.fits(model)
    tracks = ccd_diffusion.tracks.load()
    assert len(fits) == len(tracks)
    assert [f.track.name for f in fits] == [t.name for t in tracks]
    assert all(f.model == model for f in fits)
    assert sum(f.flat for f in fits) > 300


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_fit_matches_stored(model: str):
    fits = {f.track.name: f for f in ccd_diffusion.tracks.fits(model)}
    tracks = sorted(ccd_diffusion.tracks.load(), key=lambda t: t.length)[:3]
    for track in tracks:
        stored = fits[track.name]
        result = ccd_diffusion.tracks.fit(track, stored.width_depleted, model)
        assert result.width_depleted == stored.width_depleted
        assert result.orientation == stored.orientation
        assert result.critical_depth == pytest.approx(stored.critical_depth, abs=1e-3)
        assert u.isclose(result.width_max, stored.width_max, atol=0.01 * u.um)
        assert result.misfit == pytest.approx(stored.misfit, abs=1e-2)
        assert result.gain == pytest.approx(stored.gain, abs=1e-2)


@pytest.mark.parametrize("model", ["slab", "gaussian"])
def test_save(monkeypatch, tmp_path, model: str):
    stored = ccd_diffusion.tracks.fits(model)
    chips = ("FUV1", "FUV2", "SJI")
    pooled = tuple(ccd_diffusion.tracks.depleted(chip, model) for chip in chips)
    monkeypatch.setitem(_fit._path_fits, model, tmp_path / "fits.csv")
    monkeypatch.setitem(_depleted._path_depleted, model, tmp_path / "depleted.csv")
    ccd_diffusion.tracks.save(stored, pooled, model)
    _fit.fits.cache_clear()
    _depleted.depleted.cache_clear()
    try:
        reloaded = ccd_diffusion.tracks.fits(model)
        reloaded_pooled = tuple(
            ccd_diffusion.tracks.depleted(chip, model) for chip in chips
        )
    finally:
        _fit.fits.cache_clear()
        _depleted.depleted.cache_clear()
    assert len(reloaded) == len(stored)
    for a, b in zip(reloaded, stored):
        assert a.track is b.track
        assert a.model == b.model
        assert a.width_depleted == b.width_depleted
        assert a.critical_depth == b.critical_depth
        assert a.width_max == b.width_max
        assert a.misfit == b.misfit
        assert a.width_depleted_preferred == b.width_depleted_preferred
    for a, b in zip(reloaded_pooled, pooled):
        assert a.chip == b.chip
        assert a.num == b.num
        assert np.allclose(a.misfit, b.misfit)
        assert a.best == b.best


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "NUV", "SJI"])
def test_profile(chip: str):
    result = ccd_diffusion.tracks.profile(chip)
    assert isinstance(result, ccd_diffusion.tracks.Profile)
    assert result.chip == chip
    for array in (
        result.measured,
        result.error,
        result.mean,
        result.error_mean,
        result.fitted,
        result.none,
    ):
        assert array.shape == result.depth.shape
        assert np.all(np.isfinite(array))
    assert np.all(result.num > 0)
    assert np.all((result.measured > 0) & (result.measured < 1))
    assert np.all((result.mean > 0) & (result.mean < 1))
    assert np.all(result.none >= result.fitted)
    # stray charge from other hits can only lower the mean, and the median
    # is unmoved by it, so beyond the field-free layer the median must sit
    # above the mean
    deep = result.depth.ndarray > 0.6
    assert np.all(result.measured.ndarray[deep] > result.mean.ndarray[deep])


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "NUV", "SJI"])
def test_summary(chip: str):
    result = ccd_diffusion.tracks.summary(chip)
    assert isinstance(result, ccd_diffusion.tracks.Summary)
    assert result.num_flat < result.num_tracks
    assert 0 < result.same_pixel < 1
    assert 0 < result.same_pixel_error < 0.1
    assert (
        result.critical_depth[0] <= result.critical_depth[1] <= result.critical_depth[2]
    )
    assert result.width_max[0] <= result.width_max[1] <= result.width_max[2]


def test_same_pixel_ratio():
    # a Gaussian cloud is separable, and the slab's, a mixture of separable
    # Gaussians, puts two electrons in the same pixel more often than p^2
    tracks = ccd_diffusion.tracks
    args = (0.4, 5 * u.um, 0.8 * u.um)
    assert tracks.same_pixel_ratio(*args, model="gaussian") == pytest.approx(1)
    slab = tracks.same_pixel_ratio(*args, model="slab")
    assert 1 < slab < 1.1
    # deep in the depletion region the cloud is a single Gaussian again
    assert tracks.same_pixel_ratio(*args, depth=0.9) == pytest.approx(1)


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


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "NUV", "SJI"])
def test_kernel(chip: str):
    result = ccd_diffusion.tracks.kernel(chip)
    assert isinstance(result, ccd_diffusion.tracks.Kernel)
    shape = {
        ccd_diffusion.tracks.axis_depth: result.depth.size,
        "distance": result.distance.size - 1,
    }
    for array in (result.measured, result.error, result.model):
        assert array.shape == shape
        assert np.all(np.isfinite(array))
    # the fractions are a density per pixel, so they integrate to one
    width = np.diff(result.distance, axis="distance")
    for array in (result.measured, result.model):
        assert np.allclose((array * width).sum("distance"), 1, atol=0.01)
    # the peak is lower at the back surface than at the gates
    peak = result.model.max("distance")
    assert peak[dict(depth=0)] < peak[dict(depth=-1)]


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "NUV", "SJI"])
def test_widths(chip: str):
    result = ccd_diffusion.tracks.widths(chip)
    assert isinstance(result, ccd_diffusion.tracks.Widths)
    assert np.all(result.lower <= result.best)
    assert np.all(result.best <= result.upper)
    assert result.fitted.shape == result.depth.shape
    # the back surface is wider than the front
    assert result.best[dict(depth=0)] > result.best[dict(depth=-1)]


@pytest.mark.parametrize("chip", ["FUV1", "FUV2", "NUV", "SJI"])
def test_depleted(chip: str):
    result = ccd_diffusion.tracks.depleted(chip)
    assert isinstance(result, ccd_diffusion.tracks.Depleted)
    assert result.misfit.min() == 0
    assert 0 * u.um < result.best < 3 * u.um
    assert result.num == len(ccd_diffusion.tracks.flat(chip))
    for array in (
        result.critical_depth,
        result.width_max,
        result.critical_depth_mean,
        result.critical_depth_error,
        result.width_max_mean,
        result.width_max_error,
    ):
        assert array.shape == result.misfit.shape
        assert np.all(np.isfinite(array))
    assert np.all(result.critical_depth_error > 0)
    assert np.all(result.width_max_error > 0 * u.um)
    # the mean of a thousand tracks sits within a grid step of the median
    assert np.all(abs(result.critical_depth_mean - result.critical_depth) < 0.05)
    assert np.all(abs(result.width_max_mean - result.width_max) < 0.5 * u.um)
    # every flat track on the chip was fit at the pooled value
    for f in ccd_diffusion.tracks.flat(chip):
        assert f.width_depleted == result.best


def test_pooled():
    tracks = [f.track for f in ccd_diffusion.tracks.flat("SJI")][:12]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        scans = list(pool.map(ccd_diffusion.tracks.scan, tracks))
    result = ccd_diffusion.tracks.pooled("SJI", scans)
    assert result.chip == "SJI"
    assert 0 < result.num <= len(tracks)
    assert result.critical_depth_mean.shape == result.misfit.shape
    assert result.width_max_mean.shape == result.misfit.shape
    assert result.misfit.shape == ccd_diffusion.tracks.width_depleted.shape
    assert result.critical_depth.shape == result.misfit.shape


def test_charge_minimum_mirrors_the_finder():
    from . import _extract, _tracks

    assert _tracks.charge_minimum == _extract.charge_minimum
