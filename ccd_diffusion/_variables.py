import collections
import numpy as np
import astropy.units as u
import aastex
import ccd_diffusion

__all__ = [
    "variables",
]

_chips = ["FUV1", "FUV2", "NUV", "SJI"]

_names = dict(FUV1="FuvOne", FUV2="FuvTwo", NUV="Nuv", SJI="Sji")
"""The suffix of the name of a macro for each CCD."""

_depth_critical = (0.3, 0.4)
"""The depths of the bins of the model-free profile just past the critical depth."""


def _depleted() -> dict[str, float]:
    """
    How sharply the pooled misfit of each CCD picks its :math:`\\sigma_d`, and
    how the mean fit moves from :math:`\\sigma_d = 0` to it.
    """
    tracks = ccd_diffusion.tracks
    neighbor, gain, gain_track, shift_tc, shift_sm = [], [], [], [], []
    for chip in _chips:
        d = tracks.depleted(chip)
        misfit = d.misfit.ndarray
        i = int(np.argmin(misfit))
        neighbor += [misfit[j] for j in (i - 1, i + 1) if 0 <= j < misfit.size]
        gain.append(misfit[0])
        gain_track.append(misfit[0] / d.num)
        tc = d.critical_depth_mean.ndarray
        sm = d.width_max_mean.ndarray.to_value(u.um)
        shift_tc.append(tc[0] - tc[i])
        shift_sm.append(100 * (sm[i] / sm[0] - 1))
    return dict(
        neighbor_low=min(neighbor),
        neighbor_high=max(neighbor),
        gain_low=min(gain),
        gain_high=max(gain),
        gain_track_low=min(gain_track),
        gain_track_high=max(gain_track),
        shift_tc_low=min(shift_tc),
        shift_tc_high=max(shift_tc),
        shift_sm_high=max(shift_sm),
    )


def _spread_groups() -> float:
    """
    The largest difference in the mean :math:`t_c` of the core tracks between
    the two particle populations, the spacecraft rolls, or the 8 and 15 s
    exposures.
    """
    core = ccd_diffusion.figures._parameters.core()
    datasets = ccd_diffusion.tracks.datasets
    result = 0.0
    groupings = (("particles", None), ("roll", None), ("exposure", (8, 15)))
    for key, values in groupings:
        groups = collections.defaultdict(list)
        for f in core:
            groups[datasets[f.track.dataset][key]].append(f.critical_depth)
        means = [np.mean(v) for k, v in groups.items() if values is None or k in values]
        result = max(result, max(means) - min(means))
    return result


def _campaigns() -> tuple[float, float]:
    """The least and greatest mean :math:`t_c` of the campaigns with more than twenty core tracks."""
    groups = collections.defaultdict(list)
    for f in ccd_diffusion.figures._parameters.core():
        groups[f.track.dataset].append(f.critical_depth)
    means = [np.mean(v) for v in groups.values() if len(v) > 20]
    return min(means), max(means)


def _residual() -> tuple[float, float]:
    """
    The largest difference between the model-free width and that of the
    fits of the slab model, at any depth but just past the critical depth,
    and the largest just past it.
    """
    tracks = ccd_diffusion.tracks
    away, near = [], []
    for chip in _chips:
        c = tracks.comparison(chip)
        r = c.residual["slab"].ndarray.to_value(u.um)
        t = c.depth.ndarray
        critical = (t > _depth_critical[0]) & (t < _depth_critical[1])
        away.append(np.abs(r[~critical]).max())
        near.append(r[critical].max())
    return max(away), max(near)


def variables() -> list[aastex.Variable]:
    """
    The numeric values quoted in the prose, computed from the tracks so that
    the text cannot drift from the measurement.
    """
    tracks = ccd_diffusion.tracks
    frames = tracks.frames()
    fits = tracks.fits()
    sji = tracks.summary("SJI")
    summaries = {chip: tracks.summary(chip) for chip in _chips}
    comparisons = [tracks.comparison(chip) for chip in _chips]
    num_compared = sum(c.num for c in comparisons)
    misfit = {
        model: sum(c.misfit[model] for c in comparisons) for model in tracks.models
    }
    misfit_gain = -misfit["slab"]
    back = {
        model: [c.residual[model].ndarray[0].to_value(u.um) for c in comparisons]
        for model in ("slab", "gaussian")
    }
    depleted = _depleted()
    campaign_low, campaign_high = _campaigns()
    core_means = [
        np.mean(
            [
                f.critical_depth
                for f in ccd_diffusion.figures._parameters.core()
                if f.track.chip == chip
            ]
        )
        for chip in _chips
    ]
    tc_flat = [f.critical_depth for f in fits if f.flat]
    residual_away, residual_near = _residual()

    result = [
        aastex.Variable("numFrames", len(frames)),
        aastex.Variable("numFramesSaa", sum(f["saa"] == "1" for f in frames)),
        aastex.Variable("numTracks", len(fits)),
        aastex.Variable("numFlatTracks", sum(f.flat for f in fits)),
        aastex.Variable("numFlatTracksSji", sji.num_flat),
        aastex.Variable(
            "numFlatTracksGaussian",
            sum(f.flat for f in tracks.fits("gaussian")),
        ),
        aastex.Variable("sjiSamePixel", f"{sji.same_pixel:.3f}"),
        aastex.Variable("sjiSamePixelError", f"{sji.same_pixel_error:.3f}"),
        aastex.Variable("sjiCriticalDepth", f"{sji.critical_depth[1]:.2f}"),
        aastex.Variable("sjiWidthMax", f"{sji.width_max[1].to_value(u.um):.1f}"),
        aastex.Variable("sjiCriticalDepthError", f"{sji.critical_depth_error:.3f}"),
        aastex.Variable("numComparedTracks", num_compared),
        aastex.Variable("misfitGainSlab", f"{misfit_gain:,.0f}"),
        aastex.Variable("misfitGainPerTrack", f"{misfit_gain / num_compared:.1f}"),
        aastex.Variable(
            "percentBetterSlab",
            f"{100 * sum(c.better for c in comparisons) / num_compared:.0f}",
        ),
        aastex.Variable(
            "percentWorseSlab",
            f"{100 * sum(c.worse for c in comparisons) / num_compared:.0f}",
        ),
        aastex.Variable("misfitQuadraticLoss", f"{misfit['gaussian_quadratic']:,.0f}"),
        aastex.Variable(
            "misfitTiedLoss", f"{misfit['slab_tied'] - misfit['slab']:,.0f}"
        ),
        aastex.Variable("residualBackGaussian", f"{-min(back['gaussian']):.2f}"),
        aastex.Variable("residualBackSlab", f"{max(abs(r) for r in back['slab']):.2f}"),
        aastex.Variable(
            "separabilityLow",
            f"{100 * (min(s.same_pixel_ratio for s in summaries.values()) - 1):.1f}",
        ),
        aastex.Variable(
            "separabilityHigh",
            f"{100 * (max(s.same_pixel_ratio for s in summaries.values()) - 1):.1f}",
        ),
        aastex.Variable("widthResidualMax", f"{residual_away:.2f}"),
        aastex.Variable("widthResidualPeak", f"{residual_near:.2f}"),
        aastex.Variable(
            "thicknessError",
            f"{max(s.thickness_error for s in summaries.values()).to_value(u.um):.1f}",
        ),
        aastex.Variable("depletedNeighborLow", f"{depleted['neighbor_low']:,.0f}"),
        aastex.Variable("depletedNeighborHigh", f"{depleted['neighbor_high']:,.0f}"),
        aastex.Variable("depletedGainLow", f"{depleted['gain_low']:,.0f}"),
        aastex.Variable("depletedGainHigh", f"{depleted['gain_high']:,.0f}"),
        aastex.Variable("depletedGainTrackLow", f"{depleted['gain_track_low']:.1f}"),
        aastex.Variable("depletedGainTrackHigh", f"{depleted['gain_track_high']:.1f}"),
        aastex.Variable("criticalShiftLow", f"{depleted['shift_tc_low']:.2f}"),
        aastex.Variable("criticalShiftHigh", f"{depleted['shift_tc_high']:.2f}"),
        aastex.Variable("widthMaxShiftHigh", f"{depleted['shift_sm_high']:.1f}"),
        aastex.Variable("criticalQuartileLow", f"{np.percentile(tc_flat, 25):.2f}"),
        aastex.Variable("criticalQuartileHigh", f"{np.percentile(tc_flat, 75):.2f}"),
        aastex.Variable("criticalSpreadGroups", f"{_spread_groups():.2f}"),
        aastex.Variable("criticalCampaignLow", f"{campaign_low:.2f}"),
        aastex.Variable("criticalCampaignHigh", f"{campaign_high:.2f}"),
        aastex.Variable(
            "criticalSpreadChips", f"{max(core_means) - min(core_means):.2f}"
        ),
        aastex.Variable("imageVmax", ccd_diffusion.figures._image._vmax),
        aastex.Variable(
            "acceptedIsotropic",
            f"{100 * ccd_diffusion.figures._azimuth._accepted_isotropic():.0f}",
        ),
        aastex.Variable(
            "acceptedRollZero",
            f"{100 * ccd_diffusion.figures._azimuth._accepted(0):.0f}",
        ),
        aastex.Variable("thickness", f"{tracks.thickness.to_value(u.um):.0f}"),
        aastex.Variable("pixelPitch", f"{tracks.width_pixel.to_value(u.um):.0f}"),
    ]
    for chip, s in summaries.items():
        name = _names[chip]
        result += [
            aastex.Variable(
                f"widthDepleted{name}",
                f"{tracks.depleted(chip).best.to_value(u.um):.2f}",
            ),
            aastex.Variable(
                f"widthDepleted{name}Gaussian",
                f"{tracks.depleted(chip, 'gaussian').best.to_value(u.um):.2f}",
            ),
            aastex.Variable(f"widthMax{name}", f"{s.width_max[1].to_value(u.um):.1f}"),
            aastex.Variable(f"thickness{name}", f"{s.thickness.to_value(u.um):.1f}"),
        ]
    return result
