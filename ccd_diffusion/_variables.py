import astropy.units as u
import aastex
import ccd_diffusion

__all__ = [
    "variables",
]


def variables() -> list[aastex.Variable]:
    """
    The numeric values quoted in the prose, computed from the tracks so that
    the text cannot drift from the measurement.
    """
    tracks = ccd_diffusion.tracks
    frames = tracks.frames()
    fits = tracks.fits()
    sji = tracks.summary("SJI")
    tc_paper, sm_paper = tracks.paper_model()
    ccd = ccd_diffusion.ccd()
    images = {im.image: im for im in tracks.images()}

    return [
        aastex.Variable("numFrames", len(frames)),
        aastex.Variable("numFramesSaa", sum(f["saa"] == "1" for f in frames)),
        aastex.Variable("numTracks", len(fits)),
        aastex.Variable("numFlatTracks", sum(f.flat for f in fits)),
        aastex.Variable("numFlatTracksSji", sji.num_flat),
        aastex.Variable("sjiSamePixel", f"{sji.same_pixel:.2f}"),
        aastex.Variable("sjiSamePixelError", f"{sji.same_pixel_error:.2f}"),
        aastex.Variable("sjiSamePixelModel", f"{sji.same_pixel_paper:.2f}"),
        aastex.Variable("sjiCriticalDepth", f"{sji.critical_depth[1]:.2f}"),
        aastex.Variable("sjiWidthMax", f"{sji.width_max[1].to_value(u.um):.1f}"),
        aastex.Variable("modelCriticalDepth", f"{tc_paper:.2f}"),
        aastex.Variable("modelWidthMax", f"{sm_paper.to_value(u.um):.2f}"),
        aastex.Variable(
            "widthDepleted",
            f"{tracks.depleted('SJI').best.to_value(u.um):.1f}",
        ),
        aastex.Variable("imageFsnSji", images["SJI_2796"].fsn),
        aastex.Variable("imageFsnFuv", images["FUV"].fsn),
        aastex.Variable("imageVmax", ccd_diffusion.figures._image._vmax),
        aastex.Variable("thickness", f"{ccd.thickness_substrate.to_value(u.um):.0f}"),
        aastex.Variable("pixelPitch", f"{tracks.width_pixel.to_value(u.um):.0f}"),
    ]
