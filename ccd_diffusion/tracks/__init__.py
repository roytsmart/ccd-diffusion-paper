"""
Measure the charge diffusion kernel of the IRIS CCDs from glancing
particle tracks.

The tracks are stored in ``data/iris_tracks.npz`` and ``data/iris_tracks.csv``,
and the result of fitting the diffusion model to every track is stored in
``data/iris_fits.csv``.
Fitting every track takes several minutes, so the fits are only recomputed
by running ``python -m ccd_diffusion.tracks``.
"""

from ._tracks import (
    axis_slice,
    axis_pixel,
    half_width,
    width_pixel,
    Track,
    load,
    frames,
)
from ._fit import (
    axis_critical_depth,
    axis_width_max,
    axis_offset,
    axis_tilt,
    critical_depth,
    width_max,
    offset,
    tilt,
    width,
    fractions,
    loss,
    Fit,
    fit,
    fit_all,
    fits,
    save,
)
from ._samepix import (
    axis_depth,
    depth_bins,
    depth_back,
    same_pixel,
    same_pixel_model,
    paper_model,
    Profile,
    profile,
    Summary,
    summary,
)

__all__ = [
    "axis_slice",
    "axis_pixel",
    "half_width",
    "width_pixel",
    "Track",
    "load",
    "frames",
    "axis_critical_depth",
    "axis_width_max",
    "axis_offset",
    "axis_tilt",
    "critical_depth",
    "width_max",
    "offset",
    "tilt",
    "width",
    "fractions",
    "loss",
    "Fit",
    "fit",
    "fit_all",
    "fits",
    "save",
    "axis_depth",
    "depth_bins",
    "depth_back",
    "same_pixel",
    "same_pixel_model",
    "paper_model",
    "Profile",
    "profile",
    "Summary",
    "summary",
]
