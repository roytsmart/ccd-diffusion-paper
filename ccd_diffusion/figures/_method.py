import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import aastex
import ccd_diffusion

__all__ = [
    "method",
]

"""The track shown as the example."""

_length_schematic = 20
"""The length of the schematic track in pixels."""


def _unwrapped(fit: "ccd_diffusion.tracks.Fit") -> tuple[np.ndarray, np.ndarray]:
    """
    A track's cutout in the coordinates of its parent image, relative to the
    first slice, and the fitted centerline in the same coordinates.

    The cutout re-centers every slice on the integer part of the centerline,
    so this undoes that shift, inferring it from the wrapped fractional
    positions and the slope.
    """
    tracks = ccd_diffusion.tracks
    track = fit.track
    h = tracks.half_width
    charge = track.charge
    position = fit.position
    if fit.orientation < 0:
        charge = charge[{tracks.axis_slice: slice(None, None, -1)}]
        position = position[{tracks.axis_slice: slice(None, None, -1)}]
    slope = track.slope if fit.orientation > 0 else -track.slope
    p = track.position.ndarray[:: fit.orientation]
    shift = np.concatenate([[0], np.cumsum(np.round(slope - np.diff(p)))]).astype(int)
    shift = shift - shift.min()
    image = np.zeros((track.length, 2 * h + 1 + shift.max()))
    for i, k in enumerate(shift):
        image[i, k : k + 2 * h + 1] = charge[{tracks.axis_slice: i}].ndarray
    return image, position.ndarray + shift


def method() -> aastex.Figure:
    """
    How a glancing track measures the diffusion width against depth, and the
    diffusion signal in one real track.
    """
    tracks = ccd_diffusion.tracks
    # the schematic is drawn at the median fit of the FUV2 tracks
    typical = tracks.summary("FUV2")
    tc_typical = typical.critical_depth[1]
    sm_typical = typical.width_max[1]
    D = tracks.thickness.to_value(u.um)
    zf = tc_typical * D
    h = tracks.half_width
    L = _length_schematic

    # the longest flat track on the FUV2 CCD in the roll -90 campaign
    example = max(
        (
            f
            for f in tracks.fits()
            if f.flat and f.track.chip == "FUV2" and f.track.dataset == "2018may"
        ),
        key=lambda f: f.track.length,
    )
    track = example.track

    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6.5, 5),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = ax.flat

    # (a) depth as a function of distance along a schematic glancing track
    x = na.linspace(0, L, axis="x", num=200)
    ax_a.axhspan(0, zf, color="0.85", label="field-free")
    ax_a.axhspan(zf, D, color="tab:blue", alpha=0.15, label="depleted")
    na.plt.plot(x, D * x / L, ax=ax_a, color="black", linewidth=1.5, label="particle")
    sample = na.linspace(0.5, L - 0.5, axis="sample", num=9)
    sigma = tracks.width(sample / L, tc_typical, sm_typical)
    ax_a.errorbar(
        sample.ndarray,
        (D * sample / L).ndarray,
        xerr=sigma.ndarray,
        fmt="o",
        color="tab:red",
        markersize=2,
        capsize=2,
        label=r"$\pm\sigma(z)$, typical fit",
    )
    ax_a.axhline(zf, linestyle="--", color="tab:blue", linewidth=0.8)
    ax_a.text(L, zf, "$z_f$", ha="right", va="bottom", fontsize=8)
    ax_a.set_xlim(0, L)
    ax_a.set_ylim(0, D)
    ax_a.set_xlabel("distance along track (pixels)")
    ax_a.set_ylabel(r"depth $z$ ($\mu$m)")
    ax_a.set_title("(a) depth is a function of position", fontsize=8)
    ax_a.legend(loc="lower right", fontsize=6)

    # (b) the same track seen from above
    sigma = tracks.width(x / L, tc_typical, sm_typical)
    for k in range(-h, h + 1):
        ax_b.axhline(k + 0.5, color="0.85", linewidth=0.5)
    ax_b.fill_between(
        x.ndarray, -2 * sigma.ndarray, 2 * sigma.ndarray, color="tab:red", alpha=0.2
    )
    ax_b.fill_between(
        x.ndarray,
        -sigma.ndarray,
        sigma.ndarray,
        color="tab:red",
        alpha=0.4,
        label=r"$\pm\sigma(z)$, typical fit",
    )
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_xlim(0, L)
    ax_b.set_ylim(-h - 0.5, h + 0.5)
    ax_b.text(0.02, 0.05, "back surface, wide", transform=ax_b.transAxes, fontsize=7)
    ax_b.text(
        0.98, 0.05, "front, sharp", transform=ax_b.transAxes, ha="right", fontsize=7
    )
    ax_b.set_xlabel("distance along track (pixels)")
    ax_b.set_ylabel("transverse (pixels)")
    ax_b.set_title("(b) seen from above", fontsize=8)
    ax_b.legend(loc="upper right", fontsize=6)

    # (c) the example track
    image, centerline = _unwrapped(example)
    mappable = ax_c.imshow(
        image.T / 1000,
        cmap="magma",
        origin="lower",
        extent=[0, track.length, -h - 0.5, image.shape[1] - h - 0.5],
        aspect="auto",
        interpolation="none",
    )
    fig.colorbar(mappable, ax=ax_c, label="thousands of electrons")
    ax_c.plot(
        np.arange(track.length) + 0.5,
        centerline,
        color="white",
        linestyle="--",
        linewidth=0.8,
    )
    ax_c.set_xlabel("position along track (pixels)")
    ax_c.set_ylabel("transverse (pixels)")
    ax_c.set_title(
        f"(c) track {track.name} ({track.chip}), {track.length} slices, "
        f"{track.signal.mean().ndarray:.0f} e$^-$ per slice",
        fontsize=8,
    )

    # (d) the same-column probability of every slice of the example track
    t = example.depth.ndarray
    order = np.argsort(t)
    none = tracks.same_pixel_model(example, 0, 0 * u.um).ndarray
    best = tracks.same_pixel_model(
        example, example.critical_depth, example.width_max, example.width_depleted
    ).ndarray
    measured = tracks.same_pixel(example).ndarray
    ax_d.plot(t[order], none[order], color="gray", linestyle="--", label="no diffusion")
    ax_d.plot(
        t[order],
        best[order],
        color="tab:blue",
        label=rf"fit: $t_c={example.critical_depth:.2f}$, "
        rf"$\sigma_\mathrm{{max}}={example.width_max.to_value(u.um):.1f}$, "
        rf"$\sigma_d={example.width_depleted.to_value(u.um):.2f}$ $\mu$m",
    )
    ax_d.plot(t, measured, "o", color="black", markersize=2.5, label="measured")
    ax_d.axvline(example.critical_depth, color="tab:blue", linestyle=":", linewidth=0.8)
    ax_d.set_ylim(0.2, 1.02)
    ax_d.set_xlabel("fractional depth, $t = z / D$")
    ax_d.set_ylabel(r"same-column probability, $\sum_j f_j^2$")
    ax_d.set_title("(d) the diffusion signal along the track in (c)", fontsize=8)
    ax_d.legend(fontsize=6, loc="lower left")

    result = aastex.Figure("method", position="htb!")
    result.add_fig(fig, width=None)
    result.add_caption(aastex.NoEscape(r"""
How a glancing track measures the diffusion width against depth.
(a) A particle entering at the illuminated back surface ($z = 0$) and leaving
at the gates ($z = D$) crosses the field-free layer first, where the charge
it liberates diffuses until it reaches the depletion edge, so the lateral
spread (red bars, to scale in pixels, at the median fit of the \FUV{}2
tracks) is largest at the back surface and vanishes at $z = z_f$.
(b) Seen from above, the track is a wedge: about 0.4 pixels wide at one end
and pixel-sharp beyond $t_c$.
(c) A proton track on the \FUV{}2 \CCD, in the coordinates of its parent
image, with the fitted centerline dashed.
(d) The probability that two electrons deposited in the same slice of that
track are collected in the same column, $\sum_j f_j^2$ corrected for read
noise, for every slice, against the best fit and a model with no
diffusion, both evaluated at the fitted centerline.
The dip near $t = 0.2$ is where the centerline crosses a pixel boundary; the
diffusion signal is the gap between the measured points and the
no-diffusion curve."""))
    return result
