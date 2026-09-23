import aastex

__all__ = [
    "discussion",
]


def discussion() -> aastex.Section:
    result = aastex.Section("Discussion", label="sec:discussion")
    result.escape = False
    result.append(r"""
At the back surface, where \UV\ photons are absorbed, the tracks give the
kernel directly: on \SJI\ the field-free layer is \sjiCriticalDepth\ of
the thickness, the cloud is \sjiWidthMax\ $\mu$m wide, and the same-pixel
probability is $\sjiSamePixel \pm \sjiSamePixelError$, the quantity that
enters the variance of a \UV\ image.
Deeper in the sensor the tracks see a spread of under a micron,
$\sigma_d$, that a purely field-free kernel would neglect.
It is about what the drift-time diffusion across the depleted thickness
plus the micron-scale ionization column of the track itself should give,
and it is small compared with the back-surface spread, so it has little
effect on \UV\ imaging.
It cannot be left out of the fit, though: without it the wedge stretches
to absorb the floor and $t_c$ comes out too large, which is why
Equation~\ref{eq:width} carries it from the start.

The \FUV{}1 \CCD\ is narrower than the other three, at the back surface
and in the fitted $\sigma_\text{max}$ alike, and \NUV\ sits between it and
the \FUV{}2 and \SJI\ pair.
We have no explanation beyond chip-to-chip variation of the field-free
thickness, which depends on the resistivity of the wafer and the applied
bias; the same measurement on \AIA\ \CCD{}s, which are of the same
design, would show whether such variation is common.

Two systematics deserve comment.
First, the tracks are left by protons that deposit thousands of electrons
per row, whereas a \UV\ photon liberates one to three.
In thick, fully depleted sensors, slow protons leave tracks visibly
widened by the mutual repulsion of that charge \cite{Grosson2023}, and if
the same occurred in the field-free layer here the tracks would
overestimate the single-photon kernel.
The tracks themselves can test this, by comparing the kernels of faint and
bright tracks, and that comparison is the first thing to add.
Second, the depth assignment assumes every flat track crosses the full
thickness of the sensor; a track that entered or left through the edge of
the field-free layer, or that was clipped by the Bragg cut at one end,
would have its depths compressed.
The agreement between the model-free and parametric profiles, and between
campaigns, suggests such tracks are rare, but a validation on synthetic
tracks injected into real frames is the way to bound it.""")
    return result
