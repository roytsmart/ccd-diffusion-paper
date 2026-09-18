import aastex

__all__ = [
    "discussion",
]


def discussion() -> aastex.Section:
    result = aastex.Section("Discussion", label="sec:discussion")
    result.escape = False
    result.append(r"""
At the back surface, where \UV\ photons are absorbed, the tracks confirm the
field-free model on the \SJI\ and \FUV{}2 \CCD{}s: the same-pixel
probability of $\sjiSamePixel \pm \sjiSamePixelError$ on \SJI\ matches the
\sjiSamePixelModel\ predicted from the depletion thickness of the model,
which is the quantity the companion noise model needs.
Deeper in the sensor the tracks see a spread of about a micron,
$\sigma_d$, that the field-free model neglects.
It is about what the drift-time diffusion across the depleted thickness
plus the micron-scale ionization column of the track itself should give,
and it is small compared with the back-surface spread, so it has little
effect on \UV\ imaging.
It cannot be left out of the fit, though: without it the wedge stretches
to absorb the floor and $t_c$ comes out too large, which is why
Equation~\ref{eq:width} carries it from the start.
With it, the fitted $t_c$ of $\sjiCriticalDepth \pm \sjiCriticalDepthError$
on \SJI\ is to be compared with the model's \modelCriticalDepth.

The \FUV{}1 \CCD\ is narrower than the other two, at the back surface and
in the fitted $\sigma_\text{max}$ alike.
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
