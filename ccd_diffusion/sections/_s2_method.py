import aastex
import ccd_diffusion

__all__ = [
    "method",
]


def method() -> aastex.Section:
    result = aastex.Section("Method", label="sec:method")
    result.escape = False
    result.append(r"""
\subsection{Tracks}

We searched \numFrames\ level-1 images from the \IRIS\ spectrograph and
\SJI, \numFramesSaa\ of them taken inside \SAA, for straight streaks of
charge against the dark background of off-limb or unexposed regions.
The background of each frame was estimated as the trimmed mean of the frames
in the same observing sequence, the read noise from the median absolute
deviation of the residual, and pixels more than five times the read noise
above the background were labeled.
Connected groups of labeled pixels were fit with a straight line through the
charge-weighted centroid of each row, and we kept the groups spanning at
least twelve rows with a slope below 0.35 pixels per row, so that each track
is nearly aligned with the pixel grid and its cross-section is sampled once
per row.
A seven-pixel-wide cutout centered on the fitted line was extracted for each
track, which yielded \numTracks\ tracks.

\subsection{Model}

We model the charge in every slice of a track as spread uniformly along the
fitted centerline and diffused as a Gaussian with standard deviation
\begin{equation} \label{eq:width}
    \sigma(t) = \sigma_\text{max} \sqrt{1 - t / t_c}, \quad t < t_c,
\end{equation}
and zero otherwise, where $t = z / D$ is the fractional depth of the slice
below the back surface, $t_c = z_f / D$ is the fractional thickness of the
field-free region, and $\sigma_\text{max}$ is the width of the charge cloud
at the back surface.
A track that enters the back surface and exits the front surface spans the
full thickness of the sensor, so the fractional depth of slice $i$ of an
$N$-slice track is $t = (i + 1/2) / N$, up to the orientation of the track,
which we fit along with $t_c$ and $\sigma_\text{max}$ by exhaustive search
on a grid of 21 values in each of $t_c \in [0, 1]$ and
$\sigma_\text{max} \in [0, 10]$ $\mu$m.
At each grid point the offset and tilt of the centerline are adjusted, by up
to 0.6 pixels and 0.03 pixels per row, to minimize a robust misfit,
$\sum \ln(1 + r^2 / 2)$, where $r$ is the residual of the fraction of each
slice's charge in each pixel in units of the read noise.

\subsection{Selection}

The useful tracks are those which cross the full thickness of the sensor at
nearly constant energy loss.
We kept only the tracks for which the fit constrains $t_c$ to within 0.15
and improves on a model with no diffusion by at least ten units of misfit,
and for which the median charge per slice in the last third of the track is
within 50\% of that in the first third, since a rise in the deposited
charge along the track marks a particle that stopped inside the sensor.
\numFlatTracks\ tracks pass these cuts, \numFlatTracksSji\ of them on the
\SJI\ \CCD.""")
    result.append(ccd_diffusion.figures.tracks())
    return result
