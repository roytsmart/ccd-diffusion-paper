import aastex
import ccd_diffusion

__all__ = [
    "method",
]


def method() -> aastex.Section:
    result = aastex.Section("Method", label="sec:method")
    result.escape = False
    result.append(r"""
\subsection{Data}

\IRIS\ \cite{DePontieu2014} carries three \CCD{}s of the same design, two
behind the spectrograph (\FUV{}1 and \FUV{}2, which share one level-1 image)
and one behind \SJI.
They are back-illuminated, \thickness\ $\mu$m thick, with \pixelPitch\
$\mu$m pixels, and their level-1 images are the raw frames with only the
dark and the readout corrections applied, so a particle hit appears exactly
as the camera recorded it.
We searched \numFrames\ such images, \numFramesSaa\ of them taken inside
\SAA, where the flux of trapped protons at the altitude of \IRIS\ is
greatest \cite{Barth2003,Adriani2015}, drawn from the five campaigns of
Table~\ref{tab:datasets}.
Figure~\ref{fig:image} shows a spectrograph image and a slit-jaw image
taken seconds apart during one pass through \SAA: the particle hits are
dense, and a handful of them are the long glancing tracks we want.
Tracks were sought against the dark background of the spectrograph frames
and, for \SJI, in the off-limb half of the field of view.""")
    result.append(ccd_diffusion.tables.datasets())
    result.append(ccd_diffusion.figures.image())
    result.append(r"""
\subsection{Finding tracks}

The background of each frame was estimated as the trimmed mean of the
frames in the same observing sequence and subtracted, and the read noise
was estimated from the median absolute deviation of the residual.
Pixels more than five times the read noise above the background were
labeled, and a straight line was fit by least squares to the charge-weighted
centroid of each row of every connected group of labeled pixels.
We kept the groups spanning at least twelve rows (or columns) with a slope
below 0.35 pixels per row, so that each track is nearly aligned with the
pixel grid and its cross-section is sampled once per row without
resampling the pixels, and extracted a seven-pixel-wide cutout centered on
the integer part of the fitted line in every row.
This yielded \numTracks\ tracks.
The slope cut is not as costly as it sounds: Figure~\ref{fig:azimuth}
shows that the trapped protons arrive strongly aligned with the slit axis,
so most of them pass it.

\subsection{Model}

Figure~\ref{fig:method} shows the principle.
A particle that enters the back surface and exits the front surface spans
the full thickness of the sensor, so the fractional depth of slice $i$ of
an $N$-slice track is $t = (i + 1/2) / N$, up to the orientation of the
track, and the charge deposited in each slice diffuses from a known depth.
We model the charge in every slice as spread uniformly along the fitted
centerline and diffused as a Gaussian with standard deviation
\begin{equation} \label{eq:width}
    \sigma(t) = \sigma_\text{max} \sqrt{1 - t / t_c}, \quad t < t_c,
\end{equation}
and zero otherwise, where $t_c = z_f / D$ is the fractional thickness of
the field-free region and $\sigma_\text{max}$ is the width of the charge
cloud at the back surface.
This is the depth dependence of a random walk to an absorbing plane at
$z_f$, the form used to model the \UV\ noise of these sensors in the
companion article, evaluated here in fractional depth because we observe
the length of a track in slices rather than the thickness of the sensor in
microns.
The uniform spread along the tilted centerline within a slice is
approximated by adding $m^2 / 12$ to the variance, where $m$ is the slope
in pixels per row.

We fit the orientation, $t_c$, and $\sigma_\text{max}$ by exhaustive
search on a grid of 21 values in each of $t_c \in [0, 1]$ and
$\sigma_\text{max} \in [0, 10]$ $\mu$m.
At each grid point the offset and tilt of the centerline are adjusted, by
up to 0.6 pixels and 0.03 pixels per row, to minimize a robust misfit,
$\sum \ln(1 + r^2 / 2)$, where $r$ is the residual of the fraction of each
slice's charge in each pixel in units of the read noise.
This is the negative log-likelihood of a Cauchy distribution rather than a
Gaussian, so that a single stray pixel, from a delta ray or a second hit
inside the cutout, cannot dominate the fit.

\subsection{Selection}

The useful tracks are those which cross the full thickness of the sensor at
nearly constant energy loss.
We kept only the tracks for which the fit constrains $t_c$ to within 0.15
and improves on a model with no diffusion by at least ten units of misfit,
and for which the median charge per slice in the last third of the track is
within 50\% of that in the first third, since a rise in the deposited
charge along the track is the Bragg peak of a particle that stopped inside
the sensor and did not cross its full thickness.
We call these the flat tracks; \numFlatTracks\ tracks pass these cuts,
\numFlatTracksSji\ of them on the \SJI\ \CCD, and Figure~\ref{fig:gallery}
shows the longest of them alongside a stopping track.""")
    result.append(ccd_diffusion.figures.azimuth())
    result.append(ccd_diffusion.figures.method())
    return result
