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

\IRIS\ \cite{DePontieu2014} carries four \CCD{}s of the same design: two
behind the \FUV\ channel of the spectrograph (\FUV{}1 and \FUV{}2, which
share one level-1 image), one behind its \NUV\ channel, and one behind
\SJI.
They are back-illuminated, \thickness\ $\mu$m thick, with \pixelPitch\
$\mu$m pixels, and their level-1 images are the raw frames with only the
dark and the readout corrections applied, so a particle hit appears exactly
as the camera recorded it.
We searched \numFrames\ such images, \numFramesSaa\ of them taken inside
\SAA, where the flux of trapped protons at the altitude of \IRIS\ is
greatest \cite{Barth2003,Adriani2015}, drawn from the campaigns of
Table~\ref{tab:datasets}.
The campaigns added for this work were chosen among those exposing for 8 s,
since a longer exposure crowds the frame inside \SAA\ with hits that spoil
the tracks around them (the campaigns of the original measurement that
pass through \SAA\ expose for 15 s, and yielded per frame half as many of
the tracks kept below), and among those whose spectrograph readout window
lies mostly beyond the limb, as the pointing recorded in each frame's
header places it.
Figure~\ref{fig:image} shows a spectrograph image and a slit-jaw image
taken seconds apart during one pass through \SAA: the particle hits are
dense, and a handful of them are the long glancing tracks we want.
Tracks were sought only where the pointing recorded in each frame's
header places the pixel at least 15 arcseconds above the photospheric
limb, so that no track lies on the disk or in the chromosphere and
spicules seen just above the limb.""")
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
centerline and diffused as a Gaussian whose variance has two parts,
\begin{equation} \label{eq:width}
    \sigma^2(t) = \sigma_\text{max}^2 \left( 1 - \frac{t}{t_c} \right)
    + \sigma_d^2 \, g(t).
\end{equation}
The first part is the field-free layer.
It applies only for $t < t_c$, where $t_c = z_f / D$ is the fractional
thickness of the layer and $\sigma_\text{max}$ is the width of the charge
cloud at the back surface, and it is the depth dependence of a random walk
to an absorbing plane at $z_f$.
The second part is the depletion region.
Charge that reaches the edge of the depletion region, whether it was
deposited there or diffused there from the field-free layer, still has to
drift to the gates and spreads a little as it does so.
$\sigma_d$ is the spread acquired crossing the full depleted thickness, and
$g(t) = \min[(1 - t) / (1 - t_c), 1]$ scales it with the distance drifted,
so that it is one for all charge deposited in the field-free layer and
falls linearly to zero at the gates.
We work in fractional depth because we observe the length of a track in
slices rather than the thickness of the sensor in microns, and the uniform
spread along the tilted centerline within a slice is approximated by
adding $m^2 / 12$ to the variance, where $m$ is the slope in pixels per
row.

The three parameters are not fit alike.
$t_c$ and $\sigma_\text{max}$ describe the field-free layer, whose
thickness can vary across a wafer, and are fit to each track together with
its orientation, by exhaustive search on a grid of 21 values in each of
$t_c \in [0, 1]$ and $\sigma_\text{max} \in [0, 10]$ $\mu$m.
At each grid point the offset and tilt of the centerline are adjusted, by
up to 0.6 pixels and 0.03 pixels per row, to minimize a robust misfit,
$\sum \ln(1 + r^2 / 2)$, where $r$ is the residual of the fraction of each
slice's charge in each pixel in units of the read noise.
This is the negative log-likelihood of a Cauchy distribution rather than a
Gaussian, so that a single stray pixel, from a delta ray or a second hit
inside the cutout, cannot dominate the fit.
$\sigma_d$ describes the drift field, which is a property of the \CCD\ and
its bias rather than of any one track, and a single track constrains it
only weakly, so it is fit to each \CCD\ as a whole.
Every track is fit as above at each of 16 values of $\sigma_d$ from 0 to 1.5
$\mu$m, the misfits of the flat tracks, defined next, are summed at each
value, and the value that minimizes the sum is adopted for every track on
that \CCD.
Which tracks are flat depends on their fits, so the selection and
$\sigma_d$ are iterated to consistency, which takes one or two rounds.

\subsection{Selection}

The useful tracks are those which cross the full thickness of the sensor at
nearly constant energy loss.
We kept only the tracks for which the fit constrains $t_c$ to within 0.15
and improves on a model with no diffusion by at least ten units of misfit,
for which the median charge per slice in the last third of the track is
within 50\% of that in the first third, since a rise in the deposited
charge along the track is the Bragg peak of a particle that stopped inside
the sensor and did not cross its full thickness, and for which the fit
finds a depleted end, $t_c \le 0.7$, since a particle that crossed the
sensor is pixel-sharp where it left through the gates, whereas a feature
that is wide from end to end is, on the slit-jaw imager, usually a
spicule or other structure at the limb that the mask let through.
We call these the flat tracks; \numFlatTracks\ tracks pass these cuts,
\numFlatTracksSji\ of them on the \SJI\ \CCD, and Figure~\ref{fig:gallery}
shows the longest of them alongside a stopping track.""")
    result.append(ccd_diffusion.figures.azimuth())
    result.append(ccd_diffusion.figures.method())
    return result
