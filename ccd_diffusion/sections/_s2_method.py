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
shows that the trapped protons arrive strongly aligned with one axis of the
detector or the other, which one depending on the roll of the spacecraft,
so that more of them pass it than the \acceptedIsotropic\% an isotropic
population would, \acceptedRollZero\% at the roll of most of the campaigns.

\subsection{Model}

Figure~\ref{fig:method} shows the principle.
A particle that enters the back surface and exits the front surface spans
the full thickness of the sensor, so the fractional depth of slice $i$ of
an $N$-slice track is $t = (i + 1/2) / N$, up to the orientation of the
track, and the charge deposited in each slice diffuses from a known depth.
We model the charge in every slice as spread uniformly along the fitted
centerline and diffused by a charge cloud with two parts.

The first part is the field-free layer, which the charge of a slice
crosses only if $t < t_c$, where $t_c = z_f / D$ is the fractional
thickness of the layer.
Charge created there diffuses until it reaches the edge of the depletion
region, which collects it, while the back surface reflects it
\cite{Pavlov1999}.
In a layer $z_f$ thick, the fraction of the charge created a fraction
$t / t_c$ of the way across it that arrives at an offset less than $y$
along one axis is
\begin{equation} \label{eq:slab}
    F(y) = \frac{1}{2} + \frac{1}{\pi} \arctan \left[
        \frac{\sinh (\pi y / 2 z_f)}{\cos (\pi t / 2 t_c)}
    \right],
\end{equation}
whose variance falls from $z_f^2$ at the back surface to zero at the
depletion region as $z_f^2 [1 - (t / t_c)^2]$.
The cloud is not a Gaussian: it falls off exponentially, as
$e^{-\pi |y| / 2 z_f}$, and near the depletion region, where most of the
charge is collected at once but some wanders back toward the back surface
first, it has a narrow core and broad wings.
Equation~\ref{eq:slab} ties the width of the cloud at the back surface to
the thickness of the layer, $\sigma_\text{max} = z_f = t_c D$.
We leave $\sigma_\text{max}$ free instead, replacing $z_f$ by it in
Equation~\ref{eq:slab}, so that the model has the same parameters as the
Gaussian model below, and the ratio $\sigma_\text{max} / t_c$ measures the
thickness of the sensor the fit implies.
The second part is the depletion region.
Charge that reaches the edge of the depletion region, whether it was
deposited there or diffused there from the field-free layer, still has to
drift to the gates and spreads a little as it does so, which we model as a
Gaussian of variance $\sigma_d^2 \, g(t)$.
$\sigma_d$ is the spread acquired crossing the full depleted thickness, and
$g(t) = \min[(1 - t) / (1 - t_c), 1]$ scales it with the distance drifted,
so that it is one for all charge deposited in the field-free layer and
falls linearly to zero at the gates.
The variance of the cloud is then
\begin{equation} \label{eq:width}
    \sigma^2(t) = \sigma_\text{max}^2 \left[ 1 - \left( \frac{t}{t_c}
    \right)^2 \right] + \sigma_d^2 \, g(t),
\end{equation}
where the first term applies only for $t < t_c$.
Equation~\ref{eq:slab} convolved with the Gaussian has no closed form, so
we evaluate the cloud as the mixture of eight Gaussians of the
\texttt{optika} package \cite{optika}, whose variances scale as
$\sigma_\text{max}^2$ and to each of which we add $\sigma_d^2 \, g(t)$;
at a single depth it reproduces Equation~\ref{eq:slab} to about a part in
a thousand of the charge.
We work in fractional depth because we observe the length of a track in
slices rather than the thickness of the sensor in microns, and the uniform
spread along the tilted centerline within a slice is approximated by
adding $m^2 / 12$ to the variance of every Gaussian, where $m$ is the slope
in pixels per row.

To see what the shape of the cloud is worth, we also fit every track with
a Gaussian whose variance is
\begin{equation} \label{eq:gaussian}
    \sigma^2(t) = \sigma_\text{max}^2 \left( 1 - \frac{t}{t_c} \right)
    + \sigma_d^2 \, g(t),
\end{equation}
the depth dependence of the model of Janesick \cite{Janesick2001}, a random
walk to an absorbing plane at $z_f$ with no reflecting back surface, with
the same parameters searched on the same grids.

The three parameters are not fit alike.
$t_c$ and $\sigma_\text{max}$ describe the field-free layer, whose
thickness can vary across a wafer, and are fit to each track together with
its orientation, by exhaustive search on a grid of 21 values in each of
$t_c \in [0, 1]$ and $\sigma_\text{max} \in [0, 10]$ $\mu$m.
At each grid point the offset and tilt of the centerline are adjusted, by
up to 0.6 pixels and 0.03 pixels per row, to minimize a robust misfit,
\begin{equation} \label{eq:misfit}
    M = \sum_i \sum_j \ln \left( 1 + \frac{r_{ij}^2}{2} \right),
    \qquad
    r_{ij} = \frac{f_{ij} - q_{ij}}{\epsilon_i},
\end{equation}
where $f_{ij}$ is the fraction of the charge of slice $i$ measured in pixel
$j$ of the cutout, $q_{ij}$ the fraction the model puts there, and
$\epsilon_i$ the read noise of a pixel divided by the charge of the slice,
the uncertainty of a fraction from read noise alone, so that the brighter
slices count for more.
This is the negative log-likelihood of a Cauchy distribution rather than a
Gaussian, so that a single stray pixel, from a delta ray or a second hit
inside the cutout, cannot dominate the fit, and differences in $M$ are
what the thresholds below are stated in.
$\sigma_d$ describes the drift field, which is a property of the \CCD\ and
its bias rather than of any one track, and a single track constrains it
only weakly, so it is fit to each \CCD\ as a whole.
Every track is fit as above at each of 16 values of $\sigma_d$ from 0 to 1.5
$\mu$m, the misfits $M$ of the flat tracks, defined next, are summed at each
value, and the value that minimizes the sum is adopted for every track on
that \CCD.
Which tracks are flat depends on their fits, so the selection and
$\sigma_d$ are iterated to consistency, which takes one or two rounds.

\subsection{Selection}

The useful tracks are those which cross the full thickness of the sensor at
nearly constant energy loss.
We kept only the tracks for which the fit constrains $t_c$ to within 0.15
and improves on a model with no diffusion by at least ten in $M$,
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
