import aastex
import ccd_diffusion

__all__ = [
    "results",
]


def results() -> aastex.Section:
    result = aastex.Section("Results", label="sec:results")
    result.escape = False
    result.append(r"""
\subsection{The width against depth}

Figure~\ref{fig:stacked} shows the flat tracks stacked on their fitted
centerlines and the diffusion width they imply at each depth, measured two
ways.
The first assumes no depth dependence at all: for each depth bin, the
misfits of every slice in the bin are summed over a grid of trial widths
and the width which minimizes the sum is taken, so the wedge shape emerges
from the data rather than being imposed.
The second applies the same procedure to the fractions the per-track fits
of the slab model predict for the same slices, so that the fits are
measured the way the slices are, although their cloud is not a Gaussian.
The two agree to within \widthResidualMax\ $\mu$m on all four \CCD{}s at
every depth but the bins just past $t_c$, where the measured cloud is up
to \widthResidualPeak\ $\mu$m wider than the fits: it narrows less abruptly
at the edge of the depletion region than either model has it.
Beyond $t_c$ the model-free profile does not go to zero but keeps a floor
across the depletion region, under a micron just past $t_c$ and a quarter
micron near the gates, and the fits follow it: this floor is the
$\sigma_d$ term of Equation~\ref{eq:width}.
The third panel shows the kernel itself on \FUV{}2 at four depths,
integrated over a pixel: the fits follow its core as it narrows from the
back surface to the gates, while the stray charge discussed with
Figure~\ref{fig:profile} lowers the measured peak by a few hundredths and
raises the wings.

\subsection{The shape of the charge cloud}

Table~\ref{tab:models} and Figure~\ref{fig:models} compare the slab model
with the Gaussian of Equation~\ref{eq:gaussian}, each fit to every track
and its $\sigma_d$ pooled over each \CCD\ in the same way;
\numFlatTracks\ tracks are flat under the slab model and
\numFlatTracksGaussian\ under the Gaussian.
On the \numComparedTracks\ tracks flat under both, the slab model lowers
the summed misfit by \misfitGainSlab, \misfitGainPerTrack\ per track, and
it does so on every \CCD, fitting \percentBetterSlab\% of the tracks
better by more than two and \percentWorseSlab\% worse.
The gain is in the field-free layer, between the back surface and the
critical depth, where the two clouds differ (Figure~\ref{fig:models}b);
beyond $t_c$, where both are the same Gaussian, the slab model is a little
worse, by about a tenth per track in each depth bin, since the centerline
and the $t_c$ it chooses serve the field-free layer.
The gain grows with the charge per slice, which pins down the shape of the
cloud, and at the same charge it is the same for the 8 and 15 s exposures
and for the protons and the cosmic rays, so it is not the stray charge of
Section~\ref{sec:samepix}, which grows with the exposure and is absent from
the frames of the cosmic rays.
It is the shape of the cloud that the tracks prefer rather than the way
its width falls with depth: a Gaussian whose variance falls as
Equation~\ref{eq:width} has it fits the same tracks worse than
Equation~\ref{eq:gaussian}, by \misfitQuadraticLoss.
Tying $\sigma_\text{max}$ to the thickness of the field-free layer,
$t_c D$, as the slab model has it for a sensor \thickness\ $\mu$m thick,
costs \misfitTiedLoss, more than the shape gains, for reasons the next
sections take up.
The model-free widths summarize each depth bin by a single Gaussian and
do not separate the models as sharply: under both, the residuals of
Table~\ref{tab:models} are a few tenths of a micron, but at the back
surface the measured width falls below the Gaussian fits on every \CCD, by
up to \residualBackGaussian\ $\mu$m, and within \residualBackSlab\ $\mu$m
of the fits of the slab model, whose variance stops changing with depth
where the back surface reflects the charge.

\subsection{The spread inside the depletion region} \label{sec:depleted}

Figure~\ref{fig:depleted} shows how $\sigma_d$ is determined.
The misfit $M$ of Equation~\ref{eq:misfit} summed over the flat tracks on
each \CCD\ has a sharp minimum, at $\sigma_d = \widthDepletedFuvOne$,
\widthDepletedFuvTwo, \widthDepletedNuv, and \widthDepletedSji\ $\mu$m on
\FUV{}1, \FUV{}2, \NUV, and \SJI, with the neighboring grid points a tenth
of a micron away disfavored by \depletedNeighborLow\ to
\depletedNeighborHigh\ units of misfit.
The improvement over $\sigma_d = 0$ is \depletedGainLow\ to
\depletedGainHigh\ units pooled but \depletedGainTrackLow\ to
\depletedGainTrackHigh\ per track, and the value each track prefers on its
own is spread over the whole grid, from 0 to 1.5 $\mu$m, so the effect is
modest for any one track and consistent across them, which is why
$\sigma_d$ is fit to each \CCD\ as a whole.
The resolution of the centerline matters here: a centerline located only
to 0.1 pixel is uncertain by 1.3 $\mu$m, the size of the effect being
sought, and with such a grid the minimum washes out entirely.
$\sigma_d$ trades against the field-free wedge: as it rises from zero to
its adopted value, the mean $t_c$ of the flat tracks falls by
\criticalShiftLow\ to \criticalShiftHigh, most on \FUV{}1, since without it
the wedge stretches to absorb the floor, while the mean
$\sigma_\text{max}$ rises by \widthMaxShiftHigh\% at most.
Leaving $\sigma_d$ out of the model would therefore bias the critical
depth, which is why it is part of the model rather than a correction
applied afterwards.

\subsection{The critical depth and back-surface width}

Figure~\ref{fig:parameters} shows the fitted $t_c$ and $\sigma_\text{max}$
of every flat track, and Table~\ref{tab:datasets} their means in each
campaign.
Half of the flat tracks have a critical depth between
\criticalQuartileLow\ and \criticalQuartileHigh.
Its mean over the core tracks is the same to within
\criticalSpreadGroups\ for the two particle populations, the four
spacecraft rolls, and the 8 and 15 s exposures, and every campaign with
more than twenty core tracks lies between \criticalCampaignLow\ and
\criticalCampaignHigh; the four \CCD{}s differ from one another by up to
\criticalSpreadChips, \FUV{}1 lowest and \FUV{}2 highest.
The back-surface width differs more between chips, its median from
\widthMaxFuvOne\ $\mu$m on \FUV{}1 and \widthMaxNuv\ $\mu$m on \NUV\ to
\widthMaxFuvTwo\ and \widthMaxSji\ $\mu$m on \FUV{}2 and \SJI, so the pooled
scatter is mostly this chip-to-chip difference, which is expected from the
resistivity of the wafer and the applied bias and is one reason a sensor's
kernel is worth measuring rather than assuming.

The ratio of the two measures the thickness of the sensor, since the
cloud at a reflecting back surface is as wide as the field-free layer is
thick, $\sigma_\text{max} = z_f = t_c D$.
The mean $\sigma_\text{max}$ of the flat tracks divided by their mean $t_c$
is \thicknessFuvOne, \thicknessFuvTwo, \thicknessNuv, and \thicknessSji\
$\mu$m on \FUV{}1, \FUV{}2, \NUV, and \SJI, each to about
\thicknessError\ $\mu$m, against the \thickness\ $\mu$m of the design.

\subsection{The same-pixel probability} \label{sec:samepix}

For the noise statistics of a \UV\ image the quantity that matters is not
the width but the probability that two electrons deposited at the same
depth are collected in the same pixel.
For a slice with pixel fractions $f_j$ that is $p = \sum_j f_j^2$, corrected
for the read-noise contribution $\sum_j \epsilon_j^2$, and it can be read
directly off each slice with no model of the shape of the kernel.
For a separable kernel the probability that two electrons are collected in
the same pixel would be $p^2$, but the cloud of the slab model is not
separable, since an electron that takes long to reach the depletion region
spreads along both axes at once.
For the median fit of each \CCD\ at the back surface the slab model puts
the same-pixel probability \separabilityLow\ to \separabilityHigh\% above
$p^2$, and we take $\mathcal{P}$ to be $p^2$ raised by that factor.
Figure~\ref{fig:profile} shows $p$ against depth on each \CCD, and
Table~\ref{tab:tracks} lists it for the slices within $D / 10$ of the back
surface.
On the \SJI\ \CCD\ the back surface gives $\mathcal{P} = \sjiSamePixel \pm
\sjiSamePixelError$, \FUV{}2 the same to within the errors, while \NUV\
and, further still, \FUV{}1 spread their charge over fewer pixels, in the
same order as their back-surface widths.

The figure gives $p$ two ways in each depth bin, as the plain mean over
the slices and as the median of how far each slice falls short of its
own sharp-cloud value.
The two agree at the back surface and part beyond $t_c$, where the mean
settles near 0.88 on all four \CCD{}s while the median stays within 0.02
of the 0.95 of a perfectly sharp cloud, with the per-track fits between
them.
The difference is charge that does not belong to the track.
A frame taken inside the anomaly is crowded with hits, and one that
touches a track is joined to it by the finder and cut out with it.
Such charge lands only off the peak of a slice, so it can only lower
$p$, and it pulls the mean down in proportion to its charge while the
median is unmoved until it reaches half the slices.
Two checks say this is what the mean is seeing: it falls twice as far
below the sharp-cloud value in campaigns exposed for 15 s as in those
exposed for 8 s, at the same charge per slice, and it does not fall at
all for the few tracks of cosmic rays recorded outside the anomaly,
where the frames are empty.
At the back surface the cloud already spans several columns and the same
stray charge moves $p$ by under 0.02, so Table~\ref{tab:tracks} keeps the
mean, which is the quantity that enters the variance of an image.
Beyond $t_c$ the median is the measurement, and it leaves the depletion
region a few hundredths short of sharp, in the direction and about the
size of the $\sigma_d$ the fits carry.""")
    result.append(ccd_diffusion.figures.stacked())
    result.append(ccd_diffusion.figures.models())
    result.append(ccd_diffusion.tables.models())
    result.append(ccd_diffusion.figures.depleted())
    result.append(ccd_diffusion.figures.parameters())
    result.append(ccd_diffusion.figures.profile())
    result.append(ccd_diffusion.tables.tracks())
    return result
