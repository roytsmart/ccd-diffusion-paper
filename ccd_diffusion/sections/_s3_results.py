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
The second is the average of the per-track fits of Equation~\ref{eq:width}.
The two agree to within a few tenths of a micron on all three \CCD{}s at
every depth.
Beyond $t_c$ the model-free profile does not go to zero but keeps a floor
of 0.5 to 1 $\mu$m across the depletion region, and the fits follow it:
this floor is the $\sigma_d$ term of Equation~\ref{eq:width}.

\subsection{The spread inside the depletion region}

Figure~\ref{fig:depleted} shows how $\sigma_d$ is determined.
The misfit summed over the flat tracks on each \CCD\ has a sharp minimum,
at $\sigma_d = \widthDepletedSji$ $\mu$m on \SJI, \widthDepletedFuvTwo\
$\mu$m on \FUV{}2, and \widthDepletedFuvOne\ $\mu$m on \FUV{}1, with the
neighboring grid points a quarter micron away already disfavored by about
a hundred units of misfit.
The improvement over $\sigma_d = 0$ is three to four hundred units pooled
but only two or three per track, and the value each track prefers on its
own is spread over 0 to 2 $\mu$m, so the effect is modest for any one
track and consistent across them, which is why $\sigma_d$ is fit to each
\CCD\ as a whole.
The resolution of the centerline matters here: a centerline located only
to 0.1 pixel is uncertain by 1.3 $\mu$m, the size of the effect being
sought, and with such a grid the minimum washes out entirely.
$\sigma_d$ trades against the field-free wedge: as it rises from zero to
its adopted value, the median $t_c$ of the flat tracks falls by 0.05 on
every \CCD\ and the median $\sigma_\text{max}$ rises by half a micron on
\FUV{}2, since without it the wedge stretches to absorb the floor.
Leaving $\sigma_d$ out of the model would therefore bias both of the
parameters the noise model needs, which is why it is part of the model
rather than a correction applied afterwards.

\subsection{The critical depth and back-surface width}

Figure~\ref{fig:parameters} shows the fitted $t_c$ and $\sigma_\text{max}$
of every flat track, and Table~\ref{tab:datasets} their means in each
campaign.
The critical depth clusters between 0.35 and 0.50 on all three \CCD{}s,
around the \modelCriticalDepth\ of the field-free model, and the two
particle populations, the two spacecraft rolls, and the two cameras agree
to within a few hundredths.
The back-surface width does differ between chips: \FUV{}2 and \SJI\
cluster at 5.5 to 6.5 $\mu$m while \FUV{}1 clusters at 3 to 5 $\mu$m, so
the pooled scatter is mostly this chip-to-chip difference, which is
expected from the resistivity of the wafer and the applied bias and is one
reason a sensor's kernel is worth measuring rather than assuming.

\subsection{The same-pixel probability}

For the noise model the quantity that matters is not the width but the
probability that two electrons deposited at the same depth are collected
in the same pixel.
For a slice with pixel fractions $f_j$ that is $p = \sum_j f_j^2$, corrected
for the read-noise contribution $\sum_j \epsilon_j^2$, and it can be read
directly off each slice with no model of the shape of the kernel.
Since the kernel is separable, the probability that two electrons are
collected in the same pixel is $\mathcal{P} = p^2$.
Figure~\ref{fig:profile} shows $p$ against depth on each \CCD, and
Table~\ref{tab:tracks} lists it for the slices within $D / 10$ of the back
surface, alongside the prediction of the field-free model evaluated at the
fitted centerline of every track.
On the \SJI\ \CCD\ the back surface gives $\mathcal{P} = \sjiSamePixel \pm
\sjiSamePixelError$ against \sjiSamePixelModel\ from the model, and
\FUV{}2 agrees as well, while \FUV{}1 spreads its charge over fewer pixels.
Beyond $t_c$ the measured probability settles at about 0.85 rather than
the 0.95 of a perfectly sharp cloud on all three \CCD{}s, which the
per-track fits reproduce through $\sigma_d$ and the field-free model does
not.""")
    result.append(ccd_diffusion.figures.stacked())
    result.append(ccd_diffusion.figures.depleted())
    result.append(ccd_diffusion.figures.parameters())
    result.append(ccd_diffusion.figures.profile())
    result.append(ccd_diffusion.tables.tracks())
    return result
