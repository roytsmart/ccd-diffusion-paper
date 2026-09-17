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
The two agree to within a few tenths of a micron on all three \CCD{}s except
beyond $t_c$, where the model-free profile does not go to zero but keeps a
floor of 0.5 to 1 $\mu$m across the depletion region, which the
two-parameter model cannot represent.

\subsection{The fitted parameters}

Figure~\ref{fig:parameters} shows the fitted $t_c$ and $\sigma_\text{max}$
of every flat track, and Table~\ref{tab:datasets} their means in each
campaign.
The critical depth peaks at 0.40 to 0.50 on all three \CCD{}s, to the right
of the \modelCriticalDepth\ of the field-free model, and the two particle
populations, the two spacecraft rolls, and the two cameras agree to within
a few hundredths.
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
the 0.95 of a perfectly sharp cloud on all three \CCD{}s, the same floor
seen in the widths.

\subsection{Diffusion inside the depletion region}

To quantify that floor we add a third parameter to the model,
\begin{equation} \label{eq:depleted}
    \sigma^2(t) = \sigma_\text{max}^2 \left( 1 - \frac{t}{t_c} \right)
    + \sigma_d^2 \, g(t),
\end{equation}
where the first term applies only for $t < t_c$, and $g = 1$ in the
field-free layer, since that charge drifts across the full depleted
thickness, falling linearly to zero at the gates.
The grid is $t_c$ from 0.25 to 0.60, $\sigma_\text{max}$ from 2 to 8
$\mu$m, and $\sigma_d$ from 0 to 3 $\mu$m, with the same centerline
nuisance parameters as before, applied to every flat track with its
orientation taken from the two-parameter fit.
Figure~\ref{fig:depleted} shows the result.
The misfit pooled over the tracks on each \CCD\ has a sharp minimum at
$\sigma_d = \widthDepleted$ $\mu$m, with 1.5 $\mu$m already strongly
disfavored; the improvement is hundreds of units of misfit pooled but only
one or two per track, and the value each track prefers on its own is
spread over 0 to 1 $\mu$m, so the effect is modest for any one track and
consistent across them.
Adding it shifts the other two parameters: $t_c$ drops toward the model
value and $\sigma_\text{max}$ rises by about half a micron, since some of
what the two-parameter fit attributed to the field-free layer was really
this floor.
The resolution of the centerline matters here: a centerline located only
to 0.1 pixel is uncertain by 1.3 $\mu$m, the size of the effect being
sought, and with such a grid the minimum washes out entirely.""")
    result.append(ccd_diffusion.figures.stacked())
    result.append(ccd_diffusion.figures.parameters())
    result.append(ccd_diffusion.figures.profile())
    result.append(ccd_diffusion.tables.tracks())
    result.append(ccd_diffusion.figures.depleted())
    return result
