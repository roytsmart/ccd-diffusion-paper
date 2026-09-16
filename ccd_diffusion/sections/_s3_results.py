import aastex
import ccd_diffusion

__all__ = [
    "results",
]


def results() -> aastex.Section:
    result = aastex.Section("Results", label="sec:results")
    result.escape = False
    result.append(r"""
Figure~\ref{fig:tracks} shows one of the flat tracks along with the
probability that two electrons deposited in the same slice are collected in
the same column, $\sum_j f_j^2$ where $f_j$ is the fraction of the slice's
charge in column $j$, averaged over the flat tracks on each \CCD\ in bins of
fractional depth.
This quantity requires no model of the shape of the kernel, and since the
kernel is separable, the probability that two electrons are collected in the
same pixel is its square.
Table~\ref{tab:tracks} lists the medians of the fitted parameters and the
same-column and same-pixel probabilities for the slices within $D/10$ of the
back surface, alongside the same-pixel probability predicted by the
field-free model with $t_c = \modelCriticalDepth$ and
$\sigma_\text{max} = \modelWidthMax$ $\mu$m.""")
    result.append(ccd_diffusion.tables.tracks())
    return result
