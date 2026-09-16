import aastex

__all__ = [
    "introduction",
]


def introduction() -> aastex.Section:
    result = aastex.Section("Introduction")
    result.escape = False
    result.append(r"""
A back-illuminated silicon \CCD\ absorbs \UV\ light within a fraction of a
micron of its back surface, in a layer where there is no electric field to
sweep the photogenerated charge toward the gates.
The charge diffuses until it reaches the edge of the depletion region, and
the resulting charge cloud, which can be several microns wide, is shared
among neighboring pixels.
This sharing broadens the \PSF\ of the sensor \cite{Pavlov1999} and, because
the electrons liberated by one photon are shared together, it changes the
statistics of the image noise, entering the \VMR\ of a photon-transfer curve
through the probability that two electrons from the same photon land in the
same pixel \cite{Janesick2001}.

The width of the charge cloud is set by the thickness of the undepleted
layer, which depends on the resistivity of the wafer and the bias applied to
the device, both of which vary between nominally identical sensors and are
rarely known for a sensor in orbit.
Laboratory measurements use light of varying penetration depth
\cite{Stern2004,Widenhorn2010} or the split events of soft X-rays \cite{Fraser1994}, and
neither is available once the instrument has launched.

Energetic particles offer a probe that arrives for free.
A particle crossing the sensor at glancing incidence deposits charge along a
track tens of pixels long, and because it enters at the back surface and
exits at the gates, the fractional position along the track is the
fractional depth of the charge.
Cosmic-ray muons have been used in this way to measure the growth of charge
diffusion with drift distance in thick, fully depleted \CCD{}s
\cite{FisherLevine2015}, where the cloud widens with depth.
In a thin sensor with an undepleted back surface the dependence is reversed,
the cloud is widest at the back surface and collapses to a pixel-sharp line
in the depletion region, and the same tracks measure the field-free kernel
that governs \UV\ imaging.

Here we apply this method to the three \CCD{}s of \IRIS\
\cite{DePontieu2014}, whose \FUV\ and near-\UV\ photon-transfer curves
\cite{Wulser2018} motivated the question.
Section~\ref{sec:method} describes the tracks, the model of their
cross-section, and the fit; Section~\ref{sec:results} presents the kernel
of each sensor; and Section~\ref{sec:discussion} compares it with the
field-free model and discusses the systematics, including the charge
density of the tracks.""")
    return result
