import ccd_diffusion

__all__ = [
    "abstract",
]


def abstract() -> "ccd_diffusion.spie.Abstract":
    result = ccd_diffusion.spie.Abstract()
    result.append(r"""
The lateral diffusion of photogenerated charge in the undepleted layer of a
back-illuminated silicon sensor sets its point spread function and the
statistics of its noise in the ultraviolet, but it is rarely measured on a
flight sensor.
We show that glancing particle tracks, which cross the full thickness of the
sensor over a length of tens of pixels, sample the diffusion width at every
depth in a single exposure, and we use them to measure the depth-dependent
charge-diffusion kernel of the three \acs{CCD}s of the Interface Region
Imaging Spectrograph in orbit, from \numTracks\ tracks found in level-1
images taken mostly inside the South Atlantic Anomaly.
The width of the charge cloud at the back surface, the depth at which the
field-free layer ends, the spread acquired drifting across the depletion
region, and the probability that two electrons deposited at the same depth
are collected in the same pixel are measured directly and compared with a
field-free diffusion model.
On the Slit-Jaw Imager the same-pixel probability at the back surface is
$\sjiSamePixel \pm \sjiSamePixelError$, against \sjiSamePixelModel\ from the
model, and the tracks require a spread of \widthDepletedSji\ $\mu$m inside
the depletion region, which the field-free model neglects.
The method needs no laboratory access and applies to any back-illuminated
sensor in orbit.
\acresetall""")
    return result
