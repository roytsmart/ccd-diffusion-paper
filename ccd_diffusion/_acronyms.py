import aastex

__all__ = [
    "acronyms",
]


def acronyms() -> list[aastex.Acronym]:
    """
    The acronyms used in this article, each of which defines a LaTeX command
    that expands on first use and abbreviates thereafter.
    """
    return [
        aastex.Acronym("CCD", "charge-coupled device", plural=True),
        aastex.Acronym("UV", "ultraviolet"),
        aastex.Acronym("FUV", "far ultraviolet"),
        aastex.Acronym("PSF", "point spread function"),
        aastex.Acronym("IRIS", "the Interface Region Imaging Spectrograph"),
        aastex.Acronym("SJI", "the Slit-Jaw Imager"),
        aastex.Acronym("AIA", "the Atmospheric Imaging Assembly"),
        aastex.Acronym("SAA", "the South Atlantic Anomaly"),
        aastex.Acronym("PTC", "photon-transfer curve", plural=True),
        aastex.Acronym("VMR", "variance-to-mean ratio"),
    ]
