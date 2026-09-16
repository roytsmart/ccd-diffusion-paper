import ccd_diffusion

__all__ = [
    "keywords",
]


def keywords() -> "ccd_diffusion.spie.Keywords":
    """The keywords which follow the abstract."""
    return ccd_diffusion.spie.Keywords(
        [
            "charge-coupled devices",
            "charge diffusion",
            "point spread function",
            "ultraviolet detectors",
            "particle tracks",
            "IRIS",
        ]
    )
