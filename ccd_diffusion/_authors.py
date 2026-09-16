import ccd_diffusion

__all__ = [
    "authors",
]


def authors() -> list["ccd_diffusion.spie.Author"]:
    """
    The authors of this article and their affiliations.
    """

    spie = ccd_diffusion.spie

    msu = spie.Affiliation(
        key="a",
        name=(
            "Montana State University, "
            "Department of Physics, "
            "P.O. Box 173840, "
            "Bozeman, MT 59717, USA"
        ),
    )

    roy = spie.Author(
        name="Roy T. Smart",
        affiliation=msu,
        orcid="0000-0002-9997-5515",
        email="roytsmart@gmail.com",
        corresponding=True,
    )

    charles = spie.Author(
        name="Charles C. Kankelborg",
        affiliation=msu,
        orcid="0000-0002-1992-7469",
        email="kankel@montana.edu",
    )

    return [
        roy,
        charles,
    ]
