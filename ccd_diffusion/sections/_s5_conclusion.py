import aastex

__all__ = [
    "conclusion",
]


def conclusion() -> aastex.Section:
    result = aastex.Section("Conclusion", label="sec:conclusion")
    result.escape = False
    result.append(r"""
Glancing particle tracks measure the depth-dependent charge-diffusion kernel
of a back-illuminated \CCD\ in orbit, with no laboratory access.
On the \IRIS\ \CCD{}s they give the thickness of the field-free layer,
the width of the charge cloud at the back surface, which differs by a
fifth between four sensors of one design, and a spread of under a micron
acquired inside the depletion region that a field-free kernel alone would
miss.""")
    return result
