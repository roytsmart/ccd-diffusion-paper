import aastex

__all__ = [
    "conclusion",
]


def conclusion() -> aastex.Section:
    result = aastex.Section("Conclusion", label="sec:conclusion")
    result.escape = False
    result.append(r"""
Glancing particle tracks measure the depth-dependent charge-diffusion kernel
of a back-illuminated \CCD\ in orbit, with no laboratory access, and on the
\IRIS\ \CCD{}s they confirm the field-free model at the back surface and
measure the spread of about a micron acquired inside the depletion region
that the model neglects.""")
    return result
