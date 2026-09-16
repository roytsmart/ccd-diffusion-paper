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
\IRIS\ \CCD{}s they confirm the field-free model at the back surface while
revealing a small residual spread inside the depletion region.""")
    return result
