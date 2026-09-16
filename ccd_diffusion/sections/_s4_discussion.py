import aastex

__all__ = [
    "discussion",
]


def discussion() -> aastex.Section:
    result = aastex.Section("Discussion", label="sec:discussion")
    result.escape = False
    result.append(r"""
On the \SJI\ \CCD\ the same-pixel probability at the back surface is
$\sjiSamePixel \pm \sjiSamePixelError$, in agreement with the
\sjiSamePixelModel\ predicted by the field-free model.
Two features of the tracks deserve comment.
First, the fitted $t_c$, with a median of \sjiCriticalDepth\ on the \SJI\
\CCD, is somewhat larger than the \modelCriticalDepth\ of the model, and
beyond $t_c$ the measured same-column probability settles below the value
expected for a track with no diffusion at all.
Both indicate that the charge undergoes some additional spreading, of order
a micron, while drifting across the depletion region, which the model
neglects.
Second, the tracks are left by protons that deposit thousands of electrons
per row, whereas a \UV\ photon liberates one to three; whether the charge
density of the track widens the cloud, as it does for slow protons in thick
sensors \cite{Grosson2023}, is a systematic that the tracks themselves can
test by comparing the kernels of faint and bright tracks.""")
    return result
