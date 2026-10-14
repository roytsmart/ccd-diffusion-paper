import astropy.units as u
import pylatex
import ccd_diffusion

__all__ = [
    "models",
]

_chips = ["FUV1", "FUV2", "NUV", "SJI"]


def models() -> pylatex.Table:
    """The slab model against the Gaussian model, one row per CCD."""
    tracks = ccd_diffusion.tracks

    result = pylatex.Table(position="htb!")
    result.escape = False
    result.append(pylatex.Command("centering"))
    result.add_caption(pylatex.NoEscape(r"""
The slab model of Equation~\ref{eq:width} against the Gaussian model of
Equation~\ref{eq:gaussian}, on the tracks flat under both.
$\Delta M$ is the misfit of the slab model minus that of the Gaussian,
summed over the tracks, each model at its own $\sigma_d$, and the next two
columns count the tracks the slab model fits better or worse by more than
two.
The residual is the model-free width of Figure~\ref{fig:stacked}b minus
that found the same way in the fractions the fits predict, in the depth bin
at the back surface and as the root mean square over all twenty bins, for
the Gaussian model and then the slab model."""))
    result.append(pylatex.Command("label", "tab:models"))
    result.append(pylatex.Command("footnotesize"))

    tabular = pylatex.Tabular("lrrrrcc", booktabs=False)
    tabular.escape = False
    tabular.add_hline()
    tabular.add_row(
        [
            pylatex.NoEscape(s)
            for s in (
                r"\CCD",
                "tracks",
                r"$\Delta M$",
                "better",
                "worse",
                r"back residual ($\mu$m)",
                r"rms residual ($\mu$m)",
            )
        ]
    )
    tabular.add_hline()
    for chip in _chips:
        if not tracks.flat(chip):
            continue
        c = tracks.comparison(chip)
        back = {m: r[{tracks.axis_depth: 0}].ndarray for m, r in c.residual.items()}
        rms = c.residual_rms

        def pair(values: dict[str, u.Quantity]) -> str:
            g = values["gaussian"].to_value(u.um)
            s = values["slab"].to_value(u.um)
            return f"${g:+.2f} \\to {s:+.2f}$"

        def pair_rms(values: dict[str, u.Quantity]) -> str:
            g = values["gaussian"].to_value(u.um)
            s = values["slab"].to_value(u.um)
            return f"${g:.2f} \\to {s:.2f}$"

        row = [
            chip,
            f"{c.num}",
            # a comma in math mode is punctuation and takes a space after it
            "$" + f"{c.misfit['slab']:+,.0f}".replace(",", "{,}") + "$",
            f"{c.better}",
            f"{c.worse}",
            pair(back),
            pair_rms(rms),
        ]
        tabular.add_row([pylatex.NoEscape(v) for v in row])
    tabular.add_hline()

    result.append(tabular)
    return result
