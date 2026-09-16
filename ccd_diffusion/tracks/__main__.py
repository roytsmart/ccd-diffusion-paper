"""
Refit the diffusion model to every track and rewrite ``data/iris_fits.csv``.

Run with ``python -m ccd_diffusion.tracks``.
"""

import ccd_diffusion.tracks

if __name__ == "__main__":
    ccd_diffusion.tracks.save(ccd_diffusion.tracks.fit_all(ccd_diffusion.tracks.load()))
