"""
Regenerate the data distributed with this article.

``python -m ccd_diffusion.tracks extract`` fetches the level-1 images from
the archive, one block at a time, and rewrites the track cutouts, the
frame list, and the census of track directions; ``python -m
ccd_diffusion.tracks fit`` refits every track and rewrites the fits.
"""

import sys
import ccd_diffusion


def main(argv: list[str]) -> None:
    command = argv[0] if argv else "fit"
    tracks = ccd_diffusion.tracks
    if command == "fit":
        tracks.save(tracks.fit_all(tracks.load()))
    elif command == "extract":
        keep = "--keep" in argv
        found = []
        components = []
        for dataset in tracks.datasets:
            t, c = tracks.extract(dataset, keep=keep)
            found += t
            components += c
        tracks.save_tracks(found)
        tracks.save_census(components)
    else:
        raise SystemExit(f"unknown command {command!r}; use 'extract' or 'fit'")


if __name__ == "__main__":
    main(sys.argv[1:])
