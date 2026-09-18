"""
Regenerate the data distributed with this article.

``python -m ccd_diffusion.tracks select`` asks the level-1 catalog which
frames each campaign covers and rewrites the frame list, which is where
expanding the dataset starts.
``python -m ccd_diffusion.tracks extract`` fetches the level-1 images from
the archive, one block at a time, and rewrites the track cutouts, the
frame list, and the census of track directions. With ``--dataset`` it
handles one campaign and, with ``--output``, writes its results somewhere
other than the package's data directory, so that the campaigns can be
extracted separately (on separate machines, as the ``data`` workflow does)
and joined afterwards by ``python -m ccd_diffusion.tracks merge``.
``python -m ccd_diffusion.tracks fit`` refits every track and rewrites the
fits.
"""

import sys
import argparse
import pathlib
import ccd_diffusion


def main(argv: list[str]) -> None:
    tracks = ccd_diffusion.tracks
    parser = argparse.ArgumentParser(prog="python -m ccd_diffusion.tracks")
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser(
        "extract", help="fetch the images and find the tracks"
    )
    extract.add_argument(
        "--dataset",
        action="append",
        choices=list(tracks.datasets),
        help="a campaign to extract; every campaign if not given",
    )
    extract.add_argument(
        "--output",
        type=pathlib.Path,
        help="where to write the results; the package's data directory if not given",
    )
    extract.add_argument(
        "--keep", action="store_true", help="leave the images in the cache"
    )
    select = commands.add_parser("select", help="rewrite the list of frames to search")
    select.add_argument(
        "--dataset",
        action="append",
        choices=list(tracks.campaigns),
        help="a campaign to select; every campaign if not given",
    )
    select.add_argument(
        "--output",
        type=pathlib.Path,
        help="where to write the frame list; the package's data directory if not given",
    )
    merge = commands.add_parser("merge", help="join separately extracted campaigns")
    merge.add_argument(
        "directory",
        type=pathlib.Path,
        nargs="+",
        help="the --output directories of the campaigns, in any order",
    )
    commands.add_parser("fit", help="refit every track")
    args = parser.parse_args(argv)

    # the campaigns in the order of the frame list, which the data files follow
    order = {
        d: i for i, d in enumerate(dict.fromkeys(f["dataset"] for f in tracks.frames()))
    }

    if args.command == "select":
        frames = []
        for dataset in sorted(args.dataset or order, key=order.__getitem__):
            frames += tracks.select(dataset)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        tracks.save_frames(frames, args.output)
    elif args.command == "fit":
        tracks.save(*tracks.fit_all(tracks.load()))
    elif args.command == "extract":
        found = []
        components = []
        for dataset in sorted(args.dataset or order, key=order.__getitem__):
            t, c = tracks.extract(dataset, keep=args.keep)
            found += t
            components += c
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        tracks.save_tracks(found, args.output)
        tracks.save_census(components, args.output)
    elif args.command == "merge":
        found = []
        components = []
        for directory in args.directory:
            found += tracks.load(directory)
            components += tracks.load_census(directory)
        found.sort(key=lambda t: (order[t.dataset], int(t.name.rsplit("-", 1)[1])))
        components.sort(key=lambda c: (order[c.dataset], c.fsn))
        tracks.save_tracks(found)
        tracks.save_census(components)


if __name__ == "__main__":
    main(sys.argv[1:])
