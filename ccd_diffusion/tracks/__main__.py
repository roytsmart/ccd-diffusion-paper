"""
Regenerate the data distributed with this article.

``python -m ccd_diffusion.tracks select`` asks the level-1 catalog which
frames each campaign covers and rewrites the frame list, which is where
expanding the dataset starts.
``python -m ccd_diffusion.tracks plan`` says which campaigns the frame list
and the current code would extract differently from the cutouts in the
package, so that only those need fetching.
``python -m ccd_diffusion.tracks extract`` fetches the level-1 images of a
campaign from the archive, one block at a time, and writes its cutouts,
frames, and census of track directions; ``export`` writes the same files
for a campaign from the data already in the package; and ``merge`` joins
the directories either one wrote into the package's data files, which is
how the ``data`` workflow handles every campaign in its own job.
``python -m ccd_diffusion.tracks fit`` refits every track and rewrites the
fits.
``python -m ccd_diffusion.tracks browser`` writes every track and a frame
from each camera of every campaign in the form the track browser of the
documentation reads.
"""

import sys
import json
import argparse
import pathlib
import ccd_diffusion


def _campaign_arguments(parser, tracks, verb):
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(tracks.campaigns),
        help=f"a campaign to {verb}; every campaign if not given",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="where to write the results; the package's data directory if not given",
    )


def main(argv: list[str]) -> None:
    tracks = ccd_diffusion.tracks
    parser = argparse.ArgumentParser(prog="python -m ccd_diffusion.tracks")
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select", help="rewrite the list of frames to search")
    _campaign_arguments(select, tracks, "select")

    plan = commands.add_parser("plan", help="say which campaigns need extracting")
    plan.add_argument(
        "--frames",
        type=pathlib.Path,
        help="the directory holding the frame list to plan for; the package's if not given",
    )
    plan.add_argument("--json", type=pathlib.Path, help="also write the plan here")

    extract = commands.add_parser(
        "extract", help="fetch the images and find the tracks"
    )
    _campaign_arguments(extract, tracks, "extract")
    extract.add_argument(
        "--keep", action="store_true", help="leave the images in the cache"
    )

    export = commands.add_parser(
        "export", help="write a campaign's data from the package, as extract would"
    )
    export.add_argument("--dataset", required=True, choices=list(tracks.campaigns))
    export.add_argument("--output", type=pathlib.Path, required=True)

    merge = commands.add_parser("merge", help="join separately handled campaigns")
    merge.add_argument(
        "directory",
        type=pathlib.Path,
        nargs="+",
        help="the --output directories of the campaigns, in any order",
    )
    merge.add_argument(
        "--report", type=pathlib.Path, help="write a summary table here, in Markdown"
    )

    search = commands.add_parser(
        "search", help="find observations worth adding to the campaign table"
    )
    search.add_argument("--start", default="2013-07", help="the first month, YYYY-MM")
    search.add_argument("--stop", required=True, help="the last month, YYYY-MM")
    search.add_argument(
        "--radius", type=float, default=940, help="least pointing radius, arcsec"
    )
    search.add_argument(
        "--anomaly", type=int, default=30, help="fewest frames inside the anomaly"
    )
    search.add_argument(
        "--exposure", type=float, default=4, help="least exposure, seconds"
    )
    search.add_argument(
        "--output", type=pathlib.Path, help="write the candidates here as CSV"
    )

    commands.add_parser("fit", help="refit every track")

    browser = commands.add_parser(
        "browser", help="export the tracks and example frames for the documentation"
    )
    browser.add_argument("--output", type=pathlib.Path, required=True)
    browser.add_argument(
        "--no-frames", action="store_true", help="skip fetching and rendering frames"
    )
    args = parser.parse_args(argv)

    if args.command == "select":
        frames = []
        for dataset in args.dataset or list(tracks.campaigns):
            frames += tracks.select(dataset)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        tracks.save_frames(frames, args.output)

    elif args.command == "plan":
        result = tracks.plan(tracks.frames(args.frames))
        for dataset, status in result.items():
            print(f"{dataset:>10}: {status}")
        if args.json is not None:
            args.json.write_text(
                json.dumps([dict(dataset=d, status=s) for d, s in result.items()])
            )

    elif args.command == "extract":
        chosen = args.dataset or list(tracks.campaigns)
        found, components, rows = [], [], []
        for dataset in chosen:
            t, c = tracks.extract(dataset, keep=args.keep)
            found += t
            components += c
            mine = [f for f in tracks.frames() if f["dataset"] == dataset]
            rows.append(
                tracks.Provenance(
                    dataset=dataset,
                    fingerprint=tracks.fingerprint(dataset, mine),
                    frames=len(mine),
                    tracks=len(t),
                    source="extracted",
                )
            )
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        frames = [f for f in tracks.frames() if f["dataset"] in chosen]
        tracks.save_tracks(found, args.output, frames=frames)
        tracks.save_census(components, args.output)
        if args.output is None:
            for r in rows:
                r.source = ""
        tracks.save_campaigns(rows, args.output)

    elif args.command == "export":
        row = tracks.export(args.dataset, args.output)
        print(
            f"{row.dataset}: {row.frames} frames, {row.tracks} tracks, fingerprint {row.fingerprint or 'unknown'}"
        )

    elif args.command == "merge":
        result = tracks.merge(args.directory)
        lines = ["| campaign | outcome | frames | tracks |", "|---|---|---|---|"]
        for row, status in result:
            print(
                f"{row.dataset:>10}: {status}, {row.frames} frames, {row.tracks} tracks"
            )
            lines.append(f"| {row.dataset} | {status} | {row.frames} | {row.tracks} |")
        missing = [
            d for d in tracks.campaigns if d not in {r.dataset for r, _ in result}
        ]
        for dataset in missing:
            print(f"{dataset:>10}: missing")
            lines.append(f"| {dataset} | missing | | |")
        if args.report is not None:
            args.report.write_text("\n".join(lines) + "\n")

    elif args.command == "search":
        found = tracks.search(
            args.start, args.stop, args.radius, args.exposure, args.anomaly
        )
        print(
            f"{'day':>10} {'from':>5} {'hours':>5} {'radius':>6} {'roll':>5} {'exp':>4} "
            f"{'anomaly':>7} {'quiet':>5} {'seconds':>7}  obsid"
        )
        for o in found[:40]:
            print(
                f"{o.day:>10} {o.start[11:16]:>5} {o.hours:>5.1f} {o.radius:>6.0f} "
                f"{o.roll:>5.0f} {o.exposure:>4.0f} {o.anomaly:>7} {o.quiet:>5} "
                f"{o.seconds:>7.0f}  {o.obsid}"
            )
        if len(found) > 40:
            print(f"... and {len(found) - 40} more")
        if args.output is not None:
            tracks.save_search(found, args.output)

    elif args.command == "fit":
        tracks.save(*tracks.fit_all(tracks.load()))

    elif args.command == "browser":
        num = tracks.export_tracks(args.output / "tracks.json")
        print(f"{num} tracks written to {args.output / 'tracks.json'}")
        if not args.no_frames:
            rendered = tracks.export_frames(args.output / "frames")
            print(f"{len(rendered)} frames rendered in {args.output / 'frames'}")


if __name__ == "__main__":
    main(sys.argv[1:])
