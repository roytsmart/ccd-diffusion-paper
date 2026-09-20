"""
Which campaigns' cutouts are current.

Extracting a campaign turns gigabytes of images into a few hundred
kilobytes of cutouts, so the cutouts are what is worth keeping. Each
campaign carries a fingerprint over the frames it was extracted from and
over the extraction code, stored in ``data/iris_campaigns.csv``, and a
regeneration re-extracts only the campaigns whose fingerprint has changed,
taking the rest straight from the data already in the package.
"""

import ast
import csv
import dataclasses
import hashlib
import io
import pathlib
import tokenize
from ._tracks import half_width, frames, load, _directory_data
from ._select import campaigns
from ._extract import save_tracks, save_census, load_census

__all__ = [
    "hash_source",
    "extractor_hash",
    "fingerprint",
    "Provenance",
    "save_campaigns",
    "load_campaigns",
    "plan",
    "export",
    "merge",
]

_modules_extractor = ("_extract.py", "_archive.py")
"""The modules whose code decides what a campaign's cutouts contain."""


def hash_source(source: str) -> str:
    """
    A hash of the code in a module which ignores its comments, blank lines,
    line breaks, and docstrings, so that documenting or reformatting the
    extractor does not look like changing it.

    Parameters
    ----------
    source
        The text of the module.
    """
    docstrings = set()
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        for statement in body if isinstance(body, list) else []:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                docstrings.add((statement.lineno, statement.col_offset))
    # Token numbers change between Python versions, so hash their names, and
    # newer versions split an f-string into several tokens, so put it back
    # together from the source text to hash it as the one string it was.
    lines = source.splitlines(keepends=True)

    def slice_(start, stop):
        (r1, c1), (r2, c2) = start, stop
        if r1 == r2:
            return lines[r1 - 1][c1:c2]
        return lines[r1 - 1][c1:] + "".join(lines[r1 : r2 - 1]) + lines[r2 - 1][:c2]

    skip = {"COMMENT", "NL", "ENCODING"}
    digest = hashlib.sha256()
    tokens = iter(tokenize.generate_tokens(io.StringIO(source).readline))
    for token in tokens:
        name = tokenize.tok_name[token.type]
        if name in skip:
            continue
        if name == "FSTRING_START":
            depth, start = 1, token.start
            for inner in tokens:
                kind = tokenize.tok_name[inner.type]
                depth += (kind == "FSTRING_START") - (kind == "FSTRING_END")
                if depth == 0:
                    break
            name, text = "STRING", slice_(start, inner.end)
        else:
            text = token.string
        if name == "STRING" and token.start in docstrings:
            continue
        digest.update(f"{name}:{text}\n".encode())
    return digest.hexdigest()


def extractor_hash() -> str:
    """
    A hash of the extraction code, :mod:`ccd_diffusion.tracks._extract`
    and :mod:`ccd_diffusion.tracks._archive`, and of the cutout width.
    """
    digest = hashlib.sha256()
    here = pathlib.Path(__file__).parent
    for name in _modules_extractor:
        digest.update(hash_source((here / name).read_text(encoding="utf-8")).encode())
    digest.update(f"half_width={half_width}".encode())
    return digest.hexdigest()


def fingerprint(
    dataset: str, frames: "list[dict[str, str]] | tuple[dict[str, str], ...]"
) -> str:
    """
    The fingerprint of a campaign: a hash of the frames it covers and of
    the extraction code.

    Parameters
    ----------
    dataset
        The campaign.
    frames
        The frame list the campaign is drawn from, rows of
        ``iris_frames.csv``.
    """
    fsn = sorted(int(f["fsn"]) for f in frames if f["dataset"] == dataset)
    digest = hashlib.sha256()
    digest.update(extractor_hash().encode())
    digest.update(repr(fsn).encode())
    return digest.hexdigest()[:16]


@dataclasses.dataclass(eq=False)
class Provenance:
    """The provenance of one campaign's cutouts."""

    dataset: str
    """The campaign."""

    fingerprint: str
    """Its :func:`fingerprint` when its cutouts were extracted, empty if unknown."""

    frames: int
    """The number of frames it covers."""

    tracks: int
    """The number of tracks found."""

    source: str = ""
    """
    Where a separately handled campaign came from, ``extracted`` afresh or
    ``exported`` from the data already in the package; empty in the
    package's own file.
    """


_fields = ["dataset", "fingerprint", "frames", "tracks", "source"]


def save_campaigns(
    rows: list[Provenance], directory: None | pathlib.Path = None
) -> None:
    """
    Write the provenance of every campaign to ``iris_campaigns.csv``.

    Parameters
    ----------
    rows
        One row per campaign.
    directory
        The data directory of the package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    with open(pathlib.Path(directory) / "iris_campaigns.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dataclasses.asdict(r))


def load_campaigns(directory: None | pathlib.Path = None) -> list[Provenance]:
    """
    Read ``iris_campaigns.csv``, an empty list if there is none.

    Parameters
    ----------
    directory
        The data directory of the package if :obj:`None`.
    """
    if directory is None:
        directory = _directory_data
    path = pathlib.Path(directory) / "iris_campaigns.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [
            Provenance(
                dataset=r["dataset"],
                fingerprint=r["fingerprint"],
                frames=int(r["frames"]),
                tracks=int(r["tracks"]),
                source=r.get("source", ""),
            )
            for r in csv.DictReader(f)
        ]


def plan(
    frames: "list[dict[str, str]] | tuple[dict[str, str], ...]",
    directory: None | pathlib.Path = None,
) -> dict[str, str]:
    """
    Which campaigns need extracting: ``fresh`` if the package's cutouts
    carry the fingerprint the given frames and the current code would
    produce, ``stale`` if they carry another, and ``new`` if the package
    has none.

    Parameters
    ----------
    frames
        The frame list to compare against, rows of ``iris_frames.csv``.
    directory
        Where to read the package's provenance from, its data directory if
        :obj:`None`.
    """
    known = {c.dataset: c.fingerprint for c in load_campaigns(directory)}
    result = {}
    for dataset in campaigns:
        current = fingerprint(dataset, frames)
        if dataset not in known:
            result[dataset] = "new"
        elif known[dataset] == current:
            result[dataset] = "fresh"
        else:
            result[dataset] = "stale"
    return result


def export(dataset: str, output: pathlib.Path) -> Provenance:
    """
    Copy one campaign's frames, cutouts, census, and provenance out of the
    package into a directory of its own, in the layout ``extract`` writes,
    so that :func:`merge` can join it with freshly extracted campaigns.

    Parameters
    ----------
    dataset
        The campaign.
    output
        The directory to write.
    """
    mine = [f for f in frames() if f["dataset"] == dataset]
    if not mine:
        raise ValueError(f"the package has no frames for {dataset!r}")
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tracks = [t for t in load() if t.dataset == dataset]
    save_tracks(tracks, output, frames=mine)
    try:
        components = [c for c in load_census() if c.dataset == dataset]
    except FileNotFoundError:
        components = []
    save_census(components, output)
    known = {c.dataset: c for c in load_campaigns()}
    row = Provenance(
        dataset=dataset,
        fingerprint=known[dataset].fingerprint if dataset in known else "",
        frames=len(mine),
        tracks=len(tracks),
        source="exported",
    )
    save_campaigns([row], output)
    return row


def merge(
    directories: list[pathlib.Path],
    directory: None | pathlib.Path = None,
) -> list[tuple[Provenance, str]]:
    """
    Join separately handled campaigns into the package's data files.

    Parameters
    ----------
    directories
        The directories written by ``extract`` or :func:`export`, one per
        campaign, in any order.
    directory
        The data directory to write, the package's if :obj:`None`.

    Returns
    -------
    The provenance of each campaign in the order of
    :data:`ccd_diffusion.tracks.campaigns`, with what became of it:
    ``refreshed`` if extracted afresh, ``reused`` if exported and current,
    ``kept`` if exported but out of date, so that the next run redoes it.
    """
    order = {d: i for i, d in enumerate(campaigns)}
    parts = {}
    for d in directories:
        d = pathlib.Path(d)
        row = load_campaigns(d)
        if len(row) != 1:
            raise ValueError(f"{d} does not hold exactly one campaign")
        parts[row[0].dataset] = (d, row[0])
    frames_all, tracks_all, census_all, result = [], [], [], []
    for dataset in sorted(parts, key=order.__getitem__):
        d, row = parts[dataset]
        frames_all += list(frames(d))
        tracks_all += list(load(d))
        census_all += load_census(d)
    for dataset in sorted(parts, key=order.__getitem__):
        d, row = parts[dataset]
        current = fingerprint(dataset, frames_all)
        if row.source == "extracted":
            status = "refreshed"
            row.fingerprint = current
        elif row.fingerprint == current:
            status = "reused"
        else:
            status = "kept"
        result.append(
            (Provenance(dataset, row.fingerprint, row.frames, row.tracks), status)
        )
    save_tracks(tracks_all, directory, frames=frames_all)
    save_census(census_all, directory)
    save_campaigns([r for r, _ in result], directory)
    return result
