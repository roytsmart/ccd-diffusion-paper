import pathlib
import textwrap
import numpy as np
import scipy.ndimage
import ccd_diffusion
from . import _extract

_module = textwrap.dedent('''
    """The module docstring."""

    import math

    threshold = 5.0
    """An attribute docstring."""


    def area(r):
        """A function docstring."""
        # a comment
        return math.pi * r**2
    ''')


def test_hash_source_ignores_prose_and_layout():
    reference = ccd_diffusion.tracks.hash_source(_module)
    reworded = _module.replace("The module docstring.", "Reworded.")
    reworded = reworded.replace("# a comment", "# another comment")
    reworded = reworded.replace("An attribute docstring.", "Changed.")
    assert ccd_diffusion.tracks.hash_source(reworded) == reference
    # blank lines and line breaks inside brackets do not matter either
    spaced = _module.replace("\n\n\n", "\n\n").replace("area(r)", "area(\n    r\n)")
    assert ccd_diffusion.tracks.hash_source(spaced) == reference


def test_hash_source_sees_code():
    reference = ccd_diffusion.tracks.hash_source(_module)
    assert ccd_diffusion.tracks.hash_source(_module.replace("5.0", "6.0")) != reference
    assert (
        ccd_diffusion.tracks.hash_source(_module.replace("r**2", "r**3")) != reference
    )


def test_extractor_hash_is_stable():
    a = ccd_diffusion.tracks.extractor_hash()
    b = ccd_diffusion.tracks.extractor_hash()
    assert a == b
    assert len(a) == 64


def test_fingerprint():
    frames = ccd_diffusion.tracks.frames()
    a = ccd_diffusion.tracks.fingerprint("sji", frames)
    assert a == ccd_diffusion.tracks.fingerprint("sji", frames)
    assert a != ccd_diffusion.tracks.fingerprint("2014", frames)
    # dropping one frame changes it
    fewer = [
        f for f in frames if not (f["dataset"] == "sji" and f["fsn"] == "39496475")
    ]
    assert len(fewer) == len(frames) - 1
    assert ccd_diffusion.tracks.fingerprint("sji", fewer) != a
    # the order of the frames does not
    assert ccd_diffusion.tracks.fingerprint("sji", list(reversed(frames))) == a


def test_campaigns_roundtrip(tmp_path: pathlib.Path):
    rows = [
        ccd_diffusion.tracks.Provenance("2014", "abc", 975, 49),
        ccd_diffusion.tracks.Provenance("sji", "", 1158, 321, source="exported"),
    ]
    ccd_diffusion.tracks.save_campaigns(rows, tmp_path)
    loaded = ccd_diffusion.tracks.load_campaigns(tmp_path)
    assert [
        (r.dataset, r.fingerprint, r.frames, r.tracks, r.source) for r in loaded
    ] == [
        ("2014", "abc", 975, 49, ""),
        ("sji", "", 1158, 321, "exported"),
    ]
    assert ccd_diffusion.tracks.load_campaigns(tmp_path / "nowhere") == []


def test_plan(tmp_path: pathlib.Path):
    frames = ccd_diffusion.tracks.frames()
    # nothing recorded: everything is new
    assert set(ccd_diffusion.tracks.plan(frames, tmp_path).values()) == {"new"}
    # record two campaigns, one current and one out of date
    rows = [
        ccd_diffusion.tracks.Provenance(
            "2014", ccd_diffusion.tracks.fingerprint("2014", frames), 975, 49
        ),
        ccd_diffusion.tracks.Provenance("2018", "0000000000000000", 311, 85),
    ]
    ccd_diffusion.tracks.save_campaigns(rows, tmp_path)
    result = ccd_diffusion.tracks.plan(frames, tmp_path)
    assert result["2014"] == "fresh"
    assert result["2018"] == "stale"
    assert result["sji"] == "new"


def test_export_and_merge(tmp_path: pathlib.Path):
    frames = ccd_diffusion.tracks.frames()
    tracks = ccd_diffusion.tracks.load()
    parts = []
    present = [
        d
        for d in ccd_diffusion.tracks.campaigns
        if any(f["dataset"] == d for f in frames)
    ]
    assert len(present) >= 5
    for dataset in present:
        part = tmp_path / dataset
        row = ccd_diffusion.tracks.export(dataset, part)
        assert row.dataset == dataset
        assert row.source == "exported"
        assert row.frames == sum(f["dataset"] == dataset for f in frames)
        assert row.tracks == sum(t.dataset == dataset for t in tracks)
        assert ccd_diffusion.tracks.load_campaigns(part)[0].dataset == dataset
        parts.append(part)
    # pretend one campaign was extracted afresh
    rows = ccd_diffusion.tracks.load_campaigns(parts[0])
    rows[0].source = "extracted"
    ccd_diffusion.tracks.save_campaigns(rows, parts[0])

    merged = tmp_path / "merged"
    merged.mkdir()
    result = ccd_diffusion.tracks.merge(list(reversed(parts)), merged)
    assert [r.dataset for r, _ in result] == present
    statuses = dict((r.dataset, s) for r, s in result)
    first = present[0]
    assert statuses[first] == "refreshed"
    # the package records no fingerprints yet, so exported campaigns are kept, not reused
    assert all(s == "kept" for d, s in statuses.items() if d != first)

    loaded = ccd_diffusion.tracks.load(merged)
    assert sorted(t.name for t in loaded) == sorted(t.name for t in tracks)
    assert len(ccd_diffusion.tracks.frames(merged)) == len(frames)
    saved = {r.dataset: r for r in ccd_diffusion.tracks.load_campaigns(merged)}
    assert saved[first].fingerprint == ccd_diffusion.tracks.fingerprint(first, frames)
    assert all(r.source == "" for r in saved.values())
    # with those fingerprints recorded, a plan against the same frames reuses the first
    assert ccd_diffusion.tracks.plan(frames, merged)[first] == "fresh"


def test_find_joins_a_sharp_stepping_track():
    # one column every five rows with no diffusion: the rows touch only at corners
    residual = np.zeros((80, 40), np.float32)
    for r in range(20, 50):
        residual[r, 10 + (r - 20) // 5] = 900.0
    frame = dict(dataset="t", fsn="1", saa="1")
    valid = np.ones(residual.shape, bool)
    found = ccd_diffusion.tracks.find(frame, residual, valid, 3.0, 6.0)
    assert [t.length for t in found] == [30]
    assert abs(found[0].slope - 0.2) < 0.02


def test_find_needs_corner_connectivity(monkeypatch):
    residual = np.zeros((80, 40), np.float32)
    for r in range(20, 50):
        residual[r, 10 + (r - 20) // 5] = 900.0
    frame = dict(dataset="t", fsn="1", saa="1")
    valid = np.ones(residual.shape, bool)
    monkeypatch.setattr(
        _extract, "_touching", scipy.ndimage.generate_binary_structure(2, 1)
    )
    assert ccd_diffusion.tracks.find(frame, residual, valid, 3.0, 6.0) == []
