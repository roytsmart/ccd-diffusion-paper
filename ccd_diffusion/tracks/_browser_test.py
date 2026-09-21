import json
import pathlib
import numpy as np
import PIL.Image
import ccd_diffusion


def test_export_tracks(tmp_path: pathlib.Path):
    path = tmp_path / "browser" / "tracks.json"
    num = ccd_diffusion.tracks.export_tracks(path)
    fits = ccd_diffusion.tracks.fits()
    assert num == len(fits)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["num"] == num
    assert data["half_width"] == ccd_diffusion.tracks.half_width
    assert set(data["width_depleted"]) == {"FUV1", "FUV2", "SJI"}
    assert data["campaigns"][0] == fits[0].track.dataset
    columns = data["tracks"]
    assert all(len(v) == num for v in columns.values())
    assert sum(columns["flat"]) == sum(f.flat for f in fits)
    assert sum(columns["core"]) == sum(
        f.flat and 0.25 < f.critical_depth < 0.6 for f in fits
    )
    i = 5
    assert columns["name"][i] == fits[i].track.name
    assert len(columns["charge"][i]) == fits[i].track.length
    assert len(columns["charge"][i][0]) == 2 * ccd_diffusion.tracks.half_width + 1
    assert np.allclose(columns["charge"][i], fits[i].track.charge.ndarray, atol=0.5)
    assert abs(columns["tc"][i] - fits[i].critical_depth) < 1e-3


def test_frame_choices():
    chosen = ccd_diffusion.tracks.frame_choices()
    assert chosen
    keys = [(f["dataset"], f["image"][:3]) for f in chosen]
    assert len(keys) == len(set(keys))  # one frame per camera per campaign
    assert all(int(f["tracks"]) > 0 for f in chosen)
    by_key = {k: f for k, f in zip(keys, chosen)}
    for f in ccd_diffusion.tracks.frames():
        key = (f["dataset"], f["image"][:3])
        if key in by_key:
            assert int(f["tracks"]) <= int(by_key[key]["tracks"])


def test_render_frame(tmp_path: pathlib.Path):
    image = next(im for im in ccd_diffusion.tracks.images() if im.image == "FUV")
    path = tmp_path / "frame.png"
    crop = ccd_diffusion.tracks.render_frame(image.data.ndarray, image.tracks, path)
    with PIL.Image.open(path) as rendered:
        width, height = rendered.size
        pixels = np.asarray(rendered.convert("L"))
    assert width == crop["columns"][1] - crop["columns"][0]
    assert height == crop["rows"][1] - crop["rows"][0]
    assert crop["rows"][0] > 0  # the frame is cropped to what was read out
    assert pixels.max() == 255 and pixels.min() < 100  # unread white, hits dark
