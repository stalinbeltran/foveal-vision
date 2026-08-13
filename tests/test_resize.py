"""A' — derived sources (fv.datasets.resize).

The three rules the module exists to keep are each pinned by a test, because
each one fails silently: the geometry drifts a fraction of a pixel, the mask
grows classes nobody declared, or a nested level keeps the old resolution while
the top level looks right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fv.datasets.loader import SourceDataset, discover_sources
from fv.datasets.resize import ResizeConfig, ResizeError, resize_source

W, H = 48, 36


def _quad(x0, y0, x1, y1):
    return [[float(x0), float(y0)], [float(x1), float(y0)],
            [float(x1), float(y1)], [float(x0), float(y1)]]


def make_rich_source(root: Path, name: str, count: int = 4, w: int = W, h: int = H,
                     with_mask: bool = True, extra_meta: dict | None = None) -> Path:
    """A source with the FULL label shape: blocks + nested lines + words.

    conftest's make_source only writes blocks, which is all the extractor reads
    -- and a resize that only handles blocks would pass against it.
    """
    out = root / name
    (out / "images").mkdir(parents=True)
    if with_mask:
        (out / "masks").mkdir(parents=True)
    lines = []
    for i in range(count):
        img = np.full((h, w), 230, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = 4, 6, 4 + 20, 6 + 12
        img[y0:y1, x0:x1] = 40
        mask[y0:y1, x0:x1] = 255
        rel = f"images/{i:06d}.png"
        Image.fromarray(img).save(out / rel)
        rec = {
            "index": i,
            "image": rel,
            "labels": {
                "image_id": f"{name}/{i:06d}",
                "width": w, "height": h,
                "blocks": [{"block_id": "b0", "kind": "paragraph", "angle": 0.0,
                            "text": "hola mundo",
                            "box": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                            "quad": _quad(x0, y0, x1, y1)}],
                "lines": [{"block_id": "b0", "index": 0, "text": "hola mundo",
                           "box": [float(x0), float(y0), float(x1 - x0), 6.0],
                           "quad": _quad(x0, y0, x1, y0 + 6)}],
                "words": [{"block_id": "b0", "line_index": 0, "text": "hola",
                           "box": [float(x0), float(y0), 8.0, 6.0],
                           "quad": _quad(x0, y0, x0 + 8, y0 + 6)}],
                "has_overlap": False,
            },
        }
        if with_mask:
            mrel = f"masks/{i:06d}.png"
            Image.fromarray(mask).save(out / mrel)
            rec["mask"] = mrel
        lines.append(json.dumps(rec))
    (out / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"id": name, "count": count, "seed": 3, "spec_version": 1}
    meta.update(extra_meta or {})
    (out / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


@pytest.fixture()
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv("FV_ROOT", str(tmp_path))
    monkeypatch.setenv("FV_DATASETS_ROOT", str(tmp_path / "no-external"))
    from fv import settings
    root = settings.local_sources_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in
            (out / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_reduces_images_and_records_the_measured_scale(sources):
    make_rich_source(sources, "src")
    out = sources / "small"

    meta = resize_source(ResizeConfig(source="local/src", width=24), out)

    assert meta["derived"]["size"] == [24, 18]
    assert meta["derived"]["scale"] == [24 / W, 18 / H]
    assert meta["derived"]["resample"] == "lanczos"
    with Image.open(out / "images/000000.png") as img:
        assert img.size == (24, 18)
    assert _read(out)[0]["labels"]["width"] == 24


def test_scale_comes_from_the_output_not_from_the_request(sources):
    """35 -> 18 rounds up: 18/35 != 24/48, and the quads must follow the truth."""
    make_rich_source(sources, "src", w=48, h=35)
    out = sources / "small"

    meta = resize_source(ResizeConfig(source="local/src", width=24), out)

    sx, sy = meta["derived"]["scale"]
    assert meta["derived"]["size"] == [24, 18]
    assert sx == 0.5 and sy == 18 / 35   # two different scales, both measured
    assert sy != 0.5
    quad = _read(out)[0]["labels"]["blocks"][0]["quad"]
    assert quad[0] == [round(4 * sx, 2), round(6 * sy, 2)]


def test_every_nested_level_is_rescaled(sources):
    make_rich_source(sources, "src")
    out = sources / "small"

    resize_source(ResizeConfig(source="local/src", width=24), out)

    labels = _read(out)[0]["labels"]
    for level in ("blocks", "lines", "words"):
        entry = labels[level][0]
        assert entry["box"][0] == 2.0, f"{level} box kept the old resolution"
        assert entry["quad"][0] == [2.0, 3.0], f"{level} quad kept the old resolution"


def test_masks_stay_binary(sources):
    """NEAREST or the mask grows greys, which are classes nobody declared."""
    make_rich_source(sources, "src")
    out = sources / "small"

    resize_source(ResizeConfig(source="local/src", width=24), out)

    with Image.open(out / "masks/000000.png") as m:
        values = set(np.asarray(m).flatten().tolist())
    assert values <= {0, 255}, f"la mascara se interpolo: {sorted(values)}"


def test_fields_we_do_not_consume_survive(sources):
    make_rich_source(sources, "src")
    out = sources / "small"

    resize_source(ResizeConfig(source="local/src", width=24), out)

    labels = _read(out)[0]["labels"]
    assert labels["image_id"] == "src/000000"
    assert labels["has_overlap"] is False
    assert labels["blocks"][0]["text"] == "hola mundo"
    assert labels["blocks"][0]["kind"] == "paragraph"


def test_provenance_is_addressable_and_holdout_survives(sources):
    make_rich_source(sources, "src", extra_meta={"holdout": True, "recipe_id": "r-1"})
    out = sources / "small"

    meta = resize_source(ResizeConfig(source="local/src", width=24), out)

    assert meta["derived"]["from"] == "local/src"      # addressable, not declared
    assert meta["derived"]["from_declared_id"] == "src"
    assert meta["derived"]["request"] == {"width": 24}
    assert meta["holdout"] is True                      # must not be lost by reducing
    assert meta["recipe_id"] == "r-1"
    assert meta["id"] == "small" and meta["count"] == 4


def test_the_result_is_a_usable_source(sources):
    make_rich_source(sources, "src")
    resize_source(ResizeConfig(source="local/src", width=24), sources / "small")

    ids = {s["id"] for s in discover_sources()}
    assert "local/small" in ids

    ds = SourceDataset("local/small")
    sample = ds.sample_at(0)
    assert (sample.width, sample.height) == (24, 18)
    assert sample.load_image().shape == (18, 24)
    assert sample.blocks[0].bbox == (2.0, 3.0, 12.0, 9.0)


def test_upscale_is_refused_before_writing(sources):
    make_rich_source(sources, "src")
    out = sources / "big"

    with pytest.raises(ResizeError) as e:
        resize_source(ResizeConfig(source="local/src", width=96), out)

    assert e.value.code == "upscale_not_allowed"
    assert not out.exists()


def test_exactly_one_dimension(sources):
    make_rich_source(sources, "src")

    for cfg in (ResizeConfig(source="local/src"),
                ResizeConfig(source="local/src", width=24, height=18)):
        with pytest.raises(ResizeError) as e:
            resize_source(cfg, sources / "small")
        assert e.value.code == "resize_needs_one_dimension"


def test_height_drives_the_width_too(sources):
    make_rich_source(sources, "src")

    meta = resize_source(ResizeConfig(source="local/src", height=18), sources / "small")

    assert meta["derived"]["size"] == [24, 18]
    assert meta["derived"]["request"] == {"height": 18}


def test_mixed_sizes_are_refused(sources):
    make_rich_source(sources, "src", count=2)
    records = _read(sources / "src")
    records[1]["labels"]["width"] = 64
    (sources / "src" / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    with pytest.raises(ResizeError) as e:
        resize_source(ResizeConfig(source="local/src", width=24), sources / "small")
    assert e.value.code == "mixed_source_sizes"


def test_existing_destination_is_never_overwritten(sources):
    make_rich_source(sources, "src")
    out = sources / "small"
    resize_source(ResizeConfig(source="local/src", width=24), out)

    with pytest.raises(ResizeError) as e:
        resize_source(ResizeConfig(source="local/src", width=12), out)
    assert e.value.code == "destination_exists"


def test_unknown_resample_is_refused(sources):
    make_rich_source(sources, "src")

    with pytest.raises(ResizeError) as e:
        resize_source(ResizeConfig(source="local/src", width=24, resample="magic"),
                      sources / "small")
    assert e.value.code == "unknown_resample"
