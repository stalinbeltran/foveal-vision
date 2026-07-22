"""Tiny synthetic worlds built per test (tests.md: never touch real data/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_source(root: Path, name: str, count: int = 10, W: int = 48, H: int = 36,
                seed: int = 3) -> None:
    out = root / name
    (out / "images").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(count):
        img = np.full((H, W), 230, dtype=np.uint8)
        pw = int(rng.integers(14, 24))
        ph = int(rng.integers(10, 16))
        x0 = int(rng.integers(2, W - pw - 2))
        y0 = int(rng.integers(2, H - ph - 2))
        yy = y0
        while yy + 2 <= y0 + ph:
            img[yy:yy + 2, x0:x0 + pw] = 40
            yy += 4
        rel = f"images/{i:06d}.png"
        Image.fromarray(img).save(out / rel)
        quad = [[float(x0), float(y0)], [float(x0 + pw), float(y0)],
                [float(x0 + pw), float(y0 + ph)], [float(x0), float(y0 + ph)]]
        lines.append(json.dumps({
            "index": i, "image": rel,
            "labels": {"width": W, "height": H,
                       "blocks": [{"block_id": "b0", "kind": "paragraph",
                                   "angle": 0.0, "quad": quad}]}}))
    (out / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "dataset.json").write_text(json.dumps({"id": name, "count": count}),
                                      encoding="utf-8")


TINY_NET = {"N": 12, "c_frac": 0.667, "d": 2, "pen_frac": 0.1,
            "k_center": 3, "k_periph": 3, "s_center": 1, "s_periph": 1,
            "ch1": 4, "ch2": 8, "merge": "concat", "pool_mode": "avg",
            "pad_mode": "edge"}  # fovea (center_out) = 8


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """An isolated project root with one source and one extracted dataset."""
    monkeypatch.setenv("FV_ROOT", str(tmp_path))
    monkeypatch.setenv("FV_DATASETS_ROOT", str(tmp_path / "no-external"))
    from fv import settings
    make_source(settings.local_sources_root(), "mini", count=10)

    from fv.windows.extract import ExtractConfig, extract_windows
    cfg = ExtractConfig(source="local/mini", window_size=8, stride=6,
                        val_frac=0.2, test_frac=0.2, seed=1)
    manifest = extract_windows(cfg, settings.window_datasets_root() / "mini-b8")
    return {"root": tmp_path, "source": "local/mini",
            "dataset": "mini-b8", "manifest": manifest}
