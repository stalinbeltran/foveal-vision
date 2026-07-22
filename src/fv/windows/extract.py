"""B — extract labelled windows from a source.

The labelled window IS the fovea (F1b): corners are labelled against a
window_size x window_size grid window; the foveated view (context around it)
is built later, in the dataloader, from the stored full images — so the whole
foveated geometry is sweepable without re-extracting (decision C1/D23).

windows.npz arrays (formatos.md §4.1): y (N,4,3), sample_idx, window_xy,
split, images (S,H,W), images_sample_idx. No baked X.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fv.datasets.loader import SourceDataset
from fv.metrics import CORNER_NAMES
from fv.ioutils import write_json_atomic

IMAGES_BUDGET_BYTES = 1_000_000_000


class ExtractError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass
class ExtractConfig:
    source: str
    window_size: int = 16
    stride: int = 8
    target_kinds: tuple = ("paragraph",)   # paragraphs today; lines/words later
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1                          # the SPLIT seed (per image), not the training seed


def _positions(length: int, n: int, stride: int) -> list[int]:
    """Grid positions covering [0, length): stride steps, last window flush."""
    if length < n:
        return []
    xs = list(range(0, length - n + 1, stride))
    if xs[-1] != length - n:
        xs.append(length - n)
    return xs


def _assign_splits(num_samples: int, val_frac: float, test_frac: float,
                   seed: int) -> np.ndarray:
    """Split BY IMAGE (never by window: windows of one image are correlated)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_samples)
    n_val = int(round(num_samples * val_frac))
    n_test = int(round(num_samples * test_frac))
    split = np.zeros(num_samples, dtype=np.int8)
    split[order[:n_val]] = 1
    split[order[n_val:n_val + n_test]] = 2
    return split


def _corners_of(bbox: tuple) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # TL TR BR BL


def extract_windows(cfg: ExtractConfig, out_dir: Path,
                    progress=None, should_stop=None) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ExtractError("window_dataset_exists",
                           f"ya existe un dataset en {out_dir}",
                           "elige otro nombre o borralo primero: no se sobrescribe nunca")
    ds = SourceDataset(cfg.source)
    samples = ds.samples()
    if not samples:
        raise ExtractError("source_empty", f"la fuente '{cfg.source}' no tiene imagenes",
                           "elige otra fuente")

    sizes = {(s.width, s.height) for s in samples}
    if len(sizes) > 1:
        raise ExtractError("images_not_uniform",
                           f"la fuente tiene imagenes de tamanos distintos: {sorted(sizes)}",
                           "usa una fuente uniforme (o redimensionala)")
    W, H = samples[0].width, samples[0].height
    budget = len(samples) * W * H
    if budget > IMAGES_BUDGET_BYTES:
        raise ExtractError(
            "images_budget_exceeded",
            f"guardar las imagenes costaria {budget / 1e9:.2f} GB (> 1 GB)",
            "reduce la fuente o baja el numero de imagenes: no hay camino degradado")

    n = cfg.window_size
    split_by_image = _assign_splits(len(samples), cfg.val_frac, cfg.test_frac, cfg.seed)

    ys, sample_idxs, window_xys, splits = [], [], [], []
    images = np.zeros((len(samples), H, W), dtype=np.uint8)

    for si, s in enumerate(samples):
        if should_stop and should_stop():
            raise ExtractError("cancelled", "extraccion cancelada", "relanzala")
        images[si] = s.load_image()
        corners_by_type: list[list[tuple[float, float]]] = [[], [], [], []]
        for b in s.blocks:
            if b.kind not in cfg.target_kinds:
                continue
            for ci, pt in enumerate(_corners_of(b.bbox)):
                corners_by_type[ci].append(pt)
        for wy0 in _positions(H, n, cfg.stride):
            for wx0 in _positions(W, n, cfg.stride):
                y = np.zeros((4, 3), dtype=np.float32)
                for ci in range(4):
                    inside = [(cx, cy) for cx, cy in corners_by_type[ci]
                              if wx0 <= cx < wx0 + n and wy0 <= cy < wy0 + n]
                    if inside:
                        # tie-break: nearest to the window centre
                        ccx, ccy = wx0 + n / 2, wy0 + n / 2
                        cx, cy = min(inside, key=lambda p: (p[0] - ccx) ** 2 + (p[1] - ccy) ** 2)
                        y[ci] = (1.0, (cx - wx0) / n, (cy - wy0) / n)
                ys.append(y)
                sample_idxs.append(s.index)
                window_xys.append((wx0, wy0))
                splits.append(split_by_image[si])
        if progress:
            progress(si + 1, len(samples))

    y_arr = np.stack(ys).astype(np.float32)
    sample_idx = np.asarray(sample_idxs, dtype=np.int32)
    window_xy = np.asarray(window_xys, dtype=np.int32)
    split = np.asarray(splits, dtype=np.int8)
    images_sample_idx = np.asarray([s.index for s in samples], dtype=np.int32)

    out_dir.mkdir(parents=True)
    npz_path = out_dir / "windows.npz"
    np.savez_compressed(npz_path, y=y_arr, sample_idx=sample_idx,
                        window_xy=window_xy, split=split,
                        images=images, images_sample_idx=images_sample_idx)

    fingerprint = "sha256:" + hashlib.sha256(npz_path.read_bytes()).hexdigest()
    per_split = {name: int((split == i).sum()) for i, name in
                 enumerate(("train", "val", "test"))}
    positives = {c: int((y_arr[:, i, 0] >= 0.5).sum())
                 for i, c in enumerate(CORNER_NAMES)}
    manifest = {
        "format_version": 1,
        "fingerprint": fingerprint,
        "has_images": True,
        "images": {"shape": [len(samples), H, W], "bytes": int(budget),
                   "budget_bytes": IMAGES_BUDGET_BYTES},
        "source_id": cfg.source,
        "config": asdict(cfg),
        "num_samples": len(samples),
        "num_windows": int(y_arr.shape[0]),
        "corner_order": list(CORNER_NAMES),
        "label_window": "center",  # F1b: x,y normalised against the fovea window
        "windows_per_split": per_split,
        "positives_per_corner": positives,
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    split_json = {name: [int(s.index) for si2, s in enumerate(samples)
                         if split_by_image[si2] == i]
                  for i, name in enumerate(("train", "val", "test"))}
    write_json_atomic(out_dir / "split.json", split_json)
    return manifest
