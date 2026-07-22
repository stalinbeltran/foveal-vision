"""F — apply a run to a full image: sliding fovea window -> corner detections
-> NMS -> greedy TL->BR reconstruction. Returns ALL stages (without the raw
one, "the paragraph came out wrong" is not diagnosable).

The foveated view comes from THE SAME fv.fovea.build_view the dataloader uses
(contract (5)). Knobs (threshold, stride, nms_radius, min_size) are F, not D:
post-hoc, in units of the labelled window, echoed back in the payload.
"""

from __future__ import annotations

import numpy as np
import torch

from fv.fovea import build_view
from fv.metrics import CORNER_NAMES


def _positions(length: int, n: int, stride: int) -> list[int]:
    if length < n:
        return []
    xs = list(range(0, length - n + 1, stride))
    if xs[-1] != length - n:
        xs.append(length - n)
    return xs


def predict_image(model, image: np.ndarray, threshold: float = 0.5,
                  stride: int | None = None, nms_radius: float | None = None,
                  min_size: float | None = None) -> dict:
    dims = model.dims
    n = dims.center_out                      # the labelled window = the fovea
    stride = stride if stride else max(1, n // 2)
    nms_radius = nms_radius if nms_radius is not None else n / 2
    min_size = min_size if min_size is not None else 4.0
    H, W = image.shape

    xs = _positions(W, n, stride)
    ys = _positions(H, n, stride)
    views, origins = [], []
    for wy0 in ys:
        for wx0 in xs:
            v, _cov = build_view(image, wx0, wy0, dims,
                                 pool_mode=model.cfg["pool_mode"],
                                 pad_mode=model.cfg["pad_mode"])
            views.append(v)
            origins.append((wx0, wy0))
    raw = []
    if views:
        batch = torch.from_numpy(np.stack(views)).unsqueeze(1)
        with torch.no_grad():
            out = model(batch).numpy()
        scores = 1.0 / (1.0 + np.exp(-out[:, :, 0]))
        for i, (wx0, wy0) in enumerate(origins):
            for ci in range(4):
                s = float(scores[i, ci])
                if s >= threshold:
                    cx = wx0 + float(out[i, ci, 1]) * n
                    cy = wy0 + float(out[i, ci, 2]) * n
                    raw.append({"corner": CORNER_NAMES[ci], "score": s,
                                "x": round(cx, 2), "y": round(cy, 2),
                                "window": [wx0, wy0]})

    corners = _nms(raw, nms_radius)
    paragraphs = _reconstruct(corners, min_size)
    return {"raw": raw, "corners": corners, "paragraphs": paragraphs,
            "image_size": [W, H],
            "knobs": {"threshold": threshold, "stride": stride,
                      "nms_radius": nms_radius, "min_size": min_size,
                      "window_size": n}}


def _nms(dets: list[dict], radius: float) -> list[dict]:
    out = []
    for cname in CORNER_NAMES:
        group = sorted((d for d in dets if d["corner"] == cname),
                       key=lambda d: -d["score"])
        kept: list[dict] = []
        for d in group:
            if all((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 > radius ** 2
                   for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def _reconstruct(corners: list[dict], min_size: float) -> list[dict]:
    """Greedy TL->BR pairing (inherited heuristic — the place to touch if
    paragraphs come out wrong while corners come out right)."""
    tls = sorted((c for c in corners if c["corner"] == "TL"), key=lambda c: -c["score"])
    brs = [c for c in corners if c["corner"] == "BR"]
    used: set[int] = set()
    boxes = []
    for tl in tls:
        best, best_j = None, -1
        for j, br in enumerate(brs):
            if j in used:
                continue
            if br["x"] - tl["x"] >= min_size and br["y"] - tl["y"] >= min_size:
                score = tl["score"] * br["score"]
                if best is None or score > best:
                    best, best_j = score, j
        if best_j >= 0:
            used.add(best_j)
            br = brs[best_j]
            boxes.append({"x0": tl["x"], "y0": tl["y"], "x1": br["x"], "y1": br["y"],
                          "score": round(best, 4)})
    return boxes
