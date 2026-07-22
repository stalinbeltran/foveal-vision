"""What every number means — one site, every reader (contract (5) analogue).

Pure arrays: imports nothing from fv. Both the training loop (per-epoch val
metrics) and the per-window diagnostics table call these; a metric defined
twice is two copies that must agree with nothing checking it.

All position errors are reported in PIXELS OF THE LABELLED WINDOW (the fovea,
F1b) — which is fixed for a given B, so metrics stay comparable across a sweep
of the foveated geometry (contract (9) extension).
"""

from __future__ import annotations

import numpy as np

CORNER_NAMES = ("TL", "TR", "BR", "BL")
NUM_CORNERS = 4


def corner_scores(logits: np.ndarray) -> np.ndarray:
    """sigmoid(exists) — logits (N, 4, 3) -> scores (N, 4)."""
    return 1.0 / (1.0 + np.exp(-logits[:, :, 0]))


def detection_counts(scores: np.ndarray, exists_true: np.ndarray,
                     threshold: float = 0.5) -> dict:
    pred = scores >= threshold
    true = exists_true >= 0.5
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def pos_err_px(xy_pred: np.ndarray, xy_true: np.ndarray, exists_true: np.ndarray,
               window_size: int) -> float | None:
    """Mean euclidean error in window pixels over TRUE corners; None if none.

    None, never 0: absent is not zero (formatos.md §2).
    """
    mask = exists_true >= 0.5
    if not mask.any():
        return None
    d = (xy_pred - xy_true) * float(window_size)
    err = np.sqrt((d ** 2).sum(axis=-1))
    return float(err[mask].mean())


def per_window_errors(xy_pred: np.ndarray, xy_true: np.ndarray,
                      exists_true: np.ndarray, window_size: int) -> np.ndarray:
    """(N, 4) error px; NaN where there is no true corner (never 0)."""
    d = (xy_pred - xy_true) * float(window_size)
    err = np.sqrt((d ** 2).sum(axis=-1))
    err = err.astype(np.float32)
    err[exists_true < 0.5] = np.nan
    return err


def corner_evidence(y: np.ndarray) -> np.ndarray:
    """Fraction of the labelled window the corner's paragraph CAN occupy.

    y: (N, 4, 3) [exists, x, y] normalised to the labelled window. Directional
    by corner type: a TL at (fx, fy) has its body to the right and below ->
    (1-fx)(1-fy); TR -> fx(1-fy); BR -> fx*fy; BL -> (1-fx)fy.
    Geometric, no pixels, no model. Frozen against the labelled window: it is
    NOT redefined against the field of view (lesson R-b of the sibling's P4).
    NaN where the corner does not exist.
    """
    fx = y[:, :, 1]
    fy = y[:, :, 2]
    ev = np.stack([
        (1 - fx[:, 0]) * (1 - fy[:, 0]),
        fx[:, 1] * (1 - fy[:, 1]),
        fx[:, 2] * fy[:, 2],
        (1 - fx[:, 3]) * fy[:, 3],
    ], axis=1).astype(np.float32)
    ev[y[:, :, 0] < 0.5] = np.nan
    return ev


# ---------------------------------------------------------------------------
# Paragraph-level metric (protocolo.md §2): measured per IMAGE in pixels of the
# ORIGINAL image — the property that lets one holdout serve any geometry.

def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def paragraph_f1(pred_boxes: list, true_boxes: list, iou_threshold: float = 0.5) -> dict:
    """Greedy IoU matching of predicted boxes against ground truth bboxes."""
    matched_true: set[int] = set()
    tp, ious = 0, []
    for pb in pred_boxes:
        best, best_j = 0.0, -1
        for j, tb in enumerate(true_boxes):
            if j in matched_true:
                continue
            v = _iou(tuple(pb), tuple(tb))
            if v > best:
                best, best_j = v, j
        if best >= iou_threshold and best_j >= 0:
            matched_true.add(best_j)
            tp += 1
            ious.append(best)
    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "mean_iou": float(np.mean(ious)) if ious else None}
