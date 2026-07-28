"""What every number means — one site, every reader (contract (5) analogue).

Pure arrays: imports nothing from fv. Both the training loop (per-epoch val
metrics) and the per-window diagnostics table call these; a metric defined
twice is two copies that must agree with nothing checking it.

All position errors are reported in PIXELS OF THE LABELLED WINDOW (the fovea,
F1b) — which is fixed for a given B, so metrics stay comparable across a sweep
of the foveated geometry (contract (9) extension).
"""

from __future__ import annotations

import math

import numpy as np

CORNER_NAMES = ("TL", "TR", "BR", "BL")
NUM_CORNERS = 4

# ---------------------------------------------------------------------------
# Which epoch the monitor keeps. ONE definition, two readers: the training loop
# asks `monitor_improved` when to overwrite best.pt, and the sweep ranking asks
# `checkpoint_record` WHICH epoch that file came from. Written twice they drift,
# and the ranking starts describing weights that are not on disk.

# The metrics of a `val` record that can RANK (a subset: the record also carries
# precision and recall, which nobody optimises), and which way is better. It is
# the ONE place the direction lives: H reads it as OBJECTIVES (fv.sweeps.spec)
# and D names one of MONITORS to choose best.pt. It used to be written twice —
# a set {"val_f1"} here and a dict in the sweep spec — and the halves disagreed
# about names: a recipe saying `monitor: "f1"` (the bare objective, which the UI
# offered) found the value but not the direction, so best.pt kept the epoch with
# the WORST f1, silently. Hence MONITORS: a monitor is 'val_' + a val metric.
VAL_METRICS = {"f1": "max", "pos_err_px": "min", "loss": "min"}
MONITORS = tuple(f"val_{k}" for k in VAL_METRICS)


def monitor_key(monitor: str) -> str:
    """The key inside a metrics record's `val` dict: 'val_loss' -> 'loss'."""
    return monitor[4:] if monitor.startswith("val_") else monitor


def monitor_improved(value, best, monitor: str) -> bool:
    """Strictly better, so the FIRST epoch of a tie keeps the checkpoint."""
    if value is None:
        return False
    if best is None:
        return True
    higher = VAL_METRICS.get(monitor_key(monitor)) == "max"
    return value > best if higher else value < best


def checkpoint_record(records: list, monitor: str) -> dict | None:
    """The metrics record of the epoch whose weights `best.pt` holds.

    None when the monitor never measured — that run has NO best.pt (the loop
    never wrote one), so there is nothing to describe. None, never the last
    epoch as a consolation: absent is not a fallback (formatos.md §2).
    """
    best, best_rec = None, None
    for r in records:
        v = (r.get("val") or {}).get(monitor_key(monitor))
        if monitor_improved(v, best, monitor):
            best, best_rec = v, r
    return best_rec


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


def _mean_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n with TIES SHARING THE MEAN rank — the standard convention, and
    the one that keeps the two series comparable when only one of them has ties."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0   # mean of ranks i+1..j+1
        i = j + 1
    return ranks


def spearman(a, b) -> float | None:
    """Rank correlation between two series (metrica-de-tarea.md §5.2).

    Lives here and not in a script because this is THE site where "what each
    number means" is defined: a Spearman computed inside a one-off script is a
    number nobody can test, and this project has already been burnt by defining
    a number twice.

    Ties take the MEAN rank. Returns None — never 0 — when either series is
    constant or has fewer than two points: there the correlation is undefined,
    and 0 would read as «they do not correlate», which is a different claim
    (formatos.md §2: absent is not zero).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"spearman necesita dos series 1-D del mismo largo, "
                         f"llegaron {x.shape} y {y.shape}")
    if len(x) < 2:
        return None
    rx, ry = _mean_ranks(x), _mean_ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((dx ** 2).sum()) * float((dy ** 2).sum()))
    if den == 0.0:                       # a constant series has no ranking
        return None
    return float((dx * dy).sum() / den)


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
