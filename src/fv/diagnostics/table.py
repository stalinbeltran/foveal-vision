"""E x B: the per-window table — A CACHE, not an entity (inherited D1).

Pure function of (run, fingerprint of B, split, checkpoint mtime): it is
recomputed exactly, so it is not named, not listed, and deleting it loses
nothing. The key carries the checkpoint mtime — a live run rewrites best.pt
every epoch it improves. threshold is a QUERY parameter, never part of the
cache key: re-thresholding reads stored scores, which is what makes the
threshold sweep free (V8).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from fv import settings
from fv.fovea import derive_dims
from fv.inference.checkpoint import MODEL_CACHE
from fv.metrics import (corner_evidence, corner_scores, detection_counts,
                        per_window_errors)
from fv.training.registry import RunError, RunStore
from fv.windows.dataset import FoveatedWindowDataset
from fv.windows.store import WindowDatasetStore

SPLITS = {"train": 0, "val": 1, "test": 2}


def _cache_key(run: str, fingerprint: str, split: str, ckpt: Path) -> str:
    h = hashlib.sha256(
        f"{run}|{fingerprint}|{split}|{ckpt.stat().st_mtime_ns}".encode()).hexdigest()
    return h[:24]


def diagnostics_table(run_name: str, split: str = "val",
                      store: RunStore | None = None) -> dict:
    store = store or RunStore()
    cfg = store.config(run_name)
    prov = cfg.get("provenance") or {}
    if not prov.get("window_dataset", {}).get("name"):
        raise RunError("run_without_provenance",
                       f"'{run_name}' no tiene procedencia: no puede decir de que "
                       f"dataset salio, asi que no hay contra que diagnosticarlo",
                       "borralo y reentrenalo: no es comparable con nada")
    ds_name = prov["window_dataset"]["name"]
    wstore = WindowDatasetStore()
    manifest = wstore.manifest(ds_name)
    if manifest["fingerprint"] != prov["window_dataset"]["fingerprint"]:
        raise RunError("window_dataset_changed",
                       f"'{ds_name}' se reconstruyo desde que se entreno '{run_name}': "
                       f"su split ya no es el que ese best.pt uso para elegirse",
                       "reentrena contra el dataset actual: los numeros saldrian con "
                       "buena cara y medirian otra cosa")
    ckpt = store.path(run_name) / "best.pt"
    if not ckpt.exists():
        raise RunError("run_has_no_checkpoint",
                       f"'{run_name}' no tiene best.pt todavia",
                       "espera a que termine al menos una epoca")
    if split not in SPLITS:
        raise RunError("unknown_split", f"split '{split}' no existe",
                       "usa train, val o test")

    cache_dir = settings.cache_root() / "diagnostics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(run_name, manifest["fingerprint"], split, ckpt)
    cache_file = cache_dir / f"{key}.npz"
    window_size = int(manifest["config"]["window_size"])

    if cache_file.exists():
        data = np.load(cache_file)
        scores, xy_pred, y_true = data["scores"], data["xy_pred"], data["y_true"]
        window_idx = data["window_idx"]
    else:
        model = MODEL_CACHE.get(ckpt)
        net = cfg["network"]
        dims = derive_dims(net["N"], net["c_frac"], net["d"], net["pen_frac"])
        arrays = wstore.arrays(ds_name)
        ds = FoveatedWindowDataset(arrays, dims, split=SPLITS[split],
                                   pool_mode=net["pool_mode"],
                                   pad_mode=net["pad_mode"])
        if len(ds) == 0:
            raise RunError("split_empty", f"el split '{split}' esta vacio",
                           "reconstruye el dataset con ese split > 0")
        loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=0)
        logits_all, y_all = [], []
        with torch.no_grad():
            for x, y in loader:
                logits_all.append(model(x).numpy())
                y_all.append(y.numpy())
        logits = np.concatenate(logits_all)
        y_true = np.concatenate(y_all)
        scores = corner_scores(logits).astype(np.float32)
        xy_pred = logits[:, :, 1:].astype(np.float32)
        window_idx = np.flatnonzero(arrays["split"] == SPLITS[split]).astype(np.int32)
        np.savez_compressed(cache_file, scores=scores, xy_pred=xy_pred,
                            y_true=y_true, window_idx=window_idx)

    err = per_window_errors(xy_pred, y_true[:, :, 1:], y_true[:, :, 0], window_size)
    return {"run": run_name, "split": split, "window_dataset": ds_name,
            "window_size": window_size, "scores": scores, "xy_pred": xy_pred,
            "y_true": y_true, "err_px": err, "window_idx": window_idx}


def summary_payload(table: dict, threshold: float = 0.5) -> dict:
    scores, y_true, err = table["scores"], table["y_true"], table["err_px"]
    det = detection_counts(scores, y_true[:, :, 0], threshold)
    ev = corner_evidence(y_true)
    blind = ev < 0.05
    visible = ev >= 0.05
    with np.errstate(invalid="ignore"):
        err_all = float(np.nanmean(err)) if np.isfinite(err).any() else None
        err_blind = float(np.nanmean(err[blind & np.isfinite(err)])) \
            if (blind & np.isfinite(err)).any() else None
        err_visible = float(np.nanmean(err[visible & np.isfinite(err)])) \
            if (visible & np.isfinite(err)).any() else None
    positives = int((y_true[:, :, 0] >= 0.5).sum())
    return {
        "windows": int(scores.shape[0]), "positives": positives,
        "threshold": threshold, "detection": det,
        "pos_err_px": err_all,
        "pos_err_px_blind": err_blind,       # evidence < 0.05: paragraph outside the fovea
        "pos_err_px_visible": err_visible,
        "blind_share": float(np.nansum(blind) / max(1, positives)),
    }


def worst_windows(table: dict, threshold: float = 0.5, limit: int = 24,
                  offset: int = 0, outcome: str | None = None) -> dict:
    """Gallery worst-first: rank windows by max position error / misdetection."""
    scores, y_true, err = table["scores"], table["y_true"], table["err_px"]
    pred = scores >= threshold
    true = y_true[:, :, 0] >= 0.5
    fp = pred & ~true
    fn = ~pred & true
    err_filled = np.where(np.isfinite(err), err, 0.0)
    badness = err_filled.max(axis=1) + 10.0 * fp.any(axis=1) + 10.0 * fn.any(axis=1)
    order = np.argsort(-badness)
    if outcome == "fp":
        order = [i for i in order if fp[i].any()]
    elif outcome == "fn":
        order = [i for i in order if fn[i].any()]
    total = len(order)
    sel = list(order[offset:offset + limit])
    items = []
    for i in sel:
        items.append({
            "row": int(i),
            "window_idx": int(table["window_idx"][i]),
            "scores": [round(float(s), 4) for s in scores[i]],
            "xy_pred": [[round(float(v), 4) for v in p] for p in table["xy_pred"][i]],
            "y_true": [[round(float(v), 4) for v in p] for p in y_true[i]],
            "err_px": [None if not np.isfinite(e) else round(float(e), 2)
                       for e in err[i]],
        })
    return {"total": total, "offset": offset, "items": items}
