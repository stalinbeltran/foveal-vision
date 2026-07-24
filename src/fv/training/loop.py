"""The training loop: B + C + D (+ X aside) -> E.

Every gate calls fv.validation.check_run BEFORE RunStore.create; train() calls
it again as the safety net (the CLI does not pass through the API). A dataset
without val refuses to train (choosing best.pt by train loss in silence is the
measured trap). Cooperative stop at epoch end. Reproducible: same seed + same
config => same weights (tested with a control).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fv import settings
from fv.fovea import derive_dims
from fv.ioutils import write_json_atomic
from fv.metrics import corner_scores, detection_counts, pos_err_px
from fv.models.builder import build_model, full_config
from fv.training.losses import corner_loss
from fv.training.recipe import Recipe
from fv.training.registry import RunError, RunStore, environment, git_commit
from fv.validation import check_run
from fv.windows.dataset import FoveatedWindowDataset
from fv.windows.store import WindowDatasetStore


def make_optimizer(model, recipe: Recipe):
    if recipe.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=recipe.lr,
                                weight_decay=recipe.weight_decay)
    if recipe.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=recipe.lr,
                                 weight_decay=recipe.weight_decay)
    if recipe.optimizer == "sgd":
        # momentum EXPLICIT: the default 0 silently rigs any optimizer sweep
        return torch.optim.SGD(model.parameters(), lr=recipe.lr,
                               momentum=recipe.momentum,
                               weight_decay=recipe.weight_decay)
    raise RunError("unknown_optimizer", f"optimizer '{recipe.optimizer}' no existe",
                   "usa adam, adamw o sgd")


def evaluate(model, loader, recipe: Recipe, window_size: int, device: str) -> dict:
    model.eval()
    losses, all_logits, all_targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(float(corner_loss(
                logits, y, recipe.lambda_pos, recipe.pos_weight,
                recipe.smooth_l1_beta)))
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    scores = corner_scores(logits)
    det = detection_counts(scores, targets[:, :, 0])
    err = pos_err_px(logits[:, :, 1:], targets[:, :, 1:], targets[:, :, 0], window_size)
    return {"loss": float(np.mean(losses)) if losses else None,
            "f1": det["f1"], "precision": det["precision"], "recall": det["recall"],
            "pos_err_px": err}


def train(run_name: str, window_dataset: str, network_name: str, network_cfg: dict,
          recipe_name: str, recipe: Recipe, device: str = "cpu",
          sweep: str | None = None, store: RunStore | None = None,
          dataset_root: Path | None = None, progress=None, should_stop=None) -> dict:
    store = store or RunStore()
    wstore = WindowDatasetStore(dataset_root)
    manifest = wstore.manifest(window_dataset)
    net = full_config(network_cfg)

    problems = check_run(manifest, net)
    if problems:
        raise RunError(problems[0]["code"],
                       problems[0]["message"], problems[0]["hint"])

    config = {
        "format_version": 1,
        "recipe": recipe.as_dict(),
        "network": net,
        "execution": {"device": device, "num_workers": 0},  # X, outside D's identity
        "provenance": {
            "window_dataset": {"name": window_dataset,
                               "fingerprint": manifest["fingerprint"]},
            "network": {"name": network_name, "value": net},
            "recipe": {"name": recipe_name, "value": recipe.as_dict()},
            "sweep": sweep,
            "git_commit": git_commit(settings.project_root()),
            "environment": environment(device),
        },
    }
    run_dir = store.create(run_name, config)  # refuses if the name exists

    try:
        return _train_inner(run_name, run_dir, manifest, net, recipe, device,
                            store, wstore, window_dataset, progress, should_stop)
    except Exception:
        store.set_status(run_name, "error")
        raise


def _train_inner(run_name, run_dir: Path, manifest, net, recipe: Recipe,
                 device, store: RunStore, wstore, window_dataset, progress,
                 should_stop=None) -> dict:
    torch.manual_seed(recipe.seed)
    np.random.seed(recipe.seed % (2 ** 32))

    dims = derive_dims(net["N"], net["c_frac"], net["d"], net["pen_frac"])
    arrays = wstore.arrays(window_dataset)
    train_ds = FoveatedWindowDataset(arrays, dims, split=0,
                                     pool_mode=net["pool_mode"], pad_mode=net["pad_mode"])
    val_ds = FoveatedWindowDataset(arrays, dims, split=1,
                                   pool_mode=net["pool_mode"], pad_mode=net["pad_mode"])
    g = torch.Generator()
    g.manual_seed(recipe.seed)
    train_loader = DataLoader(train_ds, batch_size=recipe.batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=0)

    model = build_model(net).to(device)
    opt = make_optimizer(model, recipe)
    sched = None
    if recipe.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe.epochs)
    elif recipe.scheduler != "none":
        raise RunError("unknown_scheduler", f"scheduler '{recipe.scheduler}' no existe",
                       "usa none o cosine")

    window_size = int(manifest["config"]["window_size"])
    metrics_path = run_dir / "metrics.jsonl"
    best_value = None
    best_epoch = None
    epochs_run = 0
    cancelled = False
    stopped_early = False
    no_improve = 0
    seconds = []

    store.set_status(run_name, "running", epoch=0, pid=os.getpid())
    for epoch in range(1, recipe.epochs + 1):
        t0 = time.monotonic()
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = corner_loss(model(x), y, recipe.lambda_pos, recipe.pos_weight,
                               recipe.smooth_l1_beta)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        if sched:
            sched.step()
        val = evaluate(model, val_loader, recipe, window_size, device)
        secs = time.monotonic() - t0
        seconds.append(secs)
        epochs_run = epoch

        rec = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)),
               "val": val, "lr": float(opt.param_groups[0]["lr"]),
               "seconds": round(secs, 3)}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        monitor_value = val["loss"] if recipe.monitor == "val_loss" else val.get(
            recipe.monitor.removeprefix("val_"))
        ckpt = {"model": model.state_dict(), "config": {"model": net}, "epoch": epoch}
        torch.save(ckpt, run_dir / "last.pt")
        improved = monitor_value is not None and (
            best_value is None or
            (monitor_value > best_value if recipe.monitor in ("val_f1",)
             else monitor_value < best_value))
        if improved:
            best_value, best_epoch = monitor_value, epoch
            torch.save(ckpt, run_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1

        store.set_status(run_name, "running", epoch=epoch, pid=os.getpid())
        if progress:
            progress(epoch, recipe.epochs, rec)
        # cooperative stop: the run's own stop file OR the sweep asking its
        # in-flight point to stop (should_stop) — both cut at the epoch boundary
        if store.stop_requested(run_name) or (should_stop and should_stop()):
            cancelled = True
            break
        if recipe.patience and no_improve >= recipe.patience:
            stopped_early = True
            break

    summary = {
        "run": run_name,
        "epochs_run": epochs_run,
        "epochs_requested": recipe.epochs,
        "stopped_early": stopped_early,
        "cancelled": cancelled,
        "monitor": recipe.monitor,
        "best": best_value,           # null if the monitor never measured — NEVER ±inf
        "best_epoch": best_epoch,
        "seconds_per_epoch": round(float(np.mean(seconds)), 3) if seconds else None,
        "corner_order": list(manifest["corner_order"]),
    }
    write_json_atomic(run_dir / "summary.json", summary)
    store.set_status(run_name, "cancelled" if cancelled else "done", epoch=epochs_run)
    return summary
