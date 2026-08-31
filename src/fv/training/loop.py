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
from fv.fovea import dims_of
from fv.ioutils import write_json_atomic
from fv.metrics import (corner_scores, detection_counts, monitor_improved,
                        monitor_key, pos_err_px)
from fv.models.builder import build_model, full_config
from fv.training.losses import corner_loss
from dataclasses import replace

from fv.training.recipe import Recipe
from fv.training.sampling import VentanasPorEpoca
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
        for x, e, y in loader:
            x, e, y = x.to(device), e.to(device), y.to(device)
            logits = model(x, e)
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


# Version del formato de `last.pt`. La 1 son los checkpoints escritos antes de
# que existiera la reanudacion: llevan pesos y epoca, y NADA mas. `reanudar` los
# distingue para poder decirlo en vez de continuar en silencio con el optimizador
# en blanco, que es la clase de degradacion que no se nota hasta la curva.
FORMATO_CONTINUACION = 2


def _estado_de_continuacion(opt, sched, best_value, best_epoch, no_improve,
                            epochs_run, g) -> dict:
    """Todo lo que hace falta para seguir donde se dejo, y nada mas."""
    return {
        "format_version": FORMATO_CONTINUACION,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "best_value": best_value,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "epochs_run": epochs_run,
        # el barajado tiene que CONTINUAR, no repetirse: sin esto la epoca 11 de
        # una reanudacion ve exactamente el mismo orden que vio la 1
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        # ⚠ Y el generador del DataLoader, que es OTRO y es el que decide el
        # barajado. Se siembra con `recipe.seed` en cada llamada, asi que sin
        # guardarlo la epoca 4 de una reanudacion recibe exactamente el orden que
        # recibio la 1 -- el modelo repasa lo mismo creyendo que avanza. Es lo
        # que hace que "3 + 3" y "6 de una vez" den la MISMA curva, y hay test.
        "rng_loader": g.get_state(),
    }


def reanudar(run_name: str, *, mas: int, patience: int | None = None,
             device: str = "cpu", store: RunStore | None = None,
             dataset_root: Path | None = None, progress=None, should_stop=None,
             optimizador_limpio: bool = False) -> dict:
    """Sigue entrenando un run que ya existe, `mas` epocas mas.

    No es "entrenar otra vez con el mismo nombre": `RunStore.create` se niega a
    sobrescribir a proposito, y esta bien que se niegue. Esto retoma el MISMO
    run -- misma red, mismo dataset, misma receta, mismo `metrics.jsonl` -- desde
    el estado que dejo `last.pt`.

    La red, el dataset y la receta salen del `config.json` del run y NO se pueden
    cambiar aqui: cambiarlas seria otro run con el historial de este pegado
    detras, y las curvas mentirian. La unica excepcion es `patience`, y esta
    razonada abajo.
    """
    store = store or RunStore()
    cfg = store.config(run_name)                     # se niega solo si no existe

    # ⚠ El guard va AQUI y no en el endpoint: dos continuaciones a la vez
    # escriben el mismo `metrics.jsonl` y el mismo `last.pt`, y el resultado no
    # es de ninguna de las dos. `reconcile` cura un "running" huerfano (proceso
    # muerto), asi que esto no bloquea por un entrenamiento que ya no existe.
    estado_run = store.reconcile(run_name).get("status")
    if estado_run in ("running", "queued"):
        raise RunError(
            "run_is_running",
            f"'{run_name}' ya esta entrenando: dos continuaciones a la vez se "
            f"pisan el metrics.jsonl y el last.pt",
            "espera a que acabe, o paralo primero")

    run_dir = store.path(run_name)
    ckpt = run_dir / "last.pt"
    if not ckpt.exists():
        raise RunError(
            "run_has_no_last_checkpoint",
            f"'{run_name}' no tiene last.pt: no hay desde donde seguir",
            "si tiene best.pt, ese sirve para EVALUAR pero no para continuar "
            "(no lleva el estado del optimizador); hay que entrenar de nuevo")
    estado = torch.load(ckpt, map_location=device, weights_only=False)

    # R2: o degrada con un defecto DECLARADO, o se niega antes de empezar. Un
    # checkpoint viejo (formato 1) no lleva optimizador: continuar con Adam en
    # blanco no falla -- da una curva peor durante unas epocas y nadie lo
    # relaciona. Asi que se niega, y se puede pedir explicitamente.
    if int(estado.get("format_version", 1)) < FORMATO_CONTINUACION:
        if not optimizador_limpio:
            raise RunError(
                "checkpoint_sin_estado",
                f"el last.pt de '{run_name}' es de antes de que se guardara el "
                f"estado de entrenamiento: no lleva optimizador ni contadores",
                "continuar asi reinicia los momentos de Adam y el early-stop, y "
                "eso se ve como una curva peor sin causa aparente. Entrena de "
                "nuevo, o pide optimizador_limpio=True sabiendo lo que cuesta")
        estado = {**estado, "optimizer": None}

    recipe = Recipe(**cfg["recipe"])
    if patience is not None:
        recipe = replace(recipe, patience=int(patience))
    net = cfg["network"]
    window_dataset = cfg["provenance"]["window_dataset"]["name"]
    wstore = WindowDatasetStore(dataset_root)
    manifest = wstore.manifest(window_dataset)

    # ⚠ El mismo guard que `task_score`: si el dataset se reconstruyo, sus splits
    # ya no son los que este modelo no vio, y seguir entrenando sobre el mezcla
    # train de hoy con val de ayer -- sin un solo error.
    if manifest["fingerprint"] != cfg["provenance"]["window_dataset"]["fingerprint"]:
        raise RunError(
            "window_dataset_changed",
            f"'{window_dataset}' se reconstruyo desde que se entreno "
            f"'{run_name}': continuar mezclaria splits distintos",
            "entrena un run nuevo contra el dataset actual")

    if estado.get("optimizer") is None and not optimizador_limpio:
        raise RunError("checkpoint_sin_estado",
                       f"el last.pt de '{run_name}' no trae optimizador",
                       "entrena de nuevo, o pide optimizador_limpio=True")

    try:
        return _train_inner(run_name, run_dir, manifest, net, recipe, device,
                            store, wstore, window_dataset, progress, should_stop,
                            estado=estado, n_epocas=int(mas))
    except Exception:
        store.set_status(run_name, "error")
        raise


def _train_inner(run_name, run_dir: Path, manifest, net, recipe: Recipe,
                 device, store: RunStore, wstore, window_dataset, progress,
                 should_stop=None, estado: dict | None = None,
                 n_epocas: int | None = None) -> dict:
    """`estado` = reanudar desde el de `last.pt`; None = empezar de cero.

    ⚠ La semilla solo se siembra al EMPEZAR. Al reanudar se restaura el RNG que
    dejo la ultima epoca (mas abajo): volver a sembrar aqui haria que la epoca 11
    barajase exactamente igual que la 1, y el modelo veria dos veces el mismo
    orden creyendo que avanza.
    """
    if estado is None:
        torch.manual_seed(recipe.seed)
        np.random.seed(recipe.seed % (2 ** 32))

    dims = dims_of(net)
    arrays = wstore.arrays(window_dataset)
    train_ds = FoveatedWindowDataset(arrays, dims, split=0,
                                     pool_mode=net["pool_mode"], pad_mode=net["pad_mode"],
                                     edge_inputs=net["edge_inputs"])
    val_ds = FoveatedWindowDataset(arrays, dims, split=1,
                                   pool_mode=net["pool_mode"], pad_mode=net["pad_mode"],
                                   edge_inputs=net["edge_inputs"])
    g = torch.Generator()
    g.manual_seed(recipe.seed)
    if estado is not None and estado.get("rng_loader") is not None:
        g.set_state(estado["rng_loader"])   # el barajado CONTINUA, no se repite
    # `windows_per_epoch` iguala el presupuesto entre datasets de distinto tamano
    # (docs/barrido-stride.md 2.2). Con 0 -- el default y todo lo ya medido -- NO
    # se construye sampler y la ruta es exactamente la de siempre: cambiar esto
    # movería el f1 de todas las tablas publicadas.
    por_epoca = int(getattr(recipe, "windows_per_epoch", 0) or 0)
    if por_epoca > 0:
        train_loader = DataLoader(
            train_ds, batch_size=recipe.batch_size, num_workers=0, generator=g,
            sampler=VentanasPorEpoca(len(train_ds), por_epoca, recipe.seed))
    else:
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
    desde = 0

    if estado is not None:
        model.load_state_dict(estado["model"])
        # `None` = se pidio continuar SIN el optimizador (checkpoint viejo): se
        # deja el recien creado, que es justo la degradacion que `reanudar`
        # obliga a pedir en voz alta.
        if estado.get("optimizer") is not None:
            opt.load_state_dict(estado["optimizer"])
        if sched is not None and estado.get("scheduler") is not None:
            sched.load_state_dict(estado["scheduler"])
        best_value = estado.get("best_value")
        best_epoch = estado.get("best_epoch")
        no_improve = int(estado.get("no_improve") or 0)
        desde = int(estado.get("epochs_run") or estado.get("epoch") or 0)
        epochs_run = desde
        if estado.get("rng_torch") is not None:
            torch.set_rng_state(estado["rng_torch"])
        if estado.get("rng_numpy") is not None:
            np.random.set_state(estado["rng_numpy"])

    total = int(n_epocas if n_epocas is not None else recipe.epochs)
    store.set_status(run_name, "running", epoch=desde, pid=os.getpid())
    for epoch in range(desde + 1, desde + total + 1):
        t0 = time.monotonic()
        model.train()
        epoch_losses = []
        for x, e, y in train_loader:
            x, e, y = x.to(device), e.to(device), y.to(device)
            opt.zero_grad()
            loss = corner_loss(model(x, e), y, recipe.lambda_pos, recipe.pos_weight,
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

        # the selection rule lives in fv.metrics, so the sweep ranking can ask
        # WHICH epoch this kept without reimplementing it (and drifting)
        monitor_value = val.get(monitor_key(recipe.monitor))
        # DOS ficheros con dos propositos, y por eso NO llevan lo mismo:
        #   best.pt  -> EVALUAR. La mejor epoca segun el monitor. Solo pesos, que
        #              es lo unico que necesita `load_model` (y la pantalla de
        #              revision). Meterle el optimizador lo triplicaria de tamano
        #              para nadie.
        #   last.pt  -> CONTINUAR. La ultima epoca CON el estado entero: sin el
        #              optimizador, reanudar reinicia los momentos de Adam; sin
        #              los contadores, el early-stop y la seleccion de best.pt
        #              empiezan de cero; y sin el RNG, la epoca siguiente repite
        #              el mismo barajado que la primera.
        pesos = {"model": model.state_dict(), "config": {"model": net}, "epoch": epoch}
        improved = monitor_improved(monitor_value, best_value, recipe.monitor)
        if improved:
            best_value, best_epoch = monitor_value, epoch
            torch.save(pesos, run_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
        torch.save({**pesos, **_estado_de_continuacion(
            opt, sched, best_value, best_epoch, no_improve, epochs_run, g)},
            run_dir / "last.pt")

        store.set_status(run_name, "running", epoch=epoch, pid=os.getpid())
        if progress:
            progress(epoch, desde + total, rec)
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
        "epochs_run": epochs_run,          # ACUMULADAS, no las de esta tanda
        "epochs_requested": desde + total,
        "continued_from": desde or None,   # None = este run no se reanudo
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
