"""The metric that MATTERS: paragraph per image (protocolo.md §2) — contract ⑬.

E x A via F: it scores a run (E) against the paragraphs of the SOURCE (A) by
running full-image inference (F). It lives apart from both neighbours on
purpose:

- `fv.diagnostics` is E x B **per window** ("A CACHE, not an entity"). This is
  per IMAGE and its truth does not come from B at all — B stores the images but
  NOT the true paragraphs, only the per-window corner labels.
- `fv.inference` is F pure: apply a model to an image. This SCORES that output.

Like diagnostics it is a pure function of (run, dataset fingerprint, split,
checkpoint mtime, knobs) and therefore a cache, not an entity: deleting
data/cache/task/ loses nothing but time. Unlike diagnostics, the F knobs ARE
part of the cache key — re-thresholding there re-reads stored scores, while here
any knob changes the whole reconstruction and forces a new inference pass.

Cost measured (metrica-de-tarea.md §3.8): 28 ms per 80x60 image with window 16
and stride 8, i.e. 0.6 s for a 20-image val split. It is NOT computed per epoch
and it is NOT a sweep objective — the window proxy ranks the same on axes of D
(Spearman +0.956 aggregated) and costs nothing.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

from fv import settings
from fv.datasets.loader import SourceDataset, SourceError
from fv.diagnostics.table import SPLITS   # ONE split vocabulary, two readers
from fv.inference.checkpoint import MODEL_CACHE
from fv.inference.predict import predict_image
from fv.ioutils import read_json_retrying, write_json_atomic
from fv.metrics import paragraph_f1
from fv.training.registry import RunError, RunStore
from fv.windows.store import WindowDatasetStore

CHECKPOINT = "best.pt"   # what survives, and what Diagnostico/Predecir load


def _cache_key(run: str, fingerprint: str, split: str, ckpt: Path,
               knobs: tuple) -> str:
    raw = (f"{run}|{fingerprint}|{split}|{ckpt.stat().st_mtime_ns}|"
           + "|".join(str(k) for k in knobs))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _aggregate(per_image: list[dict]) -> dict:
    """macro (PRIMARY) and micro, both reported — never a third invented number.

    macro = mean of the per-IMAGE F1, because the unit of sample is the image
    (protocolo.md §2/§3: the effective sample size is given by the images, not
    the windows). Its sd across images buys a standard error, which is what
    turns "0.53" into "0.53 ± 0.08" — with the task metric there is a noise band
    even with a single seed.

    micro = tp/fp/fn summed over every image, then P/R/F1: the standard
    detection number, kept so images with many paragraphs can be seen to weigh.
    If macro and micro disagree loudly that IS information (a few hard images
    dominate), not something to average away.
    """
    n = len(per_image)
    f1s = [r["f1"] for r in per_image]
    mean = sum(f1s) / n
    sd = statistics.stdev(f1s) if n >= 2 else None
    tp = sum(r["tp"] for r in per_image)
    fp = sum(r["fp"] for r in per_image)
    fn = sum(r["fn"] for r in per_image)
    mp = tp / (tp + fp) if tp + fp else 0.0
    mr = tp / (tp + fn) if tp + fn else 0.0
    matched = [(r["mean_iou"], r["tp"]) for r in per_image
               if r["mean_iou"] is not None and r["tp"]]
    return {
        "macro": {
            "f1": mean,
            "sd": sd,                                    # None with one image
            "sem": (sd / math.sqrt(n)) if sd is not None else None,
            "precision": sum(r["precision"] for r in per_image) / n,
            "recall": sum(r["recall"] for r in per_image) / n,
        },
        "micro": {"f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
                  "precision": mp, "recall": mr, "tp": tp, "fp": fp, "fn": fn},
        # mean IoU over the MATCHED pairs; None when nothing matched — never 0
        # (formatos.md §2: absent is not zero, and 0 would read as "matched
        # badly" instead of "matched nothing")
        "mean_iou": (sum(v * c for v, c in matched) / sum(c for _v, c in matched)
                     if matched else None),
    }


def task_score(run_name: str, split: str = "val", *,
               threshold: float = 0.5, stride: int | None = None,
               nms_radius: float | None = None, min_size: float | None = None,
               iou_threshold: float = 0.5, window_dataset: str | None = None,
               store: RunStore | None = None) -> dict:
    """Score one run on whole images. Same gates as diagnostics_table, on
    purpose: they are THE door, and a door repeated is a door that holds.

    `window_dataset` overrides which B says the split and the source — the
    holdout path (metrica-de-tarea.md §6): a holdout is another B, extracted
    from a source the run never saw. Sharing the source is refused, because then
    it is not a holdout.
    """
    store = store or RunStore()
    cfg = store.config(run_name)
    prov = cfg.get("provenance") or {}
    if not prov.get("window_dataset", {}).get("name"):
        raise RunError("run_without_provenance",
                       f"'{run_name}' no tiene procedencia: no puede decir de que "
                       f"dataset salio, asi que no hay contra que puntuarlo",
                       "borralo y reentrenalo: no es comparable con nada")
    own_name = prov["window_dataset"]["name"]
    ds_name = window_dataset or own_name
    wstore = WindowDatasetStore()
    manifest = wstore.manifest(ds_name)
    if ds_name == own_name:
        if manifest["fingerprint"] != prov["window_dataset"]["fingerprint"]:
            raise RunError("window_dataset_changed",
                           f"'{ds_name}' se reconstruyo desde que se entreno "
                           f"'{run_name}': su split ya no es el que ese best.pt "
                           f"no vio",
                           "reentrena contra el dataset actual: las imagenes de "
                           "val ya no son las que ese modelo nunca miro")
    else:
        # the holdout guard: a different B whose source is the SAME source is
        # training data wearing another name (metrica-de-tarea.md §6)
        own_source = wstore.manifest(own_name)["source_id"]
        if manifest["source_id"] == own_source:
            raise RunError(
                "holdout_shares_source",
                f"'{ds_name}' sale de la misma fuente ('{own_source}') que el "
                f"dataset con el que se entreno '{run_name}': no es un holdout",
                "extrae el holdout de una fuente propia, de la que jamas se "
                "extraiga entrenamiento (protocolo.md §3)")
    ckpt = store.path(run_name) / CHECKPOINT
    if not ckpt.exists():
        raise RunError("run_has_no_checkpoint",
                       f"'{run_name}' no tiene {CHECKPOINT} todavia",
                       "espera a que termine al menos una epoca")
    if split not in SPLITS:
        raise RunError("unknown_split", f"split '{split}' no existe",
                       "usa train, val o test")
    indices = wstore.split_map(ds_name).get(split) or []
    if not indices:
        raise RunError("split_empty",
                       f"el split '{split}' de '{ds_name}' no tiene imagenes",
                       "reconstruye el dataset con ese split > 0")

    knobs = (threshold, stride, nms_radius, min_size, iou_threshold)
    cache_dir = settings.cache_root() / "task"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(run_name, manifest['fingerprint'], split, ckpt, knobs)}.json"
    if cache_file.exists():
        payload = read_json_retrying(cache_file)
        payload["cached"] = True
        return payload

    source_id = manifest["source_id"]
    try:
        source = SourceDataset(source_id)
    except SourceError as e:
        # the reason it matters, not just "not found": scoring against the
        # window labels instead would silently measure a different thing
        raise RunError(
            "task_needs_source",
            f"la metrica de tarea se mide contra los parrafos de la fuente "
            f"'{source_id}', y esa fuente no esta ({e.message})",
            "recupera la fuente o mide solo la metrica de ventana") from e

    kinds = set(manifest["config"]["target_kinds"])
    model = MODEL_CACHE.get(ckpt)
    per_image, used_knobs = [], None
    for index in indices:
        sample = source.sample_at(int(index))
        out = predict_image(model, sample.load_image(), threshold=threshold,
                            stride=stride, nms_radius=nms_radius,
                            min_size=min_size)
        used_knobs = out["knobs"]
        pred = [(p["x0"], p["y0"], p["x1"], p["y1"]) for p in out["paragraphs"]]
        # the kind filter is NOT optional: a dataset extracted from paragraphs
        # is not scored against lines
        true = [b.bbox for b in sample.blocks if b.kind in kinds]
        r = paragraph_f1(pred, true, iou_threshold)
        r["index"] = int(index)
        per_image.append(r)

    payload = {
        "run": run_name, "split": split, "window_dataset": ds_name,
        "source": source_id, "images": len(per_image),
        **_aggregate(per_image),
        "per_image": [{"index": r["index"], "f1": r["f1"], "tp": r["tp"],
                       "fp": r["fp"], "fn": r["fn"], "mean_iou": r["mean_iou"]}
                      for r in per_image],
        # the knobs are echoed like predict_image does: the number means nothing
        # without them, and they are what the cache key is made of
        "knobs": {**used_knobs, "iou_threshold": iou_threshold},
        "checkpoint": CHECKPOINT,
    }
    write_json_atomic(cache_file, payload)
    payload["cached"] = False
    return payload
