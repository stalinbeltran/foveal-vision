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
import json
import math
import statistics
from datetime import datetime, timezone

from .report import SMALL_SAMPLE
from pathlib import Path

from fv import settings
from fv.datasets.loader import (SourceDataset, SourceError, resolve_source,
                                source_meta)
from fv.diagnostics.table import SPLITS   # ONE split vocabulary, two readers
from fv.inference.checkpoint import MODEL_CACHE
from fv.inference.predict import RECONSTRUCT_DEFAULT, predict_image
from fv.ioutils import read_json_retrying, read_text_retrying, write_json_atomic
from fv.metrics import paragraph_f1
from fv.training.registry import RunError, RunStore
from fv.windows.store import WindowDatasetStore

CHECKPOINT = "best.pt"   # what survives, and what Diagnostico/Predecir load

# F14 (decided by the user, 2026-07-26): the protocol says the holdout is looked
# at ONCE, at the end, with the winner only — and until now nothing remembered.
# The cache makes the second look free and INVISIBLE, which is the worst shape a
# rule can have: obeyed by discipline, unfalsifiable by inspection. So every
# scoring against a holdout appends a line here, cached or not. This is the one
# place the task metric stops being a pure cache and writes into the run: it is
# deliberate, and the ledger is append-only — it records looks, never blocks one.
HOLDOUT_LEDGER = "holdout.jsonl"
HOLDOUT_SUFFIX = "-holdout"


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


def is_holdout_source(source_id: str, source_meta: dict | None = None) -> bool:
    """Is this source a holdout? ONE definition, so the ledger and any future
    guard cannot disagree about what counts.

    Two signals, both honoured: an explicit `"holdout": true` in the source's
    `dataset.json` (robust, and what should be written from now on) and the name
    convention `<algo>-holdout` (all there was before the field existed). The
    explicit field WINS in both directions — a source can declare itself not a
    holdout despite its name, because a convention should never override a
    statement (README documents both).
    """
    if source_meta is not None and "holdout" in source_meta:
        return bool(source_meta["holdout"])
    return source_id.rstrip("/").endswith(HOLDOUT_SUFFIX)


def _source_meta(source_id: str) -> dict:
    """The source's dataset.json on the CACHED path, where no SourceDataset was
    opened. A source that has since vanished simply has no metadata: the cached
    number is still valid (it was measured when the source was there), so this
    must not raise — it falls back to the name convention."""
    try:
        return source_meta(resolve_source(source_id))
    except SourceError:
        return {}


def record_holdout_touch(run_name: str, payload: dict, store: RunStore) -> None:
    """Append one line per look at a holdout (F14). Called even when the number
    came from cache: what is being recorded is that somebody LOOKED, and a cached
    look is exactly the one that used to leave no trace."""
    line = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_dataset": payload["window_dataset"], "source": payload["source"],
        "split": payload["split"], "images": payload["images"],
        "f1": payload["macro"]["f1"], "sem": payload["macro"]["sem"],
        "knobs": payload["knobs"], "checkpoint": payload["checkpoint"],
        "from_cache": bool(payload.get("cached")),
    }
    path = store.path(run_name) / HOLDOUT_LEDGER
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")


def holdout_touches(run_name: str, store: RunStore | None = None) -> list[dict]:
    """Every look at a holdout this run has had — the fact, not the promise."""
    store = store or RunStore()
    path = store.path(run_name) / HOLDOUT_LEDGER
    if not path.exists():
        return []
    return [json.loads(ln) for ln in read_text_retrying(path).splitlines() if ln.strip()]


def task_score(run_name: str, split: str = "val", *,
               threshold: float = 0.5, stride: int | None = None,
               nms_radius: float | None = None, min_size: float | None = None,
               iou_threshold: float = 0.5, window_dataset: str | None = None,
               reconstruct: str = RECONSTRUCT_DEFAULT,
               corner_tol: float | None = None,
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

    # ⚠ `reconstruct` y `corner_tol` ENTRAN en la clave, como los demas knobs de
    # F y por el mismo motivo (§ del docstring: cambiarlos obliga a re-inferir).
    # Si no entraran, cambiar de reconstruccion --o cambiar su DEFECTO-- serviria
    # numeros cacheados con la otra, bajo el mismo nombre y sin decirlo, que es
    # justo el fallo silencioso que esta cache tiene prohibido.
    knobs = (threshold, stride, nms_radius, min_size, iou_threshold,
             reconstruct, corner_tol)
    cache_dir = settings.cache_root() / "task"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(run_name, manifest['fingerprint'], split, ckpt, knobs)}.json"
    if cache_file.exists():
        payload = read_json_retrying(cache_file)
        payload["cached"] = True
        # A field ADDED to the payload does not invalidate the cache, so an entry
        # written before it exists comes back without it -- and a consumer that
        # needs it would silently read "false" (formatos.md §1: ausente != cero).
        # `small_sample` is derived from `images`, so it is filled on the way out
        # instead of forcing a re-inference of every cached run.
        payload.setdefault("small_sample", payload["images"] < SMALL_SAMPLE)
        # a cached look is STILL a look (F14): the free-and-invisible second
        # glance is precisely what the ledger exists to make visible
        if is_holdout_source(payload["source"], _source_meta(payload["source"])):
            record_holdout_touch(run_name, payload, store)
        payload["holdout_touches"] = len(holdout_touches(run_name, store))
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
                            min_size=min_size, reconstruct=reconstruct,
                            corner_tol=corner_tol)
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
        # The threshold is DEFINED once (report.SMALL_SAMPLE) and travels with the
        # number: a client that compares against its own 100 is a second definition
        # of the same fact (ui/6-numeros.md U6.2).
        "small_sample": len(per_image) < SMALL_SAMPLE,
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
    if is_holdout_source(source_id, source.meta):
        record_holdout_touch(run_name, payload, store)
    payload["holdout_touches"] = len(holdout_touches(run_name, store))
    return payload
