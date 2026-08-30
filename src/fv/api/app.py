"""The API: HTTP and nothing else (api.md §0). One resource per domain (R1),
no ambiguous words (R2), sync vs job at ~1 s (R3), every error carries
code/message/hint (R4), incremental polling (R5), aggregates server-side (R6),
names not values (R7). CORS closed to the front origin; images resolve inside
the domain, never by client path.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from fv import settings
from fv.api.jobs import JobQueue
from fv.ioutils import read_json_retrying, write_json_atomic
from fv.datasets.loader import SourceDataset, SourceError, discover_sources
from fv.fovea import (FoveaError, build_search_space, check_dims, derive_dims,
                      normalize_geometry)
from fv.inference.checkpoint import MODEL_CACHE, CheckpointError
from fv.inference.introspect import (feature_maps_payload, input_view_payload,
                                     kernels_payload)
from fv.inference.predict import predict_image
from fv.metrics import corner_evidence
from fv.models.builder import full_config, network_trace
from fv.models.store import NetworkStore, NetworkStoreError
from fv.sweeps.generate import generate_sweep
from fv.sweeps.runner import delete_sweep, prepare_sweep, run_sweep, sweep_trials
from fv.sweeps.winner import suggest_winner
from fv.sweeps.spec import SweepError
from fv.sweeps.store import SweepStore, SweepStoreError
from fv.studies.driver import (StudyError, advance, confirm, create_study,
                               delete_study)
from fv.studies.driver import status as study_status_fn
from fv.studies.store import StudyStore, StudyStoreError
from fv.training.loop import reanudar, train
from fv.training.recipe import Recipe, RecipeStore, RecipeStoreError
from fv.training.registry import RunError, RunStore
from fv.validation import check_network, check_run
from fv.windows.extract import ExtractConfig, ExtractError, extract_windows
from fv.windows.store import WindowDatasetStore, WindowStoreError

DOMAIN_ERRORS = (SourceError, ExtractError, WindowStoreError, NetworkStoreError,
                 RecipeStoreError, RunError, SweepError, SweepStoreError,
                 StudyError, StudyStoreError, CheckpointError, FoveaError)

NOT_FOUND_CODES = {"source_not_found", "sample_not_found", "window_dataset_missing",
                   "network_not_found", "recipe_not_found", "run_not_found",
                   "sweep_not_found", "study_not_found"}
CONFLICT_CODES = {"window_dataset_exists", "window_dataset_in_use", "network_exists",
                  "recipe_exists", "recipe_in_use", "run_exists", "run_is_running", "sweep_exists",
                  "run_without_provenance", "run_has_no_checkpoint", "run_belongs_to_sweep",
                  "window_dataset_changed", "split_empty", "sweep_is_running",
                  "study_exists", "step_awaiting_confirmation", "study_has_live_sweeps"}


def _http_error(e) -> HTTPException:
    code = getattr(e, "code", "error")
    status = 404 if code in NOT_FOUND_CODES else 409 if code in CONFLICT_CODES else 400
    return HTTPException(status_code=status, detail={
        "code": code, "message": getattr(e, "message", str(e)),
        "hint": getattr(e, "hint", "")})


def create_app() -> FastAPI:
    app = FastAPI(title="foveal-vision API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"], allow_headers=["*"])

    jobs = JobQueue(max_workers=1)  # CPU: torch already uses every core
    runs = RunStore()
    wstore = WindowDatasetStore()
    nstore = NetworkStore()
    rstore = RecipeStore()
    sstore = SweepStore()
    studies_store = StudyStore()

    @app.exception_handler(Exception)
    async def _domain_handler(request, exc):
        from fastapi.responses import JSONResponse
        if isinstance(exc, DOMAIN_ERRORS):
            he = _http_error(exc)
            return JSONResponse(status_code=he.status_code,
                                content={"detail": he.detail})
        raise exc

    # ------------------------------------------------------------- sources (A)
    @app.get("/sources")
    def list_sources():
        return {"sources": discover_sources(),
                "external_root": str(settings.external_datasets_root() or "")}

    @app.get("/sources/{source_id:path}/samples/{index}/image")
    def source_image(source_id: str, index: int, w: int | None = None):
        ds = SourceDataset(source_id)
        s = ds.sample_at(index)
        from PIL import Image
        img = Image.open(s.image_path).convert("L")
        if w and w < img.width:
            img = img.resize((w, int(img.height * w / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.get("/sources/{source_id:path}/samples/{index}")
    def source_sample(source_id: str, index: int):
        ds = SourceDataset(source_id)
        s = ds.sample_at(index)
        return {"index": s.index, "width": s.width, "height": s.height,
                "blocks": [{"block_id": b.block_id, "kind": b.kind,
                            "angle": b.angle, "quad": b.quad.tolist()}
                           for b in s.blocks]}

    @app.get("/sources/{source_id:path}")
    def source_meta(source_id: str):
        ds = SourceDataset(source_id)
        return {"id": source_id, "count": len(ds)}

    # ---------------------------------------------------- window datasets (B)
    @app.get("/window-datasets")
    def list_window_datasets():
        return {"window_datasets": wstore.list()}

    @app.post("/window-datasets", status_code=202)
    def build_window_dataset(body: dict):
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, {"code": "name_required",
                                      "message": "falta el nombre", "hint": ""})
        if wstore.path(name).exists():
            raise _http_error(WindowStoreError(
                "window_dataset_exists", f"ya existe '{name}'",
                "elige otro nombre: no se sobrescribe nunca"))
        cfg = ExtractConfig(
            source=body["source"],
            window_size=int(body.get("window_size", 16)),
            stride=int(body.get("stride", 8)),
            val_frac=float(body.get("val_frac", 0.15)),
            test_frac=float(body.get("test_frac", 0.15)),
            seed=int(body.get("seed", 1)))
        # fail BEFORE creating the job on a bad source
        SourceDataset(cfg.source)

        def work(is_cancelled):
            return extract_windows(cfg, wstore.path(name),
                                   should_stop=is_cancelled)
        return {"job": jobs.submit("extract", work, {"name": name})}

    def _dataset_referrers(name: str) -> list[str]:
        """Everything that would RETRAIN on this B by name: runs (have), sweeps
        (resume), studies (advance). All three break if B vanishes — so all three
        gate its deletion (R4), not just runs."""
        return (runs.used_by_dataset(name)
                + sstore.used_by_dataset(name)
                + studies_store.used_by_dataset(name))

    @app.get("/window-datasets/{name}")
    def window_dataset_detail(name: str):
        m = wstore.manifest(name)
        m["used_by"] = _dataset_referrers(name)
        return m

    @app.delete("/window-datasets/{name}")
    def delete_window_dataset(name: str):
        wstore.delete(name, _dataset_referrers(name))
        return {"deleted": name}

    @app.get("/window-datasets/{name}/windows/{index}")
    def window_pixels(name: str, index: int):
        m = wstore.manifest(name)
        arrays = wstore.arrays(name)
        if index < 0 or index >= arrays["y"].shape[0]:
            raise _http_error(WindowStoreError(
                "window_not_found", f"'{name}' no tiene la ventana {index}",
                f"indices validos: 0..{arrays['y'].shape[0] - 1}"))
        n = int(m["config"]["window_size"])
        lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
        row = lookup[int(arrays["sample_idx"][index])]
        wx0, wy0 = (int(v) for v in arrays["window_xy"][index])
        crop = arrays["images"][row][wy0:wy0 + n, wx0:wx0 + n]
        return {"index": index, "window_size": n,
                "sample_idx": int(arrays["sample_idx"][index]),
                "window_xy": [wx0, wy0],
                "y": arrays["y"][index].tolist(),
                # `y` rows are ordered by THIS dataset's corner_order: whoever
                # draws them by index needs it, and must not keep its own copy
                "corner_order": list(m["corner_order"]),
                "pixels": crop.tolist(),
                "split": int(arrays["split"][index])}

    @app.get("/window-datasets/{name}/samples/{index}/image")
    def window_dataset_image(name: str, index: int, w: int | None = None):
        """La imagen ENTERA de una muestra, sacada del `windows.npz`.

        Existe porque el npz es lo que de verdad viaja por git y las FUENTES no:
        medido el 2026-08-29 en un dev recien hecho, `discover_sources()` devolvia
        0 y con ello no habia una sola imagen que mirar. `extract.py` las guarda
        verbatim (`images[si] = s.load_image()`), asi que estos pixeles son
        exactamente los que el modelo miro -- no una reconstruccion.

        Es el gemelo de `/sources/{id}/samples/{i}/image`, con la misma firma y
        el mismo `?w=` para el movil: quien pinta no tiene que saber de cual de
        los dos vino (`image_base` en la respuesta de revision lo decide una vez).
        """
        arrays = wstore.arrays(name)
        lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
        if int(index) not in lookup:
            raise _http_error(WindowStoreError(
                "sample_not_found", f"'{name}' no guarda la muestra {index}",
                "usa un indice de los que trae el split"))
        from PIL import Image
        img = Image.fromarray(arrays["images"][lookup[int(index)]], mode="L")
        if w and w < img.width:
            img = img.resize((w, int(img.height * w / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.get("/window-datasets/{name}/windows")
    def window_list(name: str, split: str | None = None, offset: int = 0,
                    limit: int = 24, positives_only: bool = False):
        limit = min(limit, 96)  # bounded by the route, not by convention (R6)
        arrays = wstore.arrays(name)
        mask = np.ones(arrays["y"].shape[0], dtype=bool)
        if split in ("train", "val", "test"):
            mask &= arrays["split"] == ("train", "val", "test").index(split)
        if positives_only:
            mask &= (arrays["y"][:, :, 0] >= 0.5).any(axis=1)
        idxs = np.flatnonzero(mask)
        return {"total": int(idxs.size),
                "indexes": [int(i) for i in idxs[offset:offset + limit]]}

    # -------------------------------------------------------------- networks (C)
    @app.get("/networks")
    def list_networks():
        # the C defaults travel with the list (like /recipes serves Recipe()):
        # a screen that hardcodes its own copy drifts the day one changes here —
        # measured: the form said channels [16,16] while the builder derives
        # [16]*n_layers. full_config resolves them with the SAME rule the builder
        # uses, so what the form pre-fills is what an empty config would build.
        return {"networks": nstore.list(), "defaults": full_config({})}

    @app.post("/networks")
    def save_network(body: dict):
        name = body.get("name", "")
        try:
            cfg = full_config(body)
        except FoveaError as e:
            raise HTTPException(400, e.as_dict())
        # the SAME validator every other door asks (R4): check_dims alone only
        # covers the geometry, so a kernel that is even, an unknown merge or a
        # dropout outside [0,1) used to be SAVED here and refused later, at the
        # training door — a stored config that no run can ever use. Measured
        # 2026-08-27 while adding `dropout`; the hole predates it.
        problems = check_network(cfg)
        if problems:
            raise HTTPException(400, problems[0])
        nstore.save(name, cfg, overwrite=bool(body.get("overwrite")))
        return {"saved": name}

    @app.post("/networks/validate")
    def validate_network(body: dict):
        try:
            cfg = full_config(body)
        except FoveaError as e:
            return {"valid": False, "problems": [e.as_dict()]}
        problems = check_network(cfg)
        if problems:
            return {"valid": False, "problems": problems}
        trace = network_trace(cfg)
        dims = trace["dims"]
        space = build_search_space(cfg, n_layers=int(cfg["n_layers"]))
        return {"valid": True, "trace": trace,
                "ranges": {k: v for k, v in space.items() if not k.startswith("_")}}

    @app.get("/networks/{name}")
    def get_network(name: str):
        cfg = nstore.get(name)
        cfg["name"] = name
        return cfg

    @app.delete("/networks/{name}")
    def delete_network(name: str):
        nstore.delete(name)
        return {"deleted": name}

    # -------------------------------------------------------------- recipes (D)
    @app.get("/recipes")
    def list_recipes():
        # `monitor` is a closed vocabulary and the screen must not keep its own
        # copy: it was filling that select with the sweep OBJECTIVES ('f1'), a
        # different vocabulary — so the control showed 'f1' while the recipe
        # said 'val_loss', and saving 'f1' would have made best.pt keep the
        # worst epoch. Served from the same constant the gate validates against.
        from fv.metrics import MONITORS
        names = [r["name"] for r in rstore.list()]
        return {"recipes": rstore.list(),
                "defaults": Recipe().as_dict(),
                "vocabulary": {"monitor": list(MONITORS)},
                # Who pins each D BY NAME. A run or a sweep copied its values, so
                # editing one never rewrites their history; a STUDY re-resolves
                # base_recipe at every advance, so its next steps would use the
                # new values. That is what the screen must say before replacing.
                # A map apart, never inside the recipe: a field mixed into the
                # object comes back on the next save as an unknown one.
                "used_by": {n: studies_store.used_by_recipe(n) for n in names}}

    @app.post("/recipes")
    def save_recipe(body: dict):
        name = body.pop("name", "")
        overwrite = bool(body.pop("overwrite", False))
        rstore.save(name, body, overwrite=overwrite)
        return {"saved": name}

    @app.get("/recipes/{name}")
    def get_recipe(name: str):
        r = rstore.get(name).as_dict()
        r["name"] = name
        return r

    @app.delete("/recipes/{name}")
    def delete_recipe(name: str):
        # a run/sweep snapshots the recipe VALUES, so they don't pin it; a STUDY
        # carries base_recipe by NAME and re-resolves it at advance (generate),
        # so deleting it would break the study later -> refuse at the gate (R4).
        used = studies_store.used_by_recipe(name)
        if used:
            raise _http_error(RecipeStoreError(
                "recipe_in_use",
                f"la receta '{name}' la fijan los estudios: {', '.join(used)}",
                "borra esos estudios primero, o deja la receta"))
        rstore.delete(name)
        return {"deleted": name}

    # ------------------------------------------------------------------ runs (E)
    @app.get("/runs")
    def list_runs():
        return {"runs": runs.list()}

    @app.post("/runs", status_code=202)
    def create_run(body: dict):
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, {"code": "name_required",
                                      "message": "falta el nombre del run", "hint": ""})
        net = nstore.get(body["network"])          # names, not values (R7)
        recipe = rstore.get(body["recipe"])
        device = body.get("device", "cpu")         # X: aside, never in the recipe
        manifest = wstore.manifest(body["window_dataset"])
        problems = check_run(manifest, full_config(net))
        if problems:                                # 400 BEFORE job and BEFORE name
            raise HTTPException(400, problems[0])
        if runs.exists(name):
            raise _http_error(RunError("run_exists",
                                       f"ya existe un run llamado '{name}'",
                                       "elige otro nombre, o borra ese run primero: "
                                       "no se sobrescribe nunca"))

        def work(is_cancelled):
            return train(name, body["window_dataset"], body["network"], net,
                         body["recipe"], recipe, device=device, store=runs)
        return {"job": jobs.submit("train", work, {"run": name},
                                   on_cancel=lambda: runs.request_stop(name))}

    @app.post("/runs/{name}/continue", status_code=202)
    def continue_run(name: str, body: dict):
        """Sigue un run que ya existe, `more` epocas mas.

        Endpoint propio y no una bandera de `POST /runs`: aquel CREA (y se niega
        si el nombre existe, a proposito). Aqui no se eligen red, dataset ni
        receta -- salen del run --, asi que aceptarlas seria admitir campos que
        se ignoran en silencio.
        """
        if not runs.exists(name):
            raise _http_error(RunError("run_not_found", f"no existe '{name}'",
                                       "mira la lista en /runs"))
        mas = int(body.get("more", 0))
        if mas < 1:
            raise HTTPException(400, {
                "code": "bad_more", "message": "`more` son epocas ADICIONALES y >= 1",
                "hint": "para empezar uno nuevo: POST /runs"})
        st = runs.reconcile(name).get("status")
        if st in ("running", "queued"):
            raise _http_error(RunError(
                "run_is_running", f"'{name}' ya esta entrenando",
                "espera a que acabe, o paralo con POST /runs/{name}/stop"))
        patience = body.get("patience")

        def work(is_cancelled):
            return reanudar(name, mas=mas,
                            patience=None if patience is None else int(patience),
                            device=body.get("device", "cpu"), store=runs,
                            optimizador_limpio=bool(body.get("optimizador_limpio")))
        return {"job": jobs.submit("continue", work, {"run": name, "more": mas},
                                   on_cancel=lambda: runs.request_stop(name))}

    @app.get("/runs/{name}/metrics")
    def run_metrics(name: str, since: int = 0):
        if not runs.exists(name):
            raise _http_error(RunError("run_not_found", f"no existe '{name}'", ""))
        return runs.metrics_since(name, since)

    @app.get("/runs/{name}")
    def run_detail(name: str):
        cfg = runs.config(name)
        st = runs.status(name)
        summary = {}
        sp = runs.path(name) / "summary.json"
        if sp.exists():
            from fv.ioutils import read_json_retrying
            summary = read_json_retrying(sp)
        return {"name": name, "status": st, "config": cfg, "summary": summary}

    @app.post("/runs/{name}/stop")
    def stop_run(name: str):
        runs.request_stop(name)
        return {"stopping": name}

    @app.patch("/runs/{name}")
    def rename_run(name: str, body: dict):
        runs.rename(name, body.get("new_name", ""))
        return {"renamed": body.get("new_name")}

    @app.delete("/runs/{name}")
    def delete_run(name: str):
        cfg = runs.config(name)
        sweep = cfg.get("provenance", {}).get("sweep")
        if sweep and sstore.exists(sweep):
            raise _http_error(RunError(
                "run_belongs_to_sweep",  # 409 family — NOT "is running": the run
                # can be done/stopped and still be undeletable on its own
                f"'{name}' pertenece al recorrido '{sweep}'",
                "borra el recorrido entero (arrastra sus runs) o deja el run: "
                "sus puntos se comparan juntos"))
        # sweep GONE (deleted out-of-band: manual/filesystem/git checkout) -> the
        # run is an ORPHAN and "borra el recorrido entero" is impossible; deleting
        # it alone is the only way out, so allow it instead of a permanent
        # deadlock. Mirror of the reconcile idiom: heal a dangling ref, never trap.
        runs.delete(name)
        return {"deleted": name}

    # ------------------------------------------------- diagnostics (E x B cache)
    from fv.diagnostics.table import (diagnostics_table, summary_payload,
                                      worst_windows)

    @app.get("/runs/{name}/diagnostics/summary")
    def diag_summary(name: str, split: str = "val", threshold: float = 0.5):
        table = diagnostics_table(name, split, runs)
        return summary_payload(table, threshold)

    @app.get("/runs/{name}/diagnostics/windows")
    def diag_windows(name: str, split: str = "val", threshold: float = 0.5,
                     offset: int = 0, limit: int = 24, outcome: str | None = None):
        limit = min(limit, 96)
        table = diagnostics_table(name, split, runs)
        return worst_windows(table, threshold, limit, offset, outcome)

    @app.get("/runs/{name}/diagnostics/evidence")
    def diag_evidence(name: str, split: str = "val", threshold: float = 0.5,
                      blind: float = 0.05):
        table = diagnostics_table(name, split, runs)
        ev = corner_evidence(table["y_true"])
        err = table["err_px"]
        scores = table["scores"]
        true_pos = table["y_true"][:, :, 0] >= 0.5
        bands = []
        edges = [0.0, blind, 0.2, 0.5, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = true_pos & (ev >= lo) & (ev < hi)
            n = int(m.sum())
            bands.append({
                "band": f"[{lo:.2f}, {hi:.2f})", "count": n,
                "mean_score": float(scores[m].mean()) if n else None,
                "mean_err_px": float(np.nanmean(err[m & np.isfinite(err)]))
                if (m & np.isfinite(err)).any() else None})
        return {"blind_threshold": blind, "bands": bands}

    # ------------------------------------------- task metric (E x A via F, (13))
    from fv.task import task_score

    @app.get("/runs/{name}/task-score")
    def run_task_score(name: str, split: str = "val", threshold: float = 0.5,
                       stride: int | None = None, nms_radius: float | None = None,
                       min_size: float | None = None, iou_threshold: float = 0.5,
                       window_dataset: str | None = None):
        """The metric that MATTERS (paragraph per image), on demand — never on a
        poll: it re-infers whole images (0.6 s for 20, seconds for a holdout)."""
        return task_score(name, split, threshold=threshold, stride=stride,
                          nms_radius=nms_radius, min_size=min_size,
                          iou_threshold=iou_threshold,
                          window_dataset=window_dataset, store=runs)

    # ------------------------------------------------------- introspection (V1/V2/F0)
    def _model_for(name: str):
        ckpt = runs.path(name) / "best.pt"
        if not ckpt.exists():
            raise _http_error(RunError("run_has_no_checkpoint",
                                       f"'{name}' no tiene best.pt",
                                       "espera a que termine una epoca"))
        return MODEL_CACHE.get(ckpt)

    @app.get("/runs/{name}/kernels")
    def run_kernels(name: str):
        return kernels_payload(_model_for(name))

    def _probe_window(body: dict):
        """Resolve (image, window x0, y0) for a probe, validating the index
        BEFORE indexing (R4): a stale gallery thumb from another dataset must
        get a clean 400, not a 500 out-of-bounds deep in the handler."""
        arrays = wstore.arrays(body["window_dataset"])
        n = len(arrays["sample_idx"])
        i = int(body["index"])
        if not 0 <= i < n:
            raise _http_error(RunError(
                "window_index_out_of_range",
                f"la ventana {i} no existe en '{body['window_dataset']}' "
                f"(tiene {n})",
                "elige una ventana del dataset del run: quiza cambiaste de run "
                "y la galeria anterior quedo en pantalla"))
        lookup = {int(a): r for r, a in enumerate(arrays["images_sample_idx"])}
        img = arrays["images"][lookup[int(arrays["sample_idx"][i])]]
        wx0, wy0 = (int(v) for v in arrays["window_xy"][i])
        return img, wx0, wy0

    @app.post("/runs/{name}/feature-maps")
    def run_feature_maps(name: str, body: dict):
        model = _model_for(name)
        img, wx0, wy0 = _probe_window(body)
        from fv.fovea import build_view
        view, _cov = build_view(img, wx0, wy0, model.dims,
                                pool_mode=model.cfg["pool_mode"],
                                pad_mode=model.cfg["pad_mode"])
        return feature_maps_payload(model, view)

    @app.post("/runs/{name}/input-view")
    def run_input_view(name: str, body: dict):
        model = _model_for(name)
        img, wx0, wy0 = _probe_window(body)
        return input_view_payload(model, img, wx0, wy0)

    # --------------------------------------------------------------- predict (F)
    @app.post("/runs/{name}/predict")
    def run_predict(name: str, body: dict):
        model = _model_for(name)
        ds = SourceDataset(body["source"])
        s = ds.sample_at(int(body.get("index", 0)))
        image = s.load_image()
        result = predict_image(
            model, image,
            threshold=float(body.get("threshold", 0.5)),
            stride=body.get("stride"),
            nms_radius=body.get("nms_radius"),
            min_size=body.get("min_size"))
        result["truth"] = [{"quad": b.quad.tolist()} for b in s.blocks]
        return result

    # ------------------------------------------------- revision a ojo (F x B)
    #
    # Mirar N imagenes de un split con las cajas encima. Es la pregunta que la
    # metrica de tarea NO contesta: `task_score` dice cuanto acierta, esto dice
    # QUE esta fallando. Reusa exactamente la misma puerta que aquella --
    # provenance -> dataset -> split_map -> source -> predict_image-- para que
    # las cajas que se miran sean las mismas que se puntuan.
    from fv import review as review_mod
    from fv.metrics import paragraph_f1

    REVIEW_MAX = 60   # acotado por la RUTA, no por convencion: son ~30 ms de
                      # inferencia por imagen y esto va sincrono como
                      # /task-score. Sin tope, un N=200 desde el movil cuelga la
                      # peticion y parece que la pagina esta rota.

    def _reviewable() -> list[dict]:
        """Los datasets que se PUEDEN revisar: los que traen `windows.npz`.

        Es la lista corta a proposito. Un `manifest.json` sin npz describe un
        dataset cuyo dato se perdio (hay 16 asi hoy), y ofrecerlo en un select es
        prometer imagenes que no existen.
        """
        out = []
        for m in wstore.list():
            if not (wstore.path(m["name"]) / "windows.npz").exists():
                continue
            try:
                fuente_ok = bool(SourceDataset(m["source_id"]))
            except (SourceError, KeyError):
                fuente_ok = False
            out.append({
                "name": m["name"], "source": m.get("source_id"),
                "source_available": fuente_ok,
                "images": m.get("num_samples"),
                "splits": sorted((m.get("windows_per_split") or {}).keys()),
            })
        return out

    def _review_ctx(ds_name: str):
        """(manifest, source o None). Sin fuente NO se falla: las imagenes salen
        del npz, que es justo lo que si viaja por git. Lo que se pierde es la
        VERDAD (los parrafos reales viven en labels.jsonl de A), y eso se dice en
        el payload en vez de dibujar un overlay vacio que se lea como "la red no
        se dejo nada" (R2: degradar con un defecto declarado)."""
        manifest = wstore.manifest(ds_name)
        try:
            return manifest, SourceDataset(manifest["source_id"])
        except (SourceError, KeyError):
            return manifest, None

    def _runs_de(ds_name: str, todos: list | None = None) -> list[dict]:
        """Los runs de ESTE dataset, con si tienen checkpoint.

        Filtrar aqui y no en el front es lo que quita el select de 859 runs: la
        lista ya trae `window_dataset`, asi que no cuesta una lectura de mas.

        `todos` existe para no releer los 859 runs una vez por dataset al elegir
        cual se abre por defecto: quien llame en bucle pasa la lista una vez.
        """
        out = []
        for r in (runs.list() if todos is None else todos):
            if r.get("window_dataset") != ds_name:
                continue
            out.append({"name": r["name"], "status": r.get("status"),
                        "best": r.get("best"),
                        "has_checkpoint": (runs.path(r["name"]) / "best.pt").exists()})
        # los que pueden inferir primero: un select cuyo primer elemento falla
        # ensena un error antes que una imagen
        out.sort(key=lambda r: (not r["has_checkpoint"], r["name"]))
        return out

    @app.get("/review/datasets")
    def review_datasets():
        return {"datasets": _reviewable()}

    @app.get("/review/context")
    def review_context(window_dataset: str | None = None, split: str = "val",
                       count: int = 10, run: str | None = None):
        """Que hay que revisar, SIN inferir nada.

        `window_dataset` manda; `run` es opcional. Antes mandaba el run (el
        dataset salia de su procedencia) y eso obligaba al front a traerse los
        859 runs para poder elegir -- que es exactamente lo que no se queria ver.
        """
        datasets = _reviewable()
        if not datasets:
            raise _http_error(WindowStoreError(
                "no_reviewable_datasets",
                "ningun dataset de ventanas tiene windows.npz: no hay imagenes que mirar",
                "commitea el dataset en el repo de datos, o extrae uno nuevo"))
        nombres = [d["name"] for d in datasets]
        todos = runs.list()
        if window_dataset in nombres:
            ds_name = window_dataset
        else:
            # Sin peticion explicita manda el dataset que SE PUEDE mirar con
            # modelo, no el primero de la lista. Medido el 2026-08-30 en un dev
            # recien hecho: el primero era `bench-dirty1000-16-r20260827`, que
            # tiene CERO runs, asi que la pantalla abria diciendo "este dataset
            # no tiene ningun run en esta maquina" mientras el de al lado tenia
            # un `demo-*` con pesos. Si ninguno tiene checkpoint se cae al
            # primero, que es el comportamiento de siempre.
            ds_name = next(
                (n for n in nombres
                 if any(r["has_checkpoint"] for r in _runs_de(n, todos))),
                nombres[0])
        manifest, source = _review_ctx(ds_name)
        indices = wstore.split_map(ds_name).get(split) or []
        vistos = review_mod.reviewed_indices(ds_name, split)
        n = max(1, min(int(count), REVIEW_MAX))
        pend = [i for i in indices if i not in set(vistos)]
        return {
            "datasets": datasets, "window_dataset": ds_name, "split": split,
            "source": manifest.get("source_id"),
            "truth_available": source is not None,
            "image_base": _image_base(ds_name, manifest, source),
            # [W, H] de la imagen entera, del propio manifest: permite pintar el
            # hueco con SU proporcion antes de inferir, en vez de con un defecto
            # que luego salta
            "image_size": [manifest["images"]["shape"][2],
                           manifest["images"]["shape"][1]]
                          if manifest.get("images", {}).get("shape") else None,
            "runs": _runs_de(ds_name, todos), "run": run,
            # Cual abrir si el usuario no ha elegido nunca. Lo decide el servidor
            # -- que ya ordena los runs poniendo delante los que pueden inferir--
            # y no el front, para que no haya dos copias de la regla. Es lo que
            # hace que una maquina recien lanzada ensene CAJAS al abrir en vez de
            # pedir que adivines cual de sus 10 runs trajo pesos.
            "run_sugerido": next(
                (r["name"] for r in _runs_de(ds_name, todos)
                 if r["has_checkpoint"]), None),
            "total": len(indices),
            "splits": sorted(wstore.split_map(ds_name).keys()),
            "reviewed": len(vistos), "pending": len(pend),
            "next_offset": review_mod.next_unreviewed_offset(indices, vistos, n),
            "marked": review_mod.marked_in(ds_name, split),
            "storage": review_mod.donde_se_guarda(),
        }

    def _image_base(ds_name: str, manifest: dict, source) -> str:
        """De donde saca el navegador el PNG. Se decide UNA vez, aqui: si lo
        decidiera el front, tendria una segunda copia de la regla "hay fuente o
        no" y las dos podrian dejar de coincidir."""
        if source is not None:
            return f"/api/sources/{manifest['source_id']}/samples"
        return f"/api/window-datasets/{ds_name}/samples"

    @app.post("/review/batch")
    def review_batch(body: dict):
        """Infiere un RANGO del split y deja constancia de que se miro.

        `run` es OPCIONAL: sin el (o sin checkpoint) se devuelven las imagenes sin
        cajas, que es lo que permite mirar el dataset en una maquina que solo
        tiene el repo de datos. Mirar sigue quedando registrado -- mirar sin
        modelo es mirar.
        """
        ds_name = body.get("window_dataset")
        if not ds_name:
            raise _http_error(WindowStoreError(
                "window_dataset_required", "hay que decir que dataset se revisa",
                "eligelo en la lista de /review/datasets"))
        split = body.get("split", "val")
        manifest, source = _review_ctx(ds_name)
        indices = wstore.split_map(ds_name).get(split) or []
        if not indices:
            raise _http_error(RunError(
                "split_empty", f"el split '{split}' de '{ds_name}' no tiene imagenes",
                "elige otro split o reconstruye el dataset"))
        count = max(1, min(int(body.get("count", 10)), REVIEW_MAX))
        # `indices` explicitos: es como la pagina de detalle pide UNA imagen. Se
        # admite aqui en vez de darle un endpoint propio para que haya UN solo
        # camino de inferencia -- si el detalle tuviera el suyo, las cajas de la
        # miniatura y las del detalle podrian dejar de ser las mismas y nadie
        # sabria cual de las dos mirar.
        if body.get("indices"):
            pedidos = [int(i) for i in body["indices"]][:REVIEW_MAX]
            fuera = [i for i in pedidos if i not in set(indices)]
            if fuera:
                raise _http_error(RunError(
                    "index_not_in_split",
                    f"{fuera} no esta(n) en el split '{split}' de '{ds_name}'",
                    "abre la imagen desde su split, o cambia de split"))
            trozo = pedidos
            offset = indices.index(pedidos[0]) if pedidos else 0
        else:
            offset = max(0, min(int(body.get("offset", 0)),
                                max(0, len(indices) - 1)))
            trozo = indices[offset:offset + count]

        run_name = body.get("run") or None
        model = None
        if run_name:
            if not (runs.path(run_name) / "best.pt").exists():
                raise _http_error(RunError(
                    "run_has_no_checkpoint",
                    f"'{run_name}' no tiene best.pt en esta maquina: no puede inferir",
                    "los pesos de un run no viajan por git (*.pt esta en el "
                    ".gitignore del repo de datos); elige un run `demo-*`, que es "
                    "la excepcion y SI viaja, o entrena aqui, o mira sin run"))
            model = _model_for(run_name)

        arrays = None if source is not None else wstore.arrays(ds_name)
        fila = ({} if arrays is None
                else {int(a): i for i, a in enumerate(arrays["images_sample_idx"])})
        kinds = set(manifest["config"]["target_kinds"])
        marcadas = set(review_mod.marked_in(ds_name, split))
        knobs, imgs = None, []
        for idx in trozo:
            if source is not None:
                s = source.sample_at(int(idx))
                pix = s.load_image()
                W, H = s.width, s.height
                # el filtro por kind NO es opcional: es el mismo que usa
                # task_score, y sin el la "verdad" dibujada no seria la que se
                # puntua
                truth = [b.quad.tolist() for b in s.blocks if b.kind in kinds]
                true_bbox = [b.bbox for b in s.blocks if b.kind in kinds]
            else:
                pix = arrays["images"][fila[int(idx)]]
                H, W = pix.shape
                truth, true_bbox = [], []
            fila_img = {"index": int(idx), "width": int(W), "height": int(H),
                        "truth": truth, "marked": int(idx) in marcadas}
            if model is not None:
                out = predict_image(
                    model, pix,
                    threshold=float(body.get("threshold", 0.5)),
                    stride=body.get("stride"), nms_radius=body.get("nms_radius"),
                    min_size=body.get("min_size"))
                knobs = out["knobs"]
                fila_img["paragraphs"] = out["paragraphs"]
                if source is not None:
                    pred = [(p["x0"], p["y0"], p["x1"], p["y1"])
                            for p in out["paragraphs"]]
                    r = paragraph_f1(pred, true_bbox,
                                     float(body.get("iou_threshold", 0.5)))
                    fila_img.update({"f1": r["f1"], "tp": r["tp"],
                                     "fp": r["fp"], "fn": r["fn"]})
            else:
                fila_img["paragraphs"] = []
            imgs.append(fila_img)

        review_mod.record_review(
            window_dataset=ds_name, split=split,
            source=manifest.get("source_id", ""), run=run_name or "",
            indices=[int(i) for i in trozo], offset=offset,
            knobs={**(knobs or {})})
        vistos = review_mod.reviewed_indices(ds_name, split)
        return {
            "run": run_name, "window_dataset": ds_name, "split": split,
            "source": manifest.get("source_id"),
            "truth_available": source is not None,
            "inferred": model is not None,
            "image_base": _image_base(ds_name, manifest, source),
            "offset": offset, "count": len(imgs),
            "total": len(indices), "images": imgs, "knobs": knobs,
            "reviewed": len(vistos),
            "pending": len([i for i in indices if i not in set(vistos)]),
            "next_offset": review_mod.next_unreviewed_offset(
                indices, vistos, count),
            "storage": review_mod.donde_se_guarda(),
        }

    @app.get("/review/sessions")
    def review_sessions(days: int | None = None, window_dataset: str | None = None,
                        split: str | None = None):
        """El historial de rangos mirados, lo mas reciente primero."""
        return {"sessions": review_mod.reviews(
            window_dataset=window_dataset, split=split, since_days=days)}

    @app.get("/review/marks")
    def review_marks():
        return {"marks": review_mod.mark_list(),
                "storage": review_mod.donde_se_guarda()}

    @app.post("/review/marks")
    def review_set_mark(body: dict):
        return review_mod.set_mark(
            window_dataset=body["window_dataset"], split=body["split"],
            index=int(body["index"]), marked=bool(body.get("marked", True)),
            note=str(body.get("note", "")), source=str(body.get("source", "")),
            run=str(body.get("run", "")))

    # ---------------------------------------------------------------- sweeps (H)
    @app.get("/sweeps")
    def list_sweeps():
        return {"sweeps": sstore.list()}

    @app.get("/sweeps/axes")
    def sweep_axes():
        """The axis vocabulary, from the SAME constants the validator uses — so a
        select can't drift from what check_sweep/study accepts (the 'define it
        twice' trap). geometry_auto = the axes whose range may be 'auto' (their
        range is CALCULATED); channels_indexed is the special OAT sub-axis."""
        from fv.sweeps.spec import (GEOMETRY_AUTO, LOSS_WEIGHT_PARAMS,
                                    NETWORK_PARAMS, OBJECTIVES, RECIPE_PARAMS,
                                    WINDOW_SIZE_FIELDS)
        return {"network": sorted(NETWORK_PARAMS), "recipe": sorted(RECIPE_PARAMS),
                "geometry_auto": sorted(GEOMETRY_AUTO),
                "channels_indexed": "channels[i]",
                # the rest of the vocabulary, from the same constants the
                # validators use: objectives, the fields that can never be axes,
                # and the loss weights that contract (9) forbids ranking by loss
                "objectives": sorted(OBJECTIVES),
                "loss_weight_params": sorted(LOSS_WEIGHT_PARAMS),
                "window_size_fields": sorted(WINDOW_SIZE_FIELDS)}

    @app.post("/sweeps", status_code=202)
    def create_sweep(body: dict):
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, {"code": "name_required",
                                      "message": "falta el nombre", "hint": ""})
        # base by NAME or inline VALUE, never both, never neither (D-H2, formatos §4.4)
        base_name = body.get("base_network")
        base_value = body.get("base_network_value")
        if bool(base_name) == bool(base_value):
            raise HTTPException(400, {
                "code": "base_network_xor_value",
                "message": "el recorrido necesita base_network (nombre) O "
                           "base_network_value (inline), exactamente uno",
                "hint": "da el nombre de una red del catalogo, o el config inline"})
        net = nstore.get(base_name) if base_name else base_value
        recipe = rstore.get(body["base_recipe"])
        manifest = wstore.manifest(body["window_dataset"])
        problems = check_run(manifest, full_config(net))
        if problems:
            raise HTTPException(400, problems[0])
        spec = {
            "window_dataset": body["window_dataset"],
            "base_network": base_name,
            "base_network_value": net,
            "base_label": body.get("base_label"),
            "base_recipe": body["base_recipe"],
            "base_recipe_value": recipe.as_dict(),
            "space": body.get("space", {}),
            "strategy": body.get("strategy", "grid"),
            "objective": body.get("objective", "f1"),
            "budget": body.get("budget", {}),
            "device": body.get("device", "cpu"),
            "seed": body.get("seed", 1),
        }
        enriched = prepare_sweep(name, spec, net, sstore)  # 400 BEFORE reserving

        def work(is_cancelled):
            return run_sweep(name, sstore, runs)
        job = jobs.submit("sweep", work, {"sweep": name},
                          on_cancel=lambda: sstore.request_stop(name))
        return {"job": job, "points": len(enriched["points"]),
                "discarded": len(enriched["discarded"])}

    @app.post("/sweeps/generate", status_code=202)
    def generate_sweep_ep(body: dict):
        """P1: derive an inline base from B's window_size and sweep one axis
        (barrido-por-ejes.md §8). The base is validated with the same check_run
        as every door, inside generate_sweep, BEFORE reserving the name."""
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, {"code": "name_required",
                                      "message": "falta el nombre", "hint": ""})
        enriched = generate_sweep(
            name, body["window_dataset"], body["axis"], body["range"],
            base_recipe=body.get("base_recipe", "corta"),
            objective=body.get("objective", "f1"),
            strategy=body.get("strategy", "grid"),
            budget=body.get("budget", {}),
            device=body.get("device", "cpu"), seed=body.get("seed", 1),
            seeds=body.get("seeds", 1),
            winners=body.get("winners"), overrides=body.get("overrides"),
            border_px=body.get("border_px"), study=body.get("study"), sstore=sstore)

        def work(is_cancelled):
            return run_sweep(name, sstore, runs)
        job = jobs.submit("sweep", work, {"sweep": name},
                          on_cancel=lambda: sstore.request_stop(name))
        return {"job": job, "base_label": enriched["base_label"],
                "points": len(enriched["points"]),
                "discarded": len(enriched["discarded"]),
                "corrections": enriched.get("corrections", [])}

    @app.get("/sweeps/{name}/trials")
    def get_sweep_trials(name: str):
        return sweep_trials(name, sstore, runs)

    @app.get("/sweeps/{name}/winner")
    def get_sweep_winner(name: str, delta: float | None = None,
                         cost_metric: str = "seconds_per_epoch"):
        """D-W1: SUGGEST the cheapest point within δ of the best (the user
        confirms before carrying it). Omitting δ derives it from the seed
        dispersion the sweep measured (1-SE, protocolo §1.5); passing one
        overrides it. The cost metric is an input."""
        return suggest_winner(name, delta=delta, cost_metric=cost_metric,
                              store=sstore, run_store=runs)

    @app.get("/sweeps/{name}")
    def sweep_detail(name: str):
        # reconcile-on-read: a sweep whose owner process died is healed to
        # 'interrupted' here, so a crash never shows 'running' forever
        return {"name": name, "spec": sstore.spec(name),
                "state": sstore.reconcile(name)}

    @app.post("/sweeps/{name}/stop")
    def stop_sweep(name: str):
        sstore.request_stop(name)
        return {"stopping": name}

    @app.delete("/sweeps/{name}")
    def delete_sweep_ep(name: str):
        # cascade: a child run can't be deleted alone (its points compare
        # together), so the sweep owns them — orchestration in the runner
        return delete_sweep(name, sstore, runs)

    @app.post("/sweeps/{name}/resume", status_code=202)
    def resume_sweep(name: str):
        sstore.spec(name)  # 404 if missing
        sstore.clear_stop(name)  # pressing the button IS changing your mind

        def work(is_cancelled):
            return run_sweep(name, sstore, runs)
        return {"job": jobs.submit("sweep", work, {"sweep": name},
                                   on_cancel=lambda: sstore.request_stop(name))}

    # --------------------------------------------------------------- studies (I)
    @app.get("/studies")
    def list_studies():
        # each row carries the SAME derived fields the detail computes
        # (fv.studies.driver.summarize) — a list that lacks them forces the
        # screen to re-derive "what is this waiting for", and the two drift
        from fv.studies.driver import summarize
        return {"studies": [{**s, **summarize(s["progress"])}
                            for s in studies_store.list()]}

    @app.post("/studies", status_code=201)
    def create_study_ep(body: dict):
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, {"code": "name_required",
                                      "message": "falta el nombre", "hint": ""})
        plan = {k: body[k] for k in
                ("window_dataset", "base_recipe", "objective", "seeds", "axes", "budget")
                if k in body}
        return create_study(name, plan, studies_store)

    @app.get("/studies/{name}")
    def study_status(name: str):
        return study_status_fn(name, studies_store)

    @app.post("/studies/{name}/advance", status_code=202)
    def advance_study_ep(name: str, body: dict | None = None):
        """Generate the next step's sweep (inline base + carried winners) and
        launch it. The WINNER is still the user's to confirm afterwards."""
        out = advance(name, studies_store, sstore,
                      budget=(body or {}).get("budget"))
        sweep_name = out["step"]["sweep"]

        def work(is_cancelled):
            return run_sweep(sweep_name, sstore, runs)
        job = jobs.submit("sweep", work, {"sweep": sweep_name},
                          on_cancel=lambda: sstore.request_stop(sweep_name))
        return {"step": out["step"], "base_label": out["spec"]["base_label"],
                "points": len(out["spec"]["points"]),
                "discarded": len(out["spec"]["discarded"]), "job": job}

    @app.post("/studies/{name}/confirm")
    def confirm_study_ep(name: str, body: dict):
        """Record the user-confirmed winner point of the current step and carry
        it forward (§7). The point is the user's choice — usually the suggestion
        from GET /sweeps/{sweep}/winner, but the user decides."""
        point = body.get("point")
        if point is None:
            raise HTTPException(400, {"code": "point_required",
                                      "message": "falta el punto ganador a confirmar",
                                      "hint": "manda {point: {<eje>: <valor>}}"})
        return confirm(name, point, studies_store)

    @app.delete("/studies/{name}")
    def delete_study_ep(name: str):
        # cascade to the sweeps the study generated (and their runs): leaving
        # them orphaned collides on the next same-name advance (sweep_exists)
        return delete_study(name, studies_store, sstore, runs)

    # ------------------------------------------------------------------ jobs (X)
    @app.get("/jobs")
    def list_jobs():
        return {"jobs": jobs.list()}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, {"code": "job_not_found",
                                      "message": f"no existe el job {job_id}",
                                      "hint": "los jobs viven en memoria: un reinicio los olvida"})
        return j

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, {"code": "job_not_found",
                                      "message": f"no existe el job {job_id}", "hint": ""})
        return {"cancelling": job_id}

    # ------------------------------------------- UI state (remembered defaults)
    # A committable snapshot of the front's filters/forms so a working session
    # travels to the GPU server. Opaque blob: NOT a domain source of truth, so
    # no schema is enforced here — only a size bound (R6) to keep it a
    # convenience, not a data store.
    UI_STATE_MAX = 256 * 1024

    @app.get("/ui-state")
    def get_ui_state():
        p = settings.ui_state_path()
        return read_json_retrying(p) if p.exists() else {}

    @app.put("/ui-state")
    def put_ui_state(body: dict):
        import json
        if len(json.dumps(body)) > UI_STATE_MAX:
            raise HTTPException(400, {
                "code": "ui_state_too_large",
                "message": f"el estado de UI supera {UI_STATE_MAX // 1024} KB",
                "hint": "esto guarda filtros y formularios, no datos: revisa qué envías"})
        p = settings.ui_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(p, body)
        return {"saved": True, "path": str(p)}

    return app
