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
from fv.fovea import FoveaError, build_search_space, check_dims, derive_dims
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
from fv.studies.driver import StudyError, advance, confirm, create_study
from fv.studies.driver import status as study_status_fn
from fv.studies.store import StudyStore, StudyStoreError
from fv.training.loop import train
from fv.training.recipe import Recipe, RecipeStore, RecipeStoreError
from fv.training.registry import RunError, RunStore
from fv.validation import check_run
from fv.windows.extract import ExtractConfig, ExtractError, extract_windows
from fv.windows.store import WindowDatasetStore, WindowStoreError

DOMAIN_ERRORS = (SourceError, ExtractError, WindowStoreError, NetworkStoreError,
                 RecipeStoreError, RunError, SweepError, SweepStoreError,
                 StudyError, StudyStoreError, CheckpointError, FoveaError)

NOT_FOUND_CODES = {"source_not_found", "sample_not_found", "window_dataset_missing",
                   "network_not_found", "recipe_not_found", "run_not_found",
                   "sweep_not_found", "study_not_found"}
CONFLICT_CODES = {"window_dataset_exists", "window_dataset_in_use", "network_exists",
                  "recipe_exists", "run_exists", "run_is_running", "sweep_exists",
                  "run_without_provenance", "run_has_no_checkpoint",
                  "window_dataset_changed", "split_empty", "sweep_is_running",
                  "study_exists", "step_awaiting_confirmation"}


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

    @app.get("/window-datasets/{name}")
    def window_dataset_detail(name: str):
        m = wstore.manifest(name)
        m["used_by"] = runs.used_by_dataset(name)
        return m

    @app.delete("/window-datasets/{name}")
    def delete_window_dataset(name: str):
        wstore.delete(name, runs.used_by_dataset(name))
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
                "pixels": crop.tolist(),
                "split": int(arrays["split"][index])}

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
        return {"networks": nstore.list()}

    @app.post("/networks")
    def save_network(body: dict):
        name = body.get("name", "")
        cfg = full_config(body)
        problems = check_dims(cfg["N"], cfg["c_frac"], cfg["d"], cfg["pen_frac"])
        if problems:
            p = problems[0]
            raise HTTPException(400, p)
        nstore.save(name, cfg, overwrite=bool(body.get("overwrite")))
        return {"saved": name}

    @app.post("/networks/validate")
    def validate_network(body: dict):
        cfg = full_config(body)
        from fv.validation import check_network
        problems = check_network(cfg)
        if problems:
            return {"valid": False, "problems": problems}
        trace = network_trace(cfg)
        dims = trace["dims"]
        space = build_search_space(cfg["N"], cfg["c_frac"], cfg["pen_frac"])
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
        return {"recipes": rstore.list(),
                "defaults": Recipe().as_dict()}

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
        if cfg.get("provenance", {}).get("sweep"):
            raise _http_error(RunError(
                "run_is_running",  # 409 family
                f"'{name}' pertenece al recorrido "
                f"'{cfg['provenance']['sweep']}'",
                "borra el recorrido entero o deja el run: sus puntos se comparan juntos"))
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

    @app.post("/runs/{name}/feature-maps")
    def run_feature_maps(name: str, body: dict):
        model = _model_for(name)
        arrays = wstore.arrays(body["window_dataset"])
        i = int(body["index"])
        lookup = {int(a): r for r, a in enumerate(arrays["images_sample_idx"])}
        img = arrays["images"][lookup[int(arrays["sample_idx"][i])]]
        wx0, wy0 = (int(v) for v in arrays["window_xy"][i])
        from fv.fovea import build_view
        view, _cov = build_view(img, wx0, wy0, model.dims,
                                pool_mode=model.cfg["pool_mode"],
                                pad_mode=model.cfg["pad_mode"])
        return feature_maps_payload(model, view)

    @app.post("/runs/{name}/input-view")
    def run_input_view(name: str, body: dict):
        model = _model_for(name)
        arrays = wstore.arrays(body["window_dataset"])
        i = int(body["index"])
        lookup = {int(a): r for r, a in enumerate(arrays["images_sample_idx"])}
        img = arrays["images"][lookup[int(arrays["sample_idx"][i])]]
        wx0, wy0 = (int(v) for v in arrays["window_xy"][i])
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

    # ---------------------------------------------------------------- sweeps (H)
    @app.get("/sweeps")
    def list_sweeps():
        return {"sweeps": sstore.list()}

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
            winners=body.get("winners"), overrides=body.get("overrides"),
            c_frac=body.get("c_frac"), study=body.get("study"), sstore=sstore)

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
    def get_sweep_winner(name: str, delta: float = 0.0,
                         cost_metric: str = "seconds_per_epoch"):
        """D-W1: SUGGEST the cheapest point within δ of the best (the user
        confirms before carrying it). δ and the cost metric are inputs."""
        return suggest_winner(name, delta=delta, cost_metric=cost_metric,
                              store=sstore, run_store=runs)

    @app.get("/sweeps/{name}")
    def sweep_detail(name: str):
        return {"name": name, "spec": sstore.spec(name), "state": sstore.state(name)}

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
        return {"studies": studies_store.list()}

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
        studies_store.delete(name)
        return {"deleted": name}

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
