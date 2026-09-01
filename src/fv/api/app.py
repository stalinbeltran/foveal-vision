"""The API: HTTP and nothing else (api.md §0). One resource per domain (R1),
no ambiguous words (R2), sync vs job at ~1 s (R3), every error carries
code/message/hint (R4), incremental polling (R5), aggregates server-side (R6),
names not values (R7). CORS closed to the front origin; images resolve inside
the domain, never by client path.
"""

from __future__ import annotations

import dataclasses
import os
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from fv import errores, settings
from fv.api.jobs import JobQueue
from fv.ioutils import read_json_retrying, write_json_atomic
from fv.datasets.loader import SourceDataset, SourceError, discover_sources
from fv.fovea import (FoveaError, build_search_space, check_dims, derive_dims,
                      normalize_geometry)
from fv.inference import catalogo
from fv.inference.catalogo import CatalogoError
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
                 StudyError, StudyStoreError, CheckpointError, FoveaError,
                 CatalogoError)

NOT_FOUND_CODES = {"source_not_found", "sample_not_found", "window_dataset_missing",
                   "network_not_found", "recipe_not_found", "run_not_found",
                   "sweep_not_found", "study_not_found"}
# Techo del cuerpo de una subida de pesos. El `last.pt` de la red vigente son
# 2,0 MB (medido 2026-08-30) y `best.pt` 680 KB, asi que 64 MB deja sitio de
# sobra para una red bastante mayor y sigue siendo un techo: sin limite, una
# subida es un disco lleno, y un disco lleno tumba el entrenamiento que estaba
# corriendo. Se puede subir con FV_MAX_CHECKPOINT_MB para una red grande de
# verdad -- que es declararlo, no descubrirlo a mitad.
MAX_CHECKPOINT_BYTES = int(os.environ.get("FV_MAX_CHECKPOINT_MB", "64")) * 1024 * 1024

CONFLICT_CODES = {"window_dataset_exists", "window_dataset_in_use", "network_exists",
                  "recipe_exists", "recipe_in_use", "run_exists", "run_is_running", "sweep_exists",
                  "run_without_provenance", "run_has_no_checkpoint", "run_belongs_to_sweep",
                  "window_dataset_changed", "split_empty", "sweep_is_running",
                  "study_exists", "step_awaiting_confirmation", "study_has_live_sweeps",
                  "run_not_approved_for_inference", "antesala_vacia", "no_aprobada"}


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

    # AUTOCHEQUEO DE ARRANQUE. La clase de averia que ningun test encuentra es la
    # que aparece porque este PROCESO lleva vivo desde antes que el artefacto: un
    # test corre siempre una version de todo. Cargar aqui lo que la app dice poder
    # inferir convierte "lo descubre el usuario al pulsar, 8 horas despues" en
    # "lo dice el log a los 3 segundos" (2026-09-01).
    #
    # ⚠ No impide arrancar: la app sirve datasets, runs, recorridos y estudios, y
    # un `.pt` que no carga no puede tumbar todo eso. Degrada y lo DECLARA (R2).
    try:
        _arranque = catalogo.autochequeo(runs)
    except Exception as e:                              # noqa: BLE001
        # ni siquiera el chequeo puede tumbar el arranque; pero se dice que no se
        # pudo hacer, que no es lo mismo que "todo bien"
        _arranque = [{"run": "?", "ok": False, "code": "autochequeo_fallido",
                      "message": str(e), "hint": "mira el log del servicio"}]
    _rotas = [f for f in _arranque if not f["ok"]]
    for _f in _rotas:
        # queda en el log ADEMAS de en el arranque: el log del proceso se pierde
        # al reiniciar, y esto es justo lo que hay que poder mirar tres dias
        # despues ("¿desde cuando no cargaba?")
        errores.registrar(_f["code"], _f["message"], hint=_f.get("hint", ""),
                          origen="arranque", donde=f"red {_f['run']}")
    if _rotas:
        print(f"\n[AVISO] {len(_rotas)} de {len(_arranque)} redes servibles NO CARGAN "
              f"con este proceso:", flush=True)
        for f in _rotas:
            print(f"    {f['run']}: [{f['code']}] {f['message']}\n"
                  f"      -> {f['hint']}", flush=True)
        print("", flush=True)
    else:
        print(f"inferencia: {len(_arranque)} red(es) servible(s), todas cargan",
              flush=True)

    @app.exception_handler(Exception)
    async def _domain_handler(request, exc):
        from fastapi.responses import JSONResponse
        donde = f"{request.method} {request.url.path}"
        if isinstance(exc, DOMAIN_ERRORS):
            he = _http_error(exc)
            d = he.detail
            # `rechazo` y no `error`: una negativa con su razon es la puerta
            # FUNCIONANDO, y hay 109 codigos de esos. Mezclarlos con los fallos
            # inesperados haria un log que nadie lee. Van los dos, y la pantalla
            # filtra a `error` por defecto.
            errores.registrar(d.get("code", "?"), d.get("message", ""),
                              hint=d.get("hint", ""), nivel="rechazo",
                              origen="api", donde=donde)
            return JSONResponse(status_code=he.status_code,
                                content={"detail": he.detail})
        # lo INESPERADO: esto es un 500, nadie lo ha declarado y sin log se
        # pierde en cuanto el proceso se reinicie
        errores.registrar(type(exc).__name__, str(exc), nivel="error",
                          origen="api", donde=donde, traza=exc,
                          hint="no es una negativa declarada: mira la traza")
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
        """Los runs, y con cada uno si la app puede INFERIR con el.

        `has_checkpoint` lo pone E (hay un best.pt en su directorio) y
        `inference` lo pone F (de donde saldria el modelo: 'antesala' mientras
        entrena, 'catalogo' si esta aprobada, null si no hay). Son dos hechos
        distintos y por eso son dos campos: un run puede tener el fichero en
        disco y NO estar aprobado, y entonces la app no lo usa.

        El catalogo y la antesala se leen UNA vez para los 862 runs, no una por
        run: son dos lecturas de disco contra 1.724."""
        try:
            aprobadas = set(catalogo.leer()["runs"])
        except CatalogoError:
            aprobadas = set()          # el detalle se ve en GET /inference
        antesala = catalogo.antesala_completa()
        out = runs.list()
        for r in out:
            en_antesala = catalogo.CHECKPOINT_INFERENCIA in antesala.get(r["name"], [])
            r["approved"] = r["name"] in aprobadas
            r["inference"] = ("antesala" if en_antesala
                              else "catalogo" if (r["approved"] and r["has_checkpoint"])
                              else None)
        return {"runs": out}

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
    def _model_para_mirar(name: str):
        """Los pesos de ESTE run, para INTROSPECCION (V1/V2/F0: kernels, mapas,
        vista de entrada). No exige aprobacion, y esa es una decision:

        introspeccionar no es inferir. La regla del dueno ("solo las redes
        aprobadas se usan en la app para inferir") existe para que la app no
        ensene cajas de una red que nadie eligio; mirar los kernels de un run
        que has pedido por su nombre no tiene ese riesgo -- estas mirando esa
        red a proposito, y el resultado no se parece a una prediccion.

        Y si lo exigiera romperia el flujo local: `fv-train` deja `best.pt` en el
        directorio del run, no en la antesala, asi que un run recien entrenado
        aqui no se podria ni mirar sin aprobarlo antes -- o sea sin commitear
        2,7 MB solo para ver que salio."""
        p = staging_o_disco(name)
        if p is None:
            raise _http_error(RunError(*_por_que_no_infiere(name)))
        return MODEL_CACHE.get(p)

    def staging_o_disco(name: str):
        p = catalogo.staging_dir(name) / catalogo.CHECKPOINT_INFERENCIA
        if p.exists():
            return p
        p = runs.path(name) / catalogo.CHECKPOINT_INFERENCIA
        return p if p.exists() else None

    def _model_para_inferir(name: str):
        """El modelo con el que INFERIR (predecir, revisar, puntuar la tarea), o
        el motivo exacto de que no lo haya. Aqui SI manda el catalogo.

        Tres negativas distintas, y distinguirlas es el punto: el mensaje viejo
        decia siempre "espera a que termine una epoca", que sobre un run
        terminado hace dias con 107 epocas es simplemente falso -- y manda a
        esperar algo que no va a pasar."""
        ckpt, _origen = catalogo.checkpoint_de(name, runs)
        if ckpt is None:
            raise _http_error(RunError(*_por_que_no_infiere(name)))
        return MODEL_CACHE.get(ckpt)

    def _por_que_no_infiere(name: str) -> tuple:
        """(code, message, hint) para un run sin pesos utilizables."""
        st = (runs.status(name) or {}).get("status")
        if st == "running":
            return ("run_has_no_checkpoint",
                    f"'{name}' esta entrenando y aun no ha dejado un best.pt",
                    "espera a que termine una epoca")
        if (runs.path(name) / catalogo.CHECKPOINT_INFERENCIA).exists():
            # el fichero esta, pero nadie aprobo esta red: servirlo seria inferir
            # con una que no se eligio
            return ("run_not_approved_for_inference",
                    f"'{name}' tiene pesos en disco pero NO esta aprobada para "
                    f"inferir",
                    f"apruebala con POST /inference/staging/{name}/promote, o "
                    f"elige una de {catalogo.aprobadas() or '(ninguna)'}")
        return ("run_has_no_checkpoint",
                f"'{name}' no tiene pesos en esta maquina",
                "los pesos de un run solo se guardan si se APRUEBA para "
                "inferencia (fv.inference.catalogo); las aprobadas hoy son "
                f"{catalogo.aprobadas() or '(ninguna)'}. Para tener los de este "
                "run hay que reentrenarlo: los de un run que no se aprobo nunca "
                "existieron")

    # -------------------------------------------- pesos para inferencia (antesala)
    #
    # La antesala recibe los pesos MIENTRAS se entrena y no toca el repo de
    # datos; la promocion los pasa a lo definitivo Y aprueba la red. Por que las
    # dos cosas, y por que en este orden, en `fv.inference.catalogo`.
    #
    # ⚠ QUIEN PUEDE LLAMAR A ESTO. Estas rutas van dentro del mismo `app` que
    # monta `fv.api.web`, asi que heredan su puerta: token (cabecera `x-fv-token`,
    # cookie o `?t=`) salvo desde loopback. No hay una segunda puerta y no debe
    # haberla -- dos puertas divergen y la que se olvida es la que se deja
    # abierta. Lo que SI cambia respecto al resto del API es la consecuencia de
    # que se cuele alguien: esto escribe FICHEROS DE PESOS, y un `.pt` es un
    # pickle, o sea codigo. Por eso:
    #   - el nombre del fichero se comprueba contra una lista de DOS (`PESOS`),
    #     nunca se compone una ruta con lo que llega;
    #   - el nombre del run tiene que ser un nombre de directorio;
    #   - se guardan BYTES y no se hace `torch.load` en la subida: cargar es lo
    #     que ejecuta el pickle, y aqui no hay ninguna razon para hacerlo.
    # Con la puerta cerrada esto es "quien tiene el token puede escribir en la
    # antesala", que es menos de lo que ya puede hacer (el API borra runs y
    # datasets sin preguntar).

    @app.get("/inference")
    def inference_state():
        """Que redes puede usar la app para inferir, y de donde salen."""
        try:
            cat = catalogo.leer()
        except CatalogoError as e:
            raise _http_error(e)
        # El autochequeo se REPITE aqui en vez de servir el del arranque: entre
        # medias se promueve, se sube a la antesala y se retira del catalogo, asi
        # que el del arranque envejece. Lo que no envejece es el del arranque
        # como AVISO -- por eso estan los dos y dicen cosas distintas.
        chequeo = catalogo.autochequeo(runs)
        return {
            "aprobadas": cat["runs"],
            "cargan": chequeo,
            "rotas": [f for f in chequeo if not f["ok"]],
            "antesala": catalogo.antesala_completa(),
            "antesala_root": str(settings.inference_staging_root()),
            "catalogo": str(catalogo.catalogo_path()),
            "pesos": list(catalogo.PESOS),
            "checkpoint_inferencia": catalogo.CHECKPOINT_INFERENCIA,
        }

    @app.put("/inference/staging/{name}/{fichero}", status_code=201)
    async def upload_checkpoint(name: str, fichero: str, request: Request):
        """Recibe UN fichero de pesos y lo deja en la antesala de `name`.

        Cuerpo = los bytes del `.pt`, en crudo (`application/octet-stream`), no
        multipart: quien sube esto es un script, y `curl --data-binary @best.pt`
        no necesita una biblioteca de formularios. Se escribe atomico, porque el
        fichero se lee mientras se reemplaza (la revision usa el modelo con el
        entrenamiento en marcha)."""
        datos = await request.body()
        if not datos:
            raise _http_error(RunError(
                "checkpoint_vacio", f"el cuerpo de '{fichero}' llego vacio",
                "manda los bytes del .pt en el cuerpo "
                "(curl --data-binary @best.pt)"))
        if len(datos) > MAX_CHECKPOINT_BYTES:
            raise _http_error(RunError(
                "checkpoint_demasiado_grande",
                f"{len(datos)} bytes supera el limite de {MAX_CHECKPOINT_BYTES}",
                "el last.pt de la red vigente son 2,0 MB; si el tuyo es "
                "legitimamente mayor, sube FV_MAX_CHECKPOINT_MB"))
        try:
            destino = catalogo.guardar_en_antesala(name, fichero, datos)
        except CatalogoError as e:
            raise _http_error(e)
        return {"run": name, "fichero": fichero, "bytes": len(datos),
                "destino": str(destino), "antesala": catalogo.en_antesala(name)}

    @app.post("/inference/staging/{name}/promote")
    def promote_checkpoint(name: str, body: dict | None = None):
        """Antesala -> repo de datos, y aprueba la red. Es UNA decision."""
        body = body or {}
        try:
            return catalogo.promover(name, runs, motivo=body.get("motivo", ""),
                                     origen=body.get("origen", ""))
        except CatalogoError as e:
            raise _http_error(e)

    @app.delete("/inference/staging/{name}", status_code=200)
    def clear_staging(name: str):
        try:
            return {"run": name, "borrados": catalogo.limpiar_antesala(name)}
        except CatalogoError as e:
            raise _http_error(e)

    @app.delete("/inference/approved/{name}", status_code=200)
    def retire_from_catalog(name: str):
        """Saca la red del catalogo. NO borra sus pesos: retirar es reversible
        y borrar no, y ademas un peso ya commiteado no se va del historial de
        git por borrarlo del arbol."""
        try:
            return catalogo.retirar(name)
        except CatalogoError as e:
            raise _http_error(e)

    @app.get("/runs/{name}/kernels")
    def run_kernels(name: str):
        return kernels_payload(_model_para_mirar(name))

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
        model = _model_para_mirar(name)
        img, wx0, wy0 = _probe_window(body)
        from fv.fovea import build_view
        view, cov = build_view(img, wx0, wy0, model.dims,
                               pool_mode=model.cfg["pool_mode"],
                               pad_mode=model.cfg["pad_mode"])
        return feature_maps_payload(model, view, cov)

    @app.post("/runs/{name}/input-view")
    def run_input_view(name: str, body: dict):
        model = _model_para_mirar(name)
        img, wx0, wy0 = _probe_window(body)
        return input_view_payload(model, img, wx0, wy0)

    # --------------------------------------------------------------- predict (F)
    @app.post("/runs/{name}/predict")
    def run_predict(name: str, body: dict):
        model = _model_para_inferir(name)
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
        try:
            aprobadas = set(catalogo.leer()["runs"])
        except CatalogoError:
            aprobadas = set()
        antesala = catalogo.antesala_completa()
        out = []
        for r in (runs.list() if todos is None else todos):
            if r.get("window_dataset") != ds_name:
                continue
            # `has_checkpoint` aqui significa "la app PUEDE inferir con el", que
            # desde el 2026-08-31 no es lo mismo que "el fichero esta en disco":
            # una red sin aprobar no se usa aunque tenga pesos. Ese es el filtro
            # que el dueno pidio, y el sitio donde ponerlo es este -- si se
            # dejara al front, cada pantalla tendria su propia idea de cual vale.
            origen = ("antesala" if catalogo.CHECKPOINT_INFERENCIA
                      in antesala.get(r["name"], [])
                      else "catalogo" if (r["name"] in aprobadas
                                          and r.get("has_checkpoint"))
                      else None)
            out.append({"name": r["name"], "status": r.get("status"),
                        "best": r.get("best"),
                        "has_checkpoint": origen is not None,
                        "inference": origen,
                        # se MARCA, no se esconde: distinguir "no esta aprobada"
                        # de "no tiene pesos" es lo que dice si hay que aprobar
                        # algo o reentrenar algo
                        "on_disk": bool(r.get("has_checkpoint"))})
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

        # Las DETECCIONES van a peticion, y en DOS niveles, porque cuestan muy
        # distinto. MEDIDO el 2026-09-01 sobre un lote de 10 imagenes:
        #
        #   sin nada                 2 KB
        #   con esquinas y crudas   35 KB   (102 esquinas · 316 crudas)
        #
        # o sea que la nube cruda es ~3x las esquinas -- y es justo lo que una
        # miniatura no puede ni dibujar. Por eso:
        #
        #   `with_detections` -> las ESQUINAS (post-NMS). Baratas: ~10 por
        #        imagen. Las pide tambien la rejilla, que es lo que permite ver
        #        las esquinas en las miniaturas sin cargar el movil.
        #   `with_raw`        -> ademas la nube PRE-NMS. Solo la pagina de
        #        detalle, que mira UNA imagen grande y ahi el punto es ver que
        #        vio la red antes de la caja.
        #
        # Se DECLARA en la peticion en vez de deducirse de len(indices)==1: una
        # regla implicita obliga al cliente a adivinar cuando recibira el campo,
        # y el dia que la rejilla pida una sola imagen cambiaria de payload sin
        # que nadie lo hubiera pedido.
        con_detecciones = bool(body.get("with_detections"))
        con_crudas = bool(body.get("with_raw"))

        run_name = body.get("run") or None
        model = None
        if run_name:
            # `_model_for` ya distingue los tres casos (entrenando / con pesos sin
            # aprobar / sin pesos) y lo dice. Antes habia aqui un mensaje propio
            # que decia que los pesos NO viajan por git y que eligieras un run
            # `demo-*`: cierto hasta el 2026-08-30, falso desde el commit que
            # abrio la tercera excepcion del .gitignore. Dos sitios que explican
            # lo mismo divergen, y el que se olvida es el que miente.
            model = _model_para_inferir(run_name)

        arrays = None if source is not None else wstore.arrays(ds_name)
        fila = ({} if arrays is None
                else {int(a): i for i, a in enumerate(arrays["images_sample_idx"])})
        kinds = set(manifest["config"]["target_kinds"])
        marcadas = set(review_mod.marked_in(ds_name, split))
        knobs, orden, imgs = None, None, []
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
                orden = out["corner_order"]
                fila_img["paragraphs"] = out["paragraphs"]
                if con_detecciones:
                    fila_img["corners"] = out["corners"]
                if con_crudas:
                    fila_img["raw"] = out["raw"]
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
            # el vocabulario de esquinas viaja con la respuesta indexada por el
            # (`fv.metrics` es su UNICA definicion): un lector que guarde su
            # copia se desincroniza en silencio. `None` sin modelo, porque sin
            # inferencia no hay ranuras de que hablar -- y ausente no es lista
            # vacia (formatos.md 1).
            "corner_order": orden,
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

    # ----------------------------------------------------------- errores (X)
    @app.get("/errores")
    def list_errores(nivel: str | None = None, code: str | None = None,
                     origen: str | None = None, q: str | None = None,
                     desde: str | None = None, hasta: str | None = None,
                     limit: int = 100, offset: int = 0):
        """El log de errores, filtrado y PAGINADO en el servidor.

        ⚠ El filtro y las cuentas van aqui y no en el navegador (U4.3). El dueno
        pidio esto contando con que habra muchos: mandar el fichero entero para
        filtrarlo en el front es exactamente lo que deja de funcionar el dia que
        de verdad haga falta.

        Devuelve tambien las FACETAS (cuantos por nivel/code/origen/version)
        porque con un log grande la pregunta no es "ensenamelos" sino "de que
        hay": sin ellas, filtrar es adivinar un valor a ciegas.
        """
        return errores.consultar(
            nivel=nivel, code=code, origen=origen, q=q, desde=desde, hasta=hasta,
            limit=max(1, min(int(limit), 500)), offset=max(0, int(offset)),
            sin_traza=True)

    @app.post("/errores/traza")
    def traza_de(body: dict):
        """La traza de UN error, a peticion.

        No viaja en la lista porque la pantalla solo la enseña al abrir una fila
        y el sondeo es cada 5 s. Va por POST y no por GET con la clave en la URL
        porque la clave es (cuando, code, donde) --con rutas dentro-- y meter eso
        en una query string lo deja en el log de acceso de este proceso.
        """
        d = errores.consultar(code=body.get("code"), desde=body.get("cuando"),
                              hasta=body.get("cuando"), limit=20)
        for e in d["errores"]:
            if e.get("cuando") == body.get("cuando") and e.get("donde") == body.get("donde"):
                return {"traza": e.get("traza") or None}
        # ausente != vacio: "no la encuentro" y "no tenia" son cosas distintas
        return {"traza": None, "motivo": "no encuentro ese error en el log"}

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
