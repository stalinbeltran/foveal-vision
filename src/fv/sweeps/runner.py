"""The sweep runner: sequential points (worker limit 1 on CPU), each point a
first-class run named {sweep}-{i:04d}-{axis} (point_run_name) with
provenance.sweep set, validated by
THE SAME gate as every other way of training. Resume = count what finished
(only done/cancelled) and redo the rest. Cooperative stop cuts twice: between
points, and — via should_stop — the point IN FLIGHT at its next epoch boundary.
The owner PID is recorded so a crashed 'running' can be reconciled (store).
"""

from __future__ import annotations

import dataclasses
import os
import re

from fv.ioutils import read_json_retrying
from fv.metrics import checkpoint_record
from fv.sweeps.spec import OBJECTIVES, SweepError, check_sweep, expand_points
from fv.sweeps.store import SweepStore, SweepStoreError
from fv.training.loop import train
from fv.training.recipe import Recipe
from fv.training.registry import RunError, RunStore


def _axis_token(overrides: dict) -> str:
    """A compact, filename/route/cp1252-safe tag of the sweep point, so a run's
    name carries its axis value — not just an opaque index. OAT moves ONE axis
    at a time, so the first override is normally the only one; if a point moves
    several fields (a grid, or a study fixing n_layers and expanding channels)
    we tag the FIRST and let the index disambiguate the rest."""
    if not overrides:
        return ""
    k, v = next(iter(overrides.items()))
    if isinstance(v, (list, tuple)):
        vs = "x".join(str(x) for x in v)
    else:
        vs = str(v).replace(".", "p").replace("-", "m")
    return re.sub(r"[^0-9A-Za-z_]+", "", f"{k}{vs}")


def point_run_name(sweep_name: str, i: int, overrides: dict) -> str:
    """The single source of truth for a child run's name — used by BOTH the
    runner (that creates it) and sweep_trials (that maps a point back to it). If
    these two ever built the name differently the sweep would show every point as
    'pending' forever. Index stays for a stable, collision-free identity; the
    axis token rides along so the name is legible.

    `seed` is the replica axis, not the real one (§11.1): tag from the real axis
    and append the seed as a suffix, so N replicas of one value read as
    `...-batch_size10_seed1 .. _seed5` instead of five opaque indices — and the
    'group by config-minus-seed' view has a legible name to lean on."""
    seed = overrides.get("seed")
    primary = {k: v for k, v in overrides.items() if k != "seed"}
    tag = _axis_token(primary)
    if seed is not None:
        tag = f"{tag}_seed{seed}" if tag else f"seed{seed}"
    return f"{sweep_name}-{i:04d}-{tag}" if tag else f"{sweep_name}-{i:04d}"


def prepare_sweep(name: str, spec: dict, base_network: dict,
                  store: SweepStore | None = None) -> dict:
    """Validate + expand + persist. Returns the enriched spec (points declared,
    discards declared with reasons)."""
    store = store or SweepStore()
    problems = check_sweep(spec)
    if problems:
        p = problems[0]
        raise SweepError(p["code"], p["message"], p["hint"])
    valid, discarded = expand_points(spec, base_network)
    if not valid:
        # every value failed — carry WHY (each discard has its reason) so the
        # user sees which value to fix, not just "all invalid" (formatos §2)
        def _fmt(pt: dict) -> str:
            return ", ".join(f"{k}={v}" for k, v in pt.items())
        reasons = "; ".join(
            f"{_fmt(d['point'])}: {d['problems'][0]['message']}"
            for d in discarded[:8])
        if len(discarded) > 8:
            reasons += f"; … (+{len(discarded) - 8})"
        raise SweepError("no_valid_points",
                         "todos los valores del eje son geometricamente invalidos",
                         f"cada valor y su razon: {reasons}")
    enriched = dict(spec)
    enriched["name"] = name
    enriched["points"] = [p["overrides"] for p in valid]
    enriched["discarded"] = discarded
    store.create(name, enriched)
    return enriched


def run_sweep(name: str, store: SweepStore | None = None,
              run_store: RunStore | None = None, progress=None,
              only_seed: int | None = None) -> dict:
    """Start OR resume: counts finished child runs and runs the rest.

    `only_seed` corre SOLO los puntos de esa semilla, para repartir un recorrido
    entre varias maquinas (una por semilla) sin partirlo en varios recorridos.
    Recorre la lista ENTERA y salta lo ajeno, asi que el indice de cada punto
    -y con el su `point_run_name`- es el mismo que tendria corriendo aqui de
    corrido: por eso los runs de N maquinas se juntan despues en UN solo
    recorrido y `sweep_trials` los encuentra por su nombre, sin renombrar nada.

    La semilla es el eje REPLICA, y esa es justamente la razon de repartir por
    ella y no por el eje real: cada maquina mide TODOS los valores del eje, asi
    que si una maquina resulta ser mas lenta o mas rara que otra, eso entra
    igual en todos los valores que se comparan en vez de cargarselo a uno solo.
    Repartir por valor de `lr` haria lo contrario: confundir la maquina con la
    respuesta."""
    store = store or SweepStore()
    run_store = run_store or RunStore()
    spec = store.spec(name)
    base_network = spec["base_network_value"]
    base_recipe = Recipe(**spec["base_recipe_value"])
    epochs = int((spec.get("budget") or {}).get("epochs", 0) or 0)
    objective = spec.get("objective", "f1")

    valid, _ = expand_points(spec, base_network)
    mios = list(range(len(valid)))
    if only_seed is not None:
        mios = [i for i in mios if valid[i]["overrides"].get("seed") == only_seed]
        if not mios:
            # ruidoso: una maquina que no encuentra sus puntos y termina "bien"
            # deja un hueco en el recorrido que solo se ve al juntar los runs
            raise SweepError(
                "no_points_for_seed",
                f"el recorrido '{name}' no tiene ningun punto con seed={only_seed}",
                f"las semillas del recorrido son "
                f"{sorted({p['overrides'].get('seed') for p in valid})}")
    pid = os.getpid()
    store.set_state(name, "running", total=len(mios), pid=pid)
    done = 0
    for i in mios:
        point = valid[i]
        run_name = point_run_name(name, i, point["overrides"])
        if run_store.exists(run_name):
            st = run_store.status(run_name).get("status")
            if st in ("done", "cancelled"):
                done += 1
                continue
            # anything else — error, running, queued, unknown, interrupted — is a
            # point that never finished (crash/hibernation/reconciled): drop and
            # redo. Only done/cancelled count as finished; the rest we redo.
            try:
                run_store.path(run_name).joinpath("status.json").unlink(missing_ok=True)
            except OSError:
                pass
            for f in sorted(run_store.path(run_name).rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            run_store.path(run_name).rmdir()
        if store.stop_requested(name):
            store.set_state(name, "stopped", done=done, total=len(mios))
            return store.state(name)
        recipe = dataclasses.replace(base_recipe, **point["recipe_overrides"])
        # budget epochs caps the run, but it must NOT silently override a point
        # that sweeps `epochs` itself — that would collapse the axis to one value
        # and "the axis does nothing" (exactly what breaks a sweep unnoticed)
        if epochs and "epochs" not in point["recipe_overrides"]:
            recipe = dataclasses.replace(recipe, epochs=epochs)
        try:
            train(run_name, spec["window_dataset"], spec["base_network"],
                  point["network"], spec["base_recipe"], recipe,
                  device=spec.get("device", "cpu"), sweep=name, store=run_store,
                  # a stop asked of the sweep also cuts the point in flight at its
                  # next epoch boundary — not only between points
                  should_stop=lambda: store.stop_requested(name))
        except RunError as e:
            # declared, never silent: the point failed with its reason
            store.set_state(name, "running", done=done, total=len(mios), pid=pid,
                            last_error={"run": run_name, "code": e.code,
                                        "message": e.message})
            continue
        done += 1
        store.set_state(name, "running", done=done, total=len(mios), pid=pid)
        if progress:
            progress(done, len(mios), run_name)
    store.set_state(name, "done", done=done, total=len(mios))
    return store.state(name)


def delete_sweep(name: str, store: SweepStore | None = None,
                 run_store: RunStore | None = None) -> dict:
    """Delete a sweep AND its child runs, as one unit. A child run refuses to be
    deleted on its own (it belongs to the sweep — its points are compared
    together), so the sweep owns them: removing the sweep cascades to its runs.
    Refuses while anything is live, with the reason and the fix — never orphans a
    run and never deletes work in progress."""
    store = store or SweepStore()
    run_store = run_store or RunStore()
    if not store.exists(name):
        raise SweepStoreError("sweep_not_found",
                              f"no existe el recorrido '{name}'", "nada que borrar")
    state = store.state(name).get("status")
    if state in ("running", "queued"):
        raise SweepStoreError(
            "sweep_is_running", f"el recorrido '{name}' esta {state}",
            "paralo antes de borrarlo")
    children = run_store.used_by_sweep(name)
    live = [c for c in children
            if run_store.status(c).get("status") in ("running", "queued")]
    if live:
        raise SweepStoreError(
            "sweep_is_running",
            f"runs del recorrido aun en marcha: {', '.join(live)}",
            "paralos antes de borrar el recorrido")
    for c in children:            # cascade: hijos primero, luego el padre
        run_store.delete(c)
    store.delete(name)
    return {"deleted": name, "runs_deleted": children}


def sweep_trials(name: str, store: SweepStore | None = None,
                 run_store: RunStore | None = None,
                 objective: str | None = None) -> dict:
    """The points table ordered by the objective.

    `objective` overrides the sweep's own for READING ONLY — it re-ranks the same
    finished runs by another val metric (metrica-de-tarea.md §9.7: is `f1` really
    the proxy that tracks the task best, or would `pos_err_px` do better?). It
    does NOT touch the spec: the sweep still trained and checkpointed under its
    own objective, and the answer travels as `objective_overridden` so no reader
    mistakes a re-reading for what the sweep actually optimised.

    A point is scored by the objective AT THE EPOCH THAT `best.pt` KEPT, not at
    the last one. `best.pt` is what survives the run — what diagnostics load,
    what inference uses, what the winner carries forward — so the number that
    picks a winner must describe THAT file. Ranking by the last epoch scored
    weights nobody keeps: measured on the user's own sweeps, `best.pt` was not
    the last epoch in 63% of the runs of `fast-lr-2-s0-lr`, and ranking by the
    last epoch elected a different lr than ranking by the checkpoint.

    `value_last` keeps the old number in view (auditing a ranking means seeing
    both), and `epoch`/`epochs` say which epoch the value came from and how many
    ran, so a point still training is never mistaken for a finished one.
    """
    store = store or SweepStore()
    run_store = run_store or RunStore()
    spec = store.spec(name)
    own_objective = spec.get("objective", "f1")
    if objective is not None and objective not in OBJECTIVES:
        raise SweepError("unknown_objective",
                         f"objetivo '{objective}' no existe",
                         f"usa uno de {sorted(OBJECTIVES)}")
    objective = objective or own_objective
    direction = OBJECTIVES.get(objective, "max")
    base_monitor = (spec.get("base_recipe_value") or {}).get("monitor", "val_loss")
    rows = []
    for i, overrides in enumerate(spec.get("points", [])):
        run_name = point_run_name(name, i, overrides)
        row = {"trial": i, "run": run_name, "point": overrides,
               "status": None, "value": None, "value_last": None,
               "epoch": None, "epochs": None, "monitor": None,
               "seconds_per_epoch": None}
        if run_store.exists(run_name):
            row["status"] = run_store.reconcile(run_name).get("status")
            sp = run_store.path(run_name) / "summary.json"
            if sp.exists():
                summary = read_json_retrying(sp)
                row["seconds_per_epoch"] = summary.get("seconds_per_epoch")
            # the run's OWN monitor: `monitor` is a recipe field, so a sweep may
            # move it point to point — asking the base recipe would be wrong there
            monitor = base_monitor
            try:
                monitor = run_store.config(run_name).get("recipe", {}).get(
                    "monitor", base_monitor)
            except RunError:
                pass
            row["monitor"] = monitor
            m = run_store.metrics_since(run_name, 0)["records"]
            if m:
                row["epochs"] = len(m)
                row["value_last"] = (m[-1].get("val") or {}).get(objective)
                rec = checkpoint_record(m, monitor)
                if rec is not None:
                    row["value"] = (rec.get("val") or {}).get(objective)
                    row["epoch"] = rec.get("epoch")
                else:
                    # the monitor never measured -> the loop never wrote best.pt.
                    # No checkpoint, no value: the reason travels (formatos §2)
                    row["value_reason"] = {
                        "code": "no_checkpoint",
                        "message": f"'{monitor}' no midio en ninguna epoca, asi que "
                                   f"este run no tiene best.pt que rankear",
                        "hint": "usa un monitor que este definido en val "
                                "(val_loss / val_f1) y reentrena el punto"}
        rows.append(row)
    scored = [r for r in rows if r["value"] is not None]
    scored.sort(key=lambda r: r["value"], reverse=(direction == "max"))
    pending = [r for r in rows if r["value"] is None]
    monitors = sorted({r["monitor"] for r in rows if r["monitor"]})
    return {"objective": objective, "direction": direction,
            # a re-reading is never mistaken for what the sweep optimised
            "objective_overridden": objective != own_objective,
            "sweep_objective": own_objective,
            # declared, so a reader never has to guess which epoch it is looking at
            "value_from": "checkpoint",
            "monitors": monitors,
            # best.pt is chosen by `monitor` and the ranking reports `objective`:
            # when they disagree the ranking measures a checkpoint selected by a
            # DIFFERENT criterion. Legal, but the reader has to know.
            "monitor_matches_objective": monitors == [f"val_{objective}"],
            "trials": scored + pending,
            "discarded": spec.get("discarded", [])}
