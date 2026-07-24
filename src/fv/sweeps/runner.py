"""The sweep runner: sequential points (worker limit 1 on CPU), each point a
first-class run named {sweep}-{i:04d} with provenance.sweep set, validated by
THE SAME gate as every other way of training. Resume = count what finished
(only done/cancelled) and redo the rest. Cooperative stop cuts twice: between
points, and — via should_stop — the point IN FLIGHT at its next epoch boundary.
The owner PID is recorded so a crashed 'running' can be reconciled (store).
"""

from __future__ import annotations

import dataclasses
import os

from fv.ioutils import read_json_retrying
from fv.sweeps.spec import OBJECTIVES, SweepError, check_sweep, expand_points
from fv.sweeps.store import SweepStore, SweepStoreError
from fv.training.loop import train
from fv.training.recipe import Recipe
from fv.training.registry import RunError, RunStore


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
        raise SweepError("no_valid_points",
                         "todos los puntos del espacio son geometricamente invalidos",
                         "revisa los rangos: los asserts de la geometria matan esas "
                         "combinaciones (los descartes llevan su razon)")
    enriched = dict(spec)
    enriched["name"] = name
    enriched["points"] = [p["overrides"] for p in valid]
    enriched["discarded"] = discarded
    store.create(name, enriched)
    return enriched


def run_sweep(name: str, store: SweepStore | None = None,
              run_store: RunStore | None = None, progress=None) -> dict:
    """Start OR resume: counts finished child runs and runs the rest."""
    store = store or SweepStore()
    run_store = run_store or RunStore()
    spec = store.spec(name)
    base_network = spec["base_network_value"]
    base_recipe = Recipe(**spec["base_recipe_value"])
    epochs = int((spec.get("budget") or {}).get("epochs", 0) or 0)
    objective = spec.get("objective", "f1")

    valid, _ = expand_points(spec, base_network)
    pid = os.getpid()
    store.set_state(name, "running", total=len(valid), pid=pid)
    done = 0
    for i, point in enumerate(valid):
        run_name = f"{name}-{i:04d}"
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
            store.set_state(name, "stopped", done=done, total=len(valid))
            return store.state(name)
        recipe = dataclasses.replace(base_recipe, **point["recipe_overrides"])
        if epochs:
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
            store.set_state(name, "running", done=done, total=len(valid), pid=pid,
                            last_error={"run": run_name, "code": e.code,
                                        "message": e.message})
            continue
        done += 1
        store.set_state(name, "running", done=done, total=len(valid), pid=pid)
        if progress:
            progress(done, len(valid), run_name)
    store.set_state(name, "done", done=done, total=len(valid))
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
                 run_store: RunStore | None = None) -> dict:
    """The points table ordered by the objective (read from run summaries +
    last metrics line)."""
    store = store or SweepStore()
    run_store = run_store or RunStore()
    spec = store.spec(name)
    objective = spec.get("objective", "f1")
    direction = OBJECTIVES.get(objective, "max")
    rows = []
    for i, overrides in enumerate(spec.get("points", [])):
        run_name = f"{name}-{i:04d}"
        row = {"trial": i, "run": run_name, "point": overrides,
               "status": None, "value": None, "seconds_per_epoch": None}
        if run_store.exists(run_name):
            row["status"] = run_store.reconcile(run_name).get("status")
            sp = run_store.path(run_name) / "summary.json"
            if sp.exists():
                summary = read_json_retrying(sp)
                row["seconds_per_epoch"] = summary.get("seconds_per_epoch")
            m = run_store.metrics_since(run_name, 0)["records"]
            if m:
                last_val = m[-1].get("val", {})
                row["value"] = last_val.get(objective)
        rows.append(row)
    scored = [r for r in rows if r["value"] is not None]
    scored.sort(key=lambda r: r["value"], reverse=(direction == "max"))
    pending = [r for r in rows if r["value"] is None]
    return {"objective": objective, "direction": direction,
            "trials": scored + pending,
            "discarded": spec.get("discarded", [])}
