"""Piece 5 — the OAT study (I): plan validation, guided steps, channels[i]
expansion, carry-forward (barrido-por-ejes.md §6, §7; contract ⑫)."""

import pytest

from fv.studies.driver import (StudyError, advance, confirm, create_study,
                               delete_study, status, validate_plan)
from fv.studies.store import StudyStore


def _recipe(world):
    from fv.training.recipe import RecipeStore
    RecipeStore().save("corta", {"epochs": 1, "batch_size": 32, "lr": 1e-3},
                       overwrite=True)


def _plan(world, axes):
    return {"window_dataset": world["dataset"], "base_recipe": "corta",
            "objective": "f1", "seeds": 3, "budget": {"epochs": 1},
            "axes": axes}


def test_list_and_detail_agree_on_what_the_study_awaits(world):
    """La lista y el detalle describen lo mismo con los MISMOS campos.

    La lista no traía `next_axis`, así que la pantalla rellenó su columna
    «siguiente» con lo que tenía a mano (el dataset): una cabecera y una celda
    diciendo cosas distintas. Ahora ambas salen de `summarize`, y este test
    afirma la COSTURA, no la función."""
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    _recipe(world)
    client = TestClient(create_app())
    body = {"name": "seam", **_plan(world, [{"axis": "n_layers", "range": [1, 2]}])}
    assert client.post("/studies", json=body).status_code == 201
    KEYS = ("next_axis", "awaiting_confirmation", "done")
    row = next(s for s in client.get("/studies").json()["studies"]
               if s["name"] == "seam")
    detail = client.get("/studies/seam").json()
    assert all(k in row for k in KEYS), "la lista no trae lo que la pantalla pinta"
    assert {k: row[k] for k in KEYS} == {k: detail[k] for k in KEYS}
    assert row["next_axis"] == "n_layers"       # el EJE, no el dataset
    assert row["next_axis"] != row["plan"]["window_dataset"]


def test_validate_plan_rejects_unknown_axis_and_bad_auto(world):
    problems = validate_plan(_plan(world, [{"axis": "not_a_field", "range": [1]}]))
    assert any(p["code"] == "unknown_axis" for p in problems)
    problems = validate_plan(_plan(world, [{"axis": "lr", "range": "auto"}]))
    assert any(p["code"] == "auto_needs_geometry" for p in problems)
    assert validate_plan(_plan(world, [{"axis": "n_layers", "range": [1, 2]}])) == []


def test_validate_plan_refuses_N_and_c_frac_axes(world):
    # N/c_frac set center_out, which ①a ties to the fixed window_size — they are
    # derived, not swept; the plan gate must refuse them (root cause of test2).
    for axis in ("N", "c_frac"):
        problems = validate_plan(_plan(world, [{"axis": axis, "range": [8, 10, 12]}]))
        assert any(p["code"] == "axis_breaks_window_size" for p in problems)


def test_study_never_overwritten(world):
    _recipe(world)
    store = StudyStore()
    create_study("est-dup", _plan(world, [{"axis": "d", "range": [1, 2]}]), store)
    with pytest.raises(Exception):
        create_study("est-dup", _plan(world, [{"axis": "d", "range": [1, 2]}]), store)


def test_advance_generates_step_and_refuses_until_confirmed(world):
    _recipe(world)
    store, sstore = StudyStore(), None
    create_study("est1", _plan(world, [{"axis": "d", "range": [1, 2]},
                                       {"axis": "k_center", "range": "auto"}]), store)
    out = advance("est1", store)
    assert out["step"]["axis"] == "d"
    assert out["step"]["sweep"] == "est1-s0-d"
    assert out["step"]["base_label"].startswith("ws8-")
    # guides, not executes: cannot advance while the winner is unconfirmed
    with pytest.raises(StudyError) as e:
        advance("est1", store)
    assert e.value.code == "step_awaiting_confirmation"


def test_confirm_carries_winner_into_next_base(world):
    _recipe(world)
    store = StudyStore()
    create_study("est2", _plan(world, [{"axis": "d", "range": [1, 2]},
                                       {"axis": "k_center", "range": "auto"}]), store)
    advance("est2", store)
    confirm("est2", {"d": 2}, store)
    st = status("est2", store)
    assert st["winners"]["d"] == {"value": 2, "from": "est2/step-0"}
    assert st["next_axis"] == "k_center"
    # the next step's base carries d=2 (origin winner)
    out = advance("est2", store)
    fo = out["spec"]["derivation"]["field_origin"]["d"]
    assert fo["origin"] == "winner" and fo["from"] == "est2/step-0"


def test_n_layers_winner_expands_channels_placeholder(world):
    """§6.1: channels[i] expands to one sub-axis per layer once n_layers wins;
    each sub-step sweeps `channels` with that index varied."""
    _recipe(world)
    store, sstore = StudyStore(), None
    create_study("est3", _plan(world, [
        {"axis": "n_layers", "range": [1, 2, 3]},
        {"axis": "channels[i]", "range": [8, 16], "depends_on": "n_layers"}]), store)
    advance("est3", store)                 # step 0: n_layers
    confirm("est3", {"n_layers": 3}, store)  # winner: 3 layers
    out = advance("est3", store)           # step 1: expands to channels[0]
    assert out["step"]["axis"] == "channels[0]"
    assert out["step"]["space_field"] == "channels"
    from fv.sweeps.store import SweepStore
    space = SweepStore().spec(out["step"]["sweep"])["space"]
    # index 0 varied over [8,16], the other two layers held at the default 16
    assert space["channels"] == [[8, 16, 16], [16, 16, 16]]
    confirm("est3", {"channels": [8, 16, 16]}, store)
    out2 = advance("est3", store)          # step 2: channels[1], carrying [8,16,16]
    assert out2["step"]["axis"] == "channels[1]"
    space2 = SweepStore().spec(out2["step"]["sweep"])["space"]
    assert space2["channels"] == [[8, 8, 16], [8, 16, 16]]


def test_study_seeds_are_wired_into_every_axis_point(world):
    """The plan's `seeds` (D-M1) reaches the generated sweep: seeds=3 over a
    2-value axis yields 2*3 points with a seed axis — root cause of the reported
    bug (seeds validated + stored but never turned into runs)."""
    _recipe(world)
    store = StudyStore()
    from fv.sweeps.store import SweepStore
    create_study("est-seeds", {**_plan(world, [{"axis": "d", "range": [1, 2]}]),
                               "seeds": 3}, store)
    out = advance("est-seeds", store)
    assert out["step"]["seeds"] == 3
    assert out["step"]["points"] == 2 * 3
    space = SweepStore().spec(out["step"]["sweep"])["space"]
    assert space["d"] == [1, 2] and space["seed"] == [1, 2, 3]


def test_delete_study_cascades_its_sweeps_so_same_name_can_be_recreated(world):
    """Deleting a study removes the sweeps it generated (and their runs): leaving
    them orphaned makes recreating the study with the same name collide on the
    next advance (sweep_exists) — the reported bug. After a cascading delete, the
    same name is free to advance again."""
    _recipe(world)
    from fv.sweeps.runner import run_sweep
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, sstore, rstore = StudyStore(), SweepStore(), RunStore()
    create_study("est-recycle", _plan(world, [{"axis": "d", "range": [1, 2]}]), store)
    out = advance("est-recycle", store, sstore)
    sweep = out["step"]["sweep"]
    assert sstore.exists(sweep)                       # the study generated it
    run_sweep(sweep, sstore, rstore)                  # -> terminal (done), deletable
    res = delete_study("est-recycle", store, sstore, rstore)
    assert res["sweeps_deleted"] == [sweep]
    assert not sstore.exists(sweep)                   # cascaded away, not orphaned
    assert not store.exists("est-recycle")
    # the name is reusable: recreate + advance regenerates the SAME sweep name
    # with no collision (before the fix this raised sweep_exists)
    create_study("est-recycle", _plan(world, [{"axis": "d", "range": [1, 2]}]), store)
    out2 = advance("est-recycle", store, sstore)
    assert out2["step"]["sweep"] == sweep and sstore.exists(sweep)


def test_delete_study_refuses_while_a_generated_sweep_is_live(world):
    """R4: a study whose sweep is queued/running is not deleted out from under
    it — refuse BEFORE removing anything, naming the live sweep. (advance
    generates the sweep as 'queued'; the study owns it but must not orphan a
    scheduled run.)"""
    _recipe(world)
    from fv.sweeps.store import SweepStore
    store, sstore = StudyStore(), SweepStore()
    create_study("est-live", _plan(world, [{"axis": "d", "range": [1, 2]}]), store)
    sweep = advance("est-live", store, sstore)["step"]["sweep"]   # -> queued
    with pytest.raises(StudyError) as e:
        delete_study("est-live", store, sstore)
    assert e.value.code == "study_has_live_sweeps" and sweep in e.value.message
    assert store.exists("est-live") and sstore.exists(sweep)      # nothing removed


def test_missing_progress_is_reconstructed_from_plan(world):
    """progress.json is regenerable live state (gitignored); plan.json is
    committed. A fresh clone has plan.json and no progress.json — reading or
    listing the study must self-heal (step-0 progress from the plan), never
    500. Root cause of the /studies HTTP 500 (studies/test5)."""
    _recipe(world)
    store = StudyStore()
    create_study("est-fresh", _plan(world, [{"axis": "n_layers", "range": [1, 2, 3]}]), store)
    # simulate the fresh-clone / cleaned-tree state: only the committed plan.json
    (store.path("est-fresh") / "progress.json").unlink()
    # list() over all studies must not raise on the one missing progress.json
    names = [s["name"] for s in store.list()]
    assert "est-fresh" in names
    # progress is reconstructed: queue from the axis, no steps, no winners
    st = status("est-fresh", store)
    assert st["progress"]["steps"] == [] and st["progress"]["winners"] == {}
    assert st["progress"]["queue"][0]["axis"] == "n_layers"
    # and it was persisted (self-healed on disk)
    assert (store.path("est-fresh") / "progress.json").exists()


def test_full_chain_runs_and_suggests(world):
    """Integration: generate a step, run it, suggest+confirm the winner, advance."""
    _recipe(world)
    store = StudyStore()
    from fv.sweeps.runner import run_sweep
    from fv.sweeps.store import SweepStore
    from fv.sweeps.winner import suggest_winner
    from fv.training.registry import RunStore
    sstore, rstore = SweepStore(), RunStore()
    create_study("est4", _plan(world, [{"axis": "n_layers", "range": [1, 2]}]), store)
    out = advance("est4", store)
    run_sweep(out["step"]["sweep"], sstore, rstore)
    sug = suggest_winner(out["step"]["sweep"], delta=1.0, store=sstore, run_store=rstore)
    confirm("est4", sug["suggested"]["point"], store)
    assert status("est4", store)["done"] is True
