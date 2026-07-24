"""Piece 5 — the OAT study (I): plan validation, guided steps, channels[i]
expansion, carry-forward (barrido-por-ejes.md §6, §7; contract ⑫)."""

import pytest

from fv.studies.driver import (StudyError, advance, confirm, create_study,
                               status, validate_plan)
from fv.studies.store import StudyStore


def _recipe(world):
    from fv.training.recipe import RecipeStore
    RecipeStore().save("corta", {"epochs": 1, "batch_size": 32, "lr": 1e-3},
                       overwrite=True)


def _plan(world, axes):
    return {"window_dataset": world["dataset"], "base_recipe": "corta",
            "objective": "f1", "seeds": 3, "budget": {"epochs": 1},
            "axes": axes}


def test_validate_plan_rejects_unknown_axis_and_bad_auto(world):
    problems = validate_plan(_plan(world, [{"axis": "not_a_field", "range": [1]}]))
    assert any(p["code"] == "unknown_axis" for p in problems)
    problems = validate_plan(_plan(world, [{"axis": "lr", "range": "auto"}]))
    assert any(p["code"] == "auto_needs_geometry" for p in problems)
    assert validate_plan(_plan(world, [{"axis": "n_layers", "range": [1, 2]}])) == []


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
