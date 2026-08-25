"""H: expansion with declared discards, sequential run, resume, ranking."""

import pytest

from tests.conftest import TINY_NET


def _spec(world, points=0, epochs=1):
    return {
        "window_dataset": world["dataset"],
        "base_network": "tiny", "base_network_value": TINY_NET,
        "base_recipe": "quick",
        "base_recipe_value": {"epochs": epochs, "batch_size": 32, "lr": 1e-3},
        "space": {"border_px": [2, 4], "lr": [0.001, 0.003]},
        "strategy": "grid", "objective": "f1",
        "budget": {"points": points, "epochs": epochs},
    }


def test_expand_discards_invalid_geometry_with_reason(world):
    from fv.sweeps.spec import expand_points
    spec = {"space": {"overlap_fovea_px": [1, 4]}, "strategy": "grid"}
    valid, discarded = expand_points(spec, TINY_NET)   # fovea 8 -> 4 >= 8//2
    assert len(valid) == 1 and len(discarded) == 1
    assert discarded[0]["problems"][0]["code"] == "penetration_too_large"


def test_auto_ranges_come_from_fovea(world):
    from fv.fovea import build_search_space
    from fv.sweeps.spec import expand_points
    spec = {"space": {"k_center": "auto"}, "strategy": "grid"}
    valid, _ = expand_points(spec, TINY_NET)
    ss = build_search_space(TINY_NET)
    assert [p["overrides"]["k_center"] for p in valid] == ss["k_center"]


def test_sweep_runs_ranks_and_resumes(world):
    from fv.sweeps.runner import point_run_name, prepare_sweep, run_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    spec = _spec(world, points=2, epochs=1)
    enriched = prepare_sweep("sw1", spec, TINY_NET, store)
    assert len(enriched["points"]) == 2
    state = run_sweep("sw1", store, rstore)
    assert state["status"] == "done" and state["done"] == 2
    trials = sweep_trials("sw1", store, rstore)
    assert trials["objective"] == "f1"
    assert all(t["value"] is not None for t in trials["trials"])
    # every point is a first-class run with provenance.sweep set — and the run
    # name carries its axis value (point_run_name), not just an opaque index
    r0 = point_run_name("sw1", 0, enriched["points"][0])
    assert r0.startswith("sw1-0000-") and rstore.exists(r0)
    cfg = rstore.config(r0)
    assert cfg["provenance"]["sweep"] == "sw1"
    # resume is idempotent: finished points are counted, not redone
    state2 = run_sweep("sw1", store, rstore)
    assert state2["done"] == 2


def test_trials_can_be_reread_with_another_objective(world):
    """§9.7: re-rank the SAME finished runs by another val metric, without
    touching the spec — and say so, so a re-reading is never mistaken for what
    the sweep actually optimised."""
    from fv.sweeps.runner import prepare_sweep, run_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    prepare_sweep("sw-reread", _spec(world, points=2, epochs=1), TINY_NET, store)
    run_sweep("sw-reread", store, rstore)

    own = sweep_trials("sw-reread", store, rstore)
    assert own["objective"] == "f1" and own["direction"] == "max"
    assert own["objective_overridden"] is False

    other = sweep_trials("sw-reread", store, rstore, objective="loss")
    assert other["objective"] == "loss" and other["direction"] == "min"
    # the override is DECLARED and the sweep's own objective still travels
    assert other["objective_overridden"] is True
    assert other["sweep_objective"] == "f1"
    # same runs, same epoch source — only the number read changes
    assert {t["run"] for t in other["trials"]} == {t["run"] for t in own["trials"]}
    assert other["value_from"] == "checkpoint"
    assert all(t["value"] is not None for t in other["trials"])
    # the spec is untouched: asking again without the override says f1
    assert store.spec("sw-reread")["objective"] == "f1"


def test_rereading_with_an_unknown_objective_is_refused(world):
    """At the door, with the reason — not a silent table of None (R4)."""
    from fv.sweeps.runner import prepare_sweep, run_sweep, sweep_trials
    from fv.sweeps.spec import SweepError
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    prepare_sweep("sw-badobj", _spec(world, points=1, epochs=1), TINY_NET, store)
    run_sweep("sw-badobj", store, rstore)
    with pytest.raises(SweepError) as e:
        sweep_trials("sw-badobj", store, rstore, objective="paragraph_f1")
    assert e.value.code == "unknown_objective"


def test_delete_sweep_cascades_and_leaves_no_orphan(world):
    from fv.sweeps.runner import delete_sweep, point_run_name, prepare_sweep, run_sweep
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    enriched = prepare_sweep("swd", _spec(world, points=1, epochs=1), TINY_NET, store)
    run_sweep("swd", store, rstore)
    child = point_run_name("swd", 0, enriched["points"][0])
    assert rstore.exists(child)
    out = delete_sweep("swd", store, rstore)
    assert out["deleted"] == "swd" and out["runs_deleted"] == [child]
    # both gone: the sweep AND its child — nothing points at a missing parent
    assert not store.exists("swd")
    assert not rstore.exists(child)
    assert rstore.used_by_sweep("swd") == []


def test_delete_running_sweep_is_refused(world):
    from fv.sweeps.runner import delete_sweep
    from fv.sweeps.store import SweepStore, SweepStoreError
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    store.create("live", _spec(world, points=1, epochs=1))
    store.set_state("live", "running", done=0, total=1)
    with pytest.raises(SweepStoreError) as e:
        delete_sweep("live", store, rstore)
    assert e.value.code == "sweep_is_running"
    assert store.exists("live")   # refused, nothing removed


def test_sweep_stop_between_points(world):
    from fv.sweeps.runner import prepare_sweep, run_sweep
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    prepare_sweep("sw2", _spec(world, points=3, epochs=1), TINY_NET, store)
    store.request_stop("sw2")            # stop already requested: cuts at point 0
    state = run_sweep("sw2", store, rstore)
    assert state["status"] == "stopped"
    store.clear_stop("sw2")              # resume clears the request and finishes
    state2 = run_sweep("sw2", store, rstore)
    assert state2["status"] == "done"


def test_should_stop_cuts_the_point_in_flight(world):
    """Feature 1: a stop asked of the sweep cuts the point IN FLIGHT at the next
    epoch boundary, not only between points."""
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    rstore = RunStore()
    recipe = Recipe(epochs=5, batch_size=32, lr=1e-3)
    summary = train("r-inflight", world["dataset"], "tiny", TINY_NET,
                    "quick", recipe, store=rstore,
                    should_stop=lambda: True)  # stop at the first epoch boundary
    assert summary["cancelled"] is True
    assert summary["epochs_run"] == 1          # did not run all 5
    assert rstore.status("r-inflight")["status"] == "cancelled"


def test_reconcile_heals_stale_running_when_owner_is_gone(world):
    """Feature 2: a sweep whose owner process is gone (crash/restart/hibernation)
    is healed from 'running' to 'interrupted' — never 'running' forever."""
    import os
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    store.create("crashed", _spec(world, points=2, epochs=1))

    store.set_state("crashed", "running", done=0, total=2, pid=2_000_000_000)
    healed = store.reconcile("crashed")      # no process owns that PID
    assert healed["status"] == "interrupted" and healed["reason"]

    store.create("live2", _spec(world, points=2, epochs=1))
    store.set_state("live2", "running", done=0, total=2, pid=os.getpid())
    assert store.reconcile("live2")["status"] == "running"   # this process is alive

    store.create("legacy", _spec(world, points=2, epochs=1))
    store.set_state("legacy", "running", done=0, total=2)    # old sweep, no owner
    assert store.reconcile("legacy")["status"] == "running"  # never guesses


def test_resume_redoes_an_interrupted_point(world):
    """Feature 2 + runner: only done/cancelled count as finished; an interrupted
    point (reconciled after a crash) is dropped and redone on resume."""
    from fv.sweeps.runner import point_run_name, prepare_sweep, run_sweep
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    enriched = prepare_sweep("swr", _spec(world, points=2, epochs=1), TINY_NET, store)
    run_sweep("swr", store, rstore)
    child1 = point_run_name("swr", 1, enriched["points"][1])
    rstore.set_status(child1, "interrupted", epoch=0)       # simulate a crash
    run_sweep("swr", store, rstore)                         # resume
    assert rstore.status(child1)["status"] == "done"        # redone, not jammed


def test_no_valid_points_error_carries_each_reason(world):
    # when every axis value is geometrically invalid, the error must say WHICH
    # value failed and WHY — not just "all invalid" (the user's complaint)
    from fv.sweeps.runner import prepare_sweep
    from fv.sweeps.spec import SweepError
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    spec = _spec(world)
    spec["space"] = {"overlap_fovea_px": [4, 5]}  # both eat the fovea's core
    with pytest.raises(SweepError) as e:
        prepare_sweep("bad-axis", spec, TINY_NET, store)
    assert e.value.code == "no_valid_points"
    assert "overlap_fovea_px=4" in e.value.hint and "overlap_fovea_px=5" in e.value.hint
    assert "fovea_px//2" in e.value.hint          # the concrete geometric reason
    assert not store.exists("bad-axis")            # nothing reserved on failure


def test_sweeping_epochs_is_not_collapsed_by_the_budget(world):
    # the budget caps epochs, but a point that SWEEPS epochs must keep its own
    # value — otherwise the axis silently does nothing (every point same epochs)
    from fv.ioutils import read_json_retrying
    from fv.sweeps.runner import point_run_name, prepare_sweep, run_sweep
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    spec = _spec(world)
    spec["space"] = {"epochs": [1, 2]}
    spec["budget"] = {"epochs": 1}          # would have overridden both to 1
    enriched = prepare_sweep("sw-ep", spec, TINY_NET, store)
    run_sweep("sw-ep", store, rstore)
    got = sorted(read_json_retrying(
                     rstore.path(point_run_name("sw-ep", i, enriched["points"][i]))
                     / "summary.json")["epochs_requested"] for i in range(2))
    assert got == [1, 2]                     # the axis varied, not collapsed


def test_sweeping_the_fovea_is_refused_before_reserving(world):
    # ①a ties the fovea to the dataset window_size, so it cannot be an axis:
    # check_sweep must refuse WITH the reason, up front, instead of letting every
    # point fail window_size_mismatch deep in the job (the bug behind test2-s0-N
    # ending 'done 0/3' with no visible cause).
    from fv.sweeps.runner import prepare_sweep
    from fv.sweeps.spec import SweepError, check_sweep
    from fv.sweeps.store import SweepStore
    spec = _spec(world)
    spec["space"] = {"fovea_px": [8, 10, 12]}
    problems = check_sweep(spec)
    assert any(p["code"] == "axis_breaks_window_size" for p in problems)
    store = SweepStore()
    with pytest.raises(SweepError) as e:
        prepare_sweep("bad-fovea", spec, TINY_NET, store)
    assert e.value.code == "axis_breaks_window_size"
    assert not store.exists("bad-fovea")           # nothing reserved on refusal


def test_the_old_geometry_axes_are_refused_by_NAME_not_reinterpreted(world):
    """The reparameterisation renamed the geometry AND changed what `d` means.
    An old spec re-run verbatim must stop with the reason, never train a
    different network in silence — that is the whole point of the rename."""
    from fv.sweeps.runner import prepare_sweep
    from fv.sweeps.spec import SweepError, check_sweep
    from fv.sweeps.store import SweepStore
    for axis, wanted in (("N", "border_px"), ("c_frac", "border_px"),
                         ("pen_frac", "overlap_fovea_px"), ("d", "border_reduce")):
        spec = _spec(world)
        spec["space"] = {axis: [1, 2]}
        problems = check_sweep(spec)
        assert any(p["code"] == "axis_renamed" for p in problems), axis
        assert any(wanted in p["hint"] for p in problems), axis
        store = SweepStore()
        with pytest.raises(SweepError) as e:
            prepare_sweep(f"old-{axis}", spec, TINY_NET, store)
        assert e.value.code == "axis_renamed"
        assert not store.exists(f"old-{axis}")


def test_pid_alive():
    import os
    from fv.proc import pid_alive
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_000_000_000) is False
    assert pid_alive(None) is False and pid_alive(0) is False
