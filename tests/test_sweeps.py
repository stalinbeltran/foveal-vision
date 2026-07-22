"""H: expansion with declared discards, sequential run, resume, ranking."""

from tests.conftest import TINY_NET


def _spec(world, points=0, epochs=1):
    return {
        "window_dataset": world["dataset"],
        "base_network": "tiny", "base_network_value": TINY_NET,
        "base_recipe": "quick",
        "base_recipe_value": {"epochs": epochs, "batch_size": 32, "lr": 1e-3},
        "space": {"d": [1, 2], "lr": [0.001, 0.003]},
        "strategy": "grid", "objective": "f1",
        "budget": {"points": points, "epochs": epochs},
    }


def test_expand_discards_invalid_geometry_with_reason(world):
    from fv.sweeps.spec import expand_points
    spec = {"space": {"pen_frac": [0.1, 0.45]}, "strategy": "grid"}
    valid, discarded = expand_points(spec, TINY_NET)
    assert len(valid) == 1 and len(discarded) == 1
    assert discarded[0]["problems"][0]["code"] == "penetration_too_large"


def test_auto_ranges_come_from_fovea(world):
    from fv.fovea import build_search_space
    from fv.sweeps.spec import expand_points
    spec = {"space": {"k_center": "auto"}, "strategy": "grid"}
    valid, _ = expand_points(spec, TINY_NET)
    ss = build_search_space(TINY_NET["N"], TINY_NET["c_frac"], TINY_NET["pen_frac"])
    assert [p["overrides"]["k_center"] for p in valid] == ss["k_center"]


def test_sweep_runs_ranks_and_resumes(world):
    from fv.sweeps.runner import prepare_sweep, run_sweep, sweep_trials
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
    # every point is a first-class run with provenance.sweep set
    cfg = rstore.config("sw1-0000")
    assert cfg["provenance"]["sweep"] == "sw1"
    # resume is idempotent: finished points are counted, not redone
    state2 = run_sweep("sw1", store, rstore)
    assert state2["done"] == 2


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
