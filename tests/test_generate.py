"""Piece 3 — inline base + the P1 generator (barrido-por-ejes.md §8, §10)."""

import pytest

RECIPE = {"epochs": 1, "batch_size": 32, "lr": 1e-3}


def test_generated_inline_sweep_prepares_runs_and_ranks(world):
    """Base inline (no name) prepares, runs and ranks exactly like a named-base
    sweep — same gate, same machinery (§8.2)."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import run_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    enriched = generate_sweep(
        "gen1", world["dataset"], "border_px", [2, 4], base_recipe="corta",
        base_recipe_value=RECIPE, budget={"points": 2, "epochs": 1}, sstore=store)
    # D-H2: inline base -> no name, a synthetic label, and a derivation block
    assert enriched["base_network"] is None
    assert enriched["base_label"].startswith("ws8-")
    assert enriched["derivation"]["window_size"] == 8
    assert len(enriched["points"]) == 2

    state = run_sweep("gen1", store, rstore)
    assert state["status"] == "done" and state["done"] == 2
    trials = sweep_trials("gen1", store, rstore)
    assert all(t["value"] is not None for t in trials["trials"])
    # each point is a first-class run: null network name (inline), sweep set
    from fv.sweeps.runner import point_run_name
    cfg = rstore.config(point_run_name("gen1", 0, enriched["points"][0]))
    assert cfg["provenance"]["network"]["name"] is None
    assert (cfg["provenance"]["network"]["value"]["fovea_px"]
        == enriched["base_network_value"]["fovea_px"])
    assert cfg["provenance"]["sweep"] == "gen1"


def test_generator_uses_the_same_gate_before_reserving(world):
    """§10.1: a base that would not train is refused with its reason and NOTHING
    is written (no sweep dir)."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.spec import SweepError
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    # merge:sum with unequal strides is a base check_network rejects and the
    # derivator does not silently fix (it only corrects d/kernels)
    with pytest.raises(SweepError) as e:
        generate_sweep("gen-bad", world["dataset"], "border_px", [2, 4],
                       base_recipe_value=RECIPE,
                       overrides={"merge": "sum", "s_center": 2, "s_periph": 1},
                       sstore=store)
    assert e.value.code == "merge_sum_needs_equal_strides"
    assert not store.exists("gen-bad")


def test_generator_discards_invalid_points_with_reason(world):
    """§10.3: geometrically invalid expanded points go to `discarded` with the
    reason — the base itself is valid."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    enriched = generate_sweep(
        "gen2", world["dataset"], "overlap_fovea_px", [1, 4],
        base_recipe_value=RECIPE, sstore=store)
    assert len(enriched["discarded"]) == 1
    assert enriched["discarded"][0]["problems"][0]["code"] == "penetration_too_large"


def test_generator_respects_contract_9(world):
    """§8.3: the generator does not elude contract ⑨ — objective=loss with a loss
    weight in the space is rejected by the same check_sweep."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.spec import SweepError
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    with pytest.raises(SweepError) as e:
        generate_sweep("gen3", world["dataset"], "lambda_pos", [0.1, 1.0],
                       base_recipe_value=RECIPE, objective="loss", sstore=store)
    assert e.value.code == "objective_varies_with_space"


def test_sweeping_n_layers_resizes_channels(world):
    """§6.1: n_layers as an axis resizes channels to [16]*L, so every depth is a
    VALID point (not discarded for a stale channel length)."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    enriched = generate_sweep(
        "gen-nl", world["dataset"], "n_layers", [1, 2, 3],
        base_recipe_value=RECIPE, sstore=store)
    assert len(enriched["points"]) == 3 and len(enriched["discarded"]) == 0
    from fv.sweeps.spec import expand_points
    valid, _ = expand_points(enriched, enriched["base_network_value"])
    by_depth = {v["overrides"]["n_layers"]: v["network"]["channels"] for v in valid}
    assert by_depth == {1: [16], 2: [16, 16], 3: [16, 16, 16]}


def test_generated_spec_is_deterministic(world):
    """The generated base + points are a pure function of the inputs (the basis
    of CLI<->API parity)."""
    from fv.sweeps.generate import build_generated_spec
    a, _ = build_generated_spec(world["dataset"], "border_px", [2, 4], base_recipe_value=RECIPE)
    b, _ = build_generated_spec(world["dataset"], "border_px", [2, 4], base_recipe_value=RECIPE)
    assert a == b


def test_seeds_add_a_replica_axis_one_run_per_value_and_seed(world):
    """§11.1/D-M1: seeds>1 trains EVERY axis value on N seeds — the generator
    adds a second `seed` axis, so points = len(range) * seeds, one run per
    (value, seed), each carrying its own recipe seed. seed<=1 is unchanged."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import point_run_name
    from fv.sweeps.spec import expand_points
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    enriched = generate_sweep(
        "gen-seeds", world["dataset"], "border_px", [2, 4], base_recipe_value=RECIPE,
        seeds=3, sstore=store)
    assert enriched["space"]["seed"] == [1, 2, 3]      # base recipe seed = 1 -> [1,2,3]
    assert len(enriched["points"]) == 2 * 3            # each value x each seed
    valid, _ = expand_points(enriched, enriched["base_network_value"])
    # every point pairs an axis value with a distinct seed, and the seed flows
    # into the recipe overrides (so training actually uses it)
    pairs = {(v["overrides"]["border_px"], v["recipe_overrides"]["seed"]) for v in valid}
    assert pairs == {(2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)}
    # and the run name carries the seed so N replicas are legible, not opaque
    name0 = point_run_name("gen-seeds", 0, valid[0]["overrides"])
    assert name0.endswith("_seed1") and "d1" in name0


def test_seed_one_keeps_single_seed_probe_unchanged(world):
    """seeds=1 (the fast probe) adds NO seed axis: one run per value, as before."""
    from fv.sweeps.generate import build_generated_spec
    spec, _ = build_generated_spec(world["dataset"], "border_px", [2, 4],
                                   base_recipe_value=RECIPE, seeds=1)
    assert "seed" not in spec["space"]
