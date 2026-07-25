"""Piece 4 — winner carry-forward and the cost/quality rule (§7, D-W1)."""

from fv.sweeps.winner import aggregate_seeds, select_winner, winner_overrides


def _t(point, value, cost):
    return {"run": f"r-{point}", "point": {"n_layers": point},
            "value": value, "seconds_per_epoch": cost}


def test_cost_quality_rule_prefers_cheaper_within_delta():
    """D-W1: a costlier point that does not beat the best by more than δ loses.
    3 layers (best f1) but 2 layers is within δ and cheaper -> 2 wins."""
    scored = [_t(3, 0.90, 30.0), _t(2, 0.88, 12.0), _t(1, 0.60, 5.0)]
    best, suggested, frontier = select_winner(scored, "max", delta=0.05,
                                              cost_metric="seconds_per_epoch")
    assert best["point"]["n_layers"] == 3        # best objective
    assert suggested["point"]["n_layers"] == 2   # cheapest within δ
    assert {t["point"]["n_layers"] for t in frontier} == {3, 2}  # 1 is outside δ


def test_zero_delta_only_ties_with_best_are_frontier():
    scored = [_t(3, 0.90, 30.0), _t(2, 0.88, 12.0)]
    best, suggested, frontier = select_winner(scored, "max", delta=0.0,
                                              cost_metric="seconds_per_epoch")
    assert suggested["point"]["n_layers"] == 3   # only the best is within δ=0
    assert len(frontier) == 1


def test_min_direction_respects_delta():
    """For a min objective (pos_err_px), within δ means value <= best + δ."""
    scored = [_t(3, 2.0, 30.0), _t(2, 2.3, 12.0), _t(1, 5.0, 5.0)]
    best, suggested, frontier = select_winner(scored, "min", delta=0.5,
                                              cost_metric="seconds_per_epoch")
    assert best["point"]["n_layers"] == 3
    assert suggested["point"]["n_layers"] == 2   # within 0.5 and cheaper


def _seeded(point_val, seed, value, cost):
    return {"run": f"r-{point_val}-s{seed}", "point": {"n_layers": point_val, "seed": seed},
            "value": value, "seconds_per_epoch": cost}


def test_aggregate_seeds_ranks_by_mean_not_the_lucky_replica():
    """§11.1: seed is the replica axis, not one to optimise. A value whose seeds
    AVERAGE higher must win, even if a single lucky replica of another value
    scored the single best run. Aggregation collapses to one entry per value with
    the mean, the band, and n_seeds."""
    scored = [
        _seeded(2, 1, 0.95, 12.0),   # one lucky replica of n_layers=2
        _seeded(2, 2, 0.70, 12.0),
        _seeded(2, 3, 0.72, 12.0),   # mean(2) = 0.79
        _seeded(3, 1, 0.86, 30.0),
        _seeded(3, 2, 0.88, 30.0),
        _seeded(3, 3, 0.90, 30.0),   # mean(3) = 0.88 -> wins on the mean
    ]
    groups = aggregate_seeds(scored, "max", "seconds_per_epoch")
    assert len(groups) == 2
    top = groups[0]
    assert top["point"] == {"n_layers": 3}          # seed stripped from the carried point
    assert abs(top["value"] - 0.88) < 1e-9          # ranked by the MEAN
    assert top["n_seeds"] == 3 and top["seeds"] == [1, 2, 3]
    assert top["value_min"] == 0.86 and top["value_max"] == 0.90  # the band travels
    # the cost/quality rule then runs on the aggregated groups
    best, suggested, frontier = select_winner(groups, "max", delta=0.0,
                                              cost_metric="seconds_per_epoch")
    assert best["point"] == {"n_layers": 3}


def test_aggregate_seeds_singleton_is_a_noop():
    """One seed per value (the probe) -> each group is a singleton: mean = value,
    point unchanged. Same ranking as no aggregation at all (backward compatible)."""
    scored = [_t(3, 0.90, 30.0), _t(2, 0.88, 12.0)]
    groups = aggregate_seeds(scored, "max", "seconds_per_epoch")
    assert [g["point"]["n_layers"] for g in groups] == [3, 2]
    assert all(g["n_seeds"] == 1 for g in groups)


def test_winner_overrides_only_network_fields_with_from():
    point = {"n_layers": 3, "lr": 0.003}   # lr is a recipe (D) field, not carried here
    carried = winner_overrides(point, "estudio-01/paso-1")
    assert carried == {"n_layers": {"value": 3, "from": "estudio-01/paso-1"}}


def test_suggest_winner_wires_a_real_sweep(world):
    """Integration: a finished generated sweep yields a suggestion whose carried
    winner feeds the next step's derived base."""
    from fv.models.derive import derive_base
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import run_sweep
    from fv.sweeps.store import SweepStore
    from fv.sweeps.winner import suggest_winner, winner_overrides
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    generate_sweep("win1", world["dataset"], "n_layers", [1, 2],
                   base_recipe_value={"epochs": 1, "batch_size": 32, "lr": 1e-3},
                   sstore=store)
    run_sweep("win1", store, rstore)
    sug = suggest_winner("win1", delta=1.0, store=store, run_store=rstore)
    assert sug["suggested"]["point"]["n_layers"] in (1, 2)
    carried = winner_overrides(sug["suggested"]["point"], "win1")
    nxt = derive_base(8, winners=carried)
    fo = nxt["derivation"]["field_origin"]["n_layers"]
    assert fo["origin"] == "winner" and fo["from"] == "win1"
