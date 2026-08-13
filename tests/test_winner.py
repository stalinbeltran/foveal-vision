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


def test_tie_delta_is_the_noise_the_sweep_measured(world):
    """protocolo §1.5: a difference that does not clear the seed band is a TIE.
    δ defaults to the 1-SE of the best point across its replicas — the sweep's
    own measurement, not a constant someone has to remember to set."""
    from fv.sweeps.winner import tie_delta
    scored = [
        _seeded(2, 1, 0.80, 12.0), _seeded(2, 2, 0.60, 12.0), _seeded(2, 3, 0.70, 12.0),
        _seeded(3, 1, 0.69, 30.0), _seeded(3, 2, 0.70, 30.0), _seeded(3, 3, 0.71, 30.0),
    ]
    groups = aggregate_seeds(scored, "max", "seconds_per_epoch")
    assert abs(groups[0]["value"] - 0.70) < 1e-9        # both mean 0.70; ties broken by order
    delta, source = tie_delta(groups)
    # stdev([0.80,0.60,0.70]) = 0.1 -> sem = 0.1/sqrt(3) = 0.0577
    assert abs(delta - 0.1 / (3 ** 0.5)) < 1e-9
    assert "1-SE" in source and "3 semillas" in source


def test_tie_delta_refuses_to_invent_a_band_with_one_seed():
    """One replica has no dispersion. δ=0 AND the reason says why — pretending
    the noise is 0 is what crowned the lucky replica in the first place."""
    from fv.sweeps.winner import tie_delta
    groups = aggregate_seeds([_t(3, 0.90, 30.0), _t(2, 0.88, 12.0)], "max",
                             "seconds_per_epoch")
    delta, source = tie_delta(groups)
    assert delta == 0.0
    assert "una sola semilla" in source and "seeds" in source
    assert groups[0]["value_sem"] is None and groups[0]["value_std"] is None


def test_technical_tie_is_declared_not_hidden(world):
    """The user's real case: 1st beats 2nd by less than the winner's own seed
    spread. The suggestion must SAY it is a tie and name the frontier, instead of
    presenting a coin flip as a result."""
    from fv.sweeps.winner import select_winner, tie_delta, tie_reason
    scored = [
        _seeded(2, 1, 0.62, 12.0), _seeded(2, 2, 0.55, 12.0), _seeded(2, 3, 0.68, 12.0),
        _seeded(3, 1, 0.61, 30.0), _seeded(3, 2, 0.56, 30.0), _seeded(3, 3, 0.67, 30.0),
    ]
    groups = aggregate_seeds(scored, "max", "seconds_per_epoch")
    delta, _ = tie_delta(groups)
    gap = groups[0]["value"] - groups[1]["value"]
    assert gap < delta                                   # the gap is inside the noise
    best, suggested, frontier = select_winner(groups, "max", delta, "seconds_per_epoch")
    assert len(frontier) == 2                            # both survive
    assert suggested["point"]["n_layers"] == 2           # the cheaper of the tied
    reason = tie_reason(frontier, delta)
    assert "EMPATE" in reason and "no los distingue" in reason
    # control: a gap wider than the band is NOT a tie
    _, _, frontier2 = select_winner(groups, "max", gap / 2, "seconds_per_epoch")
    assert len(frontier2) == 1
    assert "despega" in tie_reason(frontier2, gap / 2)


def test_suggest_winner_derives_delta_when_none_is_given(world):
    """The API default: no δ in the query -> δ comes from the seeds, and the
    answer declares where it came from."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import run_sweep
    from fv.sweeps.store import SweepStore
    from fv.sweeps.winner import suggest_winner
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    generate_sweep("tie1", world["dataset"], "lr", [1e-3, 3e-3], seeds=2,
                   base_recipe_value={"epochs": 1, "batch_size": 32, "lr": 1e-3},
                   sstore=store)
    run_sweep("tie1", store, rstore)
    sug = suggest_winner("tie1", store=store, run_store=rstore)   # δ omitted
    assert sug["delta"] >= 0.0 and "1-SE" in sug["delta_source"]
    assert sug["best"]["n_seeds"] == 2                # 2 seeds per lr value
    assert isinstance(sug["tie"], bool) and sug["tie_reason"]
    # an explicit δ still wins, and says so
    fixed = suggest_winner("tie1", delta=0.5, store=store, run_store=rstore)
    assert fixed["delta"] == 0.5 and fixed["delta_source"] == "fijada a mano"


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
