"""I/H — read the winner of a finished sweep and carry it forward (§7).

"Arrastrar el ganador" = read the winning point of a finished sweep's ranking,
take the value of the swept axis there, and fix it in the derived base of the
next axis (a clona-y-varía: base = previous winner, space = next axis). No typing.

The winner is NOT "best objective, full stop". The user's criterion (D-W1): a
3-layer net that costs more but does not beat the 2-layer net SIGNIFICANTLY must
LOSE. So the rule is cost-adjusted with a margin δ: SUGGEST the cheapest point
whose quality is not worse than the best by more than δ (1-SE / Pareto style),
with δ and the cost metric (time vs params) IN VIEW; the user confirms before
the winner is carried. This module suggests; it never decides.
"""

from __future__ import annotations

from fv.models.builder import network_trace
from fv.sweeps.runner import sweep_trials
from fv.sweeps.spec import NETWORK_PARAMS, SweepError
from fv.sweeps.store import SweepStore
from fv.training.registry import RunStore

COST_METRICS = ("seconds_per_epoch", "num_params")


def _num_params_of(run_name: str, run_store: RunStore) -> int | None:
    if not run_store.exists(run_name):
        return None
    net = run_store.config(run_name).get("network")
    return network_trace(net)["num_params"] if net else None


def suggest_winner(name: str, delta: float = 0.0,
                   cost_metric: str = "seconds_per_epoch",
                   store: SweepStore | None = None,
                   run_store: RunStore | None = None) -> dict:
    """Suggest the winner of a finished sweep by the cost/quality rule (D-W1).

    Returns {objective, direction, delta, cost_metric, best, suggested,
    frontier, trials}: `best` is the provisional best objective, `suggested` is
    the cheapest within δ of it, `frontier` are the candidates within δ (the
    ones a confirmation must re-run with N seeds, §11.1). Nothing is decided —
    the caller confirms.
    """
    store = store or SweepStore()
    run_store = run_store or RunStore()
    if cost_metric not in COST_METRICS:
        raise SweepError("unknown_cost_metric",
                         f"métrica de coste '{cost_metric}' no existe",
                         f"usa una de {list(COST_METRICS)}")
    trials = sweep_trials(name, store, run_store)
    scored = [dict(t) for t in trials["trials"] if t["value"] is not None]
    if not scored:
        raise SweepError("no_scored_trials",
                         f"el recorrido '{name}' no tiene puntos con valor aún",
                         "espera a que terminen runs o revisa por qué no miden")
    direction = trials["direction"]
    if cost_metric == "num_params":
        for t in scored:
            t["num_params"] = _num_params_of(t["run"], run_store)
    best, suggested, frontier = select_winner(scored, direction, delta, cost_metric)
    return {
        "objective": trials["objective"], "direction": direction,
        "delta": delta, "cost_metric": cost_metric,
        "best": best, "suggested": suggested,
        "frontier": frontier, "trials": scored,
    }


def select_winner(scored: list[dict], direction: str, delta: float,
                  cost_metric: str) -> tuple[dict, dict, list[dict]]:
    """The pure cost/quality rule (D-W1): given trials sorted best-first, return
    (best, suggested, frontier). `best` is the top objective; `frontier` are the
    points within δ of it; `suggested` is the cheapest of the frontier (ties
    broken toward the better objective)."""
    best = scored[0]

    def within(t) -> bool:
        if direction == "max":
            return t["value"] >= best["value"] - delta
        return t["value"] <= best["value"] + delta

    def cost_of(t):
        c = t.get(cost_metric)
        return c if c is not None else float("inf")

    frontier = [t for t in scored if within(t)]
    suggested = min(frontier, key=lambda t: (cost_of(t),
                                             -t["value"] if direction == "max" else t["value"]))
    return best, suggested, frontier


def winner_overrides(point: dict, from_label: str) -> dict:
    """Turn a winning point's network overrides into carried winners for
    derive_base: {field: {"value": v, "from": "<sweep/step>"}} (§7.2)."""
    return {k: {"value": v, "from": from_label}
            for k, v in point.items() if k in NETWORK_PARAMS}
