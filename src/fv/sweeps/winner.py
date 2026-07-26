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

import math
import statistics

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


def suggest_winner(name: str, delta: float | None = None,
                   cost_metric: str = "seconds_per_epoch",
                   store: SweepStore | None = None,
                   run_store: RunStore | None = None) -> dict:
    """Suggest the winner of a finished sweep by the cost/quality rule (D-W1).

    Returns {objective, direction, delta, delta_source, tie, tie_reason,
    cost_metric, best, suggested, frontier, trials}: `best` is the provisional
    best objective, `suggested` is the cheapest within δ of it, `frontier` are
    the candidates within δ. Nothing is decided — the caller confirms.

    δ defaults to the noise the sweep MEASURED ITSELF (`tie_delta`): pass a
    number to override it.
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
    # seed is the replica axis, not one to optimise (recipe.py): rank the
    # per-value MEAN across seeds, never a single lucky replica (§11.1, D-M1).
    # With one seed per value each group is a singleton -> identical to before.
    groups = aggregate_seeds(scored, direction, cost_metric)
    if delta is None:
        delta, delta_source = tie_delta(groups)
    else:
        delta, delta_source = float(delta), "fijada a mano"
    best, suggested, frontier = select_winner(groups, direction, delta, cost_metric)
    return {
        "objective": trials["objective"], "direction": direction,
        "delta": delta, "delta_source": delta_source, "cost_metric": cost_metric,
        "best": best, "suggested": suggested,
        "frontier": frontier, "trials": groups,
        "tie": len(frontier) > 1, "tie_reason": tie_reason(frontier, delta),
        # the ranking's own provenance travels with the suggestion
        "value_from": trials.get("value_from"),
        "monitor_matches_objective": trials.get("monitor_matches_objective"),
    }


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def tie_delta(groups: list[dict]) -> tuple[float, str]:
    """δ by default = the noise THIS sweep measured (the 1-SE rule).

    protocolo.md §1.5: «la diferencia supera la banda de ruido; si no, es un
    empate — y no se rompe con "pero es que este subió"». The seeds are run
    precisely to measure that band, and until now `select_winner` took
    `scored[0]` without looking at it: on the user's own `fast-lr-2-s0-lr` the
    gap between 1st and 2nd was 0.0035 while the winner's own seeds spread
    0.0771 — a coin flip crowned with the face of a result.

    So δ = the standard error of the mean of the best point across its seeds.
    Anything within that is, by the project's own protocol, a tie. With a single
    seed there is no measured band: δ=0 and the reason says so — it does NOT
    pretend the noise is zero.
    """
    best = groups[0]
    sem = best.get("value_sem")
    if sem is None:
        return 0.0, ("una sola semilla en el mejor punto: no hay dispersión medida, "
                     "así que no hay banda de ruido que descontar (sube `seeds` para "
                     "poder declarar un empate)")
    return float(sem), (f"1-SE: error estándar de las {best['n_seeds']} semillas del "
                        f"mejor punto")


def tie_reason(frontier: list[dict], delta: float) -> str:
    """What the frontier means, in words — a number the reader has to interpret
    alone is a number that gets over-read."""
    # ASCII-safe on purpose: this string is printed by fv-oat/fv-study, and the
    # Windows console is cp1252 — a Greek delta there is a UnicodeEncodeError
    # that kills an unattended study. The web UI writes the δ in its own labels.
    if len(frontier) <= 1:
        return (f"el mejor punto despega del resto por más de delta={delta:.4f}: "
                f"la diferencia supera la banda de ruido medida")
    pts = ", ".join(str(t["point"]) for t in frontier[:6])
    more = f" (+{len(frontier) - 6})" if len(frontier) > 6 else ""
    return (f"EMPATE TÉCNICO: {len(frontier)} puntos caen dentro de delta={delta:.4f} "
            f"({pts}{more}). Sus diferencias no superan el ruido entre semillas, "
            f"así que este recorrido no los distingue: el sugerido es el más barato "
            f"de la frontera, no «el mejor»")


def aggregate_seeds(scored: list[dict], direction: str,
                    cost_metric: str) -> list[dict]:
    """Collapse per-run trials into one entry per axis value, averaging the
    objective (and the cost) across the seed replicas that share it. Grouping key
    is the point WITHOUT `seed`; the group's `point` drops `seed` too, so the
    carried winner is the axis value, not a replica. Each group keeps `n_seeds`
    and the min/max band (protocolo: a value is a distribution, not a point).
    Returned sorted best-first, the order select_winner expects. Each group also
    carries the dispersion of its replicas (`value_std`, `value_sem`) — that is
    the noise band the tie rule spends (tie_delta); None with one seed, never 0
    (an unmeasured band is not a zero band)."""
    groups: dict = {}
    order: list = []
    for t in scored:
        point = {k: v for k, v in t["point"].items() if k != "seed"}
        key = tuple(sorted((k, _hashable(v)) for k, v in point.items()))
        if key not in groups:
            groups[key] = {"point": point, "values": [], "costs": [],
                           "num_params": t.get("num_params"), "runs": [], "seeds": []}
            order.append(key)
        g = groups[key]
        g["values"].append(t["value"])
        g["runs"].append(t["run"])
        if "seed" in t["point"]:
            g["seeds"].append(t["point"]["seed"])
        c = t.get(cost_metric)
        if c is not None and cost_metric != "num_params":
            g["costs"].append(c)
    out = []
    for key in order:
        g = groups[key]
        n = len(g["values"])
        std = statistics.stdev(g["values"]) if n >= 2 else None
        entry = {"point": g["point"], "value": sum(g["values"]) / n,
                 "n_seeds": n, "seeds": sorted(g["seeds"]),
                 "value_min": min(g["values"]), "value_max": max(g["values"]),
                 "value_std": std,
                 "value_sem": (std / math.sqrt(n)) if std is not None else None,
                 "runs": g["runs"]}
        if cost_metric == "num_params":
            entry["num_params"] = g["num_params"]
        else:
            entry[cost_metric] = (sum(g["costs"]) / len(g["costs"])
                                  if g["costs"] else None)
        out.append(entry)
    out.sort(key=lambda e: e["value"], reverse=(direction == "max"))
    return out


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
