"""H — the sweep spec: a space over C and/or D with B fixed.

Pure validation and expansion (no torch): contract (9) — the objective cannot
be the loss if a loss weight is in the space (each point would be measured
with a different rule and lambda->0 "wins" by definition). Geometry ranges may
be "auto": they come from fv.fovea.build_search_space, never hand-written.
Geometrically invalid points are discarded WITH their reason declared, before
reserving anything.
"""

from __future__ import annotations

import itertools

from fv.fovea import build_search_space
from fv.models.builder import DEFAULT_CHANNEL, NETWORK_DEFAULTS, full_config
from fv.training.recipe import Recipe
from fv.validation import check_network

NETWORK_PARAMS = set(NETWORK_DEFAULTS)
RECIPE_PARAMS = set(Recipe().as_dict())
LOSS_WEIGHT_PARAMS = {"lambda_pos", "pos_weight", "smooth_l1_beta"}
GEOMETRY_AUTO = {"k_center", "k_periph", "s_center", "s_periph", "d"}
OBJECTIVES = {"f1": "max", "pos_err_px": "min", "loss": "min"}


class SweepError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def check_sweep(spec: dict) -> list[dict]:
    problems = []

    def bad(code, message, hint):
        problems.append({"code": code, "message": message, "hint": hint})

    space = spec.get("space", {})
    if not space:
        bad("empty_space", "el espacio esta vacio", "declara al menos un eje")
    for param in space:
        if param not in NETWORK_PARAMS | RECIPE_PARAMS:
            bad("unknown_space_param", f"'{param}' no es un campo de C ni de D",
                f"los ejes validos son {sorted(NETWORK_PARAMS | RECIPE_PARAMS)}")
    objective = spec.get("objective", "f1")
    if objective not in OBJECTIVES:
        bad("unknown_objective", f"objetivo '{objective}' no existe",
            f"usa uno de {sorted(OBJECTIVES)}")
    if objective == "loss" and LOSS_WEIGHT_PARAMS & set(space):
        bad("objective_varies_with_space",
            f"la loss no puede rankear un espacio que barre "
            f"{sorted(LOSS_WEIGHT_PARAMS & set(space))}: cada punto se mediria con "
            f"una perdida distinta y lambda->0 gana por definicion",
            "usa 'f1' o 'pos_err_px' como objetivo")
    if spec.get("strategy", "grid") not in ("grid", "random"):
        bad("unknown_strategy", f"estrategia '{spec.get('strategy')}' no existe",
            "usa grid (geometria: espacio pequeno y discreto) o random")
    for param, values in space.items():
        if values == "auto":
            if param not in GEOMETRY_AUTO:
                bad("auto_needs_geometry",
                    f"'{param}' no tiene rango calculado: 'auto' solo vale para "
                    f"{sorted(GEOMETRY_AUTO)}",
                    "da la lista de valores explicita")
        elif not isinstance(values, list) or not values:
            bad("space_values_must_be_list",
                f"el eje '{param}' debe ser una lista de valores o 'auto'",
                "p. ej. {\"lr\": [0.001, 0.003]} o {\"d\": \"auto\"}")
    return problems


def expand_points(spec: dict, base_network: dict) -> tuple[list[dict], list[dict]]:
    """-> (valid points, discarded points with reasons). A point is
    {network: {...}, recipe_overrides: {...}}."""
    base = full_config(base_network)
    ss = build_search_space(base["N"], base["c_frac"], base["pen_frac"])
    space: dict[str, list] = {}
    for param, values in spec.get("space", {}).items():
        if values == "auto":
            space[param] = ss[param]
        else:
            space[param] = list(values)

    names = sorted(space)
    combos = list(itertools.product(*(space[k] for k in names)))
    if spec.get("strategy", "grid") == "random":
        import random
        rng = random.Random(spec.get("seed", 1))
        rng.shuffle(combos)
    budget = spec.get("budget", {}) or {}
    max_points = int(budget.get("points", 0) or 0)
    if max_points:
        combos = combos[:max_points]

    valid, discarded = [], []
    for combo in combos:
        overrides = dict(zip(names, combo))
        net = dict(base)
        net.update({k: v for k, v in overrides.items() if k in NETWORK_PARAMS})
        # channels depends on n_layers (§6.1): sweeping depth WITHOUT sweeping
        # channels resizes the vector to the default rule [16]*L (§3.2), so the
        # point stays valid instead of carrying the base's stale channel length.
        if "n_layers" in overrides and "channels" not in overrides:
            net["channels"] = [DEFAULT_CHANNEL] * int(overrides["n_layers"])
        recipe_over = {k: v for k, v in overrides.items() if k in RECIPE_PARAMS}
        problems = check_network(net)
        if problems:
            discarded.append({"point": overrides, "problems": problems})
        else:
            valid.append({"overrides": overrides, "network": net,
                          "recipe_overrides": recipe_over})
    return valid, discarded
