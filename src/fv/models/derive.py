"""G/C — derive a full network base config (C) from the problem (window_size).

The base a sweep step needs is not hand-written: it is DERIVED. Contract ①a
fixes center_out(C) == window_size(B), so from W the geometry follows; every
other field is a static default (§4, from NETWORK_DEFAULTS) with carried winners
and explicit user tunables applied on top. A default invalid for this W (a `d`
that overflows downsample_range, a kernel that exceeds the band) falls to the
nearest valid value WITH its reason — never silently (barrido-por-ejes.md §5.2).

N is the only value the problem forces; it is derived, never written by hand
(the principle: todo dato es un parámetro). This module imports only fv.fovea
and its own domain (fv.models) — contract ⑦: it never sees fv.validation, whose
extra gate (kernels/merge/channels + measurability) the generator runs via
check_run before reserving anything (§10).
"""

from __future__ import annotations

from fv.fovea import (FoveaError, check_dims, derive_dims, downsample_range,
                      kernel_range, round_to_even)
from fv.models.builder import NETWORK_DEFAULTS, full_config

# The static context of a derived base (§4): everything NETWORK_DEFAULTS fixes
# EXCEPT N (derived from W) and channels (derived from n_layers). pen_frac stays
# fixed (D-G1); c_frac and d are the exposed tunables.
STATIC_FIELDS = ("c_frac", "pen_frac", "d", "n_layers", "k_center", "k_periph",
                 "s_center", "s_periph", "merge", "pool_mode", "pad_mode")
DEFAULT_C_FRAC = NETWORK_DEFAULTS["c_frac"]
C_FRAC_TOLERANCE = 0.15


def derive_geometry(window_size: int, c_frac_target: float = DEFAULT_C_FRAC,
                    c_frac_tol: float = C_FRAC_TOLERANCE) -> tuple[int, float, str | None]:
    """(N, c_frac_effective, reason). The smallest even N (D-G2) whose fovea is
    exactly W with a periphery of >=1. If none exists at c_frac_target, loosen
    c_frac to the value that hits W exactly (W/N), smallest N within tolerance,
    and RETURN the reason (D-G3) — W never moves, it comes from B."""
    W = int(window_size)
    if W < 4 or W % 2 != 0:
        raise FoveaError("window_size_must_be_even",
                         f"window_size={W} debe ser par y >= 4",
                         "reconstruye B con una ventana par: la periferia reparte simétrico")
    n_max = max(W * 4, W + 8)
    exact = [N for N in range(W + 2, n_max + 1, 2)
             if round_to_even(N * c_frac_target) == W and (N - W) // 2 >= 1]
    if exact:
        return min(exact), float(c_frac_target), None
    for N in range(W + 2, n_max + 1, 2):
        if (N - W) // 2 < 1:
            continue
        cf = W / N  # center_out = round_to_even(N * W/N) = W exactly (W even)
        if abs(cf - c_frac_target) <= c_frac_tol:
            reason = (f"ningún N par con c_frac={c_frac_target} da fóvea {W}px; "
                      f"se afloja c_frac a {cf:.4f} (N={N}) para cumplir ①a")
            return N, float(cf), reason
    raise FoveaError(
        "no_feasible_n",
        f"ningún N par da fóvea {W}px, ni aflojando c_frac ±{c_frac_tol}",
        "revisa el window_size de B o el c_frac objetivo")


def base_label(dims, n_layers: int) -> str:
    """The synthetic grouping key (D-H2), guion separator: ws16-p2-d2-L2."""
    return f"ws{dims.center_out}-p{dims.periph_out}-d{dims.d}-L{int(n_layers)}"


def _correct(cfg: dict, field: str, valid: list[int], corrections: list[dict]) -> None:
    """Fall an invalid default/winner to the nearest valid value (largest <= v,
    else smallest), recording the reason. Never silent (§5.2 step 4)."""
    if not valid:
        return
    v = cfg[field]
    if v in valid:
        return
    below = [x for x in valid if x <= v]
    new = max(below) if below else min(valid)
    corrections.append({
        "field": field, "from": v, "to": new,
        "reason": f"{field}={v} inválido para esta geometría; cae a {new} "
                  f"(rango válido {valid})"})
    cfg[field] = new


def derive_base(window_size: int, winners: dict | None = None,
                overrides: dict | None = None, c_frac: float | None = None) -> dict:
    """Derive a full base config from the problem.

    winners:   {field: {"value": v, "from": "<study/step>"}} — carried winners (§7).
    overrides: {field: v} — explicit user tunables (c_frac/d/...), U5.
    c_frac:    the target central fraction (else the user override, else default).

    Returns {config, dims, base_label, c_frac_effective, c_frac_reason,
    corrections, derivation{window_size, fractions, field_origin}}.
    """
    winners = dict(winners or {})
    overrides = dict(overrides or {})
    c_frac_target = (c_frac if c_frac is not None
                     else overrides.get("c_frac", DEFAULT_C_FRAC))
    N, c_frac_eff, cfrac_reason = derive_geometry(window_size, c_frac_target)

    cfg = {f: NETWORK_DEFAULTS[f] for f in STATIC_FIELDS}
    cfg["c_frac"] = c_frac_eff
    cfg["N"] = N
    origin: dict[str, dict] = {f: {"origin": "default"} for f in cfg}
    for f, w in winners.items():
        cfg[f] = w["value"] if isinstance(w, dict) else w
        origin[f] = {"origin": "winner",
                     "from": w.get("from") if isinstance(w, dict) else None}
    for f, v in overrides.items():
        cfg[f] = v
        origin[f] = {"origin": "user"}

    corrections: list[dict] = []
    dims = derive_dims(N, cfg["c_frac"], cfg["d"], cfg["pen_frac"])
    _correct(cfg, "d", downsample_range(dims.periph_out, N, max_original=2 * N), corrections)
    _correct(cfg, "k_center", kernel_range(dims.center_out), corrections)
    _correct(cfg, "k_periph", kernel_range(dims.periph_band), corrections)

    config = full_config(cfg)  # fills channels=[16]*n_layers (D-C2) and N
    origin.setdefault("channels", {"origin": "default"})

    problems = check_dims(config["N"], config["c_frac"], config["d"], config["pen_frac"])
    if problems:
        p = problems[0]
        raise FoveaError(p["code"], p["message"], p["hint"])

    dims = derive_dims(config["N"], config["c_frac"], config["d"], config["pen_frac"])
    return {
        "config": config,
        "dims": dims,
        "base_label": base_label(dims, config["n_layers"]),
        "c_frac_effective": c_frac_eff,
        "c_frac_reason": cfrac_reason,
        "corrections": corrections,
        "derivation": {
            "window_size": int(window_size),
            "fractions": {"c_frac": config["c_frac"],
                          "pen_frac": config["pen_frac"], "d": config["d"]},
            "field_origin": {f: origin.get(f, {"origin": "default"})
                             for f in config},
        },
    }
