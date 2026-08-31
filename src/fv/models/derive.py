"""G/C — derive a full network base config (C) from the problem (window_size).

The base a sweep step needs is not hand-written: it is DERIVED. Contract ①a
fixes fovea_px(C) == window_size(B), so the fovea comes straight from the
problem; every other field is a static default (§4, from NETWORK_DEFAULTS) with
carried winners and explicit user tunables applied on top. A default invalid for
this geometry (a `border_reduce` that does not divide the border, a kernel that
exceeds the band) falls to the nearest valid value WITH its reason — never
silently (barrido-por-ejes.md §5.2).

Since the 2026-08-25 reparameterisation the fovea is not SEARCHED for: it IS the
window size. The old `derive_geometry` looked for the smallest even N whose
fovea landed exactly on W, and loosened `c_frac` with a reason when none did —
that whole machinery existed only because the geometry was stated from the wrong
end. It survives as `legacy_border_px` for reading plans written before the
change, and nothing else calls it.

This module imports only fv.fovea and its own domain (fv.models) — contract ⑦:
it never sees fv.validation, whose extra gate (kernels/merge/channels +
measurability) the generator runs via check_run before reserving anything (§10).
"""

from __future__ import annotations

from fv.fovea import (FoveaError, border_range, check_dims, derive_dims, dims_of,
                      is_single_region, kernel_range, normalize_geometry,
                      overlap_border_range, overlap_fovea_range, reduce_range,
                      round_to_even)
from fv.models.builder import NETWORK_DEFAULTS, full_config

# The static context of a derived base (§4): everything NETWORK_DEFAULTS fixes
# EXCEPT fovea_px (it comes from W) and channels (derived from n_layers).
STATIC_FIELDS = ("border_px", "border_reduce", "overlap_fovea_px",
                 "overlap_border_px", "n_layers", "k_center", "k_periph",
                 "s_center", "s_periph", "merge", "pool_mode", "pad_mode",
                 "regions", "dropout", "edge_inputs")
DEFAULT_BORDER_PX = NETWORK_DEFAULTS["border_px"]
# the tolerance the pre-2026-08-25 derivation used when loosening c_frac (D-G3)
LEGACY_C_FRAC_TOLERANCE = 0.15


def legacy_border_px(window_size: int, c_frac: float, border_reduce: int = 2,
                     single_region: bool = False) -> int:
    """LEGACY SHIM — what a pre-2026-08-25 `c_frac` meant, in border px.

    Plans and specs written before the reparameterisation state the geometry as
    a central FRACTION, which only becomes a length once you know the N the old
    derivation would have chosen: the smallest even N whose fovea lands exactly
    on W, with a periphery of at least one cell. Reproduced here verbatim so
    those artefacts keep meaning exactly what they meant — read old, write new.
    Nothing else calls this, and nothing new should.
    """
    W = int(window_size)
    if W < 4 or W % 2 != 0:
        raise FoveaError("window_size_must_be_even",
                         f"window_size={W} debe ser par y >= 4",
                         "reconstruye B con una ventana par: el borde reparte simétrico")
    r = max(1, int(border_reduce))
    n_max = max(W * 4, W + 8)
    n_min = W if single_region else W + 2
    min_periph = 0 if single_region else 1
    for N in range(n_min, n_max + 1, 2):
        if (N - W) // 2 < min_periph:
            continue
        if round_to_even(N * float(c_frac)) == W:
            return ((N - W) // 2) * r
    # ...and the old D-G3 fallback: no even N hit W at that fraction, so the
    # derivation loosened c_frac to the value that does (W/N), smallest N within
    # tolerance. Reproduced verbatim, tolerance included, or the shim would
    # refuse geometries that used to be legal (c_frac=1.0 on a split base).
    for N in range(n_min, n_max + 1, 2):
        if (N - W) // 2 < min_periph:
            continue
        if abs(W / N - float(c_frac)) <= LEGACY_C_FRAC_TOLERANCE:
            return ((N - W) // 2) * r
    raise FoveaError(
        "no_feasible_border",
        f"ningun borde reproduce c_frac={c_frac} con una fovea de {W}px",
        "declara el borde en px: border_px (la geometria vieja ya no se escribe)")


def base_label(dims, n_layers: int) -> str:
    """The synthetic grouping key (D-H2), guion separator: ws16-p2-d2-L2.

    Deliberately UNCHANGED by the reparameterisation, and deliberately in cells:
    every sweep, study and report on disk cites labels in this shape, and a base
    that means the same thing must keep the same name. The overlaps are not in
    it — a sweep that varies one shares a base by construction, and the exact
    numbers travel in `base_network_value`.
    """
    return f"ws{dims.fovea_px}-p{dims.border_cells}-d{dims.border_reduce}-L{int(n_layers)}"


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
                overrides: dict | None = None, border_px: int | None = None) -> dict:
    """Derive a full base config from the problem.

    winners:   {field: {"value": v, "from": "<study/step>"}} — carried winners (§7).
    overrides: {field: v} — explicit user tunables (border_px/border_reduce/...), U5.
    border_px: the target border width in px (else the user override, else default).

    Returns {config, dims, base_label, corrections, derivation{window_size,
    geometry, field_origin}}.
    """
    winners = dict(winners or {})
    overrides = dict(overrides or {})

    def _asked(field, default):
        """What the caller asked for, before any derivation: an override wins,
        then a carried winner, then the static default."""
        if field in overrides:
            return overrides[field]
        if field in winners:
            w = winners[field]
            return w["value"] if isinstance(w, dict) else w
        return default

    W = int(window_size)
    if W < 4 or W % 2 != 0:
        raise FoveaError("window_size_must_be_even",
                         f"window_size={W} debe ser par y >= 4",
                         "reconstruye B con una ventana par: el borde reparte simétrico")

    # `regions` has to be known BEFORE the border is fixed: 'single' is the flat
    # control and a flat control with a ring is a DIFFERENT experiment (the bug
    # measured 2026-08-09, when asking for the flat base yielded ws16-p1-d1-L4).
    single = _asked("regions", NETWORK_DEFAULTS["regions"]) == "single"
    target_border = (border_px if border_px is not None
                     else _asked("border_px", DEFAULT_BORDER_PX))

    cfg = {f: NETWORK_DEFAULTS[f] for f in STATIC_FIELDS}
    origin: dict[str, dict] = {f: {"origin": "default"} for f in cfg}
    for f, w in winners.items():
        cfg[f] = w["value"] if isinstance(w, dict) else w
        origin[f] = {"origin": "winner",
                     "from": w.get("from") if isinstance(w, dict) else None}
    for f, v in overrides.items():
        cfg[f] = v
        origin[f] = {"origin": "user"}
    cfg["fovea_px"] = W                       # contract ①a: taken, never searched
    origin["fovea_px"] = {"origin": "problem", "from": "window_size"}
    cfg["border_px"] = int(target_border)
    if border_px is not None:
        # asked for explicitly (the --border-px flag): say so. U1.6 — an object
        # shows the definition it was made with, not a plausible-looking one.
        origin["border_px"] = {"origin": "user"}

    corrections: list[dict] = []
    if single and cfg["border_px"] != 0:
        corrections.append({
            "field": "border_px", "from": cfg["border_px"], "to": 0,
            "reason": "regions='single' es la CNN plana: una sola rama sobre todo "
                      "el input, sin borde. El borde cae a 0"})
        cfg["border_px"] = 0
    if not single:
        _correct(cfg, "border_px", border_range(W, 1), corrections)
        _correct(cfg, "border_reduce", reduce_range(cfg["border_px"]), corrections)
        _correct(cfg, "overlap_fovea_px", overlap_fovea_range(W), corrections)
        _correct(cfg, "overlap_border_px",
                 overlap_border_range(cfg["border_px"], cfg["border_reduce"]), corrections)

    dims = derive_dims(normalize_geometry(cfg), single_region=single)
    _correct(cfg, "k_center", kernel_range(dims.center_band), corrections)
    if not single:
        _correct(cfg, "k_periph", kernel_range(dims.periph_band), corrections)

    config = full_config(cfg)  # fills channels=[16]*n_layers (D-C2)
    origin.setdefault("channels", {"origin": "default"})

    problems = check_dims(normalize_geometry(config), is_single_region(config))
    if problems:
        p = problems[0]
        raise FoveaError(p["code"], p["message"], p["hint"])

    dims = dims_of(config)
    return {
        "config": config,
        "dims": dims,
        "base_label": base_label(dims, config["n_layers"]),
        "corrections": corrections,
        "derivation": {
            "window_size": W,
            # the geometry, in the spelling it is now written in: real px
            "geometry": {"fovea_px": config["fovea_px"],
                         "border_px": config["border_px"],
                         "border_reduce": config["border_reduce"],
                         "overlap_fovea_px": config["overlap_fovea_px"],
                         "overlap_border_px": config["overlap_border_px"]},
            "field_origin": {f: origin.get(f, {"origin": "default"})
                             for f in config},
        },
    }
