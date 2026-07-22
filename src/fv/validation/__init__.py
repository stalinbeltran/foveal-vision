"""Contracts (1) and (2): can this network train on this window dataset?

Pure functions of dictionaries (manifest x network config). No torch, no
training, milliseconds — which is the proof the validation sits in the right
layer (tests.md). EVERY training gate (POST /runs, fv-train, each sweep point)
calls check_run BEFORE reserving the run name; the laxest gate is the one a
sweep walks through.
"""

from __future__ import annotations

from fv.fovea import check_dims, derive_dims


def check_network(net: dict) -> list[dict]:
    """Contract (2): the foveated geometry is self-consistent."""
    problems = list(check_dims(
        int(net.get("N", 0)), float(net.get("c_frac", 0.0)),
        int(net.get("d", 1)), float(net.get("pen_frac", 0.0))))
    for key in ("k_center", "k_periph"):
        k = int(net.get(key, 3))
        if k % 2 == 0 or k < 3:
            problems.append({
                "code": "kernel_must_be_odd",
                "message": f"{key}={k}: un kernel par desalinea las mascaras (padding no entero)",
                "hint": "usa un kernel impar >= 3 (los rangos calculados solo generan impares)"})
    if net.get("merge", "concat") == "sum" and \
            int(net.get("s_center", 1)) != int(net.get("s_periph", 1)):
        problems.append({
            "code": "merge_sum_needs_equal_strides",
            "message": f"merge: sum con s_center={net.get('s_center')} != "
                       f"s_periph={net.get('s_periph')} no alinea las ramas",
            "hint": "usa merge: concat (tolera dimensiones distintas) o iguala los strides"})
    if net.get("merge", "concat") not in ("sum", "concat"):
        problems.append({
            "code": "unknown_merge",
            "message": f"merge '{net.get('merge')}' no existe",
            "hint": "usa 'sum' o 'concat'"})
    if net.get("pool_mode", "avg") not in ("avg", "max"):
        problems.append({
            "code": "unknown_pool_mode",
            "message": f"pool_mode '{net.get('pool_mode')}' no existe",
            "hint": "usa 'avg' o 'max'"})
    if not problems:
        dims = derive_dims(net["N"], net["c_frac"], net["d"], net["pen_frac"])
        if int(net.get("k_periph", 3)) > 2 * dims.periph_band + 1:
            problems.append({
                "code": "kernel_exceeds_band",
                "message": f"k_periph={net.get('k_periph')} desborda la banda periferica "
                           f"({dims.periph_band}px)",
                "hint": f"usa un kernel de la lista calculada: {_krange(dims)}"})
    return problems


def _krange(dims) -> list[int]:
    from fv.fovea import kernel_range
    return kernel_range(dims.periph_band)


def check_compatible(manifest: dict, net: dict) -> list[dict]:
    """Contract (1): (1)a the labelled window is the fovea; (1)b the view is computable."""
    problems = check_network(net)
    if problems:
        return problems
    dims = derive_dims(net["N"], net["c_frac"], net["d"], net["pen_frac"])
    window_size = int(manifest.get("config", {}).get("window_size", 0))
    if dims.center_out != window_size:
        problems.append({
            "code": "window_size_mismatch",
            "message": f"la fovea de la red es {dims.center_out}px "
                       f"(N={net['N']}, c_frac={net['c_frac']}) y el dataset etiqueta "
                       f"ventanas de {window_size}px",
            "hint": f"elige un dataset con window_size {dims.center_out}, o una red "
                    f"cuya fovea sea {window_size} (p. ej. N={window_size + 2 * dims.periph_out}, "
                    f"c_frac={window_size}/{window_size + 2 * dims.periph_out})"})
    if not manifest.get("has_images", False):
        problems.append({
            "code": "view_needs_images",
            "message": "la vista foveada se recorta de las imagenes completas y este "
                       "dataset no las guarda",
            "hint": "reconstruye el dataset (fv-extract): B guarda `images` desde el dia 0"})
    return problems


def check_measurable(manifest: dict) -> list[dict]:
    problems = []
    per_split = manifest.get("windows_per_split", {})
    if int(per_split.get("val", 0)) <= 0:
        problems.append({
            "code": "no_validation_split",
            "message": "el dataset no tiene ventanas de val, asi que no hay con que "
                       "elegir best.pt ni con que medir",
            "hint": "reconstruye el dataset con una fraccion de val > 0: sin val, elegir "
                    "checkpoint cae en la perdida de entrenamiento, en silencio"})
    return problems


def check_run(manifest: dict, net: dict) -> list[dict]:
    """The single gate: compatibility + measurability."""
    return check_compatible(manifest, net) + check_measurable(manifest)
