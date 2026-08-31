"""Contracts (1) and (2): can this network train on this window dataset?

Pure functions of dictionaries (manifest x network config). No torch, no
training, milliseconds — which is the proof the validation sits in the right
layer (tests.md). EVERY training gate (POST /runs, fv-train, each sweep point)
calls check_run BEFORE reserving the run name; the laxest gate is the one a
sweep walks through.
"""

from __future__ import annotations

from fv.fovea import (EDGE_MODES, FoveaError, REGIONS, check_dims, dims_of,
                      normalize_geometry)


def check_network(net: dict) -> list[dict]:
    """Contract (2): the foveated geometry is self-consistent."""
    regions = net.get("regions", "split")
    if regions not in REGIONS:
        # refuse the value BEFORE deriving anything: an unknown `regions` silently
        # falling back to 'split' would train a foveated net while the config —
        # and the whole comparison built on it — claims it is the flat control
        return [{"code": "unknown_regions",
                 "message": f"regions '{regions}' no existe",
                 "hint": f"usa uno de {sorted(REGIONS)}: 'split' es la red foveada "
                         f"(dos ramas enmascaradas) y 'single' la CNN plana de "
                         f"control (una rama sobre todo el input)"}]
    single = regions == "single"
    # the geometry may arrive in either spelling; normalising here means the gate
    # refuses an ambiguous config (both spellings, or a bare `d`) with its reason
    # instead of guessing which half of it to believe
    try:
        geom = normalize_geometry(net)
    except FoveaError as e:
        return [e.as_dict()]
    problems = list(check_dims(geom, single))
    # in 'single' there is no peripheral branch, so its kernel/stride/merge
    # describe nothing and are not validated against a band that does not exist
    for key in (("k_center",) if single else ("k_center", "k_periph")):
        k = int(net.get(key, 3))
        if k % 2 == 0 or k < 3:
            problems.append({
                "code": "kernel_must_be_odd",
                "message": f"{key}={k}: un kernel par desalinea las mascaras (padding no entero)",
                "hint": "usa un kernel impar >= 3 (los rangos calculados solo generan impares)"})
    if not single and net.get("merge", "concat") == "sum" and \
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
    # channels: a per-layer vector of length n_layers (D-C3). Absent means the
    # legacy ch1/ch2 form (the builder resolves it); present means it must fit.
    n_layers = int(net.get("n_layers", 2))
    channels = net.get("channels")
    if channels is not None:
        if not isinstance(channels, (list, tuple)) or len(channels) != n_layers:
            problems.append({
                "code": "channels_length_mismatch",
                "message": f"channels={channels} debe tener longitud n_layers={n_layers}",
                "hint": "da un canal por capa: una lista de longitud n_layers"})
        elif any(int(c) < 1 for c in channels):
            problems.append({
                "code": "channels_must_be_positive",
                "message": f"channels={channels} tiene un valor < 1",
                "hint": "cada capa necesita al menos 1 canal"})
    # dropout: a probability. nn.Dropout raises on p outside [0, 1), and p=1.0
    # zeroes EVERYTHING — a net that trains on nothing and still writes a run.
    # Refused here, with the reason, before the name is reserved (R4).
    try:
        dropout = float(net.get("dropout", 0.0))
    except (TypeError, ValueError):
        dropout = None
    if dropout is None or not (0.0 <= dropout < 1.0):
        problems.append({
            "code": "dropout_out_of_range",
            "message": f"dropout={net.get('dropout')!r} no es una probabilidad en [0, 1)",
            "hint": "usa 0.0 (apagado) hasta 0.9; 1.0 apagaria TODAS las neuronas "
                    "y la cabeza no veria nada"})
    # edge_inputs: entradas extra a la CABEZA sobre el borde de la IMAGEN. Se
    # rechaza aqui por lo mismo que `regions`: un valor desconocido que cayera a
    # 'off' entrenaria una red SIN la senal mientras la config -- y el punto del
    # barrido construido sobre ella -- dice que la tiene.
    edge = net.get("edge_inputs", "off")
    if edge not in EDGE_MODES:
        problems.append({
            "code": "unknown_edge_inputs",
            "message": f"edge_inputs '{edge}' no existe",
            "hint": f"usa uno de {sorted(EDGE_MODES)}: 'off' (nada), 'pad' "
                    f"(que fraccion del margen es relleno, por lado) o 'dist' "
                    f"(a que distancia esta el borde de la imagen, en foveas)"})
    elif edge == "pad":
        try:
            border = int(normalize_geometry(net).get("border_px", 0))
        except FoveaError:
            border = None      # la geometria ya se quejo arriba; no se duplica
        if border == 0:
            problems.append({
                "code": "edge_pad_needs_border",
                "message": "edge_inputs='pad' con border_px=0 mide siempre 0: sin "
                           "margen no hay relleno del que dar una fraccion",
                "hint": "usa edge_inputs='dist' (mide contra la fovea y funciona "
                        "tambien en la CNN plana) o dale un border_px > 0"})
    if not problems and not single:
        dims = dims_of(net)
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
    dims = dims_of(net)
    window_size = int(manifest.get("config", {}).get("window_size", 0))
    if dims.fovea_px != window_size:
        problems.append({
            "code": "window_size_mismatch",
            "message": f"la fovea de la red es {dims.fovea_px}px y el dataset "
                       f"etiqueta ventanas de {window_size}px",
            "hint": f"elige un dataset con window_size {dims.fovea_px}, o pon "
                    f"fovea_px={window_size} en la red (el borde no cambia: sigue "
                    f"siendo border_px={dims.border_px})"})
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
