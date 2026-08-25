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
from fv.metrics import MONITORS, VAL_METRICS
from fv.models.builder import DEFAULT_CHANNEL, NETWORK_DEFAULTS, full_config
from fv.training.recipe import Recipe
from fv.validation import check_network

NETWORK_PARAMS = set(NETWORK_DEFAULTS)
RECIPE_PARAMS = set(Recipe().as_dict())
LOSS_WEIGHT_PARAMS = {"lambda_pos", "pos_weight", "smooth_l1_beta"}
GEOMETRY_AUTO = {"k_center", "k_periph", "s_center", "s_periph",
                 "border_px", "border_reduce", "overlap_fovea_px", "overlap_border_px"}
# The fovea IS the labelled window (contract ①a), and B's window_size is fixed for
# the whole sweep — so a sweep that varies it makes EVERY point violate ①a. It is
# taken from the problem (fv.models.derive), never swept: refused HERE, with the
# reason, before any point is reserved — not silently trained and failed deep in
# the job (R4, §10). The border is NOT here: since the 2026-08-25
# reparameterisation it is an independent length and a first-class axis.
WINDOW_SIZE_FIELDS = {"fovea_px"}
# Spellings that no longer mean what they used to. Refused as axes with the
# reason, never re-interpreted: `d` used to grow the context (border = cells*d)
# and now only says how coarsely a border of fixed size is condensed, so an old
# spec re-run verbatim would train DIFFERENT networks in silence.
RENAMED_AXES = {
    "N": ("fovea_px/border_px", "N se DERIVA (fovea_px + 2*border_px/border_reduce): "
          "barre 'border_px' para cambiar cuanto contexto ve la red"),
    "c_frac": ("fovea_px/border_px", "c_frac se DERIVA de la fovea y el borde: "
               "barre 'border_px'"),
    "pen_frac": ("overlap_fovea_px", "el solape se declara en px de fovea, no como "
                 "fraccion de N"),
    "d": ("border_reduce", "'d' cambio de significado: hoy el borde es 'border_px' "
          "y 'border_reduce' solo dice cuantos px caben en una celda. Para barrer "
          "cuanto contexto ve la red, barre 'border_px'"),
}
# What H can rank by IS what a val record measures, with its direction: the same
# table fv.metrics uses to choose best.pt, read from here (it was written twice).
OBJECTIVES = dict(VAL_METRICS)


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
        if param in RENAMED_AXES:
            new_name, why = RENAMED_AXES[param]
            bad("axis_renamed",
                f"'{param}' ya no es un eje: la geometria se reparametrizo "
                f"(2026-08-25) y ahora se declara en px reales",
                f"usa {new_name}. {why}")
        elif param not in NETWORK_PARAMS | RECIPE_PARAMS:
            bad("unknown_space_param", f"'{param}' no es un campo de C ni de D",
                f"los ejes validos son {sorted(NETWORK_PARAMS | RECIPE_PARAMS)}")
        elif param in WINDOW_SIZE_FIELDS:
            bad("axis_breaks_window_size",
                f"'{param}' es la fovea, que el contrato (1)a ata al window_size "
                f"del dataset: barrerlo hace que cada punto tenga una fovea != la "
                f"ventana etiquetada",
                "la fovea se TOMA del window_size (no se barre); para variar el "
                "contexto barre 'border_px', y para cambiar la fovea "
                "usa/reconstruye un dataset con ese window_size")
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
        elif param == "monitor":
            # a monitor is not an objective: 'f1' names the value but not its
            # direction, and best.pt would keep the WORST epoch without a word
            for v in values:
                if v not in MONITORS:
                    bad("unknown_monitor", f"'{v}' no es un monitor",
                        f"usa uno de {sorted(MONITORS)} (el monitor nombra la "
                        f"metrica de val con su 'val_' delante)")
    problems.extend(_check_couple(spec.get("couple") or {}, space))
    return problems


def _check_couple(couple: dict, space: dict) -> list[dict]:
    """Valida las ATADURAS: campos que se mueven CON un eje, no contra el.

    Ver `expand_points` para que son y por que hacen falta. Aqui solo se
    comprueba que la atadura sea legible y no mienta:

    - el campo atado es un campo de C/D, y NO es a su vez un eje (seria las dos
      cosas a la vez: el producto cartesiano y la diagonal);
    - el eje del que cuelga existe en el espacio y no es `seed` (la replica);
    - hay exactamente un valor atado por valor del eje. Una lista mas corta o mas
      larga es la forma silenciosa de que la diagonal se desalinee, y entonces el
      recorrido entrena redes que nadie pidio con una tabla que sale igual de
      creible.
    """
    problems = []

    def bad(code, message, hint):
        problems.append({"code": code, "message": message, "hint": hint})

    if not isinstance(couple, dict):
        bad("couple_must_be_dict", "'couple' debe ser un diccionario "
            "{campo: {axis, values}}",
            'p. ej. {"border_reduce": {"axis": "border_px", "values": [4,5,6,8]}}')
        return problems
    for field, regla in couple.items():
        if field not in NETWORK_PARAMS | RECIPE_PARAMS:
            bad("unknown_couple_field", f"'{field}' no es un campo de C ni de D",
                f"los campos atables son {sorted(NETWORK_PARAMS | RECIPE_PARAMS)}")
            continue
        if field in space:
            bad("couple_field_is_axis",
                f"'{field}' es eje Y campo atado a la vez: o barre libre (producto "
                f"cartesiano) o va atado a otro eje (diagonal), no las dos",
                f"quita '{field}' de 'space' o quitalo de 'couple'")
            continue
        if not isinstance(regla, dict):
            bad("couple_rule_must_be_dict", f"la atadura de '{field}' debe ser "
                f"{{axis, values}}",
                'p. ej. {"axis": "border_px", "values": [4,5,6,8]}')
            continue
        eje = regla.get("axis")
        valores = regla.get("values")
        if eje not in space:
            bad("couple_axis_not_in_space",
                f"'{field}' cuelga de '{eje}', que no es un eje de este recorrido",
                f"los ejes declarados son {sorted(space)}")
            continue
        if eje == "seed":
            bad("couple_axis_is_seed",
                f"'{field}' no puede colgar de 'seed': la semilla es el eje "
                f"REPLICA, y atarle un campo haria que cada replica entrenara una "
                f"red distinta",
                "cuelgalo del eje de verdad")
            continue
        if not isinstance(valores, list) or not valores:
            bad("couple_values_must_be_list",
                f"la atadura de '{field}' necesita una lista 'values'",
                'p. ej. {"axis": "border_px", "values": [4,5,6,8]}')
            continue
        n_eje = space[eje]
        if isinstance(n_eje, list) and len(valores) != len(n_eje):
            bad("couple_length_mismatch",
                f"'{field}' trae {len(valores)} valores atados y el eje '{eje}' "
                f"tiene {len(n_eje)}: la diagonal quedaria desalineada",
                f"da exactamente {len(n_eje)} valores, uno por cada {eje}")
        elif n_eje == "auto":
            bad("couple_axis_is_auto",
                f"'{field}' cuelga de '{eje}', que es 'auto': el rango lo calcula "
                f"la geometria y no se sabe aqui cuantos valores tendra",
                f"declara el rango de '{eje}' explicito para poder atarle "
                f"'{field}'")
    return problems


def expand_points(spec: dict, base_network: dict) -> tuple[list[dict], list[dict]]:
    """-> (valid points, discarded points with reasons). A point is
    {network: {...}, recipe_overrides: {...}}.

    ATADURAS (`spec["couple"]`) — un campo que se mueve CON el eje
    ---------------------------------------------------------------
    El motor es OAT y el espacio se expande en PRODUCTO CARTESIANO, que es lo
    correcto mientras cada eje sea independiente. Pero hay preguntas donde dos
    campos tienen que moverse **a la vez** para que el experimento signifique lo
    que dice, y entonces lo que hace falta es la DIAGONAL, no el producto.

    El caso que lo trajo (estudio 1 de prioridad 1, 2026-08-25): *«¿ayuda ver mas
    contexto, a coste constante?»*. Eso es barrer `border_px` ∈ [8,10,12,16]
    manteniendo el anillo en 2 celdas, o sea `border_reduce` = `border_px`/2 ∈
    [4,5,6,8]. Como producto cartesiano serian 16 combinaciones, de las que la
    geometria acepta 7 (las que dividen) — y entre ellas `border_px`=8 saldria
    con 2 celdas Y con 1, que son DOS REDES DISTINTAS agregadas bajo el mismo
    valor del eje. La tabla saldria igual de creible y estaria midiendo otra cosa.

    Con `couple` el recorrido declara la diagonal:

        "space":  {"border_px": [8, 10, 12, 16]},
        "couple": {"border_reduce": {"axis": "border_px",
                                     "values": [4, 5, 6, 8]}}

    El campo atado entra en `overrides` como cualquier otro, asi que viaja al
    nombre del run y al agrupado por punto: queda escrito que la red de
    `border_px`=8 tenia `border_reduce`=4, en vez de quedar implicito.

    ⚠ Lo que una atadura NO es: un eje. No multiplica puntos, no se rankea por
    ella y `estudio_informe.py --eje` sigue siendo el eje de verdad. Y no se
    aplica sola por conveniencia geometrica: hay que escribirla, porque mantener
    el anillo fijo es una DECISION de diseño del experimento (la alternativa
    —`border_reduce` fijo— tambien es legal, solo que entonces N crece y el coste
    con el, y eso hay que decirlo en el plan).
    """
    base = full_config(base_network)
    ss = build_search_space(base, n_layers=int(base["n_layers"]))
    space: dict[str, list] = {}
    for param, values in spec.get("space", {}).items():
        if values == "auto":
            space[param] = ss[param]
        else:
            space[param] = list(values)

    # {eje: {valor_del_eje_por_indice: {campo: valor}}} — resuelto por POSICION
    # contra el rango del eje, que es como se declara y como lo valida check_sweep
    atados: dict[str, list[tuple[str, list]]] = {}
    for field, regla in (spec.get("couple") or {}).items():
        eje = regla.get("axis")
        if eje in space:
            atados.setdefault(eje, []).append((field, list(regla.get("values") or [])))

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
        for eje, reglas in atados.items():
            i = space[eje].index(overrides[eje])
            for field, valores in reglas:
                if i < len(valores):
                    overrides[field] = valores[i]
        net = dict(base)
        net.update({k: v for k, v in overrides.items() if k in NETWORK_PARAMS})
        # channels depends on n_layers (§6.1): sweeping depth WITHOUT sweeping
        # channels resizes the vector, so the point stays valid instead of
        # carrying the base's stale channel length.
        #
        # QUE anchura, y por que no siempre la del default: si la base declara un
        # ancho UNIFORME propio, se conserva ese. Poner [16]*L de oficio sobre una
        # base de, por ejemplo, [22,22,22,22] movería la ANCHURA al barrer la
        # PROFUNDIDAD -- dos cosas a la vez en un eje que dice medir una, y sin
        # decirlo. Es el fallo silencioso que el proyecto persigue: el eje mediría
        # otra cosa y la tabla saldría igual de creíble.
        # Para una base con [16]*L -- todos los recorridos anteriores a esto -- el
        # ancho uniforme ES 16, así que el comportamiento no cambia en nada ya
        # medido.
        if "n_layers" in overrides and "channels" not in overrides:
            base_ch = list(base.get("channels") or [])
            ancho = (int(base_ch[0]) if base_ch and len(set(base_ch)) == 1
                     else DEFAULT_CHANNEL)
            net["channels"] = [ancho] * int(overrides["n_layers"])
        recipe_over = {k: v for k, v in overrides.items() if k in RECIPE_PARAMS}
        problems = check_network(net)
        if problems:
            discarded.append({"point": overrides, "problems": problems})
        else:
            valid.append({"overrides": overrides, "network": net,
                          "recipe_overrides": recipe_over})
    return valid, discarded
