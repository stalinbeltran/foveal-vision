"""The study driver: validate the plan, and walk the OAT chain step by step.

Guides, never executes (D-H1): `advance` derives the base from the problem with
the carried winners and GENERATES the next sweep (base inline); the caller runs
it with the existing sweep machinery. `confirm` records the user-confirmed
winner, carries it forward, and — because the chain is dynamic — expands the
sub-axes a winner unlocks (n_layers=L -> channels[0..L-1]) lazily (§6.1).
"""

from __future__ import annotations

import re

from fv.metrics import MONITORS
from fv.models.builder import DEFAULT_CHANNEL, NETWORK_DEFAULTS
from fv.sweeps.generate import generate_sweep
from fv.sweeps.spec import (GEOMETRY_AUTO, LOSS_WEIGHT_PARAMS, NETWORK_PARAMS,
                            OBJECTIVES, RECIPE_PARAMS, WINDOW_SIZE_FIELDS)
from fv.sweeps.store import SweepStore
from fv.sweeps.winner import winner_overrides

CHANNELS_INDEXED = re.compile(r"^channels\[i\]$")
CHANNELS_AT = re.compile(r"^channels\[(\d+)\]$")


class StudyError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def _bad(problems, code, message, hint):
    problems.append({"code": code, "message": message, "hint": hint})


def validate_plan(plan: dict) -> list[dict]:
    """Problems of a plan, each with code/message/hint (nothing is created if any)."""
    problems: list[dict] = []
    if not plan.get("window_dataset"):
        _bad(problems, "window_dataset_required", "el estudio fija un B (window_dataset)",
             "da el nombre de un dataset de ventanas")
    obj = plan.get("objective", "f1")
    if obj not in OBJECTIVES:
        _bad(problems, "unknown_objective", f"objetivo '{obj}' no existe",
             f"usa uno de {sorted(OBJECTIVES)}")
    if int(plan.get("seeds", 3)) < 1:
        _bad(problems, "seeds_must_be_positive", "seeds debe ser >= 1",
             "1 = sondeo; sube a 3 para confirmar (D-M1)")
    axes = plan.get("axes", [])
    if not axes:
        _bad(problems, "no_axes", "el estudio no tiene ejes que barrer",
             "declara al menos un eje en 'axes' (orden = orden de barrido)")
    # Contract (9), at THIS gate too. check_sweep already refuses it, but a study
    # reaches that check inside `advance` — half a plan later, in the job. The
    # study screen simply did not offer 'loss' in its select, which HID the gap
    # instead of closing it: the laxer gate is the one an automated chain walks
    # through (same shape as the N/c_frac bug).
    loss_axes = sorted({a.get("axis") for a in axes} & LOSS_WEIGHT_PARAMS)
    if obj == "loss" and loss_axes:
        _bad(problems, "objective_varies_with_space",
             f"la loss no puede rankear un estudio que barre {loss_axes}: cada "
             f"punto se mediria con una perdida distinta y lambda->0 gana por "
             f"definicion",
             "usa 'f1' o 'pos_err_px' como objetivo del estudio")
    # The base network the study runs ON (plan-cnn-plana.md §3.1): the same plan
    # over the foveated base and over the flat control is the paired comparison.
    # Refused HERE, before anything is created — a bad field would otherwise
    # surface inside `advance`, half a study later, in the job (api.md R4).
    base_network = plan.get("base_network")
    if base_network is not None:
        if not isinstance(base_network, dict):
            _bad(problems, "base_network_must_be_a_map",
                 f"base_network={base_network!r} no es un mapa de campos de C",
                 "da {campo: valor}, p. ej. {regions: single, d: 1}")
        else:
            unknown = sorted(set(base_network) - NETWORK_PARAMS)
            if unknown:
                _bad(problems, "unknown_base_network_field",
                     f"base_network trae campos que no son de C: {unknown}",
                     f"campos válidos: {sorted(NETWORK_PARAMS)}")
            fixed = sorted(set(base_network) & WINDOW_SIZE_FIELDS)
            if fixed:
                _bad(problems, "base_network_breaks_window_size",
                     f"{fixed} fija center_out, que el contrato ①a ata al "
                     f"window_size de B: se DERIVA, no se escribe",
                     "quita N; para mover la fracción central usa el campo "
                     "'c_frac' del plan, que la derivación sí honra")
    cf = plan.get("c_frac")
    if cf is not None and not (isinstance(cf, (int, float)) and 0 < float(cf) <= 1):
        _bad(problems, "c_frac_out_of_range", f"c_frac={cf!r} debe estar en (0, 1]",
             "1.0 solo tiene sentido con base_network.regions='single' "
             "(sin anillo); si no, usa algo como 0.8")
    valid_fields = NETWORK_PARAMS | RECIPE_PARAMS
    for a in axes:
        axis = a.get("axis", "")
        rng = a.get("range", "auto")
        is_indexed = bool(CHANNELS_INDEXED.match(axis))
        if not is_indexed and axis not in valid_fields:
            _bad(problems, "unknown_axis", f"'{axis}' no es un campo de C/D ni channels[i]",
                 f"ejes válidos: {sorted(valid_fields)} o channels[i]")
        elif axis in WINDOW_SIZE_FIELDS:
            _bad(problems, "axis_breaks_window_size",
                 f"'{axis}' fija center_out (= round_to_even(N*c_frac)), que el "
                 f"contrato (1)a ata al window_size del dataset: barrerlo daria una "
                 f"fovea != la ventana etiquetada en cada punto",
                 "N y c_frac se derivan juntos del window_size (no se barren); barre "
                 "'d' para el contexto periferico, o usa un dataset con esa ventana")
        if rng == "auto" and not is_indexed and axis not in GEOMETRY_AUTO:
            _bad(problems, "auto_needs_geometry",
                 f"'{axis}' no tiene rango calculado: 'auto' solo vale para {sorted(GEOMETRY_AUTO)}",
                 "da la lista de valores explícita")
        elif rng != "auto" and (not isinstance(rng, list) or not rng):
            _bad(problems, "range_must_be_list", f"el eje '{axis}' necesita una lista o 'auto'",
                 "p. ej. [1, 2, 3]")
        elif axis == "monitor" and rng != "auto":
            # misma puerta que check_sweep: un objetivo ('f1') no es un monitor
            for v in rng:
                if v not in MONITORS:
                    _bad(problems, "unknown_monitor", f"'{v}' no es un monitor",
                         f"usa uno de {sorted(MONITORS)} (el monitor nombra la "
                         f"metrica de val con su 'val_' delante)")
    return problems


def _queue_from_plan(plan: dict) -> list[dict]:
    """The ordered concrete-axis queue. channels[i] stays a placeholder until the
    winning n_layers expands it (its length is unknown before then)."""
    q = []
    for a in plan["axes"]:
        axis = a["axis"]
        entry = {"axis": axis, "range": a.get("range", "auto"),
                 "depends_on": a.get("depends_on")}
        entry["kind"] = "channels_indexed" if CHANNELS_INDEXED.match(axis) else "field"
        q.append(entry)
    return q


def initial_progress(plan: dict) -> dict:
    """The step-0 progress for a plan: no steps run, no winners, queue derived
    from the axes. `progress.json` is regenerable live state (gitignored) whose
    committed source is `plan.json`; this reconstructs it when it is absent (a
    fresh clone, or a cleaned working tree)."""
    return {"format_version": 1, "steps": [], "winners": {},
            "queue": _queue_from_plan(plan)}


def create_study(name: str, plan: dict, store: StudyStore | None = None) -> dict:
    from fv.studies.store import StudyStore as _SS
    store = store or _SS()
    problems = validate_plan(plan)
    if problems:
        p = problems[0]
        raise StudyError(p["code"], p["message"], p["hint"])
    plan = dict(plan)
    plan.setdefault("format_version", 1)
    plan.setdefault("objective", "f1")
    plan.setdefault("seeds", 3)
    plan.setdefault("base_recipe", "corta")
    progress = initial_progress(plan)
    store.create(name, plan, progress)
    return {"name": name, "plan": plan, "progress": progress}


def _current_n_layers(winners: dict) -> int:
    w = winners.get("n_layers")
    if isinstance(w, dict):
        return int(w["value"])
    return int(NETWORK_DEFAULTS["n_layers"])


def _current_channels(winners: dict, n_layers: int) -> list[int]:
    w = winners.get("channels")
    if isinstance(w, dict):
        return list(w["value"])
    return [DEFAULT_CHANNEL] * n_layers


def _expand_channels_placeholder(desc: dict, winners: dict) -> list[dict]:
    """Turn channels[i] into one concrete axis per layer (§6.1), unlocked by the
    winning n_layers."""
    L = _current_n_layers(winners)
    return [{"axis": f"channels[{j}]", "range": desc["range"],
             "depends_on": desc.get("depends_on"), "kind": "channels_at", "index": j}
            for j in range(L)]


def _axis_and_range(desc: dict, winners: dict) -> tuple[str, object]:
    """The sweep axis field and its concrete range for a queue descriptor. A
    channels[j] step sweeps the `channels` field with index j varied over the
    range, the other indices held at the carried/base value."""
    if desc["kind"] == "channels_at":
        n_layers = _current_n_layers(winners)
        current = _current_channels(winners, n_layers)
        j = desc["index"]
        candidates = []
        for v in desc["range"]:
            vec = list(current)
            vec[j] = int(v)
            candidates.append(vec)
        return "channels", candidates
    return desc["axis"], desc["range"]


def _awaiting(progress: dict) -> dict | None:
    steps = progress["steps"]
    if steps and not steps[-1].get("confirmed"):
        return steps[-1]
    return None


def summarize(progress: dict) -> dict:
    """What a study is waiting for, derived from its progress. ONE definition,
    two readers: the detail (`status`) and the list. The list used to have no
    `next_axis` at all — the screen's «siguiente» column showed the dataset
    instead, a header and a cell disagreeing about what they meant."""
    awaiting = _awaiting(progress)
    queue = progress["queue"]
    next_axis = None
    if not awaiting and queue:
        head = queue[0]
        next_axis = (head["axis"] if head["kind"] != "channels_indexed"
                     else f"channels[i] (se expande a {_current_n_layers(progress['winners'])} sub-pasos)")
    return {"awaiting_confirmation": awaiting, "next_axis": next_axis,
            "done": awaiting is None and not queue}


def status(name: str, store: StudyStore | None = None) -> dict:
    from fv.studies.store import StudyStore as _SS
    store = store or _SS()
    plan = store.plan(name)
    progress = store.progress(name)
    return {"name": name, "plan": plan, "progress": progress,
            "steps": progress["steps"], "winners": progress["winners"],
            **summarize(progress)}


def advance(name: str, store: StudyStore | None = None,
            sstore: SweepStore | None = None, budget: dict | None = None) -> dict:
    """Derive + generate the next step's sweep (base inline, carried winners).
    Does NOT run it — the caller runs it with run_sweep. Refuses if the previous
    step still awaits the user's winner confirmation (guides, not executes)."""
    from fv.studies.store import StudyStore as _SS
    store = store or _SS()
    plan = store.plan(name)
    progress = store.progress(name)
    if _awaiting(progress) is not None:
        raise StudyError("step_awaiting_confirmation",
                         "el paso anterior espera que confirmes su ganador",
                         "confirma el ganador antes de avanzar (el estudio guía, no ejecuta)")
    queue = progress["queue"]
    if not queue:
        raise StudyError("study_done", "el estudio no tiene más ejes que barrer",
                         "revisa el ranking final o borra el estudio")
    # expand a channels[i] placeholder now that n_layers is known (§6.1)
    if queue[0]["kind"] == "channels_indexed":
        queue = _expand_channels_placeholder(queue[0], progress["winners"]) + queue[1:]

    desc = queue[0]
    winners = progress["winners"]
    axis, axis_range = _axis_and_range(desc, winners)
    step_i = len(progress["steps"])
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", desc["axis"])
    sweep_name = f"{name}-s{step_i}-{safe}"

    seeds = int(plan.get("seeds", 3))
    # The study's base network (plan-cnn-plana.md §3.1). A study fixes B, D and
    # the AXES, and derived C purely from the window_size — so the SAME plan
    # could not be run on the flat control, which is exactly the paired
    # comparison that control exists for. These go to the same derivation the
    # CLI's --overrides/--c-frac feed, so both entrances derive one base.
    enriched = generate_sweep(
        sweep_name, plan["window_dataset"], axis, axis_range,
        base_recipe=plan["base_recipe"], objective=plan["objective"],
        budget=budget or plan.get("budget", {}), winners=winners,
        overrides=dict(plan.get("base_network") or {}), c_frac=plan.get("c_frac"),
        seeds=seeds, study=name, sstore=sstore)

    step = {"step": step_i, "axis": desc["axis"], "kind": desc["kind"],
            "index": desc.get("index"), "sweep": sweep_name,
            "space_field": axis, "base_label": enriched["base_label"],
            "points": len(enriched["points"]), "discarded": len(enriched["discarded"]),
            "seeds": seeds, "confirmed": False, "winner": None}
    progress["steps"].append(step)
    progress["queue"] = queue[1:]
    store.set_progress(name, progress)
    return {"step": step, "spec": enriched}


def delete_study(name: str, store: StudyStore | None = None,
                 sstore: SweepStore | None = None, run_store=None) -> dict:
    """Delete a study AND every sweep it generated (which each cascade to their
    runs), as one unit. A study names each step's sweep deterministically, so
    leaving its sweeps behind orphans them: recreating the study with the same
    name would collide on the next advance (sweep_exists) — the reported bug.
    Refuses BEFORE deleting anything if any of its sweeps (or their runs) is
    live, so a study is never half-cascaded. Mirrors runner.delete_sweep."""
    from fv.studies.store import StudyStore as _SS
    from fv.sweeps.runner import delete_sweep
    from fv.training.registry import RunStore
    store = store or _SS()
    sstore = sstore or SweepStore()
    run_store = run_store or RunStore()
    if not store.exists(name):
        raise StudyError("study_not_found",
                         f"no existe el estudio '{name}'", "nada que borrar")
    sweeps = sstore.used_by_study(name)
    # pre-scan: refuse the WHOLE delete if any sweep or child run is live, so we
    # never delete some sweeps and then choke on a running one (R4: fail early)
    live = []
    for s in sweeps:
        if sstore.reconcile(s).get("status") in ("running", "queued"):
            live.append(s)
            continue
        live += [c for c in run_store.used_by_sweep(s)
                 if run_store.status(c).get("status") in ("running", "queued")]
    if live:
        raise StudyError(
            "study_has_live_sweeps",
            f"el estudio '{name}' tiene trabajo en marcha: {', '.join(sorted(set(live)))}",
            "paralo antes de borrar el estudio (para el recorrido/run activo)")
    for s in sweeps:                      # cascade: sweeps (y sus runs) primero
        delete_sweep(s, sstore, run_store)
    store.delete(name)                    # ... luego el estudio
    return {"deleted": name, "sweeps_deleted": sweeps}


def confirm(name: str, chosen_point: dict, store: StudyStore | None = None) -> dict:
    """Record the user-confirmed winner of the current step, carry it forward as
    a winner for the next step's derived base (§7), and advance the chain. A
    confirmed n_layers unlocks the channels[i] sub-axes lazily at the next
    advance."""
    from fv.studies.store import StudyStore as _SS
    store = store or _SS()
    progress = store.progress(name)
    step = _awaiting(progress)
    if step is None:
        raise StudyError("no_step_awaiting",
                         "no hay ningún paso esperando confirmación",
                         "genera el siguiente paso con advance")
    step["winner"] = chosen_point
    step["confirmed"] = True
    carried = winner_overrides(chosen_point, f"{name}/step-{step['step']}")
    progress["winners"].update(carried)
    step["unlocked"] = sorted(carried)
    store.set_progress(name, progress)
    return {"name": name, "confirmed_step": step["step"], "winners": progress["winners"]}
