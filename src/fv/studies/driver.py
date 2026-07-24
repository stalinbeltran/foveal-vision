"""The study driver: validate the plan, and walk the OAT chain step by step.

Guides, never executes (D-H1): `advance` derives the base from the problem with
the carried winners and GENERATES the next sweep (base inline); the caller runs
it with the existing sweep machinery. `confirm` records the user-confirmed
winner, carries it forward, and — because the chain is dynamic — expands the
sub-axes a winner unlocks (n_layers=L -> channels[0..L-1]) lazily (§6.1).
"""

from __future__ import annotations

import re

from fv.models.builder import DEFAULT_CHANNEL, NETWORK_DEFAULTS
from fv.sweeps.generate import generate_sweep
from fv.sweeps.spec import (GEOMETRY_AUTO, NETWORK_PARAMS, OBJECTIVES,
                            RECIPE_PARAMS)
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
    valid_fields = NETWORK_PARAMS | RECIPE_PARAMS
    for a in axes:
        axis = a.get("axis", "")
        rng = a.get("range", "auto")
        is_indexed = bool(CHANNELS_INDEXED.match(axis))
        if not is_indexed and axis not in valid_fields:
            _bad(problems, "unknown_axis", f"'{axis}' no es un campo de C/D ni channels[i]",
                 f"ejes válidos: {sorted(valid_fields)} o channels[i]")
        if rng == "auto" and not is_indexed and axis not in GEOMETRY_AUTO:
            _bad(problems, "auto_needs_geometry",
                 f"'{axis}' no tiene rango calculado: 'auto' solo vale para {sorted(GEOMETRY_AUTO)}",
                 "da la lista de valores explícita")
        elif rng != "auto" and (not isinstance(rng, list) or not rng):
            _bad(problems, "range_must_be_list", f"el eje '{axis}' necesita una lista o 'auto'",
                 "p. ej. [1, 2, 3]")
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
    progress = {"format_version": 1, "steps": [], "winners": {},
                "queue": _queue_from_plan(plan)}
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


def status(name: str, store: StudyStore | None = None) -> dict:
    from fv.studies.store import StudyStore as _SS
    store = store or _SS()
    plan = store.plan(name)
    progress = store.progress(name)
    awaiting = _awaiting(progress)
    queue = progress["queue"]
    next_axis = None
    if not awaiting and queue:
        head = queue[0]
        next_axis = (head["axis"] if head["kind"] != "channels_indexed"
                     else f"channels[i] (se expande a {_current_n_layers(progress['winners'])} sub-pasos)")
    return {"name": name, "plan": plan, "progress": progress,
            "steps": progress["steps"], "winners": progress["winners"],
            "awaiting_confirmation": awaiting, "next_axis": next_axis,
            "done": awaiting is None and not queue}


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

    enriched = generate_sweep(
        sweep_name, plan["window_dataset"], axis, axis_range,
        base_recipe=plan["base_recipe"], objective=plan["objective"],
        budget=budget or plan.get("budget", {}), winners=winners,
        study=name, sstore=sstore)

    step = {"step": step_i, "axis": desc["axis"], "kind": desc["kind"],
            "index": desc.get("index"), "sweep": sweep_name,
            "space_field": axis, "base_label": enriched["base_label"],
            "points": len(enriched["points"]), "discarded": len(enriched["discarded"]),
            "confirmed": False, "winner": None}
    progress["steps"].append(step)
    progress["queue"] = queue[1:]
    store.set_progress(name, progress)
    return {"step": step, "spec": enriched}


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
