"""Dispatch, the four states, and the rules that keep a green from lying.

Design points that are NOT negotiable (docs/ui/validador.md §2, §3):
  - a rule with no verb and no handler is `no_verificable`, never `ok`;
  - a check whose scope matches nothing is a CONFIGURATION violation, not a pass;
  - a verb that is declared in the spec but not built yet is `no_aplicable`
    carrying the phase that will build it -- so the report doubles as roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .extract import Check, Rule, Spec

OK = "ok"
VIOLATED = "violada"
UNVERIFIABLE = "no_verificable"
NA = "no_aplicable"

# Worst wins when a rule carries several checks.
ORDER = {OK: 0, UNVERIFIABLE: 1, NA: 2, VIOLATED: 3}

LIVE_SUBSTRATES = {"http", "dom"}

# Declared in validador.md, built in a later phase. The report says which.
PLANNED: dict[str, int] = {
    "ports_free": 4,
}


@dataclass
class Outcome:
    state: str
    detail: str = ""
    strength: str = "strong"


@dataclass
class Result:
    rule: Rule
    state: str
    strength: str
    details: list[str]


@dataclass
class Context:
    root: Path
    mode: str  # "static" | "live"
    spec: Spec
    base_url: str = ""   # backend to interrogate in --live (empty in static)
    front_url: str = ""  # vite, for the DOM substrate
    started_ports: tuple[int, ...] = ()  # what THIS run brought up


VERBS: dict[str, Callable[[Check, Context], Outcome]] = {}


def verb(name: str):
    def deco(fn):
        VERBS[name] = fn
        return fn
    return deco


def _evaluate_check(check: Check, ctx: Context, seen: set[str]) -> Outcome:
    sub = check.substrate

    if sub == "none":
        reason = check.data.get("reason")
        if not reason:
            return Outcome(VIOLATED, "substrate: none sin reason (lo exige el formato)")
        return Outcome(UNVERIFIABLE, str(reason))

    if sub == "same_as":
        target = str(check.data.get("target", ""))
        if target in seen:
            return Outcome(VIOLATED, f"same_as circular con {target}")
        if target not in ctx.spec.rules:
            return Outcome(VIOLATED, f"same_as apunta a {target}, que no existe")
        res = evaluate_rule(ctx.spec.rules[target], ctx, seen | {check.rule_id})
        return Outcome(res.state, f"misma comprobacion que {target}: {'; '.join(res.details) or 'ok'}")

    if sub == "delegated":
        target = str(check.data.get("target", ""))
        name = target.split("::")[-1]
        hits = [p for p in (ctx.root / "tests").rglob("*.py")
                if name and name in p.read_text(encoding="utf-8", errors="replace")]
        if not name:
            return Outcome(VIOLATED, "delegated sin target")
        if not hits:
            return Outcome(VIOLATED, f"delegated a {target}, que no existe en tests/")
        return Outcome(OK, f"delegada en {hits[0].relative_to(ctx.root).as_posix()}")

    kind = check.kind
    if not kind:
        return Outcome(UNVERIFIABLE, "sin verbo ni handler declarado")

    if kind in PLANNED and kind not in VERBS:
        return Outcome(NA, f"verbo no implementado (fase {PLANNED[kind]})", check.strength)

    if kind not in VERBS:
        return Outcome(VIOLATED, f"verbo desconocido: {kind}")

    if ctx.mode == "static" and sub in LIVE_SUBSTRATES:
        return Outcome(NA, f"sustrato {sub} no disponible en modo estatico", check.strength)

    try:
        out = VERBS[kind](check, ctx)
    except Exception as exc:  # a broken check is a violation of the check, not a pass
        return Outcome(VIOLATED, f"el check revento: {type(exc).__name__}: {exc}")
    out.strength = check.strength
    return out


def evaluate_rule(rule: Rule, ctx: Context, seen: set[str] | None = None) -> Result:
    seen = seen or set()
    checks = ctx.spec.checks.get(rule.id, [])
    if not checks:
        return Result(rule, VIOLATED, "strong", ["sin bloque check (lo exige el formato A2)"])
    outs = [_evaluate_check(c, ctx, seen) for c in checks]
    state = max((o.state for o in outs), key=lambda s: ORDER[s])
    strength = "weak" if any(o.strength == "weak" for o in outs) else "strong"
    details = [f"[{o.state}] {o.detail}" for o in outs if o.detail]
    return Result(rule, state, strength, details)


def run(ctx: Context, only: list[str] | None = None) -> list[Result]:
    rules = [r for r in ctx.spec.rules.values() if not only or r.id in only]
    return [evaluate_rule(r, ctx) for r in sorted(rules, key=lambda r: (r.type, r.num))]
