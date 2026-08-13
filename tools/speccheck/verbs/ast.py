"""Fase 2 -- structural facts about the front (substrate B3) and the seams (B7).

The facts come from `tools/speccheck/tsfacts.mjs` (TypeScript compiler API); this
module only judges them. The distinction that pays for the side-car: a COMMENT
mentioning `corner_order` is not a second definition of it, and only a parser can
tell the two apart.

No node -> `no_aplicable`, never `ok`.
"""

from __future__ import annotations

import json
import re
import subprocess
from fnmatch import fnmatch

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb

_FACTS: dict[str, dict] = {}


def facts(ctx: Context) -> tuple[dict | None, str]:
    key = str(ctx.root)
    if key in _FACTS:
        return _FACTS[key], ""
    try:
        proc = subprocess.run(
            ["node", "tools/speccheck/tsfacts.mjs", ".", "web/src"],
            cwd=ctx.root, capture_output=True, text=True, timeout=180, shell=True,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"no se pudo ejecutar node: {exc}"
    if not proc.stdout.strip():
        return None, f"tsfacts no devolvio JSON: {proc.stderr.strip()[:200]}"
    data = json.loads(proc.stdout)
    _FACTS[key] = data
    return data, ""


def _files(data: dict, scope: str | None):
    for rel, f in data["files"].items():
        if not scope or fnmatch(rel, scope):
            yield rel, f


@verb("single_definition")
def single_definition(check: Check, ctx: Context) -> Outcome:
    """A shared vocabulary has ONE definition. Everyone else reads it."""
    data, err = facts(ctx)
    if data is None:
        return Outcome(NA, err)
    problems: list[str] = []
    checked = 0
    for seam in check.args.get("seams") or []:
        name = str(seam.get("name", "?"))
        owner = seam.get("owner")
        markers = [str(m) for m in (seam.get("markers") or [])]
        need = int(seam.get("min_markers", 2))
        if not markers:
            return Outcome(VIOLATED, f"costura {name} sin markers: configuracion del check")
        checked += 1
        for rel, f in _files(data, seam.get("scope")):
            if owner and fnmatch(rel, str(owner)):
                continue
            hits = {s["v"] for s in f["strings"] if s["v"] in markers}
            if len(hits) >= need:
                problems.append(f"{name}: {rel} define {sorted(hits)}")
    if problems:
        return Outcome(VIOLATED, "; ".join(problems))
    return Outcome(OK, f"{checked} costura(s), una sola definicion cada una")


@verb("ast_query")
def ast_query(check: Check, ctx: Context) -> Outcome:
    data, err = facts(ctx)
    if data is None:
        return Outcome(NA, err)
    a = check.args
    scope = check.data.get("scope")
    scope = None if not scope else str(scope)
    files = list(_files(data, scope))
    if not files:
        return Outcome(VIOLATED, f"scope vacio ({scope}): configuracion del check")
    bad: list[str] = []
    did = []

    if a.get("forbid_identifiers"):
        did.append("identificadores")
        banned = set(a["forbid_identifiers"])
        for rel, f in files:
            hit = banned & set(f["identifiers"])
            if hit:
                bad.append(f"{rel}: {', '.join(sorted(hit))}")

    if a.get("forbid_string_in_timer"):
        did.append("literales dentro de un sondeo")
        banned = [str(x) for x in a["forbid_string_in_timer"]]
        for rel, f in files:
            for s in f["strings"]:
                if s["timer"] and any(b in s["v"] for b in banned):
                    bad.append(f"{rel}:{s['line']} {s['v']!r} dentro de setInterval/setTimeout")

    if a.get("string_only_in"):
        spec = a["string_only_in"]
        did.append(f"el literal {spec['value']!r}")
        allow = [str(x) for x in spec.get("files") or []]
        for rel, f in files:
            if any(fnmatch(rel, p) for p in allow):
                continue
            for s in f["strings"]:
                if str(spec["value"]) in s["v"]:
                    bad.append(f"{rel}:{s['line']}")

    if a.get("require_any_identifier"):
        did.append("identificador obligatorio")
        need = set(a["require_any_identifier"])
        for rel, f in files:
            if not need & set(f["identifiers"]):
                bad.append(f"{rel}: falta {' o '.join(sorted(need))}")

    if a.get("call_guard"):
        spec = a["call_guard"]
        did.append(f"guarda de {spec['callee']}")
        found = False
        for rel, f in files:
            for c in f["calls"]:
                if c["callee"] == spec["callee"]:
                    found = True
                    if str(spec["condition_contains"]) not in c["guard"]:
                        bad.append(f"{rel}:{c['line']} sin la guarda "
                                   f"{spec['condition_contains']!r} (guarda: {c['guard'][:60]!r})")
        if not found:
            bad.append(f"no existe ninguna llamada a {spec['callee']}: configuracion del check")

    if a.get("forbid_numeric_comparison"):
        did.append("umbral escrito en el codigo")
        for rel, f in files:
            for c in f["numericComparisons"]:
                for pat in a["forbid_numeric_comparison"]:
                    if re.search(str(pat), c["text"]):
                        bad.append(f"{rel}:{c['line']} {c['text']!r}")

    if not did:
        return Outcome(NA, "la regla no declara todavia una consulta ejecutable")
    if bad:
        return Outcome(VIOLATED, "; ".join(bad[:6]) + (f" (+{len(bad) - 6})" if len(bad) > 6 else ""))
    return Outcome(OK, f"{', '.join(did)} sobre {len(files)} fichero(s)")


@verb("settle_guard")
def settle_guard(check: Check, ctx: Context) -> Outcome:
    """U4.5, as its own handler because the bug it prevents was expensive."""
    return ast_query(check, ctx)


@verb("error_hint_propagated")
def error_hint_propagated(check: Check, ctx: Context) -> Outcome:
    """U5.2: every screen renders errors through the component that shows the hint."""
    return ast_query(check, ctx)
