"""B1 -- the spec checking itself.

Runs as preflight on every invocation: if the specification is malformed, no
result computed from it means anything, so the run aborts instead of reporting a
comfortable green. Also exposed as the `spec_lint` verb for rules that ARE about
the documents (U2.4).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..engine import OK, VIOLATED, Check, Context, Outcome, verb
from ..extract import Spec
from .catalogs import EXTRACTORS

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+?)(?:#[^)]*)?\)")
REQUIRED_BY_SUBSTRATE = {
    "none": ("reason",),
    "same_as": ("target",),
    "delegated": ("target",),
}


def preflight(spec: Spec, root: Path) -> list[str]:
    problems = list(spec.problems)

    for rid, rule in spec.rules.items():
        checks = spec.checks.get(rid)
        if not checks:
            problems.append(f"{rid} ({rule.path.as_posix()}:{rule.line}): sin bloque check")
            continue
        for c in checks:
            sub = c.substrate
            if not sub:
                problems.append(f"{rid} en {c.where}: falta `substrate`")
                continue
            for field in REQUIRED_BY_SUBSTRATE.get(sub, ()):
                if not c.data.get(field):
                    problems.append(f"{rid} en {c.where}: substrate {sub} exige `{field}`")
            if sub not in REQUIRED_BY_SUBSTRATE:
                if not c.kind:
                    problems.append(f"{rid} en {c.where}: falta `kind`")
                if c.strength not in ("strong", "weak"):
                    problems.append(f"{rid} en {c.where}: `strength` debe ser strong o weak")
            if sub in ("same_as",) and str(c.data.get("target")) not in spec.rules:
                problems.append(f"{rid} en {c.where}: same_as apunta a una regla inexistente")

    for rid in spec.checks:
        if rid not in spec.rules:
            problems.append(f"bloque check huerfano: {rid} no es una regla")

    by_type: dict[int, list[int]] = {}
    for rule in spec.rules.values():
        by_type.setdefault(rule.type, []).append(rule.num)
    for t, nums in sorted(by_type.items()):
        expected = list(range(1, len(nums) + 1))
        if sorted(nums) != expected:
            problems.append(f"tipo {t}: numeracion con huecos o repetida ({sorted(nums)})")

    for rel in spec.files:
        text = (root / rel).read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http", "mailto")):
                continue
            if not (root / rel).parent.joinpath(target).resolve().exists():
                problems.append(f"{rel.as_posix()}: enlace roto -> {target}")

    return problems


@verb("spec_lint")
def spec_lint(check: Check, ctx: Context) -> Outcome:
    which = str(check.args.get("assert", ""))
    if which == "view_ids_unique":
        text = (ctx.root / "docs/ui/2-vistas.md").read_text(encoding="utf-8")
        ids = [i for row in text.splitlines() if row.startswith("| ")
               for i in re.findall(r"F0|V\d+|FG\d+", row.split("|")[1])]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            return Outcome(VIOLATED, f"ids de vista repetidos en el catalogo: {', '.join(dupes)}")
        if not ids:
            return Outcome(VIOLATED, "no se encontro el catalogo de vistas")
        return Outcome(OK, f"{len(ids)} ids de vista, sin repetidos")
    if which == "rules_have_checks":
        missing = [r for r in ctx.spec.rules if r not in ctx.spec.checks]
        if missing:
            return Outcome(VIOLATED, f"reglas sin bloque check: {', '.join(sorted(missing))}")
        return Outcome(OK, f"{len(ctx.spec.rules)} reglas, todas con bloque check")
    if which == "extractors_resolve":
        unknown = sorted({str(c.args.get(side))
                          for checks in ctx.spec.checks.values() for c in checks
                          for side in ("left", "right")
                          if c.kind == "catalog_match" and str(c.args.get(side)) not in EXTRACTORS
                          and c.args.get(side) is not None})
        if unknown:
            return Outcome(VIOLATED, f"extractores inexistentes: {', '.join(unknown)}")
        return Outcome(OK, "todos los extractores citados existen")
    return Outcome(VIOLATED, f"assert desconocido: {which}")
