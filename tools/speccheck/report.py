"""ASCII report. U8.6 applies to this tool too: the console here is cp1252, and a
Greek delta in a last line already killed one overnight study.

Coverage is COMPUTED, never maintained by hand -- not here and not in CLAUDE.md.
"""

from __future__ import annotations

import re
import unicodedata

from .engine import NA, OK, UNVERIFIABLE, VIOLATED, Result

TYPE_NAMES = {
    1: "estructura", 2: "vistas", 3: "representacion", 4: "datos",
    5: "invariantes", 6: "numeros", 7: "operacion", 8: "lexico",
}


def ascii_(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _counts(results: list[Result]) -> dict[str, int]:
    return {s: sum(1 for r in results if r.state == s) for s in (OK, VIOLATED, UNVERIFIABLE, NA)}


def render(results: list[Result], verbose: bool = False, summary_only: bool = False) -> str:
    lines: list[str] = []
    types = sorted({r.rule.type for r in results})

    lines.append("")
    lines.append("  tipo                    ok  violada  no-verif  no-aplic   cobertura")
    lines.append("  " + "-" * 68)
    for t in types:
        rs = [r for r in results if r.rule.type == t]
        c = _counts(rs)
        covered = c[OK] + c[VIOLATED]
        pct = 100 * covered // len(rs) if rs else 0
        name = f"{t} {TYPE_NAMES.get(t, '')}"
        lines.append(f"  {name:<20} {c[OK]:>5} {c[VIOLATED]:>8} {c[UNVERIFIABLE]:>9} "
                     f"{c[NA]:>9} {pct:>10}%")

    c = _counts(results)
    covered = c[OK] + c[VIOLATED]
    strong = sum(1 for r in results if r.state in (OK, VIOLATED) and r.strength == "strong")
    total = len(results)
    lines.append("  " + "-" * 68)
    lines.append(f"  {'TOTAL':<20} {c[OK]:>5} {c[VIOLATED]:>8} {c[UNVERIFIABLE]:>9} "
                 f"{c[NA]:>9} {100 * covered // total if total else 0:>10}%")
    lines.append("")
    lines.append(f"  {total} reglas | cobertura mecanica {100 * covered // total if total else 0}% "
                 f"(fuertes {100 * strong // total if total else 0}%)")

    # A `no_aplicable` because the verb is not built yet is a ROADMAP entry, not a
    # ceiling: saying so is what keeps the coverage number from reading as final.
    pend: dict[str, int] = {}
    for r in results:
        for d in r.details:
            m = re.search(r"fase (\d)", d)
            if r.state == NA and m:
                pend[m.group(1)] = pend.get(m.group(1), 0) + 1
                break
        else:
            if r.state == NA:
                pend["live"] = pend.get("live", 0) + 1
    if pend:
        parts = [f"fase {k}: {v}" for k, v in sorted(pend.items()) if k != "live"]
        if "live" in pend:
            parts.append(f"solo con --live: {pend['live']}")
        lines.append(f"  pendientes por construir -> {' | '.join(parts)}")

    if summary_only:
        lines.append("")
        return "\n".join(ascii_(line) for line in lines)

    viol = [r for r in results if r.state == VIOLATED]
    if viol:
        lines.append("")
        lines.append(f"  VIOLADAS ({len(viol)}):")
        for r in viol:
            lines.append(f"    {r.rule.id}  {ascii_(r.rule.title)[:64]}")
            for d in r.details:
                lines.append(f"        {ascii_(d)}")

    unv = [r for r in results if r.state == UNVERIFIABLE]
    if unv:
        lines.append("")
        lines.append(f"  NO VERIFICABLES ({len(unv)}) -- lo que sigue dependiendo de una persona:")
        for r in unv:
            lines.append(f"    {r.rule.id}  {ascii_((r.details[0] if r.details else ''))[:70]}")

    if verbose:
        lines.append("")
        for r in results:
            lines.append(f"    {r.state:<15} {r.rule.id:<6} {ascii_(r.rule.title)[:52]}")
            for d in r.details:
                lines.append(f"        {ascii_(d)}")

    lines.append("")
    return "\n".join(ascii_(line) for line in lines)
