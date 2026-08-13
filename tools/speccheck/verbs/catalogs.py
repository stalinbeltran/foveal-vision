"""`catalog_match`: a table written in the documents versus what the code declares.

This is the verb that pays for the tables in docs/ui/: they stop being prose and
become the spec. Comparison is bidirectional by default -- an orphan on EITHER
side is a finding, because "the doc lists a screen nobody built" and "the code
has a screen nobody wrote down" are both the same failure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb
from .fs import iter_files

EXTRACTORS: dict[str, Callable[[Context], set[str]]] = {}

# Named in the check blocks, built in a later phase. Naming them here is what
# keeps a rule that needs them at `no_aplicable` instead of "extractor unknown".
PLANNED_EXTRACTORS: dict[str, int] = {
    "dom_data_domain": 4,
    "dom_data_view": 4,
    "dom_view_subtitles": 4,
    "backend_error_codes": 3,
    "front_error_codes": 3,
    "backend_states": 3,
    "badge_states": 4,
}


def extractor(name: str):
    def deco(fn):
        EXTRACTORS[name] = fn
        return fn
    return deco


def _read(ctx: Context, rel: str) -> str:
    return (ctx.root / rel).read_text(encoding="utf-8", errors="replace")


def _table_column(text: str, heading: str, column: int) -> set[str]:
    """Cells of a markdown table that follows `heading`, one column, cleaned."""
    out: set[str] = set()
    body = text.split(heading, 1)[-1]
    for line in body.splitlines():
        if not line.startswith("|"):
            if out:
                break  # table finished
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) <= column or set(cells[column]) <= set("-: "):
            continue
        cell = re.sub(r"\*\*|`", "", cells[column]).strip()
        cell = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cell)
        if cell and not cell.lower().startswith(("grupo", "pantalla", "vista", "|")):
            out.add(cell)
    return out


@extractor("doc_screen_labels")
def doc_screen_labels(ctx: Context) -> set[str]:
    text = _read(ctx, "docs/ui/1-estructura.md")
    return {c for c in _table_column(text, "## El mapa de pantallas", 1) if c}


@extractor("doc_screen_routes")
def doc_screen_routes(ctx: Context) -> set[str]:
    text = _read(ctx, "docs/ui/1-estructura.md")
    out: set[str] = set()
    for cell in _table_column(text, "## El mapa de pantallas", 2):
        out |= {r for r in re.split(r"[^A-Za-z0-9:/_-]+", cell) if r.startswith("/")}
    return out


@extractor("nav_labels")
def nav_labels(ctx: Context) -> set[str]:
    text = _read(ctx, "web/src/App.tsx")
    return set(re.findall(r"<NavLink to=\"[^\"]+\">([^<]+)</NavLink>", text))


@extractor("app_routes")
def app_routes(ctx: Context) -> set[str]:
    text = _read(ctx, "web/src/App.tsx")
    out = set()
    for path, element in re.findall(r"<Route path=\"([^\"]+)\" element=\{<(\w+)", text):
        if element == "Navigate":
            continue  # a redirect is not a screen
        out.add(path)
    return out


@extractor("verify_ui_routes")
def verify_ui_routes(ctx: Context) -> set[str]:
    text = _read(ctx, "scripts/verify_ui.py")
    return set(re.findall(r"check\(page, \"([^\"]+)\"", text))


@extractor("doc_testids")
def doc_testids(ctx: Context) -> set[str]:
    text = _read(ctx, "docs/ui/7-operacion.md")
    block = text.split("Los vivos hoy:", 1)[-1].split("\n\n", 1)[0]
    return set(re.findall(r"`([a-z0-9-]+)`", block))


@extractor("code_testids")
def code_testids(ctx: Context) -> set[str]:
    out: set[str] = set()
    for p in iter_files(ctx.root, "web/src/**/*.tsx"):
        out |= set(re.findall(r"data-testid=[\"'{]?([a-z0-9-]+)",
                              p.read_text(encoding="utf-8", errors="replace")))
    return out


@extractor("doc_view_ids")
def doc_view_ids(ctx: Context) -> set[str]:
    text = _read(ctx, "docs/ui/2-vistas.md")
    ids: set[str] = set()
    for cell in _table_column(text, "## El catálogo", 0):
        ids |= set(re.findall(r"F0|V\d+|FG\d+", cell))
    ids |= set(re.findall(r"\*\*(FG\d+) —", text))
    return ids


def _route_matches(concrete: str, pattern: str) -> bool:
    a, b = concrete.strip("/").split("/"), pattern.strip("/").split("/")
    if len(a) != len(b):
        return False
    return all(pb.startswith(":") or pa == pb for pa, pb in zip(a, b))


@verb("catalog_match")
def catalog_match(check: Check, ctx: Context) -> Outcome:
    args = check.args
    for side in ("left", "right"):
        name = str(args.get(side))
        if name in PLANNED_EXTRACTORS:
            return Outcome(NA, f"extractor {name} no implementado (fase {PLANNED_EXTRACTORS[name]})")
        if name not in EXTRACTORS:
            return Outcome(VIOLATED, f"extractor desconocido: {args.get(side)}")
    left = EXTRACTORS[str(args["left"])](ctx)
    right = EXTRACTORS[str(args["right"])](ctx)
    if not left or not right:
        empty = args["left"] if not left else args["right"]
        return Outcome(VIOLATED, f"el extractor {empty} no saco nada: configuracion del check")

    if args.get("normalize") == "route":
        right = {next((p for p in left if _route_matches(c, p)), c) for c in right}

    missing_right = sorted(left - right)
    missing_left = sorted(right - left)
    problems = []
    if missing_right:
        problems.append(f"en {args['left']} y no en {args['right']}: {', '.join(missing_right)}")
    if missing_left and args.get("mode") != "covers":
        problems.append(f"en {args['right']} y no en {args['left']}: {', '.join(missing_left)}")
    if problems:
        return Outcome(VIOLATED, "; ".join(problems))
    return Outcome(OK, f"{len(left)} entradas casan")
