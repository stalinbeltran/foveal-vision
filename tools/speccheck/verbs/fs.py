"""Verbs over files on disk: the ones that need no parser and no server.

`no_match_outside` with an empty `allow` means "this pattern must appear
nowhere" -- that is how a prohibition is written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb

SKIP_DIRS = {"node_modules", ".git", ".venv", "__pycache__", "data", "runs", "sweeps", "studies"}


def _expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out = []
    for opt in m.group(1).split(","):
        out.extend(_expand_braces(pattern[: m.start()] + opt + pattern[m.end():]))
    return out


def iter_files(root: Path, scope: str) -> Iterator[Path]:
    seen: set[Path] = set()
    for pat in _expand_braces(scope):
        for p in root.glob(pat):
            if not p.is_file() or p in seen:
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
                continue
            seen.add(p)
            yield p


def _empty_scope(check: Check) -> Outcome:
    return Outcome(VIOLATED, f"scope vacio ({check.args.get('scope') or check.data.get('scope')}): "
                             "configuracion del check, no un aprobado")


@verb("file_exists")
def file_exists(check: Check, ctx: Context) -> Outcome:
    path = ctx.root / str(check.args["path"])
    if path.exists():
        return Outcome(OK, f"existe {check.args['path']}")
    return Outcome(VIOLATED, f"no existe {check.args['path']}")


@verb("json_path")
def json_path(check: Check, ctx: Context) -> Outcome:
    path = ctx.root / str(check.args["file"])
    if not path.exists():
        return Outcome(VIOLATED, f"no existe {check.args['file']}")
    data = json.loads(path.read_text(encoding="utf-8"))
    node = data
    for key in str(check.args["path"]).split("."):
        if isinstance(node, list):
            try:
                node = node[int(key)]
            except (ValueError, IndexError):
                return Outcome(VIOLATED, f"{check.args['file']}: falta {check.args['path']}")
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return Outcome(VIOLATED, f"{check.args['file']}: falta {check.args['path']}")
    if "equals" in check.args and node != check.args["equals"]:
        return Outcome(VIOLATED, f"{check.args['path']} = {node!r}, se esperaba {check.args['equals']!r}")
    if "matches" in check.args and not re.search(str(check.args["matches"]), str(node)):
        return Outcome(VIOLATED, f"{check.args['path']} = {node!r} no casa con {check.args['matches']}")
    return Outcome(OK, f"{check.args['path']} presente")


@verb("must_match")
def must_match(check: Check, ctx: Context) -> Outcome:
    scope = str(check.data.get("scope", ""))
    rx = re.compile(str(check.args["pattern"]))
    minimum = int(check.args.get("min", 1))
    files = list(iter_files(ctx.root, scope))
    if not files:
        return _empty_scope(check)
    hits = sum(len(rx.findall(p.read_text(encoding="utf-8", errors="replace"))) for p in files)
    if hits >= minimum:
        return Outcome(OK, f"{hits} coincidencia(s) en {len(files)} fichero(s)")
    return Outcome(VIOLATED, f"{hits} coincidencia(s), se exigian {minimum}")


@verb("no_match_outside")
def no_match_outside(check: Check, ctx: Context) -> Outcome:
    scope = str(check.data.get("scope", ""))
    rx = re.compile(str(check.args["pattern"]))
    allow = {str(a) for a in (check.args.get("allow") or [])}
    files = list(iter_files(ctx.root, scope))
    if not files:
        return _empty_scope(check)
    bad: list[str] = []
    for p in files:
        rel = p.relative_to(ctx.root).as_posix()
        if rel in allow:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                bad.append(f"{rel}:{i}")
    if not bad:
        return Outcome(OK, f"sin coincidencias fuera de lo permitido ({len(files)} ficheros)")
    shown = ", ".join(bad[:6]) + (f" (+{len(bad) - 6})" if len(bad) > 6 else "")
    return Outcome(VIOLATED, f"{len(bad)} coincidencia(s): {shown}")
