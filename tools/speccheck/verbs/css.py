"""Fase 1 -- the palette (substrate B2).

Not a second implementation: every number here comes from
`web/scripts/palette.mjs --json`, the SAME script `npm run validate:palette` runs.
This module only maps its report onto rules and applies the policy the spec
declares in the check block (whether a WARN counts, which relief covers it).

If node is missing the state is `no_aplicable` -- never `ok`.
"""

from __future__ import annotations

import json
import subprocess

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb

_CACHE: dict[str, dict] = {}


def palette_report(ctx: Context) -> tuple[dict | None, str]:
    key = str(ctx.root)
    if key in _CACHE:
        return _CACHE[key], ""
    try:
        proc = subprocess.run(
            ["node", "scripts/palette.mjs", "--json"],
            cwd=ctx.root / "web", capture_output=True, text=True, timeout=120, shell=True,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"no se pudo ejecutar node: {exc}"
    if not proc.stdout.strip():
        return None, f"palette.mjs no devolvio JSON: {proc.stderr.strip()[:200]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"salida de palette.mjs ilegible: {exc}"
    _CACHE[key] = data
    return data, ""


def _find(report: list[dict], name: str) -> dict | None:
    return next((r for r in report if r["check"] == name), None)


@verb("css_tokens")
def css_tokens(check: Check, ctx: Context) -> Outcome:
    data, err = palette_report(ctx)
    if data is None:
        return Outcome(NA, err)
    themes = data["tokens"]
    problems: list[str] = []

    for name in check.args.get("required") or []:
        for mode in ("light", "dark") if check.args.get("both_themes") else ("light",):
            if name not in themes[mode]:
                problems.append(f"falta {name} en {mode}")

    if check.args.get("theme_parity"):
        # `dark` is built as light-overridden-by-dark, so a token present in light
        # is always present in dark; what parity means here is that every token the
        # dark block declares exists in light too, and that dark actually overrides
        # the ones that carry colour.
        light, dark = themes["light"], themes["dark"]
        missing = sorted(set(dark) - set(light))
        if missing:
            problems.append(f"solo en oscuro: {', '.join(missing)}")
        same = sorted(k for k in light if k.startswith(("--series-", "--corner-", "--div-"))
                      and light[k] == dark[k])
        if same and check.args.get("require_override"):
            problems.append(f"sin valor propio en oscuro: {', '.join(same)}")

    if problems:
        return Outcome(VIOLATED, "; ".join(problems))
    n = len(themes["light"])
    return Outcome(OK, f"{n} tokens, claro y oscuro coherentes")


def _band_outcome(check: Check, entries: list[tuple[str, dict]]) -> Outcome:
    """Map PASS/WARN/FAIL onto a state using the policy declared in the block."""
    warn_is = str(check.args.get("warn_is", "violada"))
    bad = [f"{mode}: {e['detail']}" for mode, e in entries if e["state"] == "FAIL"]
    warn = [f"{mode}: {e['detail']}" for mode, e in entries if e["state"] == "WARN"]
    if bad:
        return Outcome(VIOLATED, "; ".join(bad))
    if warn and warn_is != "ok":
        return Outcome(VIOLATED, "; ".join(warn))
    if warn:
        relief = check.args.get("relief", "")
        return Outcome(OK, f"WARN admitido por {relief}: {'; '.join(w[:110] for w in warn)}")
    return Outcome(OK, "; ".join(f"{mode}: {e['detail'][:80]}" for mode, e in entries))


@verb("palette_cvd_delta_e")
def palette_cvd_delta_e(check: Check, ctx: Context) -> Outcome:
    data, err = palette_report(ctx)
    if data is None:
        return Outcome(NA, err)
    entries = []
    for mode, m in data["modes"].items():
        for name in ("CVD separation", "Normal-vision floor"):
            e = _find(m["report"], name)
            if e:
                entries.append((mode, e))
    if not entries:
        return Outcome(VIOLATED, "el informe no trae los checks de CVD: configuracion")
    return _band_outcome(check, entries)


@verb("palette_contrast")
def palette_contrast(check: Check, ctx: Context) -> Outcome:
    data, err = palette_report(ctx)
    if data is None:
        return Outcome(NA, err)
    entries = []
    for mode, m in data["modes"].items():
        e = _find(m["report"], "Contrast vs surface")
        if e:
            entries.append((mode, e))
    out = _band_outcome(check, entries)

    # The ink is a separate question from the marks and it is NOT dismissable:
    # secondary encoding does not make unreadable text readable.
    ink_min = float(check.args.get("ink_min", 4.5))
    bad_ink = [f"{mode} {name} {ratio:.2f}:1 < {ink_min}"
               for mode, m in data["modes"].items()
               for name, ratio in m["ink"].items() if ratio < ink_min]
    if bad_ink:
        return Outcome(VIOLATED, "; ".join(bad_ink))
    return Outcome(out.state, out.detail + f" | tinta >= {ink_min}:1 en ambos temas")
