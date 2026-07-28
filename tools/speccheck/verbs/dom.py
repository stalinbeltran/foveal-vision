"""Fase 4 -- the live DOM (substrate B5) and the `data-*` declarations (B6).

This is where the three prose-only types stop being prose: a screen declares its
domain, a view declares its (fixes, varies, measures), a heatmap declares its
number twin. The validator checks the declaration EXISTS and MATCHES the
document; whether the declared triple is the RIGHT one stays human (U2.2).

One browser for the whole run: opening Chromium per rule would make `--live`
unusable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb

_PAGE: dict[str, Any] = {}


def browser(ctx: Context):
    """A live page, or (None, reason). Playwright missing -> no_aplicable."""
    if "page" in _PAGE:
        return _PAGE["page"], ""
    if not ctx.front_url:
        return None, "no hay front vivo (vite no arranco o el puerto esta ocupado)"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, "playwright no esta instalado en el venv"
    pw = sync_playwright().start()
    br = pw.chromium.launch()
    page = br.new_page(viewport={"width": 1400, "height": 950})
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    _PAGE.update({"page": page, "browser": br, "pw": pw, "errors": errors})
    return page, ""


def close_browser() -> list[str]:
    errs = _PAGE.get("errors", [])
    if "browser" in _PAGE:
        _PAGE["browser"].close()
        _PAGE["pw"].stop()
    _PAGE.clear()
    return errs


def _visit(ctx: Context, page, route: str, seed: dict | None = None) -> str:
    """Go to a route, optionally seeding localStorage first (U7.4/U5.7)."""
    if seed:
        page.goto(ctx.front_url + "/", wait_until="domcontentloaded")
        page.evaluate("(kv) => { for (const [k, v] of Object.entries(kv)) "
                      "localStorage.setItem('fv.ui.' + k, JSON.stringify(v)); }", seed)
    page.goto(ctx.front_url + route, wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    return route


def _routes(ctx: Context, scope: str) -> list[str]:
    """`*` means every screen the document lists; a concrete path means itself."""
    from .catalogs import EXTRACTORS
    if scope in ("*", ""):
        routes = sorted(EXTRACTORS["app_routes"](ctx))
    else:
        routes = [scope]
    # a route with a :param needs a real name; ask the backend for one
    out = []
    for r in routes:
        if ":" in r:
            from .http import fixtures
            fx = fixtures(ctx)
            name = {"/window-datasets/:name": fx["window_dataset"],
                    "/runs/:name": fx["run"]}.get(r)
            if not name:
                continue
            r = r.rsplit("/", 1)[0] + "/" + str(name)
        out.append(r)
    return out


@verb("dom_query")
def dom_query(check: Check, ctx: Context) -> Outcome:
    page, why = browser(ctx)
    if page is None:
        return Outcome(NA, why)
    a = check.args
    scope = str(check.data.get("scope", "*"))
    routes = _routes(ctx, scope)
    if not routes:
        return Outcome(NA, f"no hay ruta concreta para {scope}")

    bad: list[str] = []
    seen = 0
    for route in routes:
        _visit(ctx, page, route, a.get("seed_localstorage"))
        if a.get("assert_no"):
            if page.locator(str(a["assert_no"])).count():
                bad.append(f"{route}: aparece {a['assert_no']}")
        if a.get("selector"):
            # wait for it, never sleep a fixed time: the heavy screens (a dataset
            # with millions of windows) render well after any constant you pick,
            # and a timeout that expires reads exactly like a missing element.
            if not a.get("optional"):
                try:
                    page.wait_for_selector(str(a["selector"]), timeout=25000)
                except Exception:
                    pass
            loc = page.locator(str(a["selector"]))
            n = loc.count()
            if n == 0:
                if a.get("optional"):
                    continue
                bad.append(f"{route}: sin {a['selector']}")
                continue
            seen += n
            if a.get("min_count") and n < int(a["min_count"]):
                bad.append(f"{route}: {n} < {a['min_count']} de {a['selector']}")
            for attr in (a.get("must_have_attr") or []):
                missing = page.eval_on_selector_all(
                    str(a["selector"]),
                    "(els, at) => els.filter((e) => !e.getAttribute(at)).length", str(attr))
                if missing:
                    bad.append(f"{route}: {missing} nodo(s) {a['selector']} sin {attr}")
            if a.get("sibling_required"):
                # the twin may be collapsed behind a click; the contract is that
                # the element EXISTS in the same card, so click it open first
                loc.first.click()
                page.wait_for_timeout(300)
                if not page.locator(str(a["sibling_required"])).count():
                    bad.append(f"{route}: {a['selector']} sin {a['sibling_required']}")
            if a.get("assert_text_not_empty"):
                empty = page.eval_on_selector_all(
                    str(a["selector"]), "(els) => els.filter((e) => !e.textContent.trim()).length")
                if empty:
                    bad.append(f"{route}: {empty} nodo(s) sin texto")
    if bad:
        return Outcome(VIOLATED, "; ".join(bad[:5]) + (f" (+{len(bad)-5})" if len(bad) > 5 else ""))
    return Outcome(OK, f"{len(routes)} ruta(s), {seen} nodo(s) encontrados")


@verb("dom_absent_text")
def dom_absent_text(check: Check, ctx: Context) -> Outcome:
    """U8.2: the ambiguous word does not reach the screen, qualified uses aside."""
    page, why = browser(ctx)
    if page is None:
        return Outcome(NA, why)
    words = [str(w) for w in (check.args.get("words") or [])]
    allow = [str(w).lower() for w in (check.args.get("allow_qualified") or [])]
    bad: list[str] = []
    routes = _routes(ctx, str(check.data.get("scope", "*")))
    for route in routes:
        _visit(ctx, page, route)
        text = (page.inner_text("body") or "").lower()
        for w in allow:
            text = text.replace(w, " ")
        for w in words:
            if re.search(rf"\b{re.escape(w.lower())}\b", text):
                bad.append(f"{route}: «{w}»")
    if bad:
        return Outcome(VIOLATED, "; ".join(bad[:6]) + (f" (+{len(bad)-6})" if len(bad) > 6 else ""))
    return Outcome(OK, f"{len(routes)} ruta(s) sin las palabras prohibidas")


@verb("color_follows_entity")
def color_follows_entity(check: Check, ctx: Context) -> Outcome:
    """U3.7: hide one series and the survivors keep their colour."""
    page, why = browser(ctx)
    if page is None:
        return Outcome(NA, why)
    route = str(check.data.get("scope", "/sweeps"))
    _visit(ctx, page, route)
    toggle = str(check.args.get("toggle", ""))
    series = "[data-testid=sweep-curves] svg path[stroke]"
    if not page.locator(toggle).count():
        return Outcome(NA, f"{route}: no hay leyenda que conmutar (recorrido sin curvas)")
    before = page.eval_on_selector_all(series, "(els) => els.map((e) => e.getAttribute('stroke'))")
    page.locator(toggle).first.click()
    page.wait_for_timeout(500)
    after = page.eval_on_selector_all(series, "(els) => els.map((e) => e.getAttribute('stroke'))")
    kept = [c for c in after if c in before]
    if len(set(after)) != len(kept):
        return Outcome(VIOLATED, f"al ocultar una serie aparecen colores nuevos: {set(after) - set(before)}")
    return Outcome(OK, f"{len(before)} series -> {len(after)}, sin repintar a las demas")


@verb("number_has_uncertainty")
def number_has_uncertainty(check: Check, ctx: Context) -> Outcome:
    page, why = browser(ctx)
    if page is None:
        return Outcome(NA, why)
    route = str(check.data.get("scope", "*"))
    routes = _routes(ctx, route)
    marks = [str(m) for m in (check.args.get("marks") or ["±", "sem", "sd", "n ="])]
    bad = []
    for r in routes:
        _visit(ctx, page, r)
        for sel in (check.args.get("selectors") or []):
            if not page.locator(str(sel)).count():
                continue
            text = page.locator(str(sel)).first.inner_text()
            if not any(m in text for m in marks):
                bad.append(f"{r}: {sel} sin dispersion ni n")
    if bad:
        return Outcome(VIOLATED, "; ".join(bad))
    return Outcome(OK, "los bloques con numero llevan dispersion y n")
