"""Drive every screen with Playwright/Chromium and fail on any console or page
error. Assumes backend on :8010 and vite on :5173 with real data (synth-01,
synth-b16, fov-16, corta/media, fov-run-1/2, rec-d).

Usage: .venv\\Scripts\\python scripts\\verify_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
SHOTS = Path(__file__).resolve().parents[1] / "data" / "ui-shots"
SHOTS.mkdir(parents=True, exist_ok=True)

errors: list[str] = []


def check(page, path: str, ready_selector: str, shot: str, extra=None):
    page.goto(BASE + path)
    page.wait_for_selector(ready_selector, timeout=20000)
    page.wait_for_timeout(600)
    if extra:
        extra(page)
    page.screenshot(path=str(SHOTS / f"{shot}.png"), full_page=True)
    print(f"  OK {path} -> {shot}.png")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        # Fuentes: table + click a row -> viewer with quads
        def sources_extra(page):
            page.click("[data-testid=sources-table] tbody tr:has-text('synth-01')")
            page.wait_for_selector("img[alt^='imagen']", timeout=15000)
            page.wait_for_timeout(400)
        check(page, "/sources", "[data-testid=sources-table]", "01-fuentes", sources_extra)

        # Ventanas: list + detail with the raw window grid
        check(page, "/window-datasets", "[data-testid=wds-table]", "02-ventanas")

        def wdetail_extra(page):
            page.wait_for_selector("[data-testid=window-grid] canvas", timeout=20000)
        check(page, "/window-datasets/synth-b16", "[data-testid=window-grid]",
              "03-ventanas-detalle", wdetail_extra)

        # Redes: live validation panel + a broken assert shows its reason
        def networks_extra(page):
            page.wait_for_selector("[data-testid=zone-diagram]", timeout=15000)
            pen = page.locator("label.field", has_text="pen_frac").locator("input")
            pen.fill("0.45")
            page.wait_for_selector("text=penetration_too_large", timeout=10000)
            page.screenshot(path=str(SHOTS / "04b-redes-assert.png"), full_page=True)
            pen.fill("0.1")
            page.wait_for_selector("[data-testid=zone-diagram]", timeout=10000)
        check(page, "/networks", "[data-testid=validate-panel]", "04-redes", networks_extra)

        # Recetas
        check(page, "/recipes", "[data-testid=recipes-table]", "05-recetas")

        # Entrenar: the (1) compatibility line must be visible
        def train_extra(page):
            page.wait_for_selector("[data-testid=compat]", timeout=15000)
        check(page, "/train", "text=nombre del run", "06-entrenar", train_extra)

        # Recorridos: table + select rec-d -> trials ranking; (9) block live
        def sweeps_extra(page):
            page.click("[data-testid=sweeps-table] tbody tr:has-text('rec-d')")
            page.wait_for_selector("[data-testid=trials-table]", timeout=15000)
            page.fill("input[placeholder='0.5, 1.0']", "0.5, 1.0")
            objetivo = page.locator("label.field", has_text="objetivo").locator("select")
            objetivo.select_option("loss")
            page.wait_for_selector("[data-testid=nine-block]", timeout=10000)
            page.screenshot(path=str(SHOTS / "07b-recorridos-contrato9.png"), full_page=True)
            objetivo.select_option("f1")
        check(page, "/sweeps", "[data-testid=sweeps-table]", "07-recorridos", sweeps_extra)

        # Runs: list + detail with curves and provenance
        check(page, "/runs", "[data-testid=runs-table]", "08-runs")

        def run_extra(page):
            page.wait_for_selector("text=Procedencia", timeout=15000)
            page.wait_for_selector("svg[aria-label='loss']", timeout=15000)
        check(page, "/runs/fov-run-2", "text=fov-run-2", "09-run-detalle", run_extra)

        # Diagnostico: summary + gallery -> click opens the probes (F0, V1, V2)
        def diag_extra(page):
            page.wait_for_selector("[data-testid=diag-summary]", timeout=30000)
            page.wait_for_selector("[data-testid=gallery] canvas", timeout=30000)
            page.click("[data-testid=gallery] .thumb")
            page.wait_for_selector("[data-testid=probes]", timeout=30000)
            page.wait_for_timeout(800)
        check(page, "/diagnostics", "text=Diagnóstico", "10-diagnostico", diag_extra)

        # Predecir: stage with overlays + move the threshold slider live
        def predict_extra(page):
            page.wait_for_selector("[data-testid=predict-stage]", timeout=30000)
            page.wait_for_selector("[data-testid=predict-numbers]", timeout=15000)
            sliders = page.locator("input[type=range]")
            sliders.nth(1).fill("0.35")
            page.wait_for_timeout(1500)
        check(page, "/predict", "text=Predecir", "11-predecir", predict_extra)

        browser.close()

    real = [e for e in errors if "favicon" not in e]
    if real:
        print("\nERRORES DE CONSOLA/PAGINA:")
        for e in real:
            print(" ", e)
        return 1
    print(f"\nTODO OK: 11 pantallas/interacciones sin errores. Capturas en {SHOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
