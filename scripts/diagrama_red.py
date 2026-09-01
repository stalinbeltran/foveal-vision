#!/usr/bin/env python3
"""Dibuja la ESTRUCTURA de una red, en SVG, leyendola del propio codigo.

Por que se GENERA y no se dibuja a mano
---------------------------------------
Un diagrama hecho a mano es una segunda definicion de la geometria: el dia que
cambie `border_px` o `n_layers`, el codigo cambia y el dibujo se queda mintiendo
--y un diagrama que miente es peor que no tenerlo, porque se cree--. Aqui todas
las cifras salen de `network_trace` y de los parametros reales del modelo, asi
que un cambio de config produce un dibujo distinto sin tocar esto.

Es la misma regla que la UI ya cumple con los derivados (U5.4: se piden, no se
escriben) y la R4 de diseno: el acoplamiento se declara, no se deduce.

⚠ Los COLORES tambien se leen: salen de `web/src/theme/tokens.css`, que es la
unica paleta del proyecto y esta validada para daltonismo. Inventar aqui cuatro
hex seria la segunda definicion de la paleta.

    .venv/bin/python scripts/diagrama_red.py fov16-optimo-mask
    .venv/bin/python scripts/diagrama_red.py fov16-optimo --salida /tmp/otra.svg
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.fovea import build_masks                              # noqa: E402
from fv.models.builder import build_model, full_config, network_trace  # noqa: E402


def paleta() -> dict:
    """Los tokens del tema claro. Una paleta, dos lectores (la app y esto)."""
    css = (ROOT / "web" / "src" / "theme" / "tokens.css").read_text(encoding="utf-8")
    cabeza = css.split("@media", 1)[0]                        # solo el tema claro
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,8})", cabeza))


def _txt(x, y, s, *, tam=13, color="text", ancla="start", peso="normal",
         fam="ui-sans-serif, system-ui, sans-serif", P=None):
    return (f'<text x="{x}" y="{y}" font-size="{tam}" fill="{P[color]}" '
            f'text-anchor="{ancla}" font-weight="{peso}" font-family="{fam}">{s}</text>')


def _caja(x, y, w, h, *, relleno="surface", borde="border", grosor=1.5, r=8, P=None):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{P[relleno]}" stroke="{P[borde]}" stroke-width="{grosor}"/>')


def _flecha(x1, y1, x2, y2, *, color="text-dim", grosor=2, P=None, guion=None):
    d = f' stroke-dasharray="{guion}"' if guion else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{P[color]}" '
            f'stroke-width="{grosor}" marker-end="url(#punta)"{d}/>')


def svg(nombre: str) -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "networks" / f"{nombre}.yaml")
                         .read_text(encoding="utf-8"))
    cfg.pop("format_version", None)
    t = network_trace(cfg)
    d = t["dims"]
    modelo = build_model(full_config(cfg))
    entrenables = sum(p.numel() for p in modelo.parameters())
    cabeza = sum(p.numel() for n, p in modelo.named_parameters() if n.startswith("head"))
    convs = entrenables - cabeza
    cm, pm = build_masks(modelo.dims)
    canales = cfg.get("channels") or []
    L = len(canales)
    con_mascara = t["mask_channel"] != "off"
    n_edge = t["edge_features"]
    P = paleta()
    M = P["mono"] = "ui-monospace, monospace"

    # el alto depende de si hay nota al pie: sin canal de mascara no la hay,
    # y dejar el hueco haria que el dibujo pareciera cortado
    W, H = 1180, (536 if con_mascara else 432)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="ui-sans-serif, system-ui, sans-serif">',
         f'<defs><marker id="punta" viewBox="0 0 10 10" refX="9" refY="5" '
         f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
         f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{P["text-dim"]}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="{P["surface"]}"/>']

    o.append(_txt(28, 38, nombre, tam=22, peso="700", P=P))
    o.append(_txt(28, 60, f'{entrenables:,} pesos entrenables · '
                          f'fovea {d["fovea_px"]} px · borde {d["border_px"]} px '
                          f'(reduce ×{d["border_reduce"]}) · solape {d["overlap_fovea_px"]} px'
                          .replace(",", "."), tam=13, color="text-dim", P=P))

    # ---------------------------------------------------------------- 1. entrada
    x = 28
    o.append(_caja(x, 92, 210, 150, relleno="surface-2", P=P))
    o.append(_txt(x + 12, 116, "1 · lo que se recorta", tam=13, peso="600", P=P))
    o.append(_txt(x + 12, 140, f'recorte {d["original_size"]}×{d["original_size"]} px',
                  tam=12, color="text-dim", P=P))
    # la fovea dentro del margen, a escala
    esc = 90 / d["original_size"]
    bx, by = x + 12, 152
    o.append(f'<rect x="{bx}" y="{by}" width="{90}" height="{90}" '
             f'fill="{P["surface"]}" stroke="{P["text-dim"]}" stroke-width="1.5" '
             f'stroke-dasharray="4 3"/>')
    f0 = d["border_px"] * esc
    o.append(f'<rect x="{bx + f0}" y="{by + f0}" width="{d["fovea_px"] * esc}" '
             f'height="{d["fovea_px"] * esc}" fill="{P["accent"]}" opacity="0.16" '
             f'stroke="{P["accent"]}" stroke-width="1.5"/>')
    o.append(_txt(bx + 108, by + 30, "ventana", tam=11, color="accent", P=P))
    o.append(_txt(bx + 108, by + 45, "etiquetada", tam=11, color="accent", P=P))
    o.append(_txt(bx + 108, by + 62, "margen:", tam=11, color="text-dim", P=P))
    o.append(_txt(bx + 108, by + 76, "contexto", tam=11, color="text-dim", P=P))

    # -------------------------------------------------- 2. la vista + el relleno
    x = 288
    alto = 176 if con_mascara else 120
    o.append(_caja(x, 92, 232, alto, relleno="surface-2", P=P))
    o.append(_txt(x + 12, 116, "2 · muestreo foveado", tam=13, peso="600", P=P))
    o.append(_txt(x + 12, 138, f'vista {d["N"]}×{d["N"]} — centro a resolución',
                  tam=12, color="text-dim", P=P))
    o.append(_txt(x + 12, 154, f'plena, anillo reducido ×{d["border_reduce"]}',
                  tam=12, color="text-dim", P=P))
    o.append(_txt(x + 12, 178, "canal 1: la imagen", tam=12, color="text", P=P))
    if con_mascara:
        o.append(f'<rect x="{x + 12}" y="{192}" width="{208}" height="{64}" rx="6" '
                 f'fill="{P["error"]}" opacity="0.10"/>')
        o.append(_txt(x + 20, 212, "canal 2: EL RELLENO", tam=12, peso="700",
                      color="error", P=P))
        o.append(_txt(x + 20, 230, "1 − cobertura, celda a celda:", tam=11,
                      color="text-dim", P=P))
        o.append(_txt(x + 20, 246, "0 = imagen real · 1 = inventado", tam=11,
                      color="text-dim", P=P))

    # ------------------------------------------------------------- 3. las ramas
    x = 570
    for i, (etq, mask_cells, canales_in, color, y0) in enumerate((
            ("rama CENTRO", int(cm.sum()), 1, "accent", 92),
            ("rama PERIFERIA", int(pm.sum()), 2 if con_mascara else 1, "corner-br", 246))):
        o.append(_caja(x, y0, 268, 140, P=P))
        o.append(_txt(x + 14, y0 + 26, etq, tam=13, peso="600", color=color, P=P))
        o.append(_txt(x + 14, y0 + 46,
                      f'máscara: {mask_cells} de {d["N"] * d["N"]} celdas',
                      tam=11, color="text-dim", P=P))
        entra = (f'entra {canales_in} canal' if canales_in == 1
                 else f'entra {canales_in} canales (imagen + relleno)')
        o.append(_txt(x + 14, y0 + 64, entra, tam=11,
                      color="error" if canales_in == 2 else "text-dim", P=P))
        # las L convoluciones
        cw = 44
        for k in range(L):
            cx = x + 14 + k * (cw + 8)
            o.append(f'<rect x="{cx}" y="{y0 + 76}" width="{cw}" height="{40}" rx="5" '
                     f'fill="{P[color]}" opacity="{0.14 + 0.06 * k}" '
                     f'stroke="{P[color]}" stroke-width="1"/>')
            o.append(_txt(cx + cw / 2, y0 + 94, f'{canales[k]}', tam=12,
                          ancla="middle", peso="600", color=color, P=P))
            o.append(_txt(cx + cw / 2, y0 + 108,
                          f'{cfg.get("k_center" if i == 0 else "k_periph", 3)}×'
                          f'{cfg.get("k_center" if i == 0 else "k_periph", 3)}',
                          tam=9, ancla="middle", color="text-dim", P=P))
        o.append(_txt(x + 14, y0 + 132,
                      f'salida {t["branch_out"]["center" if i == 0 else "periph"][0]}×'
                      f'{t["branch_out"]["center" if i == 0 else "periph"][1]}×{canales[-1]}',
                      tam=11, color="text-dim", P=P))

    # ------------------------------------------------------------- 4. la cabeza
    x = 872
    o.append(_caja(x, 150, 280, 262, relleno="surface-2", P=P))
    o.append(_txt(x + 14, 176, "4 · la cabeza", tam=13, peso="600", P=P))
    o.append(_txt(x + 14, 200, f'concat → {t["flat_features"]:,} features'.replace(",", "."),
                  tam=12, color="text-dim", P=P))
    o.append(_txt(x + 14, 218, f'ReLU · dropout {cfg.get("dropout", 0.0)}',
                  tam=12, color="text-dim", P=P))
    if n_edge:
        o.append(f'<rect x="{x + 14}" y="{230}" width="{252}" height="{46}" rx="6" '
                 f'fill="{P["warn"]}" opacity="0.12"/>')
        o.append(_txt(x + 22, 249, f'+ {n_edge} escalares de BORDE', tam=12,
                      peso="700", color="warn", P=P))
        o.append(_txt(x + 22, 266, f'edge_inputs: {t["edge_inputs"]} — no pasan por',
                      tam=10, color="text-dim", P=P))
        o.append(_txt(x + 22, 278, "ninguna convolución", tam=10, color="text-dim", P=P))
    o.append(_txt(x + 14, 302, f'Linear {t["head_inputs"]:,} → 12'.replace(",", "."),
                  tam=13, peso="600", P=P))
    o.append(_txt(x + 14, 322, f'{cabeza:,} pesos ({100 * cabeza / entrenables:.0f} % del total)'
                  .replace(",", "."), tam=11, color="text-dim", P=P))
    # las 4 esquinas
    for k, (c, tok) in enumerate((("TL", "corner-tl"), ("TR", "corner-tr"),
                                  ("BR", "corner-br"), ("BL", "corner-bl"))):
        cx = x + 16 + k * 66
        o.append(f'<rect x="{cx}" y="{338}" width="{58}" height="{30}" rx="5" '
                 f'fill="{P[tok]}" opacity="0.16" stroke="{P[tok]}" stroke-width="1"/>')
        o.append(_txt(cx + 29, 358, c, tam=12, ancla="middle", peso="700", color=tok, P=P))
    o.append(_txt(x + 14, 398, "4 esquinas × [existe, x, y]", tam=11, color="text-dim", P=P))

    # -------------------------------------------------------------- las flechas
    o.append(_flecha(240, 167, 284, 167, P=P))
    o.append(_flecha(522, 150, 566, 150, P=P))            # vista -> centro
    o.append(_flecha(522, 210, 566, 300, P=P))            # vista -> periferia
    if con_mascara:
        # la que importa: el relleno SOLO a la periferia
        o.append(f'<path d="M 500 224 C 545 224 530 300 566 316" fill="none" '
                 f'stroke="{P["error"]}" stroke-width="2.5" marker-end="url(#punta)"/>')
    o.append(_flecha(842, 160, 868, 200, P=P))
    o.append(_flecha(842, 316, 868, 268, P=P))

    # ------------------------------------------------------------------- la nota
    if con_mascara:
        o.append(f'<rect x="28" y="{H - 110}" width="{W - 56}" height="86" rx="8" '
                 f'fill="{P["error"]}" opacity="0.07"/>')
        o.append(_txt(44, H - 84, "El canal de relleno va SOLO a la periferia, y no es "
                      "una economía", tam=13, peso="700", color="error", P=P))
        o.append(_txt(44, H - 62, f'Medido: bajo la máscara del centro la cobertura es 1,000 '
                      f'en TODAS las celdas, también en la ventana (0,0) — la fóvea está dentro '
                      f'de la imagen por construcción.', tam=12, color="text", P=P))
        o.append(_txt(44, H - 42, f'Dárselo al centro serían 144 pesos equivalentes a un '
                      f'término de sesgo. Coste real: +144 ({convs:,} en las convoluciones)'
                      .replace(",", "."), tam=12, color="text", P=P))
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("red")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    destino = Path(a.salida) if a.salida else ROOT / "docs" / f"red-{a.red}.svg"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(svg(a.red), encoding="utf-8")
    print(f"escrito {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
