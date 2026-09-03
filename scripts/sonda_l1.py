#!/usr/bin/env python3
r"""Sonda L1: ¿pueden los kernels de la primera capa aprender filtros genéricos?

Un autoencoder convolucional de UNA capa por lado sobre la MISMA vista 20x20 que
ve la red de producción. El modelo *son* los kernels: no hay nada detrás que
pueda arreglar un código malo, así que la presión cae entera sobre L1 -- que es
justo lo que no ocurre en `fov16-optimo-mask`, donde detrás hay una cabeza de
153.660 parámetros.

    .venv/bin/python scripts/sonda_l1.py --cronometrar          # UNA combinación
    .venv/bin/python scripts/sonda_l1.py --rejilla              # las 48
    .venv/bin/python scripts/sonda_l1.py --repetir-mejores 3 --semillas 1,2,3
    .venv/bin/python scripts/sonda_l1.py --tabla                # rehace la tabla
    .venv/bin/python scripts/sonda_l1.py --figuras              # rehace las figuras

⚠ NO lances la rejilla sin haber leído `docs/plan-sonda-l1-2026-09-02.md`: los
umbrales de éxito/fracaso se escriben ANTES de mirar, y ese documento es donde
están. El script no los comprueba por ti a propósito -- quien decide qué cuenta
como "gana" tiene que hacerlo sin ver el resultado.

ESTE FICHERO ES SÓLO LA CLI. La sonda vive en `src/fv/probe/`, que es lo que
pedía el encargo (§6, "módulo aislado") y lo que evita que `scripts/` vuelva a
adelantar a `src/`. Aquí sólo se lee la línea de comandos, se cablea la
geometría de producción y se imprime.

LA ÚNICA COSA QUE ESTE FICHERO DECIDE, Y POR QUÉ ESTÁ AQUÍ
----------------------------------------------------------
La geometría (`NETWORK_DEFAULTS`) se lee aquí y se pasa al módulo como un dict.
Así `fv.probe` NO importa `fv.models` -- que es la regla de aislamiento del
encargo-- y a la vez la geometría sigue teniendo una sola definición, en
`builder.py`. Duplicarla aquí sería el fallo que el aislamiento pretende evitar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch                                    # noqa: E402

from fv import settings                         # noqa: E402
from fv.models.builder import NETWORK_DEFAULTS  # noqa: E402  (aquí, no en fv.probe)
from fv.probe import (GRID_CHANNELS, GRID_KS, GRID_KS_ANCHOR, GRID_LAMBDAS,  # noqa: E402
                      L1Probe, calibrate_lambda, code_maps, comparison_table,
                      contact_sheet, prepare, run_name, train)

DATASET = "dirty1000-80px-16px-r20260827"


def _combos(con_k3: bool) -> list[tuple[int, int, float]]:
    ks = (GRID_KS_ANCHOR if con_k3 else []) + GRID_KS
    return [(K, k, l) for k in ks for K in GRID_CHANNELS for l in GRID_LAMBDAS]


def _leer_runs(salida: Path) -> list[dict]:
    filas = []
    for d in sorted(salida.glob("*/summary.json")):
        try:
            filas.append(json.loads(d.read_text()))
        except json.JSONDecodeError as e:
            print(f"  ⚠ {d}: JSON roto ({e}); se salta")
    return filas


def _tabla(salida: Path, filas: list[dict]) -> None:
    if not filas:
        print("no hay ningún run con summary.json todavía")
        return
    md = comparison_table(filas, salida / "tabla.md", salida / "tabla.csv")
    print(md)
    print(f"\ntabla en {salida/'tabla.md'} y {salida/'tabla.csv'}")


def _hoja(salida: Path, fila: dict) -> None:
    d = salida / fila["nombre"]
    f = d / "kernels_enc.npy"
    if not f.exists():
        return
    contact_sheet(
        np.load(f), d / "kernels_enc.png",
        f"{fila['nombre']}  ·  codificador",
        f"Gabor D {fila.get('gabor_delta', float('nan')):.3f} "
        f"(R2 {fila.get('gabor_r2', float('nan')):.3f} vs nulo {fila.get('gabor_r2_base', float('nan')):.3f})  ·  "
        f"enriq {fila.get('enriquecimiento', float('nan')):.2f}x  ·  "
        f"activa {fila.get('frac_activa', float('nan'))*100:.1f}%  ·  "
        f"muertos {fila.get('kernels_muertos', 0)}  ·  "
        f"R2 rec int {fila.get('r2_rec_int', float('nan')):.3f}")
    dec = d / "kernels_dec.npy"
    if dec.exists():
        contact_sheet(np.load(dec), d / "kernels_dec.png",
                      f"{fila['nombre']}  ·  decodificador",
                      f"alineación con el codificador: "
                      f"{fila.get('align_enc_dec', float('nan')):.2f} de media, "
                      f"{fila.get('align_enc_dec_min', float('nan')):.2f} la peor")


def _mapas_z(salida: Path, fila: dict, val: np.ndarray, cuantas: int = 1) -> None:
    d = salida / fila["nombre"]
    ck = d / "checkpoint.pt"
    if not ck.exists():
        print(f"  ⚠ {fila['nombre']}: sin checkpoint.pt, no puedo pintar los mapas z")
        return
    est = torch.load(ck, map_location="cpu", weights_only=True)
    m = L1Probe(est["K"], est["k"])
    m.load_state_dict(est["state_dict"])
    m.eval()
    paso = max(1, val.shape[0] // (cuantas + 1))
    for j in range(cuantas):
        x = torch.from_numpy(val[j * paso:j * paso + 1])
        with torch.no_grad():
            _, z = m(x)
        code_maps(x[0, 0].numpy(), z[0].numpy(), d / f"mapas_z_{j}.png",
                  f"{fila['nombre']}  ·  la entrada y sus {est['K']} mapas z",
                  "¿es la imagen resultante más genérica? "
                  f"(Gabor D {fila.get('gabor_delta', float('nan')):.3f})")


def main() -> int:
    p = argparse.ArgumentParser(description="Sonda L1: ¿aprende L1 filtros genéricos?")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--lote", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=None,
                   help="por defecto se MIDE: mediana de la sd local del train")
    p.add_argument("--limite", type=int, default=None,
                   help="submuestrea el train a N ventanas (de 84.000)")
    p.add_argument("--val-log", type=int, default=4096,
                   help="ventanas de val para la curva por época (las métricas "
                        "finales usan SIEMPRE el split entero)")
    p.add_argument("--semillas", default="1")
    p.add_argument("--solo", default=None, help="una combinación: k9-K32-l0.1")
    p.add_argument("--con-k3", dest="k3", action="store_true", default=True,
                   help="incluye el ancla k=3 (por defecto sí)")
    p.add_argument("--sin-k3", dest="k3", action="store_false")
    p.add_argument("--cronometrar", action="store_true",
                   help="mide UNA combinación y estima la rejilla entera. No la lanza")
    p.add_argument("--rejilla", action="store_true", help="lanza la rejilla completa")
    p.add_argument("--repetir-mejores", type=int, default=0, metavar="N",
                   help="relanza las N mejores (por Gabor D) con --semillas")
    p.add_argument("--tanteo-k", action="store_true",
                   help="el eje k a K fijo: k in {3,5,7,9} x lambda in {0, calibrada}. "
                        "8 runs, ~1,7 h. Es el eje que lleva la premisa del encargo")
    p.add_argument("--canales", type=int, default=16,
                   help="el K del --tanteo-k (por defecto 16)")
    p.add_argument("--calibrar", action="store_true",
                   help="lambda se CALIBRA por celda hasta la banda de activacion, "
                        "en vez de tomarse de la rejilla")
    p.add_argument("--objetivo-activa", type=float, default=0.10)
    p.add_argument("--tolerancia-activa", type=float, default=0.03)
    p.add_argument("--limite-calibrar", type=int, default=8000,
                   help="ventanas para la biseccion (los runs usan el train entero)")
    p.add_argument("--tabla", action="store_true",
                   help="rehace tabla.md/tabla.csv desde los runs en disco")
    p.add_argument("--figuras", action="store_true",
                   help="rehace las figuras desde los runs en disco")
    p.add_argument("--salida", default=None)
    a = p.parse_args()

    cache = Path(os.environ.get("FV_SONDA_CACHE", "/tmp/sonda-l1-cache"))
    salida = Path(a.salida) if a.salida else settings.data_root() / "sondas" / "l1"
    combos = _combos(a.k3)
    if a.solo:
        combos = [c for c in combos if run_name(c[0], c[1], c[2]) == a.solo]
        if not combos:
            raise SystemExit(f"'{a.solo}' no está en la rejilla")
    semillas = [int(s) for s in a.semillas.split(",")]

    # --- lo que NO entrena: se contesta sin tocar los datos pesados
    hace_runs = a.rejilla or a.solo or a.repetir_mejores or a.cronometrar or a.tanteo_k
    if a.tabla and not hace_runs:
        _tabla(salida, _leer_runs(salida))
        return 0

    datos = prepare(a.dataset, dict(NETWORK_DEFAULTS), a.sigma, a.eps, cache, a.limite)

    if a.figuras and not hace_runs:
        filas = _leer_runs(salida)
        for f in filas:
            _hoja(salida, f)
        for f in sorted(filas, key=lambda r: -r.get("gabor_delta", -9))[:3]:
            _mapas_z(salida, f, datos["val"])
        print(f"{len(filas)} hoja(s) de contactos y hasta 3 figuras de mapas z en {salida}")
        return 0

    if a.cronometrar:
        K, k, lam = 32, 9, 0.1          # la más cara de la rejilla
        print(f"\n[cronometrar] la combinación MÁS CARA: k={k} K={K} λ={lam}, "
              f"2 épocas para extrapolar")
        r = train(datos, K, k, lam, 1, 2, a.lote, a.lr, out_dir=None,
                  val_log=a.val_log)
        seg_ep = r["segundos"] / 2
        coste = sum(KK * kk * kk for KK, kk, _ in combos)
        total = coste * (seg_ep / (K * k * k)) * a.epocas
        print(f"\n  {seg_ep:.1f} s/época  ->  {seg_ep*a.epocas/60:.1f} min este run")
        print(f"  rejilla de {len(combos)} runs x {a.epocas} épocas  ->  "
              f"~{total/3600:.1f} h en esta máquina, extrapolado por K·k²")
        print(f"  (+ repetir las 3 mejores con 3 semillas: "
              f"~{total/len(combos)*6/3600:.1f} h más)")
        r.pop("_curva", None)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    if a.tanteo_k:
        # Punto 3 de la revision del dueno: el tanteo cubria UN punto del eje k,
        # y el eje k es la premisa entera del experimento. Ocho runs antes de las
        # 12-15 h de la rejilla.
        combos = [(a.canales, k, 0.0) for k in (GRID_KS_ANCHOR + GRID_KS)]
        a.calibrar = True

    if a.repetir_mejores:
        previos = _leer_runs(salida)
        if not previos:
            raise SystemExit(f"no hay runs en {salida}: lanza --rejilla primero")
        # Se agrupa por combinación antes de rankear: si ya hay varias semillas,
        # la mejor es la de mejor MEDIA, no la de la semilla más afortunada.
        por: dict[str, list[dict]] = {}
        for r in previos:
            por.setdefault(run_name(r["K"], r["k"], r["lambda"]), []).append(r)
        rank = sorted(por.items(),
                      key=lambda kv: -sum(x.get("gabor_delta", -9) for x in kv[1]) / len(kv[1]))
        elegidas = [n for n, _ in rank[:a.repetir_mejores]]
        print(f"[mejores] por Gabor D: {', '.join(elegidas)}")
        combos = [c for c in _combos(True) if run_name(c[0], c[1], c[2]) in elegidas]
    elif not (a.rejilla or a.solo or a.tanteo_k):
        p.error("elige --cronometrar, --tanteo-k, --rejilla, --solo, "
                "--repetir-mejores, --tabla o --figuras")

    salida.mkdir(parents=True, exist_ok=True)
    # La bisección corre sobre un subconjunto: son 2 épocas x 5-6 evaluaciones y
    # lo que se busca es el ORDEN de λ, no su cuarta cifra. Los runs de verdad
    # usan el train entero -- `--limite` es una variable de confusión JUSTO sobre
    # la métrica principal (menos ventanas -> kernels más ruidosos -> el ajuste
    # Gabor baja), así que sesga hacia el fracaso.
    dcal = datos
    if a.calibrar and a.limite_calibrar < datos["train"].shape[0]:
        gsub = torch.Generator().manual_seed(0)
        sub = torch.randperm(datos["train"].shape[0], generator=gsub)[:a.limite_calibrar]
        dcal = dict(datos, train=datos["train"][sub.numpy()])

    t0 = time.time()
    for idx, (K, k, lam) in enumerate(combos, 1):
        extra = {}
        if a.calibrar:
            print(f"\n[{idx}/{len(combos)}] calibrando λ para k={k} K={K}...")
            cal = calibrate_lambda(dcal, K, k, seed=semillas[0],
                                   objetivo=a.objetivo_activa,
                                   tolerancia=a.tolerancia_activa,
                                   val_log=a.val_log)
            extra = {"calibracion": cal}
            print(f"    → λ={cal['lambda']:.4g}  activa {cal['activa_calibrada']*100:.1f} %"
                  f"  en_banda={cal['en_banda']}  saturado={cal['saturado']}")
            if not cal["en_banda"]:
                print(f"    ⚠ esta celda NO alcanza la banda "
                      f"{a.objetivo_activa*100:.0f}±{a.tolerancia_activa*100:.0f} %: "
                      f"se corre igual y queda anotado en summary.json")
        for s in semillas:
            for lam_run, sufijo in ([(lam, "")] if not a.calibrar
                                    else [(0.0, "-l0"), (cal["lambda"], "-lcal")]):
                n = f"k{k}-K{K}{sufijo or f'-l{lam_run}'}-s{s}"
                if (salida / n / "summary.json").exists():
                    print(f"[{idx}/{len(combos)}] {n}: ya está, se salta")
                    continue
                print(f"[{idx}/{len(combos)}] {n}  ({K*k*k*2+K} parámetros)")
                r = train(datos, K, k, lam_run, s, a.epocas, a.lote, a.lr,
                          out_dir=salida, val_log=a.val_log, name=n,
                          extra=extra if sufijo == "-lcal" else {})
                r.pop("_curva", None)
                print(f"    Gabor Δ/margen {r['gabor_delta_rel']:+.3f} "
                      f"({'SUPERA' if r['gabor_supera_p95'] else 'no supera'} el p95)  "
                      f"orient {r['conc_orient_delta']:+.3f}"
                      f"{'*' if r['conc_orient_supera_p95'] else ' '}  "
                      f"banda {r['conc_banda_delta']:+.3f}"
                      f"{'*' if r['conc_banda_supera_p95'] else ' '}  "
                      f"enriq {r['enriquecimiento']:.2f}x  "
                      f"R2rec_int {r['r2_rec_int']:.3f}  "
                      f"activa {r['frac_activa']*100:.1f}%")
                _hoja(salida, r)

    filas = _leer_runs(salida)
    (salida / "resumen.json").write_text(json.dumps(
        {"dataset": a.dataset, "epocas": a.epocas, "eps": datos["eps"],
         "var": datos["var"], "limite": a.limite,
         "horas": round((time.time() - t0) / 3600, 2), "runs": filas},
        indent=2, ensure_ascii=False))
    _tabla(salida, filas)
    for f in sorted(filas, key=lambda r: -r.get("gabor_delta", -9))[:3]:
        _mapas_z(salida, f, datos["val"])
    print(f"\nresultados en {salida}")
    print("⚠ el veredicto se lee contra docs/plan-sonda-l1-2026-09-02.md, "
          "que se escribió ANTES de mirar esto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
