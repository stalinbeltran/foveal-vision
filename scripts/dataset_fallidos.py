#!/usr/bin/env python3
r"""Un dataset con los casos en que una red FALLA, para poder usarlos.

Pasa una red entrenada por todas las imagenes de un dataset de ventanas, puntua
cada imagen a nivel de PARRAFO, y escribe un dataset de ventanas NUEVO con las
peores. El resultado es un B de pleno derecho (`windows.npz` + `manifest.json` +
`split.json`): `fv-train` entrena sobre el, la web app lo lista, y nada necesita
saber de donde salio.

    # las tres redes aprobadas, sobre el dataset con el que se entrenaron
    .venv/bin/python scripts/dataset_fallidos.py --verdad ventanas

    # una sola, mirando primero sin escribir nada
    .venv/bin/python scripts/dataset_fallidos.py --nn fov16-mask-p20 --seco

    # solo el holdout de val, y solo las 50 peores, con las imagenes en PNG
    .venv/bin/python scripts/dataset_fallidos.py --nn fov16-edge-p20 \
        --split val --max-imagenes 50 --png

QUE ES UN ERROR (escrito antes de mirar ningun numero)
------------------------------------------------------
Un parrafo que la red no encontro (`fn`) o uno que se invento (`fp`),
emparejando por IoU >= --iou. Peor = mas errores; los empates los rompen el f1
de parrafo, el IoU medio de los emparejados y el indice, en ese orden y solo
para que el resultado sea el mismo cada vez que se repita.

El detalle, el choque declarado con el contrato 13 y las tolerancias medidas
estan en el modulo: `src/fv/fallidos.py`.

Codigos de salida: 0 escrito (o --seco), 1 error, 2 nada que escribir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.fallidos import (Criterio, FallidosError, crear, dataset_de,  # noqa: E402
                         nombre_dataset, ordenar)
from fv.inference.catalogo import aprobadas                           # noqa: E402
from fv.training.registry import RunStore                             # noqa: E402
from fv.windows.store import WindowDatasetStore                       # noqa: E402


def _barra(n: int, total: int) -> None:
    if n == total or n % 100 == 0:
        print(f"\r    {n}/{total} imagenes", end="", flush=True)
        if n == total:
            print()


def _resumen(res: dict) -> None:
    ev, elegidas = res["evaluacion"], res["elegidas"]
    todas = ev["per_image"]
    con_error = [r for r in todas if r["errores"] > 0]
    total_err = sum(r["errores"] for r in todas)
    print(f"  imagenes evaluadas : {len(todas)} (split {ev['split']})")
    print(f"  sin un solo error  : {len(todas) - len(con_error)}")
    print(f"  con al menos uno   : {len(con_error)}   "
          f"errores totales {total_err} "
          f"(fp {sum(r['fp'] for r in todas)} / fn {sum(r['fn'] for r in todas)})")
    if ev["verdad"]["gt_incompleta"]:
        print(f"  ⚠ verdad incompleta: {ev['verdad']['gt_incompleta']} imagen(es) "
              f"con algun parrafo cortado por el borde; fuera de la seleccion "
              f"salvo --incluir-gt-parcial")
    reparto: dict[int, int] = {}
    for r in con_error:
        reparto[r["errores"]] = reparto.get(r["errores"], 0) + 1
    if reparto:
        print("  reparto            : " + "  ".join(
            f"{k} error{'es' if k > 1 else ' '}: {v}" for k, v in sorted(reparto.items())))
    # De quien es la culpa. Sin esto, "esta red falla aqui" manda a reentrenar
    # una red que puede estar perfecta: ver `fv.fallidos.esquinas_acertadas`.
    dg = ev["diagnostico"]
    if dg["imagenes_con_error"]:
        pct = 100.0 * dg["solo_emparejado"] / dg["imagenes_con_error"]
        e = dg["esquinas"]
        print(f"  de quien es         : {dg['solo_emparejado']} de "
              f"{dg['imagenes_con_error']} ({pct:.0f} %) tienen TODAS las esquinas "
              f"bien y fallan solo al EMPAREJARLAS")
        print(f"                        esquinas (tol {dg['tol_esquina_px']:.0f} px): "
              f"tp {e['tp']}  fp {e['fp']}  fn {e['fn']}")
    print(f"  ENTRAN             : {len(elegidas)}")
    for r in ordenar(elegidas)[:5]:
        iou = "sin emparejar" if r["mean_iou"] is None else f"IoU {r['mean_iou']:.3f}"
        culpa = "emparejado" if r["solo_emparejado"] else "esquinas"
        print(f"      img {r['index']:4d} [{r['split']:5s}] "
              f"{r['errores']} error(es)  f1 {r['f1']:.3f}  {iou}  "
              f"({r['n_prediccion']} predichos / {r['n_verdad']} verdaderos)  "
              f"culpa: {culpa}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nn", action="append", metavar="RUN",
                    help="que red (repetible). Por defecto, las aprobadas en "
                         "inferencia.json del repo de datos")
    ap.add_argument("--dataset", help="dataset de ventanas base. Por defecto, "
                                      "aquel con el que se entreno la red")
    ap.add_argument("--nombre", help="nombre del dataset de salida. Por defecto "
                                     "<red acortada>-fallidos. Solo vale con UNA red")
    ap.add_argument("--verdad", choices=("fuente", "ventanas"), default="fuente",
                    help="de donde salen los parrafos verdaderos. 'fuente' es el "
                         "contrato 13 y el defecto; 'ventanas' los recompone del "
                         "propio npz cuando la fuente no esta (degradado: pierde "
                         "los parrafos cortados por el borde)")
    ap.add_argument("--split", choices=("todo", "train", "val", "test"),
                    default="todo", help="que imagenes se evaluan (por defecto todas)")

    g = ap.add_argument_group("criterio de seleccion")
    g.add_argument("--min-errores", type=int, default=1,
                   help="minimo de errores para entrar (por defecto 1)")
    g.add_argument("--max-imagenes", type=int, default=0,
                   help="tope de imagenes, las peores primero (0 = sin tope)")
    g.add_argument("--incluir-gt-parcial", action="store_true",
                   help="incluir tambien las imagenes cuya verdad no se pudo "
                        "reconstruir entera (su cuenta de errores no es creible)")

    g = ap.add_argument_group("knobs de inferencia (F)")
    g.add_argument("--umbral", type=float, default=0.5, dest="threshold")
    g.add_argument("--stride", type=int, default=None)
    g.add_argument("--nms-radio", type=float, default=None, dest="nms_radius")
    g.add_argument("--min-tam", type=float, default=None, dest="min_size")
    g.add_argument("--iou", type=float, default=0.5, dest="iou_threshold")

    g = ap.add_argument_group("el dataset de salida")
    g.add_argument("--split-salida", choices=("conservar", "rehacer", "train"),
                   default="conservar",
                   help="conservar: cada imagen mantiene el split que tenia "
                        "(defecto). rehacer: reparto nuevo. train: todo a train")
    g.add_argument("--val-frac", type=float, default=0.2)
    g.add_argument("--test-frac", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--png", action="store_true",
                   help="ademas, las imagenes elegidas como PNG en imagenes/")
    g.add_argument("--seco", action="store_true",
                   help="mide y ensena el resultado, no escribe nada")
    a = ap.parse_args()

    store, wstore = RunStore(), WindowDatasetStore()
    redes = a.nn or aprobadas()
    if not redes:
        print("no hay ninguna red aprobada en inferencia.json y no se paso --nn",
              file=sys.stderr)
        print("  -> pasa una con --nn <run>, o aprueba una (docs/inferencia.md)",
              file=sys.stderr)
        return 1
    if a.nombre and len(redes) > 1:
        print(f"--nombre es para UNA red y se pidieron {len(redes)}: "
              f"{', '.join(redes)}", file=sys.stderr)
        print("  -> quitalo (cada una toma su nombre corto) o pide una sola",
              file=sys.stderr)
        return 1

    criterio = Criterio(threshold=a.threshold, stride=a.stride,
                        nms_radius=a.nms_radius, min_size=a.min_size,
                        iou_threshold=a.iou_threshold,
                        min_errores=a.min_errores, max_imagenes=a.max_imagenes,
                        incluir_gt_parcial=a.incluir_gt_parcial)

    escritos, fallos, vacios = [], [], []
    for run in redes:
        nombre = a.nombre or nombre_dataset(run)
        try:
            base = a.dataset or dataset_de(run, store)
        except FallidosError as e:
            print(f"\n=== {run}\n  [{e.code}] {e.message}\n  -> {e.hint}",
                  file=sys.stderr)
            fallos.append(run)
            continue
        print(f"\n=== {run}  ->  {nombre}")
        print(f"  base: {base}   verdad: {a.verdad}")
        try:
            res = crear(run, dataset=base, nombre=nombre, criterio=criterio,
                        verdad=a.verdad, split=a.split,
                        split_salida=a.split_salida, val_frac=a.val_frac,
                        test_frac=a.test_frac, seed=a.seed, png=a.png,
                        seco=a.seco, store=store, wstore=wstore,
                        progreso=_barra)
        except FallidosError as e:
            if e.code == "sin_fallos":
                print(f"  {e.message}")
                vacios.append(run)
                continue
            print(f"  [{e.code}] {e.message}\n  -> {e.hint}", file=sys.stderr)
            fallos.append(run)
            continue
        _resumen(res)
        w = res["escrito"]
        if w is None:
            print("  (--seco: no se escribio nada)")
            continue
        print(f"  escrito: {w['destino']}")
        print(f"           {w['imagenes']} imagenes, {w['ventanas']} ventanas, "
              f"split {w['windows_per_split']}"
              + (f", {w['png']} PNG" if w["png"] else ""))
        escritos.append(w)

    if escritos:
        # No se commitea desde aqui, igual que `fv-train` y que `promover`: un
        # script que escribe en el historial de otro repo sin pedirlo es una
        # sorpresa. Pero "lo que no esta empujado, no existe", asi que el comando
        # va literal.
        print("\nfalta empujarlo (lo que no esta empujado, no existe):")
        print("  " + escritos[0]["commit"])
    if fallos:
        return 1
    return 2 if (vacios and not escritos) else 0


if __name__ == "__main__":
    raise SystemExit(main())
