#!/usr/bin/env python3
r"""Crea los datasets y los recorridos del barrido de stride de EXTRACCION.

Que es esto
-----------
`docs/plan-stride-2026-08-27.md` fija QUE se mide y COMO se lee, antes de mirar
nada; `docs/barrido-stride.md` explica el mecanismo. Este script solo lo traduce
a `data/window-datasets/` y `sweeps/`, como `estudio_cierre.py` hace con el suyo.
Aqui no se alquila ni se entrena nada: lo que gasta es `estudio_flota.py`.

    .venv/bin/python scripts/estudio_stride.py
    .venv/bin/python scripts/estudio_stride.py --strides 8,16 --semillas 2
    .venv/bin/python scripts/estudio_stride.py --solo-datasets

LAS CUATRO COSAS QUE HAY QUE RESPETAR SI SE TOCA ESTO
-----------------------------------------------------
1. **La rejilla de evaluacion es FIJA y la misma en todos los brazos.** Es la
   razon de ser del estudio: si val/test siguieran al stride de train, el brazo
   de stride 1 se examinaria de 2925 ventanas por imagen y el de 16 de 20, y
   comparar esos dos f1 seria comparar dos EXAMENES. Por eso `eval_stride` es un
   solo valor para todos, y por eso este script se NIEGA si encuentra un dataset
   del estudio extraido con otro (ver `_comprobar_dataset`).

2. **El `seed` de B es el mismo en todos los brazos.** `_assign_splits` solo
   depende de `(n_imagenes, val_frac, test_frac, seed)`, asi que con el mismo
   seed los cinco datasets reparten LAS MISMAS imagenes a train/val/test.
   Cambiarlo mediria el ruido del split (glosario.md, entrada `seed`).

3. **La red y la receta se HEREDAN, no se re-derivan.** Se copian de un
   recorrido que ya existe (`--hereda-de`, por defecto `ov-r26`) para que este
   estudio mida la red vigente y no una parecida. Es el mismo motivo por el que
   `estudio_cierre._hereda` existe.

4. **Los datasets se extraen de stride GRANDE a stride PEQUENO.** El de stride 1
   son 2.925.000 ventanas y es el unico que puede quedarse sin memoria; hacerlo
   el ultimo significa que cualquier fallo de configuracion aparece en el barato,
   a los segundos, y no despues de la parte cara.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.datasets.loader import SourceDataset       # noqa: E402
from fv.sweeps.runner import prepare_sweep          # noqa: E402
from fv.sweeps.spec import SweepError               # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402
from fv.windows.extract import ExtractConfig, extract_windows  # noqa: E402
from fv.windows.store import WindowDatasetStore     # noqa: E402

ESTUDIO = "stride-2026-08-27"
STRIDES = [1, 2, 4, 8, 16]
EVAL_STRIDE = 5
SEMILLAS = 5
VENTANAS_POR_EPOCA = 84_000
EPOCAS = 150
FUENTE = "local/dirty-1000-80px"
BASE_DATASET = "dirty1000-80px-16px"
HEREDA_DE = "ov-r26"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    raise SystemExit(2)


def nombre_dataset(stride: int) -> str:
    return f"{BASE_DATASET}-st{stride:02d}"


def nombre_sweep(stride: int, humo: bool = False) -> str:
    return f"stride-{'h' if humo else ''}{stride:02d}"


def _comprobar_dataset(m: dict, cfg: ExtractConfig, nombre: str) -> None:
    """Un dataset que ya existe se REUSA solo si es el que se pidio.

    Se falla en vez de avisar. Un brazo extraido con otra rejilla de evaluacion,
    otra semilla de split u otra fuente no da error al entrenar: da NUMEROS, y la
    tabla sale igual de creible midiendo otra cosa.
    """
    tiene = m.get("config") or {}
    esperado = {"source": cfg.source, "window_size": cfg.window_size,
                "stride": cfg.stride, "eval_stride": cfg.eval_stride,
                "val_frac": cfg.val_frac, "test_frac": cfg.test_frac,
                "seed": cfg.seed}
    difieren = {k: (tiene.get(k), v) for k, v in esperado.items()
                if tiene.get(k) != v}
    if difieren:
        detalle = "; ".join(f"{k}: en disco {a!r}, se pide {b!r}"
                            for k, (a, b) in sorted(difieren.items()))
        die(f"'{nombre}' ya existe pero NO es el que pide este estudio.\n"
            f"  {detalle}\n"
            f"  Borralo y vuelve a extraerlo, o usa otro nombre. No se reusa un\n"
            f"  dataset que no es el pedido: entrenaria sin fallar sobre otro dato.")


def construir_dataset(stride: int, args, wstore: WindowDatasetStore) -> dict:
    nombre = nombre_dataset(stride)
    cfg = ExtractConfig(source=args.fuente, window_size=args.window_size,
                        stride=stride, val_frac=args.val_frac,
                        test_frac=args.test_frac, seed=args.seed_split,
                        eval_stride=args.eval_stride)
    destino = wstore.path(nombre)
    if destino.exists():
        m = wstore.manifest(nombre)
        _comprobar_dataset(m, cfg, nombre)
        npz = destino / "windows.npz"
        log(f"  {nombre}: ya esta ({m['num_windows']} ventanas, "
            f"{npz.stat().st_size / 1e6:.1f} MB). No se rehace.")
        return m
    t0 = time.monotonic()
    m = extract_windows(cfg, destino)
    secs = time.monotonic() - t0
    npz = destino / "windows.npz"
    log(f"  {nombre}: {m['num_windows']} ventanas "
        f"(train {m['windows_per_split']['train']}, "
        f"val {m['windows_per_split']['val']}, "
        f"test {m['windows_per_split']['test']}) · "
        f"{npz.stat().st_size / 1e6:.1f} MB · {secs / 60:.1f} min")
    return m


def construir_sweep(stride: int, args, store: SweepStore, padre: dict) -> dict:
    nombre = nombre_sweep(stride, args.humo)
    estudio = f"{ESTUDIO}-humo" if args.humo else ESTUDIO
    if store.exists(nombre) and not args.rehacer:
        log(f"  {nombre}: ya existe. No se rehace (--rehacer para forzar).")
        return store.spec(nombre)

    receta = dict(padre["base_recipe_value"])
    receta["windows_per_epoch"] = args.windows_per_epoch
    s0 = int(receta.get("seed", 1))
    base = padre["base_network_value"]

    spec = {
        "window_dataset": nombre_dataset(stride),
        "base_network": None,
        "base_label": padre["base_label"],
        "base_network_value": base,
        "base_recipe": padre["base_recipe"],
        "base_recipe_value": receta,
        "derivation": padre.get("derivation", {}),
        "corrections": padre.get("corrections", []),
        "hereda_base_de": args.hereda_de,
        "hereda_base_medida_sobre": padre["window_dataset"],
        # El eje de este estudio NO vive en `space`: vive en el dataset. Esto es
        # una ETIQUETA DE PROCEDENCIA para que el comparador sepa que valor
        # representa cada recorrido -- no un eje. `expand_points` no lo ve.
        # Ver docs/barrido-stride.md 1.
        "eje_dataset": {"campo": "stride", "valor": stride,
                        "eval_stride": args.eval_stride,
                        "estudio": estudio},
        "space": {"seed": [s0 + k for k in range(args.semillas)]},
        "strategy": "grid",
        "objective": "f1",
        "budget": {"epochs": args.epocas},
        "device": "cpu",
        "seed": s0,
        "seeds": args.semillas,
        "study": estudio,
    }
    enriquecido = prepare_sweep(nombre, spec, base, store)
    log(f"  {nombre}: {len(enriquecido['points'])} puntos "
        f"(semillas {s0}..{s0 + args.semillas - 1}) sobre "
        f"{spec['window_dataset']}")
    return enriquecido


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fuente", default=FUENTE)
    ap.add_argument("--strides", default=",".join(str(s) for s in STRIDES),
                    help="valores del eje, separados por comas")
    ap.add_argument("--eval-stride", type=int, default=EVAL_STRIDE,
                    help="la rejilla FIJA de val/test, igual en todos los brazos")
    ap.add_argument("--semillas", type=int, default=SEMILLAS)
    ap.add_argument("--epocas", type=int, default=EPOCAS)
    ap.add_argument("--windows-per-epoch", type=int, default=VENTANAS_POR_EPOCA,
                    help="presupuesto igualado: ventanas de train por epoca en "
                         "TODOS los brazos. 0 = el pool entero (y entonces el "
                         "barrido mide el presupuesto, no la densidad)")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed-split", type=int, default=1,
                    help="el seed de B: el MISMO en todos los brazos")
    ap.add_argument("--hereda-de", default=HEREDA_DE,
                    help="recorrido del que se copian red base y receta base")
    ap.add_argument("--humo", action="store_true",
                    help="recorridos de VALIDACION: nombres `stride-hNN`, estudio "
                         "aparte y los MISMOS datasets. Sirve para probar la cadena "
                         "entera en 2 maquinas antes de gastar en 25, sin que esos "
                         "runs cuenten como puntos del estudio de verdad (si "
                         "contaran, la flota los daria por hechos y el estudio "
                         "quedaria medido con 3 epocas)")
    ap.add_argument("--solo-datasets", action="store_true")
    ap.add_argument("--solo-recorridos", action="store_true")
    ap.add_argument("--rehacer", action="store_true")
    args = ap.parse_args()

    try:
        strides = [int(s) for s in args.strides.split(",") if s.strip()]
    except ValueError:
        die(f"--strides no es una lista de enteros: {args.strides!r}")
    if not strides:
        die("--strides esta vacio")
    if args.eval_stride in strides:
        die(f"--eval-stride {args.eval_stride} esta ENTRE los strides barridos.\n"
            f"  Ese brazo habria entrenado exactamente sobre las posiciones del\n"
            f"  examen, y se llevaria una ventaja que no es el efecto que se mide.\n"
            f"  Elige una rejilla de evaluacion que no este en el eje.")
    if args.windows_per_epoch <= 0:
        log("AVISO: --windows-per-epoch 0 -> cada brazo entrena sobre su pool "
            "entero.\n  El brazo de stride 1 recibiria 146x mas pasos de "
            "gradiente que el de 16,\n  y la tabla mediria el presupuesto y no "
            "la densidad. Es legal, pero hay\n  que decirlo en el reporte.")

    store = SweepStore()
    wstore = WindowDatasetStore()

    # La fuente esta en .gitignore (/data/sources/), asi que en una maquina recien
    # hecha NO esta, y sin ella no hay datasets que extraer. Se comprueba ANTES de
    # nada y con el comando que la reconstruye al lado: descubrirlo a mitad es
    # como se dan por imposibles cosas que si se pueden hacer (CLAUDE.md del
    # coordinador, "el dataset se genera, y esta comprobado que se puede").
    if not args.solo_recorridos:
        try:
            ds = SourceDataset(args.fuente)
            if not ds.labels_path.exists():
                raise FileNotFoundError(ds.labels_path)
        except Exception as exc:                              # noqa: BLE001
            die(f"no encuentro la fuente '{args.fuente}' ({type(exc).__name__}).\n"
                f"  Vive en data/sources/, que esta en .gitignore: un clon limpio no\n"
                f"  la trae. Se RECONSTRUYE (~15-20 min de renders, reproducible):\n"
                f"      .venv/bin/python scripts/bench_dataset.py build\n"
                f"  o se copia de otra maquina que ya la tenga.")

    if not store.exists(args.hereda_de):
        die(f"no existe el recorrido '{args.hereda_de}', del que se hereda la red "
            f"base y la receta.\n  Usa --hereda-de con uno que exista.")
    padre = store.spec(args.hereda_de)
    log(f"Estudio {ESTUDIO}")
    log(f"  red base: {padre['base_label']} (heredada de {args.hereda_de})")
    log(f"  receta:   {padre['base_recipe']} + windows_per_epoch="
        f"{args.windows_per_epoch}")
    log(f"  fuente:   {args.fuente} · ventana {args.window_size} · "
        f"eval_stride {args.eval_stride} · seed de split {args.seed_split}")
    log(f"  brazos:   {strides}")
    if args.humo:
        log("  MODO HUMO: recorridos `stride-hNN` en el estudio "
            f"'{ESTUDIO}-humo'. NO son el estudio.")

    # De grande a pequeno: el caro va el ultimo (ver el docstring, punto 4).
    orden = sorted(strides, reverse=True)

    if not args.solo_recorridos:
        log("\nDatasets:")
        for s in orden:
            construir_dataset(s, args, wstore)

    if not args.solo_datasets:
        log("\nRecorridos:")
        for s in sorted(strides):
            try:
                construir_sweep(s, args, store, padre)
            except SweepError as e:
                die(f"{nombre_sweep(s, args.humo)}: {e.code}: {e.message}\n  {e.hint}")

    log("\nListo. Nada se ha alquilado ni entrenado.")
    if not args.solo_datasets:
        nombres = " ".join(f"--sweep {nombre_sweep(s, args.humo)}"
                           for s in sorted(strides))
        log(f"\n  Estimar (no gasta):\n"
            f"    .venv/bin/python scripts/estudio_estimar.py {nombres}\n"
            f"\n  Lanzar la flota (ALQUILA):\n"
            f"    scripts/desacoplar.sh .venv/bin/python scripts/estudio_flota.py \\\n"
            f"        {nombres} \\\n"
            f"        --cpu E5-26 --criba 2 --git --horas-max 6 "
            f"--prefijo stride- --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
