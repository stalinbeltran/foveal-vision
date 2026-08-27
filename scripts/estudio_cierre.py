#!/usr/bin/env python3
r"""Crea los recorridos del plan de cierre, con el criterio ya escrito.

Que es esto
-----------
`docs/plan-cierre-2026-08-26.md` fija QUE se mide y COMO se lee, antes de mirar
nada. Este script solo lo traduce a recorridos en `sweeps/`, para que la lista de
estudios y la lista de barridos no puedan divergir -- misma razon por la que
existe `estudio_prioridades.py`, y mismo reparto de responsabilidades: aqui no se
alquila ni se entrena nada. Lo que gasta es `estudio_flota.py`.

    .venv/bin/python scripts/estudio_cierre.py --dataset dirty1000-80px-16px-r20260824
    .venv/bin/python scripts/estudio_cierre.py --dataset <B> --bloque A
    .venv/bin/python scripts/estudio_cierre.py --dataset <B> --solo ov-alto

LAS DOS COSAS QUE HAY QUE RESPETAR SI SE TOCA ESTO
--------------------------------------------------
1. **Las semillas 6-10 son el punto, no un detalle.** `ov-sig` y `bp-sig` existen
   para llevar un contraste de 5-contra-5 a 10-contra-10, y eso solo funciona si
   las semillas nuevas NO se pisan con las que ya hay en `ov-fov` / `borde-ancho`
   (que son la 1 a la 5). El eje replica arranca en `base_recipe_value["seed"]`
   (`fv/sweeps/generate.py`), asi que se desplaza por ahi y no por `seed=`, que
   es otra cosa. Si esto se rompe, el estudio no falla: **repite 10 runs ya
   pagados y sigue sin poder declarar**, que es el peor de los fallos posibles.

   Y 10-contra-10 no es un numero redondo: es el TECHO. `permutation_test` se
   niega a correr por encima de C(n+m,n) = 200.000, y C(22,11) = 705.432.
   Comprobado el 2026-08-26:  2v2 -> p_min 0,333 · 5v5 -> 0,0079 ·
   10v10 -> 1,08e-5 · 11v11 -> se niega.

2. **La plana NO se re-deriva: se hereda del tanteo.** `derive_base` fuerza hoy
   `border_px = 0` cuando `regions='single'`, asi que la base que produce es
   `ws16-p0-d1-L4` (entrada 16x16) y NO la `plana-24-single` (entrada 24x24) con
   la que se corrio la fase 1. Ver el aviso de abajo. Por eso `pl-f2-*` copia
   `base_network_value` del recorrido de fase 1 en vez de volver a derivarla:
   asi fase 1 y fase 2 son la MISMA red por construccion, que es la unica forma
   de que la fase 2 continue el tanteo en vez de empezar otro estudio.

3. **El bloque B es TANTEO y no puede declarar ganador.** Son 2 semillas: el `p`
   minimo alcanzable es 0,333. Sirve para decidir a que ejes se les dedican 5
   semillas (bloque C), nunca para mover un vigente. El criterio de ascenso esta
   en el plan, §2, escrito antes de mirar.

AVISO: LA BASE DE LA PLANA NO SE PUEDE RE-DERIVAR HOY (medido 2026-08-26)
------------------------------------------------------------------------
`derive_base(16, overrides={'regions':'single',...}, border_px=4)` devuelve
`border_px = 0` -> etiqueta `ws16-p0-d1-L4`, entrada 16x16. La correccion la
aplica `derive.py` con este motivo: *"regions='single' es la CNN plana: una sola
rama sobre todo el input, sin borde"*.

Pero `plan-cnn-plana.md` §5.1 exige lo contrario y lo llama *la premisa*: la
plana tiene que ver **la MISMA AREA ORIGINAL** que la foveada, o sea **24x24**
(`N=24`, `c_frac=16/24`, `d=1`), porque comparar 16x16 contra 24x24 mediria
*cuanta imagen ve cada una* y no *como la mira*. Los recorridos `pl-t-*` de la
fase 1 son 24x24 (su `base_network_value` trae `N: 24`), y se crearon con el
codigo anterior a la reparametrizacion del 25-ago.

O sea: la correccion confunde DOS cosas que la ortografia nueva separo -- el
anillo (estructura, que `single` no tiene) y el recorte (area, que si tiene que
conservar). Consecuencia practica: **hoy `estudio_plana.py` no reproduce la base
con la que se midio la plana**. Mientras eso no se arregle en `derive.py`, la
unica forma correcta de continuar el estudio es heredar la base, que es lo que
hace `hereda_de` aqui abajo. Comprobado que la geometria heredada es legal:
`check_dims({fovea 16, border 4, reduce 1}, single=True)` -> [] (sin problemas).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.sweeps.generate import generate_sweep       # noqa: E402
from fv.sweeps.runner import prepare_sweep          # noqa: E402
from fv.sweeps.spec import SweepError               # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402
from fv.training.recipe import RecipeStore          # noqa: E402

RECIPE = "plan40"
OBJECTIVE = "f1"
EPOCHS = 150                    # el de los recorridos con los que se compara

# La foveada vigente: ws16-p2-d2-L4, 167.852 parametros.
FOVEADA = {"n_layers": 4, "channels": [16] * 4}
# La plana de control: misma entrada 24x24, ~los mismos parametros (165.430).
PLANA = {"regions": "single", "border_reduce": 1, "n_layers": 4,
         "channels": [22] * 4}

ESTUDIOS = [
    # ================================================================ bloque 0
    # NO es un estudio: es la COMPROBACION PREVIA de que el dato reproduce.
    #
    # Por que existe (medido el 2026-08-26): esta maquina se rehizo y `data/` se
    # perdio entera. El dataset se reconstruyo desde los specs congelados y
    # `bench-dirty1000-16` (stride 8) SI reproduce su huella de git -- o sea que
    # la fuente es la buena. Pero el de los estudios (stride 5) NO: misma
    # config, mismos num_windows, mismo windows_per_split, mismos
    # positives_per_corner y `split.json` identico byte a byte... y otro sha256
    # del .npz.
    #
    # Eso deja la pregunta abierta, y no es academica: ov-sig, bp-sig y pl-f2-*
    # SUMAN semillas a runs que ya existen. Si el dato no es el mismo, esos tres
    # comparan peras con manzanas y NO dan ningun sintoma -- el `p` sale igual de
    # creible. Es exactamente el fallo silencioso que el proyecto evita.
    #
    # El contraste local no vale: divergencia de CPU y divergencia de dato se
    # confunden (medido: cruzar de familia mueve el f1 hasta 0,0457). Asi que se
    # re-corre EL MISMO punto con LA MISMA semilla en la MISMA familia de CPU
    # (--cpu E5-26, donde esta medido que el entrenamiento sale identico bit a
    # bit) y se compara epoca a epoca contra lo que dejo escrito el run viejo.
    #
    # Referencia (runs/ov-fov-0010-overlap_fovea_px2_seed1/metrics.jsonl):
    #   e1 train_loss=0.4850498190031013  val_f1=0.6806736039399476
    #   e2 train_loss=0.29654462766599127 val_f1=0.8084834834834835
    #   e3 train_loss=0.2159383792818135  val_f1=0.8524697415767093
    # Iguales -> el dato reproduce y los tres recorridos que suman son validos.
    # Distintas -> no reproduce, y hay que rehacer el diseno (ver el plan §4).
    #
    # 3 epocas y patience=0: con esto basta para distinguir, y cuesta ~2 runs
    # cortos en vez de dos completos.
    {
        "bloque": "0", "name": "repro-chk",
        "que": "B0 - ¿reproduce el dato? Mismo punto, misma semilla, misma familia de CPU",
        "axis": "overlap_fovea_px", "range": [2],
        "base": FOVEADA, "border_px": 4, "epochs": 3, "semillas": 2,
        "receta_extra": {"patience": 0},   # que pare en la 3, no antes
    },

    # ================================================================ bloque A
    # Cerrar overlap_fovea_px. MEDIDO el 2026-08-26: el eje es cost-neutral en
    # parametros (167.852 en todo el rango 0..7), asi que no hay coste que
    # sopesar contra la ganancia -- a diferencia de border_reduce.
    #
    # ⚠ EL EJE ENTERO, y no solo los puntos que faltaban: `repro-chk` demostro
    # que el dato reconstruido NO es el r20260824 (ver el AVISO de abajo), asi
    # que los 20 runs de `ov-fov` estan sobre OTRO dataset y no se pueden sumar.
    # Volver a medir 0/1/2/4 no es repetir trabajo: es lo unico que hace
    # comparables 5/6/7 con ellos.
    #
    # Y sale un ancla gratis: el punto 2 ES la configuracion vigente, asi que su
    # media con 5 semillas mide CUANTO movio el dato respecto de los 0,9308 que
    # dejo escritos el #13.
    {
        "bloque": "A", "name": "ov-r26",
        "que": "A1 - overlap_fovea_px ENTERO sobre el dato nuevo, hasta la pared legal (7)",
        "axis": "overlap_fovea_px",
        # overlap_fovea_range(16) = [0..7]: 7 es el maximo que la geometria
        # admite (la rama del borde veria 14 de los 16 px de fovea). No hay
        # "mas alla": si gana 7, gana el extremo LEGAL, que es otra frase.
        "range": [0, 1, 2, 4, 5, 6, 7],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 5,
    },
    {
        "bloque": "A", "name": "ov-sig26",
        "que": "A2 - llevar el contraste decisivo a 10 semillas por punto (el techo del test)",
        "axis": "overlap_fovea_px",
        "range": [2, 4],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 5,
        "seed0": 6,             # <- las 1..5 las pone ov-r26, sobre ESTE dato
    },

    {
        "bloque": "A", "name": "ov-sig7",
        "que": "A3 - el GANADOR REAL (7) a 10 semillas contra el vigente",
        "axis": "overlap_fovea_px",
        # Solo el 7: las semillas 6-10 del punto 2 ya las pone `ov-sig26`, y las
        # 1-5 de los dos las pone `ov-r26`. Asi el contraste 7-contra-2 llega a
        # 10 contra 10 con 5 runs en vez de 20.
        #
        # Existe porque el plan NO podia saber cual seria el ganador: se escribio
        # con el 4 como candidato -- era el ganador nominal sobre el dato viejo --
        # y sobre el dato nuevo el eje sube monotono hasta la pared legal y gana
        # el 7 (p = 0,008 con 5 semillas, el suelo alcanzable con 5 contra 5).
        # Llevar a 10 contra 10 el punto que YA declara no es redundante: es lo
        # que pidio el encargo -- "verificar que el resultado sea significativo"--
        # y con 10 contra 10 el suelo del test baja a 1,08e-5.
        "range": [7],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 5,
        "seed0": 6,
    },

    # ================================================================ bloque B
    # Tanteo de 2 semillas de lo que nunca se midio. ACOTA, no declara.
    {
        "bloque": "B", "name": "wd-t",
        "que": "B1 - weight_decay: la regularizacion por la puerta barata (brecha val/train +28 %)",
        "axis": "weight_decay", "range": [0.0, 1e-5, 1e-4, 1e-3],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "opt-t",
        "que": "B2 - optimizer: y adam vs adamw es un CONTROL (con wd=0 deben salir iguales)",
        "axis": "optimizer", "range": ["adam", "adamw", "sgd"],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "lp-t",
        "que": "B3 - lambda_pos: existencia contra posicion. Se rankea por f1 (contrato 9)",
        "axis": "lambda_pos", "range": [0.5, 1.0, 2.0, 4.0],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "sb-t",
        "que": "B4 - smooth_l1_beta: donde la perdida de posicion pasa de cuadratica a lineal",
        "axis": "smooth_l1_beta", "range": [0.02, 0.08, 0.32],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "pat-t",
        "que": "B5 - patience: NO es calidad, es el criterio de parada. El 5 prueba el minimo medido (8)",
        "axis": "patience", "range": [5, 10, 20],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "mrg-t",
        "que": "B6 - merge: NO cost-neutral. MEDIDO hoy: sum=91.052 params contra concat=167.852 (0,54x)",
        "axis": "merge", "range": ["concat", "sum"],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "pool-t",
        "que": "B7 - pool_mode: con texto, max conserva trazos finos que avg difumina",
        "axis": "pool_mode", "range": ["avg", "max"],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "pad-t",
        "que": "B8 - pad_mode: solo toca ventanas de borde; se espera poco y se mide igual",
        "axis": "pad_mode", "range": ["edge", "mean", "zero"],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },
    {
        "bloque": "B", "name": "ovb-t",
        "que": "B9 - overlap_border_px: el simetrico del solape. Con borde 4 y reduce 2 SOLO admite {0,2}",
        "axis": "overlap_border_px", "range": [0, 2],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 2,
    },

    # ================================================================ bloque C
    # Verificacion de 5 semillas. Estos tres no dependen del tanteo: ya estaban
    # pendientes. Los que salgan del bloque B se anaden despues, con --solo.
    {
        "bloque": "C", "name": "bp-r26",
        "que": "C1 - border_px 8 contra 4 a 10 semillas PROPIAS: cerrar la p=0,063 medida DOS veces",
        "axis": "border_px", "range": [4, 8],
        # El anillo se queda en 2 celdas -> N no se mueve -> mismos parametros.
        # Es la MISMA atadura de `borde-ancho`, y tiene que serlo: sin ella los
        # 10 runs nuevos no serian comparables con los 10 que ya hay.
        "couple": {"border_reduce": {"axis": "border_px", "values": [2, 4]}},
        # 10 semillas AQUI, no 5 sumadas a `borde-ancho`: aquellas estan sobre
        # el dato viejo. 10 contra 10 es el techo del test exacto.
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS, "semillas": 10,
    },
    {
        "bloque": "C", "name": "pl-f2-bs",
        "que": "C2 - plana fase 2, batch_size: el tanteo dejo 170 ganando POR DENTRO",
        "axis": "batch_size", "range": [85, 170, 340],
        "hereda_de": "pl-t-bs",          # la RED de la fase 1; el DATO ya no
        # 5 semillas propias. Se hereda la red (que no se puede re-derivar, ver
        # el AVISO) pero NO las semillas: el tanteo esta sobre el dato viejo.
        # Un estudio de 5 semillas declara solo, sin apoyarse en aquel -- el
        # tanteo dijo DONDE mirar, y ese rango sigue valiendo.
        "epochs": EPOCHS, "semillas": 5,
    },
    {
        "bloque": "C", "name": "pl-f2-nl",
        "que": "C3 - plana fase 2, n_layers: con 5 semillas se ve si el 0,0000 de L6 es bimodalidad",
        "axis": "n_layers", "range": [4, 5, 6],
        "hereda_de": "pl-t-nl",          # la RED de la fase 1; el DATO ya no
        "epochs": EPOCHS, "semillas": 5,
    },
]


def _hereda(name: str, est: dict, dataset: str, receta: dict,
            store: SweepStore) -> dict:
    """Un recorrido nuevo sobre la MISMA red base que otro que ya existe.

    Para que una fase 2 continue a una fase 1 en vez de empezar otro estudio, la
    red tiene que ser la misma; y la de la plana hoy **no se puede re-derivar**
    (ver el AVISO del docstring). Asi que se copia de su spec, que es el unico
    sitio donde esa red sigue escrita entera.

    Se comprueban las DOS cosas que harian incomparables los dos recorridos, y
    se falla en vez de avisar: un estudio que mide otra red y lo dice en una
    linea de log es peor que uno que no arranca.
    """
    padre = store.spec(est["hereda_de"])
    base = padre["base_network_value"]
    # Se hereda LA RED, no los numeros. Son dos cosas distintas y confundirlas es
    # el fallo que este script existe para evitar:
    #   - la RED puede venir de un recorrido sobre otro dataset sin problema: es
    #     una definicion, no una medida.
    #   - los RUNS no: sumar semillas exige el mismo dato, y por eso `seed0` y
    #     `hereda_de` nunca se usan juntos.
    # Se anota en el spec para que quede dicho de donde salio y sobre que dato se
    # midio aquello, en vez de que el spec parezca derivado (U1.6).
    otro_dato = padre["window_dataset"] != dataset
    if otro_dato and est.get("seed0"):
        raise SweepError(
            "hereda_semillas_de_otro_dataset",
            f"'{est['hereda_de']}' se midio sobre '{padre['window_dataset']}' y "
            f"aqui se pide '{dataset}', pero se piden semillas desplazadas "
            f"(seed0={est['seed0']}), o sea SUMAR a sus runs",
            "dos medidas solo se comparan si coinciden en el dato: quita seed0 y "
            "corre las semillas enteras sobre el dato nuevo")
    if padre["base_recipe"] != RECIPE:
        raise SweepError(
            "hereda_otra_receta",
            f"'{est['hereda_de']}' usa la receta '{padre['base_recipe']}' y aqui "
            f"se usa '{RECIPE}'",
            "iguala la receta base o el eje mediria tambien el cambio de receta")

    espacio = {est["axis"]: list(est["range"])}
    s0 = int(receta.get("seed", 1))
    if est["semillas"] > 1 and est["axis"] != "seed":
        espacio["seed"] = [s0 + k for k in range(est["semillas"])]

    spec = {
        "window_dataset": dataset,
        "base_network": None,
        "base_label": padre["base_label"],
        "base_network_value": base,
        "base_recipe": RECIPE,
        "base_recipe_value": receta,
        # de donde salio la base, para que un spec heredado no parezca derivado:
        # U1.6, "un objeto ensena la definicion con la que se hizo".
        "derivation": padre.get("derivation", {}),
        "corrections": padre.get("corrections", []),
        "hereda_base_de": est["hereda_de"],
        "hereda_base_medida_sobre": padre["window_dataset"],
        "space": espacio,
        "strategy": "grid",
        "objective": OBJECTIVE,
        "budget": {"epochs": est["epochs"]},
        "device": "cpu",
        "seed": s0,
        "seeds": est["semillas"],
        "study": "cierre-2026-08-26",
    }
    return prepare_sweep(name, spec, base, store)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bloque", action="append", choices=["0", "A", "B", "C"],
                    help="crea solo estos bloques (repetible)")
    ap.add_argument("--solo", action="append",
                    help="crea solo estos recorridos (repetible)")
    ap.add_argument("--rehacer", action="store_true",
                    help="borra y rehace el que ya exista (PIERDE sus runs)")
    args = ap.parse_args()

    store = SweepStore()
    existentes = {p.name for p in (ROOT / "sweeps").iterdir() if p.is_dir()}
    receta_base = RecipeStore().get(RECIPE).as_dict()
    bloques = set(args.bloque or ["0", "A", "B", "C"])
    pedidos = set(args.solo or [])
    creados, saltados, fallidos = [], [], []

    for est in ESTUDIOS:
        name = est["name"]
        if pedidos and name not in pedidos:
            continue
        if not pedidos and est["bloque"] not in bloques:
            continue
        if name in existentes:
            if not args.rehacer:
                print(f"  = {name:10s} ya existe, se deja (--rehacer para borrarlo)")
                saltados.append(name)
                continue
            # --rehacer BORRA. Y un recorrido con runs en disco es dinero ya
            # gastado, asi que aqui se para en vez de avisar: `estudio_flota.py`
            # se salta los puntos ya `done`, o sea que borrarlos no es "empezar
            # de cero", es "volver a pagarlos". Fallo ruidoso antes que silencioso.
            hechos = sorted((ROOT / "runs").glob(f"{name}-*"))
            if hechos:
                print(f"  ! {name:10s} --rehacer NO se aplica: tiene {len(hechos)} "
                      f"runs en disco (p.ej. {hechos[0].name}).\n"
                      f"      Borrarlos es volver a pagarlos. Borra runs/{name}-* "
                      f"a mano si de verdad es lo que quieres.")
                fallidos.append(name)
                continue
            store.delete(name)
            print(f"  - {name:10s} borrado (no tenia runs), se rehace")

        # El eje replica arranca en base_recipe_value["seed"], NO en `seed=`.
        # Desplazarlo es lo que hace que ov-sig/bp-sig SUMEN semillas a los
        # recorridos que ya estan en disco en vez de repetirlas. Ver el docstring.
        receta = dict(receta_base)
        if est.get("seed0"):
            receta["seed"] = int(est["seed0"])
        receta.update(est.get("receta_extra") or {})

        try:
            if est.get("hereda_de"):
                spec = _hereda(name, est, args.dataset, receta, store)
            else:
                spec = generate_sweep(
                    name, args.dataset, est["axis"], est["range"],
                    base_recipe=RECIPE, base_recipe_value=receta,
                    objective=OBJECTIVE, budget={"epochs": est["epochs"]},
                    seeds=est["semillas"], device="cpu",
                    overrides=est["base"], border_px=est.get("border_px"),
                    couple=est.get("couple"),
                    study="cierre-2026-08-26", sstore=store,
                )
        except SweepError as e:
            print(f"  ! {name:10s} {e.code}: {e.message}\n      {e.hint}")
            fallidos.append(name)
            continue

        n = len(spec["points"])
        if spec.get("hereda_base_medida_sobre") not in (None, args.dataset):
            print(f"      ⚠ {name}: hereda LA RED de '{spec['hereda_base_de']}', que se "
                  f"midio sobre '{spec['hereda_base_medida_sobre']}'.\n"
                  f"        La red viaja; los numeros de aquel recorrido NO. Este "
                  f"declara solo, con sus 5 semillas.")
        semillas = spec["space"].get("seed", [receta["seed"]])
        desc = spec.get("discarded") or []
        print(f"  + [{est['bloque']}] {name:10s} {len(spec['space'][est['axis']])} valores "
              f"x semillas {semillas} = {n:3d} runs   [{est['que']}]")
        for d in desc:
            print(f"      descartado {d['point']}: {d['problems'][0]['message']}")
        creados.append((name, n))

    print(f"\n{len(creados)} recorridos creados, {len(saltados)} ya estaban, "
          f"{len(fallidos)} fallaron.")
    if creados:
        print(f"Runs nuevos: {sum(n for _, n in creados)}")
        print("\nSiguiente paso (esto SI gasta):")
        print("  .venv/bin/python scripts/estudio_flota.py \\")
        for name, _ in creados:
            print(f"      --sweep {name} \\")
        print("      --cpu E5-26 --sin-cpu v2 --max-price 0.12 --criba 2 --git --estimar")
    return 1 if fallidos else 0


if __name__ == "__main__":
    raise SystemExit(main())
