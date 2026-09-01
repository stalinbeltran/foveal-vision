#!/usr/bin/env python3
r"""Los recorridos del estudio de `s_center` / `s_periph`, con el criterio escrito.

Que es esto
-----------
`docs/plan-strides-rama-2026-09-01.md` fija QUE se mide y COMO se lee, ANTES de
mirar nada. Este script solo lo traduce a recorridos en `sweeps/`, igual que
`estudio_dropout.py` y `estudio_cierre.py` hacen con los suyos: aqui no se
alquila ni se entrena nada. Lo que gasta es `estudio_flota.py`.

    .venv/bin/python scripts/estudio_strides.py --dataset <B> --eje s_center --fase tanteo
    .venv/bin/python scripts/estudio_strides.py --dataset <B> --eje s_periph --fase tanteo
    .venv/bin/python scripts/estudio_strides.py --dataset <B> --eje diagonal --fase tanteo
    .venv/bin/python scripts/estudio_strides.py --dataset <B> --eje s_center --fase completo
    .venv/bin/python scripts/estudio_strides.py --dataset <B> --control          # §4.3

CINCO COSAS QUE HAY QUE RESPETAR SI SE TOCA ESTO
------------------------------------------------
1. **El rango va EXPLICITO, nunca "auto".** `check_sweep` acepta
   `{"s_center": "auto"}` porque el campo esta en `GEOMETRY_AUTO`, pero
   `stride_range(16, n_layers=4)` devuelve **[1]** (medido 2026-09-01): el
   recorrido entrenaria N veces la misma red SIN AVISAR, que es exactamente el
   fallo que costo `dropout`. La incoherencia de fondo (`builder.py:128` dice
   que `n_layers` no entra en el rango, `fovea/__init__.py:351` lo usa como
   raiz) esta anotada en el plan §3.4 y NO se arregla aqui.

2. **La fase 2 ancla en el VIGENTE, no en el ganador de la otra rama.** El plan
   §5.4 lo decide antes de tener un numero: cada eje se mide contra la misma
   base que el resto de la tabla. Encadenarlos mediria el efecto marginal y
   arrastraria el error de la fase anterior.

3. **El rango de la fase 2 lo elige TABLA_PICO, no quien lanza.** Igual que en
   `estudio_dropout.py`: para cada resultado posible del tanteo, que rango se
   barre y por que -- escrito antes de mirar. Si el rango se eligiera despues,
   "lo dijo el tanteo" no significaria nada.

4. **`p*` se DERIVA del disco.** Con las mismas funciones que el informe
   (`sweep_trials` + `aggregate_seeds`), para que el pico que elige el rango sea
   el mismo que sale en la tabla. `--pico` lo fuerza y AVISA EN VOZ ALTA.

5. ⚠ **ESTO NO SE PUEDE PROBAR EN SECO: `crear` ESCRIBE EN EL REPO DE DATOS.**
   Pasarle un `SweepStore(root=<tmp>)` NO aisla nada -- `SweepStore.destino()`
   delega en `artefactos.destino_agrupado()`, que resuelve la carpeta del mes
   contra la raiz REAL e ignora el `root` del store (comprobado el 2026-09-01:
   una prueba con store temporal dejo `sc-t`, `sp-t` e `if-c` escritos en
   `foveal-vision-data/2026/09-septiembre/sweeps/`, y hubo que borrarlos a
   mano). Crear un recorrido no gasta dinero -- solo `estudio_flota.py` gasta --
   pero si deja artefactos con nombre reservado: `store.exists()` los ve, asi
   que el intento siguiente dice "ya existe, se deja" y NO rehace el recorrido
   aunque el rango haya cambiado. Para probar el camino sin persistir, usa
   `build_generated_spec` + `expand_points`, que no escriben.

6. **El control ISO-FEATURES es un recorrido APARTE y se corre solo si hace
   falta** (plan §4.3 y criterio R5). Aisla capacidad de resolucion: la misma
   cabeza que el brazo con stride, pero con la resolucion entera. Los canales
   estan CALCULADOS, no tecleados, y se reduce solo la ultima capa: ver
   `canales_iso()`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.models.builder import network_trace          # noqa: E402
from fv.sweeps.generate import generate_sweep        # noqa: E402
from fv.sweeps.runner import es_medida, sweep_trials  # noqa: E402
from fv.sweeps.spec import SweepError                # noqa: E402
from fv.sweeps.store import SweepStore               # noqa: E402
from fv.sweeps.winner import aggregate_seeds         # noqa: E402
from fv.training.recipe import RecipeStore           # noqa: E402
from fv.training.registry import RunStore            # noqa: E402

RUNS = RunStore()

RECIPE = "plan40"
OBJECTIVE = "f1"
# 300 y no 150, plan §5.3: un run que para por el tope mide presupuesto, no
# calidad (R1 del protocolo). Aqui es casi gratis: solo el brazo s=1 puede
# consumirlo -- los demas corren entre 4x y 8x mas rapido (medido, plan §3.2).
EPOCHS = 300
ESTUDIO = "strides-rama-2026-09-01"

# La foveada VIGENTE: ws16-p2-d2-L4, 168.652 parametros, 12.800 features a la
# cabeza. La misma base que toda la tabla -- no la "mejor conocida" (#14 dejo
# border_px 8 y overlap_fovea_px 7 medidos y SIN APLICAR).
FOVEADA = {"n_layers": 4, "channels": [16] * 4}
BORDER_PX = 4
N_LAYERS = 4
CANALES_BASE = 16

# Un recorrido por eje y fase. El prefijo de Vast sale de aqui: se pasa a
# `estudio_flota.py --prefijo`, y con un workspace por tema es lo unico que
# distingue TUS maquinas de las de otra sesion (CLAUDE.md § "Varias sesiones").
EJES = {
    "s_center": {"tanteo": "sc-t", "completo": "sc-v", "prefijo": "sc-",
                 "otro": "s_periph", "espacio": "s_center", "couple": None},
    "s_periph": {"tanteo": "sp-t", "completo": "sp-v", "prefijo": "sp-",
                 "otro": "s_center", "espacio": "s_periph", "couple": None},
    # El brazo DIAGONAL: las dos ramas al mismo stride. Es el unico que recorta
    # de verdad (3.200 features con s=2, 800 con s=4) y el unico que acelera de
    # verdad -- los simples dejan una rama a stride 1, que domina el reloj:
    # MEDIDO el 2026-09-01, s=2 cuesta 0,50x en simple y 0,24x en diagonal.
    #
    # No es un tercer eje: es el eje `s_center` con `s_periph` ATADO a el
    # (`couple`), o sea una diagonal y no un producto cartesiano. Sigue siendo
    # "un eje cada vez" (OAT) y sigue costando 4 puntos, no 16.
    "diagonal": {"tanteo": "sd-t", "completo": "sd-v", "prefijo": "sd-",
                 "otro": None, "espacio": "s_center", "couple": "s_periph"},
}
CONTROL = "if-c"            # el iso-FEATURES de §4.3
PREFIJO_CONTROL = "if-"

# El tanteo: 4 valores x 2 semillas = 8 runs por eje. ACOTA, no declara (el p
# minimo alcanzable con 2 contra 2 es 0,333).
RANGO_TANTEO = [1, 2, 3, 4]
MOTIVO_TANTEO = (
    "1 es el ancla (el vigente); 2 es el punto util (mapa 10x10 alineado con "
    "N=20, campo receptivo aun dentro de la vista); 3 contesta la pregunta de "
    "la alineacion (7x7 con resto); 4 ACOTA el eje por la derecha, por la "
    "leccion de borde-ancho y patience -- un ganador pegado al borde del rango "
    "no es un optimo")

# --------------------------------------------------------------------------
# LA TABLA DEL PLAN §6, EN CODIGO. Escrita antes de tener un solo numero: para
# cada pico posible del tanteo, que rango barre la fase 2 y por que. Cubre los
# cuatro valores de RANGO_TANTEO y ninguno mas: un pico que no este aqui es un
# fallo de coherencia, no un caso a improvisar, y el script se niega.
TABLA_PICO = {
    1: ([1, 2],
        "gana el vigente: el submuestreo cuesta calidad ya en el primer paso. "
        "La validacion solo tiene que confirmar 1 contra 2 con 5 semillas -- "
        "los strides grandes ya perdieron y repetirlos es pagar por confirmar "
        "al perdedor, que es justo donde no se debe gastar"),
    2: ([1, 2, 3],
        "gana 2: se encierra por los dos lados (1 por debajo, 3 por encima) "
        "para que el optimo quede INTERIOR y no en un borde"),
    3: ([1, 2, 3, 4],
        "gana 3: se encierra con 2 y 4. Entra el 4 aunque haya perdido en el "
        "tanteo, porque con 2 semillas la diferencia no declara nada"),
    4: ([1, 2, 4, 6],
        "gana el EXTREMO del tanteo, asi que el eje NO esta acotado por la "
        "derecha: se extiende mas alla (6), como hizo borde-ancho cuando su "
        "ganador salio en el borde. Con s=6 el mapa es 4x4"),
}


def canales_iso(stride: int) -> list[int] | None:
    """Los `channels` que a stride 1 dan las MISMAS features que `stride`.

    CALCULADO, no tecleado (plan §4.3). La salida de cada rama es un cuadrado
    completo, asi que features = 2 * lado(s)^2 * c: igualarlas es dividir. Si
    no sale un entero, devuelve None y el control NO se crea con un valor
    aproximado -- un control que no iguala de verdad no controla nada.

    ⚠ Se reduce SOLO LA ULTIMA CAPA, y esa es la decision. Bajar las cuatro
    (`[c]*L`) iguala las features igual de bien, pero cambia ademas las tres
    convoluciones de en medio: MEDIDO el 2026-09-01, contra el brazo s=2 la
    diferencia total de parametros es de 8.580 (7,7 %) con `[10]*4` y de 1.740
    (1,6 %) con `[16,16,16,10]`. Un control que mueve mas cosas que la que
    controla no controla: con `[16,16,16,10]` las conv 1..L-1 son IDENTICAS a
    las del brazo con stride y solo cambia la que produce el mapa que lee la
    cabeza, que es justo lo que se quiere aislar.

    ⚠ Y el nombre es ISO-FEATURES, no iso-parametros: lo que queda EXACTO es
    la cabeza (96.012 pesos en los dos, medido). Los totales siguen difiriendo
    en ese 1,6 % de las conv, y eso se dice en el reporte en vez de redondearlo.
    """
    base = dict(FOVEADA, fovea_px=16, border_px=BORDER_PX)
    f_stride = network_trace(dict(base, s_center=stride, s_periph=1))["flat_features"]
    f_uno = network_trace(dict(base, s_center=1, s_periph=1))["flat_features"]
    # f_uno = 2*400*CANALES_BASE -> c = f_stride / (f_uno / CANALES_BASE)
    por_canal = f_uno // CANALES_BASE
    if f_stride % por_canal:
        return None
    c = f_stride // por_canal
    if c < 1:
        return None
    return [CANALES_BASE] * (N_LAYERS - 1) + [c]


def pico_del_tanteo(eje: str, store: SweepStore) -> tuple[int, int, float]:
    # `eje` es la clave de EJES; la del PUNTO es cfg["espacio"] -- en el
    # diagonal son distintas (el eje se llama "diagonal", el campo es
    # `s_center` con `s_periph` atado).
    """El mejor valor del eje en su tanteo, leido del disco con las MISMAS
    funciones que usa `estudio_informe.py`.

    Falla en vez de adivinar: un tanteo a medias no puede elegir el rango de un
    estudio que cuesta 20 runs, y un rango elegido sobre 3 de 8 es peor que
    ninguno porque no lo dice nadie.
    """
    tanteo = EJES[eje]["tanteo"]
    try:
        tabla = sweep_trials(tanteo, store=store, run_store=RUNS)
    except Exception as e:                       # noqa: BLE001 - el motivo importa
        raise SystemExit(
            f"No puedo leer el tanteo '{tanteo}': {e}\n"
            f"  -> crealo y correlo primero:\n"
            f"     .venv/bin/python scripts/estudio_strides.py --dataset <B> "
            f"--eje {eje} --fase tanteo")
    medidos = [t for t in tabla["trials"] if es_medida(t)]
    total = len(tabla["trials"])
    if not medidos:
        raise SystemExit(
            f"El tanteo '{tanteo}' no tiene ningun punto medido todavia "
            f"(0/{total}).\n"
            f"  -> mira como va:  .venv/bin/python scripts/estudio_progreso.py "
            f"--sweep {tanteo} --tabla")
    if len(medidos) < total:
        raise SystemExit(
            f"El tanteo '{tanteo}' esta a medias: {len(medidos)}/{total} runs "
            f"medidos.\n"
            f"  El rango de la fase 2 se elige con el pico del tanteo; elegirlo "
            f"sobre un tanteo incompleto es elegirlo sobre el subconjunto de "
            f"puntos que dio la casualidad de terminar primero.\n"
            f"  -> termina el tanteo, o forzalo con --pico <valor> si de verdad "
            f"es lo que quieres.")
    grupos = aggregate_seeds(medidos, tabla["direction"], "seconds_per_epoch")
    mejor = grupos[0]
    return int(mejor["point"][EJES[eje]["espacio"]]), len(medidos), float(mejor["value"])


def crear(name: str, espacio_eje: str, rango: list, semillas: int, dataset: str,
          receta: dict, store: SweepStore, motivo: str,
          overrides: dict | None = None, atado: str | None = None) -> bool:
    # `store.exists`, y no un listado de `store.root`: la raiz plana puede no
    # existir todavia, y un recorrido puede vivir en el archivo fechado
    # <anio>/<mes>/, donde un iterdir de la raiz NO lo ve. Un "no existe" falso
    # aqui crea un recorrido duplicado con el mismo nombre.
    if store.exists(name):
        print(f"  = {name:6s} ya existe, se deja. Borralo a mano si de verdad "
              f"quieres rehacerlo (sus runs son dinero ya gastado).")
        return False
    net = dict(FOVEADA, **(overrides or {}))
    # La atadura da UN valor por valor del eje -- el propio `check_sweep` se
    # niega si las longitudes no casan, porque una diagonal desalineada entrena
    # redes que nadie pidio con una tabla igual de creible.
    couple = ({atado: {"axis": espacio_eje, "values": list(rango)}}
              if atado else None)
    try:
        spec = generate_sweep(
            name, dataset, espacio_eje, list(rango),
            base_recipe=RECIPE, base_recipe_value=receta,
            objective=OBJECTIVE, budget={"epochs": EPOCHS},
            seeds=semillas, device="cpu",
            overrides=net, border_px=BORDER_PX, couple=couple,
            study=ESTUDIO, sstore=store,
        )
    except SweepError as e:
        print(f"  ! {name:6s} {e.code}: {e.message}\n      {e.hint}")
        return False
    n = len(spec["points"])
    sem = spec["space"].get("seed", [receta["seed"]])
    print(f"  + {name:6s} {len(spec['space'][espacio_eje])} valores x semillas "
          f"{sem} = {n:3d} runs")
    print(f"      eje:    {espacio_eje} = {rango}")
    print(f"      ancla:  {', '.join(f'{k}={v}' for k, v in sorted(net.items()))}")
    print(f"      motivo: {motivo}")
    # Lo que de verdad se esta comprando, en features: es el objetivo declarado
    # del estudio, asi que sale impreso al crear y no hay que ir a buscarlo.
    if atado:
        print(f"      atado:  {atado} = {espacio_eje} (diagonal, no producto)")
    print("      cabeza:")
    for v in rango:
        campos = {espacio_eje: v} | ({atado: v} if atado else {})
        tr = network_trace(dict(net, fovea_px=16, border_px=BORDER_PX, **campos))
        etiqueta = f"{espacio_eje}={v}" + (f",{atado}={v}" if atado else "")
        print(f"        {etiqueta:>22}: {tr['flat_features']:>6} features -> "
              f"{tr['num_params']:>7} params")
    for d in spec.get("discarded") or []:
        print(f"      descartado {d['point']}: {d['problems'][0]['message']}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eje", choices=sorted(EJES))
    ap.add_argument("--fase", choices=["tanteo", "completo"])
    ap.add_argument("--control", action="store_true",
                    help="crea el recorrido iso-features del plan §4.3. Solo "
                         "hace falta si el f1 baja (criterio R5)")
    ap.add_argument("--control-stride", type=int, default=2,
                    help="a que brazo con stride iguala el control (defecto 2)")
    ap.add_argument("--pico", type=int, default=None,
                    help="fuerza el pico del tanteo en vez de derivarlo del "
                         "disco. Solo para --fase completo, y avisa en voz alta")
    args = ap.parse_args()

    store = SweepStore()
    receta = dict(RecipeStore().get(RECIPE).as_dict())

    if args.control:
        canales = canales_iso(args.control_stride)
        if canales is None:
            raise SystemExit(
                f"No hay un `channels` entero que iguale las features de "
                f"stride {args.control_stride} a resolucion 1.\n"
                f"  Un control aproximado NO controla nada: la diferencia que "
                f"midiera seria en parte la del redondeo.\n"
                f"  -> usa --control-stride con un brazo que si divida (2 o 4).")
        print(f"CONTROL iso-features (plan §4.3). Aisla CAPACIDAD de "
              f"RESOLUCION: la misma cabeza que s={args.control_stride}, pero "
              f"con el mapa entero.")
        ok = crear(CONTROL, "s_center", [1], 2, args.dataset, receta, store,
                   f"channels={canales} a stride 1 da las mismas features que "
                   f"stride {args.control_stride} con channels=[16]x4. Si "
                   f"empatan, lo que pesaba era la capacidad y el stride es "
                   f"gratis; si el stride pierde, la resolucion importa",
                   overrides={"channels": canales})
        siguiente, prefijo, horas = CONTROL, PREFIJO_CONTROL, 6
    else:
        if not args.eje or not args.fase:
            raise SystemExit("hacen falta --eje y --fase (o --control)")
        cfg = EJES[args.eje]
        if args.fase == "tanteo":
            print(f"Fase 1 -- TANTEO de `{args.eje}`. 2 semillas: ACOTA, no "
                  f"declara (p minimo alcanzable 0,333).")
            if cfg["otro"]:
                print(f"  Ancla: `{cfg['otro']}` = 1, el VIGENTE -- no el "
                      f"ganador de la otra rama (plan §5.4).")
            else:
                print(f"  Las DOS ramas al mismo stride. Contesta lo que los "
                      f"dos simples no pueden: si los efectos SUMAN (plan §5.6).")
            ok = crear(cfg["tanteo"], cfg["espacio"], RANGO_TANTEO, 2,
                       args.dataset, receta, store, MOTIVO_TANTEO,
                       atado=cfg["couple"])
            siguiente, horas = cfg["tanteo"], 6
        else:
            if args.pico is not None:
                pico = int(args.pico)
                print(f"⚠ PICO FORZADO A MANO: {pico}. NO se ha derivado del "
                      f"tanteo, asi que el rango de abajo no lo respalda el disco.")
            else:
                pico, n, f1 = pico_del_tanteo(args.eje, store)
                print(f"Pico derivado de '{cfg['tanteo']}' ({n} runs medidos): "
                      f"{args.eje} = {pico} con f1 = {f1:.4f} de media.")
            if pico not in TABLA_PICO:
                raise SystemExit(
                    f"El pico {pico} no esta en la tabla del plan §6, que cubre "
                    f"{sorted(TABLA_PICO)}.\n"
                    f"  Eso significa que el tanteo midio otro rango del que el "
                    f"plan escribio, o que --pico trae un valor inventado. "
                    f"Improvisar aqui un rango es justo lo que el plan existe "
                    f"para impedir.\n"
                    f"  -> revisa docs/plan-strides-rama-2026-09-01.md §6 y "
                    f"RANGO_TANTEO.")
            rango, motivo = TABLA_PICO[pico]
            print(f"Fase 2 -- ESTUDIO COMPLETO de `{args.eje}`. 5 semillas: con "
                  f"5 contra 5 el p minimo alcanzable es 0,0079, o sea que SI "
                  f"puede declarar al 5 %.")
            ok = crear(cfg["completo"], cfg["espacio"], rango, 5, args.dataset,
                       receta, store, motivo, atado=cfg["couple"])
            siguiente, horas = cfg["completo"], 8
        prefijo = cfg["prefijo"]

    if ok:
        print("\nSiguiente paso (esto SI gasta -- ALQUILA MAQUINAS QUE FACTURAN):")
        print(f"  .venv/bin/python scripts/estudio_estimar.py --sweep {siguiente}")
        print(f"  scripts/desacoplar.sh .venv/bin/python scripts/estudio_flota.py \\")
        print(f"      --sweep {siguiente} --cpu E5-26 --criba 2 --git "
              f"--horas-max {horas} --prefijo {prefijo} --yes")
        print(f"\n  ⚠ El --prefijo NO es opcional: es lo unico que distingue tus "
              f"maquinas de las de otra sesion en la MISMA cuenta de Vast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
