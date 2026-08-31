#!/usr/bin/env python3
"""Adopta una instancia de Vast HUERFANA: vigila, se trae los pesos y la destruye.

Por que existe
--------------
`entrenar_vast.py` destruye la instancia en un `finally` de SU proceso, y su
propio docstring ya avisaba del agujero: *"Si el droplet de control muere de
golpe, el `finally` no corre y la instancia sigue facturando"*. Hasta el
2026-08-31 eso era una advertencia sin salida: si el vigilante moria, no habia
forma de volver a engancharse -- `--continuar` **alquila otra maquina** y ademas
exige un `last.pt` local que un vigilante muerto nunca llego a bajar.

PASO DE VERDAD ese dia. El entrenamiento `fov16-edge-p20` seguia corriendo en la
maquina (1 h 38 min de reloj cuando se descubrio) y el proceso local habia
desaparecido: el trabajo intacto, el que lo recoge muerto, y la factura corriendo.

⚠ POR QUE murio, que es la leccion y no el sintoma
---------------------------------------------------
Se lanzo con `telegram-coordinator/scripts/desacoplar.sh`, que usa
`systemd-run --scope`. Eso da **cgroup propio** --y por tanto sobrevive a un
`systemctl restart` del coordinador, que es para lo que se escribio-- pero el
proceso **sigue siendo hijo del que lo lanzo**. Un tree-kill al padre (al cerrar
la sesion que lo arranco) se lo lleva.

O sea: "sobrevive" llevaba complemento, y era otro. Contra el restart del
servicio, si; contra la muerte de su padre, no. Lo que si sobrevive a las dos
cosas es una **unidad transitoria** (`systemd-run` SIN `--scope`), cuyo padre es
PID 1. Por eso este script se lanza asi -- ver el bloque de uso.

Que hace
--------
Sondea la maquina; mientras `fv-train` viva se trae los pesos a la antesala (lo
mismo que hace `entrenar_vast`, con la misma funcion); cuando muere, promueve y
**destruye la instancia en un `finally`**.

    python3 scripts/adoptar_vast.py --iid 49406152 --name fov16-edge-p20

⚠ Y este script tiene EL MISMO agujero que el que rescata: si el muere, la
instancia se queda. No se arregla desde aqui --destruirse a si misma pediria el
token de Vast dentro de la maquina alquilada, y ahi no viaja ningun secreto a
proposito. Lo que hay es lo de siempre: el comando de destruccion impreso antes
de nada, `--horas-max` como tope, y `cerrable.mjs`, que desde el 2026-08-31
cuenta tambien este proceso.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Los estados en que un run YA no entrena. Mismo vocabulario que el resto del
# proyecto (`fv.training.registry`), no una lista propia que pueda divergir.
TERMINALES = {"done", "error", "cancelled", "interrupted"}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "digital-ocean-dropplet-auto-launching" / "scripts"))


def _cargar(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _cargar("entrenar_vast", ROOT / "scripts" / "entrenar_vast.py")
V = EV.V

from fv.inference import catalogo          # noqa: E402
from fv.training.registry import RunStore   # noqa: E402


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _morir_con_finally(sig, _frame):
    """SIGTERM/SIGINT -> excepcion, para que el `finally` LLEGUE A CORRER.

    ⚠⚠ Sin esto, `systemctl stop` de este vigilante mataba el proceso sin
    destruir la instancia -- y "paro el vigilante" es justo lo que uno hace
    cuando quiere terminar. Medido el 2026-08-31: se paro la unidad y la maquina
    siguio facturando; hubo que destruirla a mano.

    El `finally` de Python cubre EXCEPCIONES, no senales: SIGTERM termina el
    proceso sin desenrollar la pila. `SystemExit` si la desenrolla, asi que
    convertir la senal en excepcion es lo que hace que el unico sitio donde se
    destruye la instancia se ejecute SIEMPRE que este proceso acabe por su
    cuenta.

    (Contra SIGKILL no hay nada que hacer, y por eso ademas estan el comando
    impreso al principio y el aviso de `cerrable.mjs`.)
    """
    log(f"\n  recibida senal {signal.Signals(sig).name}: recojo y destruyo antes de salir")
    raise SystemExit(143 if sig == signal.SIGTERM else 130)


def vive_el_entrenamiento(destino: Path) -> bool:
    """¿Sigue entrenando? Se lo pregunta al ARTEFACTO, no al sistema operativo.

    `status.json` lo escribe el propio bucle de entrenamiento (`RunStore`) y sus
    estados terminales estan declarados en un sitio. Se lee del fichero que la
    sonda acaba de bajar, asi que no cuesta ni una conexion mas.

    ⚠⚠ ANTES ESTO PREGUNTABA `pgrep -f 'fv-train'` POR SSH, Y NO FUNCIONABA
    NUNCA. Medido el 2026-08-31: el entrenamiento habia terminado --`pgrep -af
    'fv-train'` en la maquina no devolvia NADA-- y la comprobacion seguia
    diciendo VIVO. El motivo es que `pgrep -f` casa contra la linea de comando
    COMPLETA de cada proceso, y el shell que ssh abre para ejecutar el sondeo
    lleva la cadena 'fv-train' en la suya: se encontraba a si mismo.

    Consecuencia: el vigilante sondeo UNA HORA un run terminado y habria seguido
    hasta `--horas-max` (6 h), facturando 0,05 $/h por nada. El error que se
    penso al escribir esto fue el falso NEGATIVO (destruir con el trabajo
    dentro), y se cubrio; el falso POSITIVO --no enterarse nunca de que
    termino-- se quedo sin cubrir, y es el que paso.

    Es exactamente la trampa que `cerrable.mjs` ya tenia documentada y resuelta
    ("un shell cuya linea MENCIONA el nombre de un trabajo casa igual que el
    trabajo"), repetida aqui por no mirarla. Preguntar por el artefacto la quita
    de raiz: `status.json` no puede confundirse consigo mismo.

    ⚠ Y la duda sigue resolviendose hacia SEGUIR VIGILANDO: si el fichero no
    esta o no se puede leer, se devuelve True. Un sondeo de mas cuesta nada;
    destruir de mas cuesta el entrenamiento entero.
    """
    p = destino / "status.json"
    try:
        estado = json.loads(p.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError) as e:
        log(f"  (sondeo: no puedo leer status.json ({e}); NO lo tomo por terminado)")
        return True
    if estado in TERMINALES:
        log(f"  status.json dice '{estado}'")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Adopta una instancia de Vast huerfana")
    ap.add_argument("--iid", required=True, help="id de la instancia ya alquilada")
    ap.add_argument("--name", required=True, help="nombre del run que entrena dentro")
    ap.add_argument("--cada", type=float, default=120, help="segundos entre sondeos")
    ap.add_argument("--horas-max", type=float, default=8.0)
    ap.add_argument("--sin-promover", action="store_true")
    ap.add_argument("--no-destruir", action="store_true",
                    help="deja la instancia viva al terminar (para depurar)")
    args = ap.parse_args()

    # antes de nada: que una senal pase por el `finally` que destruye
    signal.signal(signal.SIGTERM, _morir_con_finally)
    signal.signal(signal.SIGINT, _morir_con_finally)

    V.load_env()
    inst = V.buscar_instancia(str(args.iid))
    if not inst:
        log(f"no existe la instancia {args.iid}")
        return 2
    host, port = V.ssh_destino(inst)

    # Lo primero de todo, antes de cualquier espera: como se corta la factura a
    # mano si esto tambien muere.
    log(f"adoptando instancia {args.iid} ({host}:{port}) para el run '{args.name}'")
    log(f"  si esto muere, la factura se corta con:\n"
        f"    python3 {ROOT.parent}/digital-ocean-dropplet-auto-launching/scripts/"
        f"vast_instance.py destroy {args.iid} --yes")

    store = RunStore()
    destino = store.path(args.name) if store.exists(args.name) else store.destino(args.name)
    destino.mkdir(parents=True, exist_ok=True)
    rdir = EV.dir_remoto(host, port, args.name)
    if not rdir:
        log("  ⚠ no encuentro el directorio del run EN la maquina; sigo sondeando")

    t_fin = time.time() + args.horas_max * 3600
    try:
        while True:
            rdir = rdir or EV.dir_remoto(host, port, args.name)
            traidos = EV.traer(host, port, args.name, destino, rdir)
            log(f"  {time.strftime('%H:%M:%S')} sonda: {traidos or 'nada todavia'}")
            if not vive_el_entrenamiento(destino):
                log("  el entrenamiento termino en la maquina")
                break
            if time.time() > t_fin:
                log(f"  ⚠ tope de {args.horas_max} h: dejo de esperar y recojo lo que haya")
                break
            time.sleep(args.cada)

        # una ultima recogida: la epoca final se escribe DESPUES del ultimo sondeo
        EV.traer(host, port, args.name, destino, rdir or EV.dir_remoto(host, port, args.name))

    except SystemExit:
        # cortado por senal: los pesos que haya en la antesala se quedan ahi (no
        # se promueve algo a medias), pero el `finally` de abajo SI destruye.
        log(f"  lo bajado esta en {catalogo.staging_dir(args.name)}; no promuevo "
            f"un run cortado a mano")
        raise
    else:
        if args.sin_promover:
            log(f"\n  pesos en la antesala: {catalogo.staging_dir(args.name)}")
        else:
            try:
                r = catalogo.promover(args.name, store,
                                      motivo="entrenada en Vast (instancia adoptada)",
                                      origen="adoptar_vast.py")
                log(f"\n  pesos promovidos: {r['copiados']} -> {r['destino']}")
                log(f"  FALTA EMPUJARLO, o se pierde con la maquina:\n    {r['commit']}")
            except catalogo.CatalogoError as e:
                log(f"\n  ⚠ no se pudieron promover [{e.code}]: {e.message}\n     {e.hint}")
    finally:
        if args.no_destruir:
            log(f"\n  ⚠ --no-destruir: la instancia {args.iid} SIGUE FACTURANDO")
        else:
            log(f"\n  destruyendo instancia {args.iid}...")
            try:
                V.destruir(args.iid)
                log("  destruida")
            except Exception as e:
                log(f"  ⚠⚠ NO se pudo destruir: {e}\n"
                    f"     HAZLO A MANO: vast_instance.py destroy {args.iid} --yes")

    try:
        EV.avisar(f"run {args.name}: recogido y la instancia {args.iid} destruida")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
