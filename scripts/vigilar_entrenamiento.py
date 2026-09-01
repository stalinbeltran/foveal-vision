#!/usr/bin/env python3
"""¿Va bien el entrenamiento, y SE ESTAN GUARDANDO los pesos?

Contesta cuatro preguntas de una vez, y sale con codigo != 0 si alguna falla:

  1. ¿vive la unidad de systemd que lo vigila?
  2. ¿estan BAJANDO los pesos? (edad del .pt mas nuevo en la antesala)
  3. ¿hay una instancia de Vast viva, y tiene vigilante?
  4. ¿por que epoca va y cuanto lleva?

Por que la 2 existe y no basta con `--cada`
-------------------------------------------
Un intervalo configurado no es un fichero en disco. `entrenar_vast.py` se trae los
pesos por scp en cada sonda, y un scp que falla por red deja una linea en el log y
el entrenamiento sigue tan campante: cuando la maquina muera, lo que hay es lo que
se bajo la ultima vez que funciono. La unica comprobacion que vale es el `mtime`.

*Costo `fov-optimo-p20` entero el 2026-08-30: 69 epocas en Vast, reentrenadas.*

⚠ Y la edad se compara contra un TOPE que se pasa por argumento, no contra
`--cada`: son dos numeros distintos a proposito. `--cada` es cada cuanto se
intenta; el tope es cuanto se acepta sin noticias antes de considerarlo roto.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"
sys.path.insert(0, str(ROOT / "src"))

from fv import settings                      # noqa: E402
from fv.inference import catalogo            # noqa: E402


def _unidad_viva(unidad: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", unidad],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() == "active"
    except Exception:                        # noqa: BLE001
        return False


def _edad_unidad(unidad: str) -> float | None:
    """Segundos desde que la unidad esta activa, o None si no se sabe.

    Hace falta para la GRACIA: entre que se alquila la maquina y cae la primera
    epoca pasan minutos (alquilar, esperar al ssh, subir el payload, instalar).
    Durante ese rato NO hay pesos, y eso es lo normal, no una averia. Sin gracia
    el celador avisa en rojo en su primera vuelta -- y un aviso que salta siempre
    se deja de leer, que es lo unico que no le puede pasar a un aviso.
    """
    try:
        r = subprocess.run(
            ["systemctl", "show", "-p", "ActiveEnterTimestampMonotonic", "--value", unidad],
            capture_output=True, text=True, timeout=20)
        us = int((r.stdout or "0").strip() or 0)
        if us <= 0:
            return None
        with open("/proc/uptime") as fh:
            ahora = float(fh.read().split()[0])
        return max(0.0, ahora - us / 1e6)
    except Exception:                        # noqa: BLE001
        return None


def _instancias_vast() -> list[dict] | None:
    """Las instancias vivas, o None si NO SE PUEDE saber.

    None y no [] a proposito: 'no hay ninguna' y 'no he podido preguntar' son
    estados distintos, y confundirlos es dar por buena una factura que corre.
    Misma regla que el `NO SE` de cerrable.mjs.
    """
    try:
        sys.path.insert(0, str(LANZADOR / "scripts"))
        import vast_instance as V            # noqa: PLC0415
        return [i for i in V.instancias()
                if str(i.get("actual_status") or i.get("cur_state") or "") != "exited"]
    except Exception:                        # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="nombre del run")
    ap.add_argument("--unidad", default=None,
                    help="unidad de systemd (por defecto entrenar-<name>)")
    ap.add_argument("--max-edad", type=float, default=300.0,
                    help="segundos que se aceptan sin un peso nuevo (300 = 5 min)")
    ap.add_argument("--gracia", type=float, default=900.0,
                    help="segundos desde que arranco la unidad en los que NO "
                         "tener pesos todavia es normal (alquilar+instalar)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    unidad = args.unidad or f"entrenar-{args.name}"

    problemas: list[str] = []
    info: dict = {"run": args.name, "unidad": unidad}

    # 0. ¿YA TERMINO? Se pregunta lo PRIMERO, porque cambia el significado de
    # todo lo demas: con el run terminado, la unidad muerta y unos pesos de hace
    # horas son exactamente lo que tiene que haber. Juzgarlos como averia es la
    # otra mitad del fallo que arreglo la gracia -- un aviso que salta cuando
    # todo fue bien es un aviso que se deja de leer.
    from fv.training.registry import RunStore                  # noqa: PLC0415
    store = RunStore(settings.runs_root())
    info["status"] = (store.status(args.name) or {}).get("status")
    terminado = info["status"] in ("done", "error", "cancelled")

    # 1. el vigilante
    info["unidad_viva"] = _unidad_viva(unidad)
    if not info["unidad_viva"] and not terminado:
        problemas.append(f"la unidad '{unidad}' NO esta activa")

    # 2. ¿bajan los pesos? -- la pregunta que da nombre a este script
    edad_unidad = _edad_unidad(unidad)
    info["edad_unidad_s"] = None if edad_unidad is None else round(edad_unidad)
    arrancando = edad_unidad is not None and edad_unidad < args.gracia
    antesala = catalogo.staging_dir(args.name)
    pts = sorted(antesala.glob("*.pt")) if antesala.exists() else []
    if not pts:
        info["pesos"] = None
        if terminado:
            # ⚠ Con el run TERMINADO la antesala vacia es lo NORMAL: al promover,
            # los pesos pasan al repo de datos y la antesala se limpia. La
            # pregunta deja de ser "¿estan bajando?" y pasa a ser la unica que
            # importa entonces: "¿sobrevivio el modelo?". Se le pregunta al
            # catalogo, que es quien lo sabe.
            #
            # Preguntar por la antesala daba ROJO sobre `fov16-edge-p20`, que
            # esta aprobada y commiteada desde ayer. Un vigilante que llama
            # averia a un exito es un vigilante que se apaga.
            ck, origen = catalogo.checkpoint_de(args.name, store)
            info["pesos_definitivos"] = None if ck is None else {
                "origen": origen, "ruta": str(ck)}
            if ck is None:
                problemas.append(
                    f"el run acabo en '{info['status']}' y NO tiene pesos en "
                    f"ninguna parte (ni antesala ni catalogo): el modelo se perdio")
        elif arrancando:
            # ⚠ se DICE que se esta en gracia, no se calla: "aun no toca mirar"
            # y "mire y esta bien" no pueden leerse igual.
            info["gracia"] = (f"sin pesos todavia, pero la unidad lleva "
                              f"{edad_unidad / 60:.1f} min de los "
                              f"{args.gracia / 60:.0f} de gracia")
        else:
            problemas.append(f"NO hay ningun .pt en la antesala ({antesala})")
    else:
        edad = min(time.time() - p.stat().st_mtime for p in pts)
        info["pesos"] = {
            "dir": str(antesala),
            "ficheros": {p.name: round(p.stat().st_size / 1e6, 2) for p in pts},
            "edad_s": round(edad, 1),
            "max_edad_s": args.max_edad,
        }
        if edad > args.max_edad and not terminado:
            problemas.append(
                f"el peso mas nuevo tiene {edad / 60:.1f} min "
                f"(tope {args.max_edad / 60:.0f} min): NO se esta guardando")

    # 3. la factura
    vivas = _instancias_vast()
    if vivas is None:
        info["vast"] = "NO SE"
        problemas.append("no pude preguntar a Vast (¿token? ¿red?): NO se que hay alquilado")
    else:
        info["vast"] = [{"id": i.get("id"), "label": i.get("label"),
                         "usd_h": i.get("dph_total")} for i in vivas]
        if vivas and not info["unidad_viva"]:
            problemas.append(
                f"{len(vivas)} instancia(s) VIVAS y sin vigilante: siguen facturando. "
                f"Recogela: python3 scripts/adoptar_vast.py --iid "
                f"{vivas[0].get('id')} --name {args.name}")

    # 4. por donde va
    # RunStore y no runs_root(): resuelve el mes en que se creo el run
    runs = store.path(args.name)
    mj = runs / "metrics.jsonl"
    if mj.exists():
        filas = [json.loads(l) for l in mj.read_text().splitlines() if l.strip()]
        if filas:
            u = filas[-1]
            info["epocas"] = len(filas)
            info["ultima"] = {"epoch": u.get("epoch"),
                              "val_loss": (u.get("val") or {}).get("loss"),
                              "f1": (u.get("val") or {}).get("f1")}
    if args.json:
        print(json.dumps({**info, "problemas": problemas}, indent=1, ensure_ascii=False))
    else:
        icono = "🔴" if problemas else ("✅" if terminado else "🟢")
        print(f"{icono} {args.name} · "
              + (f"TERMINADO ({info['status']})" if terminado
                 else ("unidad viva" if info["unidad_viva"] else "unidad MUERTA"))
              + (f" · epoca {info['ultima']['epoch']}"
                 f" f1={info['ultima']['f1']:.3f}" if info.get("ultima") else "")
              + (f" · pesos hace {info['pesos']['edad_s'] / 60:.1f} min"
                 if info.get("pesos")
                 else (" · arrancando (sin pesos aun)" if info.get("gracia")
                       else (f" · pesos en el {info['pesos_definitivos']['origen']}"
                             if info.get("pesos_definitivos") else " · SIN PESOS"))))
        if info.get("gracia"):
            print(f"    · {info['gracia']}")
        if isinstance(info.get("vast"), list):
            for i in info["vast"]:
                print(f"    vast {i['id']} ({i['label']}) {i['usd_h']} $/h")
        elif info.get("vast") == "NO SE":
            print("    vast: NO SE")
        for p in problemas:
            print(f"    ⚠ {p}")
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
