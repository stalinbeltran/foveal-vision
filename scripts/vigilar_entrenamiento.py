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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    unidad = args.unidad or f"entrenar-{args.name}"

    problemas: list[str] = []
    info: dict = {"run": args.name, "unidad": unidad}

    # 1. el vigilante
    info["unidad_viva"] = _unidad_viva(unidad)
    if not info["unidad_viva"]:
        problemas.append(f"la unidad '{unidad}' NO esta activa")

    # 2. ¿bajan los pesos? -- la pregunta que da nombre a este script
    antesala = catalogo.staging_dir(args.name)
    pts = sorted(antesala.glob("*.pt")) if antesala.exists() else []
    if not pts:
        info["pesos"] = None
        problemas.append(f"NO hay ningun .pt en la antesala ({antesala})")
    else:
        edad = min(time.time() - p.stat().st_mtime for p in pts)
        info["pesos"] = {
            "dir": str(antesala),
            "ficheros": {p.name: round(p.stat().st_size / 1e6, 2) for p in pts},
            "edad_s": round(edad, 1),
            "max_edad_s": args.max_edad,
        }
        if edad > args.max_edad:
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
    from fv.training.registry import RunStore          # noqa: PLC0415
    runs = RunStore(settings.runs_root()).path(args.name)
    mj = runs / "metrics.jsonl"
    if mj.exists():
        filas = [json.loads(l) for l in mj.read_text().splitlines() if l.strip()]
        if filas:
            u = filas[-1]
            info["epocas"] = len(filas)
            info["ultima"] = {"epoch": u.get("epoch"),
                              "val_loss": (u.get("val") or {}).get("loss"),
                              "f1": (u.get("val") or {}).get("f1")}
    st = runs / "status.json"
    if st.exists():
        try:
            info["status"] = json.loads(st.read_text()).get("status")
        except json.JSONDecodeError:
            info["status"] = None

    if args.json:
        print(json.dumps({**info, "problemas": problemas}, indent=1, ensure_ascii=False))
    else:
        icono = "🔴" if problemas else "🟢"
        print(f"{icono} {args.name} · unidad {'viva' if info['unidad_viva'] else 'MUERTA'}"
              + (f" · epoca {info['ultima']['epoch']}"
                 f" f1={info['ultima']['f1']:.3f}" if info.get("ultima") else "")
              + (f" · pesos hace {info['pesos']['edad_s'] / 60:.1f} min"
                 if info.get("pesos") else " · SIN PESOS"))
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
