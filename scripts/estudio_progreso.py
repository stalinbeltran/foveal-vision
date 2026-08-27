#!/usr/bin/env python3
"""Como va uno o varios recorridos, leyendo el libro de a bordo que hay en disco.

Por que existe
--------------
Un estudio de estos dura horas y se opera desde el movil. Preguntar "¿como va?"
no puede obligar a entrar por SSH en 15 maquinas: la flota ya baja `metrics.jsonl`
y `status.json` de cada run en cada sonda y los commitea, asi que **la respuesta
esta en disco** y este script solo la lee. No toca la red, no toca Vast y no
cuesta nada.

Y por eso mismo sirve tambien despues de una caida: lo que diga aqui es
exactamente lo que la flota va a saltarse al relanzarla.

    python3 scripts/estudio_progreso.py --sweep bs5-L4 --sweep nl5-L4 --sweep d5-L4
    python3 scripts/estudio_progreso.py --sweep bs5-L4 --tabla

⚠ La tabla parcial (`--tabla`) es **parcial**: mezcla puntos terminados con puntos
a medias, cuyo mejor checkpoint todavia puede mejorar. Sirve para ver que el
estudio va por donde se esperaba, NO para decidir nada. El veredicto se saca al
cerrar, con `scripts/estudio_informe.py`, que aplica R1..R6.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.metrics import checkpoint_record             # noqa: E402
from fv.sweeps.runner import point_run_name          # noqa: E402
from fv.sweeps.spec import expand_points             # noqa: E402
from fv.sweeps.store import SweepStore               # noqa: E402
from fv.training.registry import RunStore            # noqa: E402



def leer_run(nombre: str, monitor: str, objetivo: str) -> dict:
    d = RunStore().path(nombre)
    fila = {"run": nombre, "estado": None, "epocas": 0, "valor": None,
            "epoca_ckpt": None, "s_epoca": None, "paro_por": None}
    st = d / "status.json"
    if st.exists():
        try:
            fila["estado"] = json.loads(st.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            fila["estado"] = "ilegible"
    m = d / "metrics.jsonl"
    if m.exists():
        try:
            recs = [json.loads(l) for l in m.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
        except (OSError, json.JSONDecodeError):
            recs = []
        fila["epocas"] = len(recs)
        segs = [r["seconds"] for r in recs if r.get("seconds")]
        if segs:
            fila["s_epoca"] = round(statistics.median(segs), 1)
        rec = checkpoint_record(recs, monitor)
        if rec is not None:
            fila["valor"] = (rec.get("val") or {}).get(objetivo)
            fila["epoca_ckpt"] = rec.get("epoch")
    s = d / "summary.json"
    if s.exists():
        try:
            sm = json.loads(s.read_text(encoding="utf-8"))
            fila["paro_por"] = ("patience" if sm.get("stopped_early")
                                else "cancelado" if sm.get("cancelled") else "tope")
        except (OSError, json.JSONDecodeError):
            pass
    return fila


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="append", required=True)
    ap.add_argument("--tabla", action="store_true",
                    help="ademas, la media por valor del eje (PARCIAL, no decide)")
    args = ap.parse_args()

    store = SweepStore()
    total_g = hechos_g = epocas_g = 0
    for nombre in args.sweep:
        if not store.exists(nombre):
            print(f"### {nombre}: no existe todavia")
            continue
        spec = store.spec(nombre)
        valid, _ = expand_points(spec, spec["base_network_value"])
        monitor = (spec.get("base_recipe_value") or {}).get("monitor", "val_loss")
        objetivo = spec.get("objective", "f1")
        # El eje puede vivir en el DATASET y no en `space` (barrido del stride de
        # extraccion: un dataset por valor, docs/barrido-stride.md 1). Entonces
        # `space` solo tiene la replica y esto imprimia "eje ?" y "media por ?",
        # que es un monitor que no sabe decir que esta mirando. El valor es el
        # mismo para todos los puntos del recorrido, asi que la tabla sale de una
        # fila por brazo -- que es justo lo que se quiere ver mientras corre.
        eje = next((k for k in spec["space"] if k != "seed"), None)
        eje_ds = spec.get("eje_dataset") or {}
        valor_del_dataset = None
        if eje is None and eje_ds.get("campo"):
            eje, valor_del_dataset = eje_ds["campo"], eje_ds.get("valor")
        eje = eje or "?"
        tope = int((spec.get("budget") or {}).get("epochs", 0) or 0)

        filas = []
        for i, p in enumerate(valid):
            f = leer_run(nombre, monitor, objetivo) if False else leer_run(
                point_run_name(nombre, i, p["overrides"]), monitor, objetivo)
            f["punto"] = p["overrides"]
            filas.append(f)
        hechos = [f for f in filas if f["estado"] in ("done", "cancelled")]
        vivos = [f for f in filas if f["estado"] == "running"]
        epocas = sum(f["epocas"] for f in filas)
        total_g += len(filas); hechos_g += len(hechos); epocas_g += epocas

        print(f"\n### {nombre}   eje {eje}   objetivo {objetivo}   tope {tope} epocas")
        print(f"    {len(hechos)}/{len(filas)} runs terminados · {len(vivos)} en marcha "
              f"· {epocas} epocas escritas")
        topados = [f for f in hechos if f["paro_por"] == "tope"]
        if topados:
            print(f"    ⚠ R1: {len(topados)} run(s) pararon POR EL TOPE, no por "
                  f"patience: {', '.join(f['run'] for f in topados[:4])}"
                  + (" …" if len(topados) > 4 else ""))
        for f in filas:
            if f["estado"] in ("done", "cancelled") or f["epocas"]:
                marca = {"done": "ok  ", "cancelled": "cnc ",
                         "running": ">>  "}.get(f["estado"], "    ")
                val = f"{f['valor']:.4f}" if f["valor"] is not None else "   -  "
                print(f"    {marca}{json.dumps(f['punto'], separators=(',', ':')):>28} "
                      f"{val}  ep {f['epocas']:3d}"
                      + (f"/ckpt {f['epoca_ckpt']:<3d}" if f["epoca_ckpt"] else "        ")
                      + (f" {f['s_epoca']:6.1f} s/ep" if f["s_epoca"] else "")
                      + (f"  ({f['paro_por']})" if f["paro_por"] else ""))

        if args.tabla:
            porvalor = {}
            for f in filas:
                if f["valor"] is None:
                    continue
                clave = json.dumps(f["punto"].get(eje, valor_del_dataset))
                porvalor.setdefault(clave, []).append(f)
            print(f"    --- media PARCIAL por {eje} (no decide nada) ---")
            for clave in sorted(porvalor, key=lambda c: json.loads(c)):
                rs = porvalor[clave]
                vs = [r["valor"] for r in rs]
                med = sum(vs) / len(vs)
                sem = (statistics.stdev(vs) / len(vs) ** 0.5) if len(vs) > 1 else 0.0
                cerrados = sum(1 for r in rs if r["estado"] in ("done", "cancelled"))
                print(f"    {eje}={clave:>8}  {med:.4f} +/- {sem:.4f}  "
                      f"n={len(vs)} ({cerrados} cerrados)")

    print(f"\nTOTAL: {hechos_g}/{total_g} runs terminados, {epocas_g} epocas escritas.")
    if hechos_g < total_g:
        print("Relanzar la flota continua por donde iba: salta los `done` y "
              "vuelve a repartir lo que falte.")
    print(f"(leido de runs/ a las {time.strftime('%H:%M:%S')}; la flota lo actualiza "
          f"en cada sonda y lo commitea)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
