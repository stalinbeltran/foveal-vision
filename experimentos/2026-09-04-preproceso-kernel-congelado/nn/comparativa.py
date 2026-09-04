#!/usr/bin/env python3
"""EL REPORTE COMPARATIVO ÚNICO de este experimento — leído de los `metrics.jsonl`.

    ../../.venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/comparativa.py
    ...                                                              comparativa.py --md

Existe para no transcribir a mano: transcribir un número a un README lo vuelve
indistinguible de una medida. Es el mismo motivo por el que existe
`comun/serie.py`, y por eso **este script no vuelve a leer los ficheros a su
manera**: llama a `serie.leer`, que es el único lector de `metrics.jsonl` de todo
`experimentos/`. Dos lectores acaban discrepando en qué es «mejor f1», y entonces
las dos tablas dejan de poder compararse — que es justo lo que este experimento
necesita hacer.

⚠⚠ LO QUE ESTA TABLA COMPARA, Y LO QUE NO
   Los brazos NO se comparan contra la referencia cruda a secas: sin relleno cada
   preproceso recorta, así que los cuatro llegan a la cabeza con anchos distintos
   (324 · 256 · 196 · 144) y en este régimen **manda el tamaño de la cabeza** (la
   cabeza es el 97-99 % de los parámetros entrenables). Comparar `pre-1k7` (144)
   con la referencia (324) mediría sobre todo que una cabeza es 2,25× la otra.

   La comparación válida es contra el **ancla iso-features**: un gemelo YA CORRIDO
   con exactamente el mismo número de features, el mismo dataset, la misma semilla
   y la misma receta. Sale a coste cero porque ya está pagado y en git.

⚠⚠ Y NO SE DECLARA NADA ANTES DE LA ÉPOCA 11. Está medido en esta misma serie que
   el orden a la época 3 sale INVERTIDO respecto al final: `1k3` es el peor de los
   tres a la ép. 3 (0,099) y el mejor a la 37 (0,680). Por eso este script se
   niega a imprimir veredicto antes de la 11 y lo dice en voz alta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
COMUN = EXP.parent / "comun"
sys.path.insert(0, str(COMUN))

import serie                                                     # noqa: E402

ESTE = EXP.name

# LA TABLA (R4: el acoplamiento se declara, no se deduce). Por brazo:
#   (etiqueta, subcarpeta en nn/pesos/, features, ancla iso-features, su carpeta)
# El ancla NO es una elección: es el gemelo con EXACTAMENTE esas features.
BRAZOS = [
    ("pre-1k3", "1k3", 256, "1k5 crudo", "2026-09-04-cnn-plana-1k5-sinpadding"),
    ("pre-1k5", "1k5", 196, "1k7 crudo", "2026-09-04-cnn-plana-1k7-sinpadding"),
    ("pre-1k7", "1k7", 144, None,        None),
]
# La referencia que pidió el dueño («la plana sin ningún preproceso») YA ESTA
# CORRIDA: es este experimento. Se declara como ancla, nunca como fila del
# estudio, porque está a 37 épocas y los brazos empiezan en 3.
REFERENCIA = ("referencia cruda (1k3)", 324, "2026-09-04-cnn-plana-1k3-sinpadding")

STOPS = (3, 11, 24, 37)
BANDA = 0.04          # la misma banda de ruido que fijaron los criterios previos
DECLARA_DESDE = 11    # antes de aquí el orden está medido como no fiable


def f1_en(datos, epoca: int):
    """El f1 de esa época, o None si el run no ha llegado."""
    if datos is None:
        return None
    por_ep, _best, _L = datos
    r = por_ep.get(epoca)
    return r["val"]["f1"] if r else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md", action="store_true", help="tabla en markdown")
    a = p.parse_args()

    filas, epocas_vistas = [], set()
    for etq, sub, feat, anc_etq, anc_carp in BRAZOS:
        d = serie.leer(ESTE, sub)
        anc = serie.leer(anc_carp) if anc_carp else None
        if d is not None:
            epocas_vistas.add(d[2][-1]["epoch"])
        filas.append((etq, feat, d, anc_etq, anc))
    ref = serie.leer(REFERENCIA[2])

    cab = ["brazo", "feat."] + [f"ép. {e}" for e in STOPS] + \
          ["ancla iso-features", "Δ misma época"]
    sep = " | " if a.md else "  "
    if a.md:
        print("| " + " | ".join(cab) + " |")
        print("|" + "---|" * len(cab))

    for etq, feat, d, anc_etq, anc in filas:
        cel = []
        for e in STOPS:
            v = f1_en(d, e)
            cel.append(f"{v:.3f}" if v is not None else "—")
        # La Δ se lee SIEMPRE a la misma época en los dos lados. Compararse
        # contra el «mejor f1» del ancla (que cae en la ép. 32-36) sería
        # comparar 3 épocas contra 37.
        ultima = max((e for e in STOPS if f1_en(d, e) is not None), default=None)
        if anc is None:
            delta = "sin ancla"
        elif ultima is None:
            delta = "—"
        else:
            va, vb = f1_en(d, ultima), f1_en(anc, ultima)
            delta = f"{va - vb:+.3f} (ép. {ultima})" if vb is not None else \
                    f"el ancla no llega a la ép. {ultima}"
        linea = [etq, str(feat)] + cel + [anc_etq or "—", delta]
        print(("| " if a.md else "") + sep.join(linea) + (" |" if a.md else ""))

    # La referencia va SEPARADA y con su época dicha: es un ancla de 37 épocas.
    if ref is not None:
        v37 = f1_en(ref, 37)
        print()
        print(f"referencia cruda ({REFERENCIA[1]} feat., **37 épocas**, ya corrida "
              f"y en git): f1 {v37:.3f}" if v37 else "")

    print()
    if len(epocas_vistas) > 1:
        print(f"⚠⚠ LOS BRAZOS NO ESTAN EN LA MISMA EPOCA {sorted(epocas_vistas)}: "
              f"esta tabla NO es comparable. → nn/avanzar.py --hasta "
              f"{max(epocas_vistas)}")
        return 1
    ep = epocas_vistas.pop() if epocas_vistas else 0
    if ep < DECLARA_DESDE:
        print(f"⚠⚠ NO SE DECLARA NADA: los brazos van por la época {ep} y el orden "
              f"de esta serie sólo se estabiliza hacia la {DECLARA_DESDE}.")
        print(f"   Medido en los gemelos: `1k3` es el PEOR a la ép. 3 (0,099) y el "
              f"MEJOR a la 37 (0,680).")
        print(f"   Lo que sí se puede leer hoy: que arrancó, el reloj y la forma de "
              f"la curva. → nn/avanzar.py --hasta {DECLARA_DESDE}")
    else:
        print(f"veredicto por brazo (banda de ruido ±{BANDA}), contra su ancla "
              f"iso-features a la MISMA época:")
        for etq, feat, d, anc_etq, anc in filas:
            if anc is None or d is None:
                print(f"  {etq}: sin ancla — sólo lectura descriptiva")
                continue
            va, vb = f1_en(d, ep), f1_en(anc, ep)
            if va is None or vb is None:
                continue
            dif = va - vb
            que = ("APORTA" if dif > BANDA else
                   "ESTORBA" if dif < -BANDA else "NEUTRO")
            print(f"  {etq}: {dif:+.3f} vs {anc_etq} → el preproceso {que}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
