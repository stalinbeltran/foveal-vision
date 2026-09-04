#!/usr/bin/env python3
"""EL REPORTE COMPARATIVO UNICO de este experimento — leido de los `metrics.jsonl`.

    ../../.venv/bin/python experimentos/2026-09-04-planas-sobre-preprocesado/nn/comparativa.py
    ...                                                                    comparativa.py --md

No vuelve a leer los ficheros a su manera: llama a `serie.leer`, el unico lector de
`metrics.jsonl` de todo `experimentos/`. Dos lectores acaban discrepando en que es
«mejor f1», y entonces las dos tablas dejan de poder compararse.

⚠⚠ LA COMPARACION VALIDA AQUI ES INTERNA: `1k3` CONTRA `1k5`
   Con k=3 y stride 2 los dos caen en 18 features y sus redes son identicas en forma,
   asi que lo UNICO que cambia entre ellos es con que kernel se preproceso -- que es
   justo la pregunta. Es la comparacion mas limpia de la serie y sale gratis.
   El `1k7` tiene 8 features y NO tiene ancla: su lectura es solo descriptiva.

⚠⚠ Y NO SE COMPARA CON LOS SIETE GEMELOS. Aquellos son planas de 1 capa, 16 canales y
   sin stride (2.511-19.656 params); estos tienen 2 capas, 2 canales, stride 2 y 286.
   Ponerlos en la misma tabla seria comparar dos cosas distintas.

⚠⚠ NO SE DECLARA NADA ANTES DE LA EPOCA 11. Medido dos veces en este proyecto: en los
   gemelos el orden a la ep. 3 sale INVERTIDO respecto al final, y en el experimento
   detenido la ventaja se dividio por 6 entre la 3 y la 11.

⚠⚠ Y SI LOS TRES ESTAN A f1 = 0, ESO NO ES UNA TABLA: ES UN COLAPSO
   La red predice «no hay esquina» en todas partes, que con un 13,2 % de esquinas
   positivas es un minimo local comodo. Se dice en voz alta en vez de imprimir ceros
   como si fueran un resultado -- que es como un preliminar se lee al reves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
COMUN = EXP.parent / "comun"
sys.path.insert(0, str(COMUN))
sys.path.insert(0, str(AQUI))

import serie                                                     # noqa: E402
from red_local import ENTRADAS                                   # noqa: E402

ESTE = EXP.name
# (etiqueta, subcarpeta, features, ancla iso-features). El ancla del 1k3 es el 1k5 y
# viceversa: no es una eleccion, es que salen con las MISMAS 18 features.
BRAZOS = [
    ("pre-1k3", "1k3", 18, "1k5"),
    ("pre-1k5", "1k5", 18, "1k3"),
    ("pre-1k7", "1k7", 8, None),
]
STOPS = (3, 11, 24, 37)
BANDA = 0.04
DECLARA_DESDE = 11


def f1_en(datos, epoca: int):
    if datos is None:
        return None
    por_ep, _b, _L = datos
    r = por_ep.get(epoca)
    return r["val"]["f1"] if r else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md", action="store_true")
    a = p.parse_args()

    datos = {sub: serie.leer(ESTE, sub) for _e, sub, _f, _an in BRAZOS}
    epocas = {d[2][-1]["epoch"] for d in datos.values() if d is not None}

    cab = ["brazo", "feat."] + [f"ép. {e}" for e in STOPS] + ["ancla", "Δ misma época"]
    sep = " | " if a.md else "  "
    if a.md:
        print("| " + " | ".join(cab) + " |")
        print("|" + "---|" * len(cab))
    for etq, sub, feat, anc in BRAZOS:
        d = datos[sub]
        cel = [f"{v:.3f}" if (v := f1_en(d, e)) is not None else "—" for e in STOPS]
        ult = max((e for e in STOPS if f1_en(d, e) is not None), default=None)
        if anc is None:
            delta = "sin ancla"
        elif ult is None:
            delta = "—"
        else:
            va, vb = f1_en(d, ult), f1_en(datos[anc], ult)
            delta = f"{va - vb:+.3f} (ép. {ult})" if vb is not None else "—"
        linea = [etq, str(feat)] + cel + [anc or "—", delta]
        print(("| " if a.md else "") + sep.join(linea) + (" |" if a.md else ""))

    print()
    if len(epocas) > 1:
        print(f"⚠⚠ LOS BRAZOS NO ESTAN EN LA MISMA EPOCA {sorted(epocas)}: esta tabla "
              f"NO es comparable. → nn/avanzar.py --hasta {max(epocas)}")
        return 1
    ep = epocas.pop() if epocas else 0

    # El colapso se mira ANTES que la epoca: un f1 = 0 en los tres no es un empate.
    vivos = [f1_en(datos[sub], ep) for _e, sub, _f, _a in BRAZOS]
    if vivos and all(v is not None and v == 0.0 for v in vivos):
        print(f"⚠⚠ COLAPSO: los tres brazos van a f1 = 0,000 en la epoca {ep}.")
        print( "   La red predice «no hay esquina» en todas partes. Con un 13,2 % de")
        print( "   esquinas positivas ese es un minimo local comodo, y con 2 canales")
        print( "   no sale de el (medido: subir a 16 canales da f1 0,544 a la ep. 3).")
        print( "   ⚠ Esto NO dice nada sobre el preproceso: dice que este regimen no")
        print( "     da para la tarea. Ver el README, § «el suelo de coste».")
        return 1

    if ep < DECLARA_DESDE:
        print(f"⚠⚠ NO SE DECLARA NADA: los brazos van por la epoca {ep} y el orden solo "
              f"se estabiliza hacia la {DECLARA_DESDE}.")
        return 0

    print(f"veredicto (banda ±{BANDA}), `1k3` contra `1k5` a la MISMA epoca — mismos "
          f"18 features, misma red:")
    a3, a5 = f1_en(datos["1k3"], ep), f1_en(datos["1k5"], ep)
    if a3 is None or a5 is None:
        return 0
    dif = a3 - a5
    que = ("gana el preproceso 3x3" if dif > BANDA else
           "gana el preproceso 5x5" if dif < -BANDA else
           "EMPATE: el tamano del kernel de preproceso no importa a este regimen")
    print(f"  {dif:+.3f} → {que}")
    print("  ⚠ el `1k7` no entra: 8 features, sin ancla. Solo lectura descriptiva.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
