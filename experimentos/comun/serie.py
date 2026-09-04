"""La serie de CNN planas, puesta en una tabla — leida de los `metrics.jsonl`.

Existe para no transcribir a mano: los cinco experimentos comparten stops (0, 3,
11, 24, 37 epocas), y sus tablas de README se copiaban numero a numero. Una
transcripcion mal hecha en un README es indistinguible de una medida.

    ../../.venv/bin/python experimentos/comun/serie.py
    ../../.venv/bin/python experimentos/comun/serie.py --md      # en markdown

⚠ Lee `nn/pesos/metrics.jsonl` de cada experimento, o sea LOS PESOS COMMITEADOS,
no el run vivo del repo de datos. Asi la tabla describe lo que hay en git, que es
lo unico que sobrevive a rehacer la maquina.
"""
import argparse, json, pathlib

RAIZ = pathlib.Path(__file__).resolve().parents[1]

# (etiqueta, carpeta, features a la cabeza). El orden es el de la serie.
SERIE = [
    ("4k7 zeros",      "2026-09-03-cnn-plana-4k7",            1600),
    ("4k7 replicate",  "2026-09-03-cnn-plana-4k7-replicate",  1600),
    ("2k7 zeros",      "2026-09-03-cnn-plana-2k7",             800),
    ("2k7 sin relleno","2026-09-03-cnn-plana-2k7-sinpadding",  392),
    ("1k7 sin relleno","2026-09-04-cnn-plana-1k7-sinpadding",  196),
]
STOPS = (1, 3, 11, 24, 37)


def leer(carpeta):
    p = RAIZ / carpeta / "nn" / "pesos" / "metrics.jsonl"
    if not p.exists():
        return None
    L = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {r["epoch"]: r for r in L}, min(L, key=lambda r: r["val"]["loss"]), L


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--md", action="store_true", help="tabla en markdown")
    a = a.parse_args()

    filas = []
    for etq, carp, feat in SERIE:
        d = leer(carp)
        if d is None:
            filas.append((etq, feat, None, None, None))
            continue
        por_ep, best, L = d
        filas.append((etq, feat, por_ep, best, L[-1]["epoch"]))

    sep = " | " if a.md else "  "
    # ⚠ La Δ va contra el DOBLE de features, no contra la fila de arriba. `4k7
    #   replicate` tiene los mismos 1.600 que `4k7 zeros`: es el control del
    #   relleno, no una mitad, y restarle a la fila anterior mezclaria los dos
    #   ejes justo en la columna que existe para separarlos.
    mejor_por_feat = {}
    for etq, feat, pe, b, _u in filas:
        if pe is not None and feat not in mejor_por_feat:
            mejor_por_feat[feat] = b["val"]["f1"]

    cab = ["red", "features"] + [f"ép. {e}" for e in STOPS] + ["mejor f1", "época", "Δ vs. el doble"]
    if a.md:
        print("| " + " | ".join(cab) + " |")
        print("|" + "---|" * len(cab))
    for etq, feat, por_ep, best, ult in filas:
        if por_ep is None:
            print(("| " if a.md else "") + sep.join([etq, str(feat)] + ["—"] * (len(STOPS) + 3))
                  + (" |" if a.md else "") + "   (sin correr)")
            continue
        cel = [f"{por_ep[e]['val']['f1']:.3f}" if e in por_ep else "—" for e in STOPS]
        mejor = best["val"]["f1"]
        # "el doble" con holgura del 10 %: 800 -> 392 NO es una mitad exacta
        # (es 2,04x, porque el mapa pasa de 20x20 a 14x14 mientras los canales
        # siguen en 2). 392 -> 196 SI lo es. Exigir el doble exacto dejaria la
        # columna vacia justo en la comparacion que mas se usa.
        doble = next((v for f, v in mejor_por_feat.items()
                      if 1.8 <= f / feat <= 2.2), None)
        d = "—" if doble is None else f"{mejor - doble:+.3f}"
        linea = sep.join([etq, str(feat)] + cel + [f"**{mejor:.3f}**" if a.md else f"{mejor:.3f}",
                                                   str(best["epoch"]), d])
        print(("| " + linea + " |") if a.md else linea)

    if not a.md:
        print("\nΔ es contra la red con el DOBLE de features, no contra la fila de arriba:")
        print("`4k7 replicate` tiene los mismos 1.600 que `4k7 zeros` --es el control del")
        print("relleno-- y compararlo con la fila anterior mezclaria los dos ejes.")


if __name__ == "__main__":
    main()
