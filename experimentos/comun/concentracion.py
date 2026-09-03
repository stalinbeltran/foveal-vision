"""¿Cuánto se reparten el trabajo los kernels? — medido DOS veces, a propósito.

`resumen.json` guarda `abs_media` sobre el **mapa entero**, y ese número está
dominado por el anillo del borde: en `plana-4k7` el anillo es 9,4x el interior,
así que «qué kernel responde más» acaba midiendo «qué kernel hace más anillo».
Medido el 2026-09-03: el run con `replicate` —que tiene MENOS anillo— sale con
un ratio de 16,6x sobre el mapa entero y 3,4x sobre el interior.

Aquí se recalcula sobre el **interior** (fuera del anillo de k//2) y **restando
el nivel** de cada mapa, que es la misma receta con la que se midió el anillo en
`../2026-09-03-cnn-plana-4k7/nn/porque_el_anillo.py`. No reescribe ningún
`resumen.json`: los stops son artefactos congelados, y esto lee su `mapas.npy`.

    ../../.venv/bin/python experimentos/comun/concentracion.py
"""
import argparse, pathlib, sys
import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[1]
BORDE = 3          # k//2 con k=7: el anillo que toca el relleno


def medir(mapas, borde=BORDE):
    """Devuelve (por_kernel_mapa_entero, por_kernel_interior) para un stop."""
    m = np.load(mapas)                       # (n_entradas, n_kernels, H, W), con signo
    c = slice(borde, -borde)
    entero = np.abs(m).mean(axis=(0, 2, 3))
    nivel = np.median(m[..., c, c], axis=(2, 3), keepdims=True)
    interior = np.abs((m - nivel)[..., c, c]).mean(axis=(0, 2, 3))
    return entero, interior


def ratio(v):
    return float(v.max() / v.min()) if v.min() > 0 else float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", action="append", default=None,
                   help="carpeta del experimento (repetible). Por defecto, todas las que tengan evaluacion/")
    a = p.parse_args()

    exps = [RAIZ / e for e in a.exp] if a.exp else sorted(
        d for d in RAIZ.iterdir() if (d / "evaluacion").is_dir())
    if not exps:
        sys.exit("no hay ningun experimento con evaluacion/")

    for exp in exps:
        print(f"\n== {exp.name}")
        print(f"   {'stop':<22} {'MAPA ENTERO (lo de resumen.json)':<40} {'INTERIOR, sin el nivel':<40}")
        for stop in sorted((exp / "evaluacion").iterdir()):
            mapas = stop / "mapas.npy"
            if not mapas.exists():
                continue
            ent, inte = medir(mapas)
            f = lambda v: " ".join(f"{x:.3f}" for x in v)
            print(f"   {stop.name:<22} {f(ent):<28} {ratio(ent):5.1f}x   {f(inte):<28} {ratio(inte):5.1f}x")


if __name__ == "__main__":
    main()
