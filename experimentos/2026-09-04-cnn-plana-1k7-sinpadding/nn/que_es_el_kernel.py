#!/usr/bin/env python3
"""¿QUE es el unico kernel de esta red? — la pregunta que 1 kernel deja hacer.

Con 4 o 2 kernels, la forma de cada uno se puede explicar por el reparto: uno
hace bordes porque otro hace manchas. Con UNO no hay reparto que invocar, asi que
su forma es directamente «lo que le hace falta a la tarea».

QUE MIDE, Y CONTRA QUE
  1. El reparto de energia entre los DOS canales de entrada (la vista y el
     relleno). Si el kernel gasta su energia en el canal de relleno, esta
     mirando «donde se acaba la pagina», no el texto.
  2. La componente DC (la suma). Es lo que produce el NIVEL constante de cada
     mapa, que esta medido que es ~8x la estructura del texto en las redes con
     relleno de ceros.
  3. La energia en el subespacio clasico 6-D (DC, Sobel-x, Sobel-y, laplaciano y
     las dos diagonales), CON SU NULO. Es la misma vara de medir de la sonda L1
     --se importa de `fv.probe.metrics`, no se copia-- y su nulo es `6/k^2`, o
     sea 0,1224 con k=7. Un cociente de 1,0 es indistinguible del azar.

⚠ EL NULO SE IMPRIME AL LADO, siempre. La leccion del reporte #22 fue justo esa:
  una metrica sin su nulo delante se lee como grande cuando no lo es. Y ademas se
  saca un nulo EMPIRICO (kernels aleatorios con la misma inicializacion), porque
  el 6/k^2 teorico supone una gaussiana isotropa y la init de PyTorch no lo es.

    python nn/que_es_el_kernel.py                    # el entrenado
    python nn/que_es_el_kernel.py --sin-entrenar     # el de partida, para contrastar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.probe.metrics import classic_basis          # noqa: E402
from red_local import construir                     # noqa: E402

PESOS = EXP / "nn" / "pesos"
CANALES = ("la vista", "el relleno")
N_NULO = 2000


def energia_6d(plano: torch.Tensor) -> float:
    """Fraccion de la energia del plano k x k que cae en el subespacio clasico."""
    k = plano.shape[-1]
    B = classic_basis(k).to(torch.float64)           # (6, k*k), ortonormal
    v = plano.reshape(-1).to(torch.float64)
    n = float(v @ v)
    if n == 0:
        return float("nan")
    return float(((B @ v) ** 2).sum() / n)


def nulo_empirico(k: int, canales: int, n: int = N_NULO) -> tuple[float, float]:
    """Media y p95 de `energia_6d` sobre kernels recien inicializados."""
    g = torch.Generator().manual_seed(20260904)
    vals = []
    for _ in range(n):
        w = torch.empty(canales, k, k)
        torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
        vals += [energia_6d(w[c]) for c in range(canales)]
    t = torch.tensor(vals)
    return float(t.mean()), float(t.quantile(0.95))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-entrenar", action="store_true")
    a = p.parse_args()

    m = construir(1)
    if a.sin_entrenar:
        etq = "sin entrenar (semilla 1)"
    else:
        ck = torch.load(PESOS / "best.pt", map_location="cpu", weights_only=False)
        estado = ck.get("model", ck.get("state_dict", ck))
        m.load_state_dict(estado)
        etq = f"best.pt · epoca {ck.get('epoch', '?')}"

    conv = m.center_convs[0]
    w = conv.weight.detach()[0]                      # (2, 7, 7): el UNICO kernel
    sesgo = float(conv.bias.detach()[0])
    k = w.shape[-1]
    e = (w ** 2).sum(dim=(1, 2))
    total = float(e.sum())

    print(f"\nEL UNICO KERNEL  ·  {etq}   (forma {tuple(m.center_convs[0].weight.shape)})")
    print(f"  norma L2 total: {total ** 0.5:.4f}   ·   sesgo: {sesgo:+.4f}\n")

    nulo_t = 6 / k ** 2
    nulo_m, nulo_p95 = nulo_empirico(k, w.shape[0])
    print(f"  {'canal':<12} {'energia':>8} {'DC (suma)':>11} {'|DC|/norma':>11} {'6-D':>7} {'/nulo':>7}")
    for c in range(w.shape[0]):
        frac = float(e[c]) / total if total else float("nan")
        dc = float(w[c].sum())
        nrm = float(w[c].norm())
        seis = energia_6d(w[c])
        print(f"  {CANALES[c]:<12} {frac:>7.1%} {dc:>+11.4f} {abs(dc) / max(nrm, 1e-9):>11.3f}"
              f" {seis:>7.3f} {seis / nulo_m:>6.2f}x")
    print(f"\n  nulo del 6-D: teorico 6/k² = {nulo_t:.4f} · EMPIRICO (n={N_NULO*w.shape[0]}) "
          f"media {nulo_m:.4f}, p95 {nulo_p95:.4f}")
    print(f"  ⚠ se compara contra el EMPIRICO: la init de PyTorch no es gaussiana isotropa.")
    print(f"  ⚠ un cociente ~1,0x es indistinguible del azar; pasar el p95 ({nulo_p95 / nulo_m:.2f}x) "
          f"es la barra del 5 %.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
