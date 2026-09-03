#!/usr/bin/env python3
"""La red de este experimento, AUTOCONTENIDA: no importa `fv` ni nada del repo.

Esa es toda su razón de ser. La copia viva vive en `foveal-vision/src/fv/probe/`
y va a seguir cambiando; ésta se congela con los pesos que produjo, para que
dentro de un año se pueda cargar un `.pt` sin depender de que el repo siga
teniendo la misma forma.

    python nn/modelo.py                      # comprueba los 8 pesos
    python nn/modelo.py nn/pesos/k9-K16-lcal-s1.pt

LA ESTRUCTURA, en una línea:

    x (1,20,20) → Conv2d(1→K, k×k, s=1, pad=k//2, replicate, bias) → ReLU
                → z (K,20,20)                     ← el código, el entregable
                → ConvTranspose2d(K→1, k×k, s=1, pad=k//2, SIN bias)
                → x̂ (1,20,20)

No hay nada entre codificador y decodificador: ni batchnorm, ni pooling, ni
cabeza. **El modelo SON los kernels.** El decodificador es lineal y sin sesgo a
propósito: si pudiera compensar un código malo, la presión sobre los kernels
desaparecería, que es justo el fallo que este experimento quiere evitar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SondaL1(nn.Module):
    """Autoencoder convolucional de UNA capa por lado. `K` canales, kernel `k×k`."""

    def __init__(self, channels: int, k: int) -> None:
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"k tiene que ser impar para que padding=k//2 conserve el tamaño (k={k})")
        self.K, self.k = channels, k
        # replicate = el mismo relleno que `pad_mode: edge` de producción
        self.enc = nn.Conv2d(1, channels, k, stride=1, padding=k // 2,
                             padding_mode="replicate", bias=True)
        # ⚠ sin sesgo, y torch NO admite padding_mode aquí: el anillo exterior de
        #   k//2 px se reconstruye viendo ceros. Por eso el error se reporta
        #   también sobre el interior (`r2_rec_int` en los summary.json).
        self.dec = nn.ConvTranspose2d(channels, 1, k, stride=1, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.enc(x))
        return self.dec(z), z

    @torch.no_grad()
    def renormalize(self) -> None:
        """Cada átomo del decodificador a norma L2 = 1. Se llamaba tras CADA paso.

        Sin esto el modelo tiene una salida degenerada gratis: multiplicar el
        codificador por 0,01 y el decodificador por 100 da la misma
        reconstrucción con la penalización cien veces menor, o sea que
        aprendería a hacer `z` PEQUEÑO en vez de DISPERSO.
        """
        w = self.dec.weight
        n = w.flatten(1).norm(dim=1).clamp_min(1e-8)
        w.div_(n.view(-1, 1, 1, 1))


def cargar(ruta: str | Path) -> SondaL1:
    """Reconstruye la red desde un checkpoint y devuelve el modelo en eval()."""
    e = torch.load(Path(ruta), map_location="cpu", weights_only=True)
    m = SondaL1(e["K"], e["k"])
    m.load_state_dict(e["state_dict"])
    m.eval()
    return m


def _comprobar(ruta: Path) -> None:
    import numpy as np
    m = cargar(ruta)
    n = ruta.stem
    x = torch.zeros(2, 1, 20, 20)
    xh, z = m(x)
    assert z.shape == (2, m.K, 20, 20) and xh.shape == (2, 1, 20, 20)
    npy = ruta.parents[2] / "resultados" / n / "kernels_enc.npy"
    ok = ""
    if npy.exists():
        guardados = torch.from_numpy(np.load(npy)).flatten(1)
        vivos = m.enc.weight.detach().flatten(1)
        assert torch.allclose(guardados, vivos, atol=1e-6), f"{n}: los .npy no casan con el .pt"
        ok = " · kernels .npy idénticos"
    print(f"  {n:<18} K={m.K:<3} k={m.k}  "
          f"{sum(p.numel() for p in m.parameters()):>5} parámetros  ✓ carga{ok}")


if __name__ == "__main__":
    aqui = Path(__file__).resolve().parent
    rutas = [Path(a) for a in sys.argv[1:]] or sorted((aqui / "pesos").glob("*.pt"))
    print(f"comprobando {len(rutas)} peso(s):")
    for r in rutas:
        _comprobar(r.resolve())
    print("todos cargan y coinciden con sus kernels guardados.")
