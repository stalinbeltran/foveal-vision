#!/usr/bin/env python3
"""La red de ESTE experimento: UN kernel 3×3 y `padding=0` — SIN relleno.

⚠⚠ POR QUE ESTO VIVE AQUI Y NO EN `src/fv/models/builder.py`
   Instruccion del dueno, 2026-09-03: «estos son experimentos, nada tienen que
   ver con las redes previas, ellas seran modificadas posteriormente. Si hay que
   hacer cambios al codigo tendremos que copiarlo localmente (pero si vale la
   pena, y eso depende de nuestras pruebas en estos experimentos)».
   O sea: el codigo de produccion NO se toca para probar una idea. Se prueba
   aqui, y si el numero lo respalda, entonces se decide.

QUE CAMBIA, Y POR QUE NO CABIA EN LA CONFIG
   `builder.py:145` calcula el relleno como `pc = k_center // 2`, siempre. No es
   un dato: es una expresion. Ponerlo a 0 desde una config habria pedido un campo
   nuevo en `NETWORK_DEFAULTS`, o sea tocar produccion. Aqui se hace envolviendo
   la construccion: se pide la red normal y se le sustituyen las convoluciones
   por otras identicas con `padding=0`, copiando los pesos.

   ⚠ Se COPIAN los pesos a proposito. Construir la red dos veces daria dos
   inicializaciones distintas y el experimento dejaria de ser «lo mismo salvo el
   relleno». Comprobado en `--comprobar`: el kernel sale identico bit a bit al de
   la misma config CON relleno.

QUE PASA CON EL TAMANO
   Con k=3 y sin relleno, un mapa 20x20 sale 18x18. Con UN canal, la cabeza
   recibe 1 x 18 x 18 = 324 features (400 con relleno).

   ⚠⚠ Son MAS features que las 256 del 5x5 y las 196 del 7x7: un kernel mas
   pequeno RECORTA MENOS. Y L1 se queda en 19 parametros, tres veces menos que
   el 5x5. Dos ejes a la vez y en sentidos opuestos, como en el 5x5; el criterio,
   congelado antes de correr nada, en `../instrucciones/02-criterio.md`.
   `_infer_flat_features` hace un forward de prueba, asi que la cabeza se
   dimensiona sola y no hay nada que ajustar a mano.

    python nn/red_local.py --comprobar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))

from fv.models.builder import FoveatedRegionalNN, full_config   # noqa: E402

RED_BASE = REPO / "configs" / "networks" / "plana-20-1k3.yaml"


def sin_relleno(model: nn.Module) -> nn.Module:
    """Sustituye cada Conv2d de las ramas por una identica con `padding=0`.

    Los pesos se COPIAN, no se re-inicializan: es lo que mantiene el experimento
    comparable con su gemelo.
    """
    for rama in ("center_convs", "periph_convs"):
        convs = getattr(model, rama, None)
        if convs is None:
            continue
        for i, cv in enumerate(convs):
            nueva = nn.Conv2d(cv.in_channels, cv.out_channels, cv.kernel_size,
                              stride=cv.stride, padding=0, bias=cv.bias is not None)
            with torch.no_grad():
                nueva.weight.copy_(cv.weight)
                if cv.bias is not None:
                    nueva.bias.copy_(cv.bias)
            convs[i] = nueva
    return model


class PlanaSinPadding(FoveatedRegionalNN):
    """La red del repo, con las convoluciones sin relleno y la cabeza re-medida.

    Hereda en vez de copiar las 300 lineas del builder: lo que este experimento
    prueba es UNA cosa --un kernel en vez de dos-- y reescribir el resto seria
    abrir la puerta a que divergiera del gemelo por accidente.

    ⚠ Es la MISMA clase que la del gemelo, copiada y no importada. Es un
    experimento autocontenido (regla 1 de `experimentos/README.md`): dentro de un
    ano se tiene que poder abrir esta carpeta y cargar sus pesos sin depender de
    que la de al lado siga existiendo.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        sin_relleno(self)
        # la cabeza tiene que re-dimensionarse: 20x20 -> 14x14 con k=7
        flat = self._infer_flat_features()
        self.flat_features = flat
        self.head = nn.Linear(flat + self.n_edge, 12)


def construir(semilla: int = 1) -> PlanaSinPadding:
    cfg = full_config(yaml.safe_load(RED_BASE.read_text()))
    torch.manual_seed(semilla)
    return PlanaSinPadding(cfg)


def _comprobar() -> int:
    from fv.models.builder import build_model
    cfg = full_config(yaml.safe_load(RED_BASE.read_text()))
    torch.manual_seed(1); con = build_model(cfg)
    m = construir(1)
    cv, cvc = m.center_convs[0], con.center_convs[0]
    x = torch.randn(2, 2, 20, 20)
    print(f"  con relleno : padding {cvc.padding} · salida {tuple(cvc(x).shape[1:])} "
          f"· flat {con.flat_features} · params {sum(p.numel() for p in con.parameters())}")
    print(f"  SIN relleno : padding {cv.padding} · salida {tuple(cv(x).shape[1:])} "
          f"· flat {m.flat_features} · params {sum(p.numel() for p in m.parameters())}")
    assert cv.padding == (0, 0), "el relleno no se quito"
    assert torch.equal(cv.weight, cvc.weight), "el kernel NO es el mismo"
    assert torch.equal(cv.bias, cvc.bias)
    e = torch.zeros(2, m.n_edge)
    # ⚠ el forward devuelve (B, 4, 3) --4 esquinas x [existe, x, y]--, no (B, 12):
    #   la cabeza da 12 y el modelo las reagrupa. La primera version comprobaba
    #   (2, 12) y fallaba con razon.
    assert m(x, e).shape == (2, 4, 3), "la salida deja de ser 4 esquinas x 3"
    print("  ✓ mismo kernel bit a bit · padding 0 · la salida sigue siendo 4x3 = 12")
    print("  ⚠ y la cabeza CRECE otra vez: 324 features contra 256 del 5x5 y 196 del")
    print("    7x7. L1 se queda en 19 parametros. Dos ejes a la vez.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comprobar", action="store_true")
    p.parse_args()
    raise SystemExit(_comprobar())
