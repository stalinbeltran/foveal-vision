#!/usr/bin/env python3
"""El `Dataset` que sirve el `.npz` preprocesado — sin construir ninguna vista.

    python nn/dataset.py            # cuantas ventanas por split y que forma tienen

POR QUE HACE FALTA UNO PROPIO
   `fv.windows.dataset.FoveatedWindowDataset` construye la vista foveada por item, a
   partir de las imagenes y de `window_xy` (`build_view` + `input_stack`). Aqui el
   dato YA esta preprocesado y guardado, asi que no hay nada que construir: se indexa.

   Y eso no es solo comodidad. MEDIDO el 2026-09-04 en este droplet, 3.000 items:

       construir la vista foveada   383,2 us/item   ->   32,2 s por epoca de train
       leer del npz preprocesado      4,3 us/item   ->    0,4 s por epoca

   89x mas barato el dato. En los brazos del experimento detenido, 32 de los 44-46
   s/epoca eran exactamente esto.

DEVUELVE TRES COSAS, COMO EL DE PRODUCCION
   `(x, e, y)`, porque `loop.py:322` hace `for x, e, y in train_loader`. El `e` es un
   tensor VACIO --forma `(0,)`, que en lote es `(B, 0)`-- igual que hace
   `FoveatedWindowDataset` con `edge_inputs='off'`: concatenar nada a la cabeza es la
   identidad, asi que no hay caso especial en ningun sitio.
   ⚠ Vacio y no ausente: una tupla de 2 elementos romperia el desempaquetado del bucle.

⚠ EL `.npz` SE CARGA UNA VEZ Y SE COMPARTE ENTRE TRAIN Y VAL
   `_entrenar` construye DOS datasets (split 0 y 1) del mismo fichero. Cargarlo dos
   veces serian 362 MB en RAM en el brazo 1k3, en una maquina de 3 GB. La cache es por
   ruta y guarda el array ya descomprimido.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from construir_datasets import DESTINO                            # noqa: E402

_CACHE: dict[str, dict] = {}


def npz_de(brazo: str) -> Path:
    """El fichero del brazo. Se NIEGA si no esta, con el comando que lo construye."""
    carp = sorted(DESTINO.glob(f"{brazo}-*"))
    if not carp:
        raise SystemExit(
            f"no hay dataset preprocesado para '{brazo}' en {DESTINO}\n"
            f"  construyelo:  .venv/bin/python "
            f"experimentos/2026-09-04-planas-sobre-preprocesado/nn/"
            f"construir_datasets.py --brazo {brazo}")
    return carp[0] / "preprocesado.npz"


def _cargar(ruta: Path) -> dict:
    clave = str(ruta)
    if clave not in _CACHE:
        z = np.load(ruta)
        _CACHE[clave] = {"x": z["x"], "y": z["y"], "split": z["split"]}
    return _CACHE[clave]


class DatasetPreprocesado(Dataset):
    """Las ventanas de un split del `.npz` de un brazo.

    ⚠ El `y` NO se toca: son las mismas etiquetas del dataset origen, en coordenadas
    de la ventana de 16 px. El preproceso cambio la RESOLUCION de la entrada, no lo
    que hay que predecir, asi que `pos_err_px` sigue midiendo en los mismos pixeles.
    """

    def __init__(self, brazo: str, split: int):
        d = _cargar(npz_de(brazo))
        mask = d["split"] == split
        self.x = d["x"][mask]
        self.y = d["y"][mask]
        self.brazo, self.split = brazo, split
        # `(0,)` -> en lote `(B, 0)`: concatenar nada a la cabeza es la identidad.
        self._sin_borde = torch.zeros(0, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.x[i]),
                self._sin_borde,
                torch.from_numpy(self.y[i].copy()))


def como_produccion(brazo: str):
    """Una fabrica con LA FIRMA de `FoveatedWindowDataset`, para el parche.

    `loop._entrenar` la llama como
    `FoveatedWindowDataset(arrays, dims, split=N, pool_mode=..., pad_mode=...,
                           edge_inputs=..., mask_channel=...)`.
    Aqui se ignora todo salvo el `split`: la geometria y los modos ya se aplicaron al
    construir el dataset, y volver a leerlos seria fingir que aun deciden algo.
    """
    def fabrica(arrays, dims, split: int, **kw):        # noqa: ARG001
        return DatasetPreprocesado(brazo, split)
    return fabrica


def main() -> int:
    from red_local import ENTRADAS
    print(f"{'brazo':6} {'train':>7} {'val':>7} {'test':>7} {'forma de x':>14} {'y':>8}")
    for b in ENTRADAS:
        try:
            ds = {s: DatasetPreprocesado(b, s) for s in (0, 1, 2)}
        except SystemExit as e:
            print(f"{b:6} {e}")
            continue
        x0, _e, y0 = ds[0][0]
        print(f"{b:6} {len(ds[0]):>7} {len(ds[1]):>7} {len(ds[2]):>7} "
              f"{str(tuple(x0.shape)):>14} {str(tuple(y0.shape)):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
