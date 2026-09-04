#!/usr/bin/env python3
"""`aplicaKernel` de ESTE experimento: su kernel 3x3, sin relleno.

    from aplica_kernel import aplicaKernel
    salida = aplicaKernel(imagen)        # (H,W) o (C,H,W) o (B,C,H,W) -> (1, H-3+1, W-3+1)

    python nn/aplica_kernel.py           # que kernel es, y una pasada de ejemplo

Es un ATAJO. La operacion vive UNA sola vez en `experimentos/comun/preproceso.py`
y aqui solo se le ata el kernel de esta carpeta: los tres experimentos son
gemelos y su salida va a compararse como preprocesador, asi que tres copias de la
convolucion harian que la comparacion no significara nada. Ahi estan la firma
entera, los defectos declarados y el porque de cada uno.

⚠ Y por eso este fichero SI depende de `../comun/`, al contrario que
`red_local.py`. Si falta, se niega y dice donde esta -- no cae a una copia local
(R2: o defecto declarado, o fallar antes de empezar).
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMUN = Path(__file__).resolve().parents[2] / "comun"
if not (_COMUN / "preproceso.py").exists():      # pragma: no cover - defensivo
    raise SystemExit(
        f"falta {_COMUN / 'preproceso.py'}: la convolucion vive ahi, no aqui.\n"
        f"  este atajo solo le ata el kernel de {Path(__file__).parents[1].name}.")
sys.path.insert(0, str(_COMUN))

from preproceso import aplicaKernel_1k3 as aplicaKernel   # noqa: E402
from preproceso import cargar_kernel                     # noqa: E402

__all__ = ["aplicaKernel", "cargar_kernel"]

KERNEL = "1k3"


if __name__ == "__main__":
    import numpy as np
    k = cargar_kernel(KERNEL)
    print(f"  {k}")
    demo = np.zeros((20, 20), dtype=np.float32)
    demo[8:12, 8:12] = 1.0                        # un cuadrado en medio
    y = aplicaKernel(demo)
    print(f"  una vista 20x20 -> {y.shape}  (sin relleno: encoge {k.k - 1} px)")
    print(f"  respuesta: min {y.min():+.4f}  max {y.max():+.4f}  |media| {abs(y).mean():.4f}")
    print("  ⚠ la entrada traia UN canal: el de relleno se puso a 0 "
          "(= todo pixel es real)")
