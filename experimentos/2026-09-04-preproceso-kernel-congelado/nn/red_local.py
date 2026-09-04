#!/usr/bin/env python3
"""La red de ESTE experimento: kernel CONGELADO de otro run + conv 3x3 entrenable.

    python nn/red_local.py --comprobar          # geometria, congelacion y colapso

⚠⚠ POR QUE ESTO VIVE AQUI Y NO EN `src/fv/models/builder.py`
   Instruccion del dueno, 2026-09-03: «estos son experimentos, nada tienen que
   ver con las redes previas... Si hay que hacer cambios al codigo tendremos que
   copiarlo localmente». El codigo de produccion NO se toca para probar una idea.
   Igual que `2026-09-04-cnn-plana-1k3-sinpadding/nn/red_local.py`, del que este
   fichero hereda el envoltorio `padding=0`.

QUE RED ES, EXACTAMENTE

    x (2, 20, 20)                 vista + relleno, como todos los gemelos
      -> Conv2d(2 -> 1, kf, padding=0)   CONGELADA: el kernel de `1k<kf>`
      -> ReLU                            ⚠ NO es decorativa: ver abajo
      -> Conv2d(1 -> 1, 3,  padding=0)   entrenable
      -> flatten -> ReLU -> Linear(-> 12)

   La primera capa es literalmente `aplicaKernel_1k<kf>` (`comun/preproceso.py`),
   que a su vez esta comprobado que es la capa L1 del experimento del que sale
   (diferencia maxima 0,0 contra su `stop-04/mapas.npy`).

⚠⚠ LA ReLU DE EN MEDIO ES LO QUE HACE QUE ESTE EXPERIMENTO EXISTA
   Sin ella, dos convoluciones seguidas SIN activacion son **una sola
   convolucion** de tamano `kf + 3 - 1`, con sus pesos atados al producto de las
   dos. No es una opinion: esta medido (`--comprobar`, y el mismo calculo suelto
   da diferencia 7,6e-06, o sea redondeo de float32).

   Y entonces el estudio seria degenerado, porque su gemelo SIN atar ya esta
   corrido y en git:

       preproc 1k3 + conv3  ==  una 5x5 atada   ->  gemelo libre: 1k5 (f1 0,642)
       preproc 1k5 + conv3  ==  una 7x7 atada   ->  gemelo libre: 1k7 (f1 0,618)
       preproc 1k7 + conv3  ==  una 9x9 atada   ->  gemelo libre: NO existe

   O sea que sin ReLU el brazo solo puede EMPATAR o PERDER contra un numero que
   ya esta pagado: su espacio de funciones es un subconjunto estricto. Con ReLU
   deja de serlo, y la pregunta «¿sirve de algo precocinar la entrada con un
   kernel ya aprendido?» pasa a tener una respuesta que no se sabe de antemano.

   La ReLU sale GRATIS del builder: `_branch_forward` activa entre capas y no
   despues de la ultima (`builder.py:227-233`), asi que basta con que el kernel
   congelado NO sea la ultima capa de la rama.

QUE PASA CON EL TAMANO, Y POR QUE NO HAY QUE AJUSTAR NADA A MANO
   Sin relleno se recorta DOS veces: 20 -> (20-kf+1) -> (...-3+1).

       kf=3:  20x20 -> 18x18 -> 16x16 -> 256 features
       kf=5:  20x20 -> 16x16 -> 14x14 -> 196 features
       kf=7:  20x20 -> 14x14 -> 12x12 -> 144 features

   `_infer_flat_features` hace un forward de prueba sobre un dummy de `dims.N`
   (`builder.py:274-278`), y como el kernel congelado ya vive DENTRO de la rama,
   la cabeza se dimensiona sola. Es la razon principal de meter el preproceso en
   el modelo y no en el `Dataset`: por ahi la cabeza se habria dimensionado para
   20x20 y el forward habria reventado (ruidoso) o, peor, casado por accidente.

⚠ LOS 256 / 196 FEATURES NO SON UN NUMERO CUALQUIERA
   Coinciden EXACTAMENTE con los de `1k5` (256) y `1k7` (196), que ya estan
   corridos sobre el mismo dataset, semilla, receta y stops. Son el control
   iso-features que el criterio necesita, y salen a coste cero. El de 144 no
   tiene gemelo y por eso su lectura es mas debil; esta dicho en el criterio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXP.parent / "comun"))

from fv.models.builder import FoveatedRegionalNN, full_config     # noqa: E402
from preproceso import cargar_kernel                              # noqa: E402

# La geometria y la receta se heredan del gemelo de 3x3: 2 canales de entrada,
# cabeza de 12 salidas, `regions: single`. Lo unico que cambia es que delante se
# le pone un kernel congelado. Se reusa su config y NO se crea una nueva para que
# no haya dos ficheros que puedan divergir en la geometria (R4).
RED_BASE = REPO / "configs" / "networks" / "plana-20-1k3.yaml"

# El kernel congelado se toma de `best.pt`, no de `last.pt`: de un preprocesador
# se quiere el mejor estado, no el ultimo. Queda escrito aqui porque cambia el
# resultado y no se puede deducir mirando la red.
PESOS = "best"


def _sin_relleno(conv: nn.Conv2d) -> nn.Conv2d:
    """La misma conv con `padding=0`, copiando los pesos (no re-inicializa).

    Copiar y no reconstruir es lo que mantiene este experimento comparable con
    los gemelos: dos construcciones dan dos inicializaciones distintas.
    """
    nueva = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=0, bias=conv.bias is not None)
    with torch.no_grad():
        nueva.weight.copy_(conv.weight)
        if conv.bias is not None:
            nueva.bias.copy_(conv.bias)
    return nueva


def construir(kf: int, cfg: dict | None = None) -> FoveatedRegionalNN:
    """La red del brazo `1k<kf>`: kernel congelado + ReLU + conv 3x3 entrenable."""
    if cfg is None:
        cfg = full_config(yaml.safe_load(RED_BASE.read_text()))
    model = FoveatedRegionalNN(cfg)

    kern = cargar_kernel(f"1k{kf}", pesos=PESOS)
    if kern.k != kf:                                  # pragma: no cover - defensivo
        raise SystemExit(f"el kernel de 1k{kf} mide {kern.k}x{kern.k}")

    # La conv congelada: 2 canales -> 1, exactamente `aplicaKernel_1k<kf>`.
    congelada = nn.Conv2d(kern.canales, kern.n_kernels, kf, stride=kern.stride,
                          padding=0, bias=True)
    with torch.no_grad():
        congelada.weight.copy_(kern.peso)
        congelada.bias.copy_(kern.sesgo)
    for p in congelada.parameters():
        p.requires_grad_(False)          # CONGELADA: no la toca el optimizador

    # La entrenable: la 3x3 que ya traia la config, con `padding=0` y 1 canal de
    # entrada (el mapa del kernel congelado).
    original = model.center_convs[0]
    entrenable = nn.Conv2d(1, original.out_channels, original.kernel_size,
                           stride=1, padding=0, bias=original.bias is not None)
    with torch.no_grad():
        # Se siembra desde los pesos que el builder ya habia inicializado, para
        # no meter una fuente de aleatoriedad distinta a la de los gemelos.
        entrenable.weight.copy_(original.weight[:, :1])
        if original.bias is not None:
            entrenable.bias.copy_(original.bias)

    # El ORDEN importa: la congelada NO puede ser la ultima de la rama, porque
    # `_branch_forward` no activa despues de la ultima. Ver la cabecera.
    model.center_convs = nn.ModuleList([congelada, entrenable])
    # La cabeza se re-dimensiona con la rama ya cambiada.
    flat = model._infer_flat_features()
    model.flat_features = flat
    model.head = nn.Linear(flat + model.n_edge, 12)
    return model


def _comprobar() -> int:
    """Geometria, congelacion y el colapso que justifica la ReLU."""
    ok = True
    print("geometria y parametros de los tres brazos:")
    for kf, esperado in ((3, 256), (5, 196), (7, 144)):
        m = construir(kf)
        entrenables = sum(p.numel() for p in m.parameters() if p.requires_grad)
        congelados = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        casa = m.flat_features == esperado
        ok &= casa
        print(f"  1k{kf}: features {m.flat_features:>4} "
              f"({'✓' if casa else '✗ esperaba ' + str(esperado)}) · "
              f"entrenables {entrenables:>5} · congelados {congelados:>3}")

    print("\nla capa congelada NO se mueve con un paso de optimizacion:")
    m = construir(5)
    antes = m.center_convs[0].weight.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    x = torch.randn(2, 2, 20, 20)
    e = torch.zeros(2, m.n_edge)
    m(x, e).sum().backward()
    opt.step()
    igual = torch.equal(antes, m.center_convs[0].weight.detach())
    ok &= igual
    print(f"  {'✓ intacta bit a bit' if igual else '✗ SE MOVIO'} tras un Adam.step()")

    print("\nsin ReLU las dos convoluciones colapsan en UNA "
          f"(por eso la ReLU esta puesta):")
    w1 = m.center_convs[0].weight.detach()
    b1 = m.center_convs[0].bias.detach()
    w2 = m.center_convs[1].weight.detach()
    b2 = m.center_convs[1].bias.detach()
    x = torch.randn(3, 2, 20, 20)
    compuesto = F.conv2d(F.conv2d(x, w1, b1), w2, b2)
    k1, k2 = w1.shape[-1], w2.shape[-1]
    w_eff = torch.zeros(1, w1.shape[1], k1 + k2 - 1, k1 + k2 - 1)
    for c in range(w1.shape[1]):
        w_eff[0, c] = F.conv2d(w1[0, c][None, None], torch.flip(w2, [2, 3]),
                               padding=k2 - 1)[0, 0]
    una = F.conv2d(x, w_eff, b2 + b1 * w2.sum())
    dif = float((compuesto - una).abs().max())
    print(f"  1k5 + conv3 sin activacion == una 7x7 atada · dif max {dif:.1e}")

    print("\n" + ("todo casa." if ok else "⚠ ALGO NO CASA"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    if p.parse_args().comprobar:
        return _comprobar()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
