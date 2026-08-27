"""D — cuantas ventanas consume una epoca, y cuales.

Por que existe
--------------
El pool de train de un dataset de ventanas depende del `stride` con el que se
extrajo: con ventana 16 sobre imagenes de 60x80 van de 1.755.000 ventanas
(stride 1) a 12.000 (stride 16), un factor 146,2x (medido 2026-08-27, ver
docs/barrido-stride.md 3). A epocas iguales, entrenar sobre el pool entero le
daria al dataset denso 146 veces mas pasos de gradiente que al disperso, y una
tabla que comparase los dos estaria midiendo el PRESUPUESTO, no la densidad de
la rejilla -- con exactamente la misma pinta.

Este sampler iguala el presupuesto: cada epoca consume `por_epoca` ventanas,
sea el pool grande o pequeno.

Las tres decisiones, y por que
------------------------------
1. **Sin reemplazo dentro de una pasada.** Con `por_epoca <= pool` se toma el
   prefijo de una permutacion. Muestrear con reemplazo repetiria unas ventanas
   y omitiria otras dentro de la MISMA epoca: ruido gratis que no mide nada.
2. **Con `por_epoca > pool`, permutaciones completas concatenadas.** Asi el
   pool pequeno se recorre entero k veces y el resto sale de una permutacion
   mas, en vez de sesgar hacia lo que el azar repita.
3. **Reproducible.** La semilla de cada epoca es `(seed, epoca)`, asi que misma
   semilla + misma config => mismos pesos sigue siendo cierto (contrato XI), y
   dos epocas distintas no reciben el mismo subconjunto.

⚠ El contador de epoca avanza en `__iter__`, que es donde el DataLoader empieza
una pasada. Si alguien itera el sampler dos veces en la misma epoca obtiene
subconjuntos distintos -- correcto para entrenar, y por eso este sampler NO se
usa para validar: val se recorre entero y siempre igual.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class VentanasPorEpoca(Sampler):
    def __init__(self, pool: int, por_epoca: int, seed: int):
        if pool <= 0:
            raise ValueError("el pool de train esta vacio")
        if por_epoca <= 0:
            raise ValueError("por_epoca debe ser > 0")
        self.pool = int(pool)
        self.por_epoca = int(por_epoca)
        self.seed = int(seed)
        self.epoca = 0

    def indices_de(self, epoca: int) -> np.ndarray:
        """Los indices de esa epoca. Puro: misma (semilla, epoca) => misma lista.

        Separado de `__iter__` para que un test pueda comprobarlo sin montar un
        DataLoader ni entrenar nada.
        """
        rng = np.random.default_rng([self.seed, epoca])
        if self.por_epoca <= self.pool:
            return rng.permutation(self.pool)[:self.por_epoca]
        vueltas = -(-self.por_epoca // self.pool)          # ceil
        return np.concatenate([rng.permutation(self.pool)
                               for _ in range(vueltas)])[:self.por_epoca]

    def __iter__(self):
        self.epoca += 1
        return iter(int(i) for i in self.indices_de(self.epoca))

    def __len__(self) -> int:
        return self.por_epoca
