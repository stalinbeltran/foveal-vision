# El criterio, **escrito antes de correr nada**

> Congelado el 2026-09-04 a las 10:25 UTC, **antes** de la primera época. Los resultados del 7×7
> (f1 **0,618**) y del 5×5 (f1 **0,642**) sí se conocían — son de hace unas horas y están
> commiteados; por eso la predicción de aquí se apoya en ellos y **eso se dice**, en vez de fingir
> un ciego que no existe.

## Lo que este punto pregunta

El 5×5 midió que **bajar el campo receptivo de 7 px a 5 px no cuesta nada medible**, y concluyó
que lo que la serie mide es el **tamaño de la cabeza**, no la capacidad de L1. Este run pregunta
si eso **sigue valiendo en 3 px**, que es el extremo del eje y el valor de producción.

Como en el 5×5, mueve **dos cosas a la vez y en sentidos opuestos**:

| | mapa (valid) | features | L1 | campo | total |
|---|---|---:|---:|---:|---:|
| 1k7 sin relleno | 14×14 | 196 | 99 | 7 px | 2.511 |
| 1k5 sin relleno | 16×16 | 256 | 51 | 5 px | 3.183 |
| **1k3 sin relleno** | **18×18** | **324** | **19** | **3 px** | **3.967** |

**+27 % de features** respecto al 5×5, con **tres veces menos parámetros** en L1.

## La predicción, y de dónde sale su pendiente

⚠ **No se usa la vieja regla de «~0,09 de f1 por cada mitad de features»**: el 7×7 midió que esa
recta **se desacelera** (−0,101 → −0,083 → −0,038). La pendiente que aplica en este extremo se
estima de los dos pasos limpios más cercanos:

| paso | Δ f1 | Δ log₂(features) | pendiente |
|---|---:|---:|---:|
| 392 → 196 | −0,038 | −1,000 | 0,038 |
| 196 → 256 *(y k 7→5)* | +0,024 | +0,386 | 0,062 |

Se toma **0,05 por cada factor 2** como mejor estimación local. Con log₂(324/256) = 0,340:

> **f1 esperado ≈ 0,642 + 0,05 × 0,340 ≈ 0,659**, banda **0,04** → rango **`[0,619 – 0,699]`**.

⚠ La pendiente 0,05 es una **interpolación entre dos medidas de una sola semilla**, no una
constante medida. Se escribe para que el criterio sea falsable, no porque tenga precisión.

## El discriminador PRINCIPAL, que no depende de esa pendiente

**¿Supera el 3×3 al 5×5?** Tiene **más features** (324 contra 256), así que si el tamaño de la
cabeza es lo que manda, **tiene que ser ≥ 0,642**. Es un binario que no depende de estimar
ninguna pendiente, y por eso es el que decide.

| desenlace | veredicto | qué significa |
|---|---|---|
| **f1 ≥ 0,642** *(y dentro de `[0,619 – 0,699]`)* | **el eje `k` es INERTE hasta 3 px** | tres tamaños de kernel, ningún coste medible. En esta plana, L1 no aporta selectividad: lo único que mueve el f1 es cuántas features llegan a la cabeza |
| **f1 < 0,642** *(peor que el 5×5 **con más features**)* | **3 px es donde el campo receptivo SÍ muerde** | y sería el resultado más informativo del eje: localizaría el suelo, y encima justo en el valor que usa producción |
| **f1 > 0,699** | **hay algo más** | por encima de lo que la pendiente predice. Candidato: con 19 parámetros en L1 casi no hay qué sobreajustar. Habría que mirar la brecha val/train antes de creerlo |

## ⚠⚠ Y una métrica que en `k` = 3 hay que leer distinto

El nulo de la energía en el subespacio clásico 6-D es `6/k²`:

| k | 7 | 5 | **3** |
|---|---:|---:|---:|
| nulo | 0,122 | 0,240 | **0,667** |

En `k` = 3 la base clásica abarca **6 de las 9 dimensiones**, así que el agregado casi no puede
discriminar y compararlo con el 2,24× del 7×7 sería comparar dos escalas. **Lo que sigue siendo
interpretable es el desglose**: cuánta energía es DC y cuánta no. Los dos kernels anteriores
salieron promediadores (24 % y 32 % de DC, y por debajo del azar al quitarlo); la pregunta aquí es
si con 3×3 pasa lo mismo.

## Lo que este run NO puede contestar, dicho antes

1. **Una semilla**, como los cinco anteriores. Acota, no declara.
2. **Los dos ejes siguen juntos** (features y campo receptivo). Separarlos pide recortar la vista
   de antemano, y no está hecho.
3. **Que `k` = 3 sea el valor de producción NO hace que esto diga nada sobre la foveada**: allí son
   **4 capas** apiladas, así que el campo receptivo efectivo es mucho mayor que 3 px. Aquí hay una.
