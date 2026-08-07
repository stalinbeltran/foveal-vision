# Plan de 40 h desatendidas — criterio escrito ANTES de mirar

> **Este documento se comitea antes de que exista un solo run del plan.** Su valor entero está en
> eso: las reglas de decisión de abajo se pueden comprobar contra el commit. Si se cambian después
> de ver resultados, el plan deja de decidir nada (protocolo.md §1).

Fecha: 2026-08-06. Ejecutor: `scripts/plan_40h.py`. Dataset B fijo: `dirty1000-80px-16px`.

## 0. Qué pregunta responde

Los dos estudios `d1000-*` fijaron `lr=0.0014` y `batch_size=85`, pero se midieron con un
presupuesto de 20 épocas que **ningún run agotó**: sobre los 70 runs, la mejor época es ≥16 en
todos y ≥20 en 37; 65 de 70 seguían mejorando entre la época 15 y la 20, con una caída media de
`val_loss` de 0,0127 — **más de la mitad de la amplitud completa del eje de `lr`** (0,022). Es
decir: aquellos recorridos midieron en parte velocidad de convergencia, no calidad.

Al mismo tiempo, la red sigue siendo el default derivado (`n_layers=2, channels=[16,16], k=3`), y
el reparto de parámetros dice algo que nadie había mirado:

```
head          153.612   96,9 %
center_convs    2.480    1,6 %
periph_convs    2.480    1,6 %
```

**El 97 % del modelo es la cabeza.** `n_layers` añade 4.640 parámetros por capa (+3 %): no es un
eje de capacidad, es un eje de **campo receptivo** — con `k=3, s=1, L=2` el campo receptivo es
**5×5 sobre una fóvea de 16×16**. `channels` sí es el eje de capacidad, porque ensancha la entrada
de la cabeza (`[16,16]`→`[32,32]` pasa de 158k a 326k parámetros).

Hay entonces **tres resortes distintos** y no se sabe cuál mueve la aguja. El plan criba primero y
confirma después.

## 1. Constantes fijadas antes

| | valor | de dónde sale |
|---|---|---|
| `patience` | **10** | suelo medido: sobre los 70 runs, la racha más larga sin mejorar seguida de una mejora posterior es de **6** épocas (4 épocas ocurre 4 veces). `patience=3` habría truncado 19 runs |
| `epochs` bloque 1 | **100** | tope; con `patience=10` cada config para cuando converge |
| δ (banda de ruido) | **0,0067** | 1-SE medido de las 5 semillas del mejor punto de `d1000-lr-1-s0-lr` |
| métrica de ranking | `val_loss` **del checkpoint** | es lo que rankea el proyecto y lo que carga Diagnóstico/Predecir; `f1` se reporta al lado |
| `lr`, `batch_size` | 0,0014 / 85 | ganadores confirmados de los dos estudios `d1000-*` |

Coste por época medido/estimado (batch 85; ~35 s fijos de dataloader + modelo):

| config | params | ms/paso | s/época |
|---|---|---|---|
| `L2 ch[16,16] k3` (base) | 158k | 27,4 | 65 (medido) |
| `L4 ch[16]×4 k3` | 168k | 58,8 | ~100 |
| `L2 ch[32,32] k3` | 326k | 67,8 | ~109 |
| `L2 ch[16,16] k5` | 167k | ~34 | ~72 |

## 2. Bloque 1 — cribado (1 semilla, ~10 h)

Cuatro runs secuenciales, idénticos salvo el resorte bajo prueba:

| run | resorte | config |
|---|---|---|
| `p40-screen-base` | — (referencia) | `n_layers=2, channels=[16,16], k_center=3, k_periph=3` |
| `p40-screen-depth` | profundidad / campo receptivo | `n_layers=4, channels=[16,16,16,16]` |
| `p40-screen-width` | capacidad | `channels=[32,32]` |
| `p40-screen-kernel` | campo receptivo en la fóvea | `k_center=5` (la periferia se queda en 3: las esquinas se etiquetan **solo** sobre la fóvea, contrato ①a) |

**Un cribado de una semilla no declara ningún ganador** (protocolo.md: un resultado sin N semillas
es una anécdota). Su única función es elegir dónde gastar el bloque 2.

## 3. La regla de decisión (escrita antes)

Sea `mejora_i = val_loss(base) − val_loss(config_i)` sobre el checkpoint.

1. **Candidatos** = los resortes con `mejora_i > δ` (0,0067).
2. **Si no hay candidatos** → el bloque 2 **no barre estructura**. Barre `lr` hacia abajo con rango
   `[0.0004, 0.0006, 0.0008, 0.0011, 0.0014]`. Justificación escrita antes: el ganador de
   `d1000-lr-1` es 0,0014, que es el **valor mínimo barrido**, y la pérdida crece monótonamente con
   `lr` en todo el rango — el óptimo no está acotado por abajo.
3. **Si hay candidatos** → gana el de mayor `mejora_i`; si otro cae dentro de δ del mejor, gana el
   **más barato** en s/época (regla coste/calidad, D-W1).
4. El eje del bloque 2 según el ganador:
   - profundidad → `n_layers`, rango `[1, 2, 3, 4, 5]`
   - capacidad → `channels`, rango `[[16,16], [24,24], [32,32], [48,48], [64,64]]`
   - campo receptivo → `k_center`, rango `auto` (= `[3, 5, 7]`)

## 4. Bloque 2 — confirmación (5 semillas)

- `epochs = min(100, max(30, ceil(1.25 × mayor mejor-época observada en el bloque 1)))`,
  `patience = 10`. Regla escrita antes para que el presupuesto salga **de lo medido**, no de una
  corazonada.
- `seeds = 5` (la misma N de los estudios `d1000-*`, para que las bandas sean comparables).
- **Guarda de presupuesto**, aplicada en este orden hasta que la estimación baje de **34 h**:
  1. `seeds` 5 → 3;
  2. recortar el rango a sus 3 valores más baratos.
  La estimación usa el coste por paso medido de cada punto + los ~35 s fijos de dataloader.

## 5. Reanudable por diseño

El equipo se apaga por falta de energía (confirmado por el usuario; tres runs de `d1000-*` tienen
**una sola época** de 17.339 s, 20.543 s y 6.920 s con el resto normales — 12,5 h de 40,3 h). Por
eso:

- el bloque 2 es un **recorrido**, y `run_sweep` ya rehace todo punto que no esté `done`/`cancelled`;
- el bloque 1 salta los runs ya `done` y rehace los demás;
- **volver a lanzar `scripts/plan_40h.py` continúa donde se quedó**, sin argumentos.

Lo que el script borra: **solo** runs con prefijo `p40-` que él mismo creó y que no terminaron.
Nada más.

## 6. Salida

`runs/../plan-40h-report.json` + log en `plan-40h.log`: las cuatro curvas del cribado, la regla
aplicada con sus números, y el veredicto del recorrido con su banda.
