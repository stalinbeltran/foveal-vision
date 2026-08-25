# Plan de 40 h desatendidas — criterio escrito ANTES de mirar

> ⚠ **Geometría: este documento usa la ortografía anterior al 2026-08-25.** `N`, `c_frac`,
> `d` y `pen_frac` fueron reemplazados por longitudes en px reales (`fovea_px`, `border_px`,
> `border_reduce`, `overlap_fovea_px`, `overlap_border_px`). **Ninguna red cambió** — es un
> cambio de nombre, verificado bit a bit — así que **todos los números de aquí siguen siendo
> válidos**. La traducción está en [instructionsNewNN.md](../instructionsNewNN.md) §2.1.
> Ojo con uno: **`d` cambió de significado**, no sólo de nombre. Antes agrandaba el contexto
> (`borde = celdas·d`); hoy `border_reduce` sólo dice cuánto se comprime un borde de tamaño
> fijo. Un eje `d` de aquella época medía **área y compresión a la vez**.

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

## 7. Correcciones hechas DESPUÉS de ver el cribado (2026-08-07 06:20)

> Este documento dice que cambiar las reglas tras ver resultados invalida el plan. Estas tres se
> cambiaron igual, y por eso quedan escritas aparte, con la hora, para que se juzguen: **ninguna
> toca la regla de decisión ni la métrica** — cambian el rango y el presupuesto del bloque 2, y las
> tres se habrían escrito igual sin mirar un solo número, si lo hubiera pensado mejor.

**7.1 El recorte dejaba fuera al ganador.** La guarda §4 recortaba «a sus 3 valores más baratos»,
que con eje `n_layers` son `[1,2,3]` — fuera precisamente el **4** que hay que confirmar. Un
barrido de confirmación que no contiene al candidato no confirma nada. Ahora recorta **alrededor
del ganador del cribado** (`trim_around`), que nunca lo suelta.

**7.2 El coste se estimaba al tope, no a lo que se corre.** `estimate_hours` cobraba las 77 épocas
del tope a cada punto, pero `patience=10` cortó los runs del cribado en **71, 32 y 57** de 100 —
y **el más profundo fue el que antes paró**. Costear al tope sobreestima ~2× y gasta semillas que
sí se podían pagar. Ahora se interpola entre las profundidades observadas (`epochs_for`).

**7.3 El rango es `[4, 2, 3, 5]`, en ese orden.** `[2,3,4,5]` para que rodee al ganador **y**
contenga L2, que es la red actual y la referencia de toda la afirmación (`[1..5]` no cabía y L1
está dominado). El **orden** es la mitigación: los puntos se entrenan en el orden de la lista, así
que si el presupuesto se queda corto lo que falta son los últimos. Primero el ganador (**4**) y la
referencia (**2**) — los dos que responden *¿la profundidad gana a la red actual?* — y el **5** al
final, que solo afina dónde está el óptimo. El ranking agrega por valor, no por orden: **esto no
cambia ningún resultado**, solo qué se pierde si algo se corta.

Y `BUDGET_HOURS` pasa de 34 a **36 h**: el 34 se fijó cuando el usuario ofrecía «unas 30 h», y
después ofreció 40+. Es el presupuesto declarado, no un número ajustado a un resultado. Con eso,
`[4,2,3,5] × 5 semillas` estima **34,0–34,8 h** (el micro-benchmark varía entre llamadas) y cabe
de forma estable.

## 8. Lo que dice la MÉTRICA DE TAREA sobre este veredicto (2026-08-08)

Todo lo de arriba es **f1 de ventana**, que es un proxy. Los mismos 20 runs, medidos contra los
párrafos de la fuente (200 imágenes de val, 5,4 s por run, cero entrenamiento):

| `n_layers` | ventana | **tarea** | ¿separado de L4? |
|---|---|---|---|
| **4** | 0,9244 | **0,7796** | — |
| 3 | 0,9093 | 0,7644 | p = 0,135 |
| 5 | 0,8832 | 0,7654 | p = 0,167 |
| 2 | 0,8756 | 0,7572 | **p = 0,032** |

**La conclusión del plan aguanta**: `n_layers=4` gana también por tarea, y contra L2 —que es la
comparación que el plan hace— la diferencia sobrevive a una permutación exacta de las semillas.

**Dos matices que hay que citar con ella:**

1. **La ganancia se encoge a la mitad** (+0,0488 en ventana → **+0,0224** en tarea) y **las bandas
   dejan de ser disjuntas**: la peor semilla de L4 (0,7532) queda por debajo de la mejor de L2
   (0,7689). «Bandas disjuntas» es una propiedad del proxy, no del resultado.
2. **El cribado de 1 semilla no habría visto nada por tarea**: base 0,7523 contra depth 0,7532, con
   un `sem` por run de ±0,023. Que funcionara fue suerte del proxy, no del diseño del cribado.

Detalle, la advertencia sobre L5 (su bimodalidad **no aparece** en la tarea) y `k_center=5` —el
peor por ventana y el **mejor** por tarea con 1 semilla— en
[metrica-de-tarea.md](metrica-de-tarea.md) §2 ter.

## 6. Salida

`runs/../plan-40h-report.json` + log en `plan-40h.log`: las cuatro curvas del cribado, la regla
aplicada con sus números, y el veredicto del recorrido con su banda.
