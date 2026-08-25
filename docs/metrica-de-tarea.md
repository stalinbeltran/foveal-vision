# La métrica de tarea: cablearla, dimensionarla y usarla

> **Estado (2026-07-26): FASES 1, 2 y 3b HECHAS. La 3 aplazada por decisión del usuario (F11); la
> 4 tiene todo el código escrito y espera el dato.**
>
> - **Fase 1** — el proxy de ventana validado en un eje de **D** (§2): Spearman agregado +0,956.
> - **Fase 2** — `task_score` cableada (§3): módulo `fv.task`, contrato ⑬, API, UI, los dos CLIs.
> - **Fase 3b** — ✅ **el proxy vale TAMBIÉN para C** (§2 bis): sobre el eje `d`, Spearman agregado
>   **+1,000**, mismo ganador, dentro de la frontera δ. **`OBJECTIVES` no cambia y §5.5 no se
>   ejecuta.** De regalo, la media respuesta a §9.9: **la periferia no está aportando** (§2 bis.1).
> - **§2 ter (2026-08-08)** — `n_layers=4` **llevado a la métrica de tarea** con **200 imágenes de
>   val**: gana también por tarea (0,7796 vs 0,7572 de L2, **p = 0,032**), pero **la ganancia se
>   encoge a la mitad y las bandas dejan de ser disjuntas**. El criterio de §5.4 sale **+0,800 <
>   0,90** por un único intercambio entre dos puntos que la tarea **tampoco distingue** (p = 0,897)
>   → §5.5 **sigue sin ejecutarse**, y queda anotada la reserva: **el f1 de ventana exagera el
>   hundimiento de L5**.
> - **Fase 3** — ⏸ aplazada: **F11 cerrada, no se regenera el dato por ahora**. Se sigue con 20
>   imágenes de val y la métrica de tarea queda como *informe del ganador*, nunca como criterio
>   para elegir entre puntos. §4 conserva toda la aritmética para reabrirla.
> - **Fase 4** — 🟡 el **código está hecho y probado** (selectores de dataset/split, las guardas, el
>   registro de F14, 5 tests); falta **la fuente**, que depende de F11.
> - **§9** — 8 de las 10 pruebas hechas, ninguna entrenando nada salvo la 3b. Dos **corrigen** al
>   documento (§9.4: la sd sube; §9.2: los tres defaults de F están mal). Quedan 9.8 (no hacer
>   hasta que duela) y 9.9 (bloqueada por F12, pero §2 bis.1 la contesta a medias).
>
> Lo que quede abierto va marcado como **DECISIÓN DEL USUARIO** y no se toma solo (decisiones.md).

Documento hermano de [protocolo.md](protocolo.md) §2, que fija *qué* métrica manda. Este fija
**cómo se construye, cuánto cuesta y qué hace falta para que sirva**.

---

## 1. El problema, en una frase

`fv.metrics.paragraph_f1` está escrita desde el día 1 y **no la llama nadie**. Los recorridos
rankean por métrica de **ventana** (`f1`, `pos_err_px`), que protocolo.md §2 clasifica como
*proxy*, mientras la métrica que manda —**párrafo por imagen**— no se calcula en ninguna parte.
El propio protocolo marcaba un paso obligado antes del primer recorrido grande: medir la
correlación de rangos entre las dos. Nunca se había hecho.

---

## 2. Fase 1 — HECHA: el proxy está validado (para ejes de D)

Medido el 2026-07-26 sobre los **65 runs ya entrenados** de `fast-lr-s0-lr` (13 valores de `lr`
× 5 semillas), sin entrenar nada: inferencia de imagen completa sobre las **20 imágenes de val**
de `dirty-paragraphs-fast-80px`, reconstrucción TL→BR y `paragraph_f1` con IoU ≥ 0,5.

| Correlación de rangos (Spearman) ventana ↔ tarea | valor |
|---|---|
| por run (n = 65) | **+0,736** |
| agregado por valor del eje (13 puntos × 5 semillas) | **+0,956** |

El ganador **coincide** con las dos métricas: `lr = 0,00215443`.

```
punto                  ventana    tarea
{"lr": 0.00215443}      0.6189   0.5353   <- gana en ambas
{"lr": 0.0014678}       0.6117   0.5046
{"lr": 0.001}           0.5957   0.4973
{"lr": 0.00068129}      0.5612   0.5022
{"lr": 0.00316228}      0.5549   0.5241
...
{"lr": 0.0001}          0.3243   0.3057
{"lr": 0.01}            0.1021   0.0963
```

**Consecuencia decidida por la evidencia:** el objetivo de ranking de un recorrido **sigue
siendo la métrica de ventana**. No se añade `paragraph_f1` a `OBJECTIVES`. Es barata, se calcula
por época y ordena igual que la cara.

**Los dos límites de esa conclusión, que hay que escribir en cualquier informe que la use:**

1. Se midió sobre **un eje de D** (`lr`), que no cambia la arquitectura ni la vista foveada.
   protocolo.md §2 avisa de que barrer **C** cambia *la regla de mirar a la vez que el modelo*.
   **Para ejes de C el proxy NO está validado** → §5.
2. Los knobs de F (`threshold`, `stride`, `nms_radius`, `min_size`) se usaron **por defecto**,
   sin optimizar por run.

**El hallazgo colateral, que manda sobre las fases siguientes:** la desviación típica del F1 de
párrafo **entre imágenes** es **0,372** y el val tiene **20 imágenes** → error estándar por run
**±0,083**.
> ⚠ **Esa sd quedó re-medida el 2026-07-26 en §9.4: es 0,4148 (±0,093), y sube.** El 0,372 de aquí
> se promedió sobre el recorrido `lr` **entero**, modelos de F1 0,10 incluidos, que varían poco entre
> imágenes. **Para cualquier cuenta usa la tabla de §4.1**, que ya está rehecha; este número se
> conserva porque es lo que se midió en la Fase 1 y borrarlo haría irreproducible su tabla.

Las diferencias entre puntos vecinos del recorrido son de 0,01 a 0,05. Es decir: hoy
la métrica de tarea es **más ruidosa que las diferencias que se quieren medir**, y por eso el
Spearman salta de 0,736 (por run) a 0,956 (agregando 5 semillas): el ruido está en la estimación
por run, no en el proxy. Esto es exactamente lo que protocolo.md §3 avisaba — *el tamaño de
muestra efectivo lo dan las imágenes, no las ventanas*.

**Script de referencia**: la medición se hizo con un script de un solo uso; la Fase 2 lo
convierte en código de primera clase y §5 lo repite sobre un eje de C.

---

## 2 bis. Fase 3b — HECHA (2026-07-26): **el proxy vale también para C**

**Fecha:** 2026-07-26 · **Dataset:** `dirty-paragraphs-fast-80px` (20 imágenes de val) ·
**Recorrido:** `proxy-c-d` · **Eje:** `d` (submuestreo de la periferia, dominio **C**) ·
**6 valores × 5 semillas = 30 runs**, 20 épocas, receta `corta`, objetivo de ventana `f1`, monitor
`val_loss` · **knobs de F por defecto** (los mismos que §2, a propósito: si no, las dos
correlaciones no serían comparables — ver §9.2 y F15).

> ⚠ **Traducción a la geometría de 2026-08-25 (léase antes de citar esto).** El eje se llamaba `d`
> y, con `N` fijo, movía **dos cosas a la vez**: el contexto real (`border_px = 2·d`) y la
> compresión (`border_reduce = d`), dejando el anillo siempre en **2 celdas**. En la ortografía de
> hoy, lo que esta tabla midió es **`border_px` = 2, 4, 6, 8, 10, 12 px a `N` constante** — es
> decir, *más área con el mismo coste*, no *más compresión*. La pregunta simétrica —*a igual área,
> ¿ayuda verla con más resolución?*— **no la mide nadie todavía**: exigiría fijar `border_px` y
> mover `border_reduce`, y eso sí cambia `N` y el número de parámetros.

| `d` | ventana (f1) | sem entre semillas | **tarea** (macro F1) | periferia real | entrada | s/época |
|---|---|---|---|---|---|---|
| 1 | 0,6213 | 0,0219 | 0,5421 | 2 px | 20 | 7,2 |
| **2** | **0,6244** | 0,0153 | **0,5533** | 4 px | 24 | 8,8 |
| 3 | 0,6089 | 0,0125 | 0,5065 | 6 px | 28 | 8,6 |
| 4 | 0,5965 | 0,0131 | 0,4958 | 8 px | 32 | 7,1 |
| 5 | 0,6112 | 0,0104 | 0,5342 | 10 px | 36 | 7,0 |
| 6 | 0,6063 | 0,0112 | 0,4974 | 12 px | 40 | 7,0 |

| | valor |
|---|---|
| **Spearman agregado** (n = 6 puntos) | **+1,000** — orden idéntico, punto por punto |
| Spearman por run (n = 30) | +0,257 |
| ganador por ventana | `d = 2` |
| **ganador por tarea** | **`d = 2`** — el mismo |
| frontera δ = 0,0153 (1-SE de las semillas del mejor) | `d ∈ {2, 1, 5}` → el ganador por tarea **cae dentro** |

**VEREDICTO: ✅ el proxy de ventana ordena igual que la tarea también en un eje de C.** Se cumplen
las dos condiciones de §5.4 (agregado ≥ 0,90 y ganador por tarea dentro de la frontera δ).
**`OBJECTIVES` no cambia** y §5.5 **no se ejecuta**. El criterio estaba escrito antes de mirar, y
es comprobable: las constantes `SPEARMAN_PASS`/`MIN_POINTS` viven en `scripts/proxy_vs_task.py`,
commiteado **antes** de que el recorrido terminara (protocolo.md §1).

**Las tres reservas, que hay que decir siempre que se cite esto:**

1. **El eje separa poco.** La amplitud de la métrica de ventana en los 6 puntos es **0,0278**
   (contra 0,52 en el eje `lr` de §2), y δ = 0,0153 se come **3 de los 6 puntos**. Por eso el
   Spearman **por run** se hunde a +0,257: no es que el proxy falle, es que el ruido por run
   (±0,09) es tres veces la señal que hay que ordenar. El agregado sobre 5 semillas la recupera.
2. **+1,000 con n = 6 es fuerte pero corto.** Un orden exacto entre 6 puntos tiene 1/720 de salir
   por azar, así que es señal de verdad — pero **6 puntos son 6 puntos**. Lo que este resultado
   descarta con confianza es que el proxy **invierta** el orden al mover la geometría, que era el
   miedo concreto de protocolo.md §2; no demuestra que coincidan en cualquier eje de C.
3. **Un solo eje de C.** `d` cambia cuánto contexto ve la red **sin tocar la fóvea**. Ejes que
   muevan kernels, strides o `n_layers` siguen sin medirse — aunque son los que menos tocan «la
   regla de mirar», así que el riesgo es menor que el de `d`.

### 2 bis.1 El hallazgo colateral: **la periferia no está aportando** (media respuesta a §9.9)

Es el resultado que más dice sobre el proyecto, y sale gratis de este mismo recorrido. `d` es
exactamente el mando del contexto periférico: de `d=1` (2 px de periferia real) a `d=6` (12 px).

- El máximo está en **`d = 2`**, es decir **4 px de contexto**, y `d = 1` —casi sin periferia— queda
  **segundo** en las dos métricas.
- A partir de ahí **más contexto es peor**: `d = 6` (12 px) es de los últimos en ventana y en tarea.
- El coste **no es la explicación**: los s/época apenas se mueven (7,0–8,8), así que el contexto
  extra es prácticamente gratis y **aun así no se paga**.
- ⚠ **Honestidad sobre la fuerza del efecto:** mejor − peor en tarea son **0,0576 = 1,43 SE** de la
  media de 5 semillas (±0,0403). Y `d = 5` rompe la tendencia (0,5342, por encima de `d=4` y
  `d=6`), que es justo el aspecto que tiene el ruido de 1,4 SE. **No se puede afirmar «la periferia
  estorba»; sí se puede afirmar «la periferia no está ayudando de forma medible».**

> ⚠ **Matiz de 2026-08-25, y no menor.** Este párrafo se escribió con `d` como «el mando del
> contexto». Con la ortografía nueva se ve que el mando movía área *y* compresión juntas, y que
> **el recorrido `d5-L4` (24-ago, dataset nuevo, red L4) sale al revés**: el eje sube monótono
> hasta el extremo del rango. O sea que «la periferia no aporta» vale para *aquel* dataset y
> *aquella* red (L2, 20 épocas), **no como afirmación del proyecto**. Cítese con las dos fechas.

**Qué se hace con esto:** protocolo.md §6 y F12 preguntan si fóvea+periferia gana a una CNN plana
equivalente. Este recorrido dice que, **en este dataset y con esta geometría, el margen que ese
experimento podría encontrar es pequeño o nulo** — y lo dice sin construir el control que F12
bloquea. Es información que vale antes de invertir en la comparación. Ojo: encaja con §9.1 y §9.3
(el techo de detección es ~0,66 y el cuello es la red), y con que el dataset sea de 80×60 px, donde
12 px de periferia son un quinto de la imagen.

**Reproducirlo:**

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep proxy-c-d --split val
```

Detalle completo por run y por punto en `data/proxy-c-d-3b.json` (comiteado: son ~30 números, no
carga).

---

## 2 ter. `n_layers=4`, medido con la métrica que manda (2026-08-08)

**Por qué existe este apartado:** el plan de 40 h ([plan-40h.md](plan-40h.md)) concluyó que
`n_layers` 2 → 4 sube el **f1 de ventana** de 0,8756 a 0,9244 con bandas disjuntas. Ventana es el
proxy. Esta sección lleva ese veredicto a la métrica de tarea, sin reentrenar nada: son los
**mismos 20 runs**, 5,4 s de inferencia cada uno.

**Fecha:** 2026-08-08 · **Dataset:** `dirty1000-80px-16px` · **split `val` = 200 imágenes**
(diez veces el de §2 y §2 bis: aquí el `sem` por run es **±0,023**, y el aviso de muestra pequeña
—n < 100— **no salta**) · **Recorrido:** `p40-confirm-n_layers` · **Eje:** `n_layers` (dominio
**C**) · 4 valores × 5 semillas · knobs de F por defecto.

| `n_layers` | ventana (f1) | **tarea** (macro F1) | sem entre semillas | peor–mejor semilla |
|---|---|---|---|---|
| **4** | **0,9244** | **0,7796** | 0,0074 | 0,7532 – 0,7922 |
| 3 | 0,9093 | 0,7644 | 0,0041 | 0,7553 – 0,7761 |
| 5 | 0,8832 | 0,7654 | 0,0053 | 0,7508 – 0,7803 |
| 2 | 0,8756 | 0,7572 | 0,0040 | 0,7489 – 0,7689 |

**1. La decisión aguanta: `n_layers=4` gana también por tarea.** +0,0224 sobre L2, y la diferencia
es más grande que reetiquetar las semillas: **p = 0,032** (permutación exacta, 252 arreglos, dos
colas). Los otros dos pares **no** se separan: L4 vs L3 p = 0,135, L4 vs L5 p = 0,167. El ganador
es el mismo con las dos métricas, y cae dentro de la frontera δ.

**2. ⚠ Pero la ganancia se encoge a la mitad, y las bandas ya NO son disjuntas.** La amplitud de
ventana entre L2 y L4 es 0,0488; la de tarea, **0,0224**. Y donde ventana daba bandas separadas
(peor L4 = 0,9105 > mejor L2 = 0,8804), en tarea **se solapan**: la peor semilla de L4 (0,7532)
cae por debajo de la mejor de L2 (0,7689). **Al citar el resultado del plan de 40 h hay que citar
esto**: «la profundidad gana» sigue siendo cierto sobre la media, pero *una* semilla de L4 no le
gana a *una* de L2.

**3. El criterio de §5.4, aplicado sin tocarlo, dice NO: Spearman agregado +0,800 < 0,90.** Es el
**segundo** eje de C medido y el primero que falla, así que hay que decir por qué y decirlo entero:

- El fallo es **un único intercambio de vecinos**: ventana ordena L3 > L5, tarea ordena L5 > L3.
  Con n = 4 puntos, un solo cambio adyacente hunde el Spearman de 1,000 a 0,800 — no hay valores
  intermedios que sacar.
- Y esos dos puntos **no están separados por nada**: 0,0010 de diferencia en tarea, con `sem` de
  0,0041 y 0,0053, **p = 0,897**. El proxy no se equivoca de orden: **no hay orden** que acertar.
- ⚠ **La causa mecánica sí es real y merece anotarse: el proxy de ventana exagera el hundimiento de
  L5.** La bimodalidad de L5 (plan-40h.md: 0,8471 / 0,8581 / 0,8620 contra 0,9279 / 0,9209, una
  amplitud de **0,081** en ventana) **casi no aparece en la tarea**: las mismas cinco semillas dan
  0,7508 – 0,7803, amplitud **0,029**, y los dos grupos **se entrelazan** (la semilla que peor
  arranca en ventana, 0,8471, puntúa 0,7567 en tarea — por encima de dos de las que sí arrancaron
  en otros puntos). Una red que «no arranca» según el f1 de ventana **sigue reconstruyendo
  párrafos casi igual de bien**.

**Qué se hace con esto — nada, y la razón:** §5.5 **no se ejecuta**. Cambiar `OBJECTIVES` exige que
el proxy ordene *mal*, no que empate dos puntos que la tarea tampoco distingue; aquí el ganador
coincide, la dirección coincide, y el único desacuerdo está bajo el ruido de las dos métricas. Lo
que sí cambia es **cómo se lee un empate**: con `n_layers`, dos puntos separados por menos de δ en
ventana pueden estar separados por **menos todavía** en tarea. Se anota como reserva del proxy en
ejes de C que muevan la profundidad, no como su refutación.

**Colateral, y es una advertencia sobre el método del propio plan:** el **cribado de 1 semilla no
habría visto nada** por tarea. `p40-screen-base` (L2) = 0,7523 y `p40-screen-depth` (L4) = 0,7532
— **+0,0009**, frente al `sem` de ±0,023 de un run. Las dos son la semilla 1, que resulta ser la
**peor** semilla de L4 (números idénticos a `…-0000-n_layers4_seed1`: mismo config, misma semilla,
mismos pesos — comprobación de determinismo que sale gratis). Por ventana el cribado sí separó
(+0,0454). Es exactamente protocolo.md: *un resultado sin N semillas es una anécdota*, y aquí la
anécdota habría apuntado a «la profundidad no hace nada».

Los otros dos resortes del cribado, por tarea y con la misma reserva de 1 semilla:
`p40-screen-width` (`channels=[32,32]`) **0,7242** — el peor de los cuatro, coherente con que
ventana ya lo descartara; `p40-screen-kernel` (`k_center=5`) **0,7594** — el **mejor** de los
cuatro, cuando por ventana era el **peor** (−0,0063 contra la base). Con una semilla y ±0,023 eso
no afirma nada, pero **k_center no queda descartado por la tarea** como lo quedó por la ventana:
es el candidato barato si se vuelve a barrer estructura.

**Reproducirlo:**

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep p40-confirm-n_layers --split val
```

Detalle por run y por punto en `data/p40-n_layers-task.json` (comiteado). Las p de permutación las
imprime el propio script desde `fv.metrics.permutation_test` — exacta hasta C(n+m,n) ≤ 200.000, y
**se niega** por encima en vez de pasar a muestreo en silencio: un p que cambia entre corridas no
decide nada.

---

## 3. Fase 2 — HECHA (2026-07-26): la métrica de tarea, cableada

**Objetivo:** poder pedir el número de tarea de cualquier run, con caché, y verlo en el veredicto
de un recorrido. **No** convertirlo en el objetivo del ranking (§2 lo desaconseja con datos), y
**no** calcularlo por época (§3.7).

### 3.1 Dónde vive el código

**Módulo nuevo `src/fv/task/__init__.py`.** No va dentro de `fv.diagnostics` ni de
`fv.inference`, y la razón importa:

- `fv.diagnostics` es **E×B por ventana** (su docstring dice «A CACHE, not an entity»). Esto es
  **por imagen** y además cruza **A** (la verdad viene de la fuente, no del dataset de ventanas).
- `fv.inference` es F puro: aplicar un modelo a una imagen. Esto **puntúa** F contra A.

Es un cruce de dominios y por tanto un **contrato**: se añade a
[organizacion.md](organizacion.md) §2 como **⑬ (E×A vía F)** con el texto:

> ⑬ **Métrica de tarea (E×A vía F)** — puntuar un run sobre imágenes completas exige la
> **fuente** (A) de la que salió su dataset (B), no solo B: B guarda las imágenes pero **no los
> párrafos verdaderos**. El fingerprint de B protege el split; la fuente se resuelve por
> `manifest["source_id"]`. Si la fuente no existe, se falla con razón — nunca se puntúa contra
> las etiquetas de ventana, que son otra cosa.

### 3.2 La función

```python
def task_score(run_name: str, split: str = "val", *,
               threshold: float = 0.5, stride: int | None = None,
               nms_radius: float | None = None, min_size: float | None = None,
               iou_threshold: float = 0.5,
               store: RunStore | None = None) -> dict:
```

Pasos exactos, en este orden (los tres primeros son **los mismos guardas** que
`fv.diagnostics.table.diagnostics_table`, y se repiten a propósito porque son la puerta):

1. `cfg = store.config(run_name)`; si no hay `provenance.window_dataset.name` →
   `RunError("run_without_provenance", …)` (mismo texto que diagnostics).
2. `manifest = WindowDatasetStore().manifest(ds_name)`; si
   `manifest["fingerprint"] != prov["window_dataset"]["fingerprint"]` →
   `RunError("window_dataset_changed", …)`. **Motivo:** el split cambió, así que las imágenes de
   val ya no son las que ese `best.pt` no vio.
3. `ckpt = store.path(run_name) / "best.pt"`; si no existe →
   `RunError("run_has_no_checkpoint", …)`. **Se puntúa `best.pt`, nunca `last.pt`** — es el
   fichero que sobrevive y el que el ranking ya describe (ver `sweep_trials`).
4. `split` ∈ `{"train","val","test"}` (reusar `fv.diagnostics.table.SPLITS`), si no →
   `RunError("unknown_split", …)`.
5. Índices de imagen del split: **`WindowDatasetStore().split_map(ds_name)[split]`** — ya existe,
   no leer `split.json` a mano (sería un segundo lector del mismo fichero). Si la lista está
   vacía → `RunError("split_empty", …)`.
6. Fuente: `SourceDataset(manifest["source_id"])`. Si `resolve_source` falla, deja subir su
   `SourceError` (ya trae code/message/hint) **pero envuelto con el porqué**: código
   `task_needs_source`, mensaje «la métrica de tarea se mide contra los párrafos de la fuente
   `<id>`, y esa fuente no está», hint «recupera la fuente o mide solo la métrica de ventana».
7. Verdad por imagen: `[b.bbox for b in sample.blocks if b.kind in set(manifest["config"]["target_kinds"])]`.
   **El filtro por `target_kinds` no es opcional**: un dataset extraído de párrafos no se puntúa
   contra líneas.
8. Predicción por imagen: `predict_image(model, sample.load_image(), threshold=…, stride=…,
   nms_radius=…, min_size=…)`, modelo vía `fv.inference.checkpoint.MODEL_CACHE.get(ckpt)`.
   Cajas predichas = `[(p["x0"], p["y0"], p["x1"], p["y1"]) for p in out["paragraphs"]]`.
9. `paragraph_f1(pred, true, iou_threshold)` **por imagen**, guardando el dict completo.

### 3.3 La agregación (decidida, con su razón)

Se reportan **las dos**, y **la primaria es la macro**:

- **macro** = media de los F1 **por imagen**. Es la que manda porque protocolo.md §2 dice «se
  mide **por imagen**» y porque la unidad de muestra es la imagen. Da además `sd` entre imágenes
  → un **error estándar** (`sd/√n_imágenes`) que se puede enchufar directo a la regla de empate
  de `fv.sweeps.winner.tie_delta`: **con la métrica de tarea hay banda de ruido aunque haya una
  sola semilla**.
- **micro** = `tp/fp/fn` sumados sobre todas las imágenes y luego P/R/F1. Es la métrica de
  detección estándar; se reporta para poder comparar con literatura y para ver el efecto de las
  imágenes con muchos párrafos.

**No se inventa un tercer número.** Si macro y micro discrepan mucho, eso es información (unas
pocas imágenes difíciles dominan), no un problema a promediar.

### 3.4 El payload exacto

```jsonc
{
  "run": "…", "split": "val",
  "window_dataset": "…", "source": "local/…",
  "images": 20,                       // n de imágenes puntuadas
  "macro": {                          // PRIMARIA
    "f1": 0.5353, "sd": 0.3721, "sem": 0.0832,
    "precision": 0.55, "recall": 0.54
  },
  "micro": { "f1": 0.5517, "precision": 0.56, "recall": 0.54,
             "tp": 32, "fp": 25, "fn": 27 },
  "mean_iou": 0.74,                   // de los emparejados; null si no hubo ninguno
  "per_image": [                      // para poder ver la cola, no solo la media
    {"index": 5, "f1": 0.8, "tp": 2, "fp": 0, "fn": 1, "mean_iou": 0.778}
  ],
  "knobs": {"threshold": 0.5, "stride": 8, "nms_radius": 8.0,
            "min_size": 4.0, "iou_threshold": 0.5, "window_size": 16},
  "checkpoint": "best.pt",
  "cached": true
}
```

Reglas del payload, que **no** son opcionales:

- `mean_iou` es **`null`** cuando no hubo ningún emparejamiento — nunca 0 (formatos.md §2:
  ausente ≠ cero; `paragraph_f1` ya lo devuelve así).
- `knobs` se **eco**, como ya hace `predict_image`: el número no significa nada sin ellos.
- `macro.sem` se calcula y viaja, porque es lo que convierte «0,53» en «0,53 ± 0,08».

### 3.5 La caché

Misma forma que la de diagnostics (`data/cache/task/<key>.json`), con `_cache_key` extendida:

```python
key = sha256(f"{run}|{fingerprint}|{split}|{ckpt.stat().st_mtime_ns}|"
             f"{threshold}|{stride}|{nms_radius}|{min_size}|{iou_threshold}")[:24]
```

**Los knobs SÍ entran en la clave** (a diferencia del `threshold` de diagnostics, que es un
parámetro de consulta porque re-umbralizar lee scores guardados). Aquí cambiar un knob cambia la
reconstrucción entera: hay que volver a inferir. Guardar el JSON completo, `cached: true/false`
en la respuesta.

### 3.6 API, UI y CLI

- **`GET /runs/{name}/task-score`** con query `split`, `threshold`, `stride`, `nms_radius`,
  `min_size`, `iou_threshold`. Errores por `_http_error` (ya mapea code/message/hint); los
  códigos de §3.2 se añaden a `NOT_FOUND_CODES` solo si son «no existe» (`task_needs_source` no
  lo es: es 400).
- **UI, pantalla Runs / detalle de run**: bloque «Métrica de tarea (párrafo por imagen)» con
  macro ± sem, micro, `mean_iou`, los knobs usados y el n de imágenes. **Con un aviso visible
  cuando `images < 100`**: «con N imágenes el error estándar es ±X: este número no distingue
  diferencias menores que eso» (el mismo espíritu que el empate técnico del veredicto).
- **UI, pantalla Recorridos**: en el bloque de veredicto, **solo para el punto sugerido y el
  mejor** — no para los 35 puntos. Botón «medir la tarea del ganador» que dispara el cálculo, no
  automático: es la diferencia entre 0,6 s y 20 s.
- **CLI**: `fv-oat` y `fv-study --auto` imprimen la métrica de tarea **del ganador sugerido** al
  cerrar, detrás de una bandera `--task-score` (por defecto **apagada**, para que un recorrido
  nocturno no pague inferencia que nadie pidió). ASCII en la salida (cp1252, ver CLAUDE.md).

### 3.7 Lo que esta fase NO hace, y por qué

- **No se calcula por época** dentro del bucle de entrenamiento. Costaría inferencia de imagen
  completa por época y cambiaría el coste de D; además `monitor` es de ventana por diseño.
- **No entra en `OBJECTIVES`** (§2 lo desaconseja con datos). Si el §5 lo contradice para ejes
  de C, entra entonces — y con la decisión escrita.
- **No optimiza los knobs de F.** Fijos y declarados. Barrerlos es otra funcionalidad (barata,
  porque no reentrena) y necesita su propia decisión de diseño.

### 3.8 Coste medido (no estimado)

| | medido |
|---|---|
| una imagen 60×80, ventana 16, stride 8 | **28 ms** |
| un run sobre las 20 imágenes de val | **0,6 s** |
| los 130 runs del repo | **~80 s** |

El cálculo lo domina `build_view` en Python por ventana — **63 ventanas por imagen** con este
dataset (imagen 80×60, ventana 16, stride 8: 9 columnas × 7 filas, comprobado con
`fv.inference.predict._positions`), no el modelo. Si algún día molesta, el arreglo es vectorizar
la construcción de vistas, no bajar el n de imágenes.

### 3.9 Tests que hay que escribir (uno por contrato, tests.md)

| Test | Afirma |
|---|---|
| `test_task_score_matches_a_hand_computed_case` | Sobre un mundo mínimo con 2 imágenes y cajas conocidas, macro y micro salen de `paragraph_f1` — la costura, no la función |
| `test_task_score_refuses_a_changed_dataset` | Fingerprint distinto → `window_dataset_changed`, **sin puntuar nada** |
| `test_task_score_refuses_without_source` | Fuente borrada → `task_needs_source` con razón y arreglo (no cae a las etiquetas de ventana) |
| `test_task_score_is_cached_by_knobs` | Dos llamadas iguales → `cached: true` la segunda; cambiar **un** knob → recalcula |
| `test_task_score_scores_best_not_last` | `best.pt` y `last.pt` distintos → el número es el de `best.pt` |
| `test_task_score_mean_iou_is_null_without_matches` | Sin emparejamientos → `null`, nunca 0 |
| `test_contract_13_task_metric_needs_the_source` | El contrato ⑬ de organizacion.md, en `tests/test_contracts.py` |

### 3.10 Documentación a tocar al terminar

- [organizacion.md](organizacion.md) §2: contrato ⑬ (texto en §3.1).
- [protocolo.md](protocolo.md) §2: marcar la fila «Rankear el recorrido» con el resultado de §2
  de este documento (el proxy vale para D; para C, pendiente).
- [api.md](api.md): la ruta nueva.
- [formatos.md](formatos.md) §5: mencionar `data/cache/task/` como carga. **No hay que tocar el
  `.gitignore`**: `/data/cache/` ya está ignorado (línea 223, verificado 2026-07-26).
- [README.md](../README.md): cómo pedir el número de tarea, con el comando **ejecutado**.
- El bloque de estado de [CLAUDE.md](../CLAUDE.md).

### 3.11 Lo construido, y en qué se desvía de lo escrito arriba

Todo §3 está implementado tal cual, salvo tres cosas que se anotan para que el próximo lector no
crea que se le escapó algo:

1. **`window_dataset` como parámetro llegó ya** (era §6, Fase 4). Es una línea de código y una
   guarda, y sin ella la firma habría que cambiarla después: `task_score(..., window_dataset=…)`
   puntúa contra otro B, y **rechaza** con `holdout_shares_source` si ese B sale de la misma
   fuente que el de entrenamiento. Lo que sigue pendiente de la Fase 4 es **el holdout en sí**
   (generar la fuente), no su cableado. Cuando el B es otro, la comprobación de huella **no
   aplica** (esa huella protege el split del run, no el del holdout).
2. **`fv.task.report`**: un formateador ASCII compartido por `fv-oat` y `fv-study`, en vez de dos
   copias del mismo bloque de `print`. Los CLIs solo aportan la bandera.
3. **El aviso de muestra pequeña viaja en las tres superficies** (UI, los dos CLIs) desde una
   sola regla (`n < 100`), no solo en la pantalla de Runs como decía §3.6.

**Verificado, no razonado:** el número de §2 sale **idéntico** del código nuevo — las 5 semillas
del ganador de `fast-lr-s0-lr` dan 0,5353 de media (1,9 s los 5 runs), que es la fila de la tabla
de §2. Los 7 tests de §3.9 más el ⑬ están en `tests/test_task.py` y `tests/test_contracts.py`
(**107 en verde**); las 12 pantallas Playwright pasan con los dos botones nuevos pulsados; los dos
CLIs corrieron de punta a punta bajo `PYTHONIOENCODING=cp1252`.

---

## 4. Fase 3 — PENDIENTE: dimensionar el dato (el bloqueo real)

**Esto no es código: es una decisión de investigación y una corrida del generador.**

### 4.1 La aritmética, para que se pueda rehacer

`SE = sd / √n_imágenes`, con **sd = 0,4148 medido** — el valor **re-medido el 2026-07-26 sobre los
20 runs ganadores de los 4 recorridos** (§9.4), que sustituye al 0,372 de §2:

| imágenes de val | SE de la métrica de tarea |
|---|---|
| 20 (hoy) | **±0,093** |
| 55 | ±0,056 |
| 154 | ±0,033 |
| **200** | **±0,029** |
| 346 | ±0,022 |
| 500 (el holdout de §6) | ±0,019 |

protocolo.md §3 ya pedía **~2000 imágenes 80/10/10** → 200 de val → **±0,029**, que es del orden
de las diferencias que se quieren distinguir. Es la cifra coherente: **no inventar otra**.

⚠ **La versión anterior de esta tabla usaba sd = 0,372 y la llamaba «la elección conservadora»,
razonando que con modelos mejores la sd bajaría. Se midió, y sube** (§9.4): la sd es **máxima con
modelos intermedios**, porque el F1 por imagen es casi bimodal y un modelo de F1≈0,5 reparte mitad
y mitad, mientras que uno uniformemente malo falla en todas y varía poco. El 0,372 se había
promediado sobre el recorrido `lr` entero, modelos de F1 0,10 incluidos. **La consecuencia es que
hacen falta más imágenes, no menos: el argumento de F11 se refuerza.**

Sigue en pie la regla: al regenerar, **volver a medir la sd** y rehacer esta tabla — con dato nuevo
y modelos mejores puede volver a moverse en cualquier dirección.

### 4.2 ⚠ CORRECCIÓN (2026-07-26): de dónde sale el dato de verdad

**La versión anterior de este §4.2 estaba equivocada en un punto que manda sobre toda la fase**, y
se corrige aquí con lo comprobado en disco. Decía que bastaba `scripts\make_synth_source.py`.
**No basta: ese script genera OTRO problema.**

Lo que hay en `data/sources/`, leído de sus `dataset.json`:

| Fuente | Imágenes | Origen | ¿La generó `make_synth_source.py`? |
|---|---|---|---|
| `local/synth-01` | 60, 96×72 | `{"synthetic": true, "seed": 7}` | **Sí** — barras sintéticas de juguete |
| `local/dirty-paragraphs-fast-80px` | 100, 80×60 | receta `clean-paragraphs` (`recipe_id: clean-paragraphs-copy-627fdbdc`), `spec_version: 1`, **`derived: {from: "dirty-paragraphs-fast-a86efebe", op: "resize", request: {width: 80}, scale: [0.125, 0.125], resample: "lanczos"}`** | **No** |
| `local/dirty-paragraphs-80ancho` | 20000, 80×60 | la misma receta, mismo camino | **No** |

Es decir: **todo lo medido en §2 (y los 130 runs) vive sobre texto renderizado por el proyecto
hermano `image-text-sample-generator`** (canvas 640×480, receta `clean-paragraphs`: 1–4 párrafos
por imagen, ancho 200–320 px, 18–45 palabras, fuentes 11–17 px, fondos solid/gradient/noise/
paper/lines/grid…), **reducido después a 80 px de ancho por un `resize` lanczos**. El generador
local hace rectángulos de barras: otro problema, otra dificultad, otras esquinas.

**Consecuencia:** «regenerar el dato» con `make_synth_source.py` cambiaría **el problema** a la vez
que el tamaño de muestra — exactamente el error que este mismo §4.2 advertía sobre la resolución.
Ningún número nuevo sería comparable con ninguno viejo, y encima no se sabría por cuál de las dos
razones.

**Las tres rutas posibles, con lo que cuesta cada una:**

| Ruta | Qué se hace | Comparabilidad | Coste real |
|---|---|---|---|
| **A — más de lo mismo** (recomendada si se regenera) | 2000 imágenes con la **misma receta** en `image-text-sample-generator` + **resize a 80 px** | El problema es el mismo; solo cambian las imágenes y el n | La corrida del generador (navegador headless, ~horas para 2000) **+ el resize, que NO está portado** (§4.3) |
| **B — juguete** | `make_synth_source.py --count 2000` | **Otro problema.** Sirve para probar el instrumento, **no** para concluir nada sobre la tarea | Minutos |
| **C — no regenerar** | Seguir con 20 imágenes de val | Se conserva todo | Cero, pero la métrica de tarea **no puede decidir entre puntos**, solo informar del ganador |

**La ruta A, paso a paso** (nada de esto está automatizado hoy):

1. En `c:\Desarrollo\image-text-sample-generator`: `\.venv\Scripts\python -m app.serve --port 8001`,
   y por su API `POST /datasets {recipe_id: "clean-paragraphs-copy-627fdbdc", count: 2000}` →
   `POST /datasets/{id}/build` (job; se sondea con `GET /datasets/{id}/build`). La receta exacta
   está **copiada dentro** de `data/sources/dirty-paragraphs-fast-80px/dataset.json` (campo
   `recipe`), así que se puede reproducir aunque el `recipe_id` ya no exista allí.
   **Otra semilla que la de las fuentes actuales** (`4652026051386056742` y
   `1640648715688333905`) — y anotarla.
2. Reducir a 80 px de ancho (§4.3).
3. Copiar la fuente reducida a `data/sources/<nombre>/` (fv lee `labels.jsonl` + `images/` +
   `dataset.json`; el lector es [loader.py](../src/fv/datasets/loader.py) y **exige `labels.jsonl`**).
4. Extraer el dataset de ventanas (§4.4).

### 4.3 El eslabón que falta: el `resize` no está portado

> ✅ **YA NO FALTA (2026-08-13).** F13 cerrada: el resize se portó a `fv.datasets.resize`
> (CLI `fv-resize`), con las dos reglas de abajo conservadas y 13 tests. Lo que sigue de esta
> sección se conserva porque explica **por qué** esas dos reglas no son detalles. Sigue sin
> existir la ruta `POST /sources/{id}/resize` ni su pantalla.

Comprobado (2026-07-26): en `fv` **no existe** el resize de fuentes. `docs/api.md` lo lista como
`POST /sources/{id}/resize` **previsto**, `docs/plan.md` lo dice explícitamente («resize de fuentes
no portado aún»), y `grep -rn resize src/` solo encuentra el `img.resize` del *thumbnail* de la
pantalla de Fuentes. Las fuentes de 80×60 que hay en disco se redujeron **fuera de este repo**.

De dónde se porta, si se decide la ruta A:

- `image-text-finder/src/itf/datasets/resize.py` (336 líneas) — «A' — a derived source: the same A,
  at another resolution (D19)». Es la **composición**: mueve píxeles con `itf.imageops.resize` y
  coordenadas con `itf.geometry.scale_quad`, y **reescribe solo los campos que consume, copiando
  el resto tal cual** (es un segundo productor del formato de otro proyecto).
- Dos detalles suyos que no son detalles y que hay que conservar al portar:
  **la escala se mide de la salida, no del factor pedido** (y son dos escalas, x e y — por eso el
  `dataset.json` de la fuente actual guarda `scale: [0.125, 0.125]`), y **las máscaras se
  remuestrean con NEAREST**, nunca interpolando (interpolar una máscara de etiquetas fabrica
  clases que no existen: la forma continua de *ausente ≠ cero*).

**Alternativa sin portar nada** (más barata y honesta si solo se quiere el dato): hacer el resize
**una vez, con un script de un solo uso** en el scratchpad, que reescriba `labels.jsonl`
multiplicando cada `quad` por la escala medida y guarde las imágenes reducidas — y **declararlo en
el `dataset.json` derivado** con el mismo bloque `derived` que ya usan las fuentes actuales, para
que la procedencia no se pierda. Si el resize se va a hacer más de una vez, se porta.

> **Decisión implícita que conviene tomar explícita:** ¿se porta el resize a `fv` (dominio A,
> `POST /sources/{id}/resize` + pantalla) o se hace fuera y solo entra el resultado? Hoy **no está
> registrada en decisiones.md**; anotarla al abrir la Fase 3.

### 4.4 Extraer el B nuevo (comandos reales, banderas verificadas)

`fv-extract --help` (ejecutado 2026-07-26) acepta **exactamente**: `--source --name --window-size
--stride --val-frac --test-frac --seed`. **No hay `--target-kinds`**: se queda en el default
`("paragraph",)` de `ExtractConfig`, que es justo lo que la métrica de tarea filtra (§3.2 paso 7).

```powershell
.\.venv\Scripts\fv-extract.exe --source local/<fuente-2k> --name paragraphs-2k-b16 `
  --window-size 16 --stride 8 --val-frac 0.1 --test-frac 0.1 --seed 1
```

- **`--window-size 16`**: el mismo del B actual. El contrato ①a atará `N` a él (`derive_base(16)`
  → `N=20, c_frac=0.8, d=2` → fóvea 16, comprobado), así que las redes derivadas siguen siendo las
  mismas y **la comparación red-a-red se conserva**; lo que cambia es el dato.
- **`--val-frac 0.1 --test-frac 0.1`** sobre 2000 imágenes → **200 de val** (SE ±0,029) y 200 de
  test. El split es **por imagen** (`_assign_splits`), que es lo que hace que el n efectivo sea el
  de imágenes.
- **`--stride 8`**: el del B actual. Sube el número de ventanas (coste de entrenamiento), no la
  resolución de la métrica de tarea.
- Presupuesto: `IMAGES_BUDGET_BYTES` = 1 GB; 2000 × 80 × 60 = **9,6 MB**. No estorba, y si alguna
  vez lo hiciera el error lo dice con su razón.

### 4.5 Qué hay que RE-medir después de regenerar (no se hereda nada)

1. **La sd por imagen.** Los ±0,093 de hoy salen de **sd = 0,4148**, ya re-medida sobre los 20 runs
   ganadores (§9.4). **No se dé por hecho que con más dato baja: la primera vez que se razonó eso,
   la medición dijo lo contrario.** Se re-mide con `task_score` sobre unos pocos runs del dato nuevo
   y **se rehace la tabla de §4.1** — y si sube otra vez, hacen falta más imágenes de las
   presupuestadas.
2. **La correlación proxy↔tarea (§2)**, que se midió sobre el B viejo: con otro n de val el ruido
   por run cambia y el Spearman *por run* se moverá aunque el agregado no.
3. **La Fase 3b** (§5), si ya se hubiera hecho sobre el B viejo: su conclusión es sobre ese dato.

### 4.6 Lo que NO se rompe (para que nadie borre nada por las prisas)

- Los 130 runs, 4 recorridos y 4 estudios actuales **siguen cargando y diagnosticándose** contra su
  B viejo: el fingerprint los protege, no los invalida.
- `task_score` sobre ellos sigue funcionando: apunta a su propia fuente por `source_id`.
- Lo único que se pierde es **poder comparar un número nuevo con uno viejo**. Eso no lo arregla
  ningún código: son dos datos distintos.
- **No borrar nada al regenerar.** El B viejo es la referencia de todo lo medido hasta hoy.

### 4.7 La consecuencia, que es lo que hace esto una DECISIÓN DEL USUARIO

**Un dataset nuevo tiene otro fingerprint, así que TODO lo entrenado hasta hoy deja de ser
comparable con lo nuevo.** Los 130 runs, los 4 recorridos y los 4 estudios actuales quedan como
historia. No es un problema técnico (nada se rompe: el fingerprint protege y los diagnósticos de
lo viejo siguen funcionando contra su B viejo), es una decisión de investigación:

> **DECISIÓN DEL USUARIO (pendiente):** ¿se regenera el dato ahora —perdiendo la comparabilidad
> con lo medido hasta hoy— o se sigue con 20 imágenes de val sabiendo que la métrica de tarea
> solo puede usarse como *informe del ganador*, nunca para decidir entre puntos?
>
> **Y, si se regenera, ¿por qué ruta?** (§4.2): A (misma receta del generador hermano + resize, el
> único camino que conserva el problema), o B (generador local de juguete, que sirve para probar el
> instrumento y no para concluir). La ruta A **cuesta más de lo que decía la versión anterior de
> este documento**: una corrida larga del generador **más** el resize, que no está portado.

Registrada como **F11** en [decisiones.md](decisiones.md) §2 (2026-07-26). **Claude no la toma
solo.** Junto a ella quedó **F12**: qué es exactamente «la CNN plana de coste equivalente» del
primer experimento, que hoy no se puede construir (`no_periphery`).

---

## 5. Fase 3b — HECHA (2026-07-26): la receta que se siguió

> **El resultado está en [§2 bis](#2-bis-fase-3b--hecha-2026-07-26-el-proxy-vale-también-para-c):
> ✅ pasa, Spearman agregado +1,000, mismo ganador.** Lo que sigue es la receta tal como se escribió
> antes de ejecutarla; se conserva porque **§5.5 sigue siendo la lista de lo que habría que tocar**
> si algún día otro eje de C la contradice, y porque §5.1/§5.2/§5.3 documentan cómo repetirlo.
> Coste real: **30 runs en ~68 min** de CPU (contra los ~70 min estimados), 12 s de análisis.

Era la fase **más barata y la que más podía cambiar el plan**, y por eso fue antes que la 4.

**Por qué:** §2 validó el proxy sobre `lr` (D). Un eje de C cambia **la vista foveada**, es decir
la regla de mirar. protocolo.md §2 es explícito: *aquí se barre C, así que ninguna métrica de
ranking puede depender de la vista*. Si el proxy de ventana se degrada al barrer geometría, todo
el arrastre de ganadores de un estudio OAT está eligiendo por el número equivocado.

### 5.1 El recorrido: qué se lanza exactamente, y qué cuesta

El eje es **`d`** (el submuestreo de la periferia): cambia **cuánto contexto real ve la red** sin
tocar la fóvea, así que ①a se mantiene en todos los puntos y el dataset no cambia. Es el eje de C
más limpio para esta pregunta.

```powershell
.\.venv\Scripts\fv-oat.exe --name proxy-c-d --window-dataset dirty-paragraphs-fast-80px `
  --axis d --range auto --recipe corta --epochs 20 --seeds 5 --objective f1 --task-score
```

Verificado **hoy**, sin entrenar (probes en el scratchpad):

- `build_search_space(N=20, c_frac=0.8, pen_frac=0.1)` → **`d ∈ [1, 2, 3, 4, 5, 6]`**: 6 puntos.
- Los **6 pasan `check_run`** contra `dirty-paragraphs-fast-80px` (ninguno se descarta). Lo que
  varía: `periph_real` 2→12 px y `original_size` 20→40 px (la fóvea sigue en 16 y el recorte cabe
  en la imagen de 80×60 — por eso ninguno cae con `original_size_exceeds_image`).
- Coste medido en ese dataset: **7,0 s/época de mediana** (min 6,1 / max 9,0, sobre los 65 runs de
  `fast-lr-s0-lr`). Con 20 épocas → **~140 s/run**; 6 valores × 5 semillas = **30 runs ≈ 70 min**
  de CPU limpia.
- ⚠ Esta máquina **hiberna** y **estrangula por temperatura hasta ~5×** en carga sostenida
  (CLAUDE.md). Para dejarlo desatendido: desactivar la suspensión (`powercfg`) y contar con que el
  reloj de pared puede ser mucho mayor que 70 min. Variante barata si hay prisa: `--seeds 3`
  (18 runs, ~42 min) — pero con 3 semillas la banda de ruido está peor estimada y la regla de
  empate se vuelve más laxa.
- `--task-score` al final imprime la métrica de tarea del ganador sugerido; **no sustituye** al
  análisis de §5.3, que necesita los 30 runs, no uno.

### 5.2 La pieza de código que falta: `spearman`

**No hay scipy en el venv** (comprobado: `scipy`, `pandas` y `matplotlib` no están instalados;
sí `numpy 2.5.1`, `torch 2.13.0+cpu`, `yaml`). Así que la correlación de rangos hay que
escribirla. **Dónde: `fv/metrics.py`** — es puro numpy, no importa nada de `fv`, y es *el* sitio
donde vive «qué significa cada número» (un Spearman calculado dentro de un script es un número que
nadie puede testear, y este proyecto ya se quemó con definir un número dos veces).

```python
def spearman(a, b) -> float | None:
    """Correlación de rangos. Empates -> rango MEDIO. None (nunca 0) si alguna
    de las dos series es constante: ahí la correlación no está definida, y 0
    se leería como «no correlacionan», que es otra cosa (formatos.md §2)."""
```

Implementación mínima (rango medio + Pearson sobre los rangos) y **números dorados ya calculados**
para el test, porque sin scipy no hay contra quién comparar:

| caso | x | y | `spearman` |
|---|---|---|---|
| monótona perfecta | `[1,2,3,4,5]` | `[10,20,30,40,50]` | **1,0** |
| inversa perfecta | `[1,2,3,4,5]` | `[50,40,30,20,10]` | **−1,0** |
| con empates en y | `[1,2,3,4,5]` | `[5,6,7,8,7]` | **0,8207826816681233** |
| ruidosa | `[0.10,0.20,0.30,0.40,0.50,0.60]` | `[0.11,0.30,0.25,0.55,0.44,0.70]` | **0,8857142857142857** |
| serie constante | `[1,2,3]` | `[7,7,7]` | **None** (no 0, no NaN hacia fuera) |

Test: `tests/test_metrics.py::test_spearman_matches_hand_computed_cases` con esa tabla, más
`test_spearman_is_none_for_a_constant_series`.

### 5.3 El script de análisis: `scripts/proxy_vs_task.py`

Precedente exacto de forma y tono: `scripts\verify_axes.py` (corre algo real e imprime un
veredicto). **CLI propuesto**, todo explícito:

```powershell
.\.venv\Scripts\python scripts\proxy_vs_task.py --sweep proxy-c-d --split val [--json out.json]
```

Qué hace, en orden (todas las piezas ya existen; el script **no** calcula ninguna métrica por su
cuenta):

1. `trials = sweep_trials(sweep)` → por run: `value` (la métrica de **ventana**, medida en la
   época que guardó `best.pt`), `point`, `status`. Descartar los `value is None` **contando
   cuántos** y diciéndolo (un run sin checkpoint no es un cero).
2. `task = task_score(run, split)` por cada run → `macro["f1"]` y `macro["sem"]`. Coste: ~0,4 s
   por run con 20 imágenes de val (medido); 30 runs ≈ **12 s**. Con caché, la segunda pasada es
   instantánea.
3. **Spearman por run** sobre las dos listas emparejadas (n = 30).
4. **Spearman agregado por valor del eje**: `suggest_winner(sweep)["trials"]` ya devuelve los
   grupos por valor del eje con `value` (media de ventana entre semillas) y `runs`; el valor de
   tarea del grupo es la media de `macro["f1"]` de esos `runs`. n = 6 puntos.
5. Imprimir la tabla igual que §2 (punto, ventana, tarea, n_seeds) **ordenada por ventana**, los
   dos Spearman, y el veredicto de §5.4. ASCII (cp1252).
6. `--json` guarda todo el detalle (por run y por punto) para poder rehacer la tabla sin
   reentrenar ni re-inferir.

**Lo que el script NO debe hacer**: recalcular `paragraph_f1` por su cuenta, leer `metrics.jsonl` a
mano, ni «rellenar» un run sin checkpoint. Cada número sale de su única definición.

### 5.4 El criterio, escrito ANTES de mirar (protocolo.md §1)

Se compara **el agregado**, porque es lo que decide un ganador; el por-run se reporta para ver
cuánto ruido hay, no para decidir.

- **✅ El proxy vale también para C** si: `spearman_agregado ≥ 0,90` **y** el punto ganador por
  tarea **cae dentro de la frontera δ** del ganador por ventana (`suggest_winner(...)["frontier"]`).
  La frontera, y no el top-1, porque el propio proyecto ya decidió que un ganador dentro de la
  banda de ruido es un empate (protocolo.md §1.5): exigir que coincida el top-1 exacto sería más
  duro que la regla con la que se elige de verdad. → Se anota en §2 bis y **no se cambia nada**.
- **❌ El proxy NO vale para C** si: `spearman_agregado < 0,90` **o** el ganador por tarea queda
  **fuera** de la frontera δ. → Ver §5.5.
- **⚠ Resultado no concluyente** si: menos de 4 puntos con valor, o la serie de ventana es
  prácticamente constante (todos los `d` empatan). Entonces el recorrido no distingue nada y el
  Spearman no significa nada: **repetir con más épocas** (el efecto de `d` puede necesitar más
  entrenamiento para aparecer) antes de concluir.

**Guardar el resultado en este documento como §2 bis**, con: fecha, dataset, nombre del recorrido,
la tabla, los dos Spearman, el veredicto y el nº de semillas. Un número de correlación sin fecha ni
dataset no vale nada.

### 5.5 Si el proxy NO vale: todo lo que hay que tocar para que la tarea sea el objetivo

Esto es lo que más fácilmente se subestima, así que va enumerado. **No es «añadir una línea a
`OBJECTIVES`».** Comprobado leyendo el código (2026-07-26):

1. **`fv/sweeps/spec.py`**: `OBJECTIVES = {"f1": "max", "pos_err_px": "min", "loss": "min"}` →
   añadir `"paragraph_f1": "max"`.
2. **`fv/sweeps/runner.py::sweep_trials`** — *el detalle que rompe en silencio*: hoy el valor de un
   punto sale de `(rec.get("val") or {}).get(objective)`, donde `rec` es el **registro de época**
   que guardó `best.pt` (`checkpoint_record`). El `val` de una época contiene **solo**
   `{loss, f1, precision, recall, pos_err_px}` (lo escribe `fv/training/loop.py::evaluate`). Con
   `objective = "paragraph_f1"` **todos los puntos saldrían con `value: None`** y el recorrido
   diría «no tiene puntos con valor aún» — sin error, sin pista. Hace falta una rama explícita:
   si el objetivo es de tarea, el valor se obtiene con `task_score(run, split)["macro"]["f1"]`
   (que ya puntúa `best.pt`, que es justo la semántica que `sweep_trials` documenta), y
   `value_from` pasa a decir `"task"` en vez de `"checkpoint"`.
3. **El aviso `monitor_matches_objective`**: hoy compara `monitors == [f"val_{objective}"]`. Con un
   objetivo de tarea **nunca** puede coincidir (no hay `val_paragraph_f1` por época), así que el
   aviso se dispararía siempre y dejaría de significar algo. Hay que darle un texto propio: «el
   checkpoint lo elige el monitor de ventana; el ranking mide la tarea» — que es cierto y es
   información, no una alarma repetida.
4. **La banda de ruido**: `task_score` devuelve `macro["sem"]` **por run** (dispersión entre
   imágenes). `aggregate_seeds` calcula su propia `value_sem` (dispersión entre semillas). Son dos
   ruidos distintos y **no se suman a la ligera**; decidir explícitamente cuál alimenta
   `tie_delta` (propuesta: seguir usando el de semillas, y **mostrar** el de imágenes al lado —
   pero es una decisión, no un detalle).
5. **El coste del ranking deja de ser cero**: cada lectura del ranking dispara inferencia de imagen
   completa por punto. Con la caché es 0,4 s/run la primera vez y ~0 después, **pero la UI sondea
   `/sweeps/{n}/trials` cada 3 s**: hay que asegurarse de que el valor sale de caché o el sondeo
   se vuelve carísimo. Alternativa: calcularlo al terminar cada punto (en el runner) y guardarlo
   en el `summary.json` del run.
6. **Las puertas y los vocabularios** que ya sirven `OBJECTIVES` y que heredarían el valor nuevo
   sin tocarlos (verificado por grep): `check_sweep` (H), `fv/studies/driver.py::validate_plan`
   (I), y `GET /sweeps/axes` → los `<select>` de **Recorridos** y **Estudios**. Este último es el
   que hace que no haya que tocar el front: sirve la lista desde la definición única.
7. **El contrato ⑨** hay que releerlo con esto puesto: la métrica de tarea está definida en
   píxeles de la imagen original, así que **sí** puede rankear un espacio que barre C — es la
   razón de que exista. Anotarlo en organizacion.md §2 ⑨ como el caso que lo cumple.
8. **La Fase 3 tiene que estar hecha antes.** Rankear con ±0,093 de ruido por run es peor que
   rankear con el proxy: se estaría eligiendo por el ruido con más ceremonia.
9. Tests: uno que afirme que un recorrido con objetivo de tarea **rankea** (no devuelve todo
   `None`) — la costura del punto 2, que es la que se rompe sola.

---

## 6. Fase 4 — PENDIENTE: el holdout

Heredado de protocolo.md §3 (paso 0a). El **cableado ya está** (§3.11); lo que falta es el dato y
tres piezas pequeñas. Todo lo operativo, sin dar nada por supuesto:

### 6.1 Las reglas que no se negocian

- **Fuente propia**, nombre `<fuente>-holdout`, **de la que jamás se extrae entrenamiento** — la
  fuga se hace físicamente imposible en vez de confiarla a un flag.
- **Misma configuración del generador** que la fuente de entrenamiento, **otra semilla**. Con la
  corrección de §4.2 esto significa: la misma receta `clean-paragraphs` del proyecto hermano +
  el mismo resize a 80 px, **no** `make_synth_source.py` (que haría otro problema, y entonces el
  holdout mediría la generalización a otro dataset, que es otra pregunta).
- **Tamaño ~500 imágenes** (protocolo.md §3). Con **sd 0,4148** (§9.4) → **SE ±0,019**. Coste de medirlo:
  ~19 ms por imagen (medido) → **~9,5 s por run**. Es barato: lo caro es generarlo.
- **Se toca una sola vez, al final, y solo con el ganador.** El val hace dos trabajos (elegir
  `best.pt` y rankear), así que el val del ganador está **sesgado al alza** y no se reporta.

### 6.2 Extraer el B del holdout: 100% test, verificado

El holdout es *otro dataset de ventanas*. Para la métrica de tarea **solo se usan `manifest`
(`source_id`, `config.target_kinds`) y `split.json`** — ni `windows.npz` ni el `window_size`
entran en el número. Aun así se extrae con el mismo `window_size` (16) para poder mirarlo también
por ventana si hiciera falta.

```powershell
.\.venv\Scripts\fv-extract.exe --source local/<fuente>-holdout --name <fuente>-holdout-b16 `
  --window-size 16 --stride 8 --val-frac 0 --test-frac 1 --seed 1
```

**Verificado hoy** (extracción real sobre `local/synth-01`, 60 imágenes): `--val-frac 0
--test-frac 1` deja `windows_per_split = {train: 0, val: 0, test: 5280}` y `split.json` con las
**60 imágenes en `test`**. Es lo que se quiere: **todo el holdout es holdout**, y de paso ese
dataset **no se puede usar para entrenar** — `check_measurable` lo rechaza con
`no_validation_split`, que aquí es una virtud, no un estorbo.

### 6.3 Cómo se pide el número

```powershell
Invoke-RestMethod ("http://localhost:8010/runs/<ganador>/task-score" +
  "?window_dataset=<fuente>-holdout-b16&split=test") | ConvertTo-Json -Depth 3
```

- La guarda `holdout_shares_source` salta si ese B sale de la misma fuente que el B de
  entrenamiento (400 con razón). **Comprobado en vivo** que el camino legítimo (fuente distinta)
  devuelve 200.
- Cuando el B es otro, **la comprobación de huella no aplica** (esa huella protege el split del
  run, no el del holdout) — está así a propósito y escrito en el código.
- Se reporta `macro.f1 ± macro.sem` **con el n de imágenes**, y `micro` al lado. Nada de
  redondear a un solo número sin banda.

### 6.4 Las tres piezas que faltaban (la 1 HECHA; 2 y 3 son decisiones)

1. **UI — HECHA (2026-07-26).** El bloque de tarea del **detalle de un run** trae dos `<select>`
   (**dataset**, alimentado por `GET /window-datasets`, con «el del propio run» como opción por
   defecto; y **split**), y al elegir un dataset distinto del propio aparece el aviso de que **el
   holdout se toca una sola vez, al final y solo con el ganador**, explicando *por qué* (el val
   hace dos trabajos, así que su número está sesgado al alza). El selector solo se ofrece donde
   tiene sentido: `TaskScore` lo recibe por prop `chooser`, y el **veredicto de Recorridos no lo
   lleva** — allí se mide el val del ganador, y ofrecer apuntar al holdout mientras aún se está
   eligiendo sería justo lo que el protocolo prohíbe.
2. **Registro de que se tocó — HECHA (2026-07-26, F14 decidida por el usuario).**
   `fv.task.record_holdout_touch` anexa una línea a **`runs/<run>/holdout.jsonl`** por cada
   medición contra un B de holdout, con `{when, window_dataset, source, split, images, f1, sem,
   knobs, checkpoint, from_cache}`. **Se escribe también cuando el número sale de caché** — ese era
   el punto: el segundo vistazo es gratis, y por eso era el invisible. `task_score` devuelve
   `holdout_touches` (el conteo) y la UI lo enseña en ámbar sobre el propio bloque, porque un
   registro que nadie abre no vigila nada.
   El registro es **append-only y no bloquea**: dice cuántas veces se ha mirado, no impide mirar.
   Convertir la caché en algo que escribe en el artefacto del run era la parte que había que
   decidir, y está decidida.
3. **Marcar la fuente como holdout — HECHA (2026-07-26).** Qué cuenta como holdout lo dice **una
   sola función**, `fv.task.is_holdout_source(source_id, source_meta)`, con dos señales: el campo
   **`"holdout": true`** del `dataset.json` de la fuente y, como respaldo, el convenio de nombre
   **`-holdout`**. El campo explícito **manda en los dos sentidos** — una fuente puede declararse
   *no* holdout pese a su nombre, porque un convenio jamás debe pisar una afirmación. El
   `dataset.json` se lee con `fv.datasets.loader.source_meta`, el **mismo** lector que
   `discover_sources` (antes eran dos: se unificó al escribir esto).
   Ambas señales están en el README.

### 6.5 Tests de esta fase — **los tres primeros ESCRITOS y en verde (2026-07-26)**

| Test | Afirma | |
|---|---|---|
| `test_task_score_scores_another_dataset` | Con `window_dataset=<otro B>` (otra fuente) puntúa y el payload dice **ese** dataset y **esa** fuente | ✅ |
| `test_holdout_dataset_can_be_100_percent_test` | `val_frac=0, test_frac=1` → `split.json` mete todas las imágenes en `test`, y `check_run` lo rechaza para entrenar (`no_validation_split`) | ✅ |
| `test_task_score_on_holdout_ignores_the_run_fingerprint` | Reconstruido el B **del run**, puntuar contra el holdout **sigue funcionando** (esa huella no protege este número) | ✅ |
| `test_holdout_touch_is_recorded` | Dos llamadas dejan dos líneas, **y la segunda sale de caché** — más: puntuar el val propio no escribe nada | ✅ |
| `test_holdout_is_recognised_by_flag_over_name` | El campo `"holdout"` gana al nombre **en los dos sentidos** | ✅ |

Viven en `tests/test_task.py` con una fixture `holdout` que construye **una segunda fuente y un B
100 % test** sobre ella — así el camino del holdout queda fijado **antes** de que exista el dato
real, y el día que la fuente exista no hay que re-deducir nada.

**Lo que queda de la Fase 4 es, literalmente, la fuente**: una corrida del generador hermano con la
misma receta y otra semilla, más el resize (§4.3, F13) — **aparcado con F11**. Todo el código está
construido y probado.

---

## 7. Orden recomendado y coste

| Fase | Qué | Coste real | Estado |
|---|---|---|---|
| ~~2~~ | ~~Cablear `task_score` (§3)~~ | — | ✅ **HECHA 2026-07-26** |
| ~~3b~~ | ~~Validar el proxy sobre el eje `d` (§5)~~ | **68 min** de CPU (30 runs) + `spearman` + el script + **12 s** de análisis | ✅ **HECHA 2026-07-26 → §2 bis: PASA (+1,000)** |
| **3** | Regenerar el dato (§4) | Ruta A: corrida larga del generador hermano + el resize, no portado (§4.3) | ⏸ **APLAZADA (F11 cerrada 2026-07-26: no se regenera por ahora)** |
| **4** | Holdout (§6) | El **código está hecho**; falta la fuente (~500 imágenes del generador hermano) | 🟡 **CÓDIGO HECHO** (UI, guardas, registro F14, 5 tests); **bloqueada por el dato, con F11** |

**Cómo queda el plan tras las decisiones del 2026-07-26:**

- La **3b pasó**: el proxy de ventana ordena igual que la tarea también en un eje de C, así que
  `OBJECTIVES` no cambia y **§5.5 no se ejecuta**. El ranking barato se queda.
- La **3 se aplaza** (F11): se sigue con 20 imágenes de val, y la métrica de tarea es **informe del
  ganador**, nunca criterio para elegir entre puntos. La UI y los CLIs ya lo dicen en pantalla
  siempre que n < 100. Se reabre cuando el ruido estorbe de verdad — la aritmética de §4.1 está
  lista para rehacer la cuenta.
- La **4 queda con todo el código escrito y probado** y solo esperando su fuente, que depende de la
  3. Esto es a propósito: fijar el camino con tests **antes** de tener el dato es lo que evita
  re-deducirlo el día que aparezca.
- Los **knobs de F no se tocan** (F15): la medición de §9.2 queda registrada para cuando se reabra.

**Lo que sigue teniendo más valor por CPU gastada** (ninguna es de este documento):

1. **La red es el cuello de botella** — §9.1 (techo 0,97) y §9.3 (techo de **detección** ~0,66) lo
   dicen a coro. El trabajo está en detectar esquinas, no en reconstruir párrafos.
2. **La periferia no está aportando de forma medible** (§2 bis.1). Antes de invertir en el control
   de F12, conviene saber si eso cambia con otra fóvea o con otro dataset.
3. **Las 7 imágenes que fallan siempre** (§9.5): un modo de fallo concreto que mirar con
   Diagnóstico/Predecir, y que vale más que cualquier décima de F1 promedio.

---

## 8. Resumen para quien lea esto en frío

1. La Fase 2 existe: `src/fv/task/__init__.py` (es `fv/diagnostics/table.py` con «por imagen» en
   vez de «por ventana» y con la verdad viniendo de A). Léelo antes de tocar métricas.
2. **No añadas `paragraph_f1` a `OBJECTIVES`.** §2 (eje de D) y §2 bis (eje de C) miden que el
   proxy barato **ordena igual**, así que no hay nada que ganar y sí un coste que pagar. §5.5
   sigue ahí como la lista de lo que habría que tocar el día que algún eje lo contradiga — y su
   punto 2 es el que rompe en silencio (el ranking se quedaría en `None` sin avisar).
3. **No regeneres el dataset**: F11 está cerrada en «no por ahora». Si se reabre, **lee §4.2
   antes** — el dato de verdad no lo hace el generador local, y el resize no está portado (F13).
4. Los knobs de F entran en la clave de caché; el `threshold` de diagnostics **no** entra en la
   suya. No es una incoherencia: allí re-umbralizar lee scores guardados, aquí hay que re-inferir.
   Y **no los cambies** sin reabrir F15, aunque §9.2 mida que los defaults son malos: mover uno
   mueve todos los números reportados y la caché no avisa.
5. Todo número que salga de aquí viaja con su `sem` y su n de imágenes. Un F1 de tarea sin banda
   es exactamente el error que este proyecto arregló en el ranking. El `sem` está **validado por
   bootstrap** (§9.6), así que puedes fiarte de él: lo que falta es `n`, no fórmula.
6. Casi todo lo de §9 **no necesita entrenar nada** — 8 de 10 hechas así. Si buscas la siguiente
   pregunta que valga la pena, mira el final de §7: la respuesta corta es **la red detecta mal las
   esquinas**, y ahí es donde hay 0,33 de F1 sobre la mesa.

---

## 9. Pruebas que valdría la pena hacer (9.1, 9.4 y 9.5 HECHAS el 2026-07-26)

Ordenadas por **valor / coste**. Las cuatro primeras **no entrenan nada**: usan runs que ya
existen, así que cuestan segundos o minutos. Cada una dice qué pregunta contesta y qué se hace con
la respuesta — una medición sin decisión asociada no se hace.

### 9.1 El techo de la reconstrucción — HECHA: **0,97. El cuello de botella es la red**

**Pregunta:** si las esquinas fueran perfectas, ¿qué F1 de párrafo daría la reconstrucción TL→BR
actual? Ese número es **el techo** de todo lo que se puede ganar mejorando la red.

**Cómo:** alimentar `fv.inference.predict._nms` + `_reconstruct` con las esquinas **verdaderas**
(las de la fuente, `_corners_of(bbox)` de `extract.py`, con `score=1.0`), sin modelo, y puntuar con
`paragraph_f1` sobre las mismas imágenes de val. Un script de scratchpad de ~40 líneas. **Coste:
segundos, cero entrenamiento.**

**Medido** (2026-07-26, `dirty-paragraphs-fast-80px`, 20 imágenes de val, 49 párrafos, IoU ≥ 0,5):

| | macro F1 | micro F1 | imágenes perfectas |
|---|---|---|---|
| reconstrucción sola | **0,9700** (sd 0,134, sem 0,030) | 0,9691 (tp 47, fp 1, fn 2) | **19/20** |
| NMS + reconstrucción | **0,9700** | 0,9691 | 19/20 |
| *el mejor run real de hoy* | *0,6448* | *0,6512* | — |

**La respuesta, y lo que se hace con ella:** el techo es **0,97**, así que **la red es el cuello de
botella** y el esfuerzo va ahí — no a `_reconstruct`. Los ~0,33 que separan al mejor modelo del
techo son **todos** de detección de esquinas. Dos hallazgos colaterales:

- **El NMS no cuesta nada**: con las esquinas verdaderas no suprime ni una (`nms_radius=8` con
  ventana 16). La sospecha de que dos párrafos vecinos se comieran entre sí **no se cumple** en
  este dataset.
- **El único fallo del techo es real y diagnosticable.** En la imagen 67 (3 párrafos) `_reconstruct`
  emparejó el **TL del párrafo 3** `(6,2, 0,4)` con el **BR del párrafo 2** `(73,6, 8,7)` y produjo
  una caja que cruza dos párrafos, perdiendo los dos (F1 0,4). Es la heurística heredada
  enseñando su costura: **ignora TR y BL por completo** y no comprueba ninguna consistencia
  geométrica.
  ⚠ **Matiz honesto:** con esquinas perfectas todos los scores valen 1,0, así que el desempate de
  `_reconstruct` (`score > best`, estricto) se resuelve por **orden de lista**, que es arbitrario.
  Con scores reales ese emparejamiento concreto puede salir distinto. El 0,97 es el techo *de esta
  heurística con scores uniformes*; el orden de magnitud —«el techo está cerca de 1, no de 0,7»—
  es el que manda para decidir dónde invertir, y ese no depende del desempate.
- **Cruce con 9.5:** la imagen 60 falla en **20/20** réplicas de los mejores modelos y su techo es
  **1,000**. No es un problema de reconstrucción ni de dato: es la red.

### 9.2 Barrer los knobs de F — HECHA: **los tres defaults están mal, y el óptimo es el mismo**

**Pregunta:** ¿cuánto F1 de tarea se está dejando en la mesa por usar `threshold=0.5`,
`stride=n//2`, `nms_radius=n/2`, `min_size=4` **por defecto, sin haberlos mirado nunca**?

**Cómo:** `task_score` ya acepta los cuatro y ya cachea por ellos. Un barrido pequeño sobre un run
bueno: `threshold ∈ {0.3,0.4,0.5,0.6,0.7}` × `stride ∈ {4,8}` × `nms_radius ∈ {4,8,12}` = 30
llamadas × 0,4 s ≈ **12 s por run**. Hacerlo sobre 3 runs de calidad distinta para ver si el óptimo
es el mismo (si no lo es, los knobs son parte de la comparación y no un ajuste global).

**Medido** (2026-07-26, 30 combinaciones × 3 runs de calidad deliberadamente distinta, 143 s):

| run | F1 con el default | F1 con el óptimo | ganancia | puesto del default |
|---|---|---|---|---|
| bueno (`fast-lr-2-…-0018`) | 0,6448 | **0,7095** | **+0,065** | 16/30 |
| medio (`batch_size-1-…-0001`) | 0,4800 | **0,6667** | **+0,187** | 24/30 |
| malo (`fast-lr-s0-lr-0000`, lr 1e-4) | 0,3017 | **0,5626** | **+0,261** | 19/30 |

**El óptimo es EL MISMO en los tres**: `threshold≈0,3`, `stride=4`, `nms_radius=12`. Es decir, es un
ajuste **global**, no parte de la identidad de cada run — comparar runs con knobs fijos **no está
sesgado en el orden**, pero el número absoluto está infravalorado entre 0,06 y 0,26.

⚠ **La primera rejilla no acotaba su óptimo**: los tres knobs ganaban en el **borde** ofrecido
(0,3 / 4 / 12), y una rejilla que no acota su óptimo no ha encontrado ninguno. Se extendió
(2026-07-26, 46 s) y ahora **los tres son óptimos interiores**:

- `threshold`: 0,05 → 0,49 · 0,1 → 0,59 · **0,2–0,3 → 0,67–0,71** · 0,4 → 0,67. Pico plano en
  ~0,25. Por debajo se dispara la precisión a la baja (0,40 con th 0,05) sin ganar recall.
- `nms_radius`: **12** · 16 → 0,69 · 20 → 0,66 · 24 → 0,65, y 4/8 ya eran peores. Interior.
- `stride`: 2 → 0,68 · **4 → 0,71** · 8 → 0,68. Interior. Bajar más **empeora**, no solo cuesta.

En unidades de la ventana (n=16) el óptimo es `stride = n/4` y `nms_radius = 3n/4`, contra los
`n/2` y `n/2` de hoy.

**Qué se hace con la respuesta — y qué NO hago solo:** §3.7 aplazó esto diciendo que «necesita su
propia decisión de diseño», y la necesita: cambiar los defaults **mueve todos los números que el
proyecto ha reportado** (la tabla de §2 incluida) y **la caché no avisa**, porque los knobs entran
en su clave y simplemente se recalcularía otra cosa. Registrado como **F15** en
[decisiones.md](decisiones.md). **No se tocan mientras la Fase 3b está midiendo**: usa los mismos
defaults que §2 a propósito, para que las dos correlaciones sean comparables.

**Y un hallazgo que importa a F11 más que el propio óptimo:** las ganancias son **desiguales** —el
run malo gana +0,26 y el bueno +0,065—, así que los knobs buenos **comprimen las diferencias entre
runs**: la separación bueno↔malo pasa de **0,343** (defaults) a **0,147** (óptimo), mientras el
`sem` se queda en ~0,08. Dicho de otro modo: **con los knobs bien puestos, la métrica de tarea
distingue PEOR entre modelos en relación con su ruido**, y buena parte de lo que hoy parece
«diferencia de calidad de modelo» es handicap de los knobs. Refuerza que el bloqueo es el tamaño
del val.

### 9.3 La curva F1-vs-IoU — HECHA: **falla detectar Y falla localizar, sin meseta**

**Pregunta:** con `iou_threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`, ¿cómo cae el F1?

**Cómo:** seis llamadas a `task_score` por run (ya parametrizado, ya cacheado). **Coste: 2,4 s por
run.** Se puede hacer sobre los 5 seeds del ganador actual.

**Medido** (2026-07-26, las 5 semillas del ganador de `fast-lr-s0-lr`, knobs por defecto):

| IoU | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 |
|---|---|---|---|---|---|---|
| media de las 5 semillas | 0,6584 | 0,5722 | **0,5353** | 0,4631 | 0,3438 | 0,1800 |
| caída respecto al anterior | — | −0,086 | −0,037 | −0,072 | −0,119 | −0,164 |

**La respuesta es «las dos cosas», y eso separa el trabajo en dos:**

1. **Techo de detección ≈ 0,66.** Ni aflojando el IoU a 0,3 se pasa de 0,658: **un tercio de los
   párrafos no se detecta en absoluto**, con criterio de solape casi regalado. Eso no lo arregla
   afinar la posición.
2. **No hay meseta, y la caída se acelera** (−0,04 en 0,4→0,5, pero −0,16 en 0,7→0,8): de los que
   sí se emparejan, muy pocos se emparejan **bien**. Es localización, y apunta a `pos_err_px` y a la
   cabeza de posición.

**Y una consecuencia sobre el propio 0,5:** el umbral heredado cae justo en el tramo **más plano**
de la curva (−0,037), que es el mejor sitio posible para un umbral —el número es poco sensible a
elegirlo mal— pero es una **casualidad afortunada**, no una justificación. Ahora sí está medido.

### 9.4 Re-medir la sd por imagen — HECHA: **sube a 0,4148, no baja**

**Pregunta:** la aritmética de §4.1 usa **sd = 0,372**, medida con modelos de F1 0,3–0,54. ¿Cuánto
vale con los mejores runs de hoy (`batch_size-*`, `fast-lr-*`)?

**Cómo:** `task_score` sobre los ganadores de los 4 recorridos y leer `macro["sd"]`. **Coste:
~10 s.** Rehacer la tabla de §4.1 con el valor nuevo.

**Medido** (2026-07-26, los **20 runs** de los puntos ganadores de los 4 recorridos —
`fast-lr-s0-lr`, `fast-lr-2-s0-lr`, `batch_size-1`, `batch_size-2`— × 5 semillas):

| | valor |
|---|---|
| sd por imagen, media de las 20 réplicas | **0,4148** |
| rango (min / max) | 0,3468 / 0,4507 |
| sd del **mejor** run (`fast-lr-2-…-0018`, F1 0,6448) | 0,3468 |
| lo que decía §4.1 | 0,3720 |

⚠ **La respuesta contradice el supuesto del documento.** §4.1 escribió que 0,372 era la elección
*conservadora* porque «con modelos mejores la sd puede bajar». **Sube a 0,4148.** La razón, una vez
vista, es obvia y merece quedar escrita: **la sd es máxima con modelos intermedios**, porque el F1
por imagen es casi bimodal (0 o 1) y un modelo de F1≈0,5 reparte mitad y mitad; un modelo
**uniformemente malo** acierta poco en todas y su varianza **entre imágenes es pequeña**. El 0,372
original se promedió sobre el recorrido `lr` **entero**, que incluía modelos de F1 0,10 — y esos
bajaron la media. La relación calidad↔sd sí existe (el mejor run es el de menor sd, 0,3468), pero
**por debajo de 0,372 solo llega el mejor de 20**.

**Qué se hace con la respuesta:** el argumento de F11 **se refuerza**, no se debilita — hacen falta
*más* imágenes de las que decía la tabla. §4.1 queda rehecha con 0,4148.

### 9.5 macro vs micro — HECHA: **empatan; pero el fallo se concentra en las mismas imágenes**

**Pregunta:** `task_score` ya devuelve las dos y `per_image` entero. ¿La diferencia macro−micro es
sistemática, y son siempre **las mismas** imágenes las que fallan?

**Cómo:** con los `per_image` de los 5 seeds del ganador (ya cacheados), contar cuántas imágenes
tienen F1 = 0 en las 5 réplicas. **Coste: segundos.**

**Medido** (2026-07-26, las mismas 20 réplicas de 9.4):

- **macro − micro = −0,0036 de media** (rango −0,038 a +0,039). **No mandan** las imágenes con
  muchos párrafos: las dos agregaciones dicen lo mismo, y eso es tranquilizador — el `macro` puede
  seguir siendo el primario sin estar contando otra cosa.
- **El fallo NO rota: se concentra.** Imágenes con F1 = 0, contadas sobre las 20 réplicas:

  | imagen | 60 | 48 | 20 | 34 | 87 | 8 | 16 | 5 | resto (10 imgs) |
  |---|---|---|---|---|---|---|---|---|---|
  | réplicas donde falla | **20/20** | 18/20 | 16/20 | 14/20 | 12/20 | 10/20 | 8/20 | 5/20 | ≤ 3/20 |

  **7 de 20 imágenes cargan casi todo el fallo**, y la 60 falla **siempre**, con los 20 modelos.

**Qué se hace con la respuesta:** hay **modo de fallo concreto** que mirar con Diagnóstico/Predecir,
y no es de reconstrucción — 9.1 mide el techo de la imagen 60 en **1,000**. La red no ve esas
esquinas. Mirar qué tienen en común (¿párrafos pegados al borde? ¿fondo `lines`/`grid`? ¿4 párrafos?)
vale más que cualquier décima de F1 promedio.
⚠ **Y un aviso sobre el tamaño de muestra que refuerza a 9.4:** si 7 imágenes de 20 deciden el
número, cambiar **una sola imagen** del val mueve el F1 en 0,05. Es la misma conclusión que la sd,
vista desde el otro lado.

### 9.6 Bootstrap de la banda — HECHA: **el `sem` aguanta (0,973× el ancho bootstrap)**

**Pregunta:** el `sem` que se reporta asume normalidad; con F1 por imagen acotado en [0,1] y muchos
ceros exactos, la distribución **no** es normal. ¿Da lo mismo un intervalo por bootstrap sobre las
imágenes?

**Cómo:** 20 000 remuestreos de las 20 imágenes (numpy, sin dependencias nuevas), percentiles 2,5 y
97,5. **Coste: milisegundos** sobre datos ya cacheados.

**Medido** (2026-07-26, las mismas 20 réplicas ganadoras de 9.4, semilla 7):

| | valor |
|---|---|
| ancho del IC95 bootstrap ÷ ancho del IC95 normal (`±1,96·sem`) | **0,973** de media |
| rango sobre las 20 réplicas | 0,960 – 0,986 |

**La distribución sí es bimodal**, como se sospechaba — un run típico reparte sus 20 imágenes en
**6 ceros exactos, 7 unos exactos y 7 intermedios**. Y aun así el `sem` vale: la media de 20
sorteos ya es prácticamente normal, y el intervalo simétrico sale **un 2,7% más ancho** que el
bootstrap, es decir **ligeramente conservador**, nunca optimista.

**Qué se hace con la respuesta:** el `sem` se queda tal cual en `task_score`, en la UI y en los
CLIs, y se gana confianza en él. **Y confirma dónde está el bloqueo:** no en la fórmula de la
banda, sino en la `n`. El ±0,093 no es un artefacto de suponer normalidad — es el ruido de verdad
que hay con 20 imágenes.

### 9.7 ¿Qué proxy de ventana correlaciona mejor? — HECHA: **`f1`, y no de poco**

**Pregunta:** §2 midió `f1` de ventana. ¿Y `pos_err_px`? ¿Y una combinación?

**Cómo:** los datos ya están: `sweep_trials` con `objective="pos_err_px"` sobre el mismo recorrido
`fast-lr-s0-lr`, y el mismo Spearman de §5.2. **Coste: segundos** (con la caché de tarea llena).

**Medido** (2026-07-26, los mismos 65 runs de `fast-lr-s0-lr`, re-leídos sin reentrenar nada):

| proxy de ventana | Spearman por run (n=65) | **agregado** (n=13) | ¿mismo ganador que la tarea? |
|---|---|---|---|
| **`f1`** | **+0,737** | **+0,956** | **sí** |
| `loss` | +0,608 | +0,780 | no (y fuera de la frontera δ) |
| `pos_err_px` | +0,402 | +0,544 | no (y fuera de la frontera δ) |

**Qué se hace con la respuesta: nada — y eso es el resultado.** El `objective` por defecto de los
recorridos ya era `f1`, y ahora lo es **con evidencia detrás en vez de por costumbre**. Los otros
dos no solo correlacionan peor: **eligen otro ganador**, y uno que ni siquiera cae dentro de la
banda de ruido del bueno. `pos_err_px` es el peor de los tres, lo cual encaja con 9.3 — solo mide
las esquinas que **existen**, y el techo del problema es de **detección** (~0,66), no de posición.

**Cableado para poder repetirlo:** `sweep_trials(..., objective=…)` y `suggest_winner(..., objective=…)`
re-leen los mismos runs terminados con otro proxy **sin tocar el spec**, y la respuesta lleva
`objective_overridden` + `sweep_objective` para que nadie confunda una re-lectura con lo que el
recorrido optimizó de verdad. Un objetivo desconocido se **rechaza en la puerta** con su razón, no
devuelve una tabla de `None` (R4). Dos tests en `tests/test_sweeps.py`. Desde el script:

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep fast-lr-s0-lr --objective pos_err_px
```

### 9.8 El coste: vectorizar `build_view` (si alguna vez molesta)

**Pregunta:** el coste de la métrica lo domina construir **63 vistas por imagen** en Python, no el
modelo (§3.8; el reparto exacto no está perfilado). ¿Cuánto se gana vectorizando?

**Cómo:** medir primero con `cProfile` sobre una imagen (¿es de verdad `build_view`?), y solo
entonces tocar. **Coste: una tarde**, y **no hacerlo hasta que el n de imágenes lo haga doler** —
con 20 imágenes no duele; con 500 y un barrido de knobs (9.2), empieza a doler.

**Qué se hace con la respuesta:** si el holdout de 500 imágenes × 30 runs se va a minutos, se
vectoriza; si no, se deja. **Nunca** «arreglarlo» bajando el número de imágenes: eso es cambiar la
medida para que quepa en el instrumento.

### 9.9 El primer experimento de verdad (bloqueado por F12)

protocolo.md §6: **¿fóvea + periferia gana a una CNN plana de coste equivalente?** Es *la* pregunta
del proyecto y sigue sin poderse montar, porque «la CNN plana equivalente» no se puede construir
hoy (`no_periphery` rechaza la geometría sin periferia, y `d=1` sigue siendo dos ramas
enmascaradas). **F12 en decisiones.md.** Lo que sí se puede preparar sin decidir nada:

- el eje `d` de §5 **ya es media respuesta**: `d=1` es «casi sin contexto» y `d=6` es «mucho
  contexto reducido». Si el F1 de tarea no se mueve con `d`, la periferia no está aportando y eso
  se sabe **antes** de construir el control.
- escribir el criterio (qué diferencia, medida cómo, con cuántas semillas) **antes** de medir,
  como manda protocolo.md §1.

### 9.10 Higiene: que la caché no mienta — HECHA: **escrita y en verde**

**Prueba barata que hoy no existe:** entrenar un run, medir tarea, **reentrenar el mismo run**
(otro `best.pt`, otro mtime) y comprobar que el número cambia. La clave ya incluye
`ckpt.stat().st_mtime_ns`, pero *que la clave lo incluya* y *que el número cambie* son dos cosas
distintas, y la segunda es la que le importa a quien mira la pantalla. Es un test de integración
de ~20 líneas.

**Hecha** (2026-07-26): `tests/test_task.py::test_task_score_does_not_serve_a_stale_number_after_retraining`.
Mide, comprueba que la segunda llamada **sí** sale de caché, reescribe `best.pt` con otros pesos y
vuelve a medir: `cached: false`, la entrada vieja sobrevive sin usarse (2 ficheros de caché) y **las
predicciones cambian**.

> ⚠ **Detalle que costó un intento y que conviene saber**: la afirmación natural
> («el `macro.f1` cambia») **no se puede hacer en el mundo mínimo de los tests** — un modelo de 1
> época sobre 10 imágenes de juguete puntúa `f1 = 0,0` prediga lo que prediga, así que 0,0 ≠ 0,0
> falla y el test parecería estar detectando un bug que no existe. La afirmación se apoya en el
> **número de predicciones** (`micro.fp`), que es lo que los pesos nuevos sí mueven. Es un ejemplo
> de la regla de tests.md: *afirmar la costura, no un valor bonito*.
