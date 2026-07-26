# La métrica de tarea: cablearla, dimensionarla y usarla

> **Estado: FASES 1 y 2 HECHAS (2026-07-26). 3b, 3 y 4 pendientes.** La Fase 1 (validar el
> proxy) está medida — sus números están en §2 y son la razón de que el resto tenga esta forma.
> La **Fase 2 está construida y verificada**: `fv.task.task_score`, `GET /runs/{name}/task-score`,
> bloque en el detalle de un run, botón en el veredicto de Recorridos, `fv-oat --task-score` /
> `fv-study --task-score`, 8 tests (7 de §3.9 + el contrato ⑬). Detalle de lo construido y de las
> desviaciones, al final de §3. Las fases **3b, 3 y 4** siguen pendientes: este documento las
> especifica para que otra sesión las implemente sin volver a decidir nada. Lo que queda abierto
> está marcado como **DECISIÓN DEL USUARIO** y no se toma solo (decisiones.md).

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
**±0,083**. Las diferencias entre puntos vecinos del recorrido son de 0,01 a 0,05. Es decir: hoy
la métrica de tarea es **más ruidosa que las diferencias que se quieren medir**, y por eso el
Spearman salta de 0,736 (por run) a 0,956 (agregando 5 semillas): el ruido está en la estimación
por run, no en el proxy. Esto es exactamente lo que protocolo.md §3 avisaba — *el tamaño de
muestra efectivo lo dan las imágenes, no las ventanas*.

**Script de referencia**: la medición se hizo con un script de un solo uso; la Fase 2 lo
convierte en código de primera clase y §5 lo repite sobre un eje de C.

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

`SE = sd / √n_imágenes`, con **sd = 0,372 medido** (§2):

| imágenes de val | SE de la métrica de tarea |
|---|---|
| 20 (hoy) | **±0,083** |
| 55 | ±0,050 |
| 154 | ±0,030 |
| **200** | **±0,026** |
| 346 | ±0,020 |

protocolo.md §3 ya pedía **~2000 imágenes 80/10/10** → 200 de val → **±0,026**, que es del orden
de las diferencias que se quieren distinguir. Es la cifra coherente: **no inventar otra**.

Nota honesta para quien lo implemente: 0,372 se midió con modelos mediocres (F1 de tarea 0,3–0,54).
Con modelos mejores la sd por imagen puede bajar, así que 0,372 es la elección **conservadora**.
Al regenerar, **volver a medir la sd** y rehacer esta tabla.

### 4.2 Cómo se genera

`scripts\make_synth_source.py` **ya está parametrizado** (verificado 2026-07-26): acepta
`--name --count --width --height --seed --root`, y escribe `count` en el `dataset.json`. No hay
que tocarlo.

**La resolución no se cambia de paso.** El dataset con el que se ha medido todo
(`dirty-paragraphs-fast-80px`) es de **80×60**, mientras el default del generador es 96×72:
pasarlas por alto cambiaría el problema a la vez que el tamaño de muestra, y ningún número sería
comparable. El comando, con la resolución explícita:

```powershell
.\.venv\Scripts\python scripts\make_synth_source.py --name paragraphs-2k --count 2000 `
  --width 80 --height 60 --seed 11
```

Después `fv-extract` con el mismo `window_size` que el B actual (**16**; ①a atará `N` a él) y
`val_frac`/`test_frac` que den los n de §4.1 — con 2000 imágenes, `--val-frac 0.1 --test-frac 0.1`
da 200 de val (SE ±0,026). Comprobar los nombres reales de las banderas con
`.\.venv\Scripts\fv-extract.exe --help` antes de escribirlo en el README.

`IMAGES_BUDGET_BYTES` (1 GB) en [extract.py](../src/fv/windows/extract.py) no estorba: 2000
imágenes de 80×60 son **9,6 MB**. A resoluciones mayores sí lo haría, y el error ya lo dice con
su razón.

### 4.3 La consecuencia, que es lo que hace esto una DECISIÓN DEL USUARIO

**Un dataset nuevo tiene otro fingerprint, así que TODO lo entrenado hasta hoy deja de ser
comparable con lo nuevo.** Los 130 runs, los 4 recorridos y los 4 estudios actuales quedan como
historia. No es un problema técnico (nada se rompe: el fingerprint protege y los diagnósticos de
lo viejo siguen funcionando contra su B viejo), es una decisión de investigación:

> **DECISIÓN DEL USUARIO (pendiente):** ¿se regenera el dato ahora —perdiendo la comparabilidad
> con lo medido hasta hoy— o se sigue con 20 imágenes de val sabiendo que la métrica de tarea
> solo puede usarse como *informe del ganador*, nunca para decidir entre puntos?

Registrada como **F11** en [decisiones.md](decisiones.md) §2 (2026-07-26). **Claude no la toma
solo.** Junto a ella quedó **F12**: qué es exactamente «la CNN plana de coste equivalente» del
primer experimento, que hoy no se puede construir (`no_periphery`).

---

## 5. Fase 3b — PENDIENTE: repetir la validación del proxy sobre un eje de C

Es la fase **más barata y la que más puede cambiar el plan**, y por eso va antes que la 4.

**Por qué:** §2 validó el proxy sobre `lr` (D). Un eje de C cambia **la vista foveada**, es decir
la regla de mirar. protocolo.md §2 es explícito: *aquí se barre C, así que ninguna métrica de
ranking puede depender de la vista*. Si el proxy de ventana se degrada al barrer geometría, todo
el arrastre de ganadores de un estudio OAT está eligiendo por el número equivocado.

**Receta exacta:**

1. Recorrido de **un eje de C** con semillas, sobre el dataset actual: el candidato natural es
   **`d`** (el submuestreo de la periferia), porque cambia cuánto contexto real ve la red sin
   tocar la fóvea (①a se mantiene). Rango: `"auto"` (lo calcula `downsample_range`).
   `fv-oat --name proxy-c-d --window-dataset <B> --axis d --range auto --seeds 5 --epochs 20`.
2. Cuando termine, calcular por cada run la métrica de tarea (`fv.task.task_score`, ya
   construida) y el Spearman contra la de ventana, **por run y agregado por valor del eje**,
   igual que en §2.
3. Decisión, escrita **antes** de mirar (protocolo.md §1):
   - **Spearman agregado ≥ 0,9 y el ganador coincide** → el proxy también vale para C. Se anota
     y no se cambia nada.
   - **Spearman < 0,9 o el ganador NO coincide** → el proxy **no** vale para barrer C. Entonces
     sí: añadir `paragraph_f1` a `fv.sweeps.spec.OBJECTIVES` (dirección `max`), con el coste
     asumido de 0,6 s por punto **y con la Fase 3 hecha antes** (si no, se estaría rankeando con
     ±0,083 de ruido, que es peor que el proxy).

**Guardar el resultado** en este documento como §2 bis, con la tabla y la fecha. Un número de
correlación sin fecha ni dataset no vale nada.

---

## 6. Fase 4 — PENDIENTE: el holdout

Heredado de protocolo.md §3 (paso 0a), sin cambios de diseño; se repite aquí lo operativo:

- **Fuente propia**, nombre `<fuente>-holdout`, **de la que jamás se extrae entrenamiento** — la
  fuga se hace físicamente imposible, no se confía en un flag.
- Misma configuración del generador que la fuente de entrenamiento, **otra semilla**.
- Tamaño: **~500 imágenes** (protocolo.md §3). Con sd 0,372 → SE ±0,017.
- **Se toca una sola vez, al final, y solo con el ganador.** El val ya hace dos trabajos (elegir
  `best.pt` y rankear), así que el val del ganador está sesgado al alza y **no se reporta**.
- Implementación: la métrica de tarea de §3 ya sirve tal cual — el holdout es *otro dataset de
  ventanas* extraído de la fuente holdout, y se pide `task_score(run, split="test")` sobre él…
  **y el detalle que esto exigía ya está resuelto** (§3.11): `task_score` resuelve la fuente
  desde el manifest del B **del run**, así que puntuar contra otro B se pide con el parámetro
  `window_dataset` — `GET /runs/{name}/task-score?window_dataset=<holdout-B>&split=test` — y hay
  guarda: **si el B del holdout comparte `source_id` con el B de entrenamiento, se rechaza** con
  `holdout_shares_source`, porque entonces no es un holdout. Lo que falta es **la fuente
  holdout**: generarla (`scripts\make_synth_source.py` con otra semilla, ~500 imágenes),
  extraerla y tocarla una sola vez, al final, con el ganador.

---

## 7. Orden recomendado y coste

| Fase | Qué | Coste | Bloquea a |
|---|---|---|---|
| ~~2~~ | ~~Cablear `task_score` (§3)~~ — **HECHA 2026-07-26** | — | — |
| 3b | Validar el proxy sobre el eje `d` (§5) | 1 recorrido + 40 s de medición | la decisión de cambiar el objetivo |
| 3 | Regenerar el dato (§4) | corrida del generador + reentrenar lo que importe | que la métrica de tarea decida algo |
| 4 | Holdout (§6) | una corrida del generador | el número que se reporta |

**La 2 y la 3b se pueden hacer con el dato de hoy**; la 3 es la que cuesta comparabilidad y por
eso es decisión del usuario.

---

## 8. Resumen para quien implemente esto en frío

1. La Fase 2 ya existe: `src/fv/task/__init__.py` (es `fv/diagnostics/table.py` con «por imagen»
   en vez de «por ventana» y con la verdad viniendo de A). Léelo antes de tocar métricas.
2. No añadas `paragraph_f1` a `OBJECTIVES` todavía. §2 dice por qué, con números.
3. No regeneres el dataset sin preguntar. §4.3.
4. Los knobs de F entran en la clave de caché; el `threshold` de diagnostics **no** entra en la
   suya. No es una incoherencia: allí re-umbralizar lee scores guardados, aquí hay que re-inferir.
5. Todo número que salga de aquí viaja con su `sem` y su n de imágenes. Un F1 de tarea sin banda
   es exactamente el error que este proyecto acaba de arreglar en el ranking.
