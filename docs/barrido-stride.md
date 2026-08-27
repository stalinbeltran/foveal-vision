# Barrido del `stride` de extracción: el mecanismo

Cómo se mide el efecto de la **densidad de la rejilla de muestreo** sobre la calidad de
predicción, y por qué ese eje **no puede** declararse en el `space` de un recorrido como se
declara `lr` o `n_layers`.

Este documento es el **mecanismo** (qué cambia, dónde y con qué contratos). El estudio concreto
—qué valores, cuántas semillas, qué criterio y cuánto cuesta— está en
[`plan-stride-2026-08-27.md`](plan-stride-2026-08-27.md), escrito antes de mirar ningún número.

---

## 0. Qué mide esto, y qué no

### 0.1 Cuál de los tres `stride` (desambiguación obligatoria)

[`glosario.md`](glosario.md) §1 ya avisa de que en este repo `stride` significa **tres** cosas.
Este barrido es el **primero**, y sólo el primero:

| | Cuál | Dónde vive | Este estudio |
|---|---|---|---|
| 1 | **stride de extracción/muestreo** — cada cuántos px se corta la siguiente ventana de la imagen fuente | `ExtractConfig.stride` (dominio **B**) — *es parte de la identidad de B* | **SÍ** |
| 2 | **stride de inferencia** — cada cuántos px se desliza la ventana al predecir sobre una página | knob de **F**, se ajusta sin reentrenar | NO |
| 3 | **`s_center` / `s_periph`** — el subsampleo de las convoluciones por rama | dominio **C**, ya barrible con rango calculado (`stride_range`) | NO |

Confundir el 1 con el 2 es el error que este documento existe para impedir: el 2 es gratis de
barrer sobre un modelo ya entrenado, el 1 obliga a **reconstruir el dataset y reentrenar**.

### 0.2 Calidad de predicción, NO uso final

La pregunta es: **¿predice mejor una red entrenada sobre una rejilla más densa?** Se responde con
la **métrica proxy** —`f1` / `pos_err_px` por ventana, la que `sweep_trials` rankea— y **no** con
la métrica de tarea (párrafos detectados sobre la página entera,
[`metrica-de-tarea.md`](metrica-de-tarea.md)).

No es un descuido, es el alcance: la métrica de tarea mezcla el stride de **inferencia** (el 2 de
la tabla) con el de extracción, y entonces el número ya no dice cuál de los dos lo movió. Un
estudio que quiera el efecto sobre el uso final es **otro** estudio, y necesita fijar el stride de
inferencia antes de empezar. Queda declarado como abierto en el plan, §6.

---

## 1. Por qué `stride` no puede ser un eje de `space`

`fv/sweeps/spec.py` lo rechaza hoy, y hace bien:

```python
elif param not in NETWORK_PARAMS | RECIPE_PARAMS:
    bad("unknown_space_param", f"'{param}' no es un campo de C ni de D", ...)
```

`stride` no es campo de C (la red) ni de D (la receta): es campo de **B**, el dataset de ventanas.
Un recorrido tiene **un** `window_dataset` y lo tiene fijo — el mismo `spec.py` ya deja escrito el
precedente para `fovea_px`:

> «la fóvea se TOMA del window_size (no se barre); … para cambiar la fóvea **usa/reconstruye un
> dataset con ese window_size**»

La doctrina del repo para un campo de B es, por tanto: **un dataset por valor**. Este barrido la
sigue en lugar de agujerearla.

### La forma que toma: brazos, no puntos

- **Un `window-dataset` por valor de stride**, todos desde la MISMA fuente, con el MISMO
  `window_size` y el MISMO `seed` de split.
- **Un recorrido (`sweep`) por dataset**, todos con la misma red base, la misma receta base y un
  único eje: `seed`, que es el eje **réplica**.
- La comparación entre valores del stride es **entre recorridos**, y la hace una herramienta
  nueva (§4.4) que reutiliza el ranking existente en lugar de reimplementarlo.

Cada recorrido lleva en su `spec.json` un bloque que dice de qué brazo es:

```json
"eje_dataset": {"campo": "stride", "valor": 8, "estudio": "stride-2026-08-27"}
```

`prepare_sweep` hace `enriched = dict(spec)`, así que las claves extra **sobreviven** al
persistido sin tocar H. Comprobado leyendo `fv/sweeps/runner.py:83`.

⚠ **`eje_dataset` NO es un eje.** No multiplica puntos, `expand_points` no lo ve y
`estudio_informe.py --eje` sigue sin conocerlo. Es una **etiqueta de procedencia** para que el
comparador sepa qué valor representa cada recorrido; si se le da otro uso, es que el eje se ha
colado en H por la puerta de atrás.

---

## 2. Los tres sesgos, y cómo se neutraliza cada uno

Los dos primeros son la razón de ser de este documento: un barrido de stride hecho de la manera
obvia **mide otra cosa** y la tabla sale igual de creíble.

### 2.1 El conjunto de evaluación se mueve con el eje → **rejilla de eval FIJA**

`extract_windows` corta val y test con el **mismo** stride que train. O sea que al barrer el eje
se mueve también el conjunto sobre el que se mide: con stride 1 el val tiene 2925 ventanas por
imagen y con stride 16 tiene 20. **Comparar esos dos `f1` es comparar dos exámenes distintos.**

Solución: `ExtractConfig` gana un campo `eval_stride`, que fija la rejilla de **val y test** para
todos los brazos. Sólo el split de **train** sigue al eje.

**Valor elegido: `eval_stride = 5`.** Las tres razones, en orden de peso:

1. **Es la rejilla de producción.** Todos los datasets con los que este proyecto ha entrenado
   (`dirty1000-80px-16px-*`) son stride 5. Medir sobre ella hace que los `f1` de este estudio sean
   **comparables con los de todos los estudios anteriores**, en vez de vivir en una escala propia.
2. **No regala ventaja a ningún brazo.** Si la rejilla de eval coincidiera con uno de los strides
   entrenados, ese brazo habría visto exactamente las mismas *fases* (offsets de ventana) que el
   examen. Con 5 —que no está entre los valores barridos— la cobertura de fases baja de forma
   monótona con el stride, que es justo el efecto que se quiere medir y no un artefacto:

   | brazo | posiciones de la rejilla de eval ya vistas al entrenar |
   |---|---|
   | stride 1 | 140/140 (100,0 %) |
   | stride 2 | 48/140 (34,3 %) |
   | stride 4 | 20/140 (14,3 %) |
   | stride 8 | 9/140 (6,4 %) |
   | stride 16 | 4/140 (2,9 %) |

   MEDIDO el 2026-08-27 con `fv.windows.extract._positions` sobre 60×80 px y ventana 16.
3. **Cuesta lo que cuesta hoy.** 200 imágenes de val × 140 ventanas = **28.000**, exactamente el
   `windows_per_split["val"]` del manifest de `dirty1000-80px-16px-r20260826`. El coste por época
   de la evaluación no cambia respecto a lo ya medido.

⚠ Que el brazo stride 1 cubra el 100 % de las posiciones de eval **no es fuga**: el split es por
imagen (§2.3), así que las *imágenes* de val no las ha visto ninguno. Lo que comparte es la
geometría de la rejilla, que es el tratamiento.

### 2.2 El presupuesto de entrenamiento se mueve con el eje → **ventanas/época FIJAS**

Con ventana 16 sobre imágenes de 60×80, el número de ventanas de train por imagen va de 2925
(stride 1) a 20 (stride 16): un factor **146,2×**. A épocas iguales, el brazo de stride 1
recibiría 146 veces más pasos de gradiente que el de stride 16, y la tabla no distinguiría
«rejilla más densa» de «146 veces más entrenamiento».

Solución: `Recipe` gana `windows_per_epoch`. Cada época consume **exactamente** ese número de
ventanas del pool de train, sea el pool grande o pequeño.

**Valor elegido: `windows_per_epoch = 84_000`**, que es el tamaño del split de train del dataset
de producción. Con eso:

- todos los brazos hacen **el mismo número de pasos de gradiente por época** (84.000/85 = 988 con
  el batch 85 de `plan40`), que además es el mismo que ha hecho **todo estudio anterior** de este
  repo;
- los coeficientes medidos de `estudio_estimar.py` (`S_EPOCA_REF` = 40 s/época) **siguen valiendo
  sin corrección**;
- el brazo stride 16 recorre su pool de 12.000 ventanas 7 veces por época; el de stride 1 saca
  84.000 frescas de un pool de 1.755.000.

**Qué se compara entonces**: a **igual cómputo**, ¿ayuda tener más posiciones distintas de las que
tirar? Esa es la pregunta del usuario, y es la única versión de ella que se puede pagar (§4 del
plan tiene el número).

**Política de muestreo** (hay que escribirla, porque «coger 84.000 de N» admite variantes que dan
resultados distintos):

- `W <= pool`: prefijo de una permutación aleatoria (sin reemplazo).
- `W > pool`: `ceil(W/pool)` permutaciones independientes concatenadas y truncadas a `W`. **No**
  muestreo con reemplazo: éste sobre-representaría unas ventanas y omitiría otras dentro de la
  misma época, que es ruido gratis.
- El generador se siembra con `(recipe.seed, epoch)`, de forma que **misma semilla + misma config
  ⇒ mismos pesos** sigue siendo cierto (contrato ⑪).

### 2.3 La fuga train/test — **ya estaba resuelta; no la rompas**

Con solape entre ventanas, partir *por ventana* metería vecinas casi idénticas a los dos lados del
corte, y a stride 1 dos ventanas contiguas comparten 15/16 de su superficie. `extract.py` ya lo
impide:

```python
def _assign_splits(...):
    """Split BY IMAGE (never by window: windows of one image are correlated)."""
```

Y el contrato ⑧ de [`tests.md`](tests.md) lo tiene testeado («el split es **por imagen**»). Este
barrido no toca eso; sólo hay que **no** tocarlo al añadir `eval_stride`, porque la tentación de
recortar el coste del eval partiendo por ventana está justo ahí.

Corolario que sí hay que respetar: **el `seed` de B es el mismo en los cinco brazos.** Como
`_assign_splits` sólo depende de `(num_samples, val_frac, test_frac, seed)`, los cinco datasets
comparten *las mismas imágenes* en train, val y test. Cambiarlo mediría el ruido del split
(glosario §1, entrada `seed`).

---

## 3. La rejilla del estudio (números, con su procedencia)

Fuente `local/dirty-1000-80px`: 1000 imágenes de **60×80 px**; ventana **16**; `val_frac` =
`test_frac` = 0,2 ⇒ 600 imágenes de train, 200 de val, 200 de test.

| stride | pos. Y | pos. X | ventanas/imagen | total | **train** |
|---|---|---|---|---|---|
| 1 | 45 | 65 | 2925 | 2.925.000 | 1.755.000 |
| 2 | 23 | 33 | 759 | 759.000 | 455.400 |
| 4 | 12 | 17 | 204 | 204.000 | 122.400 |
| 8 | 7 | 9 | 63 | 63.000 | 37.800 |
| 16 | 4 | 5 | 20 | 20.000 | 12.000 |
| *5 (sólo eval)* | *10* | *14* | *140* | *140.000* | *84.000* |

MEDIDO el 2026-08-27 con la función real del repo:

```
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); \
  from fv.windows.extract import _positions; \
  print(len(_positions(60,16,1))*len(_positions(80,16,1)))"
```

**Contraste contra dato real**: la fila de stride 5 da 140.000 totales y 84.000 de train, que es
exactamente lo que declara `data/window-datasets/dirty1000-80px-16px-r20260826/manifest.json`
(`num_windows: 140000`, `windows_per_split.train: 84000`). La aritmética de arriba reproduce un
dataset que ya existe, así que no es una fórmula escrita de memoria.

⚠ La rejilla **no es uniforme**: `_positions` añade una última ventana *a ras* del borde cuando el
paso no cae justo (por eso stride 8 da 7 posiciones en Y y no 6). Es igual para todos los brazos,
así que no sesga la comparación, pero explica por qué los conteos no son `(L-n)/s + 1` a secas.

---

## 4. Qué cambia, por dominio

Cuatro cambios. Los tres primeros son aditivos y con default que **preserva el comportamiento
actual byte a byte**; el cuarto es una herramienta nueva.

### 4.1 B — `eval_stride` en `ExtractConfig`

```python
@dataclass
class ExtractConfig:
    ...
    stride: int = 8
    eval_stride: int | None = None   # None = el de train (comportamiento de siempre)
```

`extract_windows` recorta con `cfg.stride` las imágenes de train y con
`cfg.eval_stride or cfg.stride` las de val y test.

**Compatibilidad**: con `eval_stride=None` el recorrido de posiciones es idéntico, en el mismo
orden, así que el `windows.npz` de cualquier dataset ya extraído sale **bit a bit igual** y su
`fingerprint` no se mueve. Es lo que tiene que comprobar un test (§7), no lo que hay que creerse.

**Lo que NO cambia**: el split sigue siendo por imagen; `positives_per_corner` y
`windows_per_split` siguen contándose sobre lo que de verdad hay.

### 4.2 D — `windows_per_epoch` en `Recipe`

```python
windows_per_epoch: int = 0   # 0 = todas (comportamiento de siempre)
```

En `_train_inner`, cuando es > 0, el `DataLoader` de train recibe un `sampler` que entrega
exactamente esas ventanas por época con la política de §2.2. Con 0 **no se construye ningún
sampler**: la ruta actual (`shuffle=True`) queda intacta.

⚠ `Recipe` es dominio D y «cambiarlo cambia el resultado», así que el campo entra en
`RECIPE_PARAMS` y pasa a ser barrible. Es correcto —es un presupuesto, como `epochs`— pero
significa que **las specs ya escritas no lo llevan**: al leerlas, un campo ausente vale 0 y todo
sigue igual.

### 4.3 La flota — un payload con VARIOS datasets

Hoy `cargar_sweeps` se niega si los recorridos de una flota no comparten dataset:

> «los recorridos de una misma flota tienen que compartir dataset de ventanas … Lanza una flota
> por dataset.»

Para este estudio eso obliga a 5 flotas, y **eso rompe los dos monitores** (§6). Así que
`estudio_flota.py` pasa a aceptar varios: `construir_payload` mete un directorio por dataset y
cada recorrido entrena con el suyo, que ya viene declarado en su propio `spec.json`.

**Lo que hay que respetar si se toca esto:**

1. **La ruta de un solo dataset no cambia.** Con un dataset, el tar que se sube y el comando que
   se ejecuta son los mismos que hoy. Este script gasta dinero: una regresión aquí no se ve, se
   factura.
2. **Sigue faltando el npz ⇒ sigue muriendo antes de alquilar.** La comprobación de existencia se
   hace **para todos** los datasets, y antes de tocar Vast. Descubrir a mitad que falta un npz
   son máquinas alquiladas para nada.
3. **El tamaño del payload se imprime.** Con el brazo de stride 1 dentro, el tar deja de ser
   despreciable y se sube a cada máquina; si crece, hay que verlo en el log y no en la factura.

### 4.4 Nuevo — `scripts/estudio_stride.py` (crear) y `scripts/estudio_stride_informe.py` (leer)

- **`estudio_stride.py`** traduce el plan a datasets + recorridos, como `estudio_cierre.py` hace
  con el suyo. No alquila ni entrena nada. Extrae los `window-datasets` que falten y prepara los
  `sweeps` con su `eje_dataset`.
- **`estudio_stride_informe.py`** es el veredicto entre brazos. **No reimplementa el criterio**:
  toma de cada recorrido sus `sweep_trials`, les pone el valor del stride como si fuera el punto,
  y llama a las funciones que ya existen —`aggregate_seeds` para la media por valor con su banda,
  `suggest_winner` para el ganador, `permutation_test` para el contraste—. Un número definido dos
  veces es un número que acaba divergiendo (es la razón de ser de `estudio_informe.py`).

  La costura es exacta porque `aggregate_seeds` agrupa por el punto **sin** `seed`: dándole
  `point = {"stride": s, "seed": k}` agrupa por stride sin tocar una línea de `winner.py`.

---

## 5. Compatibilidad: lo que no puede moverse

| Qué | Por qué | Cómo se comprueba |
|---|---|---|
| `windows.npz` de los datasets ya extraídos | son la base de 700+ runs ya pagados | test: extraer con `eval_stride=None` da el mismo `fingerprint` |
| El `f1` de un run con `windows_per_epoch=0` | todas las tablas publicadas | test: mismo seed + misma config ⇒ mismos pesos (contrato ⑪, ya existe) |
| El tar de una flota de un solo dataset | es lo que gasta dinero | test: el listado del tar con 1 dataset es el de hoy |
| El split por imagen | contrato ⑧ | test ya existente en `test_contracts.py` |

---

## 6. Los monitores de las máquinas del estudio

El estudio corre en máquinas de Vast alquiladas por segundo; lo que vigila que no se quede una
facturando sin producir ya existe y **se reutiliza**. Lo que hay que arreglar son dos supuestos
que este estudio rompe por ser el primero multi-dataset.

### 6.1 Lo que ya sirve tal cual

| Herramienta | Qué mira | Cuesta |
|---|---|---|
| `scripts/estudio_progreso.py` | cuántos puntos van, leyendo el libro de a bordo que la flota commitea | nada: no entra por SSH |
| `scripts/vigilante_avance.py` | máquina a máquina: alquilada y muda (>25 min), congelada (>20 min sin latido), huérfana (lote `done` y sigue viva) → la destruye | nada |
| `telegram/executors/estudio-progreso.json` | lo anterior, desde el móvil | nada |
| `apagar-vast` (repo del lanzador) | el freno de emergencia: destruye todo | nada |

Los umbrales del vigilante son de tiempo, no de dataset, así que valen sin tocar: 25 min de gracia
contra un arranque real medido de ~5 min hasta la primera época, y 20 min sin latido contra una
época peor observada de 161,6 s.

### 6.2 Dos fallos que este estudio dispara, y que hay que arreglar con él

**(a) El vigilante relanza todos los recorridos en UNA flota.** `relanzar()` construye
`estudio_flota.py --sweep A --sweep B …` con todos los que faltan. Con los cinco brazos en
datasets distintos, esa llamada muere hoy en `cargar_sweeps` con «tienen que compartir dataset» —
o sea que **el relanzamiento automático no relanzaría nada**, y el síntoma sería el peor de los de
este proyecto: un estudio que parece vivo y no avanza. Lo arregla §4.3.

**(b) `pgrep -f estudio_flota.py` era global.** El vigilante no relanza si ve una flota viva
(regla 4, y es correcta: dos flotas alquilarían dos veces para los mismos puntos). Pero la
comprobación no distinguía **de qué estudio** es la flota que ve. Con una flota de otro estudio
corriendo en la misma máquina —que es exactamente lo que pasaba en este servidor el 2026-08-27—,
el vigilante de este estudio **no habría relanzado nunca**: un estudio que parece vigilado y no
avanza, sin un solo error.

Es también la razón de fondo por la que este estudio va a **una sola flota** en vez de cinco: con
cinco, cada vigilante vería a las otras cuatro.

`flota_viva(sweeps)` mira ahora la línea de comandos y sólo cuenta la flota que menciona alguno de
**sus** recorridos. Sin lista, se comporta como antes.

Los dos arreglos son la misma pregunta en dos capas y hacen falta los dos: el prefijo dice de
quién son las **máquinas**, y esto dice de quién es la **flota**.

### 6.3 El aviso, que no puede esperar dentro del turno

Nada que espere a que termine el estudio puede vivir en un turno de Telegram: el proceso muere al
responder. Se desacopla con `scripts/desacoplar.sh` (cgroup propio, sobrevive a un
`systemctl restart`) y el aviso lo da `notify.mjs`, como ya hacen los ejecutores `estudio` y
`bench`. **La fuente de verdad no es el aviso, es el disco**: `sweeps/<brazo>/flota.json`,
`runs/*/status.json` y el log en `/tmp/`.

---

## 7. Tests esperados

Método de [`tests.md`](tests.md): se testea **la costura**, no la función, y un contrato sin test
es un comentario.

### 7.1 No-regresión (los que protegen lo ya pagado)

| Test | Afirma |
|---|---|
| `test_eval_stride_default_is_bit_identical` | extraer con `eval_stride=None` da el MISMO `windows.npz` — **misma huella sha256, byte a byte** — que antes del campo |
| `test_eval_stride_none_equals_explicit_same_value` | declararlo igual que el de train tampoco mueve nada (control) |
| `test_windows_per_epoch_zero_keeps_the_old_loader` | con 0 la época recorre el pool entero, como siempre |
| `test_recipe_accepts_old_specs_without_the_field` | `Recipe(**spec["base_recipe_value"])` sobre un spec ya escrito sigue construyendo: ausente = 0, no error |
| `test_payload_accepts_a_bare_string` · `test_multi_dataset_payload_carries_all` | el tar de **un** dataset es el de siempre y no arrastra los demás |

### 7.2 El mecanismo nuevo

| Test | Afirma |
|---|---|
| `test_eval_stride_splits_use_their_own_grid` | train sigue a `stride`, val/test a `eval_stride`; los conteos por split son los que dan las posiciones reales |
| `test_eval_stride_same_images_across_strides` | dos brazos con distinto `stride` y el mismo `seed` tienen **idéntico** `split.json` — misma imagen, mismo lado |
| `test_eval_stride_shared_grid_gives_same_eval_count` | los brazos se examinan del mismo número de ventanas aunque su train sea de tamaños muy distintos |
| `test_eval_stride_invalid_is_refused_with_reason` · `test_eval_stride_travels_to_the_manifest` | un valor imposible se rechaza con razón y arreglo; el válido viaja al manifest |
| `test_eval_stride_cannot_be_one_of_the_arms` | la rejilla de evaluación **no puede ser uno de los strides barridos**: ese brazo habría entrenado sobre las posiciones exactas del examen |
| `test_windows_per_epoch_exact_count` · `_caps_the_epoch` · `_above_pool_repeats` | cada época consume exactamente `W` ventanas, con `W` mayor y menor que el pool, medido sobre el `DataLoader` de verdad |
| `test_windows_per_epoch_no_replacement_within_pass` · `_repeats_by_whole_permutations` | con `W ≤ pool` ninguna sale dos veces; con `W > pool` el reparto va por permutaciones completas (cuentas 2 o 3, nunca 0 ni 5) |
| `test_windows_per_epoch_is_reproducible` · `_advances_each_pass` | misma semilla ⇒ misma secuencia (contrato ⑪); semilla y época distintas ⇒ distinta (controles) |
| `test_multi_dataset_payload_carries_all` | el tar de N datasets los lleva los N |
| `test_multi_dataset_missing_npz_dies_before_renting` | falta un npz ⇒ error con su razón **antes** de tocar la API de Vast |
| `test_vigilante_sobrantes_respects_ajena` · `test_vigilante_prefix_is_a_parameter` · `test_vigilante_only_sees_fleets_of_its_own_study` | los dos fallos de §6.2, cerrados: la rama de sobrantes respeta el veredicto de `juzgar`, el prefijo es parámetro y se hereda al relanzar, y `flota_viva` sólo cuenta la flota de sus propios recorridos |
| `test_progreso_names_an_axis_that_lives_in_the_dataset` | el monitor de progreso sabe nombrar un eje que no está en `space` (imprimía «eje ?»), sin pisar el eje de un recorrido normal |
| `test_sonda_projection_respects_the_budget` | la sonda proyecta s/época desde la época, no desde el pool: con el presupuesto igualado anunciaba 20,9× de más |
| `test_stride_informe_groups_by_stride` | el comparador agrupa por valor de stride con `aggregate_seeds`, y la media coincide con la calculada a mano |
| `test_stride_informe_refuses_mixed_eval_grid` · `_refuses_mixed_budget` | brazos con distinta rejilla o distinto presupuesto **no se juntan en una tabla** |
| `test_stride_informe_refuses_a_sweep_without_the_label` · `_refuses_when_the_study_has_no_arms` · `_orders_arms_by_stride` | sin `eje_dataset` no se adivina el valor; sin brazos no se inventa una tabla; el orden es por stride |
| `test_humo_sweeps_are_a_separate_study` | los recorridos de validación **no** se llaman como los del estudio: si lo hicieran, la flota los daría por hechos y el estudio quedaría «medido» con 3 épocas |
| `test_stride_informe_never_contradicts_itself_with_one_arm` | con un solo brazo, R1 decía a la vez «la densidad no compra nada» y «el eje no queda cerrado por arriba». Y es el caso **normal**: es lo que sale al mirar mientras la flota corre |
| `test_stride_informe_says_when_there_is_no_noise_band` | sin réplicas δ = 0 y **cualquier** diferencia «supera» δ: R1 corona ganador y R3 marca ruptura por construcción. Un «δ = 0,0000» parece precisión y es ausencia |
| `test_estudio_stride_refuses_a_missing_source_with_the_fix` | la fuente está en `.gitignore`: se comprueba **antes de nada** y el error dice el comando que la reconstruye, no sólo que falta |

**37 tests**, en `tests/test_stride.py`. La suite entera pasa de 192 a **229**.

⚠ Este recuento se queda viejo solo. Si no cuadra con
`pytest tests/test_stride.py --collect-only`, manda el segundo.

### 7.3 Lo que NO se testea

Que un stride sea mejor que otro. Es resultado de investigación, no contrato: rompe por razones
legítimas y entrena a la gente a arreglar el umbral ([`tests.md`](tests.md) §5). El veredicto vive
en el plan.

---

## 8. Lo que este mecanismo NO hace

- **No convierte `stride` en un eje de `space`.** Sigue rechazado por `check_sweep`, y con el
  mismo mensaje.
- **No mide el efecto sobre la métrica de tarea** (§0.2).
- **No toca el stride de inferencia** ni el de las convoluciones.
- **No hace comparables recorridos con `eval_stride` distinto.** El comparador se niega si los
  brazos no comparten rejilla de evaluación: sería la trampa de §2.1 disfrazada de tabla.
- **No decide cuántas semillas hacen falta.** Eso es del plan, y depende del contraste que se
  quiera poder declarar.
