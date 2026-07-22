# Organización del proyecto por dominios

Separa qué es **la red**, qué es **el dato**, qué es **el entrenamiento** y qué es **el modelo
entrenado**; y dónde se tocan. Heredado de `image-text-finder` (ver [herencia.md](herencia.md)),
adaptado a la red foveada de [../instructionsNewNN.md](../instructionsNewNN.md).

Regla de lectura: cada dominio tiene **identidad propia** (algo que se puede nombrar, listar,
borrar) y **una sola razón para cambiar**. Donde dos dominios comparten un valor, eso es un
**contrato**, numerado en §2.

---

## 1. Los dominios

| # | Dominio | Qué es | Identidad | Almacén |
|---|---------|--------|-----------|---------|
| **A** | **Fuente** | Imágenes + geometría de párrafos. Lo produce *otro* proyecto | `id` = ruta relativa a la raíz de datasets | externo, solo-lectura (+ `data/sources/` para derivadas) |
| **B** | **Dataset de ventanas** | Lo que se etiqueta: imágenes + etiquetas por ventana | `<name>` = subdir | `data/window-datasets/<name>/` |
| **C** | **Red foveada** | La arquitectura **y la geometría de su entrada**. Config puro | `<name>.yaml` | `configs/networks/*.yaml` |
| **D** | **Receta** | Hiperparámetros de entrenamiento que definen el resultado | `<name>.yaml` | `configs/recipes/*.yaml` |
| **E** | **Run** | El modelo entrenado: pesos + métricas + procedencia | `<name>` = subdir | `runs/<name>/` |
| **H** | **Recorrido** | Un espacio sobre **C y/o D** con B fijo → muchos E | `<name>` | `sweeps/<name>/` |
| **F** | **Inferencia** | Aplicar un E a una imagen completa | — (operación, no cosa) | — |
| **G** | **Geometría foveada** | Lo que *todos* comparten: dimensiones derivadas, construcción de la vista, máscaras, rangos | — | `src/fv/fovea/` |
| **X** | **Ejecución** | `device`, `num_workers`, concurrencia. Cuesta tiempo, no cambia el resultado | — (transversal) | `src/fv/api/jobs.py` |

Observaciones que ordenan todo lo demás:

- **C, D y H son sustantivos**: se nombran, se guardan, se comparan y se reutilizan. Es la
  condición para que un recorrido exista («un espacio sobre C/D con B fijo» exige poder nombrar
  C y D) y para que la procedencia de un run se sostenga sola.
- **F es un verbo**: no tiene almacén; es una llamada sobre un E. En la UI es un panel de
  resultados, no una entidad listable.
- **G nace como módulo el día 0, no cuando duela.** En el proyecto hermano la geometría de la
  ventana acabó duplicada entre extracción e inferencia (su contrato ⑤) porque «el sitio
  correcto no existía». Aquí el sitio existe desde el principio: `fv.fovea`.

### A — Fuente

Igual que en el proyecto hermano, **compatible tal cual**: `labels.jsonl` con imágenes y
`blocks[].quad` de párrafos, producido por `image-text-sample-generator` (su `SAMPLE_FORMAT.md`
manda). Solo-lectura; la raíz se apunta con una variable de entorno (`FV_DATASETS_ROOT`).

Se heredan también, porque son mecanismo probado (herencia.md §2):

- **A′ — fuentes derivadas** (resize proporcional, solo reducir, geometría reescalada recursiva,
  bloque `derived` con `from` direccionable + `scale` real medida). Van a `data/sources/`,
  segunda raíz; A externa nunca se escribe.
- **El índice de offsets** (`data/cache/sources/`): mirar UNA imagen no parsea el `labels.jsonl`
  entero. Caché recomputable con fecha+tamaño del fichero en la clave.

### B — Dataset de ventanas

Lo definen sus parámetros de extracción: `source`, la definición de la **ventana etiquetada**
(tamaño, stride o política de muestreo — y si es el centro o el campo completo: **F1b** de
decisiones.md), el criterio de etiqueta (**cerrado, C9**: esquinas de párrafo por tipo,
`[exists, x, y]` por TL/TR/BR/BL; `target_kinds` decide qué bloques de A se etiquetan —
párrafos hoy, líneas y palabras después), `split{train,val,test}` y `seed`.

**La decisión estructural heredada más importante (D23 del hermano): B guarda las imágenes
completas y las etiquetas por ventana; la vista foveada NO se hornea en B — se construye en el
dataloader**, con la geometría que declare C. Consecuencias:

- **Toda la geometría foveada (`N`, `c_frac`, `d`, `pen_frac`) es barrible sin reconstruir B.**
  Cambiar `d` es una línea de config de C, no una re-extracción. Es exactamente lo que un
  recorrido de configuraciones necesita.
- B declara en su manifest **qué ofrece**: tamaño de imagen, `has_images`, márgenes disponibles.
  C declara **qué necesita**: `original_size = center_out + 2·periph_out·d`. El validador casa
  ambos (contrato ①).
- El coste del camino perezoso está medido en el hermano (P4.2): construir la vista por ítem es
  viable **solo vectorizado** (~0,1–0,2 ms/ventana); un doble bucle Python lo hace 48× más lento.
  El presupuesto de `images` también: se rechaza por encima de un umbral (1 GB) con la razón, no
  hay camino degradado.
- **El split es por imagen, no por ventana**: ventanas de la misma imagen jamás caen en splits
  distintos (los ejemplos de una imagen están correlacionados). Se testea.

Produce `windows.npz` (arrays paralelos + `images`) y `manifest.json` con `fingerprint`
(contrato ⑧). Detalle en [formatos.md](formatos.md) §4.1.

### C — Red foveada

`NetworkConfig`, siguiendo instructionsNewNN.md. Parámetros fundamentales (todo lo demás se
deriva):

```yaml
# configs/networks/<name>.yaml
N: 20            # lado de la entrada compuesta
c_frac: 0.8      # fracción del centro
d: 2             # downsampling de la periferia
pen_frac: 0.1    # penetración hacia el centro
n_layers: 2      # capas conv por rama
k_center: 3      # kernel de la rama central   (impar; rango calculado)
k_periph: 3      # kernel de la rama periférica
s_center: 1      # stride rama central          (rango calculado)
s_periph: 1
ch1: 32
ch2: 64
merge: concat    # sum | concat  (§7 de instructionsNewNN.md; concat si strides difieren)
pool_mode: avg   # avg | max  para reducir la periferia (decisión abierta, se barre)
head: corners    # 4×[exists, x, y] por tipo TL/TR/BR/BL (C9, decisiones.md)
```

**La cabeza es la de esquinas del hermano, no el clasificador de la spec** (C9): el
`nn.Linear(ch2, num_classes)` con `adaptive_avg_pool2d(feat, 1)` de instructionsNewNN.md §6 era
código de referencia, y el avg-pool global **destruye la posición** que la cabeza predice. La
cabeza consume el `feat` fusionado aplanado (con `merge: concat`, la dimensión aplanada la
calcula un tensor dummy, como el `_infer_flat_features` del hermano).

- **Los derivados no se escriben**: `center_out`, `periph_out`, `penetration`, `periph_band`,
  `periph_real`, `original_size` y `padding = k // 2` **se calculan** en `fv.fovea.derive_dims`
  y se validan (contrato ②). Un derivado escrito a mano es una copia que diverge.
- **Los rangos de búsqueda también son funciones** (`kernel_range`, `stride_range`,
  `downsample_range`, `build_search_space`) y viven en `fv.fovea`, no en el runner del
  recorrido: H los consume, no los define.
- **No posee**: pesos (E), `lr`/`epochs` (D), de dónde salen las ventanas (B).
- Es config puro: `build_model(dict)` construye desde un diccionario, listable y comparable sin
  tocar un dataset.

### D — Receta

**El catálogo de hiperparámetros del hermano se hereda casi entero** (herencia.md §2), con sus
definiciones y notas de barrido — cada campo del formulario lleva su definición en línea, y un
hiperparámetro sin definición en el catálogo no está terminado:

- **Optimización**: `lr` (log-scale), `optimizer`, **`momentum` explícito** (la trampa medida:
  SGD a momentum 0 pierde siempre), `weight_decay`, `batch_size` (acoplado a `lr`; **es D, no
  X** — contrato ⑩), `grad_clip`.
- **Duración**: `epochs`, `scheduler` (sin él, barrer `lr` optimiza un régimen que no usarás),
  `warmup_epochs`, `patience`/`min_delta` (parada temprana per-run ≠ poda del recorrido).
- **Pérdida** (heredada entera con la cabeza, C9):
  `L = Σ_c [BCE(exists_c) + λ·exists_c·smoothL1(x_c, y_c)]` con `lambda_pos` (el
  hiperparámetro más interesante y el que más cuidado pide al rankear — contrato ⑨),
  `pos_weight` (partir del desbalance medido en el manifest de B) y **`smooth_l1_beta`
  explícito** (~0.05–0.1: el default 1.0 con coords en [0,1] anula el Huber — la pérdida sería
  MSE pura sin que nadie lo decidiera, trampa medida).
- **Selección y réplica**: `monitor` explícito; `seed` (**eje de réplica, no un hiperparámetro a
  optimizar**; distinto del `seed` de B — glosario).
- `augment` y `sampler` **fuera a propósito** hasta que alguien los pida con su test: los flips
  y rotaciones invalidan etiquetas direccionales en silencio.

**El criterio de pertenencia**: si cambiarlo cambia los pesos resultantes, es D. Si cambia la
forma del modelo o de su entrada, es C. Si solo cambia cuánto tarda, es X.

### E — Run

`runs/<name>/`: `config.json` (todo congelado + `provenance` + `execution` aparte),
`metrics.jsonl` (append-only, pollable con `?since=`), `best.pt`, `last.pt`, `summary.json`,
`status.json` (estado **explícito**: `queued|running|done|error|cancelled`), `stop.json` (la
petición de parada, cooperativa a fin de época). Formato en [formatos.md](formatos.md) §4.2.

- **La procedencia guarda nombre Y valor** de C y D (el valor reproduce, el nombre agrupa), la
  **huella** de B, el commit de git y el `environment` (python/torch/plataforma) — imprescindible
  aquí porque el plan **es** comparar CPU-hoy con GPU-mañana.
- El checkpoint es **autodescriptivo** (contrato ④): `load_model()` reconstruye C sin YAML.
- Un run **no se sobrescribe jamás**; el nombre se reserva solo tras pasar el validador.

### H — Recorrido

**La razón de ser del proyecto**: probar rápidamente varias configuraciones de red y varios
parámetros para hallar los apropiados. Un recorrido es *un espacio explorado con B fijo*, y —
**a diferencia del hermano, y por diseño desde el día 0** (allí fue la decisión D22 que quedó
abierta)— **el espacio puede cubrir C además de D**: barrer `k_center`, `s_center`, `d` o
`c_frac` es exactamente el caso de uso central (instructionsNewNN.md §3).

Lo define:

- **Lo fijo**: un B concreto (huella incluida). Lo que no barres, se queda fijo (contrato ⑧).
- **El espacio**: campos de C y/o D con su rango. Los rangos de geometría **se piden a
  `fv.fovea.build_search_space`**, no se escriben a mano; el runner **valida cada punto** con el
  mismo validador de las demás puertas y descarta los geométricamente inválidos **antes** de
  reservar nombre (los asserts de instructionsNewNN.md §2 hacen que muchas combinaciones de
  `c_frac`/`pen_frac` no existan).
- **La estrategia**: `grid` para la geometría (espacio pequeño y discreto por construcción);
  `random`/TPE (optuna) para lo continuo (`lr`, canales, dropout). Optuna es el **motor**, no la
  organización: un trial **no** es un run — lanza uno y guarda su nombre.
- **El objetivo** (métrica escalar y dirección) — con la restricción del contrato ⑨.
- **El presupuesto**: puntos, épocas por punto, poda (la palanca nº1 en CPU: en el hermano, 14
  de 30 puntos podados), y si el presupuesto se cuenta **en épocas o en segundos** — con redes
  de coste distinto (barrer C), la poda por tiempo favorece a las pequeñas sin decirlo; se
  declara en el spec.
- **Los hijos**: los E que genera, con `provenance.sweep` puesto.

**El estado vive en disco** (`sweeps/<name>/spec.json` + `optuna.db` + los runs) y el recorrido
**se reanuda** tras un reinicio — en esta máquina (hibernación) y en un server (deploys) es
condición de supervivencia, no comodidad. Ejecución secuencial (límite de workers 1 en CPU),
cancelación cooperativa.

Esto es lo que el usuario llama **«recetas de recorrido»**: specs con nombre que encadenan la
serie completa sin intervención humana — versión corta en CPU para validar el instrumento, el
mismo spec con más presupuesto en GPU.

### F — Inferencia

Aplicar un E a una imagen completa: ventana foveada deslizante → detecciones de esquina por
ventana → NMS → **reconstrucción TL→BR** (heredada como punto de partida: heurística, no red —
mejora sola si el modelo mejora, y es donde tocar si los párrafos salen mal con esquinas
buenas). Devuelve **las tres etapas**, no solo la última. Sus knobs (`threshold`, stride de
inferencia, radio de NMS, `min_size`) son **de F, no de D**: se ajustan post-hoc sin
reentrenar, en unidades de la ventana (el payload devuelve los knobs con que se calculó).

**F construye la vista foveada con el mismo `fv.fovea.build_foveated_input` que el dataloader**
(contrato ⑤). Los bordes de imagen: el relleno de la vista viaja con su **máscara de cobertura**
(fracción real por celda, nunca binarizada, nunca blanco — herencia.md §2, lección de P4 del
hermano).

### G — Geometría foveada

El módulo que todos comparten y nadie duplica — `src/fv/fovea/`:

| Función | Qué |
|---|---|
| `derive_dims(N, c_frac, d, pen_frac)` | `center_out`, `periph_out`, `penetration`, `periph_band`, `periph_real`, `original_size` + los asserts (contrato ②) |
| `build_foveated_input(img, ...)` | La entrada compuesta N×N (muestreo **excluyente**) — instructionsNewNN.md §5 |
| `build_masks(N, ...)` | Máscaras **contributivas** de las dos ramas (se suman en la penetración) — §6 |
| `kernel_range`, `stride_range`, `downsample_range`, `build_search_space` | Los rangos como funciones — §3 |

**No importa nada de `fv`**: arrays y aritmética. Es lo que permite que B (dataloader), C
(builder), F (inferencia) y H (espacios) lo consuman sin ciclos, y lo que lo haría extraíble si
un tercer proyecto lo pide.

### X — Ejecución

`device`, `num_workers`, límite de concurrencia. Vive **fuera** de la identidad de D
(`execution` aparte en el `config.json` del run). La misma receta en CPU y en GPU es **la misma
receta**. `num_workers` es X puro; **`batch_size` no** (contrato ⑩).

---

## 2. Dónde interactúan (los contratos)

Numeración heredada del hermano donde el contrato es el mismo; adaptada donde cambia.

### ① B ↔ C — la ventana etiquetada y la vista computable ← **el crítico**

Dos mitades, como el ①a/①b del hermano:

- **①a — lo que B etiqueta es lo que C predice.** La ventana etiquetada de B no se mueve al
  cambiar la vista; la cabeza de C responde por esa ventana. Si B y C declaran tamaños de
  ventana distintos, se rechaza con `window_size_mismatch` antes de reservar nombre.
- **①b — la vista que C pide es computable con lo que B guarda.** C necesita
  `original_size = center_out + 2·periph_out·d` píxeles alrededor del centro de la ventana; B
  declara `has_images` y el tamaño de sus imágenes. Si no alcanza (imagen más pequeña que
  `original_size`, o B sin `images`), se rechaza con la razón (`view_needs_images`,
  `original_size_exceeds_image`) — **nunca se rellena**: formatos.md §2.

**Lo que lo sostiene**: toda puerta que entrene (`POST /runs`, `fv-train`, cada punto del
recorrido) pregunta a **`fv.validation.check_run`** antes de reservar el nombre. Función pura de
dos diccionarios (manifest × config), milisegundos, sin torch. La puerta más laxa es por la que
entra un recorrido.

### ② C consigo misma — la geometría foveada es consistente

Los asserts de instructionsNewNN.md §2 son un contrato, no comentarios:

```python
center_out % 2 == 0                # el anillo reparte simétrico
2 * periph_out + center_out == N
penetration < center_out // 2      # el núcleo no desaparece
k impar; padding = k // 2          # un kernel par desalinea máscaras
```

Viven en `fv.fovea.derive_dims` y el validador los ejecuta en **todas** las puertas — incluido
el runner de H, porque un espacio sobre `c_frac`/`pen_frac` genera puntos inválidos y hay que
descartarlos con la razón, no reventar dentro del job. `POST /networks/validate` los expone
síncronos a la UI (con la traza de dimensiones derivadas).

### ③ B + C + D → E — nombre y valor, y la huella

El run guarda, de C y D, **el valor (para reproducir) y el nombre (para agrupar)**; de B, la
ruta **más la huella del contenido**. Sin el nombre no se puede preguntar «¿qué runs usaron la
red X?» — la pregunta que H hace todo el rato. Sin la huella, un B reconstruido bajo el mismo
nombre es indistinguible (⑧).

### ④ E → F — el checkpoint se describe solo

`load_model()` lee `ckpt["config"]["model"]` y reconstruye la red **incluida su geometría de
entrada** — F nunca necesita el YAML de C ni el dataset B. Un `.pt` es portable (CPU → GPU).

### ⑤ B ↔ F — la vista foveada es una sola

El dataloader (entrenamiento) y la inferencia construyen la vista con **la misma función** de
`fv.fovea`. El test no pregunta «¿es correcta `build_foveated_input`?» sino **«¿ven
entrenamiento e inferencia la misma vista?»** (identidad del módulo + vista bit-idéntica sobre
la misma ventana). Duplicar esas líneas es lo que sale natural al escribir F; el test impide que
la duplicación vuelva.

### ⑦ G → todos — la dirección de dependencias

`fv.fovea`, `fv.metrics`, `fv.validation`, `fv.matrixview` **no importan nada de `fv`**.
`fv.models` importa solo de `fovea`; la red no sabe que A existe. Se testea leyendo imports
(tests.md §4).

### ⑧ H ↔ B, C — comparabilidad

> **El instrumento de medida no puede ser parte del experimento.**

- Todos los runs de un recorrido comparten el mismo B (misma huella). Si el espacio cubre C, los
  puntos son comparables **porque el B y la métrica lo son**, no porque la red lo sea.
- **Holdout por encima de B** (heredado, D16): un conjunto de imágenes apartado una vez, su
  propia fuente `<nombre>-holdout`, que ninguna configuración de B toca jamás. La métrica que lo
  hace posible se mide **por imagen**, no por ventana — así el mismo holdout sirve aunque cambie
  la geometría (protocolo.md §3).
- Los **dos `seed`** no se mezclan: el de B fija el split (fijo en un recorrido); el de D es el
  eje de réplica.

### ⑨ H ↔ D — el objetivo no puede depender de lo que barres

Si la pérdida tiene pesos (`lambda_*`) y esos pesos están en el espacio, **la loss no puede ser
el objetivo**: cada punto se mediría con una regla distinta y λ→0 «gana» por definición. El
validador del spec lo rechaza con `objective_varies_with_space`, **antes de reservar nada y sin
cargar optuna**. El objetivo correcto es la métrica de tarea por imagen (protocolo.md §2).

**Extensión propia de este proyecto**: si el espacio cubre C, el objetivo tampoco puede ser
nada cuya escala dependa de la geometría barrida (p. ej. un error medido «en celdas de la vista»
cambia de significado al cambiar `d`). Las métricas se definen sobre **la imagen original**, en
píxeles de la imagen, siempre.

### ⑩ X ↔ D — el device no es una receta

`device`/`num_workers` fuera de la identidad de D, congelados aparte en `execution`. **La trampa
concreta al llegar la GPU: `batch_size` es D.** Subirlo «porque ahora cabe» invalida la
comparación con todo lo de CPU; un batch mayor es un punto nuevo del espacio.

### ⑪ X ↔ D — reanudar solo es X si es bit-exacto

*N épocas de tirón == N−k épocas + reanudar k*, hasta el último bit — o el run lo **declara** en
su procedencia (`resumed_at`). Reanudar desde `{model, config, epoch}` sin optimizador, RNG,
scheduler y `best_monitor` produce un run con la misma procedencia y **otros pesos**, en
silencio. Reanudar el **recorrido** (re-encolar puntos) sí es seguro y es lo que se construye;
reanudar dentro de un run es diseño aplazado (formatos.md §4.2.2) hasta que se mida el ahorro.

---

## 3. Trampas conocidas

La lista razonada, con su evidencia, está en organizacion.md §3 del proyecto hermano y resumida
en [herencia.md](herencia.md) §4 y en CLAUDE.md. No se duplica aquí; la regla que las engloba:

> **Casi todas eran defaults.** Nadie las eligió: aparecieron por no elegir. El fallo típico de
> este dominio no revienta — produce un artefacto bien formado que mide otra cosa.

Trampas **nuevas, propias de la red foveada** (de instructionsNewNN.md y de P4 del hermano):

1. **Un kernel par desalinea las máscaras** (padding no entero). Los rangos calculados solo
   generan impares; un YAML escrito a mano puede no hacerlo → assert en el validador.
2. **`sum` con strides distintos por rama no alinea** (§7 de la spec). La elección
   `merge: sum|concat` cambia el `forward`; decidirla **antes** de lanzar un recorrido que barra
   strides — el validador rechaza `merge: sum` + `s_center != s_periph`.
3. **Enmascarar DESPUÉS de convolucionar** (opción B de la spec) es frágil (±1px al reconstruir
   la máscara a la resolución de salida). La decisión tomada es la A: enmascarar antes. No la
   reabras al «optimizar».
4. **El relleno de los bordes nunca es blanco/cero sin máscara**: cero significa «no hay texto»,
   acierta por casualidad y enseña una regla falsa. La máscara de cobertura lleva la fracción
   real por celda y acompaña al canal que puede tener relleno.
5. **`avg_pool2d` puede borrar trazos finos en la periferia** (medido en el hermano: reducir
   mucho deja la tinta en 0,2 %). `pool_mode` es un eje a barrer, no una constante.
6. **Una máscara desplazada o transpuesta no falla**: entrena igual con contexto mal etiquetado.
   La vista de depuración de entrada (V19 en ui.md) existe para cazarla, con la cobertura como
   número.
