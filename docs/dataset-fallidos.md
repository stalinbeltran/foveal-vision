# Los casos en que una red falla, como dataset

`scripts/dataset_fallidos.py` (módulo: `src/fv/fallidos.py`) pasa una red por todas
las imágenes de un dataset de ventanas, puntúa cada imagen **a nivel de párrafo**, y
escribe un **dataset de ventanas nuevo** con las peores.

Lo que sale es un **B de pleno derecho** — `windows.npz` + `manifest.json` +
`split.json`, el formato de [formatos.md §4.1](formatos.md) —, así que `fv-train`
entrena sobre él y la web app lo lista sin que nada sepa de dónde salió. Ese es el
punto: un informe de «estas 87 imágenes salieron mal» se lee una vez; un dataset se
entrena, se vuelve a inferir y se compara consigo mismo dentro de seis meses.

```bash
# las redes aprobadas en inferencia.json, sobre el dataset con el que se entrenaron
.venv/bin/python scripts/dataset_fallidos.py --verdad ventanas

# mirar sin escribir
.venv/bin/python scripts/dataset_fallidos.py --nn fov16-mask-p20 --seco

# desde Telegram
/use fallidos      →      --nn fov16-mask-p20 --verdad ventanas --seco
```

## Qué es un error, y qué es «peor» — escrito antes de mirar ningún número

Un **error** es un párrafo que la red no encontró (`fn`) o uno que se inventó (`fp`),
emparejando por IoU ≥ `--iou` (0,5). Es exactamente `fv.metrics.paragraph_f1`, la
métrica que importa ([protocolo.md](protocolo.md) §2): aquí **no se define un segundo
número**.

**Peor = más errores.** Los empates los rompen, en este orden: menor f1 de párrafo,
menor IoU medio de los emparejados (**sin emparejar nada = lo peor**, no cero — ausente
no es cero), e índice de imagen. Los dos últimos escalones existen sólo para que el
resultado sea el mismo cada vez que se repita.

Entran las imágenes con al menos `--min-errores` (1), hasta `--max-imagenes` (0 = sin
tope). El tope se queda con **las peores**, no con las primeras.

## ⚠ El diagnóstico que hay que leer: de quién es la culpa

«La red falla en esta imagen» son **dos averías distintas con arreglos opuestos**, y sin
separarlas un dataset de fallos manda a reentrenar redes que están perfectas:

| Qué pasó | De quién es | Qué se toca |
|---|---|---|
| **no vio la esquina** | de la red (C/D) | más capacidad, más datos, otro punto del barrido |
| **la vio y el emparejado la juntó mal** | de **F** | `_reconstruct` de `fv/inference/predict.py`, un voraz TL→BR heredado |

Cada imagen lleva su `esquinas: {tp, fp, fn}` y una bandera `solo_emparejado`, y el
resumen da el reparto. **Medido el 2026-09-02** sobre las tres redes aprobadas y las
1000 imágenes de `dirty1000-80px-16px-r20260827`:

| red | imágenes con ≥1 error | de ésas, **sólo emparejado** | esquinas (tol 8 px) |
|---|---:|---:|---|
| `demo-fov16-optimo` | 433 | **188 (43 %)** | tp 9612 · fp 211 · fn 396 |
| `fov16-edge-p20` | 487 | **289 (59 %)** | tp 9788 · fp 261 · fn 220 |
| `fov16-mask-p20` | 351 | **262 (75 %)** | tp 9933 · fp 150 · fn 75 |

O sea: las tres detectan esquinas muy bien (`fov16-mask-p20` recupera el **99,25 %**), y
la mayor parte de lo que se ve como «la red falla» es el emparejado. Se confirmó **a
ojo** dibujando verdad y predicción sobre cuatro imágenes: las cajas predichas unen el
TL de un párrafo con el BR de **otro**. Es literalmente lo que avisa el docstring de
`_reconstruct`: *«the place to touch if paragraphs come out wrong while corners come out
right»*.

⚠ **Ese reparto NO entra en el criterio de selección**, a propósito: el criterio se
declaró antes de medir, y cambiarlo después de mirar es lo que este proyecto no hace.
El diagnóstico informa; no ordena.

✅ **Y ese diagnóstico llevó al arreglo (2026-09-02).** `_reconstruct` usaba dos de los cuatro
tipos de esquina; usar las cuatro (`--reconstruir quad`) sube el f1 de tarea **+0,18 a +0,28** y
recorta los errores de párrafo entre un 74 % y un 89 %. El defecto sigue siendo el heredado
—cambiarlo movería todo lo publicado—, así que los tres datasets de abajo están construidos con
`tlbr`. Todo, en [reconstruccion-parrafos.md](reconstruccion-parrafos.md).

## ⚠ La verdad: dos vías, y la segunda hay que pedirla

El [contrato ⑬](organizacion.md) dice que la verdad de párrafos sale de la **fuente**
(A), y que si no está se falla — «nunca se puntúa contra las etiquetas de ventana».

- **`--verdad fuente`** (defecto): los bloques de A, como hace `fv.task`. Si la fuente no
  está, **falla** con `verdad_necesita_fuente` y dice cuál es la otra puerta. No se cae
  sola al camino de al lado: eso sería medir una cosa creyendo medir otra.
- **`--verdad ventanas`**: recompone las cajas desde `y` del propio `.npz`. El extractor
  guarda la esquina **verbatim** (`y[ci] = (1, (cx−wx0)/n, (cy−wy0)/n)`), así que
  `wx0 + y·n` la devuelve. No es lo que ⑬ prohíbe —⑬ prohíbe puntuar contra las
  etiquetas *como si fueran* párrafos—, pero **es un camino degradado**, y por eso hay
  que pedirlo.

**Por qué existe:** la fuente `local/dirty-1000-80px` se perdió al rehacer la máquina
(sus PNG nunca entraron en git) y el `windows.npz` sí está commiteado. Sin esta vía, los
fallos de las tres redes aprobadas no se pueden mirar.

**Lo que cuesta, medido:** un párrafo **cortado por el borde de la imagen** no se
recupera — su esquina cae fuera de la rejilla y ninguna ventana la ve. En
`dirty1000-80px-16px-r20260827` son **13 de 1000**. Esas imágenes salen marcadas
`gt_completa: false` y **fuera de la selección**, porque su cuenta de errores no es
creíble: lo que la red detectó ahí se contaría como invento suyo. `--incluir-gt-parcial`
las mete.

**Lo que NO cuesta:** nada de precisión. Medido el 2026-09-02 sobre ese dataset, la
dispersión de una misma esquina vista desde ventanas distintas es **0,0 px exactos**, y
dos esquinas distintas del mismo tipo en una imagen nunca están a menos de **9,52 px**.
La tolerancia (`TOL_PX = 0,25`) vive en mitad de ese hueco. Y hay test que compara la
reconstrucción contra la fuente, imagen a imagen, en un mundo donde existen las dos.

De dónde salió la verdad queda escrito en `manifest.json → fallidos.verdad`: dentro de un
año, estos números no se pueden leer como si se hubieran medido contra A.

## El split del dataset de salida

`--split-salida`:

- **`conservar`** (defecto): cada imagen mantiene el split que tenía en el dataset base.
  Es el defecto porque es el único que no miente sobre lo que la red evaluada vio.
- **`rehacer`**: reparto nuevo por imagen (`--val-frac`, `--test-frac`, `--seed`). Para
  cuando el dataset va a entrenar una red **nueva**, a la que el split viejo no le dice
  nada.
- **`train`**: todo a train.

## Qué deja en disco

```
window-datasets/<red-corta>-fallidos/
    windows.npz       los seis arrays, recortados a las imágenes elegidas
    manifest.json     el contrato de siempre + el bloque `fallidos`
    split.json        índices de A por split
    fallos.json       el diagnóstico por imagen: predicción, verdad, errores, esquinas
    imagenes/*.png    sólo con --png
```

- **`sample_idx` conserva los índices originales de A** — no se renumera. Es lo que
  permite cruzar una fila de aquí con el dataset base y con la fuente, y es correcto por
  contrato: `sample_idx` nunca indexó `images`.
- **El `fingerprint` es el del `.npz` nuevo.** Si el criterio deja pasar *todas* las
  imágenes, el npz sale byte a byte igual que el base y la huella coincide: eso no es un
  fallo del recorte, es la prueba de que no toca lo que no debe.
- **Se construye al lado (`<nombre>.parcial`) y se renombra.** Un corte a mitad —el bot
  se reinicia— no puede dejar un directorio con manifest y sin npz que además bloquearía
  el reintento.
- **No se sobrescribe nunca**, como `fv-extract`: `--nombre` para otro.
- **No commitea.** Imprime el comando, como `fv-train` y como `promover`.

⚠ `task_score` **no** puede puntuar una red contra su propio dataset de fallos: comparte
`source_id` con el de entrenamiento, así que salta `holdout_shares_source`. Es correcto
—no es un holdout— y es justo por eso que este script trae su propia evaluación.

## Los tres que ya están hechos (2026-09-02)

| dataset | red | imágenes | ventanas | npz |
|---|---|---:|---:|---:|
| `optimo-fallidos` | `demo-fov16-optimo` | 427 | 59 780 | 1,17 MB |
| `edge-fallidos` | `fov16-edge-p20` | 481 | 67 340 | 1,29 MB |
| `mask-fallidos` | `fov16-mask-p20` | 346 | 48 440 | 0,93 MB |

Los tres con `--verdad ventanas`, `--min-errores 1`, sin tope y `--split-salida
conservar`. Verificado entrenando `fov16-mask` 2 épocas sobre `mask-fallidos`:
`val_f1` 0,15 → 0,54, `best.pt` escrito.
