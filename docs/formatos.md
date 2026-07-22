# Formatos en disco

Los artefactos que este proyecto escribe y lee, y cómo evolucionan sin romper lo ya generado.
Son **contratos**: los escribe un módulo y los leen varios, meses después. Un cambio descuidado
no da una excepción — da resultados peores sin que nadie se entere.

Heredado del proyecto hermano (formatos.md de `image-text-finder`), que pagó cada regla con un
fallo real.

---

## 1. La regla que gobierna todo: **ausente ≠ cero**

> **Rellenar un campo ausente solo es legal si el consumidor no lo usa. Un lector que necesita
> un campo ausente falla, alto y con la razón; nunca lo inventa.**

Y su afinación — **dato** ausente ≠ **declaración** ausente:

| Tipo | Ejemplo | Ausente significa | Comportamiento |
|---|---|---|---|
| **Dato** | el array `images` | No se sabe / no se tiene — no se puede inventar | **Fallar** |
| **Declaración** | `has_images` en el manifest | «No lo tengo» | Default «no»: rechaza de más, nunca de menos |

Por eso los manifests no necesitan migración: la declaración que falta se lee como `False` y el
validador se niega. **El default correcto sale solo.**

## 2. Política de versionado

Cada artefacto lleva `format_version` (entero), que **nace en 1 y no hay v0** (no existen datos
anteriores). Reglas:

| Cambio | ¿Bump? | El lector |
|---|---|---|
| Añadir campo cuyo default reproduce lo viejo | No | Lee ambos |
| Añadir campo que el consumidor **necesita** | No — pero **falla si falta** (§1) | Error con razón |
| Cambiar significado/unidad/orden/escala de un campo | **Sí** | Rechaza o migra explícito |
| Quitar un campo | **Sí** | — |

No se bumpea por costumbre: el sniffing aditivo + la declaración en el manifest cubren lo demás.

## 3. El criterio caché vs artefacto

> **Se versiona/guarda lo que no se puede recalcular; lo recomputable es un caché.**

Un caché: (a) es función pura de cosas con identidad, (b) se puede borrar sin perder nada,
(c) lleva la huella de sus entradas **en la clave** — y si está memmapeado, la huella va **en el
nombre del directorio** (en Windows no se puede borrar un fichero mapeado: una versión nueva es
un directorio nuevo). Todo caché con constructor concurrente lleva **lock de build**.

Cachés previstos (todos en `data/cache/`, gitignoreados): índice de offsets de fuentes, filas
memmapeadas de B si hace falta, tabla de diagnóstico por ventana (clave con **mtime del
checkpoint**: un run vivo reescribe `best.pt`).

## 4. Los artefactos

### 4.1 `data/window-datasets/<name>/` — el dataset de ventanas (B)

**`windows.npz`** — arrays paralelos de largo N (nº de ventanas etiquetadas) + las imágenes:

| Array | Forma | Qué |
|---|---|---|
| `y` | (N, 4, 3) float32 | `[exists, x, y]` por esquina, orden de `corner_order` (C9). `x, y` **normalizados a [0,1] dentro de la ventana etiquetada** (la que fije F1b) |
| `sample_idx` | (N,) int32 | De qué imagen de A salió (índice de A) |
| `window_xy` | (N, 2) int32 | `(x0, y0)` de la ventana etiquetada en esa imagen |
| `split` | (N,) int8 | 0 train / 1 val / 2 test — **por imagen**, jamás por ventana |
| `images` | (S, H, W) uint8 | Las imágenes de A **enteras**. Largo S (imágenes), no N |
| `images_sample_idx` | (S,) int32 | Fila de `images` → índice de A. `sample_idx` **no** indexa `images` (un filtro sobre A los desalinea — trampa medida) |

**No hay array `X`**: la vista foveada se construye en el dataloader a partir de
`images[...]` + `window_xy` + la geometría de C (organizacion.md §1-B, contrato ⑤). Es la
diferencia deliberada con el hermano, y lo que hace barrible la geometría sin re-extraer.

- El extractor **mide** `S×H×W` y rechaza sobre presupuesto (`images_budget_exceeded`, 1 GB)
  con el número y el arreglo (reducir la fuente, menos imágenes). Sin camino degradado.
- Imágenes de tamaño no uniforme: `images_not_uniform`. La variante ragged no se construye
  hasta que una fuente la pida.

**`manifest.json`** — el contrato (el `.npz` es la carga):

```jsonc
{ "format_version": 1,
  "fingerprint": "sha256:…",           // contrato ⑧ — cambia si cambia el contenido
  "has_images": true,
  "images": {"shape": [S, H, W], "bytes": …, "budget_bytes": …},
  "source_id": "…",
  "config": { /* el WindowExtractConfig entero: ventana, criterio de etiqueta, split, seed */ },
  "num_samples": S, "num_windows": N,
  "corner_order": ["TL","TR","BR","BL"],      // semántica, no adorno: si se pierde, y significa otra cosa
  "label_window": "center",                    // F1b: contra qué ventana van normalizados x,y
  "windows_per_split": {"train": …, "val": …, "test": …},
  "positives_per_corner": {"TL": …, …} }       // el desbalance, para pos_weight sin abrir el .npz
```

Lo que no se puede deducir del array (órdenes, mapeos entero→split, semántica de `y`) **vive en
el manifest**: si se pierde, los datos cargan y significan otra cosa.

**`split.json`** — índices **de A** por split (permite el cruce A×B de Predecir).

### 4.2 `runs/<name>/` — el run (E)

| Fichero | Qué |
|---|---|
| `config.json` | `format_version`, la receta (D), la red por valor (C), `provenance` (§4.2.1) y `execution` (X) **aparte** |
| `metrics.jsonl` | Una línea JSON por época: `{epoch, train_loss, val:{…}, lr, seconds}`. **Append-only**; el lector descarta la última línea si está a medias |
| `best.pt` | `{"model": state_dict, "config": dict, "epoch": int}` — el **entregable** (contrato ④) |
| `last.pt` | Ídem — el punto de guardado. El estado de reanudación bit-exacta (§4.2.2) es diseño aplazado |
| `summary.json` | `{run, epochs_run, epochs_requested, stopped_early, cancelled, monitor, best, final}` — `best` es `null` si el monitor no midió, **jamás ±inf** (Infinity no es JSON) |
| `status.json` | Estado **explícito**: `queued \| running \| done \| error \| cancelled` |
| `stop.json` | La **petición** de parada: `{requested_at, reason}` — cooperativa, se lee a fin de época |

**Escritura en Windows**: todo JSON reescribible va con fichero temporal + `os.replace`, **con
reintento con deadline en escritor y lector** (medido en el hermano: un lector y un escritor
peleándose 4 s dieron 5111 replaces fallidos; el patrón POSIX no porta). `metrics.jsonl` es la
excepción: append-only, nunca se reemplaza.

#### 4.2.1 `provenance`

```jsonc
{ "provenance": {
    "window_dataset": {"name": "…", "fingerprint": "sha256:…"},
    "network":        {"name": "…", "value": { … }},   // nombre agrupa, valor reproduce
    "recipe":         {"name": "…", "value": { … }},
    "sweep":          null,                            // o el recorrido padre
    "git_commit":     "…",                             // o la razón de no saberlo — nunca null silencioso
    "environment":    {"python": "…", "torch": "…", "platform": "…", "device": "cpu|cuda:…"}
} }
```

`environment.device` importa aquí más que en el hermano: **el plan es comparar CPU con GPU**.

#### 4.2.2 Reanudación dentro de un run — diseño, NO construido

Reanudar es X **solo si es bit-exacto** (contrato ⑪). Haría falta guardar en `last.pt`:
`optimizer.state_dict()`, `scheduler.state_dict()` (o `null` = «no tiene», distinto de faltar),
estados RNG (`torch`, `numpy`), `best_monitor` (sin él, la primera época tras reanudar machaca
un `best.pt` superior), y truncar `metrics.jsonl` a `epoch` líneas. **No se construye hasta
medir el ahorro** (decisiones.md); reanudar el *recorrido* (re-encolar puntos) sí se construye.

### 4.3 `configs/` — las definiciones (C y D)

`configs/networks/*.yaml` y `configs/recipes/*.yaml`. YAML porque los escribe una persona; **se
versionan en git: son fuente, no artefactos**. `format_version` es del fichero y se quita al
congelar dentro de un run.

En un YAML de red van **solo los parámetros fundamentales** (organizacion.md §1-C). Los
derivados (`center_out`, `original_size`, `padding`…) no se escriben nunca: se calculan en
`fv.fovea.derive_dims` — un derivado escrito es una copia que diverge.

### 4.4 `sweeps/<name>/` — el recorrido (H)

- **`spec.json`** — nuestro: lo fijo (B + huella), el espacio (campos de C/D con rango o
  `"auto"` → `build_search_space`), estrategia, objetivo (+ dirección), presupuesto (y su
  unidad: épocas o segundos), estado.
- **`optuna.db`** — del motor. **La frontera importa**: los trials de optuna no son nuestros
  runs; un trial lanza un run (`{sweep}-{trial:04d}`) y guarda su nombre.

Reanudable desde disco: contar los terminados en `optuna.db` y correr el resto.

### 4.5 Fronteras: lo que no es nuestro formato

`labels.jsonl` (A) lo define `image-text-sample-generator` (`SAMPLE_FORMAT.md`). No lo
especificamos: especificamos **qué consumimos** (`index`, `width/height`,
`blocks[].{kind,angle,quad}` — `quad` (4,2) horario desde TL). Si escribimos derivadas (resize),
somos un **segundo productor obligado a seguir al primero**: los campos que no consumimos se
copian tal cual — salvo coordenadas, que se reescalan **todas, recursivamente, o ninguna**
(`box`, `lines[]`, `words[]` anidados: la trampa medida).

### 4.6 `data/sources/<name>/` — fuente derivada (A′)

Mismo formato de A + bloque `derived`:

```jsonc
{ "derived": {
    "from": "…",                 // id DIRECCIONABLE del padre (no el declarado: no es único — medido)
    "from_declared_id": "…",
    "op": "resize",
    "request": {"width": 320},   // width XOR height
    "size": [320, 240],
    "scale": [0.5, 0.5],         // sx, sy REALES medidos de la salida, no el factor pedido
    "resample": "lanczos",
    "created": "…" } }
```

`derived` ausente ⇒ fuente original (la ausencia significa; §1). Solo reducir
(`upscale_not_allowed`, comprobado contra **todas** las muestras). Cuidado medido: reducir mucho
borra la tinta aunque la geometría siga bien — mirar una derivada antes de extraer de ella.

## 5. Qué se versiona en git

> **Se versiona la descripción, se ignora la carga.**

```gitignore
/data/
!/data/window-datasets/*/manifest.json
!/data/window-datasets/*/split.json
/runs/*/*.pt
# config.json, metrics.jsonl, summary.json de runs → versionados
/sweeps/*/optuna.db
# spec.json → versionado
```

Medido en el hermano: descripción 105 KB vs carga 38,5 MB — el 0,3 % del peso, y es el registro
de la investigación. Contra asumido: `git status` sucio mientras un run vive.

## 6. Qué se testea de aquí

En `test_contracts.py`: manifest y `.npz` cuadran; el split es por imagen; la huella cambia con
el contenido y no con el nombre; un B sin `images` → la puerta se niega con la razón; `summary`
sin `±inf`; los órdenes/semántica viajan en el manifest.
