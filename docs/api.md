# API

El contrato del API REST que expone este proyecto a la web app. Aplica los dominios de
[organizacion.md](organizacion.md); **si el API necesita un recurso que no es un dominio, o
mezcla dos, el error está en el API**. Heredado del hermano (sus reglas R1–R7 se adoptan tal
cual; herencia.md §1).

---

## 0. La capa

```
web app ──HTTP──▶ API ──llamadas──▶ fv (el dominio) ──▶ librerías
```

**El API posee HTTP y nada más.** Toda la lógica vive en `fv` y debe poder usarse sin el API —
los CLI (`fv-extract`, `fv-train`, `fv-sweep`) lo prueban. Regla mecánica: **si una función de
`app.py` no menciona HTTP, no es del API.**

## 1. Reglas

- **R1 — Un recurso por sustantivo del dominio.** Ni recursos de conveniencia ni dominios sin
  recurso.
- **R2 — Las palabras ambiguas no entran al vocabulario.** No hay `/models` (¿C o E?) ni
  `/datasets` (¿A o B?): son `/networks`, `/runs`, `/sources`, `/window-datasets`.
- **R3 — Síncrono o job según el tiempo (~1 s).** Extraer, entrenar, recorrer → **job (202)**.
  CRUD, validar, kernels, predecir una imagen, diagnósticos cacheados → síncrono.
- **R4 — Un error dice por qué y cómo se arregla**: `{code, message, hint}`. `code` es contrato
  (la UI reacciona a él). **Se valida antes, no durante**: 400 al entrar, jamás stack trace
  dentro del hilo del job. 400 = petición imposible; 404 = no existe; 409 = choca con el estado.
- **R5 — Polling incremental**: `GET /runs/{name}/metrics?since=N → {records, next}`. Nunca se
  reenvía el historial; `GET /runs/{name}` no incluye métricas.
- **R6 — Los agregados se calculan en el servidor.** El navegador nunca recibe 10⁵ filas; las
  tablas van filtradas y paginadas con `limit` acotado por la ruta.
- **R7 — Si se entrenó con ello, tiene identidad reproducible.** `POST /runs` y `POST /sweeps`
  aceptan **nombres** de red y receta. **Excepción única: la base inline de un recorrido generado
  por un estudio** (I, D-H2): sin nombre de catálogo, pero con `base_label` (agrupa) +
  `base_network_value` (reproduce) + `derivation.field_origin` (audita de dónde salió cada campo)
  — la procedencia se sostiene igual (contrato ③). Ninguna **otra** puerta acepta valores inline;
  `POST /runs` sigue exigiendo nombres.

## 2. El mapa de recursos

| Dominio | Recurso |
|---|---|
| **A** Fuente | `/sources` (+ `POST /sources/{id}/resize` → job) |
| **B** Dataset de ventanas | `/window-datasets` |
| **C** Red foveada | `/networks` (+ `POST /networks/validate`) |
| **D** Receta | `/recipes` |
| **E** Run | `/runs` |
| **E×B** Diagnóstico | `/runs/{name}/diagnostics/*` — caché, todo GET idempotente |
| **H** Recorrido | `/sweeps` |
| **I** Estudio (schedule OAT) | `/studies` (plan comiteable; genera recorridos, no ejecuta) |
| **F** Inferencia | `/runs/{name}/predict` |
| **X** Jobs | `/jobs` (+ `POST /jobs/{id}/cancel`, cooperativo) |
| **UI** Estado recordado | `/ui-state` (`GET`/`PUT`) — blob opaco de filtros/formularios, NO dominio |

## 3. Lo no evidente, recurso a recurso

### `/sources` (A)

Como el hermano: lista, metadatos, `samples` (anotables con el split de un B:
`?window_dataset=`), imagen por índice (`?w=` para miniaturas), y `resize` → job con sus 400
(`resize_needs_one_dimension`, `upscale_not_allowed` contra **todas** las muestras,
`source_exists`, `source_not_found`). Las rutas se resuelven **dentro del dominio** — no existe
`GET /image?path=`: allowlist de raíces (403 fuera) y CORS cerrado al origen del front. Aquí no
es teórico: **este API acabará corriendo en un server con GPU.**

### `/window-datasets` (B)

`GET` lista+manifest · `POST` → job · `GET /{name}` (manifest + fingerprint + `used_by`) ·
`DELETE /{name}` (**409 con la lista** si algún run lo referencia) ·
`GET /{name}/windows?split=&offset=&limit=` (paginado) · `GET /{name}/windows/{i}` (una ventana:
píxeles del recorte crudo, etiqueta, procedencia — **sin modelo**: inspeccionar el dato no exige
un run).

### `/networks` (C)

CRUD + **`POST /networks/validate`**: puro, síncrono, sin guardar. Devuelve las **dimensiones
derivadas** (`center_out`, `periph_out`, `penetration`, `periph_band`, `original_size`), los
**rangos calculados** (`kernel_range`, `stride_range`, `downsample_range` para ese `N`), el nº
de parámetros y la traza por rama — o un 400 con qué assert falló y cómo arreglarlo
(`center_not_even`, `penetration_too_large`, `kernel_must_be_odd`,
`merge_sum_needs_equal_strides`…). Alimenta en vivo la pantalla Redes: **el usuario ve lo que
`N` y las fracciones implican antes de guardar.**

### `/recipes` (D)

CRUD simple. El cuerpo es el catálogo de organizacion.md §1-D. `device`/`num_workers` **no
están** (contrato ⑩).

### `/runs` (E)

Como el hermano, entero: `POST` → job con **nombres** + `device` aparte; valida con
`fv.validation.check_run` **antes de crear el job y antes de reservar el nombre** (contratos
①②); 409 si el nombre existe — jamás se sobrescribe; `GET /{name}` con procedencia completa;
`/metrics?since=`; `PATCH` renombra (409 si corre); `DELETE` (409 si corre o si un recorrido lo
referencia); `POST /{name}/stop` cooperativo.

### `/runs/{name}/diagnostics` (E×B) — caché, no entidad

GET idempotentes sobre `(run, split)`; la tabla por ventana se calcula al primer GET y se
invalida sola (clave: run + huella de B + split + **mtime del checkpoint**). Agregados en el
servidor; `threshold` es parámetro de **consulta**, no de la clave — releer columnas guardadas
es lo que hace gratis el barrido de umbral. Las negativas con razón:
`run_without_provenance`, `run_has_no_checkpoint`, `window_dataset_changed` (la huella no
cuadra: contrato ⑧ cobrándose), `split_empty` → 409; parámetros imposibles → 400.

Los endpoints concretos (PR, mapas de error, galería peor-primero…) se fijan con F1; el patrón
es el del hermano.

### Introspección (`/runs/{name}/…`)

`GET /kernels` · `POST /feature-maps` · `POST /input-view` (la entrada canal a canal con su
máscara de cobertura — la vista de depuración fundamental aquí, ui.md V19) · sondas según ui.md.
Todos devuelven el payload de `matrixview` (números + min/max/mean + `truncated`) y **declaran
el trabajo de color** (`sequential | diverging`): el cliente no puede saber si mira un peso con
signo o una activación.

**Particularidad foveada**: kernels y feature maps van **por rama** (`branch: center|periph`).
Un filtro de la rama periférica opera sobre la vista reducida: el payload lo dice
(`branch`, `region`), porque pintarlos como si vieran la imagen original mentiría sobre la
escala.

### `/sweeps` (H)

```
GET  /sweeps
POST /sweeps            → job   {name, window_dataset, space, strategy, objective, budget, seed_policy}
GET  /sweeps/{name}             spec + progreso
GET  /sweeps/{name}/trials      tabla ordenada por objetivo
POST /sweeps/{name}/stop        cooperativo (corta entre puntos)
POST /sweeps/{name}/resume → job  (retira la petición de parada; 409 si corre o si cumplió presupuesto)
DELETE /sweeps/{name}           cascada: borra el recorrido Y sus runs hijos (409 si algo corre)
```

**Borrar en cascada, no huérfanos.** Un run hijo se niega a borrarse solo (`DELETE /runs/{n}` →
409: sus puntos se comparan juntos); el recorrido es su dueño, así que `DELETE /sweeps/{n}` los
borra a todos —hijos primero, padre después— y devuelve `{deleted, runs_deleted}`. Se niega con
`sweep_is_running` si el recorrido o cualquier hijo sigue en marcha: nunca borra trabajo vivo ni
deja un run apuntando a un padre inexistente.

- `space` admite campos de **C y/o D**; los rangos de geometría admiten `"auto"` (los pone
  `build_search_space(N, …)`).
- **Red base por nombre o inline** (D-H2, R7): el cuerpo lleva `base_network` (nombre) **o**
  `base_network` `null` + `base_network_value` + `base_label` + `derivation` (lo genera un estudio).
  El API **exige uno de los dos** (nombre XOR valor): `base_missing` si faltan ambos,
  `base_ambiguous` si vienen los dos. El validador `check_run`/`check_network` corre igual sobre el
  config resuelto — inline no es una puerta más laxa (contrato ⑫).
- 400 **antes de reservar nada**: `objective_varies_with_space` (⑨),
  `objective_depends_on_geometry` (⑨-extensión), puntos de geometría inválidos se **descartan
  declarándolos** en el spec resultante (no silenciosamente).
- Sobrevive a reinicios: el `lifespan` re-encola lo que quedó a medias desde disco.

### `/studies` (I) — el estudio planifica, no ejecuta

El schedule OAT como recurso. **No lanza jobs propios**: guía la generación de recorridos (H) paso
a paso (D-H1, contrato ⑫).

```
GET  /studies
POST /studies                    {name, window_dataset, base_recipe, objective, seeds, axes[]}
GET  /studies/{name}             plan + progreso (pasos, ganadores confirmados, sub-ejes desbloqueados)
POST /studies/{name}/next-sweep  → deriva la base del problema + arrastra ganadores → devuelve el
                                   spec de recorrido (base inline) PROPUESTO; no lo lanza
POST /studies/{name}/confirm     {step, winner} fija el ganador CONFIRMADO por el usuario y lo
                                   arrastra al siguiente paso (D-W1: la herramienta sugiere, el
                                   usuario aprieta el gatillo)
DELETE /studies/{name}           borra el plan; 409 si un recorrido hijo corre (no huérfanos)
```

- **`next-sweep` no reserva nada**: devuelve el spec para que el usuario lo revise y lance con
  `POST /sweeps` (base inline). La derivación (`window_size` → `N`/geometría) pasa por `check_run`;
  si el default estático es inválido para ese `W` cae al válido con su razón, y si ningún `N` es
  factible **afloja `c_frac` con la razón registrada** (D-G3) o devuelve 400.
- **El ganador lo propone el servidor, lo confirma el cliente**: `GET /studies/{name}` incluye,
  por paso terminado, la sugerencia coste/calidad (`δ`, métrica de coste) y los candidatos de la
  **frontera** con sus N semillas — pero no arrastra nada hasta el `confirm`.
- **Longitud dinámica**: fijar un ganador puede expandir un eje en sub-pasos (`channels[i]`); el
  progreso lo refleja, el presupuesto se cuenta al correr la cadena.

### `/runs/{name}/predict` (F)

Devuelve **todas las etapas** (por-ventana crudo → fusión → resultado por imagen), no solo la
última — sin la cruda, «salió mal» no es diagnosticable. Los knobs van en **unidades de la
ventana**, el payload los devuelve (las respuestas llegan desordenadas con sliders en vivo), y
el cliente que no sabe qué mandar manda `null` y adopta el default de F.

### `/ui-state` (conveniencia, no dominio)

`GET` devuelve `{}` cuando no hay nada guardado (ausente ≠ cero: nunca 404). `PUT` escribe un
`state/ui-state.json` **comiteable** (viaja al server con GPU) tras validar un **tope de tamaño**
(`ui_state_too_large`, R6): es un blob opaco de filtros/formularios del front, no una segunda
fuente de verdad para A–H. El front lo usa como snapshot de su `localStorage`; ni el API ni el
esquema de dominio dependen de su contenido.

## 4. Dónde el API hace cumplir los contratos

| Contrato | Dónde |
|---|---|
| ①② geometría y compatibilidad B↔C | `POST /runs` → 400 antes de crear job (y `check_run` otra vez dentro de `train()`: el CLI no pasa por el API) |
| ③ B en uso | `DELETE /window-datasets/{n}` → 409 con la lista |
| ⑧ huella | diagnostics → 409 `window_dataset_changed` |
| ⑨ objetivo | `POST /sweeps` → 400, validación pura sin optuna |
| ⑩ X fuera de D | `device` fuera de `/recipes`, en el cuerpo de `POST /runs` |
| ⑫ I planifica, H ejecuta | `POST /studies/{n}/next-sweep` deriva+valida (`check_run`), no reserva; `confirm` arrastra el ganador |
| ③ base inline reproducible | `POST /sweeps` acepta `base_label`+`base_network_value`+`derivation` (nombre XOR valor) |
| R7 | `POST /runs` exige nombres; `POST /sweeps` nombre **o** base inline con procedencia (única excepción) |

> La regla que sostiene ① no es el endpoint: es que **todas las puertas** (`POST /runs`,
> `fv-train`, cada punto del recorrido) preguntan a la misma función **antes de reservar el
> nombre**. Validar después de reservar deja un `runs/<name>/` muerto y el reintento contesta
> «ya existe».
