# foveal-vision — instrucciones para Claude

**El mismo problema que `image-text-finder`, con otra red y con recorridos automáticos.** La
tarea es idéntica al proyecto hermano: detección de esquinas de párrafo por ventana (`TL, TR,
BR, BL` con `[exists, x, y]`) y reconstrucción de párrafos en la imagen — más adelante líneas,
y luego palabras. Lo que cambia: **la red** es una NN con **muestreo foveado y ramas por
región**, totalmente parametrizada (el centro a resolución completa; la periferia abarca más
área a menor resolución; dos ramas convolucionales que se suman en la banda de penetración), y
**los recorridos (sweeps) barren automáticamente las configuraciones de esa red** además de las
recetas.

**La especificación de la red es [instructionsNewNN.md](instructionsNewNN.md)** — ese documento
manda sobre todo lo que toque la arquitectura. Su principio rector gobierna el proyecto entero:
*todo dato es un parámetro*; las dimensiones y los **rangos de búsqueda se calculan** a partir de
`N` y unas pocas fracciones, nunca se escriben a mano.

El objetivo operativo: poder **preparar series de runs secuenciales** —pruebas cortas en esta
máquina (CPU), luego largas en un server con GPU— que recorran configuraciones de red y
parámetros **sin intervención humana** (recetas de recorrido), y poder **verificar cada objeto
creado** (fuente, dataset, red, run, recorrido, análisis) desde una web app.

---

## Estado actual — léelo primero

> **2026-07-25 — BORRAR UN ESTUDIO ARRASTRA SUS RECORRIDOS (antes los dejaba huérfanos).**
> `StudyStore.delete` borraba solo el estudio (plan+progress), no los recorridos que generó
> (`{estudio}-s{i}-{eje}`). Al borrar y **recrear un estudio con el mismo nombre**, el recorrido de
> la versión anterior quedaba huérfano y el siguiente `advance` chocaba con `sweep_exists` (nunca
> sobrescribe — correcto) sin salida. Ahora `fv.studies.driver.delete_study` **arrastra** a los
> recorridos del estudio (por `spec.study`, vía `SweepStore.used_by_study`) y estos a sus runs
> (reusa `delete_sweep`); rechaza ANTES de borrar nada si algún recorrido/run está `queued`/`running`
> (`study_has_live_sweeps`, 409 — R4). El endpoint `DELETE /studies/{name}` usa el arrastre. **87
> tests en verde** (+2 de contrato: ciclo borrar→recrear→avanzar sin colisión; guarda de vivo).
> ⚠ Espejo del arrastre recorrido→runs; simétrico con la auditoría CRUD del 2026-07-25.
>
> **2026-07-25 — SEMILLAS DEL ESTUDIO: N SEMILLAS EN CADA PUNTO (antes se ignoraban).** El `seeds`
> del plan de un estudio se **validaba y guardaba pero nunca generaba runs**: cada punto del eje se
> entrenaba con una sola semilla (la del recipe base), pidieras 1, 3 o 5. Por decisión del usuario
> se implementó **N semillas en cada punto**: el generador (`fv.sweeps.generate.build_generated_spec`)
> añade un **segundo eje `seed=[s0..s0+N-1]`** junto al eje real cuando `seeds>1`, así el barrido
> produce `len(rango)·N` runs (uno por `(valor, seed)`) y la banda existe en todo el eje. Cableado:
> `studies.advance` pasa `plan["seeds"]`; también `fv-oat --seeds` y `POST /sweeps/generate {seeds}`.
> El nombre del run lleva la semilla (`…-d2_seed1 .. _seed3`, `point_run_name` seed-aware). El
> **ganador agrega por valor de eje** (`fv.sweeps.winner.aggregate_seeds`): rankea la **media** de
> las semillas + banda min–max + `n_seeds`, nunca la réplica con suerte (recipe.py: seed es el eje
> RÉPLICA). `seeds=1` = sondeo rápido de antes (sin eje de semilla). Divergencia registrada en
> [barrido-por-ejes.md](docs/barrido-por-ejes.md) §11.1: el esquema D-M1 «confirmación N-en-frontera»
> queda como modo alternativo NO construido. **85 tests en verde** (+5 de contrato); verificado en
> vivo con `fv-oat --seeds 2` sobre `synth-b16` (2 runs, seeds 1/2 distintas en disco, ganador
> agregado). ⚠ **El estudio `batchSize_fast-80px-5seeds` que ya corría se generó ANTES del arreglo:
> su recorrido no tiene eje de semilla (sigue en 5 runs, seed=1). Para obtener las 5 semillas hay
> que borrarlo y recrearlo.**
>
> **2026-07-25 — CURVAS DEL RECORRIDO: OVERLAY MULTI-RUN EN RECORRIDOS.** Al elegir un recorrido,
> la pantalla **Recorridos** superpone las curvas de val (loss / f1 / pos_err_px) de **todos sus
> runs** en tres small-multiples (la misma vista V14 de RunDetail, pero N líneas). Dos modos, por
> decisión del usuario: **líneas por run** (una por run, leyenda de casillas para ocultar/mostrar
> cada uno o todos, énfasis al pasar el ratón) y **media ± banda** (agrupa runs que comparten
> config salvo `seed`: línea media + sombra min–max; con 1 semilla la banda es degenerada, a
> propósito y anunciado en el subtítulo). El color **sigue a la entidad** (índice de trial / orden
> de grupo), nunca al rank, así ocultar/ordenar no repinta a los demás. La paleta de 8 hues vive en
> `tokens.css` (`--series-1..8`, claro/oscuro), validada CVD para las dos superficies; identidad
> siempre por leyenda+etiqueta, nunca por color solo. Los datos ya existían: `/sweeps/{n}/trials` +
> `/runs/{r}/metrics` (fan-out en el poll de 3 s; runs terminales se traen una vez). **Cero rutas
> nuevas de backend.** Verificado: `web` typecheck+build limpio; Playwright sobre el recorrido vivo
> `batchSize_fast-80px-s0-batch_size` (5 runs) — ambos modos, toggle y ocultar/mostrar todo, **sin
> errores de consola**. `LineChart` extendido (banda + serie atenuada + leyenda opcional) sin
> romper RunDetail. `SweepCurves.tsx` es nuevo. ⚠ `scripts\verify_ui.py` sigue apuntando a
> `test4-s0-n_layers` (borrado): su interacción de Recorridos hay que reapuntarla al recorrido vivo.
>
> **2026-07-25 — AUDITORÍA CRUD CROSS-PÁGINA + REFRESCO DE LISTAS EN VIVO.** Se revisó el CRUD de
> las 12 pantallas buscando «borrar aquí rompe allá». Arreglos:
> - **Diagnóstico/Predecir** ya no piden un run recordado en `localStorage` que fue borrado o
>   renombrado (la migración de sufijo de eje dejó nombres huérfanos): **gatean por pertenencia a la
>   lista, no por verdad**, y la doomed-request ya no revienta ni pisa la carga válida. Además
>   **refrescan la lista de runs en vivo** (sondeo 3 s, como Runs/Recorridos/Estudios), gateando la
>   carga pesada con un booleano `runReady`/`sourceReady` estable para no re-computar en cada pasada.
> - **Borrado cruzado por-nombre (R4):** borrar un **dataset B** que fija un **recorrido**
>   (`spec.window_dataset`) o un **estudio** (`plan.window_dataset`), o una **receta D** que fija un
>   estudio (`plan.base_recipe`), devolvía 200 y rompía la otra pantalla *dentro del job* al
>   reanudar/avanzar. Ahora se rechaza en la puerta con **409 + razón+arreglo** nombrando al referente
>   (`SweepStore/StudyStore.used_by_dataset`, `StudyStore.used_by_recipe`; código `recipe_in_use`). La
>   asimetría es deliberada: las referencias **instantánea** (run/recorrido copian los VALORES de C/D
>   inline) SÍ permiten borrar C/D sin romper el run ni su diagnóstico.
> - **Verificado:** `tests/test_crud_integration.py` fija el grafo (instantánea no obliga, por-nombre
>   sí; borrar el recorrido de un estudio lo deja legible). **80 tests en verde**; guards probados en
>   vivo por HTTP; ciclo crear→borrar y refusal-en-UI por Playwright, 12 rutas sin errores de consola.
> - **Artefactos:** por decisión del usuario **no se borró nada**. Todos los runs/recorridos/estudios
>   actuales **cargan** — la nota de §13 sobre checkpoints incompatibles (`fov-run-*`, `cli-run-1`)
>   quedó **OBSOLETA**: hoy cargan y se diagnostican/predicen sin error.
>
> **2026-07-24 — EJES DE BARRIDO: `N`/`c_frac` REHUSADOS + BUDGET NO COLAPSA `epochs` +
> VERIFICADOR DE TODOS LOS EJES.** Un recorrido generado por un estudio (`test2-s0-N`) quedaba en
> `done 0/3` sin razón visible: barría el eje `N` con `c_frac` fijo, y como `center_out =
> round_to_even(N·c_frac)` está atado por ①a al `window_size` del dataset, **cada punto** daba una
> fóvea != la ventana. `expand_points` solo validaba geometría (`check_network`), no la costura con
> el dataset (`check_run`), así que los 3 puntos pasaban como "válidos" y morían con
> `window_size_mismatch` dentro del job (`RunError`→`continue`) — la trampa que R4 prohíbe. Arreglos:
> - **`N` y `c_frac` no son ejes** (`WINDOW_SIZE_FIELDS`): se rechazan en las **dos puertas** —
>   `check_sweep` (H) y `validate_plan` (I)— con `axis_breaks_window_size` (razón + arreglo: barre
>   `d`, o usa un dataset con esa ventana), antes de reservar nada. Como ningún otro eje toca
>   `center_out`, `check_network` por punto sigue bastando.
> - **Budget no colapsa `epochs`:** el runner solo aplica `budget.epochs` si el punto no barre
>   `epochs` (antes lo pisaba siempre → el eje no hacía nada, en silencio).
> - **`scripts\verify_axes.py`:** corre un recorrido real por CADA eje de C y D. Verificado hoy:
>   **26/26 ejes** (11 red + 13 receta + `N`/`c_frac` rehusados), 0 fallos. Ejes probados listados
>   en el README (§«Qué ejes se pueden barrer»).
>
> **74 tests en verde**. Artefactos muertos del usuario borrados (`studies/test2`,
> `sweeps/test2-s0-N`, `sweeps/test1-s0-N`). Fresco y cargable sigue siendo `fov-16-param` +
> `oat-d-demo`.
>
> **2026-07-24 — PARADA DE RECORRIDOS: CORTE EN VUELO + RECONCILIACIÓN DE ESTADO MUERTO.** Dos
> arreglos sobre la parada de recorridos (H):
> - **Corte del punto en vuelo (feat 1):** `train` acepta `should_stop`; el runner le pasa
>   `lambda: store.stop_requested(name)`, así una parada pedida al recorrido corta el punto **en
>   marcha** en su siguiente frontera de epoch, no solo entre puntos. La cooperación seguía siendo
>   entre puntos: un run largo ignoraba la parada hasta acabar.
> - **Reconciliación de `running` muerto (feat 2):** quien marca un recorrido/run `running` graba
>   su **PID dueño** (`fv.proc.pid_alive`, portable Win/POSIX sin psutil). `SweepStore.reconcile`
>   y `RunStore.reconcile` sanan `running`→`interrupted` cuando el proceso dueño ya no existe
>   (caída/reinicio del API/hibernación) — se llama al leer (`GET /sweeps`, `GET /sweeps/{name}`,
>   `sweep_trials`). Cierra la trampa heredada "un crash queda running para siempre". Erra seguro:
>   dueño vivo o sin PID (legacy) → intacto. El runner ahora redó todo lo no-(done|cancelled), así
>   que un punto `interrupted` se rehace al reanudar. `interrupted` es terminal: borrable y
>   reanudable; badge ámbar en la UI. **69 tests en verde**, 12 pantallas Playwright sin errores,
>   reconciliación verificada por HTTP. Causa raíz de `test1-s0-n_layers` pegado en running: su job
>   murió y nadie leía el `stop.json`.
>
> **2026-07-24 — BARRIDO POR EJES (OAT) IMPLEMENTADO Y VERIFICADO.** Se construyeron las cinco
> piezas de [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) §14, respetando las decisiones
> cerradas de su §13:
> - **Builder paramétrico (C)**: `fv.models.builder` honra `n_layers` y `channels` por capa
>   (D-C3), stride solo en la 1ª capa (D-S1); no-regresión bit-exacta para `n_layers=2` con
>   `channels=[16,32]`. Lee `ch1/ch2` viejo, escribe siempre `channels`.
> - **Derivador de base (G/C)**: `fv.models.derive` — de `window_size` deriva `N`/geometría
>   (①a), defaults estáticos, ganadores arrastrados, corrección de inválidos con razón, `N` mínimo
>   (D-G2), afloje de `c_frac` con razón (D-G3).
> - **Base inline + generador P1 (H)**: `fv.sweeps.generate` + CLI **`fv-oat`** + `POST
>   /sweeps/generate`. `base_network=null` + `base_label` + `derivation` (D-H2). Barrer `n_layers`
>   redimensiona `channels` a `[16]*L` (§6.1).
> - **Arrastre del ganador (I/H)**: `fv.sweeps.winner` — regla coste/calidad con δ (D-W1),
>   sugiere y el usuario confirma; `GET /sweeps/{name}/winner`.
> - **Estudio OAT (I, dominio nuevo `studies/`)**: `fv.studies` (plan.json comiteable +
>   progress.json vivo), CLI **`fv-study`**, endpoints `/studies/*`, pantalla **Estudios**. Guía
>   y no ejecuta; expande `channels[i]` al fijar `n_layers`.
>
> **65 tests en verde** (~25 s) incluyendo el contrato ⑫ (estudio↔recorrido) y **las 12
> pantallas con Playwright sin un solo error de consola** (`scripts\verify_ui.py`). README con
> comandos **ejecutados** (`fv-oat`, `fv-study` verificados de punta a punta).
>
> ⚠ **Consecuencia de §13 (deuda de pesos = 0):** el builder paramétrico renombró los módulos
> conv (`center_conv1` → `center_convs.0`), así que **los checkpoints previos ya no cargan**.
> `load_model` lo rechaza limpio (`checkpoint_incompatible`, 400) en vez de un 500. Los runs de
> ejemplo antiguos (`fov-run-*`, `cli-*`, `dirty-*`, `rec-d`, `demo-seeds`) quedan **no cargables
> por diseño**: reentrénalos o descártalos. Fresco y cargable: run **`fov-16-param`** + recorrido
> inline **`oat-d-demo`** (base `ws16-p2-d2-L2`). Diagnóstico/Predecir usan `fov-16-param`.
>
> ---
>
> **2026-07-21 — IMPLEMENTACIÓN BASE COMPLETA Y VERIFICADA.** El sistema entero está construido y
> probado de punta a punta en esta máquina: paquete `fv` (fovea/datasets/windows/models/
> training/inference/diagnostics/sweeps/studies/validation/metrics/matrixview), API FastAPI, front
> Vite+React con las **diez pantallas**, CLIs (`fv-extract`, `fv-train`, `fv-sweep`, `fv-oat`,
> `fv-study`, `fv-api`) — un test por contrato más el muestreo foveado
> contra los números de la spec. Verificado además: el flujo completo por HTTP (extract →
> train → diagnóstico → predict → sweep), los CLIs (bit-idénticos al API con la misma
> semilla), las negativas con razón+arreglo antes de reservar nombre. El README lleva los comandos
> **ejecutados**, no razonados.
>
> **Decisiones cerradas en la implementación** (registradas en decisiones.md §4): F1=C9
> (cabezas de esquina), **F1b=C10: las esquinas se etiquetan SOLO sobre la fóvea**
> (`center_out == window_size` de B, contrato ①a; la periferia es contexto), C11 (relleno
> `pad_mode: edge` + máscara de cobertura solo para depurar — F0), C12 (anillo por pooling
> anisótropo por zonas, co-registrado: el código §5 de la spec no tipa para d>1, ver
> decisiones.md).
>
> **Datos de ejemplo vivos en el repo**: fuente `local/synth-01` (60 img 96×72, regenerable
> con `scripts\make_synth_source.py`), dataset `synth-b16`, red `fov-16` (migrada a `channels:
> [16,32]`), recetas `corta`/`media`. **Cargables con el builder actual**: run `fov-16-param` y
> recorrido inline `oat-d-demo`. Los runs/recorridos anteriores (`fov-run-*`, `cli-*`, `dirty-*`,
> `rec-d`, `demo-seeds`) conservan su metadata comiteada pero **sus checkpoints no cargan** (§13):
> son historia, no artefactos vivos.
>
> **Pendiente, por orden de valor**: (1) el primer experimento real (protocolo.md §6:
> ¿fóvea+periferia gana a una CNN plana de coste equivalente? — control con `d=1`/`c_frac`→1
> o red plana equivalente, N semillas, criterio escrito antes); (2) el holdout y el dato de
> verdad (fuente del generador reducida con resize — el resize aún NO está portado, decisión
> al llegar); (3) V16/occlusion pre-muestreo (diseño en ui.md); (4) poda en el runner de
> recorridos (hoy corre todos los puntos); (5) pantalla Entrenar: el estimador solo usa runs
> comparables (hecho) pero no hay curva de coste por punto del recorrido.
>
> **Servidores dev**: al cerrar esta sesión quedaron corriendo backend (:8010) y vite (:5173)
> — pararlos o reusarlos.
>
> Nada de lo documentado está construido ni verificado. Cuando un documento cita código
> (`loop.py:166`, `extract.py:127`), habla del **proyecto hermano** — es la evidencia que motivó
> el diseño, no código de este repo.

**Al terminar una fase, actualiza estas líneas.** Es lo único que le dice a la siguiente sesión
dónde está.

---

## Regla permanente: la organización por dominios manda

**[docs/organizacion.md](docs/organizacion.md) es la fuente de verdad sobre cómo se organiza
este sistema. Léelo antes de cualquier cambio y respeta sus fronteras.** Aplica a todo cambio,
por pequeño que parezca — un campo nuevo en un config es exactamente donde las fronteras se
rompen.

Los demás documentos, en orden de lectura:

| | |
|---|---|
| [instructionsNewNN.md](instructionsNewNN.md) | **La red.** Geometría foveada, parámetros, rangos calculados, código de referencia |
| [docs/organizacion.md](docs/organizacion.md) | **La raíz.** Dominios (A–I, X, G) y contratos ①–⑫ donde se tocan |
| [docs/herencia.md](docs/herencia.md) | Qué viene de `image-text-finder`, qué se adapta y qué se descarta |
| [docs/protocolo.md](docs/protocolo.md) | Cuándo un resultado es creíble. **Léelo antes de sacar conclusiones de un entrenamiento** |
| [docs/api.md](docs/api.md) · [docs/ui.md](docs/ui.md) | La organización proyectada sobre HTTP y sobre pantallas |
| [docs/plan.md](docs/plan.md) | El plan de ejecución, por fases verticales |
| [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) | **IMPLEMENTADO (2026-07-24).** Barrido OAT (un eje a la vez) con base derivada del problema, defaults estáticos, arrastre del ganador y estudio (dominio I). Ver `fv.models.derive`, `fv.sweeps.generate/winner`, `fv.studies`, CLIs `fv-oat`/`fv-study` |
| [docs/formatos.md](docs/formatos.md) · [docs/tests.md](docs/tests.md) | Los artefactos en disco; qué se testea |
| [docs/decisiones.md](docs/decisiones.md) | Lo que sigue sin decidir, y qué bloquea. **No tomes tú una decisión que esté ahí: pregunta** |
| [docs/glosario.md](docs/glosario.md) | Las palabras que significan dos cosas |

Reglas que estos documentos fijan y que se citan aquí porque se incumplen solas:

- **Ausente ≠ cero** (formatos.md §2): un lector que necesita un campo ausente **falla con la
  razón**; nunca lo inventa ni lo rellena.
- **Toda restricción se valida antes, con razón y arreglo** (api.md R4): un `400` al entrar vale
  mil veces más que un stack trace dentro del hilo del job media hora después.
- **Toda puerta que entrene pregunta al mismo validador** antes de reservar el nombre. Dos
  comprobaciones separadas se desincronizan, y la puerta más laxa es por la que entra un
  recorrido automático.
- **Un run no se sobrescribe jamás** (409 con la razón).
- **Un contrato sin test es un comentario** (tests.md): los contratos van a
  `tests/test_contracts.py`, los no implementados en `xfail(strict=True)`.
- **Un resultado sin N semillas es una anécdota** (protocolo.md).

### Los dominios (resumen; el detalle está en organizacion.md)

| | Dominio | Es | Vive en |
|---|---|---|---|
| **A** | Fuente | Imágenes + geometría de párrafos (proyecto externo, solo-lectura) | `src/fv/datasets/` |
| **B** | Dataset de ventanas | Lo que se etiqueta: imágenes completas + etiquetas por ventana. **La vista foveada NO se hornea aquí: se construye en el dataloader** | `src/fv/windows/`, `data/window-datasets/` |
| **C** | Red foveada | `N`, fracciones, kernels/strides por rama, fusión. Config puro, cero datos | `src/fv/models/`, `configs/networks/` |
| **D** | Receta | Hiperparámetros de entrenamiento que definen el resultado | `src/fv/training/`, `configs/recipes/` |
| **E** | Run | Modelo entrenado: pesos + métricas + procedencia | `runs/<name>/` |
| **H** | Recorrido | Un espacio sobre **C y/o D** con B fijo → muchos E, sin intervención humana | `src/fv/sweeps/`, `sweeps/` |
| **I** | Estudio (OAT) | Un plan ordenado de ejes sobre **H** con B fijo → muchos recorridos; guía, **no ejecuta** | `src/fv/studies/`, `studies/` |
| **F** | Inferencia | Aplicar un E a una imagen completa (ventana foveada deslizante) | `src/fv/inference/` |
| **G** | Geometría foveada | `derive_dims`, `build_foveated_input`, `build_masks`, rangos calculados. **Un solo módulo, todos lo importan** | `src/fv/fovea/` |
| **X** | Ejecución | `device`, `num_workers`, concurrencia. **Cuesta tiempo, no cambia el resultado** | `src/fv/api/jobs.py` |

### Antes de tocar nada, pregúntate a qué dominio pertenece

El criterio, en orden:

1. ¿Cambia **la forma del modelo o de su entrada**? → **C** (`N`, `c_frac`, `d`, `pen_frac`,
   kernels, strides, `merge`, `pool_mode`, `dropout` son C — *incluida la geometría del muestreo
   foveado*, aunque suene a datos: es la red quien define qué vista consume).
2. ¿Cambia **los pesos resultantes** sin cambiar la forma? → **D** (`lr`, `batch_size`, pesos de
   la pérdida).
3. ¿Solo cambia **cuánto tarda**? → **X**. Nunca dentro de la identidad de D. **`batch_size` es
   D, no X** — subirlo al pasar a GPU invalida la comparación con lo entrenado en CPU (contrato ⑩).
4. ¿Se ajusta **sin reentrenar**, sobre un modelo ya hecho? → **F** (`threshold`, stride de
   inferencia, NMS). Barrer esto no cuesta horas; no lo metas en D.

Si un cambio necesita tocar dos dominios, eso es un **contrato**: está numerado en
organizacion.md §2. Respétalo explícitamente o actualiza el doc.

---

## Contexto de trabajo

- **Hoy solo CPU (esta máquina). Habrá un server con GPU** para los recorridos largos. Por eso X
  está separado de D **desde el diseño**: si no, lo entrenado en CPU queda incomparable con lo
  de GPU. Y por eso `environment` (python/torch/plataforma/device) va en la procedencia de cada
  run.
- **El flujo objetivo son recorridos secuenciales desatendidos**: una receta de recorrido (H)
  nombra el espacio y el presupuesto, se lanza, y corre puntos de uno en uno guardando runs de
  primera clase. Primero versiones cortas aquí (pocas épocas, dataset pequeño) para validar el
  instrumento; el mismo spec, con más presupuesto, en la GPU.
- En CPU, **el límite de workers concurrentes es 1**: torch ya usa todos los núcleos. En GPU se
  reevalúa (es X: no cambia resultados).
- **El espacio de geometría foveada es pequeño y discreto por construcción** (los rangos los
  calculan las funciones de instructionsNewNN.md §3: con N=20, ~3·2·2·1·varios puntos) →
  **grid exhaustivo**. Optuna se reserva para lo continuo: `lr`, canales, dropout
  (instructionsNewNN.md §9).

## Convenciones

- **Idioma**: el usuario se comunica en español; documentación de alto nivel en español. El
  código (identificadores, docstrings) en inglés.
- **Commits**: cada tarea terminada acaba en un commit descriptivo. Además, **cada cambio
  solicitado por el usuario, una vez completado, se cierra con su propio commit descriptivo.**
- **Stack**: Python 3.12 (PyTorch no tiene wheels para 3.14) + PyTorch + FastAPI + Vite/React.
  En Windows el intérprete será `.\.venv\Scripts\python.exe`. Paquete `fv`, layout `src/`.
- **Tests**: `.\.venv\Scripts\python -m pytest -q` desde la raíz, antes de commitear código.
- **README verificado**: antes de decir que un comando documentado funciona, **ejecútalo** en
  PowerShell tal como está escrito (regla global del usuario). Nunca presentar una instrucción
  no probada como verificada.
- `data/`, `runs/` y `sweeps/` son artefactos: **se versiona la descripción (configs, métricas,
  manifests, specs), se ignora la carga (`.npz`, `.pt`, `optuna.db`)** — formatos.md §5.
- **Enlaces a ficheros en las respuestas**: siempre en formato markdown `[texto](ruta)` con la
  ruta **relativa a la raíz del workspace** (nunca backticks ni ruta pelada), para que sean
  clickeables en la extensión de VSCode. **No envuelvas el enlace entre paréntesis** ni pegues
  puntuación al `)` de cierre: `(... [x](ruta) ...)` rompe la detección del enlace y deja de ser
  clickeable. Déjalo suelto o sepáralo con `—`, dos puntos, o una coma con espacio.
  **Los enlaces solo abren ficheros de texto (código fuente), no imágenes** — verificado
  2026-07-23: un `.png` no abre al clicar aunque esté rastreado por git (no es el git-ignore, es
  el tipo binario). Para una imagen (capturas de `data/ui-shots/`, etc.) NO ofrezcas un link
  markdown que no abre: da la ruta para `Ctrl+P`/Go-to-File, o muéstrala inline con la tool Read.

## Observaciones de esta máquina (medidas en el proyecto hermano — no re-aprenderlas)

- **La máquina HIBERNA en entrenamientos nocturnos largos**: suspende el proceso. Para un
  recorrido desatendido, desactivar la suspensión (`powercfg`) o contar con que se pausa.
- **Throttling térmico**: en carga sostenida los runs se ralentizan ~5×. Los presupuestos de un
  recorrido nocturno deben contarlo.
- **La consola de Windows es cp1252**: los CLIs imprimen ASCII (un `→` en un `--help` revienta
  con `UnicodeEncodeError`).
- **Los JSON de estado se escriben con temporal + `os.replace`, y en Windows con reintento en
  los dos lados** (escritor y lector): Windows no reemplaza un fichero con un handle abierto.
  Detalle en formatos.md §4.2.
- **Hay Playwright y Chromium en esta máquina: la UI SE PUEDE ver.** Los navegadores están en
  `%LOCALAPPDATA%\ms-playwright\`; hace falta `pip install playwright` en el venv del proyecto
  (los navegadores ya están, no hace falta `playwright install`). No entregar UI diciendo «no
  puedo verlo» sin haber mirado.
- **Al verificar UI: reinicia el backend** — un server stale da 404 engañosos sobre rutas nuevas.

## Trampas heredadas: no las reproduzcas

Medidas en `image-text-finder` (lista completa y razonada en
[docs/herencia.md](docs/herencia.md) §4 y organizacion.md §3). **Casi todas eran *defaults***:
nadie las eligió, aparecieron por no elegir. Construir desde cero no protege de ellas — las
invita:

- **SGD sin momentum** si solo pasas `lr` y `weight_decay` → cualquier comparación de
  optimizadores queda sesgada a favor de Adam.
- **Un hilo por job sin límite** → un recorrido de 20 puntos son 20 entrenamientos peleándose
  por los mismos núcleos. En CPU el límite es 1.
- **Sobrescritura silenciosa de runs** (`mkdir(exist_ok=True)` + truncar métricas) — quien la
  pisa es justo un recorrido que autogenera nombres.
- **Un dataset sin val** elige `best.pt` por train loss sin avisar → se niega, no se degrada.
- **Estado de run deducido del disco** → un crash queda «running» para siempre. `status.json`
  explícito.
- **Augmentation con flips/rotaciones sin reetiquetar** → enseña basura en silencio (las
  etiquetas de posición/región son direccionales).
- **Definir un número dos veces** (una métrica calculada en dos sitios) → módulo único
  `fv.metrics`, y un test que afirma la costura, no la función.
- **Lógica de dominio dentro de `app.py`** → si una función no menciona HTTP, no es del API.
- **Medir con un val diminuto**: los ejemplos de una misma imagen están correlacionados; el
  tamaño de muestra efectivo lo dan las **imágenes**, no las ventanas. El dato es sintético:
  generar más es gratis.
- **Optimizar un proxy sin validarlo**: la métrica que manda es la de la tarea real (párrafo
  bien reconocido por imagen), no la de ventana — protocolo.md §2.
