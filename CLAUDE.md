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

> **2026-07-21 — IMPLEMENTACIÓN COMPLETA Y VERIFICADA.** El sistema entero está construido y
> probado de punta a punta en esta máquina: paquete `fv` (fovea/datasets/windows/models/
> training/inference/diagnostics/sweeps/validation/metrics/matrixview), API FastAPI, front
> Vite+React con las **nueve pantallas**, CLIs (`fv-extract`, `fv-train`, `fv-sweep`,
> `fv-api`) y **31 tests en verde** (~14 s) — un test por contrato más el muestreo foveado
> contra los números de la spec. Verificado además: el flujo completo por HTTP (extract →
> train → diagnóstico → predict → sweep), los CLIs (bit-idénticos al API con la misma
> semilla), las negativas con razón+arreglo antes de reservar nombre, y **las 11
> pantallas/interacciones con Playwright sin un solo error de consola**
> (`scripts\verify_ui.py`; capturas en `data\ui-shots\`). El README lleva los comandos
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
> con `scripts\make_synth_source.py`), dataset `synth-b16`, red `fov-16`, recetas
> `corta`/`media`, runs `fov-run-1/2`, `cli-run-1`, recorridos `rec-d`, `cli-sweep-1`.
> En `fov-run-2` (12 épocas) ya se ve el fenómeno que la fóvea ataca: err ciega 4,06 px vs
> visible 2,55 px.
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
| [docs/organizacion.md](docs/organizacion.md) | **La raíz.** Dominios (A–H, X, G) y contratos ①–⑪ donde se tocan |
| [docs/herencia.md](docs/herencia.md) | Qué viene de `image-text-finder`, qué se adapta y qué se descarta |
| [docs/protocolo.md](docs/protocolo.md) | Cuándo un resultado es creíble. **Léelo antes de sacar conclusiones de un entrenamiento** |
| [docs/api.md](docs/api.md) · [docs/ui.md](docs/ui.md) | La organización proyectada sobre HTTP y sobre pantallas |
| [docs/plan.md](docs/plan.md) | El plan de ejecución, por fases verticales |
| [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) | **DISEÑO, sin implementar.** Barrido OAT (un eje a la vez) con base derivada del problema, defaults estáticos, arrastre del ganador y schedule. El código se genera en otra sesión |
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
