# Plan de ejecución

> **Estado (2026-07-21): las fases 0–7 están construidas y verificadas** en una sola pasada de
> implementación (ver CLAUDE.md — estado): 31 tests en verde, flujo completo por HTTP y CLI,
> 11 pantallas verificadas con Playwright. Los xfails no llegaron a existir: los contratos
> nacieron implementados con su test en verde. Queda la **fase 8 (GPU)** y la investigación
> (protocolo.md §6). Diferencias deliberadas respecto al plan original: sin optuna (runner
> propio grid/random secuencial y reanudable; optuna entra si hace falta TPE/poda), sin poda
> todavía, resize de fuentes no portado aún, matrixview copiado (no extraído a claude-libs —
> D2 sigue abierta).

Fases **verticales** (backend + front por dominio), heredando el método del hermano: cada fase
acaba con la app arrancando, los tests pasando (`pytest -q` en verde), **sus xfails quitados**
(tests.md §1) y un commit. El README se actualiza con los comandos **ejecutados y verificados**.

El objetivo de negocio manda sobre el orden: **llegar cuanto antes a poder lanzar series de
runs cortas y verificables en CPU**, porque ese instrumento es lo que luego se replica en la
GPU con más presupuesto.

---

## Fase 0 — decisiones y esqueleto de specs

- **F1 ya está cerrada** (C9: cabezas de esquina — la tarea es la del hermano). Cerrar **F1b**
  (ventana etiquetada: ¿centro o campo completo?) y **F2** (fuente y resolución) de
  [decisiones.md](decisiones.md) §1, con el usuario — no se deciden solas.
- Completar en los docs los huecos que dependan de ellas (`label_window`, normalización de
  `x, y`, endpoints de diagnóstico).

## Fase 0.5 — los contratos, en xfail

- `tests/test_contracts.py`: los ~11 contratos de tests.md §3, todos `xfail(strict=True)` con
  su `reason` citando documento y fase. Ninguno puede pasar (no hay `src/`) — definen el destino.

## Fase 1 — esqueleto y paleta

- `pyproject.toml` (paquete `fv`, layout `src/`, extras `train`/`api`/`sweep`/`dev`), venv 3.12,
  scripts CLI declarados.
- Front Vite+React con el shell de pantallas vacías, `tokens.css` y `validate:palette` portados.
- Decidir **D2** para `matrixview` (extraer vs copiar) — es la primera pieza que el front pide.
- README con montar/correr **verificados**.

## Fase 2 — G y A: la geometría y las fuentes

- **`fv.fovea` completo**: `derive_dims` (+ asserts, contrato ②), `build_foveated_input`,
  `build_masks`, rangos calculados. Con los tests numéricos de instructionsNewNN.md §3–§4 como
  oráculo (quita los xfail de ②, ②b, y los tests de muestreo).
- `fv.datasets`: lector de A, índice de offsets, resize (portados del hermano, adaptando
  imports). Allowlist de raíces + CORS (C7).
- Pantallas **Fuentes** y el comparador **FG3** si es barato (solo necesita `fovea` + A).
- Quita xfail: ②, ②b, ⑦ (la tabla de imports ya se puede afirmar), ⑧ parcial (huella).

## Fase 3 — B, C, D: los sustantivos

- `fv.windows`: extractor (ventanas etiquetadas según F1 + `images`), manifest con huella,
  dataloader **perezoso y vectorizado** (la vista por ítem, contrato ⑤ lado B), CLI
  `fv-extract`.
- `fv.models`: `NetworkConfig` → `FoveatedRegionalNN` (instructionsNewNN.md §6), store YAML,
  `validate` con dimensiones derivadas y rangos.
- `fv.training.recipe`: el catálogo de D con defaults puestos a propósito (momentum, scheduler).
- `fv.validation.check_run` — la función pura que todas las puertas llamarán.
- Pantallas **Ventanas** (con el detalle del dato crudo), **Redes** (con FG1, el diagrama de
  zonas en vivo) y **Recetas**.
- Quita xfail: ①(validador), ⑤(dataloader), ⑩.

## Fase 4 — E: entrenar

- `fv.training.loop`: bucle con val obligatorio, checkpoints, `metrics.jsonl`, escritura
  atómica con reintento (Windows), parada cooperativa, procedencia completa (nombre+valor+huella
  +commit+environment). `fv-train` y `POST /runs` por la **misma puerta**. Cola de jobs
  (límite 1, cancelación, persistencia) — decidir D2 para la cola.
- Pantallas **Entrenar** (con estimación de coste honesta) y **Runs** (lista + detalle, V14).
- Quita xfail: ①(HTTP), ③, ④, ⑪(reproducibilidad).
- **Hito**: el flujo dato → red → receta → run funciona de punta a punta, por CLI y por UI.

## Fase 5 — E×B: diagnóstico

- `fv.metrics` (un número, un sitio) + la tabla por ventana como **caché** (clave con mtime del
  checkpoint) + agregados en servidor.
- Pantalla **Diagnóstico**: galería peor-primero, V7, V8 y **F0 (vista de entrada canal a canal
  con máscaras)** — la vista de depuración fundamental.

## Fase 6 — F: inferencia e introspección

- `fv.inference`: `load_model` (④), predict por imagen con todas las etapas, la vista foveada
  desde `fv.fovea` (quita el xfail de ⑤ lado F), ModelCache con mtime.
- V1/V2 **por rama**, V16, FG2 (contribución por rama). Pantalla **Predecir** con knobs en
  unidades de ventana.
- **Aquí se corre el paso 2 del protocolo** (validar el proxy) y el paso 1 (suelo de ruido) en
  versión corta.

## Fase 7 — H: recorridos

- `fv.sweeps`: spec (con `"auto"` → `build_search_space`, validación ⑨/⑨-ext pura), store,
  runner secuencial con optuna, poda, reanudación desde disco, descarte declarado de puntos
  geométricamente inválidos. `fv-sweep` CLI — **el recorrido debe poder lanzarse sin la UI ni
  el API**, porque en el server puede no haber navegador.
- Pantalla **Recorridos** con V12/V13.
- Quita xfail: ⑨.
- **Hito**: una receta de recorrido corta corre desatendida en esta máquina de principio a fin,
  sobrevive a un reinicio, y cada punto es un run con procedencia verificable en la UI.

## Fase 8 — GPU

- Probar el paquete en el server: `environment` con `device: cuda`, `num_workers` ajustado (X),
  **`batch_size` intacto** (⑩). Verificar reproducibilidad CPU↔GPU *documentando* la diferencia
  esperable (misma semilla en GPU no garantiza bit-exactitud — se declara, no se esconde).
- Lanzar el primer recorrido real (presupuesto grande) con el spec ya validado en CPU.

## Después: investigación

Con el instrumento montado, la cola de análisis vive en [protocolo.md](protocolo.md) §6. El
primer experimento ya está fijado: **¿la fóvea + periferia gana a una CNN plana de coste
equivalente?** — control y tratamiento sobre el mismo B, N semillas, criterio escrito antes de
mirar (el análogo del P4 del hermano, que es la evidencia de que hay algo que ganar).
