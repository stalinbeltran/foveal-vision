# foveal-vision

Detección de esquinas de párrafo con una **red foveada de dos ramas** (centro a resolución
completa, periferia reducida con mayor campo visual) y **recorridos automáticos** que barren
las configuraciones de la red además de las recetas. El mismo problema que
[`image-text-finder`](../image-text-finder), con otra red — ver
[docs/herencia.md](docs/herencia.md).

La especificación de la red es [instructionsNewNN.md](instructionsNewNN.md); el diseño del
sistema vive en [docs/](docs/) y las instrucciones para Claude en [CLAUDE.md](CLAUDE.md).

**Todos los comandos de este README se ejecutaron y verificaron** en Windows 11 con PowerShell,
desde la raíz del repo (base 2026-07-21; el barrido por ejes / estudios OAT, 2026-07-24).

## Requisitos

- **Python 3.12** (PyTorch no tiene wheels para 3.14; verificado con 3.12.10, `py -3.12`).
- **Node.js 18+** con npm (verificado con Node 24 / npm 11).

## Montar

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,api,dev]" playwright
cd web; npm install; cd ..
```

`playwright` es opcional (solo `scripts\verify_ui.py`); los navegadores ya están en
`%LOCALAPPDATA%\ms-playwright\` en esta máquina — **no** hace falta `playwright install`.

## Datos

Las fuentes (A) se buscan en dos raíces: `FV_DATASETS_ROOT` (por defecto,
`..\image-text-sample-generator\data\datasets` si existe) y `data\sources\` (locales, con
prefijo `local/`). Para arrancar sin el generador, hay un generador sintético:

```powershell
.\.venv\Scripts\python.exe scripts\make_synth_source.py --name synth-01 --count 60
```

> Verificado: 60 imágenes de 96×72 en `data\sources\synth-01`. Si el nombre ya existe, se
> niega (exit 2): nada se sobrescribe en silencio.

## Construir un dataset de ventanas (B)

```powershell
.\.venv\Scripts\fv-extract.exe --source local/synth-01 --name synth-b16 --window-size 16 --stride 8
```

> Verificado: 5280 ventanas de 60 imágenes, splits 3696/792/792, positivos ~430 por esquina.
> `window_size` es **la ventana etiquetada = la fóvea de la red** (F1b). B guarda las imágenes
> completas: la vista foveada se construye en el dataloader, así que la geometría (`N`, `d`,
> `c_frac`…) se barre **sin re-extraer**.
>
> Sobre una fuente grande se niega antes de escribir nada — verificado contra la fuente real
> del generador (20 000 × 640×480): `[images_budget_exceeded] guardar las imagenes costaria
> 6.14 GB (> 1 GB)`.

## Correr la app

Dos procesos, puertos explícitos (el **8010** evita chocar con el 8000 del proyecto hermano;
el **5173** está fijado con `strictPort` y en la allowlist de CORS del backend):

```powershell
# terminal 1 — backend
.\.venv\Scripts\python.exe -m fv.api --host 127.0.0.1 --port 8010
```

```powershell
# terminal 2 — front (proxya /api al backend)
cd web
npm run dev          # http://localhost:5173
```

Funcionan las diez pantallas: **Fuentes, Ventanas, Redes, Recetas, Entrenar, Recorridos,
Estudios, Runs, Diagnóstico y Predecir**. Las redes y recetas se crean desde la UI (Redes
valida en vivo: dimensiones derivadas, rangos calculados y el diagrama de zonas; la red se
edita por `n_layers` + `channels` por capa). **Estudios** encadena barridos por ejes (OAT):
deriva la base del problema, arrastra el ganador y guía paso a paso.

Al seleccionar un **Recorrido** se superponen las curvas de val (loss / f1 / pos_err_px) de
**todos sus runs** en tres small-multiples: una línea de color por run, con leyenda de casillas
para ocultar/mostrar cada uno (o todos), y un conmutador **líneas por run ↔ media ± banda**
(la banda agrupa los runs que comparten config salvo la semilla; con una sola semilla queda
degenerada, a propósito). Útil para ver el espacio entrenar y cazar el punto que diverge.

## Entrenar sin la UI

Hacen falta tres cosas **con nombre**: un dataset (B), una red (C) y una receta (D). Se crean
desde la UI o dejando YAMLs en `configs\networks\` y `configs\recipes\`.

```powershell
.\.venv\Scripts\fv-train.exe --name cli-run-1 --window-dataset synth-b16 --network fov-16 --recipe corta --device cpu
```

> Verificado (red `fov-16`: N=20, c_frac=0.8, d=2; receta `corta`: 3 épocas): ~4 s/época,
> `val_loss` 0.38→0.30, f1 0.22 en la época 3. **Y bit-idéntico al mismo run lanzado por el
> API** (misma semilla ⇒ mismos números hasta el último decimal): dos puertas, un resultado.
>
> Las negativas llegan **antes de reservar el nombre**, con razón y arreglo — verificado:
> `[network_not_found] ... -> las redes disponibles son: fov-16` (exit 2, sin `runs\x\` a
> medias).

## Recorridos (sweeps) sin la UI

La "receta de recorrido" es un YAML; `d: auto` usa el **rango calculado** por la geometría:

```powershell
.\.venv\Scripts\fv-sweep.exe --name cli-sweep-1 --spec sweep-example.yaml
```

> Verificado: 3 puntos (grid `d × lr`), 0 descartados, corre secuencial, y al final imprime el
> ranking por el objetivo. Reanudar tras un corte: el mismo comando **sin `--spec`**. El estado
> vive en disco (`sweeps\<name>\`), así que sobrevive a reinicios e hibernaciones.
>
> Pensado para el server con GPU: el CLI no necesita ni el API ni un navegador. El mismo spec
> validado corto en CPU se lanza allí con más presupuesto (`--device cuda`).

## Barrido por ejes (OAT): generar la red, no escribirla

En vez de teclear a mano los ~14 campos de una red, se **derivan del problema**: del
`window_size` del dataset sale `N` y la geometría (contrato ①a), y el generador barre **un solo
eje**. El único ingreso manual es dataset + eje + rango (diseño en
[docs/barrido-por-ejes.md](docs/barrido-por-ejes.md)).

```powershell
.\.venv\Scripts\fv-oat.exe --name mi-oat --window-dataset synth-b16 --axis k_center --range auto --recipe corta --epochs 1
```

> Verificado (2026-07-24): base inline `ws16-p2-d2-L2` derivada de la ventana de 16px, eje
> `k_center` con su **rango calculado** `[3, 5, 7]` → 3 puntos válidos, 0 descartados, corre
> secuencial e imprime el ranking. `--axis n_layers --range "[1,2,3]"` redimensiona `channels`
> a `[16]*L` en cada punto (§6.1); un eje inválido para la geometría cae al válido con su razón.

`--seeds N` entrena **cada valor del eje con N semillas** (añade un eje `seed` réplica): el
barrido pasa a `len(rango)·N` runs (`…-d2_seed1 .. _seedN`) y el ganador se rankea sobre la
**media** por valor, no sobre la réplica con suerte (recipe.py: seed es el eje réplica). Sin la
bandera, `seeds=1` = sondeo de una semilla. En un **estudio**, el campo `seeds:` del plan hace lo
mismo en cada paso.

> Verificado (2026-07-25): `fv-oat --seeds 2` sobre `synth-b16` → 2 runs con semillas 1 y 2
> distintas en disco; `GET /sweeps/{n}/winner` agrega los dos en una entrada por valor de eje
> (media + banda + `n_seeds`).

### Cómo se lee un ranking (y cuándo NO hay ganador)

Dos reglas que cambian lo que significa la tabla:

- **El valor de un punto es el de su checkpoint**, no el de la última época. `best.pt` se elige
  por `monitor` y es lo que cargan Diagnóstico, Predecir y el arrastre de un estudio; el ranking
  mide **ese** fichero. La columna «época» dice de dónde sale cada número, y la UI avisa cuando
  `monitor` y `objective` no son el mismo (entonces el ranking mide checkpoints seleccionados por
  otro criterio — legal, pero hay que saberlo).
- **Un ganador dentro del ruido es un empate.** `δ` por defecto = el error estándar de la media
  del mejor punto entre sus semillas (regla 1-SE); todo lo que caiga dentro es indistinguible y
  se declara **empate técnico**, nombrando la frontera. `--delta` (CLI) o la caja δ (UI) lo fijan
  a mano; vacío = medido. Con una sola semilla no hay banda: `δ=0` **y se dice por qué**.

> Verificado (2026-07-26) sobre los recorridos reales del repo: `batch_size` gana de verdad
> (batch 100 despega de δ=0.0133); `fast-lr-s0-lr` empata sus dos primeros; `fast-lr-2-s0-lr`
> empata **los seis** puntos — 30 runs que no distinguen nada. Y rankear por checkpoint en vez de
> por última época **cambia el ganador** de ese recorrido (lr 0.0014 → 0.00168).

Un **estudio** encadena varios ejes (descenso por coordenadas): fija el ganador de cada paso
como base del siguiente y **expande sub-ejes** (`channels[i]` al fijar `n_layers`). El plan es un
YAML comiteable; `--auto` recorre la cadena confirmando el ganador sugerido (regla coste/calidad):

```powershell
.\.venv\Scripts\fv-study.exe --name mi-estudio --plan estudio-example.yaml --auto
```

> `--delta` es opcional: sin él, el margen se **mide** de la dispersión entre semillas de cada
> paso (1-SE). Pasarlo (`--delta 0.02`) lo fija a mano en toda la cadena.

> Verificado (2026-07-24): con un plan `n_layers → channels[i]`, el ganador `n_layers=1` encogió
> la base a `ws16-p2-d2-L1` y expandió `channels[i]` a un solo paso `channels[0]` — cadena
> completa desatendida. El estudio **guía y no ejecuta** por diseño: desde la web app el ganador
> lo confirma el usuario (pantalla **Estudios**); `--auto` es para la validación corta en CPU.

### La métrica de tarea (párrafo por imagen)

El ranking de un recorrido se mide **por ventana** (`f1`, `pos_err_px`): es barata, se calcula
por época y —medido— **ordena igual** que la cara en ejes de receta (Spearman +0,956 agregado,
[docs/metrica-de-tarea.md](docs/metrica-de-tarea.md) §2). La métrica que **manda** es otra: si el
párrafo se reconoce bien **en la imagen completa**. Se pide a parte, cuando hace falta, porque
cuesta inferencia de imagen completa (~0,6 s por run con 20 imágenes de val):

```powershell
Invoke-RestMethod "http://localhost:8010/runs/fov-16-param/task-score?split=val" | ConvertTo-Json -Depth 3
```

> Verificado (2026-07-26): devuelve `macro` (**la primaria**: media por imagen, con `sd` y
> **`sem`**), `micro` (tp/fp/fn sumados), `mean_iou` (`null` si no hubo emparejamientos, nunca 0),
> `per_image`, los knobs de F resueltos y `cached`. Se puntúa `best.pt` contra los párrafos de la
> **fuente** (contrato ⑬): sin la fuente falla con `task_needs_source` en vez de puntuar contra
> las etiquetas de ventana, que miden otra cosa. La segunda llamada sale de caché; cambiar
> cualquier knob (`threshold`, `stride`, `nms_radius`, `min_size`, `iou_threshold`) re-infiere.
>
> Reproducido el número de la Fase 1 con el código nuevo: las 5 semillas del ganador de
> `fast-lr-s0-lr` (`lr=0,00215443`) dan **0,5353** de media — el mismo valor de la tabla de
> metrica-de-tarea.md §2, en 1,9 s.

En la web app: bloque **«Métrica de tarea (párrafo por imagen)»** en el detalle de un run, y
botón **«Medir la tarea del ganador sugerido»** en el veredicto de Recorridos (solo el sugerido y
el mejor, nunca los 35 puntos). En los CLIs, detrás de una bandera **apagada por defecto** para
que un recorrido nocturno no pague inferencia que nadie pidió:

```powershell
.\.venv\Scripts\fv-oat.exe --name tarea-demo --window-dataset synth-b16 --axis lr --range "[0.001,0.002]" --recipe corta --epochs 1 --task-score
.\.venv\Scripts\fv-study.exe --name tarea-estudio --plan estudio-example.yaml --auto --task-score
```

> Verificado (2026-07-26) de punta a punta, también con `PYTHONIOENCODING=cp1252`: los dos
> imprimen la métrica del **ganador sugerido** (una línea por semilla + la media), con el aviso
> explícito de tamaño de muestra. Los artefactos de la demo se borraron después.

**Con 20 imágenes de val el error estándar es ±0,093**, y las diferencias entre puntos vecinos de
un recorrido son de 0,01 a 0,05: hoy este número sirve para **informar del ganador**, no para
decidir entre puntos. La UI y los CLIs lo dicen en pantalla siempre que n < 100 — y **el número
que imprimen sale siempre del `sem` del propio payload**, nunca de esa cifra escrita. Subirlo es
regenerar el dato (§4 del doc), que **cuesta la comparabilidad con lo entrenado hasta hoy** y por
eso está esperando decisión (F11 en [docs/decisiones.md](docs/decisiones.md)).

> La sd por imagen se **re-midió el 2026-07-26** sobre los 20 runs ganadores de los 4 recorridos
> (§9.4 del doc): **0,4148**, no el 0,372 con el que se hizo la primera aritmética. Contra lo que el
> documento suponía, **sube** — la sd es máxima con modelos intermedios (F1 por imagen casi bimodal),
> y el 0,372 se había promediado incluyendo modelos de F1 0,10. Hacen falta *más* imágenes, no menos.

#### Puntuar contra un holdout

El val hace **dos trabajos** —elegir `best.pt` y rankear—, así que el número del val del ganador
está **sesgado al alza** y no se reporta como resultado. Para eso está el holdout: otro dataset de
ventanas (B) extraído de una **fuente propia de la que jamás se entrena**. El camino está cableado
de punta a punta; **lo que falta es la fuente**, que depende de F11/F13.

```powershell
# el holdout es "todo test": asi no se puede usar para entrenar, ni por accidente
.\.venv\Scripts\fv-extract.exe --source local/<fuente>-holdout --name <fuente>-holdout-b16 `
  --window-size 16 --stride 8 --val-frac 0 --test-frac 1 --seed 1

Invoke-RestMethod ("http://localhost:8010/runs/<ganador>/task-score" +
  "?window_dataset=<fuente>-holdout-b16&split=test") | ConvertTo-Json -Depth 3
```

En la web app, el bloque de tarea del **detalle de un run** trae dos selectores (**dataset** y
**split**); al elegir un dataset distinto del propio aparece el aviso de que **el holdout se toca
una sola vez, al final y solo con el ganador**.

> **Convenio, mientras no exista una marca en disco**: la única señal de que una fuente es holdout
> es que su nombre acabe en **`-holdout`**. Un campo `"holdout": true` en su `dataset.json` sería
> más robusto y está propuesto; hoy no existe (F14 en [docs/decisiones.md](docs/decisiones.md)
> cubre además si se **registra** que el holdout se miró — hoy nada lo recuerda, y la caché hace
> que la segunda mirada sea gratis e invisible).
>
> Lo que el código **sí** garantiza hoy, con test: un B que salga de **la misma fuente** que el de
> entrenamiento se rechaza con `holdout_shares_source` (no sería un holdout, sería entrenamiento con
> otro nombre); un B 100 % test **no puede entrenar** (`no_validation_split`); y la huella del
> dataset del run **no bloquea** el número del holdout, porque esa huella protege el split del run,
> no el del holdout.

### ¿El proxy de ventana ordena igual que la tarea?

La pregunta que decide si el ranking barato vale. Se contesta sobre un recorrido **ya
entrenado**, sin reentrenar nada:

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep fast-lr-s0-lr --split val
```

> Verificado (2026-07-26): reproduce **exactamente** la tabla de la Fase 1 sobre los 65 runs de
> `fast-lr-s0-lr` — Spearman **+0,737** por run (0,7368) y **+0,956** agregado por valor del eje,
> ganador `lr=0,00215443` con las dos métricas, en 40,6 s la primera vez y 0,6 s con la caché
> llena. Imprime también la frontera δ y dice si el ganador por tarea cae dentro; `--json` guarda
> el detalle por run y por punto.
>
> El veredicto **nombra el eje y su dominio** (`eje(s): lr — dominio: D (la receta: no cambia la
> vista)`): la pregunta de la Fase 3b es precisamente C-contra-D, así que un «vale para C» tras
> medir `lr` sería una mentira con forma de resultado.
>
> El script **no calcula ninguna métrica por su cuenta**: la de ventana sale de `sweep_trials` (la
> época que guardó `best.pt`), la de tarea de `fv.task.task_score`, la correlación de
> `fv.metrics.spearman` y la frontera de `suggest_winner`. Un run sin checkpoint se **descuenta
> diciéndolo**, nunca se rellena con un cero. Códigos de salida: `0` el proxy vale, `1` no vale,
> `2` no concluyente (el recorrido no distingue nada).

El criterio de aceptación está escrito **antes** de mirar (protocolo.md §1), como constantes del
script: Spearman agregado ≥ 0,90 **y** el ganador por tarea dentro de la frontera δ.

`--objective` re-lee **los mismos runs terminados** con otro proxy de ventana, sin tocar el spec ni
reentrenar, para preguntar cuál sigue mejor a la tarea:

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep fast-lr-s0-lr --objective pos_err_px
```

> Verificado (2026-07-26) sobre `fast-lr-s0-lr`, Spearman agregado: **`f1` +0,956** · `loss` +0,780 ·
> `pos_err_px` +0,544. Los dos últimos además **eligen otro ganador**, y fuera de la banda de ruido
> del bueno. El default `f1` era el correcto — ahora con evidencia, no por costumbre. La re-lectura
> se declara en la salida (`objective_overridden`) para que nunca se confunda con lo que el
> recorrido optimizó de verdad, y un objetivo desconocido se rechaza en la puerta.

### Qué ejes se pueden barrer

Cualquier campo de la **red (C)** o de la **receta (D)** es un eje válido — **excepto `N` y
`c_frac`**. Esos dos fijan `center_out = round_to_even(N·c_frac)`, que el contrato ①a ata al
`window_size` del dataset (fijo en todo el recorrido): barrerlos daría una fóvea distinta de la
ventana etiquetada en cada punto, así que se **rechazan en las dos puertas** (recorrido y estudio)
con razón y arreglo — para variar el contexto periférico barre `d`, y para cambiar la fóvea usa un
dataset con esa ventana. Para comprobar que **todos** los ejes corren de punta a punta:

```powershell
.\.venv\Scripts\python.exe scripts\verify_axes.py --dataset synth-b16
```

> Verificado (2026-07-24): **26/26 ejes** (11 de red + 13 de receta + `N`/`c_frac` rehusados),
> 0 fallos. Corre un recorrido real por eje (`generate_sweep` + `run_sweep`), entrena sus puntos
> en un store temporal y comprueba que se miden. Ejes de red: `d, pen_frac, n_layers, k_center,
> k_periph, s_center, s_periph, channels, merge, pool_mode, pad_mode`. Ejes de receta: `lr,
> optimizer, momentum, weight_decay, batch_size, epochs, scheduler, patience, lambda_pos,
> pos_weight, smooth_l1_beta, monitor, seed`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

> Verificado (2026-07-26): **107 passed** en ~43 s — un test por contrato (organizacion.md §2,
> con el ⑫ estudio↔recorrido y el nuevo **⑬ métrica de tarea**), el builder paramétrico
> (no-regresión `n_layers=2`), el derivador de base, el generador OAT, el arrastre del ganador, el
> rechazo de `N`/`c_frac` como ejes, que el budget no colapsa el eje `epochs`, las costuras de
> `task_score` (`tests/test_task.py`: caché por knobs, `best.pt` y no `last.pt`, `mean_iou` null,
> holdout que comparte fuente) y el flujo completo por HTTP.

## Verificar la UI con navegador

Con backend y front corriendo (y los datos de arriba creados):

```powershell
.\.venv\Scripts\python.exe scripts\verify_ui.py
```

> Verificado (2026-07-26): recorre las **12 pantallas/interacciones** con Chromium (incluye la
> pantalla Estudios, el clic en la galería de Diagnóstico → sondas, el bloqueo del contrato ⑨ en
> Recorridos, los sliders de Predecir y **el botón de la métrica de tarea** en el detalle de un run
> y en el veredicto de un recorrido, con su aviso de tamaño de muestra), falla ante cualquier error
> de consola, y deja capturas
> en `data\ui-shots\`. Diagnóstico/Predecir usan `fov-16-param` (entrenado con el builder
> paramétrico): los checkpoints anteriores son incompatibles a propósito (barrido §13).

## Por dónde empezar a leer

[CLAUDE.md](CLAUDE.md) abre con el estado y enlaza los documentos en orden:
organización por dominios, protocolo experimental, formatos, API, UI, tests, decisiones,
glosario y plan.
