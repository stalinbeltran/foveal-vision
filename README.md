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

Un **estudio** encadena varios ejes (descenso por coordenadas): fija el ganador de cada paso
como base del siguiente y **expande sub-ejes** (`channels[i]` al fijar `n_layers`). El plan es un
YAML comiteable; `--auto` recorre la cadena confirmando el ganador sugerido (regla coste/calidad):

```powershell
.\.venv\Scripts\fv-study.exe --name mi-estudio --plan estudio-example.yaml --auto --delta 0.02
```

> Verificado (2026-07-24): con un plan `n_layers → channels[i]`, el ganador `n_layers=1` encogió
> la base a `ws16-p2-d2-L1` y expandió `channels[i]` a un solo paso `channels[0]` — cadena
> completa desatendida. El estudio **guía y no ejecuta** por diseño: desde la web app el ganador
> lo confirma el usuario (pantalla **Estudios**); `--auto` es para la validación corta en CPU.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

> Verificado (2026-07-24): **65 passed** en ~25 s — un test por contrato (organizacion.md §2, con
> el nuevo ⑫ estudio↔recorrido), el builder paramétrico (no-regresión `n_layers=2`), el derivador
> de base, el generador OAT, el arrastre del ganador y el flujo completo por HTTP.

## Verificar la UI con navegador

Con backend y front corriendo (y los datos de arriba creados):

```powershell
.\.venv\Scripts\python.exe scripts\verify_ui.py
```

> Verificado (2026-07-24): recorre las **12 pantallas/interacciones** con Chromium (incluye la
> pantalla Estudios, el clic en la galería de Diagnóstico → sondas, el bloqueo del contrato ⑨ en
> Recorridos y los sliders de Predecir), falla ante cualquier error de consola, y deja capturas
> en `data\ui-shots\`. Diagnóstico/Predecir usan `fov-16-param` (entrenado con el builder
> paramétrico): los checkpoints anteriores son incompatibles a propósito (barrido §13).

## Por dónde empezar a leer

[CLAUDE.md](CLAUDE.md) abre con el estado y enlaza los documentos en orden:
organización por dominios, protocolo experimental, formatos, API, UI, tests, decisiones,
glosario y plan.
