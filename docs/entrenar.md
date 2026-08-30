# Entrenar la red foveada: con cuántas épocas quieras, y continuando

**Qué contesta este documento:** con qué parámetros se entrena hoy, cómo se lanza con un
número de épocas dado, cómo se **continúa** un entrenamiento parado, y qué fichero de pesos
sirve para cada cosa.

Los parámetros salen de los estudios; el índice de qué está cerrado y con cuánta evidencia
vive en [`ESTADO.md` del repo central](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/ESTADO.md).
Aquí no se re-argumenta ninguno: se enlaza y se aplica.

---

## 1. Qué red entrenar: hay DOS configuraciones, y no dan lo mismo

| Config | Qué es | Cuándo usarla |
|---|---|---|
| **`fov16-vigente`** | la red sobre la que están medidas **todas las tablas publicadas** (`ws16-p2-d2-L4`) | para **comparar** con lo ya medido |
| **`fov16-optimo`** | el vigente **con los dos ejes que los estudios cerraron al 5 % y nunca se aplicaron** | para entrenar **el mejor modelo** que la evidencia respalda |

Las dos tienen **167.852 parámetros**. La diferencia es qué ve la red:

| | `border_px` | `border_reduce` | `overlap_fovea_px` | recorte real | parámetros |
|---|---:|---:|---:|---:|---:|
| `fov16-vigente` | 4 | 2 | 2 | 24×24 | 167.852 |
| `fov16-optimo` | **8** | **4** | **7** | **32×32** | 167.852 |

Lo que los estudios midieron para justificar el cambio (los dos en el [#14]):

- `border_px` 4 → 8: `p` = **0,006**, Δ = **+0,0096** (10 semillas contra 10), y **1,33× más
  rápido por época**.
- `overlap_fovea_px` 2 → 7: `p` < **0,001**, Δ = **+0,0124**, y el óptimo es la **pared legal**
  (`overlap_fovea_range(16)` = [0..7]) — no está acotado por evidencia sino por la geometría.

Los dos superan la regla del proyecto para mover el vigente (`p` < 0,05 **y** Δ > δ). `ESTADO.md`
los marca «4 → 8» y «2 → 7», **no aplicado**: `fov16-optimo` es aplicarlos.

### ⚠ Tres cosas que hay que saber antes de usar `fov16-optimo`

1. **`border_px` va ATADO a `border_reduce`, y el resumen no lo dice.** El estudio ([#9]) barrió
   los dos a la vez para dejar el anillo en **2 celdas** y `N` en 20 — por eso el coste es
   constante. Subir `border_px` a 8 **solo** (con `border_reduce` = 2) da `N` = 24 y **235.436
   parámetros: +40 %, otra red**, y ya no es «a coste constante».
   *Medido el 2026-08-29 con `derive_dims` + `build_model`:*

   ```bash
   .venv/bin/python -c "import sys;sys.path.insert(0,'src');\
   from fv.models.builder import full_config,build_model;\
   from fv.fovea import derive_dims;\
   c=full_config({'fovea_px':16,'border_px':8,'border_reduce':4,'overlap_fovea_px':7,\
   'n_layers':4,'channels':[16]*4,'regions':'split'});\
   print(derive_dims(c).N, sum(p.numel() for p in build_model(c).parameters()))"
   #   -> 20 167852     (con border_reduce=2 sale 24 235436)
   ```

2. **Los dos ejes se midieron POR SEPARADO**, cada uno contra el vigente. **Juntos no se han
   medido.** `fov16-optimo` es la mejor apuesta que da la evidencia, no un resultado.

3. **Todo eso es f1 de VENTANA, un proxy** que está medido que **exagera** (en `n_layers` la
   ganancia real fue la mitad). **Ningún eje ha pasado por la métrica de tarea.**

## 2. La receta: `plan40`

`configs/recipes/plan40.yaml` **es** la receta vigente, copiada del `base_recipe_value` de los
recorridos. No hay una segunda: duplicarla sería crear dos que pueden divergir.

`lr` 0,0014 · `batch_size` 85 · `optimizer` adam · `weight_decay` 0,0 · `scheduler` none ·
`monitor` val_loss · `patience` 10 · `pos_weight` 1,0 · `lambda_pos` 1,0 · `smooth_l1_beta` 0,08

⚠ **`epochs` es una guarda, no un ajuste.** Medido sobre los 630 runs con curvas, la época más
alta fue **130** y ninguno llegó a 150. Se pisa desde la línea de comandos sin tocar el fichero.

## 3. Entrenar

```bash
cd ~/src/foveal-vision
.venv/bin/fv-train \
  --name mi-run \
  --window-dataset dirty1000-80px-16px-r20260827 \
  --network fov16-optimo \
  --recipe plan40 \
  --epochs 40
```

`--name` tiene que ser **nuevo**: un run no se sobrescribe nunca. Para seguir uno que ya existe,
`fv-continue` (abajo).

### Cuánto tarda

| Máquina | s/época | 40 épocas | 100 épocas |
|---|---:|---:|---:|
| **este droplet, 2 vCPU** *(medido 2026-08-29 sobre `dirty1000-80px-16px-r20260827`)* | **142–176** | ~1 h 35 – 2 h | ~4 – 5 h |
| las de los estudios *(medido, [#14])* | 46,0 | ~31 min | ~1 h 17 |

⚠ **El rango de arriba no es ruido de medida: es la máquina compartida.** Las dos épocas se
midieron seguidas, en el mismo run — 141,6 s la primera con la máquina libre y 176,3 s la
segunda mientras corrían la web app y otras cosas. Con 2 vCPU, cualquier otra cosa que corra
alarga el entrenamiento. Cuenta con el número alto.

⚠ **Más de unos minutos no cabe en un turno**: lánzalo desacoplado y avísate al terminar, o se
muere con el proceso que lo lanzó.

```bash
"$COORD_HOME/scripts/desacoplar.sh" sh -c '
  cd ~/src/foveal-vision
  .venv/bin/fv-train --name mi-run --window-dataset dirty1000-80px-16px-r20260827 \
      --network fov16-optimo --recipe plan40 --epochs 40 > /tmp/mi-run.log 2>&1
  node "$COORD_HOME/scripts/notify.mjs" "entrenamiento mi-run terminado"' &
```

## 4. Continuar un entrenamiento

```bash
.venv/bin/fv-continue --name mi-run --more 20
```

`--more` son épocas **adicionales**, no el total. Retoma el **mismo** run: misma red, mismo
dataset, misma receta, y el mismo `metrics.jsonl` — las épocas siguen numerando desde donde se
quedaron.

**La red, el dataset y la receta no se pueden cambiar al continuar**, y por eso no son banderas:
serían otro run con el historial de éste pegado detrás, y las curvas mentirían. La única
excepción es `--patience`, porque un run que paró por early-stop volvería a parar en la primera
época si se restaura su contador:

```bash
.venv/bin/fv-continue --name mi-run --more 20 --patience 0     # 0 = sin early-stop
```

### Que continuar sea FIEL no es gratis, y tiene test

«Continuar» sólo significa algo si el resultado es el mismo que no haber parado. Hay un test que
lo fija (`tests/test_continuar.py`): **3 épocas + 3 continuadas dan la misma curva, época a
época, que 6 de una vez.** Para que eso se cumpla, `last.pt` guarda —además de los pesos— el
**optimizador** (sin sus momentos, Adam reempieza), los **contadores** (`best`, `no_improve`: sin
ellos el early-stop y la selección del mejor checkpoint empiezan de cero) y **los tres
generadores**: el de torch, el de numpy y **el del `DataLoader`**, que es el que decide el
barajado y es el que más fácil se olvida — sin él, la época 4 recibe exactamente el orden que
recibió la 1 y el modelo repasa lo mismo creyendo que avanza.

⚠ **Un run entrenado antes de esto no se puede continuar bien**, y el comando **se niega** en vez
de hacerlo mal: sus checkpoints no llevan optimizador, y continuarlos no falla — da una curva
peor sin causa aparente. Si aun así quieres:

```bash
.venv/bin/fv-continue --name viejo --more 10 --optimizador-limpio
```

### ✅ Verificado el 2026-08-30 sobre el dataset real

No es el mundo de pruebas: `fov16-optimo` + `plan40` sobre `dirty1000-80px-16px-r20260827`,
en este droplet. Se entrenó **1 época**, se **continuó 2 más**, y la curva siguió bajando en vez
de reempezar — que es lo único que hace que «continuar» signifique algo:

| época | de dónde | `train_loss` | `val_loss` | f1 (ventana) | s |
|---:|---|---:|---:|---:|---:|
| 1 | `fv-train --epochs 1` | 0,4574 | 0,2726 | 0,739 | 141,6 |
| 2 | `fv-continue --more 2` | 0,2054 | 0,1853 | 0,860 | 176,5 |
| 3 | *(la misma)* | **0,1600** | **0,1460** | **0,892** | 201,6 |

El resumen quedó con `continued_from: 1` y `epochs_run: 3` (acumuladas, no las de la última
tanda). Los dos ficheros de pesos, con el tamaño que explica por qué son dos:
`best.pt` **680.793 B** contra `last.pt` **2.050.697 B** — el 3× es el estado del optimizador.

Y el ciclo entero cierra: ese `best.pt` aparece en el selector de `/review` y **dibuja las cajas
sobre las imágenes del split** (comprobado con navegador, 20 cajas en 10 imágenes, sin errores de
consola).

⚠ **3 épocas no es un modelo entrenado**, es la prueba de que la máquina funciona. Para uno de
verdad, 40–100 épocas con la receta entera.

## 5. Los pesos: DOS ficheros, dos propósitos

Cada run deja los dos en su directorio (`fv.settings.runs_root()`):

| Fichero | Para qué | Qué lleva |
|---|---|---|
| **`best.pt`** | **evaluar** | la mejor época según el monitor. **Sólo pesos** — es lo que lee `load_model`, la métrica de tarea y la pantalla de revisión |
| **`last.pt`** | **continuar** | la última época **con el estado entero** (optimizador, contadores, generadores) |

No llevan lo mismo a propósito: meterle el optimizador a `best.pt` lo engordaría ~3× para nadie,
y quitárselo a `last.pt` haría que continuar no fuese continuar.

⚠ **Los `.pt` NO viajan por git** (`*.pt` está en el `.gitignore` del repo de datos). Un modelo
entrenado vive **sólo en la máquina donde se entrenó**: si la rehaces, se pierde. Lo que sí viaja
es el dataset y la configuración, o sea la receta para volver a obtenerlo.

## 6. Evaluar con imágenes

Una vez hay `best.pt`, la pantalla **Revisar** (`/review`) lo ofrece en su selector de run y
dibuja las cajas sobre las imágenes del split. Es la comprobación a ojo; el número está en la
métrica de tarea:

```bash
curl -s "http://127.0.0.1:8010/api/runs/mi-run/task-score?split=val" | python3 -m json.tool
```

⚠ **La métrica de tarea necesita la FUENTE** (se puntúa contra los párrafos reales de A, que
viven en `labels.jsonl`), y las fuentes no viajan con el código. Si no está, `/review` enseña las
imágenes desde el `windows.npz` pero **sin la verdad**. Publicarla es un comando: ver
[README § «Publicar una fuente»](../README.md).

[#9]: https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/08-agosto/2026-08-26-borde-ancho.md
[#14]: https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/08-agosto/2026-08-26-cierre-parametros.md
