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

🔒 **Y entrenar NO guarda la red para inferencia.** Desde el 2026-08-31 los pesos de un run
**no se conservan por defecto**: sólo los de las redes que el dueño aprueba una a una, y sólo
ésas puede usar la web app. Al terminar, la red se aprueba con **una** orden —
`POST /api/inference/staging/<run>/promote`, que copia los pesos al repo de datos y anota la
entrada en `inferencia.json`. Qué se guarda, por qué, y cómo llegan los pesos mientras se
entrena: **[docs/inferencia.md](inferencia.md)**.

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

## 4 bis. Entrenar en una máquina de Vast (varias CPU)

Este droplet tiene **2 vCPU** y una época cuesta 142–176 s. Una máquina de Vast con 12 vCPU sale
por **~0,05 $/h**, así que entrenar allí es más rápido y más barato que ocupar el servidor de
control durante horas.

```bash
cd ~/src/foveal-vision
set -a; . ~/.config/dev-secrets.env; set +a          # el token de Vast
.venv/bin/python scripts/entrenar_vast.py --name mi-run --epochs 40
.venv/bin/python scripts/entrenar_vast.py --name mi-run --continuar --epochs 40
```

Alquila **una** máquina, sube el código y el dataset, entrena, **se trae los pesos en cada
sonda** y la destruye. Sin argumentos usa `fov16-optimo` + `plan40` sobre
`dirty1000-80px-16px-r20260827`.

### Cambia de máquina si se vuelve lenta, y no pasa del presupuesto

El marketplace da máquinas **muy distintas por el mismo precio**, y una buena se puede volver
lenta a mitad (otro inquilino le come los núcleos). Son **dos casos distintos y se miden
distinto** — el criterio, escrito antes de mirar:

| Caso | Cómo se ve | Umbral | Mínimo de épocas |
|---|---|---|---|
| **se degradó** | contra **sí misma**: mediana de las 3 últimas épocas contra la de las 3 primeras *en esa máquina* | **1,35** | 6 |
| **nació lenta** | contra la **mejor** mediana de esta corrida (contra sí misma no se ve: es lenta desde la época 1) | **1,6** | 3 |

El 1,35 no es inventado: es el que ya usa `estudio_flota --umbral-degradacion`. Y **sin una mejor
con la que comparar, la primera máquina nunca es «lenta»** — cambiarla sería tirar la única
referencia que hay.

Cambiar de máquina es posible **porque `last.pt` viaja**: se destruye la lenta, se alquila otra y
se **continúa**. Es el pago directo de que continuar sea fiel.

```bash
.venv/bin/python scripts/entrenar_vast.py --name mi-run \
    --epochs 300 --patience 20 --presupuesto 5 --max-cambios 6 --aviso-cada 1
```

- `--presupuesto` es un techo **duro** en dólares, y **se mira antes de alquilar la siguiente**
  (con un margen del 10 %): descubrir que no cabe con la máquina ya encendida es justo el gasto
  que el techo existe para evitar.
- `--max-cambios` evita que una racha de máquinas malas queme el presupuesto a base de arranques.
- `--aviso-cada` manda un aviso a Telegram con épocas, f1, s/época y gasto. **El aviso nunca
  rompe el entrenamiento**: la fuente de verdad es el log y el run en disco.

⚠ **La destrucción vive en el `finally` de `una_maquina`, no en el bucle**, y eso es deliberado:
si viviera fuera, cada camino de salida nuevo sería una fuga posible. Tiene test.

### El criterio de parada: `patience`

`--patience 20` (con `--epochs 300` de guarda). Es **más alta que el vigente**, y a propósito:
`ESTADO.md` mide que 20 ganó a 10 **las dos veces** (+0,0028 y +0,0027) y que el eje está
**abierto por arriba**; 10 sigue siendo el vigente sólo porque esa diferencia no llega a δ.

Aquí el objetivo es **un modelo**, no una fila de tabla, y una época cuesta ~0,0012 $ — así que
esperar más es prácticamente gratis.

⚠ **Y por eso un run entrenado así NO es comparable con las tablas publicadas.** La paciencia
usada queda en el `config.json` del run, que es donde hay que mirarla.

### ✅ Verificado el 2026-08-30: entrenado en Vast, continuando lo de aquí

No es una prueba con datos de juguete: se **continuó** el run `fov-optimo-3ep` (que llevaba 3
épocas entrenadas en este droplet) **2 épocas más en una máquina de Vast de 9,3 vCPU**, y el
resultado volvió aquí.

| época | dónde | `train_loss` | `val_loss` | f1 | s/época |
|---:|---|---:|---:|---:|---:|
| 1–3 | este droplet (2 vCPU) | 0,4574 → 0,1600 | 0,2726 → 0,1460 | 0,739 → 0,892 | 142–202 |
| **4** | **Vast (9,3 vCPU)** | 0,1397 | 0,1429 | **0,902** | **83** |
| **5** | **Vast** | **0,1295** | **0,1268** | **0,912** | **84** |

- **La curva siguió bajando al cruzar de máquina**: la continuación es real, no un reinicio.
  `summary.json` quedó con `epochs_run: 5` y `continued_from: 3`.
- **Los pesos bajaron en CADA sonda**, que es lo que se quería comprobar. El log lo dice literal:
  `[bajados: metrics.jsonl, status.json, config.json, summary.json, best.pt, last.pt]` — las tres
  veces.
- **2,4× más rápido** que aquí (83 s contra 202 s), a 0,0516 $/h.
- **La instancia se destruyó sola**: 5,5 min, **0,0048 $**.
- Y lo que quedó sirve para las dos cosas: `best.pt` carga con `load_model` y la pantalla
  `/review` dibuja sus cajas; `last.pt` trae `format_version: 2`, optimizador y los generadores,
  o sea que se puede seguir entrenando.

**Coste de toda la prueba, los cuatro intentos incluidos: ~0,025 $.**

### La diferencia con `estudio_flota.py`, que es la razón de que exista

La flota corre **barridos**: muchos puntos cortos, y su producto es la tabla. Por eso su libro de
a bordo se trae sólo texto y **los pesos se quedan en la máquina** hasta el tar final — está
escrito en su propio docstring, y para un barrido es correcto.

Aquí el producto es **el modelo**. Un `best.pt` que sólo baja al final es un modelo que se pierde
entero si la máquina se cae en la última época. Así que `TRAER` incluye `best.pt` y `last.pt`, y
bajan **en cada sonda** (~2,7 MB). Consecuencia directa: un run entrenado en Vast **se puede
continuar**, porque `last.pt` trae el estado del optimizador.

### Lo que NO garantiza, y hay que decirlo

**La destrucción va en un `finally` de este proceso.** Si el droplet de control muere de golpe, el
`finally` no corre y **la instancia sigue facturando**. No hay interruptor dentro de la máquina
alquilada porque destruirse a sí misma pediría el token de Vast, y ahí **no viaja ningún secreto**
a propósito (`ENVIA` son cuatro rutas y hay test).

Lo que sí hay:

- el `iid` y el comando exacto de destrucción se imprimen **antes que nada más**;
- `node scripts/cerrable.mjs` (en el coordinador) cuenta las instancias vivas y pone el server en
  🔴 mientras alguna respire;
- `--horas-max` corta el entrenamiento —no la factura— si algo se cuelga.

⚠ **Y un run continuado en OTRA máquina no es bit a bit el mismo** que si no se hubiera parado.
`reanudar` restaura los tres generadores, y el payload lleva las **mismas versiones** a los dos
lados (medido el 2026-08-30: `torch 2.13.0+cpu` y `numpy 2.5.2` aquí y allí), así que la
diferencia que queda es **la CPU**: otro juego de instrucciones da otro redondeo. Para entrenar un
modelo da igual; para publicar una tabla comparable, no — es la misma razón por la que la flota
prefiere repetir un punto a reanudarlo.

### ⚠ Antes de nada: una clave de Vast PROPIA, y comprobada

`VAST_SSH_KEY_FILE` apuntaba por defecto a `~/.ssh/do_droplet` —la clave de DigitalOcean—, esa
clave **estaba en la cuenta de Vast** (comprobado por huella, y la API respondía
`SSH key already associated with instance`) **y las instancias la rechazaban igual**:
`Permission denied (publickey)`, tres máquinas seguidas.

Generar una clave dedicada y registrarla lo arregló a la primera:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_ed25519 -N "" -C "vast-$(hostname)"
cd ~/src/digital-ocean-dropplet-auto-launching
VAST_SSH_KEY_FILE=~/.ssh/vast_ed25519 python3 scripts/vast_instance.py register-key
```

...y luego se entrena con `VAST_SSH_KEY_FILE=~/.ssh/vast_ed25519`.

⚠ **La causa de que la primera no valiera NO está establecida** — sólo que estaba en la cuenta y
aun así no entraba. Lo que sí está medido es que la dedicada funcionó en la máquina siguiente.
⚠ Y **la lista de claves de una instancia se fija al crearla**: registrar una clave después no
sirve para las que ya están alquiladas.

### Las DOS trampas de conectarse, que costaron los dos primeros intentos

Las dos estaban **ya medidas en este repo**, dentro de `estudio_flota.py`, y no reutilizarlas es
lo que las hizo volver a morder (2026-08-30):

1. **El banner de sshd no es el login.** `V.esperar_ssh` comprueba que sshd contesta, y eso llega
   **antes** de que la clave esté en `authorized_keys`. El primer intento murió con
   `Permission denied (publickey)` **con la clave correctamente registrada** en la cuenta — media
   hora buscando un problema de claves que no existía. `estudio_flota.sellar` reintenta 12 veces
   justo por esto, con la medida al lado: *2026-08-24, sin reintentos, 3 de 5 máquinas fallaban en
   el primer comando autenticado*.
2. **El `host:puerto` que da la API al arrancar puede no ser el definitivo.** Por eso la flota no
   usa `ssh_destino` a secas sino `resolver_destino`, con reintentos: *medido el 2026-08-24, la API
   devolvió el mismo destino para dos instancias distintas mientras arrancaban*. El segundo intento
   se quedó 5 minutos llamando a un destino leído **una sola vez**, al principio, y se rindió.

`entrenar_vast.conectar()` cubre las dos de una vez: **re-pregunta el destino en cada vuelta** y lo
comprueba con un comando **autenticado**, no con el banner. Tiene test (que además comprueba que no
se llama a `esperar_ssh`).

3. **Y lo que escondía las dos: `V.ssh_capture` se come `stderr`.** Devuelve sólo `stdout`, y el
   motivo por el que SSH falla —`Permission denied (publickey)`, `Connection refused`— viaja por
   `stderr`. Sin él, un **rechazo de clave** y una **máquina que aún no levanta sshd** se ven
   exactamente igual: `rc=255` y nada más. Doce minutos de reintentos ciegos y un diagnóstico
   apuntando al sitio equivocado. `entrenar_vast._ssh` envuelve la llamada para conservar `stderr`
   (no se toca la de `vast_instance`, que la comparten otros), y ahora un rechazo de clave **falla
   rápido y dice qué hacer** en vez de esperar — la misma asimetría que `sellar` ya aplicaba:
   *el transporte mejora esperando, la autenticación no*.

**Y el arreglo se pagó solo en el intento siguiente**: con `stderr` conservado, el log dijo
`Connection refused` — o sea **transporte**, sshd todavía no escuchaba— en vez del `rc=255` mudo
de antes. Es exactamente la distinción que costó los doce minutos, y ahora se lee de un vistazo.

⚠ **Y las dos veces la instancia se destruyó sola** — el `finally` hizo su trabajo: 0,1 min
(0,0001 $) y 5,0 min (0,0035 $). El coste de los dos fallos junto fue **menos de medio céntimo**,
que es exactamente lo que se compra teniendo el camino de destrucción escrito antes de alquilar.

## 5. Los pesos: DOS ficheros, dos propósitos

Cada run deja los dos en su directorio (`fv.settings.runs_root()`):

| Fichero | Para qué | Qué lleva |
|---|---|---|
| **`best.pt`** | **evaluar** | la mejor época según el monitor. **Sólo pesos** — es lo que lee `load_model`, la métrica de tarea y la pantalla de revisión |
| **`last.pt`** | **continuar** | la última época **con el estado entero** (optimizador, contadores, generadores) |

No llevan lo mismo a propósito: meterle el optimizador a `best.pt` lo engordaría ~3× para nadie,
y quitárselo a `last.pt` haría que continuar no fuese continuar.

⚠ **Los `.pt` de un run NO viajan por git** (`*.pt` está en el `.gitignore` del repo de datos). Un
modelo entrenado vive **sólo en la máquina donde se entrenó**: si la rehaces, se pierde. Lo que sí
viaja es el dataset y la configuración, o sea la receta para volver a obtenerlo.

**Y esa receta tarda horas, que es lo que se descubre tarde.** Medido el 2026-08-30 en un dev
recién nacido: los 10 runs de `dirty1000-80px-16px-r20260827` estaban ahí con sus métricas —el
mejor, `fov-optimo-p20`, con f1 **0,9430**— y ninguno con pesos, así que la pantalla **Revisar**
sólo podía enseñar las imágenes sin cajas. Volver a entrenarlo costó **~100 s/época** en las 2
vCPU del dev, o sea ~1 h 20 min hasta la época del mejor checkpoint. Es re-derivable, pero **no en
el momento en que hace falta**, que es al abrir la app.

## Los pesos SÍ se guardan — dos por run, y sólo dos

**Desde el 2026-08-30, y la razón la puso el dueño: hay que poder probar a mano un entrenamiento.**
Sin pesos, la web app enseña las imágenes sin cajas, la métrica de tarea no se puede puntuar, y
«la red detecta mal» no se distingue de «no hay red». Un run que sólo deja métricas es un número
que hay que creerse.

| | qué es | tamaño | para qué |
|---|---|---:|---|
| `best.pt` | la **mejor** época según el monitor | 665 KB | **probarlo a mano** |
| `last.pt` | la **más actualizada**, con el estado entero | 2,0 MB | continuar el run |

*(medidos el 2026-08-30 sobre `fov16-optimo`, 168.652 parámetros)*

**Sólo en el repo de datos.** En el repo de código `/runs/`, `/sweeps/` y `/studies/` están
ignorados enteros: un peso nunca entra ahí.

⚠ **Y sólo esos dos.** Cualquier otro `.pt` sigue fuera del git.

⚠ **La cadencia es parte de la regla.** git guarda **todas** las versiones que se commitean, así
que subir `last.pt` en cada época son 2 MB por época y por run — 140 MB en un run de 70 épocas, y
gigabytes en un barrido. Por eso la flota los trae cada `FV_EPOCAS_POR_PESOS` épocas (**25** por
defecto) y **siempre en el último tirón**, antes de destruir la máquina. Dos o tres versiones por
run, no setenta.

⚠ **Un run lanzado a mano no lo commitea nadie.** La flota tiene su libro de a bordo
(`estudio_flota.py --git`); `fv-train` sólo te imprime el comando al terminar. Si no lo corres, los
pesos mueren con la máquina.

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
