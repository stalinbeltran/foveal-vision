# Qué redes se guardan para inferir, y por dónde llegan sus pesos

**Qué contesta este documento:** qué redes conservan sus pesos, cuáles puede usar
la web app para inferir, cómo llegan al server mientras se entrena, y cómo pasan
a lo definitivo.

---

## 1. La regla, y no es opcional

> **Los pesos de un run NO se guardan por defecto.** Sólo se conservan —y sólo se
> pueden usar para inferir en la web app— los de las redes que **el dueño aprueba
> una a una**.

**Hoy la lista es exactamente una:** `demo-fov16-optimo`, con sus dos ficheros
`best.pt` y `last.pt`. Las demás se añaden **cuando el dueño lo ordene**
(«guarda esta nn para inferencia», «guarda sus pesos»), no antes y no por que
hayan salido bien.

### Por qué la regla es ésa

*Medido el 2026-08-31:* hay **862 runs** en el repo de datos y cada uno son
**2,7 MB** de pesos (`best.pt` 680 KB + `last.pt` 2,0 MB). Guardarlos todos son
**~2,3 GB** en un repo que hoy pesa **49 MB** — y git guarda **todas** las
versiones que se commitean, no la última. La mayoría de esos runs son puntos de un
barrido: lo que se lee de ellos es el número de su tabla, no el modelo.

Y por qué la lista es **explícita** y no una heurística («el mejor f1», «los de
este mes»): una heurística decide sola y en silencio, y lo que está en juego es
que git se llene de una y que la web app infiera con **una red que nadie eligió**.
Una lista se lee, se discute y se revierte.

⚠ **La otra cara, que también está medida:** hasta el 2026-08-30 no se guardaba
**ninguno**, y eso costó `fov-optimo-p20` — 69 épocas en Vast, f1 0,9430,
desaparecido al rehacer la máquina y reentrenado desde cero (~1 h 40 min). La
regla no es «no guardes»: es **«guarda lo que se va a usar, y dilo»**.

## 2. Dónde vive cada cosa

```
<repo de datos>/inferencia.json          el CATÁLOGO: qué redes están aprobadas
<repo de datos>/…/runs/<run>/best.pt     lo DEFINITIVO (commiteado, viaja por git)
<repo de datos>/…/runs/<run>/last.pt

<repo de código>/data/inferencia/<run>/  la ANTESALA (fuera de git, muere con la máquina)
```

| | qué es | quién lo pone | sobrevive a rehacer la máquina |
|---|---|---|---|
| **antesala** | lo que llega **mientras se entrena** | el endpoint (§4) | ❌ no, y es deliberado |
| **definitivo** | lo aprobado | la promoción (§5) | ✅ si se commitea |
| **catálogo** | la decisión | la promoción, o a mano | ✅ |

### Por qué la antesala no está en el repo de datos

Porque **git guarda todas las versiones que se commitean**. `entrenar_vast.py` se
trae los pesos **en cada sonda** —para poder mirar el modelo con el entrenamiento
en marcha, y para no perderlo entero si la máquina se cae en la última época— y
eso son decenas de `last.pt` de 2 MB por run. El `.gitignore` del repo de datos ya
avisa: *«140 MB en un run de 70 épocas, y gigabytes en un barrido»*.

⚠ **Y tampoco es `data/cache/`, aunque se le parezca.** La caché se puede borrar
sin perder nada; la antesala guarda **los únicos pesos de un entrenamiento en
curso**, y borrarla a mitad pierde horas de máquina. Un directorio con otra regla
de borrado es otro directorio.

## 3. Qué red usa la app para inferir

`fv.inference.catalogo.checkpoint_de` mira **dos** sitios, en este orden:

1. **antesala** — hay un entrenamiento en marcha y ésta es la versión que acaba de
   bajar. Gana, a propósito: durante un entrenamiento la buena es la más nueva.
2. **definitivo**, y **sólo si la red está aprobada**.

⚠ **Un `.pt` en el repo de datos que no esté en el catálogo NO se usa.** Es algo
que se coló (una copia a mano, un tar desempaquetado), y servirlo haría inferir
con una red que nadie eligió — justo lo que la lista existe para impedir.

⚠ **Estar en la antesala no es estar aprobada.** Se puede mirar cómo va, y eso es
distinto de «esta red se conserva»: lo primero es provisional y muere con la
máquina, lo segundo es un commit. `promover` es la frontera, y es explícita.

### Lo que ve la app

`GET /api/runs` trae por cada run **dos campos distintos**, y la distinción es el
punto:

| campo | quién lo pone | qué dice |
|---|---|---|
| `has_checkpoint` | E (`RunStore`) | hay un `best.pt` en su directorio |
| `inference` | F (el catálogo) | `"antesala"` · `"catalogo"` · `null` |

Un run puede tener el fichero y **no** poder inferir. En Predecir y en Revisar se
**marcan, no se esconden** (`⛔` sin aprobar, `🟡` en antesala): un run escondido no
se distingue de uno que no existe, y entonces no sabes si hay que **aprobarlo** o
**reentrenarlo**.

## 4. El endpoint: recibir pesos mientras se entrena

```
PUT    /api/inference/staging/<run>/<best.pt|last.pt>   sube UN fichero a la antesala
GET    /api/inference                                   qué hay aprobado y en antesala
POST   /api/inference/staging/<run>/promote             antesala -> definitivo + aprueba
DELETE /api/inference/staging/<run>                     limpia la antesala
DELETE /api/inference/approved/<run>                    retira del catálogo (no borra pesos)
```

El cuerpo del `PUT` son **los bytes del `.pt` en crudo**, no multipart: quien sube
esto es un script.

```bash
T=$(cat ~/.config/fv-web-token)          # desde fuera de la máquina
curl -sf -X PUT -H "x-fv-token: $T" \
     --data-binary @best.pt \
     http://<dev>:8010/api/inference/staging/mi-run/best.pt

# desde la propia máquina, el loopback pasa sin token
curl -sf -X PUT --data-binary @best.pt \
     http://127.0.0.1:8010/api/inference/staging/mi-run/best.pt
```

### La puerta, y por qué no hay una segunda

Estas rutas van en el **mismo `app`** que monta `fv.api.web`, así que heredan su
puerta: **token** (cabecera `x-fv-token`, cookie o `?t=`) salvo desde loopback.
No hay una segunda puerta y **no debe haberla** — dos puertas divergen, y la que se
olvida es la que se deja abierta.

⚠ **Lo que sí cambia es la consecuencia de que se cuele alguien.** El resto del API
borra runs y datasets; esto **escribe ficheros de pesos**, y un `.pt` es un pickle,
o sea **código**. Por eso:

- el nombre del fichero se comprueba contra una lista de **dos** (`best.pt`,
  `last.pt`), por igualdad — nunca se compone una ruta con lo que llega;
- el nombre del run tiene que ser un **nombre de directorio** (sin `/`, sin `..`),
  y se comprueba **en el módulo**, no sólo en el endpoint;
- se guardan **bytes**: no se hace `torch.load` en la subida. Cargar es lo que
  ejecuta el pickle, y aquí no hay ninguna razón para hacerlo;
- hay un **techo de tamaño** (`FV_MAX_CHECKPOINT_MB`, 64 por defecto): sin techo,
  una subida es un disco lleno, y un disco lleno tumba el entrenamiento que estaba
  corriendo.

⚠ **Y una advertencia sobre usarlo DESDE una máquina alquilada.** Hoy
`entrenar_vast.py` se trae los pesos por **ssh (PULL)** y por eso
[su docstring dice que «ahí no viaja ningún secreto a propósito»](../scripts/entrenar_vast.py).
Empujar desde la máquina alquilada (PUSH) invierte eso: haría falta mandarle el
token del API y exponer el puerto del dev. **El PULL sigue siendo el camino
recomendado**; este endpoint existe para el resto de clientes (un entrenamiento en
otra máquina propia, un script local, la promoción) y **no sustituye** al PULL
mientras nadie decida pagar ese precio.

### La escritura es atómica, y no es un detalle

El temporal va **al lado del destino**, no en `/tmp`, y luego `replace`:

- `best.pt` **se lee mientras se reemplaza** — la pantalla de revisión usa el
  modelo con el entrenamiento en marcha. Con un rename atómico quien lee obtiene
  la versión vieja o la nueva, **nunca media**.
- `os.replace` sólo es atómico **dentro del mismo sistema de ficheros**. Con el
  temporal en `/tmp` (que en muchas máquinas es un tmpfs aparte) daría `EXDEV`.
  Es la misma trampa que ya está anotada en `entrenar_vast.traer`.

## 5. La promoción: antesala → definitivo

**Cuando el entrenamiento termina**, `best.pt` y `last.pt` pasan de la antesala al
directorio del run en el repo de datos, y la red **queda aprobada**:

```bash
curl -sf -X POST http://127.0.0.1:8010/api/inference/staging/mi-run/promote \
     -H 'Content-Type: application/json' \
     -d '{"motivo": "el modelo que vamos a usar en la app"}'
```

**Copiar y aprobar son la misma orden**, no dos pasos: unos pesos en el repo de
datos que nadie aprobó son 2,7 MB que git no suelta nunca y que la app no usaría.

⚠ **No commitea.** Igual que `fv-train`, devuelve el comando exacto en la respuesta
(`commit`) y lo deja al usuario: promover no debería escribir en el historial de
nadie sin que se lo pidan. **Pero el commit es parte del encargo** — lo que no está
empujado, no existe:

```bash
cd ~/src/foveal-vision-data && git add -A \
  && git commit -m 'pesos de mi-run para inferencia' && git push
```

### Retirar no borra

`DELETE /api/inference/approved/<run>` saca la red del catálogo y **no toca los
ficheros**. Separado a propósito: retirar es reversible y borrar no — y un peso ya
commiteado **no se va del historial de git** por borrarlo del árbol, así que
borrarlo daría una sensación de limpieza que no es cierta.

## 6. El ciclo completo, de una vez

```
   entrenando                    al terminar                 para que sobreviva
┌───────────────┐            ┌──────────────────┐          ┌──────────────────┐
│ PUT …/staging │  ────────► │ POST …/promote   │ ───────► │ git add/commit/  │
│ (cada sonda)  │            │ copia + aprueba  │          │ push (a mano)    │
└───────────────┘            └──────────────────┘          └──────────────────┘
   data/inferencia/            repo de datos +               remoto
   (fuera de git)              inferencia.json
      🟡 se puede mirar            ✅ la app la usa              ✅ sobrevive
```

## 7. Lo que este diseño NO hace

- **No commitea solo.** Un `git push` automático desde un endpoint escribiría en el
  historial sin que nadie lo pida, y en un repo compartido.
- **No borra la antesala sola.** Si un entrenamiento muere a mitad, sus pesos se
  quedan ahí — que es lo que se quiere: son las únicas horas de máquina que hay.
  Se limpian con `DELETE`, a mano y sabiendo lo que se tira.
- **No recupera los runs previos.** Los 861 runs sin pesos del repo de datos **no
  los tienen en ninguna parte**: no se guardaron nunca. La única forma de tener el
  modelo de uno de ellos es **reentrenarlo**.
- **No decide qué red aprobar.** Eso es del dueño, siempre.
