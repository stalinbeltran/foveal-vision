# CNN plana de 1 capa y 4 kernels 7×7 — misma entrada y salida que la foveada

> **Tiene un GEMELO de 2 kernels**: [`../2026-09-03-cnn-plana-2k7/`](../2026-09-03-cnn-plana-2k7/).
> Lo único que cambia entre los dos es `channels`; todo lo demás —geometría, receta, semilla, y
> las **mismas 10 ventanas**— es idéntico, y los stops caen en las mismas épocas (0, 3, 11, 24,
> 37) para poder ponerlos uno al lado del otro.

**Encargo del 2026-09-03** ([`instrucciones/01-encargo.md`](instrucciones/01-encargo.md)).
Entrenamiento **en curso, por avances**. Hechos cuatro: **37 épocas, 24,1 min** en este droplet
(2 vCPU). **0 máquinas alquiladas, 0 $.**

---

## 1. La red

Misma **entrada** y misma **salida** que `fov16-optimo-mask`; lo único distinto es la estructura
de en medio. Config: [`configs/networks/plana-20-4k7.yaml`](../../configs/networks/plana-20-4k7.yaml).

```
x (2, 20, 20)   ← la MISMA vista 20×20 + el MISMO canal de relleno
   → Conv2d(2 → 4, 7×7, stride 1, padding 3)      ← UNA capa, CUATRO kernels
   → (sin ReLU: es la última capa, y el repo la deja con signo a propósito)
   → flatten (1.600) + los 4 escalares de borde
   → Linear(1604 → 12)                            ← 4 esquinas × [existe, x, y]
```

| | foveada `fov16-optimo-mask` | esta plana |
|---|---|---|
| regiones | `split` (fovea + periferia, con máscaras) | **`single`** (una rama, sin máscaras) |
| capas | 4 | **1** |
| kernels por capa | 16 | **4** |
| tamaño de kernel | 3×3 | **7×7** |
| entrada | 20×20, 2 canales en la periferia | **20×20, 2 canales** |
| salida | 12 | **12** |
| parámetros | 168.044 | **19.656** |

**Receta: `plan40`** — los parámetros óptimos **de la foveada** (`lr` 0,0014, adam, `batch` 85,
`patience` 10, semilla 1). ⚠ Nadie los ha ajustado a una plana de una capa; se usan porque un
punto de partida medido vale más que uno inventado, y porque es lo que pidió el dueño.

## 2. Los avances, y que se pueden añadir épocas

Lo pedía el encargo explícitamente, y **funciona con los comandos que ya existían**:

```bash
fv-train    --name plana-4k7-s1 --window-dataset dirty1000-80px-16px-r20260827 \
            --network plana-20-4k7 --recipe plan40 --epochs 1
fv-continue --name plana-4k7-s1 --more 2        # ← añade épocas al run YA entrenado
```

`fv-continue` reanuda desde `last.pt` y **continúa la curva**, no la reinicia: restaura el
estado del optimizador y el RNG del barajado, así que la época 2 no vuelve a ver el orden de la 1.

| avance | épocas | reloj | `val_loss` | f1 | err. posición |
|---|---:|---:|---:|---:|---:|
| `fv-train --epochs 1` | 1 | 37,5 s | 0,5218 | 0,109 | 3,71 px |
| `fv-continue --more 2` | 2 → 3 | 75,2 s | 0,3749 | 0,582 | 2,89 px |
| `fv-continue --more 8` | 4 → 11 | 5 min 05 s | 0,2691 | 0,792 | 2,44 px |
| `fv-continue --more 13` | 12 → 24 | 8 min 42 s | 0,2397 | 0,827 | 2,28 px |
| `fv-continue --more 13` | 25 → 37 | 8 min 24 s | **0,2297** *(ép. 34)* | **0,844** *(ép. 36)* | **2,22 px** |

**Total: 37 épocas, 24,1 min.** El primer avance fue el «~2 minutos» que pedía el encargo; los
otros tres los eligió el dueño.

### ⚠ La mejora ya es MÁS PEQUEÑA QUE EL RUIDO

| avance | épocas | Δ `val_loss` | Δ f1 | por época |
|---|---:|---:|---:|---|
| 2º | 8 | **−0,106** | +0,210 | −0,0133 |
| 3º | 13 | −0,029 | +0,035 | −0,0022 |
| **4º** | 13 | **−0,010** | +0,017 | **−0,0008** |

El rendimiento por época ha caído **17×** desde el segundo avance. Y en el cuarto la `val_loss`
oscila entre **0,2297 y 0,2663** — una banda de **0,037**, que es **casi cuatro veces** lo que ha
mejorado en todo el avance. Cada época nueva mueve el número más por dónde cae el barajado que
por aprendizaje.

⚠ **Y la mejor época ya no es la última**: la mejor `val_loss` es la **34** (0,2297) y el run
acabó en la **37** (0,2377). O sea que `best.pt` ≠ `last.pt`, y **`patience` está a 3 de 10** —
siete épocas más sin mejorar y el próximo avance para solo.

⚠ **Los stops evalúan `last.pt`, no `best.pt`**, y desde este avance eso importa: `stop-04` es la
época **37**, no la 34. Es lo correcto para «cómo va el entrenamiento» —es el estado desde el que
se continúa— pero no es el mejor modelo del run.
*(la foveada de referencia está en f1 0,954 y 1,05 px, con 8,5× los parámetros y 4 capas)*

⚠ **La foveada tiene 168.044 parámetros, no los 168.844 que dicen otros documentos del repo.**
Lo cazó el `verificador` el 2026-09-03: ese número sale de `network_trace`, que cuenta
`state_dict().values()` en vez de `parameters()`, así que suma los **dos buffers de máscara** de
20×20 (400 + 400 = 800) que **no se aprenden**. Con `regions: single` no hay máscaras, así que
en la plana los dos métodos coinciden — y comparar los dos números tal cual le regalaba 800
parámetros a la foveada. **El bug de `network_trace` es anterior a este experimento**
(`builder.py:338`, commit `5432c9c15`) y **no lo he tocado**: ese número está citado en varios
sitios y cambiarlo de tapadillo desalinearía la documentación. Queda dicho para que lo decidas.

**Los pesos están en `foveal-vision-data/2026/09-septiembre/runs/plana-4k7-s1/`** (`best.pt` para
evaluar, `last.pt` para continuar). ⚠ **Sin commitear a propósito**: el encargo dice «guarda los
pesos localmente… no hace falta commit hasta que yo te indique». Lo que **sí** entra en git son
las salidas de la evaluación.

## 3. La evaluación de este experimento

⚠ **No se evalúa la salida típica de la red.** El encargo lo dice: lo que se mira es la
**entrada pasada por los kernels** — 4 imágenes por cada imagen de entrada.

- **El set de visualización son 10 ventanas de validación**, elegidas al azar **una vez** y
  congeladas en [`../comun/set-visualizacion.json`](../comun/set-visualizacion.json). Se reusan en
  todos los stops **y en el experimento gemelo de 2 kernels**: dos stops sólo son comparables
  sobre las mismas entradas.
- ⚠ **El evaluador, el set y las imágenes de entrada viven en [`../comun/`](../comun/)**, no aquí.
  Desde que existe [`cnn-plana-2k7`](../2026-09-03-cnn-plana-2k7/) —idéntico salvo `channels`—
  comparar sus stops con los de aquí sólo significa algo si la medida es **la misma**. Dos copias
  del evaluador derivarían y la comparación se volvería una ilusión sin que nada fallara.
- **Se evalúa antes de entrenar y en cada stop.** El `stop-00` es la red **sin entrenar**, y es
  *la misma* red que luego entrena: la init se reproduce con la semilla de la receta, y está
  **comprobado** que construir los datasets en medio no consume el RNG global.

| qué | dónde |
|---|---|
| **las 10 entradas** *(compartidas con el gemelo)* | [`../comun/entradas/`](../comun/entradas/) |
| `stop-00-sin-entrenar` — red recién inicializada | [`evaluacion/stop-00-sin-entrenar/`](evaluacion/stop-00-sin-entrenar/) |
| `stop-01-3epocas` — tras el primer avance (~2 min) | [`evaluacion/stop-01-3epocas/`](evaluacion/stop-01-3epocas/) |
| `stop-02-11epocas` — tras el segundo avance (+8 épocas) | [`evaluacion/stop-02-11epocas/`](evaluacion/stop-02-11epocas/) |
| `stop-03-24epocas` — tras el tercero (+13 épocas, ~8 min) | [`evaluacion/stop-03-24epocas/`](evaluacion/stop-03-24epocas/) |
| `stop-04-37epocas` — tras el cuarto (+13 épocas, ~8 min) | [`evaluacion/stop-04-37epocas/`](evaluacion/stop-04-37epocas/) |

### Las entradas

![las 10 entradas](../comun/entradas/entradas.png)

`../comun/entradas/` trae `entradaNN.png` (la vista 20×20) y `entradaNN-relleno.png` (el
segundo canal), más la hoja de arriba. **Están fuera de los `stop-*/` a propósito**: el set está
congelado, así que son idénticas en todos los stops — copiarlas en cada uno invitaría a creer que
pueden cambiar entre uno y otro, que es justo lo que no puede pasar para que dos stops se
comparen.

⚠ **Se guardan los DOS canales porque la red ve los dos.** El de relleno es `1 − cobertura`:
0 = píxel real, 1 = inventado por `pad_mode: edge`. Y sale un dato del sorteo que conviene tener
delante al leer los mapas: **5 de las 10 ventanas tocan el borde de la imagen** (#2, #4, #6, #7 y
la #9, que toca dos a la vez). En esas, parte de lo que responde el kernel es relleno, no imagen.

Cada stop trae **40 PNG** (`entradaNN-kernelJ.png`), `mapas.npy` con los valores crudos,
`resumen.json` y **tres** figuras:

| figura | para qué |
|---|---|
| **`entrada-y-salidas.png`** | **la de revisar**: una columna por entrada y una fila por kernel, como `entradas.png`. «Qué hace el kernel 2 en las diez» se lee recorriendo **una** fila |
| `montaje.png` | la salida **cruda**, entradas en filas. ⚠ Sale como placas planas de color: ver el aviso de abajo |
| `montaje-sin-nivel.png` | lo mismo que el montaje, con la mediana de cada mapa restada |

⚠ **Un stop es una foto en el tiempo, y `--run` lee siempre `last.pt`.** Volver a correr la
etiqueta de un stop viejo después de seguir entrenando lo **reescribe con los pesos de hoy**.
Pasó el 2026-09-03 al añadir la tira: `stop-01-3epocas` quedó con los pesos de la época 11
—montaje y tira idénticos a los del `stop-02`— y sólo se notó porque dos ficheros tenían
exactamente el mismo tamaño. Se recuperó de git. Ahora:

- el script **se niega** si la etiqueta del stop ya existe con otro modelo (hace falta `--rehacer`);
- **`--solo-figuras`** rehace las figuras de un stop viejo **desde su `mapas.npy`**, sin tocar la
  red. El artefacto de registro de un stop es su `mapas.npy`; las figuras son **derivadas**.

```bash
E=experimentos/2026-09-03-cnn-plana-4k7
python experimentos/comun/aplicar_kernels.py --exp $E --red plana-20-4k7 --stop 05-Nepocas --run plana-4k7-s1
python experimentos/comun/aplicar_kernels.py --exp $E --red plana-20-4k7 --stop 01-3epocas --solo-figuras
```

### Qué se ve, tras 37 épocas

![entrada y salidas](evaluacion/stop-04-37epocas/entrada-y-salidas.png)

**Los kernels cogen las líneas de texto** — claro en las entradas #3, #6 y #8, y el canto
vertical del bloque en la #4 y la #9.

**Y los cuatro kernels se están repartiendo el trabajo**, que es lo que más se ve al comparar
stops:

| | k0 | k1 | k2 | k3 |
|---|---:|---:|---:|---:|
| norma L2 · ép. 3 | 0,845 | 0,735 | 0,771 | 0,847 |
| norma L2 · ép. 11 | 1,274 | 0,743 | 0,976 | 1,485 |
| norma L2 · **ép. 24** | **1,554** | 0,852 | 1,113 | **1,869** |
| \|respuesta\| media · ép. 3 | 0,339 | 0,222 | 0,303 | 0,459 |
| \|respuesta\| media · ép. 11 | 0,686 | 0,154 | 0,420 | 0,962 |
| \|respuesta\| media · ép. 24 | 0,838 | 0,142 | 0,432 | 1,125 |
| norma L2 · **ép. 37** | **1,685** | 1,014 | 1,151 | **2,020** |
| \|respuesta\| media · **ép. 37** | **0,811** | **0,147** | 0,436 | **1,096** |

**El reparto se ha consolidado, no revertido.** k0 y k3 siguen creciendo (respuesta ×2,5 y ×2,5
desde la época 3) y **k1 sigue apagándose**: su norma sube un poco —0,735 → 0,852— pero su
respuesta **baja otra vez**, de 0,154 a 0,142, o sea **−36 %** desde la época 3. k2 se ha quedado
plano desde la 11 (0,420 → 0,432).

⚠ **En la práctica quedan dos kernels y medio, y el patrón AGUANTA.** De la época 24 a la 37 las
respuestas están congeladas (k0 0,838 → 0,811; k1 0,142 → 0,147; k2 0,432 → 0,436; k3 1,125 →
1,096) aunque las **normas siguen creciendo**. O sea: los pesos se hacen más grandes y la
respuesta no cambia — el reparto es estable. **k3 responde 7,5× más que k1.** Con sólo 4 kernels,
la red está usando la mitad de su primera capa.

⚠ Con una semilla y 24 épocas esto **acota, no declara**.

⚠ **Hay DOS montajes, y el crudo engaña.** `montaje.png` es la salida tal cual y sale como
placas planas de color. No es un fallo del pintado: está **medido** que el nivel constante de
cada mapa es **8,2× la estructura del texto** (0,300 contra 0,037), así que con una escala común
el rizo que interesa es invisible. `montaje-sin-nivel.png` le resta a cada mapa su mediana y
toma la escala del **p99 del interior** — el anillo de 3 px queda saturado a propósito.

### El anillo rojo del borde: es el PADDING DE CEROS, no las máscaras

Preguntado por el dueño el 2026-09-03. Tres cosas se confunden fácil aquí, así que se separaron
midiendo, sobre el modelo de la época 37 y las 10 ventanas del set:

**1. Esta red NO tiene máscaras.** Con `regions: single`, `build_masks` no se llama nunca
(`builder.py:145`) y el modelo no tiene **ningún** buffer — comprobado listando `named_buffers()`:
sale vacío. Las máscaras `center_mask`/`periph_mask` sólo existen con `regions: split`.

**2. Tampoco es el canal de relleno** (el que se llama `mask_channel`, de ahí la confusión). Si
lo fuera, el anillo saldría sólo en las 5 ventanas que tocan el borde de la imagen. Sale en las
**diez** — y las dos que **no** tocan borde son justo las de **mayor** ratio anillo/interior:

| | #1 | #2 ✎ | #3 | #4 ✎ | #5 | #6 ✎ | #7 ✎ | #8 | #9 ✎ | #10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anillo / interior | **39,8×** | 12,1× | 6,2× | 8,0× | 13,0× | 5,6× | 9,9× | 7,1× | 7,2× | **38,8×** |

*(✎ = la ventana toca el borde de la imagen)*. Y separando por canal, el anillo viene **0,927 de
la vista** contra **0,119 del relleno**: el 11 %.

**3. Es el padding de ceros de la convolución.** Mismo modelo, misma entrada, cambiando **sólo**
el relleno:

| relleno de la conv | \|anillo\| | \|interior\| | ratio |
|---|---:|---:|---:|
| **`zeros`** (el real, es el defecto de `nn.Conv2d`) | **0,741** | 0,079 | **9,4×** |
| `replicate` | 0,211 | 0,079 | 2,7× |

**El mecanismo:** la vista es papel con valor ≈1 casi en todas partes. `nn.Conv2d(..., padding=3)`
rellena con **ceros** fuera del borde, así que en el anillo de 3 px el kernel ve una mezcla de
papel y ceros y su respuesta se desplaza en ≈ (suma de los pesos truncados) × 1. Ese salto es del
orden del **nivel**, que a su vez es ~8× el rizo del texto — por eso el anillo domina la figura.

#### ⚠ Y esto destapa una incoherencia del proyecto, que no es sólo de este experimento

La vista se construye con `pad_mode: edge` **precisamente para no rellenar con ceros**:

> *decisión C10: never plain zeros — zero means "no ink" and teaches a false rule*
> — `fv/fovea/__init__.py:603`

Pero la **convolución** sí rellena con ceros (`builder.py:191`, sin `padding_mode`), y eso
reintroduce un paso más adentro exactamente lo que `pad_mode: edge` evita. **Aplica igual a
producción**: `fov16-optimo-mask` usa la misma línea. Ahí el kernel es 3×3, así que el anillo es
de **1 px** (el 19 % de las celdas de una vista 20×20, contra el 51 % aquí con 7×7).

⚠ **NO lo he cambiado**, y no es un cambio menor: `padding_mode='replicate'` en `_make_branch`
alteraría **todas** las redes del repo y dejaría los checkpoints guardados con otro significado.
Y no está medido que perjudique al f1 — la cabeza es una `Linear` sobre las 1.600 features y
puede aprender a ignorar el anillo. Lo que sí está medido es que **gasta capacidad y domina la
visualización**. Queda dicho para que lo decidas.

#### ⚠ Corrección de lo que decía antes este README

Esta sección decía que *«el anillo del borde aporta poco: el marco es 1,18× el interior con
padding de ceros y 1,08× con replicate»*. **Ese número comparaba mal**: era la media de
`|respuesta|` **con el nivel dentro**, y el nivel es común a todo el mapa, así que diluía el
efecto. Comparado como se ve en la figura —quitando a cada mapa su nivel— es **9,4× contra
2,7×**, no 1,18× contra 1,08×.

**Las tres hipótesis que probé para el aspecto de placa plana** (que es otra cosa, y sigue
valiendo): dominaba el canal de relleno ❌ (el 90-97 % viene de la vista) · dominaba el DC del
kernel ❌ *(rechazada con el criterio equivocado: ver abajo)* · **el nivel constante aplasta la
escala ✅** (0,300 contra 0,037, o sea 8,2×). El nivel sí nace de la media del kernel por un papel
casi uniforme, así que la segunda era correcta en el mecanismo — la descarté comparando
`|suma|/L2` contra su máximo teórico, cuando lo que decide es nivel **contra rizo**.

### ⚠⚠ Y esto PUEDE estar afectando a las redes ya entrenadas — medido en producción

Pedido por el dueño el 2026-09-03. La pregunta no es si el anillo se ve, sino **si importa para
las redes que ya están entrenadas**. Medido sobre **`fov16-mask-p20`**, la red foveada entrenada
(62 épocas, f1 0,954), sobre las mismas 10 ventanas y cambiando **sólo** el relleno de sus
convoluciones (`nn/contaminacion_produccion.py`):

| rama | celdas cuya salida cambia | \|cambio\| medio | ...y en las celdas afectadas |
|---|---:|---:|---:|
| centro | 97 / 400 — **24 %** | 4,0 % de la escala | **16 %** |
| **periferia** | 256 / 400 — **64 %** | 22,3 % de la escala | **35 %** |

**El 64 % del mapa de la periferia depende de con qué se rellena el borde.** No es un efecto de
un píxel: con 4 capas de 3×3 la contaminación entra 4 px por lado, y `20² − 12² = 256` celdas de
400 quedan dentro de ese anillo. En el centro entra menos porque su máscara ya lo recorta.

#### El mecanismo, y por qué es una pérdida real y no sólo un artefacto

La vista 20×20 es un **recorte de una imagen más grande**. Para una ventana que no esté pegada al
borde de la imagen, **los píxeles vecinos EXISTEN**: `build_view` los tiene disponibles y
`pad_mode: edge` los usaría. Pero `nn.Conv2d(..., padding=p)` no los pide: rellena con **ceros**.

O sea que en el anillo la red **sustituye contexto real por «no hay tinta»**, que es exactamente
la regla falsa que la decisión C10 del proyecto prohíbe:

> *decisión C10: never plain zeros — zero means "no ink" and teaches a false rule*
> — `fv/fovea/__init__.py:603`

La vista respeta C10; la convolución no. Es la misma decisión tomada dos veces con criterios
opuestos, un nivel más adentro — el patrón que este proyecto ya tiene anotado.

#### ⚠ Lo que este número NO dice, y es importante

**Que la red sea SENSIBLE a la elección no significa que la elección le haya HECHO DAÑO.**
`fov16-mask-p20` se **entrenó** con relleno de ceros, así que ha aprendido a convivir con él;
cambiárselo al inferir la rompe, y eso es lo único que mide la tabla de arriba. La pregunta de
verdad —*«¿habría salido mejor entrenada con `replicate`?»*— **no está medida, y no se puede
responder sin correr el control.**

Hay un argumento en contra de que importe, y hay que ponerlo: el artefacto es **idéntico en todas
las ventanas** (siempre es el borde de la vista), así que la cabeza —una `Linear` sobre las 12.800
features— puede aprender a descontarlo. Lo que cuesta seguro es **capacidad**: 256 de 400 celdas
de la periferia llevan una señal que en parte es el relleno, no la imagen.

⚠ Y **no explica** la caída de recall en el borde de la IMAGEN (0,608 contra 0,939) que motivó
`mask_channel: coverage`: ese efecto sí depende de dónde esté la ventana, y éste no.

#### Cuatro soluciones, ordenadas por lo que cuestan. NINGUNA implementada

| | qué | coste | qué se rompe |
|---|---|---|---|
| **A** | **Medirlo antes de tocar nada.** Un run de control idéntico a `fov16-mask-p20` con `replicate`, y comparar f1 y `val_loss` | **~45 min** *(medido: 43,3 s/época × 62)* | nada |
| **B** | **`conv_pad_mode` como DATO de la config**, con defecto `zeros` = lo de hoy. Una línea en `_make_branch` (`builder.py:191`) más el campo en `NETWORK_DEFAULTS` | ~1 h de código + tests | nada si el defecto es `zeros`: toda red existente sigue significando lo mismo |
| **C** | **Cambiar el defecto a `replicate`** | 1 línea | ⚠ **todas** las redes del repo cambian de significado. Los checkpoints siguen cargando —no cambia ni un parámetro— y por eso el fallo sería **silencioso**: las tablas publicadas dejarían de ser reproducibles sin que nada falle |
| **D** | **Vista con margen y convolución `valid`**: construir la vista con `L·(k//2)` px extra de imagen REAL, convolucionar sin relleno y quedarse con las 20×20 centrales. Es la única que no inventa nada | alto: toca `build_view`, el dataloader, la inferencia y la geometría | ⚠ cambia el tamaño del tensor de entrada; y **no vale para ventanas pegadas al borde de la imagen**, donde el contexto no existe y hay que rellenar igual |

**El orden recomendado es A → B → decidir.** A responde la única pregunta que importa y no rompe
nada; B es la forma que este repo usa para todo lo demás —el mando es un dato, no un `if`— y deja
el eje barrible. **C es la trampa**: es la más fácil de escribir y la única cuyo fallo no avisa.

⚠ Y si se hace A, el control tiene que ser **`replicate` en el ENTRENAMIENTO**, no sólo al
inferir. Cambiar el relleno a una red ya entrenada mide otra cosa —lo de la tabla de arriba— y
leerlo como «el relleno es malo» sería justo el error que este experimento acaba de cometer con
las dos primeras hipótesis del anillo.

## 4. Qué hay aquí

| | |
|---|---|
| [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md) | el encargo tal como llegó (commit `9926bd0b2`) |
| [`../comun/aplicar_kernels.py`](../comun/aplicar_kernels.py) | la evaluación, **compartida con el gemelo**: congela el set, aplica los kernels, guarda las imágenes y los montajes |
| [`nn/porque_el_anillo.py`](nn/porque_el_anillo.py) | separa las tres causas del anillo (máscaras · canal de relleno · padding) |
| [`nn/contaminacion_produccion.py`](nn/contaminacion_produccion.py) | cuánto del mapa de `fov16-mask-p20` depende del relleno |
| [`../comun/entradas/`](../comun/entradas/) | las 10 entradas en PNG: vista y canal de relleno |
| `evaluacion/stop-*/` | un directorio por stop |
| los pesos | **fuera de git**, en `foveal-vision-data/2026/09-septiembre/runs/plana-4k7-s1/` |

## 5. Lo siguiente — pendiente de tu decisión

El encargo dice avisarte al terminar el primer avance para decidir si seguimos con pasos de
2 min o los alargamos. **Ahí está.** A 37,5 s/época:

| paso | épocas | acumulado |
|---|---:|---:|
| 2 min | +3 | 14 |
| 5 min | +8 | 19 |
| 15 min | +24 | 35 |
| 30 min | +48 | 59 |

Un paso se lanza con `fv-continue --name plana-4k7-s1 --more N` y luego
`python nn/aplicar_kernels.py --stop 03-<N>epocas --run plana-4k7-s1`.

⚠ **Y un aviso sobre `patience`:** la receta trae `patience: 10`, así que si en algún avance
pasan 10 épocas sin mejorar la `val_loss`, el run **para solo** antes de agotar las épocas
pedidas. Con avances cortos no ha llegado a pasar; en un paso de 30 min puede. Se cambia sólo
para esa continuación con `fv-continue --patience 0` (sin early-stop).
