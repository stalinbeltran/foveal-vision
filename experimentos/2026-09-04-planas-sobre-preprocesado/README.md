# Planas de verdad sobre datasets preprocesados **construidos antes de entrenar**

**Estado: los TRES DATASETS ESTAN CONSTRUIDOS** (2026-09-04). Falta la red y el entrenamiento.

| dataset | carpeta | forma | disco | muestra |
|---|---|---|---|---|
| `1k3` | `data/preprocesado/1k3-relu/` | (140000, 1, **18, 18**) | 180 MB | [`muestras/1k3.png`](muestras/1k3.png) |
| `1k5` | `data/preprocesado/1k5-relu/` | (140000, 1, **16, 16**) | 144 MB | [`muestras/1k5.png`](muestras/1k5.png) |
| `1k7` | `data/preprocesado/1k7-relu/` | (140000, 1, **14, 14**) | 112 MB | [`muestras/1k7.png`](muestras/1k7.png) |

Construidos en **1 min 20 s cada uno** (este droplet, 2 vCPU, 0 $), y `--comprobar` valida los
tres contra su manifiesto. Cada uno lleva su `manifiesto.json` con la huella de los pesos del
kernel, y su propia copia de la muestra.

### Las muestras: las MISMAS 10 ventanas en los tres

Una fila por ventana: la vista 20×20 que entró y el mapa que el dataset guarda. Los índices salen
de `comun/set-visualizacion.json`, el set congelado que comparten los siete gemelos, así que las
tres imágenes se pueden poner una al lado de otra — es la misma ventana en la misma fila.

⚠ **Se leen DEL `.npz` construido, no se recalculan.** Es la diferencia entre una figura y una
comprobación: si el constructor tuviera un fallo —el lookup de imágenes, el orden de las filas, la
escala— una figura recalculada saldría perfecta y el dataset seguiría roto.

⚠ **Y se ve la ReLU**: no hay una sola celda azul en los tres PNG, porque no quedan negativos.
Ceros por la ReLU: **15 %** (1k3), 13 % (1k5), 7 % (1k7).

## Qué arregla respecto al intento anterior

[`2026-09-04-preproceso-kernel-congelado`](../2026-09-04-preproceso-kernel-congelado/) metió el
kernel congelado **dentro** del modelo, como capa 0. Daba el mismo tensor, pero cambiaba lo que se
medía — y el dueño lo detectó leyendo la tabla de estructura: *«las entradas de las cnn son
siempre data 20x20???»*.

| | intento anterior (**detenido**) | **este** |
|---|---|---|
| preproceso | capa 0 del modelo, al vuelo | **dataset construido antes** |
| entrada a la red | `(2, 20, 20)` — la vista de siempre | **el mapa preprocesado** (18² · 16² · 14²) |
| convoluciones | **2** → no era «plana» | **1** → plana de verdad |
| no-linealidad tras el kernel | sí, en el forward | sí, **dentro de `aplicaKernel`** |

El encargo pedía *«3 datasets pre-procesados… y con ellos genera 3 cnn planas»*. Una «plana» en
esta serie es **una** convolución + cabeza, como los siete gemelos. Eso es lo que se hace aquí.

## La forma

```
paso 1 (una vez, antes de entrenar)
  ventana 32×32 px → build_view → vista (2,20,20) → aplicaKernel_1k<kf> (pad 0)
                                                  → mapa (1, 20−kf+1, 20−kf+1)   ← EL DATASET

paso 2 (entrenamiento, una plana de verdad)
  mapa (1,H,W) → Conv2d(1→1, 3, pad 0) → flatten → ReLU → concat 4 edge → Linear(→12)
```

| brazo | kernel | **dataset** | tras conv k=3 | features | ancla iso-features ya pagada |
|---|---|---|---|---:|---|
| `pre-1k3` | 3×3 | **18×18** | 16×16 | 256 | `1k5 crudo` → f1 0,642 |
| `pre-1k5` | 5×5 | **16×16** | 14×14 | 196 | `1k7 crudo` → f1 0,618 |
| `pre-1k7` | 7×7 | **14×14** | 12×12 | 144 | ninguna |
| *referencia* | — | *(2,20,20)* | 18×18 | 324 | **es** `1k3 crudo` → f1 0,680 |

⚠ **La referencia sigue sin haber que lanzarla**: una plana de un 3×3 sin relleno sobre la
entrada cruda **es** `2026-09-04-cnn-plana-1k3-sinpadding`, ya corrida, 37 épocas, 0 $.

## La no-linealidad va DENTRO de `aplicaKernel` — decidido por el dueño (2026-09-04)

> *«El dataset debe ser generado con las funciones que aplican kernel, y esas funciones ya deben
> aplicar la no-linearidad»*

Así que `construir_datasets.py` **no tiene ningún flag de activación**: llama a `aplicaKernel` y
lo que salga es lo que el preprocesador define como su salida. Hoy eso es **`relu`**
(`preproceso.ACTIVACION_POR_DEFECTO`), y queda escrito en el manifiesto de cada dataset — no
porque se pueda elegir, sino porque dos datasets con distinta activación no serían comparables y
el nombre del fichero no lo diría.

**Por qué importa, y no es cosmético:** sin no-linealidad el preproceso es una operación
**lineal**, así que la plana que entrene encima haría `conv(conv(x))` sin nada en medio — o sea
**una sola convolución** de tamaño `k+2` con los pesos atados — y cada brazo sería un subconjunto
estricto de un gemelo ya corrido, capaz sólo de empatar o perder. Con la ReLU dentro, eso no pasa.

⚠ **Su precio, medido** (2026-09-04, sobre las 10 ventanas del set congelado): la ReLU tira el
**15 %** de las celdas en `1k3`, el **13 %** en `1k5` y el **7 %** en `1k7`. Es información real
(la respuesta del kernel viene con signo) y se pierde a propósito.

⚠ **`activacion='ninguna'` sigue existiendo, y hace falta**: es lo que usa
`preproceso.py --comprobar` para demostrar que el kernel es **literalmente** la capa L1 de su red,
contra los `mapas.npy` guardados, que están sin activar. Las dos cosas conviven — la identidad se
prueba sin activar, el preproceso se usa activado.

## ⚠⚠ Este experimento calcula LA MISMA FUNCION que el detenido

Con la ReLU dentro de `aplicaKernel`, la cadena es idéntica a la del experimento parado:
`relu(kernel ⊛ x)` → conv entrenable → flatten → ReLU → cabeza. Cambia **dónde** ocurre el
preproceso, no qué se calcula.

| | preproceso | convs entrenables | ¿misma función? |
|---|---|---|---|
| `preproceso-kernel-congelado` (detenido, ép. 11) | capa 0 del modelo | 1 (+1 congelada) | **sí** |
| **este** | dataset construido antes | **1** (plana de verdad) | **sí** |

**Lo que se gana** es lo que pidió el dueño: la red es una plana de verdad, el dataset se puede
inspeccionar antes de entrenar, y **~33 % menos de reloj por época** porque la convolución
congelada deja de recalcularse en cada época.

**Lo que NO se gana: evidencia independiente.** Los números del detenido (ép. 3 y 11) son una
**previsión** de los de aquí, no una segunda medida. Sólo cambiará la inicialización de la conv
entrenable. Conviene saberlo antes de leer la tabla como si fueran dos estudios.

## ⚠ Y el canal de relleno se pierde, y está medido que vale

El kernel consume `(vista, relleno)` y devuelve **un** mapa, así que la plana ya no puede pesar
el relleno por su cuenta. El reporte #19 midió que ese canal sube el recall del último píxel de
**0,608 a 0,974**. `--con-relleno` lo conserva recortado como 2º canal; por defecto **no**, que
es la lectura literal del encargo. **Sigue abierto** si quieres conservarlo.

## Dónde caen los datasets, y por qué no en git

**~435 MB** en float32 los tres, contra un repo de datos de **197 MiB**. Y son re-derivables
exactamente de (kernel commiteado + dataset origen commiteado + opciones), así que la regla 4 de
[`../README.md`](../README.md) manda enlazarlos, no guardarlos. Van a `data/preprocesado/`
(**fuera de git**); lo que se commitea es el script y el `manifiesto.json` con su huella.

⚠ **Este server es efímero, así que reconstruir es un paso del arranque, no un extra.**
`--comprobar` distingue **«no está»** de **«está y no casa con su manifiesto»**, que es el caso
peligroso: un `best.pt` reentrenado daría otro dataset con el mismo nombre. Por eso la huella se
calcula sobre los **pesos** del kernel, no sobre su nombre.

## Cómo se hace, cuando digas

```bash
cd ~/src/foveal-vision
EXP=experimentos/2026-09-04-planas-sobre-preprocesado

.venv/bin/python $EXP/nn/construir_datasets.py --plan       # no escribe nada
.venv/bin/python $EXP/nn/construir_datasets.py --todos      # ~435 MB, unos minutos
.venv/bin/python $EXP/nn/construir_datasets.py --comprobar
```

⚠ **Estado real (2026-09-04):** `--plan` y `--comprobar` **sí** se han corrido y funcionan
(ninguno escribe nada). `--plan` da 434,6 MB los tres. La **construcción de verdad**
(`--todos`) **no se ha lanzado nunca**, así que esa parte sigue sin probar.

## Lo que falta por escribir

- `nn/red_local.py` — la plana sobre el mapa preprocesado: `Conv2d(1→1, 3, pad 0)` + cabeza.
  ⚠ No puede usar `FoveatedRegionalNN` tal cual: `_infer_flat_features` dimensiona la cabeza
  con `dims.N` (= 20) y aquí la entrada es 18/16/14, así que saldría una cabeza de 324 y
  reventaría al primer lote. Hace falta un `Dataset` propio y dimensionar con la forma real.
- `nn/entrenar_local.py`, `nn/avanzar.py`, `nn/comparativa.py` — se heredan casi enteros del
  experimento detenido, que ya los tiene probados (`--patience 0`, la negativa a declarar antes
  de la ép. 11, el lector único de `metrics.jsonl`).
- `instrucciones/02-criterio.md` — **escrito antes de mirar**, como manda R13. El criterio del
  detenido sirve de base: mismas anclas iso-features, misma banda de ruido 0,04, misma regla de
  no declarar antes de la época 11.

## Lo que ya se sabe y no hay que volver a pagar

- **El orden a 3 épocas no es fiable**, medido dos veces: en los gemelos (`1k3` es el peor a la
  ép. 3 y el mejor a la 37) y en el experimento detenido (la ventaja se desploma ÷6 entre la ép. 3
  y la 11).
- **La ventaja de un kernel congelado es de velocidad de convergencia, no de calidad** — medido en
  el detenido, con ReLU. Es la hipótesis a batir aquí.
- **La cabeza es el 97-99 % de los parámetros entrenables**, así que la comparación válida es
  contra el ancla iso-features y no contra la referencia.
