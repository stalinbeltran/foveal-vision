# Planas de verdad sobre datasets preprocesados **construidos antes de entrenar**

**Estado: los TRES DATASETS y las TRES ESTRUCTURAS de red están hechos** (2026-09-04).
Falta el entrenamiento.

| dataset | carpeta (repo de DATOS) | forma | en git | en crudo | muestra |
|---|---|---|---:|---:|---|
| `1k3` | `preprocesado/1k3-relu/` | (140000, 1, **18, 18**) | 34 MB | 181 MB | [`muestras/1k3.png`](muestras/1k3.png) |
| `1k5` | `preprocesado/1k5-relu/` | (140000, 1, **16, 16**) | 31 MB | 143 MB | [`muestras/1k5.png`](muestras/1k5.png) |
| `1k7` | `preprocesado/1k7-relu/` | (140000, 1, **14, 14**) | 33 MB | 110 MB | [`muestras/1k7.png`](muestras/1k7.png) |

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
  ventana 32×32 px → build_view → vista (2,20,20) → aplicaKernel_1k<kf> (pad 0, ReLU)
                                                  → mapa (1, 20−kf+1, ·)   ← EL DATASET

paso 2 (la red: CNN plana con los ÓPTIMOS de la foveada)
  mapa (1,H,W) → 4 × [Conv2d(→16, k=3, pad=1) + ReLU entre capas]
               → flatten → ReLU → Linear(→12) → (4 esquinas × 3)
```

### Los parámetros, y de dónde sale cada uno

Salen de **`estudios-redes-neuronales/ESTADO.md`**, sección «Red foveada». ⚠ El encargo decía
`reportes/README.md`, pero ése es el **historial cronológico** y él mismo avisa: *«el estado no
vive aquí: en qué quedó cada parámetro está en `../ESTADO.md`»*.

⚠⚠ **Y esa tabla tiene dos columnas que no siempre coinciden: «vigente» y «óptimo medido».** El
encargo pide los **óptimos**, así que es lo que se toma.

| parámetro | valor | evidencia en ESTADO.md |
|---|---|---|
| `n_layers` | **4** | cerrado: 2 → 0,9066 · 3 → 0,9246 · **4 → 0,9341** · 5 → 0,9136 |
| `k_center` | **3** | cerrado: 5 y 7 son peores **y más caros** |
| `channels` | **[16]×4** | cerrado 20/20: 24 y 32 no aportan, 8 hace daño. 16 es el suelo útil |
| `s_center` | **1** | no barrible: un solo valor legal con esta geometría |
| `dropout` | **0,0** | tanteo; 0,1 es el **peor** de los cuatro |
| `regions` | **single** | es lo que significa «plana» aquí (`plana-24-single.yaml`), no «una capa» |

**Los que NO se heredan** — y son exactamente *«los parámetros afectados por los datasets de
entrada»* del encargo: `fovea_px`, `border_px`, `border_reduce`, `overlap_*` describen cómo se
construye la **vista** desde la página, y aquí la entrada ya es un mapa preprocesado de tamaño
fijo. Esos mandos **ya se aplicaron** al construir el dataset — que usa la geometría de
`plana-20-1k3.yaml`, con los óptimos medidos `border_px: 8` y `overlap_fovea_px: 7`.
`k_periph`/`s_periph`/`mask_channel` tampoco: no hay rama periférica, y el canal de relleno lo
consumió el kernel congelado.

### ⚠⚠ El relleno de la convolución es `k//2`, como la foveada — NO 0

`builder.py:145` calcula `pad = k_center // 2`, así que *«los parámetros óptimos de la foveada»*
traen relleno **`same`**, y con él la resolución **no cae** por las capas: 18×18 sigue siendo
18×18 tras las cuatro.

Los siete gemelos usan `padding=0` —por eso se llaman `-sinpadding`—, pero eso era **el eje de
aquellos experimentos, no un óptimo medido**: ESTADO.md no tiene ninguna fila que diga que 0 gane.
⚠ La otra opción existe y **cambia el ancho de la cabeza 3,2×** (5.184 → 1.600 en el `1k3`); si
prefieres ésa, es un parámetro y se cambia en una línea.

### Las tres estructuras

```
$ .venv/bin/python experimentos/2026-09-04-planas-sobre-preprocesado/nn/red_local.py
```

| brazo | entrada | tras las 4 convs | features | convs | cabeza | **total** |
|---|---|---|---:|---:|---:|---:|
| `1k3` | (1, **18, 18**) | (16, 18, 18) | 5.184 | 7.120 | 62.220 | **69.340** |
| `1k5` | (1, **16, 16**) | (16, 16, 16) | 4.096 | 7.120 | 49.164 | **56.284** |
| `1k7` | (1, **14, 14**) | (16, 14, 14) | 3.136 | 7.120 | 37.644 | **44.764** |

**Las convoluciones son idénticas en las tres — 7.120 parámetros exactos — y lo único que cambia
es la cabeza.** No es una intención: es una invariante, y `red_local.py --comprobar` la verifica
capa por capa, además de contrastar la forma declarada contra el `.npz` real. 12 tests en
`tests/test_planas_preprocesado.py`.

⚠ **La cabeza sigue siendo el 84-90 % del modelo** (62.220 de 69.340), así que el confound de
siempre no desaparece: la comparación válida es a igualdad de ancho de cabeza, no entre brazos.

### ⚠ Un hueco que hay que tapar ANTES de entrenar: los escalares de borde

El `.npz` guarda `x`, `y` y `split` — **no `window_xy`**, así que `edge_features` (los 4 escalares
que le dicen a la cabeza si la ventana toca el borde de la página) **no se puede calcular al
entrenar**. Las redes se construyen con `n_edge=0`, o sea **sin** esa entrada, y la foveada sí la
lleva (`edge_inputs: pad`).

Se arregla añadiendo los 4 escalares —o `window_xy`— al dataset y reconstruyendo. Hay un test que
lo fija: si algún día se añaden, **falla** y obliga a mirar la cabeza.

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

## Dónde caen los datasets: **commiteados en el repo de datos**

**Orden del dueño (2026-09-04): «has push de los datasets».** Van a
[`foveal-vision-data/preprocesado/`](https://github.com/stalinbeltran/foveal-vision-data/tree/main/preprocesado),
con su manifiesto y su muestra al lado.

⚠ **Lo propuesto era lo contrario** —enlazarlos en vez de guardarlos, porque se rehacen
exactamente en 1 min 20 s cada uno y la regla 4 de [`../README.md`](../README.md) dice que lo
regenerable se enlaza—. El dueño decidió guardarlos; la excepción queda anotada con su motivo en
vez de darse por inaplicable.

⚠⚠ **Y caben SÓLO porque se comprimen.** GitHub rechaza cualquier fichero de más de **100 MB**, y
en crudo miden 180/144/112 MB — **los tres lo pasaban**. `savez_compressed` los deja en 34/31/33 MB
(~4×, sin perder un bit) porque el 12-15 % de las celdas son **cero exacto** (la ReLU) y el resto
es suave. Si alguien vuelve a `np.savez` a secas, **el push se rechaza**.

⚠ **Y no se bajó a float16**, que era la salida obvia para caber: habría cambiado el dato de
entrada del estudio (pérdida máxima medida **1,2e-04**). Comprimir no pierde nada y ya basta.

El pack del repo de datos pasa de 196,76 MiB a ~295 MiB.

⚠ **`--comprobar` distingue «no está» de «está y no casa con su manifiesto»**, que es el caso
peligroso: un `best.pt` reentrenado daría otro dataset con el mismo nombre. Por eso la huella se
calcula sobre los **pesos** del kernel, no sobre su nombre.

## Cómo se hace, cuando digas

```bash
cd ~/src/foveal-vision
EXP=experimentos/2026-09-04-planas-sobre-preprocesado

.venv/bin/python $EXP/nn/construir_datasets.py --plan       # no escribe nada
.venv/bin/python $EXP/nn/construir_datasets.py --todos      # ~4 min los tres
.venv/bin/python $EXP/nn/construir_datasets.py --comprobar
```

## Lo que falta por escribir

- **Los 4 escalares de borde en el dataset** (ver arriba): hoy las redes van con `n_edge=0`.
- `nn/dataset.py` — un `Dataset` que sirva el `.npz` preprocesado con su `split`.
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
