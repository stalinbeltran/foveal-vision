# Planas de verdad sobre datasets preprocesados **construidos antes de entrenar**

## ⚠⚠ ENTRENADO a la época 3 — y el resultado es un **COLAPSO**, no una tabla

**Los tres brazos dan f1 = 0,000.** La red predice «no hay esquina» en todas partes, que con un
13,2 % de esquinas positivas es un mínimo local cómodo. La pérdida sí baja (1,04 → 0,80) y ahí se
queda.

⚠ **Esto NO dice nada sobre el preproceso.** Dice que **este régimen no da para la tarea** — y era
el desenlace nº 2 que el [criterio](instrucciones/02-criterio.md) escribió *antes* de mirar:
*«si los tres brazos salen mal a la vez, el desenlace más probable no es "el preproceso no sirve"
sino "este régimen no da para esta tarea"»*.

### El suelo de coste, aislado: **son los CANALES, no el stride ni las capas**

Mismo `.npz`, misma receta `plan40`, mismas 3 épocas. Sólo cambia la red *(medido 2026-09-04,
este droplet)*:

| variante | params | features | **f1 @ép. 3** |
|---|---:|---:|---:|
| **la mínima** (2 capas · **2 can.** · s=2) | 286 | 18 | **0,000** |
| stride 1 (2 capas · **2 can.** · s=1) | 4.774 | **392** | **0,000** |
| 3 capas (3 capas · **2 can.** · s=2) | 132 | 2 | **0,000** |
| **16 canales** (2 capas · **16 can.** · s=2) | 4.220 | 144 | **0,544** |
| control (4 capas · 16 can. · s=1) | — | — | **0,688** |

**Lo decisivo es la segunda fila contra la cuarta:** `stride 1` tiene **más** parámetros (4.774 vs
4.220) y **2,7× más features** (392 vs 144) — y da 0,000 mientras la de 16 canales da 0,544. O sea
que **no es capacidad en parámetros ni anchura de cabeza: son los canales**.

**Se pueden conservar las 2 capas, el stride 2 y el sin-relleno.** El único mando que hay que
soltar es `channels`.

### Lo que sí quedó verificado

- El **dataset y el camino de entrenamiento son correctos**: el mismo `.npz` con una red mayor da
  f1 0,688 a la época 3. Si el fallo fuera de datos, ninguna red aprendería.
- **7,3-8,6 s/época** contra los 44-46 s de la versión de 4 capas: los tres brazos a la época 3 en
  **1 min 36 s**, 0 $.
- La reanudación incremental está montada (`avanzar.py --hasta N`, `--patience 0`).

**Estado: los 3 datasets, las 3 estructuras y el entrenamiento incremental están hechos.**

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

### Los parámetros — **redefinidos el 2026-09-04 para abaratar**

> *«Como queremos resultados preliminares, vamos a reducir el "costo" al mínimo. Las capas de la
> nn serán 2, por ahora. Cada capa va a tener solo 2 canales. El padding del kernel va a ser
> siempre 'sin padding'. Cada capa va a reducir el tamaño de los features. Además, el stride va a
> ser la mitad del ancho del kernel (redondeado).»*

| parámetro | valor | de dónde sale |
|---|---|---|
| `n_layers` | **2** | el dueño, 2026-09-04 *(antes 4, el óptimo foveado)* |
| `channels` | **[2]×2** | el dueño, 2026-09-04 *(antes [16]×4)* |
| `padding` | **0** — sin relleno | el dueño, 2026-09-04 |
| `stride` | **2** | «la mitad del kernel, redondeado» → `(3+1)//2` |
| `k` | **3** | ⚠ **no es una elección: es lo único que cabe** (abajo) |
| `dropout` | **0,0** | ESTADO.md, sigue siendo el óptimo y no se movió |
| `regions` | **single** | es lo que significa «plana» aquí |

#### ⚠⚠ `k` no lo dijiste, y resulta que no hacía falta: sólo k=3 sobrevive

Con 2 capas, sin relleno y stride = mitad de `k`, el mapa se agota antes de la segunda capa para
cualquier `k` mayor. Medido con `n₂ = (n − k)//s + 1`:

| | `1k3` 18×18 | `1k5` 16×16 | `1k7` 14×14 | |
|---|---|---|---|---|
| **k=3, s=2** | 18→8→**3** | 16→7→**3** | 14→6→**2** | ✅ las tres viven |
| k=5, s=3 | 18→5→1 | 16→4→**0** | 14→4→**0** | ❌ mueren dos |
| k=7, s=4 | 18→3→**0** | 16→3→**0** | 14→2→**−1** | ❌ mueren las tres |

Así que `k=3` está **forzado por tu propia definición**, no heredado del óptimo foveado — aunque
coincida con él.

⚠ **«Redondeado» lo leo como media hacia arriba**: `(k+1)//2` = 2. Truncar (`k//2` = 1) daría
18→16→14 y una cabeza de **392** features en vez de 18 — 20× más. Si querías truncar, dilo.

### Las tres estructuras

```
$ .venv/bin/python experimentos/2026-09-04-planas-sobre-preprocesado/nn/red_local.py
```

| brazo | entrada | tras conv0 | tras conv1 | features | convs | cabeza | **total** |
|---|---|---|---|---:|---:|---:|---:|
| `1k3` | (1, **18, 18**) | (2, 8, 8) | (2, 3, 3) | **18** | 58 | 228 | **286** |
| `1k5` | (1, **16, 16**) | (2, 7, 7) | (2, 3, 3) | **18** | 58 | 228 | **286** |
| `1k7` | (1, **14, 14**) | (2, 6, 6) | (2, 2, 2) | 8 | 58 | 108 | **166** |

**Las convoluciones son idénticas en las tres — 58 parámetros exactos — y lo único que cambia es
la cabeza.** `red_local.py --comprobar` lo verifica capa por capa; 14 tests en
`tests/test_planas_preprocesado.py`.

#### ⚠ El coste baja 242×, y eso tiene un precio que hay que decir

De **69.340** parámetros a **286**. Es lo que pediste, pero con 18 features llegando a una cabeza
de 12 salidas, el riesgo real es que un resultado malo mida **«la red es demasiado pequeña»** y no
«el preproceso no sirve». Un preliminar acota, no declara — y menos éste.

#### ⚠ Lo bueno que trae, y no se buscaba: casi desaparece el confound de la cabeza

Con k=3 y stride 2, el `1k3` y el `1k5` caen **los dos en 18 features**. O sea que esos dos brazos
son **iso-features por construcción**, que es justo lo que en la versión de 4 capas había que
corregir con anclas externas. El `1k7` se queda en 8 y sigue sin ancla.

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
