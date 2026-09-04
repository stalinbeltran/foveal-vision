# Planas de verdad sobre datasets preprocesados **construidos antes de entrenar**

**Estado: PLANIFICADO. No se ha construido ni entrenado nada** (2026-09-04). El dueño paró el
intento anterior y pidió que los datasets de entrada se construyan **antes** del entrenamiento.

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
| ReLU entre kernel y conv | sí | **no** (ver la decisión abierta) |

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

## ⚠⚠ La decisión que hay que tomar ANTES de construir nada: `--activacion`

`aplicaKernel` devuelve el mapa **sin activar y con signo** (es la capa L1 de esas redes, cuya
última capa no lleva ReLU a propósito). Guardarlo así o pasarle una ReLU **no es un detalle de
formato**: decide si el estudio puede salir a favor.

| | qué guarda el dataset | consecuencia |
|---|---|---|
| **`--activacion ninguna`** *(lo literal)* | el mapa con signo | la plana hace `conv(conv(x))` **sin no-linealidad en medio** = **una sola conv** de tamaño `kf+2`. Cada brazo es un **subconjunto estricto** de un gemelo ya corrido → sólo puede **empatar o perder** |
| **`--activacion relu`** | `max(0, mapa)` | no colapsa; el preproceso es un extractor de rasgos de verdad. Es lo que hacía el intento anterior |

Con `ninguna`, los equivalentes son exactos: `1k3`+plana ≡ una 5×5 atada (gemelo libre `1k5`,
0,642); `1k5`+plana ≡ una 7×7 atada (gemelo libre `1k7`, 0,618). **Sigue siendo una pregunta
legítima** —*«¿cuánto cuesta congelar y factorizar?»*— pero conviene saber antes de pagarla que
no puede salir a favor.

**No elijo por ti**: es tu decisión y va escrita en el manifiesto de cada dataset, porque dos
datasets con distinta activación no son comparables y el nombre del fichero no lo diría.

## ⚠ Y el canal de relleno se pierde, y está medido que vale

El kernel consume `(vista, relleno)` y devuelve **un** mapa, así que la plana ya no puede pesar
el relleno por su cuenta. El reporte #19 midió que ese canal sube el recall del último píxel de
**0,608 a 0,974**. `--con-relleno` lo conserva recortado como 2º canal; por defecto **no**, que
es la lectura literal del encargo. Es la segunda cosa que decidir.

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

.venv/bin/python $EXP/nn/construir_datasets.py --plan --activacion <ninguna|relu>
.venv/bin/python $EXP/nn/construir_datasets.py --todos --activacion <ninguna|relu>
.venv/bin/python $EXP/nn/construir_datasets.py --comprobar
```

⚠⚠ **`construir_datasets.py` está escrito pero NO EJECUTADO** (2026-09-04): el dueño pidió no
correr nada más. `--plan` no escribe nada y es por donde hay que mirarlo primero.

## Lo que falta por escribir

- `nn/red_local.py` — la plana sobre el mapa preprocesado. **No se puede escribir del todo hasta
  decidir `--activacion`**, porque de eso depende cuántos canales trae el dataset y si tiene
  sentido comparar contra los gemelos libres.
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
