# Preproceso con un kernel CONGELADO — un experimento, tres brazos

**¿Sirve de algo alimentar una CNN plana con la salida de un kernel ya aprendido y congelado, en
vez de con los píxeles crudos?**

Encargo del dueño del 2026-09-04, literal en
[`instrucciones/01-encargo.md`](instrucciones/01-encargo.md). El criterio, **escrito antes de
correr nada**, en [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md).

## ⛔ DETENIDO en la época 11 de 37 por orden del dueño (2026-09-04)

**No se avanza más, y sus números NO contestan el encargo.** El dueño detectó el fallo al leer la
tabla de estructura: *«las entradas de las cnn son siempre data 20x20???»*.

**El fallo, dicho claro: el encargo pedía TRES CNN PLANAS sobre TRES DATASETS PREPROCESADOS, y
esto son tres redes de DOS convoluciones sobre el dataset de siempre.** Una «plana» en esta serie
es **una** convolución + cabeza (así son los siete gemelos). Al meter el kernel congelado *dentro*
del modelo como capa 0:

- la entrada siguió siendo `(2,20,20)` en vez de ser el mapa preprocesado (18×18 · 16×16 · 14×14);
- las redes pasaron a tener **dos** convoluciones, o sea que dejaron de ser planas;
- y el preproceso dejó de ser un **paso previo** para ser **parte de la red**, que es justo la
  distinción que el encargo pedía.

La decisión de meterlo en el modelo está razonada en este README (coste, corrección del canal de
relleno) y **el razonamiento no era falso** — pero contestaba a otra pregunta. Un argumento
correcto sobre la pregunta equivocada sigue siendo un fallo.

⚠ **Lo que SÍ vale de aquí, y por qué no se borra.** Sus tres brazos son exactamente *«el mismo
preproceso PERO con una ReLU en medio»*, y su geometría y su ancho de cabeza (256 · 196 · 144)
son **idénticos** a los del experimento nuevo. O sea que queda como el **brazo con ReLU** de una
comparación que ahora tiene sentido:

| | preproceso | convs entrenables | entrada a la red | no-linealidad tras el kernel |
|---|---|---|---|---|
| **este (detenido)** | capa 0 del modelo, al vuelo | 1 (+1 congelada) | (2,20,20) | sí, en el forward |
| **el nuevo** | **dataset construido antes** | **1** (plana de verdad) | el mapa preprocesado | sí, **dentro de `aplicaKernel`** |

⚠⚠ **Y desde que la ReLU vive dentro de `aplicaKernel` (orden del dueño, 2026-09-04), los dos
calculan EXACTAMENTE LA MISMA FUNCION**: `relu(kernel ⊛ x)` → conv entrenable → flatten → ReLU →
cabeza. Lo único que cambia es **dónde** ocurre el preproceso —al construir el dataset o en cada
forward— y que el nuevo entrena una red de **una** convolución en vez de dos.

Eso tiene dos consecuencias que conviene no confundirse:

1. **Los números de aquí son una PREVISION de los del nuevo**, no evidencia independiente. Mismo
   dataset, misma receta, misma semilla y misma función; sólo cambiará la inicialización de la
   conv entrenable, así que se espera lo mismo dentro de la banda de ruido.
2. **El nuevo será ~33 % más rápido por época**, porque la convolución congelada deja de
   recalcularse en cada época (aquí costaba 44-46 s/época contra los 32-34 de los gemelos).

Y su hallazgo se sostiene solo: **la ventaja del kernel congelado se desploma ÷6 entre la época 3
y la 11**, o sea que es velocidad de convergencia y no calidad. Eso vale para el nuevo también.

→ **El experimento que contesta el encargo es
[`2026-09-04-planas-sobre-preprocesado/`](../2026-09-04-planas-sobre-preprocesado/).**

## Lo que quedó medido antes de parar (época 11 de 37)

```
$ .venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/comparativa.py
```

| brazo | feat. | ép. 3 | ép. 11 | ép. 24 | ép. 37 | ancla iso-features | Δ misma época |
|---|---|---|---|---|---|---|---|
| pre-1k3 | 256 | 0.569 | 0.625 | — | — | 1k5 crudo | **+0.030** → NEUTRO |
| pre-1k5 | 196 | 0.509 | 0.579 | — | — | 1k7 crudo | **+0.055** → aporta |
| pre-1k7 | 144 | 0.449 | 0.542 | — | — | — | sin ancla |

*referencia cruda (324 feat., **37 épocas**, ya corrida y en git): f1 **0,680***

### ⚠⚠ Lo que hay que llevarse de aquí: **la ventaja se DESPLOMA**

| | ép. 3 | ép. 11 | |
|---|---:|---:|---|
| `pre-1k3` vs `1k5 crudo` | **+0,186** | **+0,030** | ÷6,2 |
| `pre-1k5` vs `1k7 crudo` | **+0,309** | **+0,055** | ÷5,6 |

**Es exactamente lo que el criterio escribió antes de mirar**: la L1 congelada ya viene
entrenada, así que la cabeza arranca sobre features útiles en vez de sobre ruido — y esa ventaja
es de **velocidad de convergencia**, no de calidad. El kernel libre la alcanza en cuanto tiene
épocas para aprender el suyo.

⚠ **`pre-1k5` sale «aporta» por +0,055 contra una banda de ±0,04: está a un pelo del ruido y la
tendencia va a la baja.** Leerlo hoy como «el preproceso funciona» sería justo el error que este
diseño intenta evitar. El criterio dice que la ép. 11 es **provisional** y que sólo la 37 entra en
el reporte central; a este ritmo, lo esperable es que los dos brazos con ancla acaben en NEUTRO.

⚠ **Y a la época 3 el veredicto habría sido el contrario** (+0,186 y +0,309, o sea «aporta» con
holgura en los dos). Ésa es la razón de que `comparativa.py` se niegue a declarar antes de la 11,
y de que ni siquiera la 11 sea firme.

## Qué es cada brazo

```
x (2,20,20)  →  [kernel CONGELADO kf, padding 0]  →  ReLU  →  [conv 3×3 entrenable, padding 0]
             →  flatten  →  ReLU  →  Linear(→ 12)
```

El kernel congelado es el `best.pt` del experimento `1k<kf>-sinpadding` — literalmente su capa
L1, comprobado a diferencia **0,0** contra el `mapas.npy` que aquel dejó escrito
(`comun/preproceso.py --comprobar`).

| brazo | kernel | mapa | features | entrenables | congelados |
|---|---|---|---:|---:|---:|
| `pre-1k3` | 3×3 de `1k3` | 20→18→16 | 256 | 3.142 | 19 |
| `pre-1k5` | 5×5 de `1k5` | 20→16→14 | 196 | 2.422 | 51 |
| `pre-1k7` | 7×7 de `1k7` | 20→14→12 | 144 | 1.798 | 99 |

## ⚠⚠ La ReLU de en medio es lo que hace que este experimento exista

Dos convoluciones seguidas **sin activación entre medias son una sola convolución** de tamaño
`kf + 3 − 1`. Medido, no argumentado: `nn/red_local.py --comprobar` da diferencia máxima
**7,2e-07** entre la composición y su equivalente de una sola capa (redondeo de float32).

Sin ReLU, cada brazo sería un **subconjunto estricto** de un gemelo ya corrido —`pre-1k3` ⊂ una
5×5 libre (= `1k5`), `pre-1k5` ⊂ una 7×7 libre (= `1k7`)— o sea que sólo podría **empatar o
perder** contra un número que ya está pagado. Con ReLU deja de serlo. La ReLU sale gratis del
builder: `_branch_forward` activa entre capas y no tras la última, así que basta con que la capa
congelada no sea la última de la rama.

## ⚠ Y el confound que NO se quita: los cuatro brazos tienen cabezas distintas

324 · 256 · 196 · 144 features, y la cabeza es el **97-99 %** de los parámetros entrenables en
los cuatro. Esta serie ya midió que aquí **manda el tamaño de la cabeza** (≈0,09 de f1 por cada
factor 2). Por eso la comparación válida **no** es contra la referencia cruda, sino contra el
**ancla iso-features**: el gemelo con exactamente esas features, mismo dataset, semilla y receta.
Los 256 y 196 caen justo sobre `1k5` y `1k7`, así que dos de los tres brazos tienen su control
**a coste cero**. El de 144 no lo tiene, y su lectura es más débil — está dicho en el criterio.

## Cómo se sigue (es reanudable: ése era medio encargo)

```bash
cd ~/src/foveal-vision
.venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/avanzar.py --hasta 24
.venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/comparativa.py
```

Stops: **3 · 11 · 24 · 37**, los mismos que los siete gemelos — que caigan en las mismas épocas
es lo que permite ponerlos uno al lado del otro sin explicar nada.

⚠⚠ **`avanzar.py` mueve LOS TRES y se niega si se desincronizan.** No es cortesía: `loop.py:376`
corta por `patience`, y `no_improve` se **restaura** de `last.pt`, así que un brazo que ya tocó
`patience` correría **una** época aunque le pidas trece — y la tabla compararía la ép. 24 de uno
contra la 37 de otro sin que nada fallara. Por eso los tres se crean con **`--patience 0`** (queda
congelado en su `config.json`) y `avanzar.py --comprobar` sale con código ≠ 0 si no coinciden.

⚠ **Hay que reanudar por `entrenar_local.py`, no con `fv-continue` a pelo**: el brazo congelado se
monta en el parche en memoria. Con `fv-continue` suelto se construiría una plana normal (cabeza de
324) y el `state_dict` no casaría. Falla ruidosamente, que es lo que se quiere, pero el camino
bueno es éste.

⚠ **La capa congelada se comprueba al terminar cada tramo**, no se da por hecho: `entrenar_local.py`
contrasta el kernel del checkpoint contra `comun/preproceso.py`. Un `requires_grad=False` que
alguien deshaga sin querer convierte esto en otro experimento —uno donde L1 también entrena— y los
números seguirían saliendo igual de creíbles.

## Coste, medido

**0 máquinas alquiladas · 0 $ · CPU de este dev.** 11 épocas × 3 brazos = **~25 min** acumulados (los 3 primeras épocas fueron ~7 min; el tramo 3→11, ~18 min).

⚠ Un brazo cuesta **44-46 s/época** contra los **32,4-34,1 s** de los gemelos (*medido
2026-09-04, este droplet de 2 vCPU, máquina descargada*): **+33 %** por la convolución congelada
extra. No es el +1,7 % que sugiere una estimación por lote — está medido.

### Por qué NO se fue a Vast, que es lo que pedía el encargo

Los siete gemelos corrieron aquí, 0 $. Pero el motivo que pesa no es el dinero:
`scripts/entrenar_vast.py:60-64` avisa de que *«un run continuado en OTRA maquina no es bit a bit
el mismo… Para entrenar un modelo da igual; para publicar una tabla comparable, no»*. El encargo
pide justo eso que se rompe —**reanudar por tramos** y **una tabla comparativa única**— y encima
compararse con siete gemelos ya medidos en esta CPU. Alquilar metería una variable de máquina en
el único eje que el estudio quiere aislar.

**Si hubiera que ir a Vast igualmente:** `scripts/entrenar_vast.py --name plana-pre1k5-s1
--continuar`, lanzado con `desacoplar-persistente.sh` (R11: quien enciende tiene que poder
apagar), un brazo por instancia. Y entonces la tabla tiene que decir en qué máquina corrió cada
tramo, porque deja de ser comparable bit a bit.

## Los «3 datasets pre-procesados»: son la capa 0 del modelo, no tres ficheros

El `windows.npz` no guarda ventanas —guarda las 1.000 imágenes y las posiciones— y la vista 20×20
la construye `FoveatedWindowDataset.__getitem__` al vuelo. Materializar las 140.000 ventanas ya
convolucionadas son **435 MB** en float32, contra un repo de datos de **197 MiB**, para guardar
algo que está **medido** como re-derivable exactamente. La regla 4 de
[`../README.md`](../README.md) dice qué hacer con eso: *«lo que no se puede regenerar, se guarda;
lo que sí, se enlaza»*.

Y hay un argumento de **corrección**, no sólo de coste: `aplicaKernel` sobre una imagen suelta
rellena el segundo canal con el defecto declarado `relleno=0`; dentro del modelo el tensor llega
como `(B,2,20,20)` con la **cobertura real** del dataset, que es exactamente lo que vio la L1
original. Un dataset materializado tiene que reproducir eso a mano, y equivocarse ahí **no falla:
sale otro número**.

Lo observable que sustituye a los ficheros: `nn/red_local.py --comprobar` (geometría, congelación
y el colapso sin ReLU) y `comun/preproceso.py --comprobar` (que el kernel es la L1 de su red).

**Si hiciera falta materializarlos igualmente**, va como caché **no commiteada** con su huella
`sha256(kernel ‖ fingerprint del dataset ‖ opciones)`, que se niega si no casa — no dentro de
`window-datasets/`, donde `check_compatible` lo rechaza porque el manifest exige `images` +
`window_xy`. Compraría ~5× de reloj, y hoy eso son 7 minutos.

## Cómo se comprueba que esto está bien

```bash
cd ~/src/foveal-vision
.venv/bin/python experimentos/comun/preproceso.py --comprobar         # dif 0,0 en los tres
.venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/red_local.py --comprobar
.venv/bin/python experimentos/2026-09-04-preproceso-kernel-congelado/nn/avanzar.py --comprobar
.venv/bin/python -m pytest -q tests/test_preproceso.py                # 37, el contrato del kernel
```

## Lo que falta

- **Llegar a la época 11**, que es la primera con veredicto (y luego 24 y 37).
- **El brazo `pre-1k7` no tiene ancla iso-features** (nadie ha corrido una plana libre de 144).
- **Una semilla por brazo**: acota, no declara.
- **El reporte final va al repo central** `estudios-redes-neuronales/reportes/estudios/2026/
  09-septiembre/`, con su fila al final de `reportes/README.md`, y **0 instancias · 0 $ · CPU de
  este dev** escrito explícitamente en vez de dejar el hueco. Se escribe cuando haya veredicto
  (ép. 37), no ahora.
  ⚠ Los siete gemelos de esta serie **tampoco tienen reporte central todavía**: es una deuda
  anterior a este experimento, y lo suyo sería cerrarla con un solo reporte de la serie entera.
