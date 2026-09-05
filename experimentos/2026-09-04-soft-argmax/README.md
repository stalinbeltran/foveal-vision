# `soft-argmax`: leer las coordenadas como esperanza sobre un mapa de calor

**Encargo del dueño, 2026-09-04.** Mismo cuerpo que `plana-4k7`, mismo dataset **no
preprocesado**, y la cabeza de coordenadas sustituida: en vez de regresar `x, y` con una
`Linear`, produce un mapa de calor por esquina, lo pasa por un softmax y toma la **esperanza**
de la posición. El encargo y las siete decisiones, en
[`instrucciones/01-encargo.md`](instrucciones/01-encargo.md); el criterio, **congelado y
empujado a `main` antes de la primera época**, en
[`instrucciones/02-criterio.md`](instrucciones/02-criterio.md).

## ✅ Qué salió: el error de posición cae **3,9×**, y NO es la pila conv

2026-09-04 23:39 → 2026-09-05 04:07 UTC · **4 h 28 min** · **0 máquinas · 0 $** (el coste es
reloj de este droplet) · 3 brazos × 37 épocas · semilla 1 · `dirty1000-80px-16px-r20260827`.

Todos en la época que guarda su `best.pt` (elegida por `val_loss`, la misma regla en los cuatro):

| | época | f1 | **`pos_err_px`** | borde | interior | borde/int |
|---|---:|---:|---:|---:|---:|---:|
| **ancla** [`plana-4k7-s1`](../2026-09-03-cnn-plana-4k7/) | 34 | 0,8398 | **2,224** | 3,088 | 1,961 | 1,57× |
| **A** soft-argmax | 37 | 0,8657 | **0,571** | 0,940 | 0,459 | 2,05× |
| **B** soft-argmax + dispersión (λ=0,1) | 34 | 0,8576 | **0,559** | 0,838 | 0,474 | 1,77× |
| **C** *control*: misma pila, lectura `Linear` | 37 | 0,8533 | **1,414** | 2,094 | 1,207 | 1,74× |

`pos_err_px` va en píxeles de la ventana etiquetada (16 px) y se mide sobre las **14.724
esquinas verdaderas** de validación — **las mismas en los cuatro**, porque la máscara es
`exists_true`, la verdad del terreno, no lo que cada red predice.

**Se cumple el primer desenlace del criterio: A baja más de 0,15 px *y* le gana también a C.**
Y el resultado se descompone limpio, porque A y C comparten todo menos la lectura:

```
   ancla  2,224 px
     │  −0,810   ← la PILA CONV (C: mismos 4 mapas, leídos por una Linear global)
   C     1,414 px
     │  −0,843   ← la LECTURA soft-argmax (A: los mismos mapas, leídos como esperanza)
   A     0,571 px
```

Las dos mitades son casi iguales y **se componen**. La pila sola ya vale, pero el soft-argmax
vale otro tanto **encima de ella** — y lo hace con **1,66× menos cabeza** que C (19.289 contra
32.096 parámetros), que se le dio a propósito de más para que este resultado no pudiera ser de
tamaño.

⚠ **Como referencia lejana**: la foveada de producción `fov16-mask-p20` —4 capas, sobre este
**mismo** dataset y esta misma métrica— llega a **1,120 px**. A la baja con una plana de **una**
capa. ⚠ Pero detecta mucho peor (f1 0,954 contra 0,866): son dos cosas distintas y `pos_err_px`
sólo mide la primera.

## ⚠⚠ El control FALLÓ, y hay que leerlo antes que el titular

El criterio pedía que el f1 no se moviera más de ±0,01 respecto al ancla, porque `exists` no se
toca. **Se movió en los tres brazos**: A +0,0259 · B +0,0178 · C +0,0135.

**Pero se movió también en C, que no lleva soft-argmax.** O sea que lo que mueve el f1 es la
**pila conv** (y con ella la co-adaptación del cuerpo, que es compartido y se entrena), no la
lectura. Aun así, A y C difieren +0,0124 en f1, que sigue pasando el umbral: **no se puede
afirmar que el soft-argmax deje `exists` intacto.**

⚠ **Lo que el fallo del control NO contamina es el titular**, y ésa es la parte que importa:
`pos_err_px` se promedia sobre las esquinas marcadas por `exists_true`, así que **las cuatro
redes se evalúan sobre exactamente el mismo conjunto de 14.724 esquinas**. Que una detecte más
o menos no cambia ni una de las que entran en la media. La comparación de posición es válida
aunque la de detección no lo sea.

## El sesgo de contracción existe, está medido, y lo tapa la ganancia

Era el riesgo estructural del diseño: el 24,1 % de las esquinas cae en el **primer o último
píxel** de la ventana, y la esperanza de un softmax llega mal a los extremos.

**Se ve, en el sitio donde tenía que verse.** El ancla ya sufre en el borde (1,57× su interior);
A sufre **más** (2,05×). O sea que sí, la lectura por esperanza penaliza el borde de más.

⚠ **Y aun así el borde de A (0,940 px) es mejor que el INTERIOR del ancla (1,961 px).** El sesgo
es real y queda **completamente tapado** por la ganancia. Habría sido invisible sin el desglose:
mirando sólo el agregado, «A mejora 3,9×» y ya.

Que no sea peor se debe en parte a la decisión 4 —la rejilla abarca la **vista entera**, o sea
`[-0,375 · 1,375]` en unidades de fóvea, y no `[0,1]`—, así que el softmax tiene masa que
colocar *fuera* del objetivo. **No está medida la variante con la rejilla recortada**, así que
cuánto de esto lo aporta esa decisión es una conjetura razonada, no un número.

## B (el regularizador de dispersión): **inerte en el agregado, pero REDISTRIBUYE**

| | `pos_err_px` | borde | interior | β aprendida |
|---|---:|---:|---:|---:|
| A | 0,571 | 0,940 | 0,459 | 1,695 |
| B (λ=0,1) | 0,559 | **0,838** | 0,474 | **2,046** |

La diferencia global es **−0,012 px**, muy por debajo del umbral de 0,15: por el criterio,
**B ≈ A** y el regularizador no declara nada.

⚠ **Pero el desglose va justo por donde el mecanismo predice**: concentrar el mapa mejora el
**borde** un 11 % y empeora el **interior** un 3 %, y la β aprendida sube de 1,695 a 2,046 (mapas
más picudos). Es coherente de principio a fin — y **es una semilla y un solo λ**: acota, no
declara. Si el borde llega a importar, es el eje que hay que barrer.

## Las figuras

| | |
|---|---|
| [`resultados/mapas-de-calor-A-con-esquinas.png`](resultados/mapas-de-calor-A-con-esquinas.png) | los 4 mapas de probabilidad sobre 10 ventanas **con esquinas**, con la verdadera (círculo verde) y la predicha (cruz azul) |
| [`resultados/mapas-de-calor-A.png`](resultados/mapas-de-calor-A.png) | las mismas 10 ventanas **compartidas** de [`comun/`](../comun/), para poder ponerlo al lado del resto de la serie |
| ídem `-B` | el gemelo con regularizador: mapas visiblemente más concentrados |

⚠ **En una ventana SIN esquina la posición no está supervisada** (`corner_loss` la multiplica por
`exists_true`, `losses.py:19`), así que ahí la cruz azul no significa nada. Por eso hay un set
propio: de las 10 compartidas sólo 2 traen alguna esquina, y la figura salía casi vacía. El set
propio está congelado en [`evaluacion/set-con-esquinas.json`](evaluacion/set-con-esquinas.json)
con su regla de selección, para que dos figuras se puedan comparar.

## ⚠⚠ Y el precio, que NO es de parámetros: **+0,2 % de params, ×266 de CÓMPUTO**

Ésta es la trampa del diseño, y no se ve en ninguna tabla de tamaño:

| | params de cabeza | **MACs de cabeza / ventana** |
|---|---:|---:|
| ancla `Linear(1604, 12)` | 19.260 | **19.248** |
| A/B (pila conv → mapas) | 19.289 (+0,2 %) | **5.126.416 (×266)** |

Una `Linear` global se aplica **una vez** por ventana; una pila convolucional se aplica **en cada
una de las 400 posiciones**. La red entera pasa a costar **×30**, y eso se midió con el reloj:
**149-161 s/época** contra los **38,9** del ancla, en esta misma máquina.

**Igualar parámetros no iguala cómputo**, y para un eje que se lea como «cabeza» la intuición
falla en la dirección cara. Si esto llega a producción, es el número que manda.

## Lo que este experimento NO contesta

1. **Una semilla por brazo.** El umbral (0,15 px) sale de la dispersión época a época del ancla
   —un **suelo**, no el ruido entre réplicas, que sigue sin medirse—. La ganancia de A es
   **11× ese suelo**, así que no es fragilidad de umbral; pero «acota, no declara» sigue
   valiendo.
2. **λ = 0,1 no está optimizado**, sólo *dimensionado* (8,3 % del término de posición al arrancar).
3. **El cuerpo es compartido y se entrena**: a las 37 épocas cada brazo tiene su propio cuerpo.
   Es lo que hace que el control de f1 no pueda ser limpio.
4. **Nada de esto se ha probado en la foveada de producción**, que es de 4 capas. Esta serie vive
   entera en planas de una capa.
5. **La rejilla extendida no tiene control.** No se corrió la variante recortada a `[0,1]`, así
   que su aportación al resultado del borde está razonada, no medida.

## Cómo repetirlo

```bash
cd ~/src/foveal-vision
.venv/bin/python experimentos/2026-09-04-soft-argmax/nn/red_local.py --comprobar   # la red y su presupuesto
COORD_HOME="$HOME/src/telegram-coordinator" \
  "$COORD_HOME/scripts/desacoplar-persistente.sh" soft-argmax \
  experimentos/2026-09-04-soft-argmax/nn/cadena.sh                                 # los 3 brazos (~4,5 h)
.venv/bin/python experimentos/2026-09-04-soft-argmax/nn/leer_criterio.py           # la lectura
.venv/bin/python experimentos/2026-09-04-soft-argmax/nn/mapas_de_calor.py          # las figuras
```

`cadena.sh` es **reanudable** (salta los brazos ya hechos), que es lo que le permite vivir en una
unidad con `Restart=on-failure` sin entrar en el bucle que este proyecto ya pagó dos veces.
`src/fv/` **no se toca**: el parche vive en memoria y se comprueba al final de cada corrida.
