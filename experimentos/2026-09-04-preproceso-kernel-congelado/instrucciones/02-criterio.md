# El criterio, **escrito antes de correr los tres brazos**

> Congelado el 2026-09-04. En este momento no existe ningún run de este experimento: el único
> que llegó a crearse fue una prueba de humo del brazo `pre-1k5`, **borrada** antes de escribir
> esto.
>
> ⚠ **Lo que sí se vio en esa prueba, y se dice para no fingir un ciego perfecto:** dos épocas
> del brazo `pre-1k5`, f1 **0,322** (ép. 1) y **0,489** (ép. 2), y un coste de **44-45 s/época**.
> Por eso el criterio de aquí está escrito **en relativo** —contra los gemelos ya pagados— y no
> contra un número absoluto. Y por eso el desenlace «arranque más rápido» está declarado abajo
> **como la hipótesis a batir**, no como un hallazgo: ya sé que ocurre en las dos primeras épocas
> de uno de los tres.

## La pregunta

¿Sirve de algo alimentar una CNN plana con la salida de un kernel **ya aprendido y congelado**,
en vez de con los píxeles crudos?

## Los tres brazos, y el ancla de cada uno

Cada brazo es: `x (2,20,20) → [kernel congelado kf, padding 0] → ReLU → [conv 3×3 entrenable,
padding 0] → flatten → ReLU → Linear(→12)`. El kernel congelado es el `best.pt` del experimento
`1k<kf>-sinpadding`, y es literalmente su capa L1.

| brazo | mapa | features | entrenables | **ancla iso-features ya pagada** |
|---|---|---:|---:|---|
| `pre-1k3` | 16×16 | **256** | 3.142 | `1k5-sinpadding` → f1 **0,642** |
| `pre-1k5` | 14×14 | **196** | 2.422 | `1k7-sinpadding` → f1 **0,618** |
| `pre-1k7` | 12×12 | **144** | 1.798 | **ninguna** — ver «lo que no puede contestar» |
| *referencia (cruda)* | 18×18 | *324* | *3.967* | **es** `1k3-sinpadding` → f1 **0,680** |

Las anclas **no son una elección**: salen de que sin relleno cada preproceso recorta, y los
256/196 caen exactamente sobre los de dos gemelos ya corridos con el mismo dataset, semilla,
receta y stops. Es un control iso-features a coste cero.

## ⚠⚠ El confound que este experimento NO puede quitarse de encima, dicho antes

**Los cuatro brazos llegan a la cabeza con anchos distintos: 324 · 256 · 196 · 144.** Y esta
serie ya midió que en este régimen **manda el tamaño de la cabeza**: la cabeza se lleva el 97-99 %
de los parámetros entrenables en los cuatro. La tendencia medida sobre los seis gemelos es
≈ **0,09 de f1 por cada factor 2 de features**.

Por eso **la comparación válida no es «brazo contra referencia», es «brazo contra su ancla
iso-features»**. Comparar `pre-1k7` (144) con la referencia (324) mide sobre todo que una cabeza
es 2,25× la otra, que es un número que ya está pagado y no hace falta volver a comprar.

## ⚠⚠ Y por qué hay una ReLU en medio: sin ella el estudio sería degenerado

Dos convoluciones seguidas **sin activación entre medias son una sola convolución** de tamaño
`kf + 3 − 1`, con los pesos atados al producto de las dos. Medido, no argumentado:
`nn/red_local.py --comprobar` da diferencia máxima **7,2e-07** entre la composición y la conv
equivalente (redondeo de float32).

Sin ReLU, el espacio de funciones de cada brazo sería un **subconjunto estricto** del de un
gemelo ya corrido (`pre-1k3` ⊂ una 5×5 libre = `1k5`; `pre-1k5` ⊂ una 7×7 libre = `1k7`), o sea
que sólo podría empatar o perder contra un número que ya está en git. Con ReLU deja de serlo y la
pregunta pasa a tener respuesta desconocida. **La ReLU sale gratis del builder** (`_branch_forward`
activa entre capas y no tras la última), con sólo poner la capa congelada la primera.

## Los desenlaces, a **época fija**

Sea **A** el f1 del ancla iso-features del brazo, leído **a la misma época** que el brazo.
Banda de ruido **0,04**, la misma que fijaron los criterios del 1k5 y el 1k7 y que sale de la
oscilación de f1 en las últimas 9 épocas de los runs anteriores.

| si el f1 del brazo cae en | veredicto | qué significa |
|---|---|---|
| **> A + 0,04** | **el preproceso APORTA** | un kernel congelado y bien elegido bate a uno libre del tamaño efectivo equivalente. Sería el resultado que justifica seguir |
| **A ± 0,04** | **el preproceso es NEUTRO** | congelar L1 no cuesta ni da; lo que manda sigue siendo la cabeza. Desenlace más probable a juzgar por la serie |
| **< A − 0,04** | **el preproceso ESTORBA** | comprimir 2 canales a 1 con un kernel fijo tira información que la red libre sí usaba |

## El criterio de **cuándo se puede leer**, que aquí es la mitad del diseño

⚠⚠ **A 3 épocas NO se declara nada, y esto está medido en esta misma serie.** El orden a 3 épocas
está **invertido** respecto al final:

| | ép. 3 | ép. 11 | ép. 37 | |
|---|---:|---:|---:|---|
| `1k3` (324 f) | **0,099** ← el peor | 0,647 | **0,680** ← el mejor | se da la vuelta |
| `1k5` (256 f) | **0,384** ← el mejor | 0,595 | 0,642 | |
| `1k7` (196 f) | 0,200 | 0,524 | 0,618 | |

Un informe leído a la época 3 habría concluido *«1k3 es el peor de los tres»*, que es lo
contrario de lo que resultó. **El ranking se estabiliza hacia la época 11.**

Por eso:

1. **Ép. 3 — sólo se publica «arrancó / no arrancó»**, el reloj y la forma de la curva. Ningún
   veredicto de la tabla de arriba se emite aquí. Es el feedback gradual que pidió el dueño, y
   sirve para eso.
2. **Ép. 11 — primera lectura con veredicto**, marcada como provisional.
3. **Ép. 24 y 37 — lectura firme**, y sólo la de 37 entra en el reporte central.

⚠ **Y hay una razón concreta para esperar que los brazos arranquen rápido y eso NO signifique
nada:** su L1 ya viene entrenada, así que la cabeza empieza a aprender sobre features útiles en
vez de sobre ruido. La ventaja de las primeras épocas es **esperada por construcción**; lo que el
estudio pregunta es si sobrevive a la época 37.

## Lo que este experimento NO puede contestar, dicho antes

1. **Una semilla por brazo. Acota, no declara.** El suelo de ruido entre semillas medido en esta
   serie (~0,0007 en réplicas técnicas, hasta 0,04 entre épocas tardías) hace que diferencias
   menores que la banda no sean interpretables.
2. **`pre-1k7` (144 features) no tiene ancla iso-features.** Nadie ha corrido una plana libre de
   144. Su lectura se apoya en la tendencia extrapolada (≈0,578 esperado), y **una extrapolación
   no es una medida**: se marca como tal o no se lee.
3. **Se mueven DOS cosas a la vez**, como en todos los gemelos: el brazo cambia *qué* alimenta a
   la red **y** el tamaño de la cabeza. Las anclas iso-features separan lo primero de lo segundo,
   pero sólo para los dos brazos que tienen ancla.
4. **El canal de relleno deja de estar disponible por separado.** El kernel congelado consume
   `(vista, relleno)` y devuelve **un** mapa, así que aguas abajo la red ya no puede pesar el
   relleno por su cuenta. El reporte #19 midió que ese canal vale mucho (recall del último píxel
   0,608 → 0,974). Es una tercera cosa que se mueve, y no está controlada.
5. **Nada de esto mueve producción.** Mide una plana de una capa; la foveada tiene 4 capas.
   Instrucción del dueño (2026-09-03): un experimento no cambia el código de producción hasta que
   el número lo respalde.
