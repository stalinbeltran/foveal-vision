# El criterio, **escrito antes de correr nada**

> Congelado el 2026-09-04, **antes** de la primera época de los tres brazos. Lo que sí se conocía
> al escribirlo —y se dice, en vez de fingir un ciego que no existe— son las curvas del ancla
> `plana-4k7-s1` y de los otros cinco de la serie plana, que llevan commiteadas desde el 03 y el
> 04 de septiembre, y las tres medidas sobre el `windows.npz` que están más abajo.

## Lo que este experimento pregunta

**¿Leer las coordenadas como esperanza sobre un mapa de calor (soft-argmax) baja `pos_err_px`
frente a regresarlas con una `Linear`, a igualdad de cuerpo y de presupuesto de cabeza?**

No pregunta si la red detecta mejor: `exists` no se toca.

## El ancla, y por qué esta época y no otra

`plana-4k7-s1`, mismo cuerpo, mismo dataset, misma semilla, misma receta. Se compara **la época
que `best.pt` guarda**, que `plan40` elige con `monitor: val_loss` — la misma regla en los cuatro,
que es lo que la hace una comparación y no una elección:

| | época | f1 | **pos_err_px** | val_loss |
|---|---:|---:|---:|---:|
| **ancla `plana-4k7-s1`** | 34 | **0,8398** | **2,224** | 0,2297 |
| *(su última época, 37)* | 37 | 0,8352 | 2,311 | — |
| *(referencia lejana: la foveada de producción `fov16-mask-p20`)* | 62 | 0,9542 | 1,120 | — |

## El suelo de ruido, medido

No hay segunda semilla de esta plana, así que **no se puede dar el ruido entre réplicas**. Lo que
sí se puede dar es un **suelo**: la dispersión época a época del ancla ya estabilizada.

| últimas N épocas del ancla | `pos_err_px` sd | f1 sd |
|---|---:|---:|
| 5 | **0,068** | 0,0038 |
| 10 | 0,140 | 0,0051 |

⚠ **Es un suelo, no el ruido de verdad**: la variación entre semillas es de otra clase y aquí
está **sin medir**. Un resultado que no despegue de este suelo, desde luego no declara nada.

Umbrales que salen de ahí (≈2 sd de la ventana de 5):

- **mueve `pos_err_px`** si la diferencia con 2,224 supera **0,15 px**;
- **el f1 NO se ha movido** si se queda dentro de **±0,01** de 0,8398.

## Las tres medidas del `windows.npz` que condicionan el diseño

*Medidas el 2026-09-04 sobre `dirty1000-80px-16px-r20260827`, 140.000 ventanas / 72.380 esquinas
verdaderas:*

1. **0** esquinas fuera de `[0,1)`; el rango va de `0,000000` a `0,999375`.
2. **24,1 %** de las esquinas cae en el **primer o último píxel** de la ventana de 16; el 44,8 %
   en los dos de fuera. Ésta es la razón de la decisión 4: una rejilla que abarcara exactamente
   `[0,1]` haría **inalcanzable por construcción** a un cuarto del dato — el mismo tipo de fallo
   que la rejilla de λ de la sonda L1.
3. La media del objetivo es `x = 0,5002 · y = 0,4932`, y el centroide de la rejilla elegida es
   **exactamente 0,5**: el brazo A sin entrenar predice `(0,4992 · 0,4993)`. Arranca en la media,
   no en un sitio arbitrario.

## Los tres brazos

| | cabeza `(x,y)` | params de cabeza | vs ancla |
|---|---|---:|---:|
| **ancla** `plana-4k7-s1` | `Linear(1604, 12)` entera | 19.260 | — |
| **A** `sargmax-a-s1` | pila conv → 4 mapas → softmax(β) → esperanza | 19.289 | **+0,2 %** |
| **B** `sargmax-b-s1` | igual que A, + penalización de la dispersión (λ=0,1) | 19.289 | +0,2 % |
| **C** `sargmax-c-s1` | **la misma pila, los mismos 4 mapas**, leídos por `Linear` global | 32.096 | +66,6 % |

**C es el control que hace legible el experimento.** Sin él, si A gana no se sabría si gana el
soft-argmax o gana la pila conv que se le añadió debajo. ⚠ Y C se deja **con más cabeza que
nadie, a propósito**: si A le gana con 19.289 parámetros contra 32.096, la ventaja no puede ser
de tamaño. Es un control generoso, y por tanto adverso en la dirección correcta.

## La predicción

**`pos_err_px` de A en `[1,90 · 2,40]`, y el desenlace más probable es «no se mueve».**

De dónde sale la banda, y por qué no es optimista:

- **A favor**: la esperanza sobre un mapa es el sesgo inductivo del problema («¿dónde está?»), y
  la pila añade campo receptivo (7×7 sobre el 7×7 del cuerpo → 13×13 efectivo) donde el ancla
  sólo tiene una lectura sin estructura espacial.
- **En contra, y pesa más**: la `Linear` global del ancla **ya conserva el «dónde» perfectamente**
  —un peso libre por cada (canal, posición) sobre un mapa de 20×20—, así que aquí no hay el cuello
  de botella que el soft-argmax resuelve en las arquitecturas donde se hizo famoso (donde la
  alternativa es un *pooling* global que destruye la posición).
- **Y el sesgo de contracción sigue existiendo**, aunque la rejilla extendida lo alivie: llegar a
  `u = 0` exigiendo masa a la izquierda del objetivo es más difícil que emitir un número.

## Los desenlaces, y qué significa cada uno

| desenlace | veredicto |
|---|---|
| **A baja `pos_err_px` en más de 0,15 px** *y* **le gana también a C** | **el soft-argmax aporta**, y no es la pila: C tiene la misma pila y 1,66× la cabeza. Es el único desenlace que justifica mirar la foveada |
| **A baja, pero C baja igual o más** | **lo que aporta es la PILA CONV**, no la lectura. Resultado útil y barato: una cabeza más profunda es mucho más fácil de llevar a producción que un cambio de parametrización |
| **A no despega de 2,224 ± 0,15** | **inerte en esta red**: con una `Linear` global sobre un mapa de 20×20 no hay cuello que abrir. Es el desenlace que la predicción da por más probable, y lo interesante entonces sería si B (mapas concentrados) se separa de A |
| **A empeora** | el **sesgo de contracción** manda: el 24,1 % del borde se paga más caro que lo que se gana. Se comprobaría mirando el error **por posición** (borde contra interior) antes de creerlo |
| **el f1 se mueve más de ±0,01** | ⚠ **el control ha fallado** y el resultado no es atribuible. Causa esperable: el cuerpo es **compartido**, así que sus features co-adaptan a la cabeza nueva. Habría que decirlo, no explicarlo a posteriori |

**B se lee sólo contra A**, no contra el ancla: la única diferencia entre los dos es λ. Si B ≈ A,
el regularizador es inerte al valor probado; si B < A, concentrar el mapa es parte del mecanismo.

## Lo que este experimento NO puede contestar, dicho antes

1. **Una semilla por brazo.** Acota, **no declara** — y sin réplicas el ruido entre semillas queda
   sin medir, así que los umbrales de arriba son un **suelo**, no un test.
2. **λ = 0,1 no está optimizado.** Está *dimensionado*: con el mapa plano de la inicialización la
   dispersión vale 0,3639 y el término de posición 0,4362, así que λ=0,1 lo deja en el **8,3 %**
   de la posición. Está medido que es un empujón; **no** está medido que sea un buen valor. Un
   brazo que dependiera de λ pediría barrerlo, y eso es otro estudio.
3. **El cuerpo es compartido y se entrena.** Los brazos no son «las mismas features leídas de otra
   forma»: a las 37 épocas cada uno tiene su propio cuerpo. Es inevitable sin congelar el cuerpo,
   y congelarlo mediría otra cosa.
4. **Nada de esto dice nada sobre la foveada de producción.** Allí son 4 capas y una cabeza mucho
   mayor; este experimento vive entero en una plana de una capa, como el resto de la serie.
5. **`pos_err_px` se mide sólo sobre las esquinas VERDADERAS** (`fv.metrics.pos_err_px`), así que
   no mezcla la calidad de la detección con la de la posición. Es lo que hace que el f1 pueda ser
   control de verdad.
