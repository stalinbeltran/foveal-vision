> Encargo del dueño, 2026-09-04 (por Telegram):

Quiero hacer un experimento empleando soft-argmax, y el mismo dataset de entrada de los otros
experimentos (los no pre-procesados).

Dime qué necesitas

---

Se le contestó con siete decisiones y sus recomendaciones, y respondió:

> Hazlo con tus recomendaciones

**Lectura, o sea qué se decidió y con qué queda amarrado.** «Soft-argmax» es una forma de leer
**las coordenadas**: en vez de regresarlas con una `Linear`, se produce un mapa de calor por
esquina, se pasa por un softmax y se toma la **esperanza** de la posición. Toca `x, y`; no toca
`exists`.

Las siete decisiones, tal como se acordaron:

| # | Decisión | Lo acordado |
|---|---|---|
| 1 | **Sobre qué cuerpo** | `plana-20-4k7` — la más fuerte de la serie plana (f1 0,840) y la única con **4 canales**, que es lo que hace falta para que un mapa de calor no sea una función afín de un único mapa |
| 2 | **Qué sustituye** | **sólo `(x, y)`**. `exists` se queda con su `Linear` de siempre, y por eso el f1 es el **control** y `pos_err_px` el discriminador |
| 3 | **Presupuesto de la cabeza** | **igualado** al del ancla (+0,2 %). La serie ya midió que lo que mueve el f1 es el tamaño de la cabeza; sin igualar, esto volvería a medir eso |
| 4 | **La rejilla** | **la vista entera 20×20**, que en unidades de la ventana etiquetada abarca `[-0,375 · 1,375]` — le da al softmax masa que colocar fuera de `[0,1]` |
| 5 | **La temperatura β** | **aprendida** (un escalar, parametrizado en log) |
| 6 | **Cuántos runs** | **tres**: soft-argmax · soft-argmax + regularizador de dispersión · el control de la pila conv |
| 7 | **El criterio** | escrito y congelado antes de correr nada: [`02-criterio.md`](02-criterio.md) |

**El dataset es el que pidió**: `dirty1000-80px-16px-r20260827`, el mismo de los siete
experimentos de esta carpeta, **sin** pasar por `aplicaKernel`. Los preprocesados
(`foveal-vision-data/preprocesado/1k{3,5,7}-relu`) quedan fuera a propósito.

Se hereda todo lo demás de la serie: semilla 1, receta `plan40`, 37 épocas, `src/fv/` intacto
con el parche en memoria, y el mismo `windows.npz`.

⚠ **Y una cosa que este experimento NO es**: no es un punto más del eje `k` de la serie plana.
Aquélla mueve el **cuerpo** (cuántos kernels y de qué tamaño) con la cabeza fija; ésta deja el
cuerpo quieto y mueve **la cabeza**. Por eso su ancla es un run concreto —`plana-4k7-s1`— y no
la tabla de los seis.
