# Plan: los estudios de prioridad 1 y 2, con su criterio escrito ANTES

Fecha de redacción: **2026-08-25**, y la fecha importa: esto se escribe **antes de mirar
ningún resultado**, que es la única forma de que las reglas de abajo decidan algo. Un
criterio escrito después de ver la tabla no es un criterio, es una justificación.

Origen: [`reportes/2026/08-agosto/parametros-y-prioridad-de-estudios.md`](../reportes/2026/08-agosto/parametros-y-prioridad-de-estudios.md)
§6, que ordenó el inventario entero por *(evidencia de que hay algo que ganar) × (lo que
cuesta) × (si desbloquea otra pregunta)*. Este documento no reordena nada: coge esa lista y
le pone el **cómo se lee** a cada entrada.

## 0. Lo que se hereda, y no se vuelve a discutir

- **Dataset**: `dirty1000-80px-16px-r20260824` para todos, sin excepción. Mezclar datasets
  dentro de un estudio entrenaría media tabla sobre otro dato.
- **Red base foveada**: la vigente `ws16-p2-d2-L4` — `border_px` 4, `border_reduce` 2,
  `overlap_fovea_px` 2, `n_layers` 4, `channels` [16]×4, `regions` split.
- **Red base plana**: `plana-24-single` — `regions` single, `border_px` 4, `border_reduce` 1,
  `n_layers` 4, `channels` [22]×4.
- **Receta**: `plan40` (`lr` 0,0014 · `batch_size` 85 · `patience` 10 · `adam` · `monitor`
  val_loss).
- **Semillas**: 5 (1..5) en todo lo que declara ganador. Con 5 contra 5 la permutación exacta
  da 252 arreglos y el p mínimo alcanzable es 1/126 ≈ **0,0079**: R4 puede declarar
  significación al 5 %.
- **Reglas R1..R6**: las de [`plan-tres-ejes.md`](plan-tres-ejes.md) §5, tal cual. Las calcula
  `scripts/estudio_informe.py` con las funciones del proyecto, no se re-implementan aquí.
- **Tope de épocas**: 150, salvo donde se diga otra cosa y se diga por qué (E7).

Recordatorio de R1, que es la que más veces ha invalidado un estudio aquí: **un punto cuyos
runs paran por el tope de épocas y no por `patience` mide presupuesto, no calidad.** Si el
ganador es uno de ésos no se declara ganador, se reporta `budget-limited`.

---

## 1. Prioridad 1

### E1 — `borde-ancho`: ¿ayuda ver más contexto, **a coste constante**?

**Eje**: `border_px` ∈ [**4**, 8, 10, 12, 16] px, con el anillo **fijo en 2 celdas**, o sea
`border_reduce` atado ∈ [2, 4, 5, 6, 8]. 5 semillas → **25 runs**.

**Por qué así, y no barriendo `border_px` a secas.** `N = fovea + 2·(border_px/border_reduce)`
y la cabeza es `Linear(2·C·N², 12)` — el 97 % de los parámetros. Si se barre `border_px` con
`border_reduce` fijo, **N crece con el eje** y el estudio mide «más área **y** más parámetros»,
que son dos cosas. Con el anillo fijo en 2 celdas **N = 20 en los cinco puntos**: mismo
tensor de entrada, mismo número de parámetros, mismo coste por época. Lo único que cambia es
**cuántos píxeles reales de la imagen** se condensan en ese anillo. Eso es exactamente la
pregunta de la visión foveada, y hasta la reparametrización del 2026-08-25 no se podía ni
escribir (hacía falta mover `N` y `c_frac` a la vez, y el motor es OAT).

Se implementa con la **atadura** (`couple`) que se añadió para esto — ver `spec.py`,
`expand_points`.

**Por qué el rango empieza en 4 y no en 8.** 4 px **es el vigente**, y R4 exige contrastar
contra el vigente *medido en este mismo recorrido*. Además 4 px con `border_reduce` 2 es
justamente la red vigente, así que el punto ancla no cuesta ningún diseño nuevo. Los puntos
8, 10, 12 y 16 continúan la serie que `proxy-c-d` y `d5-L4` ya midieron como 2, 4, 6 y 8 px
—subiendo de forma monótona— y que se cortó con el ganador en el borde (p = 0,063).

⚠ **Techo declarado antes de mirar**: a 16 px el recorte real es 48×48 sobre imágenes de
60×80, y el **26,4 % del anillo es relleno replicado** (`pad_mode: edge`), no imagen
([instructionsNewNN.md](../instructionsNewNN.md) §2.2). El rango para ahí a propósito. **Si
gana 16, el veredicto NO es «sigue subiendo, ampliad el rango»**: es «sigue subiendo *y* el
siguiente paso empieza a medir el relleno», y para pasar de ahí hace falta otro dataset con
imágenes mayores, no otro barrido. Esto se escribe ahora justamente porque es la tentación
que `plan-tres-ejes.md` §7.3 ya reconoció haber tenido con `d`.

**Vigente**: `border_px` = 4.

### E2 — terminar el afinado de la plana

Dos tanteos a medias, con **2 semillas**, que sólo acotan y **no declaran ganador**:

- `pl-t-bs` (`batch_size`): faltan **7 de 10 runs**.
- `pl-t-nl` (`n_layers`): faltan **10 de 10** (nunca arrancó).

Y después, la **fase 2 con 5 semillas** sobre el rango que los tanteos acoten. El rango de la
fase 2 **no se escribe aquí porque depende de lo que salga**, y eso es legítimo en un diseño
de dos fases *siempre que la regla de cómo se elige esté escrita antes*. Es ésta: **la fase 2
barre el ganador nominal del tanteo y sus dos vecinos**; si el ganador nominal es un extremo
del rango del tanteo, se añade un punto más allá para que deje de serlo.

⚠ `pl-t-lr` ya está terminado (10/10) y trae una semilla con **f1 exactamente 0,0000** en
`lr`=0,0028 — un entrenamiento que no arrancó. Eso **no se promedia con las demás y ya está**:
es el fallo bimodal que [`plan-plana.md`](plan-plana.md) §6.1 documenta en la plana con L5/L6.
Se reporta como lo que es, un punto con una réplica muerta, y su media queda marcada.

### E3 — foveada vs plana **por métrica de tarea**

La pregunta que da nombre al proyecto, y sigue sin contestar. **0 entrenamientos nuevos**: se
reusan los ganadores de E2 y el vigente foveado, y se mide con `scripts/proxy_vs_task.py`.

**Cómo se lee, escrito antes**: `paragraph_f1` sobre imagen completa, 5 semillas por
arquitectura, **permutación exacta** de las semillas. Gana una arquitectura si p < 0,05 **y**
la diferencia supera δ (1 SE de las semillas del mejor).

⚠ **Lo que NO vale como respuesta**: el f1 de **ventana**. Los 0,96 de la plana contra los
0,93 de la foveada no son comparables — las dos redes ven áreas distintas — y citarlos como
la comparación del proyecto es el error que
[`plan-cnn-plana.md`](plan-cnn-plana.md) §4 prohíbe explícitamente.

### E4 — los knobs de inferencia (F)

**No se ejecuta como cambio.** Es **decisión F15, del usuario**: aplicar los defaults buenos
(`threshold` ≈0,25–0,3 · `stride` n/4 · `nms_radius` 3n/4) sube la métrica de tarea entre
+0,065 y +0,261 con los pesos que ya hay, pero **mueve todos los números que el proyecto ha
publicado** y además **comprime la separación entre un modelo bueno y uno malo** (0,343 →
0,147) con el mismo ruido, o sea que la métrica distingue **peor** entre modelos.

Lo que sí se hace aquí: **medirlo otra vez sobre los modelos vigentes y dejarlo en el
reporte**, para que la decisión se tome con el número delante y no de memoria. Aplicarlo o no
lo dice el usuario.

---

## 2. Prioridad 2

Todos sobre la **foveada vigente**, 5 semillas, tope 150 épocas salvo E7.

| # | Recorrido | Eje | Valores | Runs | Vigente |
|---|---|---|---|---|---|
| E5 | `pw-fov` | `pos_weight` | 1, 2, 4, 8 | 20 | 1 |
| E6 | `mon-fov` | `monitor` | `val_loss`, `val_f1` | 10 | `val_loss` |
| E7 | `sch-fov` | `scheduler` | `none`, `cosine` | 10 | `none` |
| E8 | `ch-fov` | `channels` | 8, 16, 24, 32 (uniformes) | 20 | 16 |
| E9 | `kc-fov` | `k_center` | 3, 5, 7 | 15 | 3 |
| E10 | `ov-fov` | `overlap_fovea_px` | 0, 1, 2, 4 | 20 | 2 |
| E10b | `red-fov` | `border_reduce` con `border_px`=8 | 4, 2, 1 | 15 | — |

### Lo que hay que declarar antes de mirar, estudio por estudio

**E5 `pos_weight`** — contrato ⑨: es un peso de la pérdida, así que el objetivo **no puede
ser `loss`** (cada punto se mediría con una regla distinta). Se rankea por `f1`, que es el
default. Es la hipótesis más plausible de mejora grande sin probar: el cuello de botella está
**medido y es de detección** — con esquinas perfectas la reconstrucción da 0,97 y el mejor
modelo real da 0,64.
⚠ **Y por eso mismo el proxy engaña aquí más que en ningún otro eje**: subir `pos_weight`
detecta más y se equivoca más, y el `val_f1` de **ventana** ya castiga los falsos positivos,
mientras que lo que se quiere mover es el párrafo entero. **Si gana algo distinto de 1, R5
no es opcional**: se mide con métrica de tarea antes de arrastrarlo.

**E6 `monitor`** — hoy el checkpoint se elige por `val_loss` y el ranking es por `val_f1`.
⚠ Cuidado con leer el resultado: el brazo `monitor: val_f1` **elige su checkpoint con la
misma métrica con la que luego se le puntúa**, así que parte con ventaja mecánica. Eso no lo
invalida —es exactamente lo que se propone hacer— pero **el veredicto tiene que decirlo**, y
la ganancia sólo cuenta como real si sobrevive a R5 (métrica de tarea).

**E7 `scheduler`** — **tope de épocas 100 en los dos brazos, y no 150**. Razón: `cosine` usa
`T_max = recipe.epochs`, o sea el **tope**, para planificar la bajada. Con tope 150 y parada
real entre las épocas 32 y 81 (medido en los 65 runs del 25-ago), el `lr` sólo bajaría a ~0,75
de su valor inicial y el estudio mediría «cosine casi sin aplicar». Con tope 100 la bajada es
real (~0,35 en la época 60) y el tope sigue por encima de la parada más tardía observada (81),
así que **manda `patience`, no el reloj** — que es R1. Los dos brazos comparten tope, o el
brazo `none` estaría corriendo otro experimento.

**E8 `channels`** — el indicio de 1 semilla dice que la anchura no aporta. Ojo: **el interés
real puede estar hacia abajo**. Si 8 canales empatan con 16, la red vigente es el doble de
cara que hace falta, y eso es un resultado tan publicable como una mejora.

**E9 `k_center`** — el único parámetro donde el proxy y la tarea se **contradicen en el
signo** con la evidencia que hay (con 1 semilla, `k_center`=5 fue el **peor** por f1 de
ventana y el **mejor** por métrica de tarea). Por eso aquí **R5 se aplica siempre**, gane lo
que gane: el estudio es tanto sobre el parámetro como sobre si el proxy sirve.

**E10 `overlap_fovea_px`** — el mando exclusivo de esta arquitectura, nunca medido.
**El 0 es el punto que más dice**: hace las dos ramas **disjuntas**, o sea que es el control
de la elección de solape contributivo de [instructionsNewNN.md](../instructionsNewNN.md) §7 —
y hasta la reparametrización del 2026-08-25 **no se podía ni escribir** (el suelo era 1 px).

**E10b `border_reduce` con `border_px` fijo** — la otra mitad de la pregunta del borde:
*a igual área de contexto, ¿ayuda verla con más resolución?*
⚠ **NO es cost-neutral, y hay que leerlo con eso delante.** Con `border_px`=8, los reduce 4,
2 y 1 dan anillos de 2, 4 y 8 celdas → **N = 20, 24 y 32**, o sea **+44 % y +156 % de
parámetros en la cabeza**. Si gana el reduce menor, parte de la ganancia es capacidad y no
resolución, y el veredicto **tiene que decir que las dos cosas están confundidas**. Va después
de E1 en importancia porque primero conviene fijar **cuánta** área conviene.

---

## 3. Coste declarado antes de gastarlo

152 runs nuevos a **0,054 $/run** (medido el 2026-08-25 sobre 65 runs: 3,49 $) ≈ **8,2 $**,
más el peaje de arranque+subida+instalación, que se paga **por máquina** y por eso crece con
el reparto. Se comprueba con `--estimar` antes de alquilar nada, y el número real sale de
`flota.json`, no de esta estimación.

**El freno**: los recorridos se destruyen en un `finally`, y si algo se corta a mitad el
apagado manual es `/use apagar-vast`. Las máquinas facturan por segundo mientras existan.
