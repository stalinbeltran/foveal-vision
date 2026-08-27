# Plan de cierre de parámetros — 2026-08-26

**Qué es esto.** El criterio de lectura de tres bloques de estudios, escrito **antes de mirar
ningún resultado**. Es la regla de siempre del proyecto: el veredicto vive aquí, en el documento
que fijó el criterio, y los reportes de
[`telegram-coordinator/reportes/`](https://github.com/stalinbeltran/telegram-coordinator/tree/main/reportes)
resumen *qué se corrió, cuándo, con cuántas máquinas y qué costó* y enlazan aquí.

Lo dispara la tabla resumen de
[`reportes/README.md`](https://github.com/stalinbeltran/telegram-coordinator/blob/main/reportes/README.md)
del coordinador, que a 2026-08-26 deja tres huecos:

1. **`overlap_fovea_px` no está acotado por arriba** y su ganador nominal (4) no llega a
   significación (`p` = 0,270 con 5 semillas).
2. **Siete parámetros de la foveada siguen «sin medir»** — nunca barridos con semillas.
3. **`border_px` = 8 contra 4 lleva `p` = 0,063 medido dos veces** y la plana sigue en tanteo.

Los recorridos los crea [`scripts/estudio_cierre.py`](../scripts/estudio_cierre.py), para que
plan y barridos no puedan divergir: un estudio que está aquí y no allí no se corre, y uno que
está allí y no aquí no tiene criterio.

---

## 0. Las reglas de lectura, que no cambian

Son las del proyecto, repetidas aquí para no tener que ir a buscarlas:

- **El vigente sólo se mueve si `p` < 0,05 Y la diferencia supera δ.** Un ganador nominal con
  `p` = 0,063 no mueve nada, por muchas veces que gane.
- **δ** es 1 SE (error estándar) del mejor punto, como en el resto de los informes
  (`estudio_informe.py`, `delta_fuente`).
- **El contraste es `permutation_test` exacto, de dos colas, sobre la diferencia de MEDIAS**
  (`src/fv/metrics.py`). No es pareado: la semilla *k* de dos redes distintas sólo comparte el
  flujo del generador de azar.
- **Todo esto es f1 de VENTANA, un proxy** que está medido que exagera. **Ningún eje ha pasado
  todavía por la métrica de tarea (R5)**, y este plan tampoco la corre.
- **2 semillas acotan, no declaran.** Con `n` = `m` = 2 el `p` mínimo alcanzable es 0,333
  (C(4,2) = 6, dos colas ⇒ 2/6); con 2 contra 2 no hay forma de bajar del 5 %. Un tanteo dice
  *dónde mirar*, nunca *quién gana*.

### El techo del contraste, y por qué el diseño se para justo ahí

`permutation_test` es **exacto o no es**: se niega a correr si C(n+m, n) > 200.000 en vez de
pasarse en silencio a muestreo, porque un `p` que significa dos cosas distintas bajo el mismo
nombre no decide nada.

| n contra m | C(n+m, n) | `p` mínimo alcanzable | ¿lo admite el test? |
|---:|---:|---:|:--:|
| 2 – 2 | 6 | 0,333 | sí |
| 5 – 5 | 252 | 0,008 | sí |
| **10 – 10** | **184.756** | **1,1 · 10⁻⁵** | **sí, y es el último** |
| 11 – 11 | 705.432 | — | **no** |

**10 contra 10 es el máximo que este proyecto puede contrastar sin cambiar de test.** Por eso los
estudios de significación de abajo piden exactamente 5 semillas más sobre los 5 que ya hay: no es
un número redondo, es el techo.

---

## 0 bis. La comprobación previa: **el dato NO reproduce** (medido, 2026-08-26)

Esto se descubrió al preparar el plan y **cambia el diseño de tres bloques**, así que va antes que
ellos.

### El síntoma

La máquina se rehízo y `data/` se perdió entera. El dataset se reconstruyó desde los specs
congelados (`bench_dataset.py build`, 12,4 min) y:

| dataset | stride | ¿reproduce su huella de git? |
|---|---:|---|
| `bench-dirty1000-16` (el del benchmark) | 8 | **SÍ** — `sha256:6268a2f5…`, igual a la de git |
| `dirty1000-80px-16px-r20260824` (**el de los estudios**) | 5 | **NO** — git dice `3df67624…`, sale `ac875e22…` |

Que el primero reproduzca dice que **la fuente es la buena**: los mil renders salen iguales. Y del
segundo coinciden **campo a campo** `num_windows` (140.000), `windows_per_split`
(84.000/28.000/28.000), `positives_per_corner` (17.043 · 17.564 · 19.198 · 18.575),
`images.shape` y `source_id`; y `split.json` es **idéntico byte a byte**. Sólo cambia el sha256
del `.npz`. La extracción además es determinista aquí (dos extracciones seguidas dan el mismo
hash), así que la diferencia es real y no del momento en que se corrió.

### Por qué no bastaba con eso, y por qué se decidió entrenando

Una huella distinta con los mismos resúmenes admite **dos explicaciones que llevan a decisiones
opuestas**, y la huella no las distingue:

1. **misma información, otra compresión** → el dato sirve, todo el plan vale;
2. **otra información** (posiciones de esquina movidas sub-píxel) → es **otro dataset**.

Y no es una duda académica: `ov-sig`, `bp-sig` y `pl-f2-*` estaban diseñados para **sumar semillas
a runs que ya existen**. Con la explicación (2), esos tres comparan peras con manzanas **y no dan
ningún síntoma** — el `p` sale igual de creíble. Es exactamente el fallo silencioso que este
proyecto evita por escrito.

⚠ **El contraste local no vale**, y por eso no se usó: en esta máquina la época 1 dio
`train_loss` 0,4923 contra 0,4850 y `val_f1` 0,6889 contra 0,6807 — pero **divergencia de CPU y
divergencia de dato se confunden** (medido: cruzar de familia mueve el f1 hasta 0,0457). Un número
que admite dos causas no decide.

### La medida

`repro-chk`: **el mismo punto** (`overlap_fovea_px` = 2), **la misma semilla** (2), **la misma
familia de CPU** —E5-2630 v4 contra el E5-2683 v4 del original, los dos E5-26xx v4, donde está
medido que el entrenamiento sale **idéntico bit a bit**— y **los mismos 8 hilos** de torch. Tres
épocas, `patience` = 0. Coste: **0,0164 $**.

| época 1 | `repro-chk` (dato de hoy) | `ov-fov-0011` (dato r20260824) | Δ |
|---|---:|---:|---:|
| `train_loss` | 0,4462163726167322 | 0,4484938883624737 | −2,28·10⁻³ |
| `val_loss` | 0,2742790627208623 | 0,31479289938103067 | −4,05·10⁻² |
| `val_f1` | 0,7813885915277565 | 0,6786845310596833 | **+1,03·10⁻¹** |
| `pos_err_px` | 2,398193359375 | 2,409182548522949 | −1,10·10⁻² |

Las tres épocas van en la misma dirección: **el dato de hoy es más fácil**.

**El dato que cierra la pregunta es el `train_loss` de la época 1.** Con la misma inicialización y
el mismo orden de ejemplos —los dos los fija la semilla—, y en una familia de CPU donde el
entrenamiento es bit a bit idéntico, la primera época **no tiene de dónde sacar una diferencia**
más que del dato. Ahí no ha habido tiempo de que se acumule ninguna divergencia numérica.

**Veredicto: es OTRO dataset.** Se le pone nombre nuevo —`dirty1000-80px-16px-r20260826`— que es
la convención que este repo ya usó dos veces (r20260823 → r20260824). El r20260824 **no se pisa**.

### La causa, y por qué se creía descartada

La misma de siempre en este repo, y **es reincidente**: la CDN de Playwright devuelve **403 desde
este proveedor** («this service is not available in your location»). El 24-ago se resolvió
rasterizando con `google-chrome-stable`; hoy se resolvió trayendo el Chromium que Playwright fija
**desde otra CDN** (`registry.npmmirror.com`). Otro binario, otros píxeles.

⚠ **Y la diferencia es más fina que la del 24-ago, que es lo que la hace peligrosa.** Aquella
movió los `positives_per_corner` y por eso se vio a simple vista comparando manifests. Ésta **no
los mueve** —coinciden los cuatro— así que **ningún campo del manifest la delata**. Sólo el
`.npz`, y sólo si alguien compara la huella. Lección para el índice, escrita por la acción que la
dispara: **al reconstruir un dataset, comparar la huella NO es opcional, y si no coincide, que los
resúmenes sí coincidan no absuelve** — hay que entrenar un punto conocido y comparar la curva.

### Lo que se lleva por delante

| | antes | ahora |
|---|---|---|
| `ov-sig` | 5 semillas **sumadas** a las de `ov-fov` | **`ov-r26`**: el eje **entero** {0,1,2,4,5,6,7} × 5, sobre el dato nuevo (35 runs) |
| — | — | **`ov-sig26`**: {2,4} × semillas 6–10, sobre el dato nuevo (10 runs) |
| `bp-sig` | 5 semillas sumadas a `borde-ancho` | **`bp-r26`**: {4,8} × **10 semillas propias** (20 runs) |
| `pl-f2-*` | 3 semillas sumadas al tanteo | **5 semillas propias**; del tanteo se hereda **la red**, no los números |
| bloque B | — | **sin cambios**: son auto-contenidos |

**Y sale un ancla gratis**: el punto `overlap` = 2 de `ov-r26` **es la configuración vigente**, así
que su media con 5 semillas mide **cuánto movió el dato** respecto de los 0,9308 que dejó escritos
el [#13]. Eso es justo lo que el commit del 24-ago prometía publicar y nunca se publicó.

⚠ **Lo que esto le hace al inventario entero, y hay que decirlo**: los **630 runs** con curvas en
disco están sobre datasets que **ya no se pueden reconstruir aquí**. La tabla resumen del
coordinador sigue siendo el mejor mapa que hay, pero sus números y los de este plan **no se
comparan entre sí**, sólo se leen en paralelo.

---

## 1. Bloque A — cerrar `overlap_fovea_px`

### Qué se sabe ya (reporte [#13], 20/20 runs)

| `overlap_fovea_px` | f1 (5 semillas) | `p` contra el vigente |
|---:|---:|---|
| 0 (ramas disjuntas) | 0,9186 | **0,032** ← el solape **aporta** |
| 1 | 0,9273 | 0,444 |
| **2 (vigente)** | **0,9308** | — |
| 4 | 0,9372 | 0,270 |

El 0 declara y dice que el solape sirve. El 4 gana +0,0065 y **no** declara. Y 4 **era el borde
del rango**, así que no se sabe qué hay más allá.

### El hecho nuevo que cambia el diseño: el eje es GRATIS

⚠ La tabla del coordinador dice que cerrar este eje hay que sopesarlo «contra su coste, porque el
solape sube N». **Eso es falso, y está medido hoy.**

`N = fovea_px + 2·(border_px // border_reduce)` **no contiene el solape**: N = 20 en todo el
recorrido. Lo que el solape mueve es la *banda* de la rama periférica
(`periph_band = border_cells + overlap_fovea_px`), y con enmascarado ambas ramas siguen
consumiendo el mismo tensor.

**Medido el 2026-08-26** con
`.venv/bin/python -c "…build_model(cfg)… sum(p.numel() for p in m.parameters())"` sobre la base
`ws16-p2-d2-L4`:

| `overlap_fovea_px` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parámetros | 167.852 | 167.852 | 167.852 | 167.852 | 167.852 | 167.852 | 167.852 | 167.852 |
| `periph_band` | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

**Es cost-neutral en parámetros en todo su rango legal**, como `border_px` con el anillo atado
(estudio `borde-ancho`) y **a diferencia** de `border_reduce`, que es el confound abierto. Así que
aquí no hay nada que sopesar: si un punto alto gana, se aplica sin pagar nada.

> El s/época observado en [#13] (46,7 · 48,1 · 58,3 · 48,2 para 0 · 1 · 2 · 4) **no es monótono**
> y no ordena con el eje: a parámetros idénticos, eso es ruido de máquina alquilada, no coste del
> solape. No se usa como argumento en ningún sentido.

### El tope no lo pone el presupuesto: lo pone la geometría

`overlap_fovea_range(16)` = **[0 … 7]** (`src/fv/fovea/__init__.py`: `0 .. fovea/2 - 1`).
Comprobado hoy con `build_search_space`. En 7 la rama periférica ve 7 px de fóvea por lado, o sea
**14 de los 16**, y la fóvea deja de tener parte exclusiva. **No hay «más allá de 7»**: cerrar por
arriba aquí significa llegar a la pared.

### Los dos recorridos

| recorrido | eje | rango | semillas | runs | para qué |
|---|---|---|---|---:|---|
| **`ov-r26`** | `overlap_fovea_px` | **{0, 1, 2, 4, 5, 6, 7}** | 1–5 | 35 | el **eje entero** sobre el dato nuevo: acota por arriba hasta la pared **y** rehace la parte que ya no es comparable |
| **`ov-sig26`** | `overlap_fovea_px` | **{2, 4}** | **6–10** | 10 | llevar el contraste decisivo a **10 contra 10** |

⚠ **`ov-r26` mide el eje entero y no sólo {5,6,7}, que era el plan original.** No es repetir
trabajo: los 20 runs de `ov-fov` están sobre **otro dataset** (§0 bis), así que volver a medir
0/1/2/4 es lo único que hace comparables 5/6/7 con ellos. `ov-sig26` sí puede usar semillas
6–10 porque las 1–5 de esos dos puntos **las pone `ov-r26`, sobre este mismo dato**.

### Criterio, escrito antes de mirar

**A-1 (acotar por arriba).** El eje queda **cerrado por los dos lados** si existe algún punto de
{5, 6, 7} cuya media sea **menor** que la del mejor punto interior. Si el máximo cae en **7**, el
eje **no queda cerrado por evidencia sino por la geometría**, y así hay que escribirlo: «gana el
extremo legal», que es una frase distinta de «gana 7».

**A-2 (significación).** Con las 10 semillas por punto:

- Si **`p` < 0,05 y la diferencia (4 − 2) > δ** → el vigente **pasa a 4**. Es aplicable de
  inmediato: mismos 167.852 parámetros.
- Si **`p` ≥ 0,05** → el vigente **se queda en 2**, y el eje se declara **«acotado, y el óptimo
  interior no se distingue del vigente ni con 20 semillas»**. Eso **no** es un empate por falta de
  datos: con 10 contra 10 el suelo del test es 1,1 · 10⁻⁵, así que un `p` alto ahí es una medida,
  no una limitación.

**A-3.** Si algún punto de {5, 6, 7} supera a 4, se contrasta **ese** contra el vigente con las
semillas que tenga, y se dice con cuántas.

⚠ **Lo que este bloque NO contesta**: si el solape ayuda a la *tarea*. Todo es f1 de ventana.

---

## 2. Bloque B — el tanteo de lo que nunca se midió (2 semillas)

**Por qué tanteo y no 5 semillas de entrada.** Son nueve ejes que nadie ha tocado nunca. Gastar
5 semillas en los nueve son ~130 runs para descubrir —probablemente— que la mayoría no mueve
nada. El tanteo de 2 semillas cuesta la mitad larga y contesta *«¿hay algo aquí?»*, que es la
pregunta correcta cuando no hay ni un indicio. Es exactamente lo que se hizo con la plana
(`pl-t-lr`, `pl-t-bs`).

**Y lo que un tanteo NO puede hacer, escrito antes de tener la tentación**: declarar un ganador.
Con 2 contra 2 el `p` mínimo es 0,333. Ningún resultado de este bloque mueve un vigente.

### Los nueve recorridos

| recorrido | eje | rango | runs | ¿cost-neutral? | por qué se mira |
|---|---|---|---:|---|---|
| **`wd-t`** | `weight_decay` | {0, 1e-5, 1e-4, 1e-3} | 8 | **sí** | **La regularización por la puerta barata.** Medido sobre 612 runs: brecha val/train mediana **+28 %**, y 390 pasan del 20 %. Es el prometido «10 ter»: existe en la receta, está en 0,0 y nunca se movió |
| **`opt-t`** | `optimizer` | {adam, adamw, sgd} | 6 | sí | Nunca comparado. ⚠ Con `weight_decay` = 0, **adam y adamw deberían salir idénticos**: eso es un **control**, no un punto — si difieren, hay algo mal en el código, no en el hiperparámetro |
| **`lp-t`** | `lambda_pos` | {0,5, 1, 2, 4} | 8 | sí | El reparto entre error de existencia y de posición. ⚠ Contrato ⑨: **se rankea por f1, nunca por loss** — cada punto tiene una loss distinta y λ→0 «ganaría» por definición |
| **`sb-t`** | `smooth_l1_beta` | {0,02, 0,08, 0,32} | 6 | sí | Trampa heredada ya desactivada (el 1,0 de PyTorch haría MSE puro sin avisar). Barrerlo es ajuste fino; misma reserva del ⑨ |
| **`pat-t`** | `patience` | {5, 10, 20} | 6 | sí (cambia el reloj) | ⚠ **No es un eje de calidad, es el criterio de parada**, y se mide como tal. El mínimo seguro está medido indirectamente en **8** sobre 70 runs, así que **5 está por debajo a propósito**: es el punto que dice si ese 8 aguanta |
| **`mrg-t`** | `merge` | {concat, sum} | 4 | **NO** ⚠ | Cómo se cosen las dos ramas. **Medido hoy: `sum` da 91.052 parámetros contra 167.852, o sea 0,54×.** No es una elección de estilo: es **otra red, y más pequeña**. Si `sum` empata, la noticia es que sobra la mitad de la red |
| **`pool-t`** | `pool_mode` | {avg, max} | 4 | sí (167.852 los dos) | Cómo resume la periferia cada bloque comprimido. Con texto, `max` conserva trazos finos que `avg` difumina |
| **`pad-t`** | `pad_mode` | {edge, mean, zero} | 6 | sí | Qué se pone cuando la ventana se sale de la imagen. Efecto esperado pequeño: sólo toca ventanas de borde |
| **`ovb-t`** | `overlap_border_px` | **{0, 2}** | 4 | sí (167.852 los dos) | El simétrico del solape: cuánta *periferia* ve la fóvea. Grado de libertad nuevo (2026-08-25). ⚠ **Con el borde vigente de 4 px y `border_reduce` = 2 sólo admite esos dos valores** (`overlap_border_range(4, 2)` = [0, 2], comprobado hoy): esto es lo máximo que se puede preguntar hoy, y es honesto decir que es poco |

**52 runs.** Estimado a los 0,054 $/run medidos el 25-ago: **≈2,8 $**.

### Criterio, escrito antes de mirar

Un eje del bloque B **pasa a verificación de 5 semillas** (bloque C) si cumple **cualquiera** de:

1. la separación entre su mejor y su peor punto **supera 0,010 de f1** — el doble del ruido
   típico entre semillas en estos recorridos (`value_std` ≈ 0,005–0,009 en [#13]); **o**
2. su mejor punto supera al vigente por **más de 1 SE** de las 2 semillas de ese punto; **o**
3. es **`mrg-t` y `sum` no pierde más de 0,010** — porque ahí la noticia no es ganar, es
   **empatar con 0,54× de parámetros**, y eso hay que confirmarlo antes de creérselo.

Un eje que no cumpla ninguna se declara **«tanteado, sin señal»** y se cierra ahí, con sus números
escritos. **No** se declara «no importa»: se declara que con 2 semillas no asoma nada, que es lo
que se midió.

### Lo que NO se barre, y por qué — con el número al lado

Esto es la mitad del encargo «que todos sean medidos»: los que **no se pueden** medir hoy tienen
que quedar dichos con su razón, o el hueco se lee como olvido.

| Parámetro | Por qué no entra | Comprobado hoy con |
|---|---|---|
| **`k_periph`** | **Un solo valor legal.** `kernel_range(periph_band = 4)` = **[3]**. La banda periférica son `border_cells + overlap_fovea_px` = 2 + 2 = 4 px, y un kernel debe caber en ~la mitad. **No es barrible con la geometría vigente**; lo sería con el borde ancho (con `border_px` = 16 la banda llega a 10 y admite [3, 5]) | `build_search_space(geom, n_layers=4)` |
| **`s_center`** | **Un solo valor legal.** `stride_range(16, 4)` = **[1]**: el producto acumulado de 4 capas no puede pasar de `región/4` = 4, y ya `2⁴` = 16 lo rompe | ídem |
| **`s_periph`** | **Un solo valor legal.** `stride_range(4, 4)` = **[1]** | ídem |
| **`momentum`** | **Inerte hoy**: sólo actúa con `optimizer: sgd`, y el vigente es `adam`. Queda **condicionado** a `opt-t`: si `sgd` sale competitivo, se barre entonces; si no, barrerlo mide una constante | `recipe.py` («applies to sgd») |
| **`epochs`** | **Es guarda, no ajuste — y hoy no ata.** Medido el 2026-08-26 sobre los **630 runs con curvas en disco**: la época más alta observada es **130** y **ninguno** llegó a 150. Subir el tope daría runs **idénticos**, porque quien para es `patience`. Barrerlo hacia arriba es pagar por repetir. *(Hacia abajo sí cambiaría algo, pero eso es medir el recorte, no el parámetro.)* | recuento sobre `runs/*/metrics.jsonl` |
| **`fovea_px`** | **No barrible por contrato**: es la ventana etiquetada de B. La puerta lo rechaza (`axis_breaks_window_size`); cambiarla exige **regenerar el dataset** | `docs/organizacion.md` contrato ①a |
| **`dropout`** | **Está en tres documentos y NO en el código.** `NETWORK_DEFAULTS` no lo tiene y el `forward` no aplica ninguna capa; `full_config` filtra por `NETWORK_DEFAULTS`, así que ponerlo de eje **entrenaría N veces la misma red y no avisaría**. Va después de `wd-t`, y sólo si la regularización mueve algo | commit `06cc8c2a` |
| **`regions`** | **No es eje a propósito**: comparar así sería tramposo (las dos redes verían áreas distintas). Es la comparación foveada-contra-plana, con bases separadas | `plan-cnn-plana.md` §3 |
| **`min_size`** | **Es de inferencia (F): no cuesta alquiler.** Se mide **en local** con `knobs_f.py`, sin flota | §3 |

---

## 3. Bloque C — la verificación completa (5 semillas)

Se lanza **después** del tanteo, y su contenido depende de él: los ejes que pasen el criterio de
§2 más los tres cierres que ya estaban pendientes y no dependen de nada.

| recorrido | eje | rango | semillas | runs | qué cierra |
|---|---|---|---|---:|---|
| **`bp-r26`** | `border_px` (con `border_reduce` atado) | {4, 8} ↔ {2, 4} | **1–10** | 20 | **La `p` = 0,063 medida dos veces**, ahora con **10 semillas propias** ⇒ 10 contra 10. Es lo que el propio informe pide: *más semillas en esos dos puntos, no un rango más ancho* |
| **`pl-f2-bs`** | `batch_size` (plana) | {85, 170, 340} | 1–5 | 15 | Fase 2 de la plana; el tanteo dejó 170 ganando **por dentro** |
| **`pl-f2-nl`** | `n_layers` (plana) | {4, 5, 6} | 1–5 | 15 | Fase 2 de la plana. ⚠ **L6 dio f1 = 0,0000 en una de sus dos semillas**: con 5 se ve si es bimodalidad o una casualidad |

Los tres llevan **semillas propias y completas**, no sumadas: el tanteo y `borde-ancho` están sobre
el dato viejo (§0 bis). De `pl-t-*` se hereda **la red** —que hoy no se puede re-derivar, ver el
aviso de abajo— y **no los números**: un estudio de 5 semillas declara solo, y el tanteo ya cumplió
su papel diciendo *dónde* mirar.
| *(los que pase el bloque B)* | — | — | 1–5 | ~20–40 | según §2 |

**Lo creado y lanzado suma 147 runs** (A: 45 · B: 52 · C: 50). Estimación de `estudio_flota.py`
antes de lanzar: **reloj 3,8 h · 86,3 máquina-horas · 7,57 $** en el escenario central (6,93 $
optimista, 9,90 $ pesimista) — *estimación, no medida*. El coste real va en el reporte.


### ⚠ Hallazgo al preparar esto: la base de la plana **no se puede re-derivar hoy**

No es un detalle de implementación, así que va en el plan y no sólo en el código.

**Medido el 2026-08-26**, con
`derive_base(16, overrides={'regions':'single','border_reduce':1,...}, border_px=4)`:

| | lo que sale hoy | lo que la fase 1 midió |
|---|---|---|
| etiqueta | `ws16-p0-d1-L4` | `plana-24-single` |
| `border_px` | **0** | 4 |
| entrada | **16×16** | **24×24** |

`derive.py` fuerza `border_px = 0` cuando `regions='single'`, con este motivo escrito:
*«regions='single' es la CNN plana: una sola rama sobre todo el input, sin borde»*. Pero
[`plan-cnn-plana.md`](plan-cnn-plana.md) §5.1 exige justo lo contrario y lo llama **la premisa**:
la plana tiene que ver **la misma área original** que la foveada —24×24, porque la periferia
comprime ×2— o la comparación mediría *cuánta imagen ve cada una* en vez de *cómo la mira*.

La corrección **confunde dos cosas que la reparametrización del 25-ago separó**: el *anillo*
(estructura, que `single` efectivamente no tiene) y el *recorte* (área, que sí tiene que
conservar). Que la geometría es legal está comprobado:
`check_dims({fovea 16, border 4, reduce 1}, single=True)` → `[]`, sin problemas.

**Consecuencia práctica, y es la que importa: hoy `estudio_plana.py` no reproduce la base con la
que se midió la plana.** Mientras eso no se arregle en `derive.py`, cualquier recorrido nuevo de
la plana que se derive desde cero mide **otra red**, y lo haría en silencio — el nombre del
recorrido no lo delata.

Por eso `pl-f2-*` **hereda** `base_network_value` del recorrido de fase 1 en vez de re-derivarlo
(`_hereda` en `estudio_cierre.py`), y comprueba dataset y receta antes de aceptar la herencia. No
es una preferencia de estilo: es la única forma hoy de que la fase 2 continúe el tanteo en vez de
empezar otro estudio con el mismo nombre.

⚠ **Esto queda como arreglo pendiente en `derive.py`, y no se ha hecho aquí**: cambiar la
semántica de `regions='single'` toca la puerta que valida *todos* los entrenamientos, y hacerlo a
la vez que se lanza una flota mezcla dos cosas que tienen que poder fallar por separado.

⚠ **`pl-f2-nl` con L6 puede volver a dar ceros.** Está previsto: si una semilla colapsa, la media
del punto **no se cita** —es el promedio de una moneda— y se reporta el recuento de colapsos, como
en [#10].

### Lo que este plan deja explícitamente abierto

1. **La pregunta del proyecto** (¿gana la foveada?) sigue necesitando la **métrica de tarea**
   sobre las dos redes afinadas. La fase 2 de la plana la **desbloquea**; no la contesta.
2. **El confound de `border_reduce`** (capacidad contra resolución) **no se toca aquí.**
   Desconfundirlo pide un diseño a parámetros igualados —subir N bajando `channels`—, y eso es un
   estudio propio con su propio criterio, no una fila más.
3. **Ningún eje pasa por R5.** En `k_center` eso no es una laguna cualquiera: es el único eje
   donde proxy y tarea se contradicen **en el signo**.
