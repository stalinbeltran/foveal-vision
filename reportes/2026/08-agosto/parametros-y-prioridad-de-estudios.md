# Todos los parámetros, qué se ha barrido ya, y en qué orden conviene seguir

Fecha: **2026-08-25**. Alcance: las **dos redes** que el proyecto entrena hoy — la **foveada**
(`regions: split`, la hipótesis del proyecto) y la **CNN plana** (`regions: single`, el control) —
y los parámetros de entrenamiento que comparten.

Este informe **no mide nada nuevo**: recoge lo que ya está en disco (23 recorridos, 478 runs) y en
los planes de `docs/`, y lo ordena. Todo número que aparece aquí sale de un artefacto o de un
documento del repo, con su enlace.

---

## 0. Cómo leer esto, en cuatro ideas

Si no eres experto, estas cuatro reglas del proyecto explican por qué la columna «¿barrido?» dice
«no» en sitios donde sí hubo mediciones:

1. **Un parámetro está «barrido» sólo si se midió con 5 semillas.** Una *semilla* es el número que
   inicializa los pesos al azar; el mismo entrenamiento con otra semilla da un resultado algo
   distinto. Medir un valor una sola vez no distingue el efecto del parámetro del efecto del azar.
   *«Un resultado sin N semillas es una anécdota»* — [protocolo.md](../../../docs/protocolo.md).
2. **Un barrido sólo vale si el entrenamiento paró por convergencia, no por el reloj.** Si el tope
   de épocas corta a todos los runs, se midió *velocidad de convergencia*, no *calidad*. Eso
   invalidó los tres estudios de `batch_size` de julio — [plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §2.1.
3. **Un óptimo pegado al borde del rango no es un óptimo**, es el final de la regla: no se sabe si
   más allá sigue subiendo. Por eso `d` y `batch_size` siguen «abiertos por la derecha».
4. **El f1 de ventana es un *proxy*.** Lo que de verdad importa es reconstruir el párrafo entero en
   la imagen (`paragraph_f1`). Están correlacionados (+0,956 en ejes de entrenamiento, +1,000 en el
   eje `d`), pero **el proxy exagera**: en `n_layers` la ganancia real fue la mitad —
   [metrica-de-tarea.md](../../../docs/metrica-de-tarea.md).

**Un dato de coste para calibrar todo lo que sigue:** con el reparto en flota medido el 2026-08-25,
**65 runs costaron 3,49 $ y 8,6 h de reloj** ([plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §7.5).
O sea ≈ **0,054 $ por run**. Un estudio típico (4 valores × 5 semillas = 20 runs) ronda **1,1 $ y
1–2 h**. El cómputo ya no es el recurso escaso: **el recurso escaso es decidir qué se pregunta**.

---

## 1. Estructura de la red — dominio **C**

Definen **la forma del modelo y de lo que ve**. Cambiar uno cambia la red: hay que reentrenar.

Vigentes: foveada `ws20-c0.8-d2-L4` (≈**167.852** parámetros) · plana `plana-24-single`
(≈**165.430**, elegida a propósito para que ambas tengan el mismo tamaño).

| Parámetro | Qué es, en cristiano | Foveada | Plana | ¿Barrido con 5 semillas? | Qué se sabe |
|---|---|---|---|---|---|
| **`N`** | Lado, en píxeles, del tensor cuadrado que entra en la red. La imagen se recorre con una ventana y cada ventana se convierte en un cuadrado de N×N. | 20 | 24 | **No, y no es barrible** | Está atado por el contrato ①a al tamaño de ventana del dataset. Se **deriva**, no se barre: la puerta lo rechaza (`axis_breaks_window_size`, [spec.py:31](../../../src/fv/sweeps/spec.py#L31)). Para cambiarlo hay que regenerar el dataset. |
| **`c_frac`** | Qué fracción de ese cuadrado es **fóvea** (la parte a resolución completa, donde se etiquetan las esquinas). El resto es periferia. | 0,8 → fóvea 16 px | 0,667 → fóvea 16 px | **No, y no es barrible** | Igual que `N`: juntos fijan que la fóvea mida exactamente la ventana etiquetada (16 px). |
| **`d`** | Cuánto **comprime la periferia**. `d=2` significa que cada píxel del anillo exterior resume un bloque 2×2 del original: se ve **más área** por el mismo coste, pero borrosa. Es la idea central de la visión foveada. | **2** | 1 (n/a) | **Sí, dos veces** — `proxy-c-d` (6×5, red L2) y `d5-L4` (4×5, red vigente) | ⚠ **ABIERTO POR LA DERECHA.** El eje **sube de forma monótona**: 1 → 0,9310 · 2 → 0,9341 · 3 → 0,9362 · **4 → 0,9408**, y `d=4` es además **el más barato** (39,8 s/época contra 46,6). El ganador es el extremo del rango, con p = 0,063 contra el vigente. [plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §7.3 reconoce que recortar el rango a `[1..4]` fue un error de diseño. |
| **`pen_frac`** | Anchura de la **banda de penetración**: la franja donde la rama central y la periférica se solapan y ambas aportan. Con 0,1 son 2 px en N=20. Es el «pegamento» entre las dos vistas. | 0,1 | 0,1 (n/a) | **NUNCA** | Sin medir jamás, ni con una semilla. Es un mando **propio de esta arquitectura**: si el solape importa, aquí se vería. |
| **`n_layers`** | Cuántas capas convolucionales tiene cada rama. Más capas = **campo receptivo mayor** (cada neurona final «ve» más contexto), no más capacidad: el 97 % de los parámetros está en la cabeza final. | **4** | **4** | **Sí, cuatro veces** — `p40-confirm-n_layers`, `nl5-L4` (foveada); `plana-confirm-s0`, `plana-screen-s0` (plana) | **CERRADO por los dos lados.** Gana 4: 2 → 0,9066 (p = 0,008) · 3 → 0,9246 (p = 0,040) · **4 → 0,9341** · 5 → 0,9136. ⚠ **A partir de 5 el entrenamiento a veces no arranca**: `sem` 7× mayor y semillas bimodales (0,8585 … 0,9415). En la plana pasa lo mismo, y con L5/L6 hubo semillas con f1 exactamente 0,0000 ([plan-plana.md](../../../docs/plan-plana.md) §6.1). Para pasar de 4 haría falta cambiar la arquitectura (residuales, otra inicialización), no el número. |
| **`channels`** | Cuántos **filtros** (detectores de patrón) aprende cada capa. Es la «anchura» de la red, frente a `n_layers`, que es la «profundidad». | [16,16,16,16] | [22,22,22,22] | **NO** (sólo 1 semilla, en `p40-screen-width`, borrado) | Con 1 semilla, doblar canales dio **+0,0046** — dentro del ruido — con **2× parámetros y más coste** ([plan-40h.md](../../../docs/plan-40h.md) §2). Indicio de que la anchura no es el cuello de botella, **pero no es un estudio**. Los 22 canales de la plana no son un óptimo: se eligieron para igualar parámetros con la foveada. |
| **`k_center`** | Tamaño del **kernel** (la ventanita que barre la imagen) en la rama central. 3 = mira 3×3 píxeles a la vez. | 3 | 3 | **NO** (1 semilla) | ⚠ El dato de 1 semilla es **contradictorio, y por eso interesante**: `k_center=5` fue **el peor por f1 de ventana** (−0,0063) y a la vez **el mejor de los cuatro por métrica de tarea** (0,7594). Candidato barato si se vuelve a mirar estructura ([plan-40h.md](../../../docs/plan-40h.md) §8). |
| **`k_periph`** | Lo mismo en la rama periférica. Como la periferia está comprimida, un kernel de 3 abarca ahí `3·d` píxeles originales. | 3 | n/a | **NO** (1 semilla, recorrido borrado) | Sin evidencia utilizable. |
| **`s_center`** | **Paso** (stride) del kernel central: cada cuántos píxeles se aplica. Subirlo **reduce el coste** y la resolución de salida. Sólo actúa en la primera capa (D-S1). | 1 | 1 | **NO** (1 semilla, borrado) | Es el mando de **coste**, no tanto de calidad. Sin medir con semillas. |
| **`s_periph`** | Ídem en la periferia. | 1 | n/a | **NO** (1 semilla, borrado) | ⚠ Con `merge: sum` los dos strides deben ser iguales (`merge_sum_needs_equal_strides`). |
| **`merge`** | Cómo se juntan las dos ramas: `concat` las pega una detrás de otra (más parámetros en la cabeza), `sum` las suma píxel a píxel (exige la misma forma). | concat | n/a | **NUNCA** | Elección discreta de 2 valores, nunca medida. Barata de probar (2 valores × 5 semillas = 10 runs). |
| **`pool_mode`** | Cómo resume la periferia cada bloque comprimido: `avg` (promedio) o `max` (el más brillante). Con texto, `max` conserva trazos finos que `avg` difumina. | avg | avg | **NUNCA** | 2 valores, nunca medido. **Interactúa con `d`**: cuanto más comprime, más importa cómo resume. |
| **`pad_mode`** | Qué se pone cuando la ventana se sale de la imagen: `edge` repite el borde. | edge | edge | **NUNCA** | Efecto esperado pequeño (afecta sólo a ventanas de borde). Baja prioridad. |
| **`regions`** | **Qué arquitectura es**: `split` = dos ramas enmascaradas (foveada) · `single` = una sola rama sobre todo el input (**CNN plana**, el control del proyecto). | split | single | **No como eje** (a propósito) | Es barrible, pero comparar así sería tramposo: las dos redes verían áreas distintas. La comparación se hace con **redes base separadas y el mismo plan** ([plan-cnn-plana.md](../../../docs/plan-cnn-plana.md) §3). **La comparación en sí no se ha hecho todavía.** |

---

## 2. Entrenamiento — dominio **D** (la «receta»)

No cambian la forma de la red, **cambian los pesos que salen**. Vigente: receta
[plan40.yaml](../../../configs/recipes/plan40.yaml), la misma para las dos arquitecturas.

| Parámetro | Qué es, en cristiano | Vigente | ¿Barrido con 5 semillas? | Qué se sabe |
|---|---|---|---|---|
| **`lr`** (learning rate) | El **tamaño del paso** con que se corrigen los pesos en cada actualización. Demasiado grande y el entrenamiento rebota o diverge; demasiado pequeño y no llega. Es el hiperparámetro más importante de casi cualquier red. | **0,0014** | **Sí, cinco veces** (foveada: `fast-lr` 13×5, `fast-lr-2`, `d1000-lr-1`, `p40-lr-L4` 4×5, `lr-alto-L4` 3×3) | **CERRADO por los dos lados** y **plano en medio**. Entre 0,00035 y 0,0014 no mueve la aguja (amplitud 0,0062 de f1; el ganador nominal 0,0006 da p = 0,341); por encima sí degrada. ⚠ **No se ha re-medido sobre el dataset del 24-ago**, que resultó ser más fácil ([plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §7.7). En la **plana** sólo hay tanteo de 2 semillas (§5.1). |
| **`batch_size`** | Cuántas ventanas se procesan juntas antes de corregir los pesos. Grande = menos correcciones por época, pero más rápidas y menos ruidosas. **Es D y no ejecución**: cambiarlo al pasar a GPU invalidaría la comparación (contrato ⑩). | **85** | **Sí, cuatro veces** (`batch_size-1`, `-2`, `d1000-batch_size-1` — los tres **inválidos** por tope de épocas — y `bs5-L4`, válido) | ⚠ **ABIERTO POR LA DERECHA** en el estudio válido: **plano entre 57 y 192** (0,9302 – 0,9351) y sólo cae de verdad en 38 (p = 0,024). El ganador nominal fue **192, el extremo**. El tanteo `bs-alto-fov` (§5.1) ya sugiere que **por encima empieza a bajar**. Nota práctica: 192 es 1,08× más rápido por época sin pérdida medible — **subir el batch abarata gratis**. |
| **`epochs`** | Cuántas pasadas completas sobre el dataset. Hoy es un **tope de seguridad**, no un ajuste: quien decide cuándo parar es `patience`. | 100–300 según estudio | n/a (no es un eje de calidad) | Su papel es de **guarda**: un tope que se agota falsea el resultado. Los 65 runs del 25-ago pararon todos por `patience`, entre las épocas 32 y 81. Con batches grandes hay que subirlo (300 en `bs-alto-*`). |
| **`patience`** | Cuántas épocas seguidas sin mejorar se aguantan antes de parar («parada temprana»). Evita sobreajustar y ahorra reloj. | **10** | **NUNCA** barrido | Medido indirectamente: el mínimo seguro es **8** (la racha más larga sin mejorar seguida de una mejora, sobre 70 runs). ⚠ **Mete varianza por la puerta de atrás**: cada semilla para donde quiere, y la que entrena más lejos suele aterrizar más abajo ([plan-40h.md](../../../docs/plan-40h.md) §5). |
| **`optimizer`** | El algoritmo que aplica las correcciones: `adam` / `adamw` (adaptan el paso por parámetro) o `sgd` (paso fijo + inercia). | **adam** | **NUNCA** | Nunca comparado en este proyecto. `adamw` sólo se diferencia de `adam` si `weight_decay` > 0 — hoy es 0, así que **hoy serían idénticos**. |
| **`momentum`** | La «inercia» de SGD: arrastra parte del paso anterior. **Sólo aplica a `sgd`.** Se declara explícito porque dejarlo en 0 por defecto es una trampa heredada que hunde a SGD en cualquier comparación. | 0,9 (inactivo) | **NUNCA** | Irrelevante mientras el optimizador sea adam. |
| **`weight_decay`** | **Regularización**: penaliza pesos grandes para que la red no memorice el dataset. | **0,0** | **NUNCA** | Nunca tocado. El dato es sintético y abundante (140.000 ventanas), así que el sobreajuste no parece el problema — pero eso es una conjetura, no una medida. |
| **`scheduler`** | Si el `lr` **baja durante el entrenamiento** (`cosine`) o se queda fijo (`none`). Bajarlo al final suele afinar el resultado. | **none** | **NUNCA** | Nunca medido, y es de los pocos mandos con una mejora esperable a priori. ⚠ **Interactúa con `patience` y con `epochs`**: `cosine` usa el tope de épocas para planificar la bajada, así que con parada temprana el efecto cambia. |
| **`lambda_pos`** | Cuánto pesa el error de **posición** de la esquina frente al de **existencia**. La pérdida es `BCE(existe) + λ · error_de_posición`. | 1,0 | **NUNCA** | ⚠ Contrato ⑨: un recorrido que barra esto **no puede rankear por `loss`** (cada punto se mediría con una regla distinta y λ→0 «ganaría» por definición). Se rankea por `f1`. |
| **`pos_weight`** | Cuánto pesa un **positivo** (hay esquina) frente a un negativo en la clasificación. Subirlo empuja a la red a **detectar más**, a costa de más falsos positivos. | 1,0 | **NUNCA** | **Directamente relacionado con el cuello de botella conocido**: un tercio de los párrafos no se detecta en absoluto, y no es culpa de la reconstrucción ([metrica-de-tarea.md](../../../docs/metrica-de-tarea.md) §9.1 y §9.3). Éste es el mando que mueve ese equilibrio **desde el entrenamiento**. |
| **`smooth_l1_beta`** | El punto donde el error de posición pasa de cuadrático a lineal (hace la pérdida robusta a atípicos). Con coordenadas en [0,1] el valor por defecto de PyTorch (1,0) convertiría la pérdida en pura MSE **sin avisar**; por eso está explícito en 0,08. | 0,08 | **NUNCA** | Trampa heredada ya desactivada. Barrerlo es ajuste fino; misma reserva del contrato ⑨. |
| **`monitor`** | Qué métrica decide **cuál época se guarda** como `best.pt`. | **`val_loss`** | **NUNCA** | ⚠ **Hoy el monitor NO coincide con el objetivo**: se guarda el checkpoint de menor `val_loss` pero se rankea por `val_f1`. Es legal y está declarado (`monitor_matches_objective`), pero significa que **el ranking describe un checkpoint elegido con otro criterio**. Probar `monitor: val_f1` es barato y nadie lo ha hecho. |
| **`seed`** | La semilla del azar. **No es un hiperparámetro a optimizar: es el eje réplica.** Elegir la semilla con mejor resultado es engañarse. | 1..5 | — | Los estudios usan 5. ⚠ Hallazgo colateral: dentro de la familia de CPU `E5-26xx`, el mismo run en máquinas distintas sale **idéntico bit a bit** (8 pares comprobados); al cruzar de familia diverge hasta 0,0457 de f1. |

---

## 3. Inferencia — dominio **F** (se ajustan **sin reentrenar**)

No entran en ningún recorrido porque **no cuestan horas**: se aplican sobre un modelo ya hecho. Son
el mejor ratio ganancia/coste de todo el inventario.

| Parámetro | Qué es | Vigente | Óptimo medido |
|---|---|---|---|
| **`threshold`** | A partir de qué confianza se acepta que hay una esquina. Bajarlo detecta más y se equivoca más. | 0,5 | **≈0,25–0,3** (pico plano, óptimo interior) |
| **`stride`** | Cada cuántos píxeles se desliza la ventana sobre la imagen. Menos paso = más ventanas = más lento y más cobertura. | `n/2` (8 px) | **`n/4`** (4 px) — interior: 2 px es **peor**, no sólo más caro |
| **`nms_radius`** | Radio para fusionar detecciones duplicadas de la misma esquina («non-maximum suppression»). | `n/2` | **`3n/4`** (12 px) — interior |
| **`min_size`** | Tamaño mínimo para aceptar un párrafo reconstruido. | 4,0 | sin medir |

**El resultado, y es grande:** con los mismos pesos, corregir los tres defaults sube la métrica de
tarea **+0,065** (modelo bueno), **+0,187** (medio) y **+0,261** (malo). Los defaults actuales
quedaban en el puesto 16–24 de 30 combinaciones. Medirlo entero costó **143 segundos**.

⚠ **Está deliberadamente sin aplicar** — decisión **F15**, del usuario: cambiar los defaults
**mueve todos los números que el proyecto ha publicado**, y la caché no avisaría. Y hay un efecto
secundario medido que hay que sopesar: con los knobs buenos, la separación entre un modelo bueno y
uno malo **se comprime** (0,343 → 0,147) mientras el ruido se queda igual, o sea que la métrica
**distingue peor entre modelos** aunque el número absoluto sea mejor.

---

## 4. Ejecución — dominio **X** (no cambian el resultado, sólo el reloj)

`device` (cpu/gpu), `num_workers`, hilos de torch, concurrencia. **Están fuera de la receta a
propósito** (contrato ⑩): si se colaran dentro, lo entrenado en CPU quedaría incomparable con lo
entrenado en GPU. Hoy se fijan **8 hilos de torch en todas las máquinas**, para que dos máquinas con
distinto número de núcleos no entrenen distinto.

---

## 5. Lo que ya se corrió: los 23 recorridos, y cuáles cuentan

| Recorrido | Eje | Valores × semillas | Red | ¿Cuenta como estudio? |
|---|---|---|---|---|
| `fast-lr`, `fast-lr-2` | `lr` | 13×5, 6×5 | foveada L2 | Sí, pero sobre red y dataset viejos |
| `d1000-lr-1` | `lr` | 7×5 | foveada L2 | Sí — encontró el **borde derecho** |
| `p40-lr-L4` | `lr` | 4×5 | foveada **L4** | Sí — eje **plano**, el vigente se queda |
| `lr-alto-L4`, `-b` | `lr` | 3×3 | foveada L4 | Sí (3 semillas) — cierra por la derecha |
| `batch_size-1`, `-2`, `d1000-batch_size-1` | `batch_size` | 7×5 (×3) | foveada L2 | **NO** — los 105 runs pararon por el tope de 20 épocas |
| **`bs5-L4`** | `batch_size` | 5×5 | foveada L4 | **Sí** — el bueno |
| `p40-confirm-n_layers` | `n_layers` | 4×5 | foveada | Sí |
| **`nl5-L4`** | `n_layers` | 4×5 | foveada L4 | **Sí** — replica el anterior sobre el dato nuevo |
| `proxy-c-d` | `d` | 6×5 | foveada L2 | Sí, pero L2 + tope de 20 épocas |
| **`d5-L4`** | `d` | 4×5 | foveada L4 | **Sí** — y deja el eje abierto |
| `plana-screen-s0/s1` | `n_layers`, `lr` | 6×1, 5×1 | plana 16 | **NO** — 1 semilla |
| `plana-confirm-s0/s1` | `n_layers`, `lr` | 3×5, 3×5 | plana 16 | Sí, pero **la red base era la equivocada** (N=16, sin el área que ve la foveada) |
| `bs-alto-fov` / `bs-alto-pl` | `batch_size` alto | 4×2 | foveada / plana 24 | **Tanteo** — acota, no declara |
| `pl-t-lr` | `lr` | 5×2 | plana 24 | **Tanteo** |
| `pl-t-bs` | `batch_size` | 5×2 | plana 24 | **Tanteo, INCOMPLETO** (3 de 10 runs) |
| `pl-t-nl` | `n_layers` | 5×2 | plana 24 | **Sin correr** (0 runs) |

### 5.1 Lo que dicen los tanteos que acaban de terminar (2 semillas — **no declaran ganador**)

`batch_size` alto, medido hoy:

| foveada (`bs-alto-fov`) | f1 | s/época | · | plana (`bs-alto-pl`) | f1 | s/época |
|---:|---:|---:|---|---:|---:|---:|
| **192** | **0,9386** | 30,6 | · | **170** | **0,9658** | 40,9 |
| 384 | 0,9362 | 34,0 | · | 340 | 0,9601 | 50,1 |
| 768 | 0,9316 | 39,9 | · | 680 | 0,9575 | 75,3 |
| 1536 | 0,9259 | 63,5 | · | 1360 | 0,9557 | 86,7 |

**Lectura provisional: el eje ya no es plano ahí arriba — baja monótonamente, y además se
encarece.** Con esto `batch_size` **queda acotado por la derecha** (el mejor del tanteo es el
extremo *inferior*, que es justo la condición que faltaba) y el vigente 85–192 sobrevive. Falta la
fase de 5 semillas para publicarlo.

`lr` de la plana (`pl-t-lr`): 0,00035 → 0,9633 · 0,0007 → **0,9649** · 0,0014 → 0,9615 · 0,0028 →
0,9442 · 0,0056 → 0,8858. **Acotado por la derecha** (0,0056 se hunde), óptimo interior en la zona
0,0007–0,0014: la misma meseta ancha que en la foveada.

⚠ **Los 0,96 de la plana contra los 0,93 de la foveada NO son la comparación del proyecto** y no
deben citarse como tal: son f1 **de ventana** y las dos redes ven áreas distintas.
[plan-cnn-plana.md](../../../docs/plan-cnn-plana.md) §4 exige métrica **de tarea** y 5 semillas para
esa comparación. Es, eso sí, un motivo más para hacerla ya.

---

## 6. Prioridad: qué estudiar y en qué orden

Criterio de ordenación: **(evidencia de que hay algo que ganar) × (lo que cuesta) × (si desbloquea
otra pregunta)**. Los costes son extrapolaciones de los 0,054 $/run medidos el 25-ago.

### Prioridad 1 — hacer ya

| # | Estudio | Coste | Por qué es lo primero |
|---|---|---|---|
| **1** | **`d ∈ [4,5,6]`, 5 semillas** (foveada) | 15 runs · ≈0,8 $ · ~1 h | El único eje con **evidencia directa sin cerrar**: sube monótono, el ganador está en el borde (p = 0,063) y `d=4` es **además el más barato**. Y no es un ajuste cualquiera: `d` **es la variable que define la visión foveada**. Si sube hasta 6, la tesis del proyecto se refuerza; si cae, queda acotada. [plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §7.3 ya lo dejó escrito como *«la pregunta abierta y la única accionable»*. |
| **2** | **Terminar el afinado de la plana**: `pl-t-bs` (los 7 runs que faltan), `pl-t-nl` (10) y la **fase 2 con 5 semillas** | ≈45 runs · ≈2,5 $ | **Bloquea la pregunta central del proyecto.** Comparar la foveada afinada contra una plana sin afinar mediría el afinado, no la arquitectura ([plan-cnn-plana.md](../../../docs/plan-cnn-plana.md) §6). Ya está a medias. |
| **3** | **Foveada vs plana por métrica de tarea**, 5 semillas, permutación exacta | 0 entrenamientos nuevos si se reusan los runs de (2) | **Es la pregunta que da nombre al proyecto** ([protocolo.md](../../../docs/protocolo.md) §6) y sigue sin contestar. Todo lo demás es afinar sin saber si la arquitectura gana. |
| **4** | **Aplicar (o rechazar) los knobs de F** | **0 $, minutos** | +0,065 a +0,261 de métrica de tarea **con los pesos que ya hay**. Es la mayor ganancia por euro del inventario. ⚠ Requiere **decisión del usuario (F15)**, porque re-escala todos los números publicados, y hay que sopesar que comprime la separación entre modelos. |

### Prioridad 2 — nunca medidos, con motivo para pensar que importan

| # | Estudio | Coste | Por qué |
|---|---|---|---|
| **5** | **`pos_weight`** ∈ {1, 2, 4, 8}, 5 semillas | 20 runs · ≈1,1 $ | El cuello de botella está **medido y es de detección**: con esquinas perfectas la reconstrucción da 0,97 y el mejor modelo real da 0,64 — un tercio de los párrafos no se detecta. `pos_weight` es el mando que ataca eso desde el entrenamiento (`threshold` lo ataca desde la inferencia). **Es la hipótesis más plausible de mejora grande que nadie ha probado.** |
| **6** | **`monitor: val_f1` vs `val_loss`**, 5 semillas | 10 runs · ≈0,6 $ | Hoy el checkpoint se elige por una métrica y se rankea por otra. Es barato y cierra una incoherencia declarada en todos los informes. |
| **7** | **`scheduler: cosine`** vs `none`, 5 semillas | 10 runs · ≈0,6 $ | Mejora esperable a priori y nunca medida. ⚠ Hay que diseñar con cuidado su interacción con `patience`. |
| **8** | **`channels` (anchura)** ∈ {8, 16, 24, 32}, 5 semillas | 20 runs · ≈1,1 $ | El indicio de 1 semilla dice que no aporta, pero **es un indicio, no un estudio**, y se tomó sobre L2 con 20 épocas. Además puede ir **hacia abajo**: si 8 canales empatan, la red es la mitad de cara. |
| **9** | **`k_center`** ∈ {3, 5, 7}, 5 semillas, **con métrica de tarea** | 15 runs · ≈0,8 $ | El único parámetro donde proxy y tarea se **contradicen en el signo** con la evidencia que hay. Barato y potencialmente revelador sobre el propio proxy. |
| **10** | **`pen_frac`** ∈ {0,05, 0,1, 0,2}, 5 semillas | 15 runs · ≈0,8 $ | Mando **exclusivo de esta arquitectura** —cuánto se solapan las dos vistas— y nunca mirado. La periferia ya salió una vez «sin aportar de forma medible»; esto dice si el problema es *cómo se cosen*, no *cuánta* hay. |

### Prioridad 3 — completar el mapa, sin prisa

| # | Estudio | Por qué está abajo |
|---|---|---|
| **11** | `merge` (concat / sum), `pool_mode` (avg / max) | 10 runs cada uno. Elecciones discretas nunca medidas; `pool_mode` gana interés **sólo si `d` sube** (comprimir más hace que importe más cómo se resume). |
| **12** | `s_center` / `s_periph` | Son mandos de **coste**, no de calidad. Interesan si algún día el reloj aprieta. |
| **13** | `weight_decay`, `optimizer`, `smooth_l1_beta`, `lambda_pos` | Sin indicio de que sean el problema. `optimizer` está casi vacío hoy (con `weight_decay=0`, adam ≡ adamw). |
| **14** | `patience`, `epochs` | No son calidad: son **criterio de parada y guarda**. Ya están en valores seguros medidos. |
| **15** | `pad_mode` | Afecta sólo a las ventanas de borde. |
| **16** | Re-medir **`lr`** sobre el dataset del 24-ago | Cerrado por los dos lados, pero sobre el dato anterior — y el dato nuevo movió la escala +0,0095. Es una **replicación**, no un descubrimiento. |
| **17** | `N` / `c_frac` | **No son barribles**: cambiarlos exige regenerar el dataset de ventanas. Es un estudio de otro tipo (¿qué tamaño de ventana conviene?), no un eje. |

### Lo que este informe recomienda **no** hacer

- **Subir `n_layers` por encima de 4 tal cual.** Está medido dos veces en las dos arquitecturas: no
  es que sea peor, es que **no arranca de forma fiable**. Antes hay que cambiar algo estructural.
- **Repetir los barridos de `lr` en la zona plana.** Cinco recorridos dicen lo mismo.
- **Fiarse de un cribado de 1 semilla para recortar un rango.** Es la lección literal de
  [plan-tres-ejes.md](../../../docs/plan-tres-ejes.md) §7.3: *«un estudio que se declara inválido
  para decidir el ganador tampoco vale para decidir dónde mirar»*.

---

## 7. Resumen en una pantalla

- **Barridos y cerrados:** `lr` (foveada, por los dos lados) y `n_layers` (por los dos lados, en las
  dos arquitecturas).
- **Barridos y abiertos por un lado:** **`d`** (sube hacia la derecha — *lo más accionable que hay*)
  y `batch_size` (el tanteo de hoy sugiere que ya se cierra por arriba; falta confirmarlo).
- **Medidos con 1 semilla, o sea sin medir:** `channels`, `k_center`, `k_periph`, `s_center`,
  `s_periph`.
- **Nunca tocados:** `pen_frac`, `merge`, `pool_mode`, `pad_mode`, `weight_decay`, `scheduler`,
  `optimizer`, `momentum`, `patience`, `lambda_pos`, `pos_weight`, `smooth_l1_beta`, `monitor`.
- **No barribles por diseño:** `N` y `c_frac` (los fija el dataset).
- **Gratis y sin aplicar:** los tres knobs de inferencia, con +0,065 a +0,261 medidos.
- **Sin contestar:** la pregunta del proyecto — **¿gana la foveada a la CNN plana?**
