# Los primeros resultados con la geometría nueva: el borde quiere ser **ancho y nítido**

Fecha: **2026-08-26**. Alcance: los ocho recorridos de prioridad que la flota corrió entre el 25 y
el 26 de agosto, leídos desde los artefactos del repo con `scripts/estudio_informe.py`, que aplica
las reglas R1–R6 escritas **antes** en
[plan-prioridades-2026-08-25.md](../../../docs/plan-prioridades-2026-08-25.md).

⚠ **Este informe no midió nada.** Los runs los produjo la flota; aquí sólo se leen, se ordenan y se
cruzan con lo que la [reparametrización de la geometría](../../../instructionsNewNN.md) dejó
preguntado. Cada número sale de `sweeps/<nombre>/informe.json` y se puede reproducir con el comando
que acompaña a cada tabla.

---

## 0. El titular, en tres líneas

1. **El borde difuso quiere ser más ancho *y* más nítido, y las dos cosas se midieron por
   separado** — que es exactamente lo que la reparametrización hizo posible.
2. **A coste constante (N=20), el óptimo del ancho es interior: 8 px.** El eje **queda acotado por
   los dos lados** por primera vez; a 16 px es *peor* que el vigente de 4.
3. ⚠ **Pero pagando, se gana bastante más**: con esos 8 px vistos **sin comprimir**
   (`border_reduce=1`, N=32) el f1 sube a **0,9574** contra 0,9341 del vigente, con **p = 0,008**,
   el mínimo alcanzable con 5 semillas. Es **la mejora más grande y más significativa del
   inventario entero**, y cuesta **1,8× por época**.

---

## 1. El ancho del borde, a coste constante — `borde-ancho`

**La pregunta**: *¿ayuda ver más contexto sin pagar más?* Se barre `border_px` con
`border_reduce` **atado** (mecanismo `couple`) para que el anillo se quede en 2 celdas y **N=20 en
los cinco puntos**: mismo tensor, mismos parámetros, mismo coste.

```powershell
.\.venv\Scripts\python.exe scripts\estudio_informe.py --sweep borde-ancho --vigente 4
```

| `border_px` | `border_reduce` | N | f1 (media de 5) | sem | s/época | p vs. vigente |
|---:|---:|---:|---:|---:|---:|---:|
| 4 *(vigente)* | 2 | 20 | 0,9341 | 0,0022 | 36,8 | — |
| **8** | 4 | 20 | **0,9408** | 0,0021 | 42,5 | **0,063** |
| 10 | 5 | 20 | 0,9385 | 0,0019 | 41,0 | 0,167 |
| 12 | 6 | 20 | 0,9376 | 0,0016 | 47,4 | 0,214 |
| 16 | 8 | 20 | 0,9321 | 0,0024 | 50,5 | 0,563 |

**R1 ✅** los 25 runs pararon por `patience` (33–87 épocas, tope 150): mide calidad, no presupuesto.

**Lo que cierra.** El eje venía **abierto por la derecha** desde `d5-L4`: allí ganaba el extremo del
rango (los 8 px, entonces escritos `d=4`) y no se sabía si seguía subiendo. **Ahora se sabe: no.**
Hace pico en 8 y baja, y **16 px es peor que el vigente de 4**. Es un óptimo **interior**, que es la
única clase de óptimo que significa algo.

**Y confirma la predicción física.** El análisis que acompañó a la reparametrización
([instructionsNewNN.md §2.2](../../../instructionsNewNN.md)) advertía que sobre imágenes de 60×80
el borde deja de ser contexto y empieza a ser **relleno replicado**: a 16 px, el 26 % del anillo es
`pad_mode: edge`. La caída aparece justo ahí. No es una coincidencia que valga como prueba, pero sí
es la clase de acuerdo que da confianza en el modelo mental.

⚠ **R4 ❌ — el vigente no se mueve.** 8 px le saca +0,0067 con **p = 0,063**, y el umbral escrito
antes era 5 %. Otra vez el borde: es el segundo recorrido consecutivo en el que este eje se queda a
las puertas. Con el resultado de §2 en la mano, la pregunta deja de ser interesante en esta forma.

---

## 2. La resolución del borde — `red-fov`. **El resultado grande**

**La pregunta**, que es la otra mitad y **no se podía formular antes de la reparametrización**: *a
igual área de contexto, ¿ayuda verla con más resolución?* Se fija `border_px = 8` y se mueve
`border_reduce`.

```powershell
.\.venv\Scripts\python.exe scripts\estudio_informe.py --sweep red-fov --vigente 2
```

| `border_reduce` | celdas de anillo | N | f1 (media de 5) | sem | s/época | p vs. `2` |
|---:|---:|---:|---:|---:|---:|---:|
| **1** *(sin comprimir)* | 8 | **32** | **0,9574** | 0,0017 | 69,8 | **0,008** |
| 2 | 4 | 24 | 0,9472 | 0,0013 | 44,1 | — |
| 4 | 2 | 20 | 0,9408 | 0,0021 | 39,4 | 0,040 |

**R1 ✅** los 15 runs pararon por `patience` (35–73 épocas).

**Sube monótono, y la diferencia es real.** `border_reduce=1` gana a `2` por **+0,0102 con
p = 0,008** — el p **mínimo alcanzable** con 5 contra 5 semillas. Y `4` pierde contra `2` con
p = 0,040. Las tres bandas min–max **no se solapan**: 0,9548–0,9641 · 0,9429–0,9499 · 0,9333–0,9446.

**Contra el vigente del proyecto** (`border_px` 4, `border_reduce` 2 → 0,9341) la mejora es de
**+0,0233**, la mayor del inventario con diferencia.

⚠ **No es gratis, y hay que decirlo al citarlo.** N pasa de 20 a 32, y la cabeza es
`Linear(2·C·N², 12)` — el 97 % de los parámetros: **+156 %**. El reloj lo confirma: **69,8 s/época
contra 36,8** del vigente, **1,9×**. Este eje **no es cost-neutral y nunca pretendió serlo**; por
eso se midió aparte del §1, y por eso las dos tablas no se pueden leer como si fueran el mismo
experimento.

**Y queda abierto por la izquierda**: `border_reduce=1` es el extremo del rango — no hay nada más
nítido que «sin comprimir». Lo que **no** está acotado es la combinación: con el borde sin comprimir,
¿cuál es el ancho óptimo? El §1 lo respondió **a 2 celdas**, no a 8.

---

## 3. El solape entre las dos vistas — `ov-fov`

**La pregunta**, también **nueva**: la spec eligió que las dos ramas *contribuyan* en una banda
compartida, y esa elección **nunca se había contrastado** porque el parámetro tenía suelo de 1 px.
Ahora `overlap_fovea_px = 0` es legal, y es el control: **ramas disjuntas**.

| `overlap_fovea_px` | f1 (media) | sem | n | s/época | p vs. `2` |
|---:|---:|---:|---:|---:|---:|
| 4 | 0,9379 | 0,0079 | 2 | 48,2 | 0,467 |
| 2 *(vigente)* | 0,9311 | 0,0045 | 4 | 58,3 | — |
| 1 | 0,9273 | 0,0026 | 5 | 48,1 | 0,468 |
| **0** *(disjuntas)* | **0,9186** | 0,0030 | 5 | 46,7 | 0,063 |

⚠⚠ **Este recorrido NO declara nada, y hay que resistirse a que lo parezca.** Está **incompleto**:
16 de 20 puntos, con **3 runs excluidos por quedarse a medias** (la máquina murió antes de que
convergieran). El punto de 4 px sólo tiene **2 semillas**, y con 2 contra 5 sólo hay **15 arreglos**
de permutación: el p **mínimo alcanzable es 0,133**, así que **R4 no puede declarar significación al
5 % aunque la diferencia fuese enorme**.

**Lo único que se puede decir, y con reservas**: la tendencia es **monótona y va en el sentido de la
spec** — más solape, mejor; y el **0 (ramas disjuntas) es el peor de los cuatro** por un margen que
dobla a cualquier otro salto (−0,0125). Si eso se sostiene al completarlo, sería la primera
evidencia directa a favor del solape contributivo. **Hay que terminarlo antes de citarlo.**

---

## 4. Los ejes que se cierran en negativo (y eso también es información)

Cuatro recorridos completos, 5 semillas, que **descartan** mandos que estaban en la lista de
«nunca medidos, con motivo para pensar que importan». Ninguno mejora al vigente:

| Recorrido | Eje | Ganador | Vigente | Veredicto |
|---|---|---|---|---|
| `pw-fov` | `pos_weight` | **1,0** *(el vigente)* | 1,0 | ⚠ **Se cierra en contra, y fuerte**: 4,0 pierde −0,0204 y 8,0 pierde −0,0561, ambos con **p = 0,008**. |
| `kc-fov` | `k_center` | **3** *(el vigente)* | 3 | Se cierra en contra: 5 pierde −0,0114 (p = 0,024) y 7 pierde −0,0134 (p = 0,008). |
| `mon-fov` | `monitor` | `val_f1` (+0,0059) | `val_loss` | **p = 0,214** — no cruza. El vigente se queda; la incoherencia monitor≠objetivo **no cuesta nada medible**. |
| `sch-fov` | `scheduler` | `none` *(el vigente)* | `none` | `cosine` da −0,0012 con **p = 0,857**: no hace absolutamente nada. |
| `ch-fov` | `channels` | `[16]×4` *(el vigente)* | `[16]×4` | Incompleto (18/20, 2 excluidos). Ni ensanchar (24, 32) ni estrechar (8) mejora; **8 canales se hunde** (0,9021 con sem 0,0147). |

⚠ **`pos_weight` merece una línea aparte** porque era **la hipótesis más plausible del inventario**:
el cuello de botella está medido y es de *detección* (un tercio de los párrafos no se detecta), y
`pos_weight` era el mando que atacaba eso desde el entrenamiento. **No funciona: empeora, y mucho.**
Eso reorienta el problema — si subir el peso del positivo hunde el f1, lo que falla no es el
equilibrio de la pérdida.

---

## 5. Qué se deduce, y qué haría falta medir ahora

**La geometría del borde es el eje que más mueve la aguja de todo el inventario**, y hasta esta
semana no se podía preguntar bien. Las dos mitades separadas dan:

- **cuánto contexto**: óptimo interior en 8 px *cuando el anillo tiene 2 celdas*;
- **con cuánta resolución**: cuanto más nítido mejor, +0,0233 sobre el vigente, con el mejor p del
  proyecto — a cambio de **1,9× de reloj y 2,5× de parámetros**.

**Lo que falta, por orden:**

1. ⚠ **El plano `(border_px, border_reduce)` no está barrido, sólo sus dos ejes por el punto
   vigente.** El §1 fijó el anillo en 2 celdas y el §2 fijó el ancho en 8 px. El mejor punto medido
   —8 px sin comprimir, N=32— es la **esquina** de las dos rectas, no un óptimo demostrado. La
   pregunta natural: con `border_reduce=1`, ¿cuál es el `border_px` óptimo? Puede estar por debajo
   de 8 (menos área, más nítida) y salir más barato.
2. **Terminar `ov-fov`** (4 puntos) y **`ch-fov`** (2 puntos). Son los dos únicos incompletos y el
   primero contrasta una elección de diseño de la spec.
3. ⚠ **Nada de esto se ha llevado a la métrica de tarea.** Todo son f1 de **ventana**, que es el
   proxy. El proxy está validado en ejes de D (+0,956) y en un eje de C (+1,000, el viejo `d`), pero
   [ya exageró una vez](../../../docs/metrica-de-tarea.md) por un factor de dos en `n_layers`.
   **+0,0233 de ventana no son +0,0233 de párrafos bien reconstruidos**, y la comparación que da
   nombre al proyecto —foveada contra plana— sigue sin hacerse.
4. **La comparación con la plana hay que rehacerla si se adopta N=32.** La plana `plana-24-single`
   se eligió para igualar parámetros con la foveada de N=20. Contra una foveada de N=32 **ya no es
   un control de coste equivalente**, y compararlas mediría el tamaño.

---

## Procedencia

Todos los números salen de `sweeps/<nombre>/informe.json`, generados por
`scripts/estudio_informe.py`, que no re-implementa ningún criterio: aplica R1–R6 con las funciones
del proyecto (`sweep_trials`, `suggest_winner`, `fv.metrics.permutation_test`).

⚠ **Un arreglo hecho al escribir esto**: `estudio_informe.py` tenía el eje con default `"lr"`
**cableado**. Si no se pasaba `--eje`, no fallaba: imprimía la tabla entera con el eje a `None` y un
ganador llamado `None` — creíble y falsa, que es peor que un error. Ahora el eje **se deriva del
`space` del recorrido** y sólo se pasa a mano para releerlo por otro campo; si el espacio no tiene
un único eje, se niega diciéndolo. Es el mismo dato en dos sitios: el spec ya lo sabía.
