# La CNN plana contra la foveada — el criterio, escrito antes de medir

**Estado: el criterio y el código. Ni un run entrenado todavía.**
Este documento cierra la decisión **F12** de [decisiones.md](decisiones.md) y construye el control
que el **§6 de [protocolo.md](protocolo.md)** lleva pidiendo desde el día 1: *¿la periferia foveada
mejora el reconocimiento respecto a una CNN plana del mismo coste?*

Se escribe **antes** de que exista un solo run, como `plan-40h.md` y `plan-lr-L4.md`, y por la
misma razón: un criterio escrito después de ver los números no es un criterio.

---

## 1. Qué estaba roto en la pregunta

«Poner la periferia a cero» **no** produce una CNN plana. Medido el 2026-08-09:

1. **La guarda lo prohíbe**: `c_frac→1` sale `no_periphery` en las tres formas probadas
   (`N=16 c_frac=1,0`, `N=20 c_frac=1,0`, `N=18 c_frac=0,95`).
2. **Y saltándose la guarda, tampoco.** La penetración es `max(1, round(N·pen_frac))` y **nunca
   puede ser 0**, así que con `periph_out=0` la máscara periférica no se apaga: se convierte en un
   anillo sobre el borde de la propia ventana. Con `pen_frac=0,1` la rama «periférica» sigue viendo
   **112 de 256 px** de la fóvea; con `pen_frac=0,0`, **60 de 256**. La segunda rama no muere nunca.

La causa es que **hay dos grados de libertad confundidos en uno**:

| | pregunta | parámetro hoy |
|---|---|---|
| ① | ¿la periferia **comprime**? | `d` (`d=1` = no comprime) |
| ② | ¿hay **dos ramas enmascaradas** por región, o **una sola** sobre todo el input? | **ninguno** |

② no era expresable. Por eso la nota de F12 decía que `d=1` «sigue siendo dos ramas enmascaradas»:
ese es exactamente el hueco. La CNN plana del §6 es **②, no ①**.

### 1.1 Y el contrato ①a acota lo que se puede pedir

`center_out == window_size` del dataset (16 px en `dirty1000-80px-16px`), y
`center_out = round_to_even(N·c_frac)`. Por tanto **`N = 16 + 2·periph_out`**:

- `periph_out = 0` → `N = 16`: la red solo puede ver **la ventana**, sin contexto. Es el único
  caso de una sola rama expresable con `c_frac`, y es **el control más pobre de los cuatro**.
- Cualquier área mayor exige `periph_out ≥ 1`, es decir **anillo**, es decir dos ramas.

De ahí que `c_frac=1,0` no baste y haga falta ②.

---

## 2. El cambio: `regions`, un campo de C

Campo nuevo en el dominio **C**, `regions ∈ {"split", "single"}`, default **`"split"`** — el
comportamiento de hoy, bit a bit.

- **`split`** (default): lo de siempre. Dos ramas convolucionales sobre el input compuesto,
  enmascaradas por región, que se funden por `merge`.
- **`single`**: **una** rama sobre el input `N×N` **sin enmascarar**. `k_periph`, `s_periph` y
  `merge` dejan de aplicar. La cabeza sigue prediciendo las 4 esquinas de la ventana etiquetada
  (los 16×16 centrales), así que **①a se sigue cumpliendo y el dataset B es el mismo**.

**`build_masks` no se toca.** En modo `single` simplemente no se llama. Es deliberado: esa función
la importa todo el proyecto, y cambiarla habría puesto en riesgo la geometría de los 164 runs ya
medidos. El único aflojamiento en `fv.fovea` es que `no_periphery` y `penetration_too_large` dejan
de aplicar **cuando `regions="single"`**, donde no describen nada.

### 2.1 Lo que esto NO cambia

- Ningún artefacto existente. `full_config` rellena `regions="split"` por ausencia, así que los
  runs, redes y recorridos ya en disco significan exactamente lo que significaban.
- El nombre de los módulos en el `state_dict` de `split` (`center_convs.N`, `periph_convs.N`), así
  que **los checkpoints actuales siguen cargando**.

---

## 3. La familia de controles

Todos contra **el mismo B** (`dirty1000-80px-16px`, ventana 16 px), **la misma receta** (`plan40`)
y **las mismas semillas**. Todos con `n_layers=4`, `channels=[16,16,16,16]` salvo donde se diga.

| | control | `regions` | `N` | `c_frac` | `d` | área vista | ramas | qué aísla |
|---|---|---|---|---|---|---|---|---|
| **base** | foveada L4 | split | 20 | 0,8 | 2 | 24×24 | 2 | — (es el campeón actual) |
| **A** | mismo tensor de entrada | single | 20 | 0,8 | 1 | 20×20 | 1 | mismo coste de convolución por capa |
| **B** | **misma área original** | single | 24 | 16/24 | 1 | 24×24 | 1 | **la comparación central**: misma información, sin foveación |
| **C** | solo la ventana | single | 16 | 1,0 | — | 16×16 | 1 | ¿el contexto periférico aporta algo? |
| **D** | mismos parámetros | single | 24 | 16/24 | 1 | 24×24 | 1 | misma capacidad (canales ↑ hasta ~168k params) |
| **E** | foveada sin comprimir | split | 20 | 0,8 | 1 | 20×20 | 2 | separa «dos ramas» de «comprimir» |

**B es la que contesta al §6.** Las demás acotan *por qué* gana o pierde:

- Si **base > B**: foveación gana con la información igualada → la arquitectura está justificada.
- Si **base ≈ B**: la foveada no daña, pero tampoco aporta; su ventaja sería solo de coste.
- Si **base < B**: comprimir la periferia **destruye** información útil, y lo barato sale caro.
- **C** dice cuánto vale el contexto *en absoluto*. Si `C ≈ base`, toda esta arquitectura sobra.
- **E** separa las dos hipótesis: si `E ≈ base`, comprimir es gratis; si `E > base`, la compresión
  es la que estorba (y no las dos ramas).
- **D** es el guardarraíl de capacidad. ⚠ Con reserva: el plan de 40 h midió que **ensanchar canales
  no mejora** (+0,0046, dentro de δ, con 2× parámetros y **más** coste), así que igualar por
  parámetros probablemente le regale a la plana capacidad que no sabe usar. Se mide igual, porque
  no medirlo dejaría abierta la objeción «ganó por tener más cabeza».

### 3.1 Por qué son estudios separados y no un eje

`N` y `c_frac` están **rehusados como ejes** (`axis_breaks_window_size`, [spec.py:31](../src/fv/sweeps/spec.py#L31)),
y los controles varían `N`. Así que **no** son puntos de un recorrido: son **redes base distintas**,
y se corre el **mismo plan** sobre cada una. Es la comparación pareada, que además es lo que la
disciplina OAT del proyecto pide.

`regions` **sí** es eje barrible (no toca `center_out`), por si alguna vez interesa un A/B directo
con todo lo demás fijo — pero **no es como se responde al §6**, porque las áreas vistas difieren.

---

## 4. El criterio, antes de mirar

1. **N semillas o no es nada.** 5 semillas por control, como el recorrido de `lr`. Un control con
   una semilla no entra en la tabla.
2. **Se decide por la métrica de tarea** (`paragraph_f1` por imagen), **no** por el f1 de ventana.
   Está medido (2026-08-08) que el proxy **exagera**: en `n_layers` la ganancia se encogió a la
   mitad (+0,0488 → +0,0224) y las bandas disjuntas dejaron de serlo. El f1 de ventana se reporta
   al lado, como informe, nunca como veredicto.
3. **La diferencia tiene que sobrevivir a `fv.metrics.permutation_test`** (exacto, 252 arreglos con
   5+5 semillas). Se publica el **p**, salga como salga.
4. ⚠ **El suelo de ruido ya conocido**: `sem` por run ≈ **±0,023** con 200 imágenes de val
   (`dirty1000`). Una diferencia menor que eso **no se afirma**, se declara indistinguible.
5. **Se publica el resultado que salga.** Si la plana empata o gana, eso es el resultado —
   exactamente como R3 del plan de `lr`. Esta arquitectura es la hipótesis del proyecto, no su
   conclusión.

## 5. Coste

Sin medir todavía en esta máquina. Referencia: la foveada L4 va a **106 s/época** y el recorrido de
`lr` estimó ~70 épocas hasta que salta `patience`. Seis controles × 5 semillas es del orden de
**30 runs**, comparable al recorrido que corre ahora (~33 h). **No se lanza nada hasta que
`p40-lr-L4` cierre**, y el coste se mide con la máquina libre — el micro-benchmark bajo carga ya
mintió una vez (34 h estimadas contra 22 h reales, plan-40h.md §7).

---

## 6. Los parámetros óptimos DE la plana — criterio escrito antes (2026-08-25)

> Añadido antes de que exista un solo run de esta parte. §1–§5 fijaron *qué red* es el control;
> esto fija *con qué hiperparámetros* se la entrena, que es un paso previo obligatorio: comparar
> la foveada afinada contra una plana sin afinar mediría el afinado, no la arquitectura.

### 6.1 La red: `plana-24-single`, y por qué esos números

La comparación exige dos cosas a la vez, y ninguna es negociable:

1. **La misma entrada — la misma información, no el mismo tensor.** La foveada tiene un tensor
   de 20×20, pero el área **original** que cubre es **24×24**: la periferia comprime ×2. MEDIDO:
   `dims_of(...).original_size == 24`. Así que la plana es `N=24`, `c_frac=16/24` (para que
   `center_out` siga siendo la ventana de 16, contrato ①a) y `d=1`.
2. **Aproximadamente los mismos parámetros.** Con `channels=[16]×4` la plana sale a **117.724**
   contra **167.852** de la foveada: **0,70×**. No es un detalle: la cabeza es el 92 % del modelo,
   y una rama sobre 24×24 da 9.216 *features* planas contra las 12.800 de dos ramas sobre 20×20.
   Ensanchando a **22 canales**: **165.430, o sea 0,99×**. MEDIDO.

O sea que ésta es la red que funde los controles **B** y **D** de §3 en una sola: misma área *y*
mismos parámetros. Se hace así porque son las dos objeciones que se le pueden poner al resultado
—«vio menos» y «tenía menos cabeza»— y separarlas costaría dos estudios enteros.

⚠ **`d` no se barre, y es el único eje del estudio foveado que no se replica.** Con
`regions=single`, subir `d` agranda el **área original** (`d=2` → 32×32, medido), así que dejaría
de ser «la misma entrada» y rompería justo la premisa. Se dice aquí para que la ausencia no se lea
como un olvido.

⚠ **Un arreglo que hizo falta para que el eje `n_layers` midiera lo que dice.** `expand_points`
reescribía los canales a `[16]×L` al mover la profundidad, así que sobre una base de `[22]×4`
habría movido **anchura y profundidad a la vez**, sin decirlo. Ahora conserva el ancho uniforme de
la base. Para las bases `[16]×L` —todos los recorridos anteriores— el ancho uniforme *es* 16, así
que **no cambia nada de lo ya medido** (163 tests en verde).

### 6.2 Dos fases, y qué puede decir cada una

**Fase 1 — tanteo (2 semillas, rangos anchos).** Su único trabajo es **acotar**. Un óptimo no se
hereda al cambiar de arquitectura: es literalmente el motivo por el que existió
[plan-lr-L4.md](plan-lr-L4.md), que tuvo que rebarrer `lr` al cambiar de L2 a L4. Partir de los
rangos de la foveada sería suponer la respuesta.

| eje | rango del tanteo | span | por qué |
|---|---|---|---|
| `lr` | 0,00035 · 0,0007 · 0,0014 · 0,0028 · 0,0056 | **16×** | el óptimo de la foveada (0,0014) queda en el centro, como ancla, no como predicción |
| `batch_size` | 24 · 43 · 85 · 170 · 340 | **14×** | igual; y cubre los dos vecindarios que los estudios viejos señalaron (25 y 85) |

⚠ **Con 2 semillas la permutación exacta da 2 arreglos: el tanteo NO puede declarar ningún
ganador, y no lo intenta.** Es la misma regla que [plan-40h.md](plan-40h.md) §2 escribió para su
cribado. Lo que sí distingue —y basta para acotar— es una zona donde el entrenamiento converge de
otra donde diverge o se arrastra.

⚠ **`n_layers` no entra en el tanteo, a propósito.** No hay nada que acotar: el rango natural es
discreto y pequeño, y el estudio foveado ya lo dejó cerrado por los dos lados en `[2..5]`. Se
barrerá directamente en la fase 2 con ese mismo rango, que es además lo que hace comparables los
dos estudios.

**Fase 2 — final (5 semillas, rangos acotados por la fase 1).** Mismas reglas R1–R6 de
[plan-tres-ejes.md](plan-tres-ejes.md) §5, mismo reparto —**una máquina por recorrido × semilla**—
y mismo tope de 150 épocas con `patience` decidiendo. Los rangos de la fase 2 se pasan por la línea
de órdenes **a propósito**: así la decisión que se toma al ver el tanteo queda escrita en el
comando y aquí, en vez de escondida en una constante.

### 6.3 Coste

Tanteo: 20 runs, **una máquina por punto**, ~1 h de reloj y **0,96–1,22 $** estimados. Se reparte
por punto y no por semilla porque un tanteo lento no sirve de tanteo: por semilla serían 3-4 h.

⚠ El reparto por punto es aceptable **aquí** porque el tanteo no decide nada (§6.2) y porque
`--cpu 'E5-26'` fija la familia, con lo que el ruido de máquina es cero
([plan-lr-alto.md](plan-lr-alto.md) §7.4, ahora confirmado con 5 pares idénticos bit a bit en
[plan-tres-ejes.md](plan-tres-ejes.md) §7). La fase 2, que sí decide, vuelve al reparto por semilla.

**MEDIDO antes de lanzar**: la plana va a **125 ms/paso** contra 113 de la foveada en el droplet de
control (2 vCPU) — **1,11× más lenta**, que es lo que cuesta una rama sobre 24×24 con 22 canales
frente a dos sobre 20×20 con 16.
