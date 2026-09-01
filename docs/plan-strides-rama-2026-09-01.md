# Plan — `s_center` / `s_periph`: cuántas features llegan a la cabeza (2026-09-01)

**Estado: el mecanismo YA existe y está verificado como barrible; NO se ha medido nada
con semillas.** Este documento fija **qué se mide y cómo se lee, antes de mirar**, que es
la regla de [protocolo.md](protocolo.md) §1 y la **R13** de
[reglas-de-diseño](https://github.com/stalinbeltran/telegram-coordinator/blob/main/docs/reglas-de-diseno.md).
Escribirlo después convierte cualquier resultado en la confirmación de lo que ya se creía.

**Origen del encargo:** el inventario de parámetros
([#síntesis 2026-08-25](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/sintesis/2026/08-agosto/2026-08-25-parametros-y-prioridad-de-estudios.md))
deja `s_center` y `s_periph` en el puesto **12**, descritos como *«mandos de coste, no de
calidad; interesan si algún día el reloj aprieta»*, medidos con **1 semilla y borrados**.
El dueño pide reabrirlos por otra razón: **el número de features que llegan a la cabeza es
desproporcionado para lo simple del problema**.

---

## 0. ⚠ SUPUESTOS PENDIENTES DE CONFIRMAR — este plan NO está congelado

Seis decisiones están tomadas **por defecto** abajo y esperan confirmación. Hasta que se
confirmen, este documento es un borrador: **congelarlo es responder §0**. Se listan aquí
arriba, y no enterradas, porque cambiar cualquiera de ellas cambia lo que el estudio mide.

| # | Decisión | Defecto tomado aquí | Dónde se discute |
|---|---|---|---|
| 1 | rango del tanteo | `{1, 2, 3, 4}` | §5.2 |
| 2 | ancla de `s_periph` | **independiente**: `s_center` = 1 (el vigente), no el ganador | §5.4 |
| 3 | control iso-features | **NO se corre** (+4 runs) | §4.3 |
| 4 | tope de épocas | **300**, no 150 | §5.3 |
| 5 | brazo diagonal `s_center` = `s_periph` | **NO se corre** | §7 |
| 6 | prioridad | **detrás de `do-v`**; no adelanta nada | §8 |

---

## 1. La observación que lo motiva, en números

Base vigente `ws16-p2-d2-L4`, la de todos los estudios de la tabla. *Medido el 2026-09-01
con `fv.models.network_trace`:*

| | |
|---|---:|
| lado de la vista compuesta (`N`) | **20 px** |
| celdas de salida **por rama** | 20 × 20 = **400** |
| canales por rama | 16 |
| features que llegan a la cabeza (`concat` de dos ramas) | **12.800** |
| salidas de la red (4 esquinas × [existe, x, y]) | **12** |
| **razón features : salidas** | **1.067 : 1** |
| parámetros totales | 168.652 |
| **de ellos, en la `Linear` final** | 153.612 = **91,1 %** |

La red tiene **una** capa densa, y esa capa se lleva nueve de cada diez pesos. Es el mismo
hecho que ya motivó poner ahí el `dropout` (*«donde está el 97 % de los parámetros»*,
`builder.py`); esto lo ataca por el otro lado: **no regularizando la cabeza, sino
haciéndola más pequeña.**

### 1.1 Y de dónde salen esas 12.800

De una **decisión de arquitectura** que conviene tener presente al leer los resultados: las
máscaras se aplican a la **entrada** (opción A), no a la salida, así que **cada rama
convoluciona sobre toda la vista de 20×20 y produce un mapa completo de 20×20** — aunque la
máscara del centro cubra 256 celdas y la del anillo otras 256.

⚠ **Lo que NO es** — y se comprobó antes de escribirlo, porque parecía obvio: esas celdas
de fuera de la máscara **no están muertas**. *Medido el 2026-09-01* comparando la salida de
cada rama ante entrada nula y ante entrada aleatoria: con `s`=1 están vivas **784 de 800**
(98 %). Con kernel 3 y 4 capas el campo receptivo es de 9 px, así que la respuesta cruza la
frontera de la máscara y las celdas de fuera siguen dependiendo del dato. **Las únicas 16
constantes** son el cuadrado central de la rama periférica, a más de 4 px del anillo.

O sea: la cabeza es desproporcionada por su **tamaño frente al problema** (1.067 features
por salida), no porque esté leyendo ceros. Eso es lo que este estudio ataca.

## 2. Qué son exactamente estos dos mandos

Del [glosario](glosario.md): en este proyecto hay **tres** cosas llamadas *stride*, y sólo
una es ésta.

| stride | dominio | qué hace | dónde se estudia |
|---|---|---|---|
| de **extracción** | B | cada cuántos px se corta la siguiente ventana etiquetada | [plan-stride-2026-08-27.md](plan-stride-2026-08-27.md) |
| de **inferencia** | F | cada cuántos px se evalúa al recorrer una página | knob de F, gratis |
| **`s_center` / `s_periph`** | **C** | el paso del kernel **dentro de cada rama** | **este documento** |

Dos hechos del código que fijan el alcance (`fv/models/builder.py`):

1. **El stride va SOLO en la primera capa** (D-S1); las demás son stride 1. El submuestreo
   total es `s`, no `s^L`. Eso es lo que mantiene a `n_layers` fuera de la ecuación.
2. **`merge: concat`** (el default vigente) **aplana y concatena**, así que **tolera que las
   dos ramas tengan formas distintas**. Es lo que permite estudiarlos por separado. Con
   `merge: sum` el validador exige `s_center == s_periph`
   (`merge_sum_needs_equal_strides`), y `sum` **nunca se ha medido**.

**Respuesta directa a «¿es posible manejarlos por separado?»: sí, y sin tocar código.**

## 3. Lo que ya está medido sin gastar un céntimo

### 3.1 Features y parámetros

*Medido el 2026-09-01 con `fv.models.network_trace` sobre `ws16-p2-d2-L4`.* La salida de una
rama es `floor((20 + 2·1 − 3)/s) + 1`:

| `s` | mapa por rama | features (una rama a `s`, la otra a 1) | params | vs base | features (**ambas** a `s`) | params | vs base |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 20×20 | **12.800** | 168.652 | 1,000× | 12.800 | 168.652 | 1,000× |
| **2** | 10×10 | 8.000 | 111.052 | 0,658× | **3.200** | 53.452 | 0,317× |
| **3** | 7×7 | 7.184 | 101.260 | 0,600× | 1.568 | 33.868 | 0,201× |
| **4** | 5×5 | 6.800 | 96.652 | 0,573× | 800 | 24.652 | 0,146× |

⚠ **Las dos ramas producen exactamente el mismo mapa**, así que **`s_center` y `s_periph`
recortan la cabeza en la misma cantidad**. Cualquier diferencia entre los dos estudios será,
por construcción, **de contenido de información y no de tamaño**. Es lo que hace la
comparación limpia — y es la razón de fondo por la que la pregunta *«¿cuál de los dos?»*
tiene respuesta.

### 3.2 Velocidad

*Medido el 2026-09-01 en este droplet (2 hilos, lote 85, forward+backward+step, 20
repeticiones tras 3 de calentamiento). El **absoluto no es transferible** a la máquina de
Vast; la **razón** sí, aproximadamente.*

| `s` (ambas ramas) | ms/paso | ratio |
|---:|---:|---:|
| 1 | 104,9 | 1,00× |
| 2 | 28,4 | **0,27×** |
| 3 | 17,2 | 0,16× |
| 4 | 13,2 | 0,13× |

⚠ **Ese ratio es un techo, no una promesa.** Mide **sólo el modelo**: el dataloader, que no
baja con el stride, no está incluido. En el entrenamiento real el ahorro será menor.

**Aun así, la consecuencia de diseño se sostiene: este eje NO es cost-neutral, es
cost-NEGATIVO.** Es el único de la cola así — `dropout` y `edge_inputs` son neutrales,
`patience` y `border_reduce` son caros. Un empate aquí **ya es una victoria**, y eso cambia
el criterio de §6.

### 3.3 Campo receptivo

*Calculado: `RF = k + (k−1)·s·(L−1)` con k=3, L=4.*

| `s` | campo receptivo | sobre una vista de 20 px |
|---:|---:|---|
| 1 | 9 px | ve menos de la mitad |
| 2 | 15 px | ve tres cuartos |
| 3 | 21 px | **excede la vista entera** |
| 4 | 27 px | excede con margen |

Éste es el confound principal y tiene su apartado.

### 3.4 Que el eje funciona hoy, sin tocar código

*Verificado el 2026-09-01:* `check_sweep` acepta `{"s_center": [1,2,3,4]}`;
`build_generated_spec` + `expand_points` producen **8 puntos válidos, 0 descartados**; y
cada punto **construye una red distinta de verdad** (flat 12.800 / 8.000 / 7.184 / 6.800).
Es la comprobación que costó `dropout` —que estaba en tres documentos y en ningún dict, así
que un barrido habría entrenado N veces la misma red— y aquí está hecha antes de gastar.

⚠⚠ **PERO `"auto"` NO SIRVE PARA ESTE EJE, y falla en silencio.** `check_sweep` lo acepta
(`s_center` está en `GEOMETRY_AUTO`), pero *medido el 2026-09-01*:

```
stride_range(center_band=16, n_layers=4) == [1]      # UN solo valor
stride_range(periph_band=4,  n_layers=4) == [1]
```

Un sweep con `"s_center": "auto"` sobre la base vigente entrenaría **N veces la misma red
sin avisar** — exactamente el fallo de `dropout`, por otra puerta. **El rango va explícito,
siempre.** Y hay una incoherencia real detrás, que este plan **no arregla** y que conviene
anotar donde se vea:

> `builder.py:128` dice *«the branch stride goes on the FIRST layer only … so the total
> subsampling is `s` regardless of depth → **n_layers stays out of stride_range**»*, pero
> `fovea/__init__.py:351` calcula `s_max = (region/4)^(1/n_layers)`, o sea que **sí** usa
> `n_layers` y como raíz. El comentario describe el código de hoy; la función describe el de
> antes de D-S1. Con L=4 el rango calculado colapsa a `[1]`.

**Arreglarlo es un cambio de código aparte** (`stride_range` debería ser
`range(1, max(1, region//4)+1)`, sin la raíz), no forma parte de este estudio, y **tiene que
llevar su test** (R14/R17). Se anota aquí para que no se pierda.

## 4. Los confounds — la parte que decide si el resultado significa algo

Subir el stride **no cambia una cosa: cambia tres a la vez**. Si el f1 baja, sin más brazos
no se puede decir cuál de las tres lo bajó. Escribirlo antes es lo que impide leer «menos
resolución hace daño» cuando lo que pasó fue «menos parámetros hace daño».

### 4.1 Resolución espacial de la salida (lo que se quiere medir)
La cabeza recibe menos posiciones donde mirar. Para un problema de **posición** —decir
*dónde* está una esquina— es plausible que duela, y es la pregunta del estudio.

### 4.2 Capacidad (confound)
La cabeza pierde parámetros en la misma proporción: 168.652 → 111.052 con un solo stride a
2. La red **puede** empeorar por falta de capacidad y no por falta de resolución.

### 4.3 El control que lo aísla, y por qué está apagado por defecto

Hay un control **iso-features** exacto y barato, gracias a que la salida es un cuadrado
completo: reducir `channels` a `s`=1 hasta que la cabeza reciba lo mismo.

| brazo | `s` | `channels` | features | **cabeza** | params totales | qué aísla |
|---|---:|---|---:|---:|---:|---|
| vigente | 1 | [16,16,16,16] | 12.800 | 153.612 | 168.652 | — |
| stride 2 (un lado) | 2 | [16,16,16,16] | 8.000 | **96.012** | 111.052 | menos resolución **y** menos params |
| **control** | 1 | **[16,16,16,10]** | 8.000 | **96.012** | 109.312 | **misma cabeza, resolución entera** |

Si el brazo de stride y el control empatan → **lo que pesaba era la capacidad**, y el stride
es gratis. Si el stride pierde contra el control → **la resolución importa**, que es un
resultado mucho más fuerte.

⚠ **Se llama iso-FEATURES y no iso-parámetros, y la diferencia se dice en vez de
redondearse.** Lo que queda **exacto** es la cabeza: 96.012 pesos en los dos brazos. Los
totales difieren en **1.740 (1,6 %)**, porque bajar canales también toca las convoluciones.
*Medido el 2026-09-01.*

⚠ **Y por eso se reduce SÓLO LA ÚLTIMA CAPA.** `[10]×4` iguala las features igual de bien,
pero mueve las tres convoluciones de en medio y la diferencia total sube a **8.580 (7,7 %)**
*(medido)*. Con `[16,16,16,10]` las conv 1..L−1 son **idénticas** a las del brazo con
stride, y sólo cambia la que produce el mapa que lee la cabeza — que es exactamente lo que
se quiere aislar. Un control que mueve más cosas que la que controla no controla.

⚠ **Sólo existe para `s`=2 y `s`=4.** Con `s`=3 el mapa es 7×7 y no hay un `channels` entero
que iguale las features; el script devuelve `None` y **se niega a crear el control** en vez
de aproximarlo, porque la diferencia que midiera sería en parte la del redondeo.

**Cuesta +2 runs por eje (+4 en total, ≈+0,15 $ estimado).** Está **apagado** por defecto
porque el tanteo es para acotar, no para explicar, y porque sólo hace falta *si el f1 baja*.
Es el supuesto **#3** de §0.

### 4.4 Campo receptivo (confound, y no se puede quitar)
Por §3.3, `s` y el campo receptivo se mueven juntos: son la misma operación. **No hay brazo
que los separe** a kernel fijo. Lo que sí se puede es **acotar el rango** para que el efecto
no cambie de naturaleza a mitad de tabla: a partir de `s`=3 el campo receptivo excede la
vista entera y la última capa deja de ser local. Es el argumento principal para el
supuesto **#1**.

### 4.5 Alineación (menor, pero se lee mal si no está escrito)
`s`=2 y `s`=4 dividen a N=20 (mapas de 10×10 y 5×5, rejilla alineada con la frontera
fóvea/anillo). `s`=3 **no** divide: 7×7 con resto, y la rejilla se desalinea. Si `s`=3 sale
peor de lo que su tendencia sugiere, ése es el primer sospechoso — y no «el stride 3 es
malo».

## 5. El diseño

### 5.1 Lo que se mantiene idéntico, y por qué

| Qué | Valor | Por qué |
|---|---|---|
| dataset | `dirty1000-80px-16px-r20260827` | el de `do-t`, `do-v` y `ei-t`: comparabilidad |
| base | `ws16-p2-d2-L4` | la vigente, la de toda la tabla — **no** la «mejor conocida» (#14 dejó `border_px` 8 y `overlap_fovea_px` 7 medidos y sin aplicar) |
| receta | `plan40` (lr 0,0014, batch 85, `patience` 10, monitor `val_loss`) | la vigente |
| `merge` | **`concat`** | es lo que permite strides distintos por rama (§2) |
| objetivo | `f1` | la métrica proxy de siempre |
| CPU | `E5-26` | dentro de la familia el entrenamiento sale bit a bit idéntico: el ruido de máquina es cero |

### 5.2 El rango: `{1, 2, 3, 4}` — supuesto #1

- **1** es el ancla: el vigente.
- **2** es el punto útil: −37 % de features por rama, rejilla alineada, campo receptivo
  todavía dentro de la vista. Es donde el dueño dice que quiere quedarse
  (*«no necesito que los features se reduzcan demasiado»*).
- **3** cuesta poco y contesta la pregunta de la alineación (§4.5).
- **4** **acota el eje por la derecha**. Está por la lección de `borde-ancho` y `patience`:
  un ganador pegado al borde del rango no es un óptimo, es el final de la regla. Con 4 el
  mapa es 5×5 y el campo receptivo 27 px: si aún así empata, el resultado es fuerte.

**No entra `s` ≥ 5**: el mapa sería 4×4 y el rango ya está acotado por 4.

### 5.3 `epochs` = 300 — supuesto #4

El tope vigente es 150. Se sube por la misma razón que lo subió `bs-alto`: **un run que
para por el tope mide presupuesto, no calidad** (R1 del protocolo), y ni el `dropout` ni el
`patience` de este eje están medidos. Aquí es **casi gratis**: por §3.2 los brazos con
stride corren entre 4× y 8× más rápido, así que el tope alto sólo lo puede consumir el brazo
`s`=1 — que es exactamente el que se compara con todo lo anterior.

### 5.4 Dos tanteos, `s_center` primero — supuesto #2

`generate_sweep` barre **un eje**, así que la petición («primero uno, luego el otro») encaja
sin forzar nada.

| fase | sweep | eje | ancla | runs | prefijo Vast |
|---|---|---|---|---:|---|
| 1 | **`sc-t`** | `s_center` | `s_periph` = 1 | 4 × 2 = **8** | `sc-` |
| 2 | **`sp-t`** | `s_periph` | **`s_center` = 1** | 4 × 2 = **8** | `sp-` |

⚠ **La fase 2 ancla en `s_center` = 1 (el vigente), NO en el ganador de la fase 1.** Es una
decisión, no un descuido:

- **A favor (lo elegido): efecto puro.** Cada eje se mide contra la misma base que todo lo
  demás en la tabla, y el resultado se puede citar sin arrastrar el error de la fase 1. Es
  también lo único que hace comparable *«¿cuál de las dos ramas aguanta mejor el
  submuestreo?»*, que es la pregunta interesante dado que las dos recortan lo mismo (§3.1).
- **En contra: no mide el efecto marginal.** Si los dos ganan por separado, **no está
  demostrado que sumen**: el brazo diagonal es otro punto (§7, supuesto #5).

**Coste estimado del conjunto: 16 runs, ≈0,5 $ y ~3 h de reloj** *(estimado, no medido: se
extrapola de `do-t` — 8 runs, 5 máquinas, 0,3626 $, 3 h 21 min — aplicando el ahorro de
§3.2 sólo en parte, porque el dataloader no baja).*

### 5.5 Las semillas: 2 en el tanteo, con lo que eso implica

**Un tanteo NO declara ganador.** Con 2 contra 2, el `p` mínimo alcanzable es **0,333**;
hacen falta 5v5 para bajar a 0,0079. El tanteo **acota**; la validación (`sc-v` / `sp-v`,
5 semillas, 20 runs) sólo se corre si §6 lo pide.

## 6. El criterio, escrito ANTES de mirar

`δ` = la banda de ruido que **este** estudio mida, por la regla de 1 SE que el proyecto ya
usa (`tie_delta` sobre las semillas de cada brazo). No se fija a mano para no elegir el
umbral después de ver la tabla.

⚠ **Y aquí el criterio NO es simétrico, a diferencia de `dropout` y `edge_inputs`.** Aquéllos
eran cost-neutrales, así que sólo ganaban mejorando. **Éste es cost-negativo** (§3.2): un
stride que **empata** ya paga, porque deja la misma calidad más barata y libera reloj para
todos los estudios que vengan detrás. El criterio lo dice explícitamente.

**R1 — Punto de corte (la pregunta principal).** Para cada eje, el **stride más grande cuya
media de f1 quede dentro de δ del brazo `s`=1**. Ése, y no el mejor, es la recomendación
práctica: la cabeza más pequeña que no pierde calidad.
- *Si el punto de corte es 4* → el eje **no queda cerrado por arriba** y la frase correcta es
  «gana el extremo, no sabemos dónde deja de valer».
- *Si es 1* → el submuestreo **cuesta calidad ya en el primer paso**, y eso también es un
  resultado: dice que la resolución de salida es un cuello de botella real.

**R2 — Se adopta un stride** (se lleva a validación de 5 semillas) si se cumplen **las dos**:
1. su f1 medio queda **dentro de δ** de `s`=1 (empatar basta: es cost-negativo), **y**
2. recorta la cabeza en **≥ 30 %** de sus parámetros — o el cambio no compensa la vuelta de
   reloj.

**R3 — Se cierra en el tanteo** si se cumple **cualquiera**:
1. **`s`=2 ya pierde por más de δ en los dos ejes** → el submuestreo no es viable en esta
   arquitectura, y la vía para encoger la cabeza es otra (§7); **o**
2. la amplitud entre `{1,2,3,4}` **no llega a 0,010** (el doble del ruido típico entre
   semillas; el mismo umbral del bloque B del #14, del tanteo de `patience` y de `ei-t`) →
   la meseta es plana, el eje es **puramente de coste** y se adopta el más barato que
   cumpla R2 **sin** pagar 5 semillas para confirmar un empate.

**R4 — El desempate entre ramas.** Si los dos ejes admiten stride, **gana el que más
recorte a igual pérdida**; si empatan también en eso, **gana `s_periph`**, y el motivo se
escribe aquí antes de mirar: el anillo ya está condensado por `border_reduce` (2 px reales
por celda) y es la parte de la vista que el diseño declara *de contexto*, no *de precisión*.
Submuestrear ahí contradice menos la hipótesis del proyecto que submuestrear la fóvea.

**R5 — Si el f1 baja, ANTES de concluir se corre el control iso-features** de §4.3
(+2 runs). Sin él, «la resolución importa» y «la capacidad importa» son indistinguibles, y
la primera es una afirmación sobre la visión foveada que este estudio no puede sostener sola.

## 7. Lo que este estudio NO contesta

- **No mide la métrica de tarea**, sólo el f1 por ventana. El proxy ya exageró una vez por un
  factor de dos (`n_layers`). Un ganador aquí es candidato, no adopción.
- **No mide el brazo diagonal** (`s_center` = `s_periph`), que es el que de verdad recorta
  (3.200 features con 2, 800 con 4). Dos ganadores por separado **no demuestran que sumen**.
  Es +4 runs y es el supuesto **#5**.
- **No toca `merge`.** `sum` exige strides iguales y **nunca se ha medido**; si el diagonal
  entra alguna vez, `merge` entra con él, porque a formas iguales `sum` divide la cabeza por
  dos otra vez. Es un plano, no una recta.
- **No toca `channels`** salvo como control (§4.3). `channels` es el **otro** mando que
  encoge la cabeza, medido con 1 semilla y borrado igual que éstos. Si el objetivo declarado
  es «menos features», los dos compiten por el mismo puesto y merecen compararse.
- **No cambia la arquitectura.** Que cada rama produzca un mapa completo de 20×20 para
  cubrir 256 celdas de máscara (§1.1) es una decisión de diseño, no un parámetro. Recortar
  cada rama a su región **no** es un ahorro tan grande como parece (98 % de las celdas están
  vivas), pero sigue siendo una vía distinta a ésta y no se estudia aquí.
- **No arregla `stride_range`** (§3.4). Sólo lo documenta y lo evita.

## 8. Prioridad y cierre — supuesto #6

⚠ **Va DETRÁS de `do-v`**, que es lo que estaba pendiente y ya tiene su plan escrito (ver el
`CLAUDE.md` del coordinador). Este documento **no lo adelanta**. Frente a `ei-t`, que está
en la misma cola sin prioridad asignada, hay un argumento para ponerlo antes —es
cost-negativo, así que abarata todo lo que venga detrás— y otro para ponerlo después —`ei-t`
ataca un límite del muestreo y esto ataca el coste—. **Lo decide el dueño.**

Al terminar: reporte en
`estudios-redes-neuronales/reportes/estudios/2026/<mes>/<fecha>-strides-rama-tanteo.md`, con
inicio y fin en **UTC**, **instancias alquiladas** (no las que trabajaron), **coste real** y
el apartado de **«lo que quedó pendiente»**. Y su fila **al final** de la tabla de
`reportes/README.md`, sin tocar las anteriores.
