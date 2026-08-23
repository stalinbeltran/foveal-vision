# Cerrar `lr` por la DERECHA sobre L4 — criterio escrito ANTES de mirar

> **Este documento se commitea antes de que exista un solo run de este recorrido.** Su valor
> entero está en eso: las reglas de abajo se pueden comprobar contra el commit. Si se cambian
> después de ver resultados, el plan deja de decidir nada (protocolo.md §1). Lo que se cambie
> igual va en un apartado aparte, con la hora, como en [plan-40h.md](plan-40h.md) §7 y
> [plan-lr-L4.md](plan-lr-L4.md) §6.

Fecha: 2026-08-23. Recorrido: **`lr-alto-L4`**. Dataset B: **`dirty1000-80px-16px-r20260823`**
(§2 dice por qué el nombre es nuevo, y es importante).

## 0. Qué pregunta responde

[plan-lr-L4.md](plan-lr-L4.md) barrió `lr` sobre la red L4 en `[0,00035 … 0,0014]` con 5 semillas
y cerró: **el eje es plano y el vigente (`lr = 0,0014`) se queda**. Contestó R3 —el óptimo **sí**
queda acotado por la izquierda, porque ganó 0,0006 y no el extremo— pero dejó abierto el lado
contrario:

⚠ **`0,0014` es el extremo DERECHO de todo lo que se ha medido sobre L4, y es el valor vigente.**
O sea que el valor que arrastra el proyecto está **pegado al borde de su rango**, que es
literalmente el defecto que aquel documento denuncia en §0 sobre el estudio anterior: *«un óptimo
en el borde no es un óptimo: es el final de la regla»*. Sobre L4 nadie ha medido por encima.

Lo que hay por encima es evidencia de **otra red**: `d1000-lr-1` barrió `[0,0014 … 0,0031]` con
`n_layers = 2` y la pérdida crecía monótonamente con `lr`. Un `lr` óptimo no se hereda al cambiar
la profundidad — es el motivo por el que plan-lr-L4 existió.

**La pregunta: sobre L4, ¿empeora `lr` al subir de 0,0014?** Si empeora, el vigente queda acotado
**por los dos lados** y la cuestión `lr` se cierra. Si no empeora, hay más que mirar arriba.

## 1. Constantes fijadas antes

| | valor | de dónde sale |
|---|---|---|
| base de red | **la misma que `p40-lr-L4`**: `ws16-p2-d2-L4` (`n_layers=4`, `channels=[16,16,16,16]`) | derivada igual, con los mismos `overrides`. Verificado: el `base_label` del `spec.json` de los dos recorridos coincide |
| receta | `plan40` (`batch_size` 85, `patience` 10, adam, sin scheduler) | el vigente; solo se mueve `lr` — es OAT |
| tope de épocas | **150** | alto a propósito: **`patience` tiene que ser quien pare**, no el tope (R1) |
| `seeds` | **3** | pedido por el usuario para esta primera vuelta. ⚠ Es **menos** que las 5 de los estudios anteriores: §5 dice qué se pierde |
| métrica de ranking | `val_f1` **del checkpoint** | metrica-de-tarea.md §9.7 |
| δ (banda de ruido) | **1-SE de las semillas del mejor punto**, calculada al cerrar | `suggest_winner`, la regla del proyecto |
| device | `cpu`, con **8 hilos de torch en las tres máquinas** | §4: máquinas distintas no deben entrenar distinto por tener más núcleos |

**El rango: `[0,0014 · 0,0020 · 0,0028]`** — log-espaciado (factor ~1,41), sube **2×** desde el
vigente, que aquí es el extremo **izquierdo**.

Por qué así y no de otra forma:

- **Empieza en el vigente**, que no es un punto gastado: es el **ancla**. Se vuelve a medir en las
  condiciones de hoy, y eso es lo que permite leer los dos valores nuevos contra algo conocido en
  vez de contra un número de otro estudio y otra máquina.
- **Sube 2× y no más.** Sobre L2 la pérdida ya crecía en `[0,0014 … 0,0031]`; si sobre L4 pasa lo
  mismo, 2× basta para verlo. Estirar más gastaría presupuesto en confirmar lo evidente.
- **Es la dirección barata.** La ley medida en plan-lr-L4.md §6 es `épocas ∝ lr^-0,287`: subir el
  `lr` **acorta** los runs. Los tres puntos de este recorrido son los más cortos que esta red
  puede dar, que es justo lo que pide una primera vuelta rápida.

**El orden de los puntos es `[0,0014 · 0,0020 · 0,0028]`**, y el orden es la mitigación (misma
idea que plan-40h.md §7.3): si el presupuesto se corta, lo que sobrevive es el ancla. El ranking
agrega por valor, no por orden: **esto no cambia ningún resultado**, solo qué se pierde si algo se
corta.

## 2. ⚠ El dato NO es bit a bit el de `p40-lr-L4`, y hay que decirlo

Al reconstruir el dataset en esta máquina (efímera, recién hecha) desde la fuente commiteada
`dirty-1000-80px` del repo del lanzador, la huella **no coincide** con la del manifest que
`dirty1000-80px-16px` tiene en git desde el 2026-07-27:

```
manifest de git : sha256:4327325b8e30267cef148e898867320fc7608b2fe88c2c36ad02481c387eb7e1
hoy             : sha256:13786b86492721806eede8e584812393468cba002bcc6d6b0b6d673f4772daef
```

Lo que **sí** coincide, campo a campo: `num_windows` (140.000), `windows_per_split`
(84.000/28.000/28.000), `positives_per_corner` (los cuatro exactos), `images.shape`,
`label_window` y `corner_order`. O sea que el contenido resumido es idéntico y lo que difiere está
por debajo de él.

Tres cosas comprobadas antes de sacar conclusiones, para que no se lea como una sospecha:

1. **La extracción es determinista**: dos extracciones seguidas aquí dan huella idéntica y los
   mismos positivos. No es ruido de serialización de esta máquina.
2. **`src/fv/windows/extract.py` no se toca desde el 2026-07-21**, anterior a los dos manifests.
   No es un cambio de código.
3. Con la config del **benchmark** (`bench-dirty1000-16`, manifest del 2026-08-14) la huella
   tampoco coincide, y ahí cambian **hasta los positivos por esquina**. O sea que la fuente ha
   cambiado por lo menos una vez entre julio y agosto.

**La consecuencia, dicha entera: no se puede afirmar que estos runs entrenen sobre el mismo dato
que los de `p40-lr-L4`.** Por eso el dataset lleva **nombre nuevo**
(`dirty1000-80px-16px-r20260823`) en vez de reutilizar el de julio — es la regla que el propio
descriptor del dataset tiene escrita: *«Si algún día cambia, cambia también el nombre en vez de
reutilizarlo»*. Reutilizarlo habría hecho que dos datasets distintos compartieran nombre, que es
la forma de que la no-comparabilidad no se note nunca.

**Qué se salva y qué no:**

- **Se salva el estudio entero.** Los 9 runs entrenan sobre **el mismo `windows.npz`**, extraído
  una vez aquí y enviado hecho a las tres máquinas (§4). Todas las comparaciones *dentro* de este
  recorrido —que son las que contestan la pregunta— son válidas.
- **No se salva la comparación directa con los números de `p40-lr-L4`.** El `lr = 0,0014` de aquel
  estudio dio `val_f1` 0,9244 ± 0,0041 (5 semillas). El de éste es una medida **nueva**, y
  compararlas mide *también* la diferencia del dato. Se reportará como lo que es, no como
  continuidad.

**Y eso convierte el ancla en algo más útil de lo previsto**: la distancia entre el 0,0014 de hoy
y el 0,9244 de julio es una medida de cuánto movió el dato. Se publica ese número, con esta
advertencia al lado.

## 3. Cómo se lee el resultado (escrito antes)

**R1 — validez antes que ranking.** Un punto cuyos runs paran **por el tope de 150** y no por
`patience` mide presupuesto, no calidad. Si el ganador es uno de ésos, **no se declara ganador**:
se reporta `budget-limited`. Aquí el riesgo es bajo —subir el `lr` acorta— pero la regla se aplica
igual, porque el proyecto ya publicó dos estudios con ese defecto sin notarlo.

**R2 — el ganador.** Media de las semillas del `val_f1` del checkpoint, δ = 1-SE de las semillas
del mejor punto, exactamente `suggest_winner`. Sin regla nueva.

**R3 — la pregunta de verdad: ¿queda acotado por la derecha?** El óptimo está acotado por la
derecha si y solo si el ganador **no es 0,0028**. Si gana el extremo derecho, la respuesta es
*sigue sin acotar por arriba*, y **eso se publica como tal** — no es excusa para lanzar otro
recorrido en silencio.

**R4 — ¿alguno le gana al vigente?** El ganador contra `lr = 0,0014` **medido en este mismo
recorrido** (no contra el de julio, §2). Con 3 semillas la permutación exacta da 20 arreglos, así
que **el p mínimo alcanzable es 0,10**: con este tamaño **R4 no puede declarar significación al
5 %**, y por tanto **el vigente se queda pase lo que pase**. Se dice ahora, antes de ver nada,
para que no se lea el resultado como si pudiera mover el `lr` del proyecto.

**R5 — nada se arrastra sin pasar por la métrica de tarea.** Si algún día este resultado fuera a
arrastrarse, antes se mide con `scripts/proxy_vs_task.py` (metrica-de-tarea.md §2 ter). Con 3
semillas no se arrastra nada, así que aquí sólo queda apuntado.

## 4. Cómo se corre: una máquina por SEMILLA

`scripts/estudio_flota.py` alquila **una máquina por semilla** y cada una corre los 3 valores de
`lr` de su semilla. El reloj pasa a ser el de la semilla más lenta en vez de la suma (el estudio
de julio fueron 36,9 h de reloj para 20 runs en secuencia).

**Se reparte por semilla y no por valor del eje, y es la decisión que más importa.** La semilla es
el eje réplica, no la pregunta: repartiendo así, cada máquina mide **todos** los valores de `lr`,
de modo que si una máquina es más lenta o más rara, esa rareza entra por igual en todos los
valores que se comparan — es un bloque y se cancela. Repartir por valor de `lr` haría lo
contrario: la máquina quedaría **confundida con la respuesta**, y un `lr` podría ganar por haberle
tocado el host bueno. Eso no se arregla después con estadística.

Tres detalles que sostienen la comparabilidad, y por qué:

- **El dataset se extrae UNA vez aquí y viaja hecho** (2,5 MB). Las tres máquinas leen el mismo
  fichero byte a byte. Extraerlo en cada una sería pedir que tres extracciones coincidan: promesa
  más fuerte, ganancia ninguna.
- **8 hilos de torch en las tres**, fijados por `OMP_NUM_THREADS`. Sin fijarlos, una máquina con
  21 núcleos y otra con 9 no sólo irían a distinta velocidad: repartirían las reducciones en coma
  flotante de forma distinta. ⚠ Lo que **no** se puede igualar es el juego de instrucciones del
  procesador (AVX2 / AVX-512): dos CPU distintas pueden redondear distinto, y eso queda como
  ruido entre semillas — que es exactamente donde el diseño por bloques lo pone.
- **Los nombres de los runs los pone el índice global del punto** dentro del recorrido, así que
  los runs de las tres máquinas se juntan en un solo `runs/` y forman el recorrido entero sin
  renombrar nada.

### Máquinas siempre distintas, y el que falla queda apuntado

En Vast.ai varias ofertas pueden ser del **mismo host** (una por GPU libre), así que «las 3 más
baratas» son a menudo 3 réplicas de la misma máquina: comparten CPU, disco y suerte, y si se cae
se lleva las tres semillas. `elegir_ofertas_distintas` coge **una oferta por `machine_id`** aunque
la siguiente cueste más.

Y un host que falla vuelve a salir mañana en el catálogo, más barato que el resto, así que la
elección por precio vuelve a caer en él — ya pasó: *«el barrido del 2026-08-21 cayó dos veces en
la misma oferta rota»*. Por eso el fallo se apunta en **`vast-bloqueadas.json` del repo del
lanzador, que se commitea**: la máquina de control es efímera y lo que no está en el remoto no
existe.

**Qué se apunta y qué no** (la parte que evita que la lista negra se coma el mercado):

- **Sí**: no arranca, sshd no contesta, falla la subida, falla la instalación, el proceso muere
  sin dejar código de salida, se agota el plazo. Eso es la máquina.
- **No**: el entrenamiento arranca y termina con puntos fallidos. Eso es código o dato, se
  repetiría en cualquier host, y bloquear por ello vaciaría el catálogo sin arreglar nada.
- **Caduca a los 30 días** desde el último fallo, y la regla va escrita junto a la constante: los
  hosts se arreglan y se actualizan, y un bloqueo eterno sólo crece hasta dejar la búsqueda sin
  ofertas sin decir por qué.

## 5. Coste, presupuesto y lo que cuestan 3 semillas

Con `épocas ∝ lr^-0,287` anclada en las 47 épocas medidas a `lr` = 0,0014:

| `lr` | épocas est. | por máquina |
|---|---|---|
| 0,0014 | 47 (**medido** en julio) | |
| 0,0020 | ~42 | |
| 0,0028 | ~39 | |
| **total** | **~128 épocas** | y cada máquina corre las 128 |

**Medido el 2026-08-23** en el droplet de control (2 vCPU, `DO-Regular`) con este mismo
recorrido: **103,9 s/época** de media sobre 5 épocas (105,6 · 99,7 · 95,4 · 104,7 · 114,2). Es el
mismo orden que las 103-106 s/época que plan-lr-L4.md midió para esta red, lo que da confianza en
que el dato de §2 se comporta igual aunque su huella no cuadre.

A esa velocidad serían ~3,7 h por máquina. Las máquinas alquiladas tienen 8 hilos en vez de 2, así
que se espera bastante menos — **pero no está medido, y por eso no se escribe un número aquí**: lo
dirá la primera época de cada máquina. Coste de alquiler: **≈ 0,16 $/h entre las tres**, techo de
0,94 $ si las tres vivieran 6 h (el plazo por semilla, tras el cual se destruyen).

⚠ **Lo que cuestan 3 semillas en vez de 5**, dicho antes de mirar: la banda de cada punto se
ensancha (~29 % más de error estándar) y, sobre todo, **R4 pierde toda capacidad de declarar
significación** (§3). Esta vuelta sirve para **ver la forma del eje** y para estrenar el reparto en
paralelo, no para mover el vigente. Si la forma resulta interesante, la confirmación se corre con
5 semillas — que con este script cuesta 5 máquinas y el mismo reloj.

## 6. RESULTADO (2026-08-23 20:37) — **el eje cae a la derecha; `lr` queda acotado por los dos lados**

9/9 puntos, **139 min de reloj** (2 h 19 min) y **0,2952 $**. Los 9 runs pararon por `patience`
(R1 ✅), entre 30 y 72 épocas, ninguno cerca del tope de 150.

| `lr` | ventana (f1) | sem | min | max | épocas (3 semillas) |
|---|---|---|---|---|---|
| **0,0014** (vigente) | **0,9246** | 0,0003 | 0,9240 | 0,9251 | 36 · 54 · 66 |
| 0,0020 | 0,9055 | 0,0092 | 0,8883 | 0,9200 | 30 · 65 · 70 |
| 0,0028 | 0,8998 | 0,0080 | 0,8878 | 0,9150 | 54 · 56 · 72 |

**R3 ✅ — LA PREGUNTA QUEDA CONTESTADA: el óptimo está acotado por la derecha.** El ganador es
`0,0014` y **no** el extremo derecho, que era la condición escrita en §3. Junto con
[plan-lr-L4.md](plan-lr-L4.md) §7 —que lo acotó por la izquierda— **`lr` sobre L4 queda ahora
cerrado por los dos lados**, y el vigente deja de estar pegado a un borde: es el defecto que §0
denunciaba, y se cierra.

**Y la caída no es marginal: las bandas son DISJUNTAS.** Las tres semillas de `0,0014` (0,9240 –
0,9251) están por encima de las tres de `0,0020` (0,8883 – 0,9200) *y* de las tres de `0,0028`
(0,8878 – 0,9150). No hay solape: ninguna réplica del vigente pierde contra ninguna réplica de un
`lr` más alto.

**R4 — el contraste, y lo que NO puede decir.** Ambos valores dan `p = 0,100` contra el vigente
con permutación exacta. Ese 0,100 **es el suelo**: con 3 contra 3 hay 20 arreglos y el p mínimo
alcanzable es 2/20. Que se toque el suelo significa que **ninguna reetiquetación de las semillas
produce una diferencia mayor que la observada** — es la evidencia más fuerte que este tamaño puede
dar. Pero sigue sin cruzar el 5 %, así que, como se escribió antes de mirar, **el vigente se queda**
por regla, no por el resultado. Aquí eso da igual: el resultado y la regla dicen lo mismo.

### 6.1 El ancla desactiva el aviso de §2

`lr = 0,0014` re-medido hoy da **0,9246** contra **0,9244** de julio: **+0,0002**, con un `sem` de
0,0041 en aquella medida. O sea que el cambio de fuente documentado en §2 —huella distinta, mismo
resumen— **no movió la medida de forma apreciable**.

⚠ Con cuidado, que es una comprobación y no una demostración: dice que en **este punto** las dos
versiones del dato dan lo mismo dentro del ruido. No prueba que los `.npz` sean idénticos, y el
dataset conserva su nombre nuevo — la evidencia de que difieren sigue en pie. Pero quita el motivo
para desconfiar de comparar estos números con los de julio.

### 6.2 Dos cosas que no se esperaban, y que valen para planificar

**a) Por encima del óptimo, `lr` más alto NO converge antes.** La ley que plan-lr-L4.md §6 midió,
`épocas ∝ lr^-0,287`, se ajustó con dos puntos **por debajo** de 0,0014, y ahí funcionó. Extendida
hacia arriba **falla, y de signo**: predecía ~42 y ~39 épocas para 0,0020 y 0,0028, y salieron
medias de **55 y 61** — más que las 52 del vigente. Consecuencia práctica: **esa extrapolación sólo
vale a la izquierda del óptimo**; usarla para presupuestar la zona alta subestima el coste.

**b) Subir el `lr` no sólo empeora la media: dispara la varianza.** El `sem` pasa de **0,0003** en
el vigente a **0,0092** y **0,0080** — treinta veces más. La peor semilla de `0,0020` (0,8883) está
0,032 por debajo de la mejor (0,9200), mientras que las tres del vigente caben en 0,0011. Un `lr`
alto no da un modelo un poco peor: da un modelo **impredecible**, y eso con una sola semilla no se
ve. Es otra ilustración de *un resultado sin N semillas es una anécdota*.

### 6.3 Qué costó, y qué habría costado en serie

| | este recorrido | `p40-lr-L4` (julio) |
|---|---|---|
| runs | 9 (3 valores × 3 semillas) | 20 (4 × 5) |
| reloj | **2 h 19 min** | 36 h 54 min |
| máquinas | 3 alquiladas, una por semilla | 1 (el equipo del usuario) |
| coste | 0,2952 $ | electricidad + 36 h de máquina ocupada |

Las tres máquinas midieron a **36,3 · 50,5 · 53,3 s/época**. En secuencia en la más lenta, los 9
runs habrían sido ~7 h 40 min de reloj: el reparto los deja en 2 h 19 min, que es lo que tarda la
semilla más lenta.

⚠ **El número de vCPU no predijo la velocidad.** La máquina de **16 vCPU** (semilla 1) fue la **más
lenta** (53,3 s/época) y la de **9,3 vCPU** (semilla 2) la más rápida (36,3). Como los hilos de
torch estaban fijados a 8 en las tres (§4), lo que quedó a la vista es la velocidad **por núcleo**,
y ahí el catálogo varía casi 1,5×. La elección de oferta filtra hoy por número de núcleos y precio,
que resulta ser un mal criterio para este trabajo: mirar `cpu_ghz` es la mejora obvia, y está sin
hacer.

### 6.4 La lista negra se estrenó sola

A los 75 segundos del lanzamiento, la máquina **45390** aceptó el alquiler, levantó `sshd` y
**rechazó la clave** (`Permission denied (publickey)`) a través del proxy `ssh5.vast.ai`. Quedó
apuntada en `vast-bloqueadas.json`, se destruyó (1,2 min, 0,0011 $) y la semilla 3 se reintentó
sola en **otra** máquina (29155), que terminó su trabajo sin incidencias.

⚠ **No está comprobado que la culpa fuera de esa máquina.** El fallo apareció en una instancia
enrutada por proxy, y podría ser una carrera entre el banner de `sshd` y la instalación de la clave
—que le pasaría a cualquier host— en vez de un defecto del host. Las otras dos máquinas, con
conexión directa, no lo sufrieron; con un solo caso no se puede distinguir. Queda apuntado aquí
porque si vuelve a pasar **en máquinas distintas y siempre por proxy**, el bloqueo estaría culpando
al host equivocado y lo que habría que arreglar es la espera. El bloqueo caduca a los 30 días, que
es lo que acota el daño de esa duda.

### 6.5 Qué NO contesta esto

- **Con 3 semillas no se mueve nada del proyecto** (§3, R4). Si se quisiera *usar* este resultado
  para algo más que cerrar la pregunta, la confirmación va con 5 semillas — que con este script son
  5 máquinas y el **mismo** reloj.
- **Es f1 de ventana, un proxy.** R5 (medir con `scripts/proxy_vs_task.py`) no se ha corrido: aquí
  no se arrastra ningún ganador, así que no hacía falta. Si alguna vez se arrastra, primero eso.
- **La zona entre 0,0014 y 0,0020 no está medida.** El eje cae, pero dónde empieza a caer
  exactamente no se sabe. Nadie lo ha preguntado todavía.

## 7. SEGUNDA CORRIDA (2026-08-23 23:36) — el mismo estudio con **una máquina por run**

`lr-alto-L4-b` es el **mismo recorrido, campo por campo** (comprobado al crearlo: los 15 campos
del `spec` coinciden salvo el nombre), corrido con `--reparto run`: 9 máquinas, una por punto, en
vez de 3, una por semilla. Se hizo para ganar reloj y para poder comparar el coste de los dos
repartos.

### 7.1 Lo que costó cada reparto

| | `seed` (3 máq.) | `run` (9 máq.) | |
|---|---:|---:|---:|
| **reloj** | 139,1 min | **55,4 min** | **−60 %** |
| **coste** | 0,2952 $ | 0,3656 $ | +24 % |
| máquina-minutos facturados | 392,7 | 368,5 | −6 % |
| · de eso, trabajo (épocas) | 381,8 | 335,5 | −12 % |
| · de eso, **peaje** | 10,9 | 31,3 | +187 % |
| precio medio de máquina | 0,0449 $/h | 0,0596 $/h | **+33 %** |

**El reparto fino sale barato: 2,5× menos reloj por 7 céntimos más.**

⚠ Pero el +24 % **no viene de donde se esperaba**, y esto es lo que había que medir. El peaje se
triplicó, sí (3,5 min por máquina × 9 en vez de × 3) — pero en absoluto son 20 minutos de máquina,
un 5 % del total. Los **máquina-minutos totales incluso bajaron un 6 %**. Lo que subió el coste fue
el **precio medio de la máquina (+33 %)**: pedir 12 máquinas *distintas* en vez de 6 obliga a bajar
más en la lista ordenada por precio, y ahí ya no están las gangas.

O sea que **el coste del paralelismo fino no es el peaje, es agotar las ofertas baratas.** Escala
distinto: el peaje crece lineal con el número de máquinas y es pequeño; el precio medio crece
según lo profundo que haya que rascar en el catálogo, y eso depende del mercado ese día. Para
estudios más grandes es el segundo el que hay que vigilar.

### 7.2 ¿Dio la misma respuesta? Sí en lo que decide; no en lo accesorio

| `lr` | corrida A (`seed`) | corrida B (`run`) | dif |
|---|---:|---:|---:|
| **0,0014** | 0,9246 | 0,9226 | −0,0020 |
| 0,0020 | 0,9055 | 0,8900 | −0,0154 |
| 0,0028 | 0,8998 | 0,9054 | +0,0056 |

**El ganador es el mismo (`0,0014`) y el veredicto de §6 aguanta entero**: R3 se cumple igual (no
gana el extremo derecho) y los dos contrastes vuelven a dar `p = 0,100`, el suelo. La conclusión
del estudio **no depende del reparto**.

⚠ **Pero el orden de los perdedores se dio la vuelta**: A dijo `0,0020 > 0,0028` y B dice
`0,0028 > 0,0020`. Ese orden **no es un resultado**, y §6 ya avisaba de por qué: en la zona alta el
`sem` es diez veces mayor que en el vigente. La réplica lo confirma en vez de descubrirlo.

### 7.3 ⚠ Lo que la réplica NO compra: semillas

Es tentador juntar las dos corridas y decir «ya tengo 6 semillas por valor», que daría 924
arreglos y **sí** podría declarar significación al 5 %. **Sería falso.** Las dos corridas usan las
**mismas semillas 1, 2 y 3**: el mismo flujo de números aleatorios sobre el mismo dato. No son seis
réplicas independientes, son **tres medidas hechas dos veces**. Juntarlas para ganar potencia
estadística sería contar cada semilla dos veces.

Lo que la réplica sí compra es otra cosa, y resultó valer más.

### 7.4 El hallazgo: **el resultado es idéntico bit a bit dentro de una familia de CPU, y diverge entre familias**

Como las dos corridas usan las mismas semillas, cada run tiene su **gemelo exacto** en la otra:
mismo código, mismo dato, misma semilla, **otra máquina**. Esa diferencia *es* el efecto de la
máquina, aislado.

| `lr` | semilla | A (f1) · CPU | B (f1) · CPU | dif |
|---|---|---|---|---:|
| 0,0014 | 2 | 0,9247 · Xeon E5-2680 v4 | 0,9247 · Xeon E5-2673 v3 | **0,0000** |
| 0,0020 | 2 | 0,9081 · Xeon E5-2680 v4 | 0,9081 · Xeon E5-2680 v4 | **0,0000** |
| 0,0028 | 2 | 0,9150 · Xeon E5-2680 v4 | 0,9150 · Xeon E5-2680 v4 | **0,0000** |
| 0,0014 | 1 | 0,9251 · Xeon Silver 4108 | 0,9219 · Xeon E5-2680 v4 | −0,0033 |
| 0,0014 | 3 | 0,9240 · EPYC 7551 | 0,9213 · Xeon E5-2680 v4 | −0,0027 |
| 0,0020 | 3 | 0,9200 · EPYC 7551 | 0,9194 · Xeon E5-2690 v4 | −0,0006 |
| 0,0028 | 1 | 0,8966 · Xeon Silver 4108 | 0,9042 · Xeon E5-2680 v4 | +0,0076 |
| 0,0028 | 3 | 0,8878 · EPYC 7551 | 0,8969 · Xeon E5-2680 v4 | +0,0091 |
| **0,0020** | **1** | **0,8883 · Xeon Silver 4108** | **0,8426 · Core i7-6700** | **−0,0457** |

**La separación es perfecta: 3 de 3 pares dentro de la familia Xeon E5-26xx v3/v4 salen idénticos
—al cuarto decimal y con el MISMO número de épocas— y 6 de 6 pares entre familias distintas
salen diferentes.** Los tres idénticos son todos de la semilla 2, que en la corrida A cayó en un
E5-2680 v4 y en la B en tres máquinas E5 distintas (v3 y v4, Haswell y Broadwell): microarquitectura
distinta, **mismo juego de instrucciones vectoriales** (AVX2/FMA3), y el entrenamiento es
reproducible bit a bit. Los que divergen son los que cruzaron a **Xeon Silver 4108** (Skylake-SP,
con AVX-512), a **AMD EPYC** (Zen) o a **Core i7-6700**.

Es la confirmación empírica del aviso que §4 dejó escrito sin poder medirlo: *«lo que no se puede
igualar es el juego de instrucciones del procesador; dos CPU distintas pueden redondear distinto»*.
Se puede, y aquí está el número.

**Y la magnitud depende de la zona del eje**, que es lo que lo hace accionable:

- En el **vigente** (`0,0014`, zona estable): la máquina mueve **≤ 0,0033**, frente a un efecto
  medido de ~0,020–0,035. El ganador está a salvo con cualquier reparto.
- En la **zona alta** (`lr` ≥ 0,0020, entrenamiento inestable): la máquina llega a mover **0,0457**
  — **más que el efecto que el estudio mide**. Con `lr` alto, dos máquinas distintas dan respuestas
  distintas, y eso explica del todo el vuelco de §7.2: no fue mala suerte de las semillas, fue el
  cambio de CPU.

### 7.5 Qué hacer con esto

1. **`--reparto run` es seguro para decidir el ganador** cuando el efecto buscado es mayor que
   ~0,005, y da 2,5× de reloj por un 24 % de coste. Para este estudio, la elección correcta.
2. **Fijar la familia de CPU convierte el ruido de máquina en cero.** No es una mejora marginal:
   dentro de Xeon E5 v3/v4 el resultado sale idéntico bit a bit. `--cpu` en
   `estudio_flota.py` (y `elegir_ofertas_distintas`) hace justo eso, y es lo que hay que usar
   cuando el efecto buscado sea pequeño o la zona sea inestable.
3. **Nunca reportar el orden de puntos que caen dentro de ~0,01 sin fijar la CPU.** El vuelco de
   §7.2 es exactamente ese error, y se habría publicado como resultado si no llega a haber réplica.

### 7.6 ⚠ Hasta dónde llega lo medido (y dónde deja de estar comprobado)

Lo comprobado son **tres pares** de runs gemelos, todos dentro de **Xeon E5-2673 v3 (Haswell) y
E5-2680 v4 (Broadwell)**, que comparten AVX2 + FMA3. Eso es todo.

**No está comprobado** que la igualdad se extienda al resto de la familia. En particular, el filtro
`--cpu "E5-26"` es una **subcadena**, y deja pasar también **E5-26xx v2 (Ivy Bridge), que no tiene
AVX2**: por el propio razonamiento de §7.4 —lo que manda es el juego de instrucciones vectoriales—
una v2 debería divergir de una v3/v4, pero **no se ha medido**. Al pedir 12 máquinas con ese filtro
el catálogo devolvió una `E5-2660 v2` entre ellas.

Si hace falta la garantía de verdad, el filtro tiene que ser más estrecho (`E5-2680 v4`, por
ejemplo) y el precio a pagar es un catálogo más pequeño. Con `E5-26` a secas, el precio medio subió
sólo un **2 %** (0,0605 → 0,0618 $/h, medido el 2026-08-23), así que estrechar más sigue siendo
barato.

Y hay una comprobación que **este estudio no hizo y que cerraría el asunto**: correr el mismo run
dos veces **en la misma máquina**. Si saliera idéntico, quedaría probado que la única fuente de
divergencia es la CPU; si no, habría algo más (hilos, versión de librería, orden de reducción) y
fijar la familia no bastaría. No se hizo, y por eso §7.4 dice «dentro de esta familia salió
idéntico» y no «es determinista».
