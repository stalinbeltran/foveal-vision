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
