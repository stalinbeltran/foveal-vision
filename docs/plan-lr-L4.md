# Re-barrer `lr` sobre `n_layers=4` — criterio escrito ANTES de mirar

> **Este documento se comitea antes de que exista un solo run del plan.** Su valor entero está en
> eso: las reglas de abajo se pueden comprobar contra el commit. Si se cambian después de ver
> resultados, el plan deja de decidir nada (protocolo.md §1). Lo que se cambie igual va en un
> apartado aparte, con la hora, como se hizo en [plan-40h.md](plan-40h.md) §7.

Fecha: 2026-08-08. Dataset B fijo: `dirty1000-80px-16px`. Recorrido: **`p40-lr-L4`**.

## 0. Qué pregunta responde, y por qué hay que rehacerla

`lr = 0,0014` es el valor que arrastra todo el proyecto desde el estudio `d1000-lr-1`. Tiene **dos
defectos conocidos**, los dos medidos:

1. **Se fijó sobre `n_layers=2`**, y desde el 2026-08-08 la red es **L4** ([plan-40h.md](plan-40h.md)).
   Un `lr` óptimo no se hereda al cambiar la profundidad.
2. ⚠ **Quedó pegado al borde izquierdo de su rango, sin acotar.** El espacio barrido fue
   `[0,0014 … 0,0031]`, la pérdida **crece monótonamente** con `lr` en todo él, y **el ganador es
   el valor mínimo**. Un óptimo en el borde no es un óptimo: es el final de la regla.

Y un tercer defecto que hace la medida aún menos fiable: aquellos 70 runs corrieron con
`patience = 0` y **tope de 20 épocas, que los 70 agotaron**. Se midió velocidad de convergencia, no
calidad — y esa confusión **penaliza justo al `lr` bajo**, que es el candidato. El `lr` bajo ganó
**a pesar** del handicap.

## 1. Constantes fijadas antes

| | valor | de dónde sale |
|---|---|---|
| base de red | **el ganador L4 exacto** (`n_layers=4`, `channels=[16,16,16,16]`, resto igual) | copiado bit a bit de `runs/p40-confirm-n_layers-0000-…/config.json`; verificado antes de crear el recorrido |
| receta | `plan40` (`batch_size` 85, `patience` 10, adam, sin scheduler) | los ganadores vigentes; solo se mueve `lr` — es OAT |
| **tope de épocas** | **150** | alto a propósito: **`patience` tiene que ser quien pare**, no el tope. Es el bug de plan-40h.md §0, y aquí muerde más porque un `lr` bajo converge más lento |
| `seeds` | **5** | la misma N de los estudios `d1000-*` y del plan de 40 h, para que las bandas se puedan comparar |
| métrica de ranking | `val_f1` **del checkpoint** | §9.7 de metrica-de-tarea.md la midió como el mejor proxy de la tarea (+0,956 contra +0,780 de `loss`) |
| δ (banda de ruido) | **1-SE de las 5 semillas del mejor punto**, calculada al cerrar | la regla del proyecto (`suggest_winner`), no un número escrito a mano |

**El rango: `[0,00035 · 0,0006 · 0,0009 · 0,0014]`** — log-espaciado (factor ~1,5), baja **4×**
desde el valor vigente, que es el extremo derecho.

⚠ **Esto se desvía de un rango pre-escrito, y hay que decirlo.** plan-40h.md §3.2 dejó escrito
`[0,0004 · 0,0006 · 0,0008 · 0,0011 · 0,0014]` **antes** de que existiera nada de esto. Se cambia
por dos razones, las dos anteriores a ver un solo número de este recorrido: (a) aquel rango se
diseñó para **L2 con 20 épocas fijas**, donde el `lr` bajo estaba penalizado, y aquí `patience`
quita esa penalización; (b) es casi lineal — gasta 3 de sus 5 puntos en la zona alta y baja solo
3,5× — cuando el objetivo entero es **acotar el óptimo por la izquierda**. Decisión del usuario,
tomada sobre las dos opciones puestas una al lado de la otra.

**El orden de los puntos es `[0,00035 · 0,0014 · 0,0006 · 0,0009]`**, y el orden es la mitigación
(misma idea que plan-40h.md §7.3): los puntos se entrenan en el orden de la lista, así que si el
presupuesto se corta lo que falta son **los interiores**. Primero el extremo caro e incierto
(0,00035) y después el vigente (0,0014) — los dos que contestan *¿el óptimo está más a la
izquierda?*. **El ranking agrega por valor, no por orden: esto no cambia ningún resultado**, solo
qué se pierde si algo se corta.

## 2. Etapa A — la sonda (1 run, ≤ 4,4 h)

**Por qué existe:** el coste de este recorrido depende de **cuántas épocas tarda `patience` en
saltar con un `lr` bajo**, y ese número **nadie lo ha medido**. Lo único que hay es *un* punto
(L4 a `lr` = 0,0014 para en 32–61 épocas, media 47) y una extrapolación `épocas ∝ 1/lr` hecha
desde él. Los 65 runs de `fast-lr-s0-lr`, que barren 13 valores de `lr`, **no sirven** para
calibrarla: tenían `patience = 0` y **los 65 toparon en la época 20**.

La sonda es **el primer run del recorrido de verdad** (`lr` = 0,00035, semilla 1, tope 150), no un
experimento aparte: si se continúa, no se tira nada.

**Qué mide:** (a) las épocas hasta que salta `patience`, y con ellas el coste real de cada punto;
(b) de paso, si ese `lr` le hace algo al f1 de ventana frente al vigente.

**La regla de decisión posterior, escrita ahora:**

1. **Si la sonda para por `patience` en E épocas** → se re-estima el recorrido entero con E medido
   (escalando `1/lr` desde el punto real más cercano, no desde 0,0014). Si el total supera **40 h**,
   se aplican las guardas **en este orden**: ① `seeds` 5 → 3; ② quitar un punto **interior**
   (0,0006 o 0,0009) — **nunca un extremo**, porque los extremos son la pregunta.
2. **Si la sonda topa en 150 épocas** → ese punto **no mide calidad, mide presupuesto**, y no se
   arregla con más tope en esta máquina. Se reporta así, se **sube el suelo del rango** y se dice
   que la zona por debajo de ese `lr` **queda sin medir** — no se declara ganador sobre un punto
   truncado.
3. ⚠ **La comparación de la sonda contra el vigente es informativa y NO decide nada.** Es una
   semilla, y este proyecto acaba de medir lo que eso vale: el cribado del plan de 40 h, con una
   semilla, daba **+0,0009** en la métrica de tarea entre L2 y L4 (metrica-de-tarea.md §2 ter),
   frente a un `sem` por run de ±0,023. *Un resultado sin N semillas es una anécdota.*

## 3. Etapa B — el recorrido, y cómo se lee (escrito antes)

**R1 — validez antes que ranking.** Un punto cuyos runs paran **por el tope** y no por `patience`
mide presupuesto, no calidad. Si el ganador es uno de esos, **no se declara ganador**: se reporta
`budget-limited` y se dice qué haría falta. Esta regla existe porque el proyecto ya publicó dos
estudios con ese defecto sin notarlo.

**R2 — el ganador.** Media de las 5 semillas del `val_f1` del checkpoint, δ = 1-SE de las semillas
del mejor punto, exactamente `suggest_winner`. Sin regla nueva.

**R3 — la pregunta de verdad: ¿queda acotado?** El óptimo está **acotado por la izquierda** si y
solo si el ganador **no es 0,00035**. Si vuelve a ganar el extremo izquierdo, la respuesta es
*sigue sin acotar*, y eso es **un resultado que se publica como tal**, no una excusa para lanzar
otro recorrido en silencio.

**R4 — ¿le gana al vigente?** El ganador contra `lr = 0,0014` con **permutación exacta** de las
semillas (`fv.metrics.permutation_test`, dos colas, 252 arreglos). Se declara mejora solo si la
diferencia **supera δ** *y* **p ≤ 0,05**. Con cualquier otro resultado el vigente **se queda**:
cambiar el `lr` que arrastra todo el proyecto por una diferencia dentro del ruido es exactamente
lo que protocolo.md prohíbe.

**R5 — nada se arrastra sin pasar por la métrica de tarea.** Antes de que el ganador se lleve a
ningún sitio, se mide con `scripts/proxy_vs_task.py`. Es la lección de metrica-de-tarea.md §2 ter,
medida hace unas horas: sobre `n_layers`, la ventana **exageró** la ganancia al doble y convirtió
bandas solapadas en disjuntas.

## 4. Reanudable por diseño

El equipo se apaga por falta de energía (confirmado por el usuario) y **hiberna**. Por eso:

- es un **recorrido**, y `run_sweep` rehace todo punto que no esté `done`/`cancelled`;
- **relanzarlo continúa donde se quedó**, y hay watchdog (`scripts/plan_lr_L4_watchdog.ps1`).

## 5. Coste — lo que se cree hoy, y lo que lo puede tumbar

Con la extrapolación `épocas ∝ 1/lr` desde el único punto medido, y 106 s/época en L4:

| `lr` | épocas est. | h/run | × 5 semillas |
|---|---|---|---|
| 0,00035 | 150 (tope) | 4,4 | 22,1 |
| 0,0006 | ~110 | 3,2 | 16,1 |
| 0,0009 | ~73 | 2,2 | 10,8 |
| 0,0014 | 47 (**medido**) | 1,4 | 6,9 |

**≈ 56 h** si todo topa; **≈ 31 h** si `patience` corta pronto. La sonda existe para saber cuál de
las dos. ⚠ Y dos avisos ya medidos en esta máquina: el **throttling térmico** ralentiza ~5× en
carga sostenida, y el **micro-benchmark de coste miente bajo carga** (plan-40h.md, punto 7: 34 h
estimadas contra 22 h reales con la máquina libre).
