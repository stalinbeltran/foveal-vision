# Re-barrer `lr` sobre `n_layers=4` — criterio escrito ANTES de mirar

> ⚠ **Geometría: este documento usa la ortografía anterior al 2026-08-25.** `N`, `c_frac`,
> `d` y `pen_frac` fueron reemplazados por longitudes en px reales (`fovea_px`, `border_px`,
> `border_reduce`, `overlap_fovea_px`, `overlap_border_px`). **Ninguna red cambió** — es un
> cambio de nombre, verificado bit a bit — así que **todos los números de aquí siguen siendo
> válidos**. La traducción está en [instructionsNewNN.md](../instructionsNewNN.md) §2.1.
> Ojo con uno: **`d` cambió de significado**, no sólo de nombre. Antes agrandaba el contexto
> (`borde = celdas·d`); hoy `border_reduce` sólo dice cuánto se comprime un borde de tamaño
> fijo. Un eje `d` de aquella época medía **área y compresión a la vez**.

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

**La tarea programada** (registrada el 2026-08-09, `fv-lrL4-watchdog`): cada **10 min** + al
iniciar sesión, `StartWhenAvailable`, sin parar con la batería, `IgnoreNew`.

⚠ **Va como tarea del usuario, no como SYSTEM.** El watchdog anterior era SYSTEM; registrarla así
**exige elevación** y esta sesión no la tiene (`Acceso denegado`). La consecuencia hay que decirla:
una tarea de usuario **no corre hasta que el usuario inicia sesión**, así que tras un apagón el
recorrido se reanuda **al entrar**, no al arrancar la máquina. Para el caso de uso —un equipo de
escritorio que el usuario enciende— alcanza; si alguna vez hace falta que reanude sin nadie
delante, hay que registrarla elevada.

```powershell
# desregistrarla al terminar el recorrido
Unregister-ScheduledTask -TaskName "fv-lrL4-watchdog" -Confirm:$false
```

**Qué está verificado del watchdog, y qué no.** Verificado el 2026-08-09 con el recorrido **vivo**:
(a) lo detecta y **no duplica** (`LastTaskResult 0`, ninguna línea de relanzamiento, ningún proceso
nuevo); (b) la **sonda de permisos ejecutada desde el contexto de la tarea** —no desde una consola—
arranca un proceso con el working dir del proyecto y escribe en su carpeta. **No verificado**: el
relanzamiento real de un recorrido muerto, que exigiría matar el entrenamiento. La sonda existe
precisamente porque probar `& python` no ejerce ni el cwd ni la redirección, que es donde falla.

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

## 6. RESULTADO DE LA SONDA (2026-08-08 16:31) — y la regla §2 aplicada

`p40-lr-L4-0000-lr0p00035_seed1`: **70 épocas**, mejor la **60**, `stopped_early: true` — **paró
`patience`, no el tope**. 2 h 00 min, **103 s/época**. Cae la rama 1 de §2: el punto es válido
(R1) y el recorrido se re-estima con el número medido.

⚠ **La extrapolación `épocas ∝ 1/lr` era mala, y por mucho.** Predecía ~188 épocas (topando en
150); salieron **70**. Con los dos puntos medidos —47 épocas a `lr` = 0,0014 y 70 a 0,00035— la ley
real es **`épocas ∝ lr^-0,287`**, no `lr^-1`. Convergir con `lr` bajo cuesta **bastante menos** de
lo que se temía. Es la razón entera por la que la sonda existía.

| `lr` | épocas | h/run | × 5 semillas |
|---|---|---|---|
| 0,00035 | **70 (medido)** | 2,04 | 10,2 |
| 0,0006 | ~60 | 1,75 | 8,7 |
| 0,0009 | ~53 | 1,56 | 7,8 |
| 0,0014 | **47 (medido)** | 1,37 | 6,9 |

**Total ≈ 33,6 h**, de las que **2,0 ya están hechas** → **quedan ~31,5 h**. Está **por debajo del
umbral de 40 h** de §2.1, así que **no se aplica ninguna guarda**: el recorrido se corre como está
especificado, 4 valores × 5 semillas. La regla estaba escrita antes y se cumple sin tocarla.

⚠ **Sensibilidad honesta**: las 5 semillas de `lr` = 0,0014 pararon entre 32 y 61 épocas (±30 %).
Con esa dispersión el total va de **~23 h a ~44 h**, y el extremo alto **sí** cruzaría el umbral.
No se re-planifica por eso: la regla mira la estimación central, y el recorrido es reanudable y
parable.

**Lo que la sonda insinúa sobre la pregunta** (y solo insinúa): `val_f1` del checkpoint **0,9254**,
contra **0,9105** del mismo par red-semilla a `lr` = 0,0014. Va en la dirección de «el óptimo está
más a la izquierda». ⚠ **Es una semilla y no decide nada** (§2.3): el cribado del plan de 40 h, con
una semilla, se equivocó de signo en la métrica de tarea. La respuesta la dan las 5 semillas y R3.

## 7. RESULTADO (2026-08-10 22:12) — **el eje es plano; el vigente se queda**

20/20 puntos, **36,9 h** de cómputo reales contra 33,6 estimadas (dentro de la banda de
sensibilidad de §6, que iba de 23 a 44 h).

| `lr` | ventana (f1) | sem | **tarea** | sem | épocas (5 semillas) |
|---|---|---|---|---|---|
| 0,0014 (vigente) | 0,9244 | 0,0041 | 0,7796 | 0,0074 | 32 · 35 · 51 · 58 · 61 |
| 0,0009 | 0,9260 | 0,0010 | 0,7809 | 0,0072 | 45 · 47 · 47 · 50 · 59 |
| **0,0006** | **0,9293** | 0,0020 | 0,7863 | 0,0101 | 49 · 53 · 55 · 58 · 67 |
| 0,00035 | 0,9231 | 0,0016 | **0,7892** | 0,0031 | 60 · 62 · 62 · 70 · 71 |

**R1 ✅ — el recorrido es válido.** Los **20 runs** pararon por `patience` (`stopped_early: true`
en 20/20), entre 32 y 71 épocas, **ninguno cerca del tope de 150**. Ningún punto mide presupuesto.
El tope alto era caro y era lo correcto.

**R2 — ganador por ventana: `lr = 0,0006`** (0,9293), y `suggest_winner` lo declara *distinguible*
con δ = 0,0020.

**R3 ✅ — el óptimo QUEDA ACOTADO por la izquierda, y esa era la pregunta.** El ganador es
**interior**, y el extremo izquierdo (0,00035) es **el peor de los cuatro** por ventana. Bajar el
`lr` deja de ayudar antes de 0,00035: la anomalía que motivó todo esto —un óptimo pegado al borde—
**está cerrada**. (Por la métrica de tarea el orden es el contrario y el ganador *sí* es el borde;
ver R5, donde nada de eso se separa del ruido.)

**R4 ❌ — pero NO le gana al vigente, así que el vigente se queda.** `0,0006` contra `0,0014`:
**+0,0049 de ventana con p = 0,341** (permutación exacta, 252 arreglos), y **+0,0066 de tarea con
p = 0,651**. El umbral escrito antes era **p ≤ 0,05**. **`lr` sigue siendo 0,0014** — no porque
0,0006 sea peor, sino porque no hay con qué distinguirlos.

> ⚠ **Y aquí sale un hallazgo que vale más que el recorrido: δ y la permutación no dicen lo mismo,
> y δ es la optimista.** Sobre los **mismos 20 números**, `suggest_winner` imprime *«el mejor punto
> despega del resto por más de δ = 0,0020: la diferencia supera la banda de ruido medida»* mientras
> la permutación exacta da **p = 0,341**. No es un bug: δ es **1-SE de las semillas del mejor punto
> y solo de ese**, así que (a) ignora la dispersión del punto contra el que compara —aquí `0,0014`
> tiene un `sem` **dos veces mayor**— y (b) 1 SE es una banda de ~68 % sobre *una* media, no una
> prueba de diferencia entre dos. La regla está escrita como criterio de **empate** (protocolo.md
> §1.5) y para eso sirve; **la frase que imprime afirma más de lo que el número aguanta**, y esa
> frase es la que lee un estudio OAT al arrastrar un ganador. Los veredictos ya publicados **no se
> caen** —`n_layers` L4 vs L2 son 12× δ y p = 0,032—, pero cualquiera cuyo margen esté cerca de δ
> hay que releerlo. **Anotado como pregunta abierta, no arreglado aquí**: cambiar la regla de
> selección toca todos los estudios y es decisión del usuario.

**R5 — la métrica de tarea: el eje no distingue nada.** El orden por tarea es **monótono al revés**
del de ventana (0,00035 el mejor, 0,0014 el peor), pero **ninguna diferencia se separa de
reetiquetar las semillas**: el ganador contra los otros tres da **p = 0,817 · 0,341 · 0,302**. Con
eso, el Spearman agregado de **−0,200** **no es un hallazgo sobre el proxy**: es ruido ordenando
ruido. `scripts/proxy_vs_task.py` ahora lo dice solo — se le añadió la guarda que declara
**`no_concluyente`** cuando ningún par de tarea baja de p = 0,05, en vez de un «NO» que se leería
como *el proxy falla*.

### 7.1 La conclusión, en una frase

**Entre 0,00035 y 0,0014, el `lr` no cambia nada medible en L4** — amplitud 0,0062 de ventana y
0,0096 de tarea, con todos los pares por encima de p = 0,2. Junto con `d1000-lr-1`, que sí midió
degradación **por encima** de 0,0014, el dibujo que queda es una **meseta ancha** que llega al
menos hasta 0,00035 y se rompe hacia arriba. Aquel estudio encontró **el borde derecho de la
meseta**, no un óptimo.

**Qué se hace con esto:** nada, y eso es el resultado. `lr = 0,0014` se queda por R4; el eje queda
**cerrado** (R3) y no merece más cómputo. Lo caro fue descubrir que es plano — pero un eje que no
mueve la aguja es exactamente lo que hay que dejar de barrer en los estudios siguientes.

**Reproducirlo:**

```powershell
.\.venv\Scripts\python.exe scripts\proxy_vs_task.py --sweep p40-lr-L4 --split val
```

Detalle por run y por punto en `data/p40-lr-L4-task.json` (comiteado).
