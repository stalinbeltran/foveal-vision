# foveal-vision — instrucciones para Claude

**El mismo problema que `image-text-finder`, con otra red y con recorridos automáticos.** La
tarea es idéntica al proyecto hermano: detección de esquinas de párrafo por ventana (`TL, TR,
BR, BL` con `[exists, x, y]`) y reconstrucción de párrafos en la imagen — más adelante líneas,
y luego palabras. Lo que cambia: **la red** es una NN con **muestreo foveado y ramas por
región**, totalmente parametrizada (el centro a resolución completa; la periferia abarca más
área a menor resolución; dos ramas convolucionales que se suman en la banda de penetración), y
**los recorridos (sweeps) barren automáticamente las configuraciones de esa red** además de las
recetas.

**La especificación de la red es [instructionsNewNN.md](instructionsNewNN.md)** — ese documento
manda sobre todo lo que toque la arquitectura. Su principio rector gobierna el proyecto entero:
*todo dato es un parámetro*; las dimensiones y los **rangos de búsqueda se calculan** a partir de
`N` y unas pocas fracciones, nunca se escriben a mano.

El objetivo operativo: poder **preparar series de runs secuenciales** —pruebas cortas en esta
máquina (CPU), luego largas en un server con GPU— que recorran configuraciones de red y
parámetros **sin intervención humana** (recetas de recorrido), y poder **verificar cada objeto
creado** (fuente, dataset, red, run, recorrido, análisis) desde una web app.

---

## Estado actual — léelo primero

> **⏳ 2026-08-09 — HAY UN RECORRIDO CORRIENDO: `p40-lr-L4` (~31 h, 19 puntos por hacer).**
> Re-barrido de `lr` sobre la red **L4**, porque el valor vigente (0,0014) se fijó sobre **L2**, con
> 20 épocas fijas que los 70 runs agotaron, y **quedó pegado al borde izquierdo de su rango**.
> **El criterio está escrito antes de mirar en [docs/plan-lr-L4.md](docs/plan-lr-L4.md)** y
> comiteado antes de que existiera un solo run (`68df83b`). Si lees esto y sigue corriendo, **no lo
> toques**: es reanudable y hay watchdog (`fv-lrL4-watchdog`, tarea de usuario, cada 10 min).
> - Rango `[0,00035 · 0,0006 · 0,0009 · 0,0014]`, 5 semillas, tope **150** épocas para que pare
>   `patience` y no el tope. Orden `[0,00035 · 0,0014 · 0,0006 · 0,0009]`: si se corta, lo que falta
>   son los **interiores**, no los extremos, que son la pregunta.
> - **La sonda ya midió lo que nadie había medido**: a `lr`=0,00035, `patience` salta en la **época
>   70** (no en las ~188 que predecía `épocas ∝ 1/lr`). La ley real es **`épocas ∝ lr^-0,287`**, así
>   que el recorrido cuesta **33,6 h** y no 56. La regla §2.1 (umbral 40 h) → **sin guardas**.
> - **Al cerrar hay que aplicar R1–R5 del documento**, y en particular: **R3** — si vuelve a ganar
>   el extremo izquierdo, «sigue sin acotar» es **el resultado que se publica**; y **R5** — nada se
>   arrastra sin pasar por `scripts/proxy_vs_task.py` (la ventana exageró la ganancia al doble en
>   `n_layers`, medido el 2026-08-08).
> - Al terminar: `Unregister-ScheduledTask -TaskName "fv-lrL4-watchdog" -Confirm:$false`.

> **2026-08-08 — LA PROFUNDIDAD GANA: `n_layers=4` CONTRA LAS 2 DE HOY, SIN SOLAPAMIENTO.**
> Plan desatendido de ~37 h (30,4 h de cómputo, 24 runs) especificado **antes** en
> [docs/plan-40h.md](docs/plan-40h.md) y comiteado antes de que existiera un solo run (`b8545db`).
> Lo que hay que saber:
> 1. **`n_layers` 2 → 4 sube el f1 de ventana de 0,8756 a 0,9244**, 5 semillas cada uno, y **las
>    bandas son disjuntas**: la peor semilla de L4 (0,9105) gana a la mejor de L2 (0,8804). `f1` y
>    `loss` coinciden en el orden y ninguno declara empate (δ=0,0041). Cuesta 106 s/época contra 60.
> 2. ⚠ **`n_layers` NO es un eje de capacidad, es de campo receptivo.** El **97 % de los parámetros
>    está en la cabeza** (153.612 de 158.572); una capa más añade 4.640 (+3 %). Por eso **doblar los
>    canales no sirve** (`ch[32,32]`: +0,0046, dentro de δ, con 2× los parámetros y **más** coste) y
>    **agrandar el kernel tampoco** (`k_center=5`: −0,0063, peor que la base). Lo que gana es
>    **apilar capas**, no ensancharlas.
> 3. **El óptimo está acotado por los dos lados**: L3 = 0,9093 y L5 = 0,8832. Pero L5 **no es peor,
>    es inestable**: sus semillas son **bimodales** — 0,8471 / 0,8581 / 0,8620 contra 0,9279 /
>    0,9209. O arranca o no arranca, sin valores intermedios; sem 0,0170 contra 0,0041 de L4. Para
>    pasar de 4 capas hay que cambiar **algo más que el número** (residuales, otra inicialización).
> 4. ⚠ **El presupuesto de 20 épocas de los estudios `d1000-*` medía velocidad, no calidad.** Los 70
>    runs tienen su mejor época ≥16, y 65 seguían mejorando entre la 15 y la 20 (caída media 0,0127
>    de `val_loss`, **más de la mitad de la amplitud completa del eje de `lr`**). Con `patience=10`
>    la misma base pasa de f1 0,8437 a 0,8789. **`patience` mínimo seguro = 8**: la racha más larga
>    sin mejorar seguida de una mejora es de 6 épocas (medido sobre esos 70 runs).
> 5. ⚠ **`patience` mete varianza por la puerta de atrás**: cada semilla para donde quiere (32–71
>    épocas) y **la que entrena más lejos aterriza más abajo**. Las bandas de este recorrido son más
>    anchas que las de los estudios de época fija, y no es ruido de medición, es el criterio de
>    parada.
> 6. **Dos bugs del propio plan, encontrados corriendo** y anotados en plan-40h.md §7: la guarda de
>    presupuesto recortaba el rango **a los 3 valores más baratos, dejando fuera al ganador**; y
>    arrastrar la config ganadora **menos el eje** no basta — hay que soltar también **lo acoplado**
>    (`channels` con `n_layers`), o `check_run` rechaza la base. Es otra vez «el mismo dato en dos
>    sitios»: la profundidad vive en `n_layers` **y** en `len(channels)`.
> 7. ⚠ **El micro-benchmark de coste miente bajo carga**: medido con un entrenamiento ocupando los
>    núcleos daba 34 h para el mismo recorrido que, con la máquina libre, sale en 22 h.
> **Artefactos**: `p40-screen-{base,depth,width,kernel}` (cribado, 1 semilla) + recorrido
> **`p40-confirm-n_layers`** (20 runs, 4 valores × 5 semillas). Receta nueva `plan40`.
> **Lo que sigue teniendo más valor**: ~~(a) la métrica de tarea~~ **hecha, ver la nota de abajo**;
> (b) probar residuales para desbloquear >4 capas; (c) re-barrer `lr`/`batch_size` sobre L4, porque
> se fijaron sobre L2 y con 20 épocas — y el `lr` ganador quedó **pegado al borde izquierdo** de su
> rango, sin acotar.

> **2026-08-08 (2) — LA PROFUNDIDAD GANA TAMBIÉN POR TAREA, PERO LA MITAD Y SIN BANDAS DISJUNTAS.**
> Paso (a) de la nota anterior, hecho sobre los **mismos 20 runs**, sin reentrenar nada (5,4 s de
> inferencia por run). Detalle en [docs/metrica-de-tarea.md](docs/metrica-de-tarea.md) §2 ter y en
> [docs/plan-40h.md](docs/plan-40h.md) §8. Lo que hay que saber:
> 1. **`n_layers=4` gana con las dos métricas**: tarea **0,7796** contra **0,7572** de L2, y la
>    diferencia sobrevive a una **permutación exacta** de las semillas (**p = 0,032**, 252 arreglos).
>    L3 y L5 **no** se separan de L4 (p = 0,135 y 0,167). El split de este dataset son **200
>    imágenes** de val, así que el aviso de muestra pequeña ya no salta (`sem` por run ±0,023).
> 2. ⚠ **Corrige el titular de la nota anterior**: la ganancia se encoge a la mitad (+0,0488 →
>    **+0,0224**) y **las bandas se solapan** en tarea (peor L4 = 0,7532 < mejor L2 = 0,7689).
>    «Bandas disjuntas» era una propiedad **del proxy**, no del resultado. Cítese siempre así.
> 3. ⚠ **El f1 de ventana EXAGERA el hundimiento de L5.** Su bimodalidad —amplitud 0,081 en
>    ventana— es de **0,029** en tarea, y los dos grupos **se entrelazan**: una red que «no arranca»
>    según la ventana sigue reconstruyendo párrafos casi igual de bien.
> 4. **El criterio de §5.4 sale NO** (Spearman agregado **+0,800** < 0,90) y aun así **§5.5 no se
>    ejecuta**: el fallo es **un solo intercambio** entre L3 y L5, dos puntos que la tarea **tampoco
>    distingue** (0,0010 de diferencia, **p = 0,897**). No hay orden que acertar. Queda anotado como
>    reserva del proxy en ejes de profundidad, no como su refutación.
> 5. ⚠ **El cribado de 1 semilla no habría visto nada por tarea**: base 0,7523 vs depth 0,7532
>    (+0,0009 contra ±0,023). Que el plan funcionara fue suerte del proxy. Y **`k_center=5`, el peor
>    por ventana, es el mejor de los cuatro por tarea** (0,7594) — con 1 semilla no afirma nada,
>    pero **no queda descartado**: es el candidato barato si se vuelve a barrer estructura.
> 6. **Pieza nueva `fv.metrics.permutation_test`**: exacta hasta C(n+m,n) ≤ 200.000 y **se niega**
>    por encima en vez de pasar a muestreo en silencio (un p que cambia entre corridas no decide
>    nada). La imprime `proxy_vs_task.py` para todo recorrido con varias semillas — porque la
>    correlación dice si las métricas *coinciden*, nunca si la diferencia es *real*.
> ⚠ **Verificado, no razonado**: **132 tests** (+5), los **dos comandos del README reproducen sus
> números documentados** (+0,737 / +0,956 y el veredicto OK de `fast-lr-s0-lr`: sin regresión), y el
> comando nuevo ejecutado en PowerShell tal como está escrito. Detalle comiteado en
> `data/p40-n_layers-task.json`. **Cero entrenamiento, cero artefactos borrados.**

> **2026-07-28 — EL SOBRE DEL FICHERO SE ESCAPÓ POR LA LISTA: GUARDAR UNA RECETA DABA 400.**
> El usuario reportó `[unknown_recipe_fields] campos desconocidos: ['format_version']` en
> **Recetas**. Otra vez **el mismo dato en dos sitios**, y otra vez en una costura sin numerar:
> `RecipeStore.get()` quitaba el sobre del fichero (`name`, `format_version`) y **`list()` no**;
> la pantalla rellena su formulario con **una fila de la lista** y lo devuelve tal cual, así que
> `save()` rechazaba como «campo desconocido» **lo que el propio API acababa de servir**. Peor:
> el formulario se recuerda (`localStorage`), así que **un clic en cualquier fila envenenaba el
> guardado para siempre**, también con un nombre nuevo. Arreglo en la única definición posible —
> `fv.ioutils.strip_envelope` / `with_envelope` — aplicada a **las tres puertas** de las **dos**
> tiendas de config (D y C: `NetworkStore.list()` filtraba igual, sin dar error todavía) y al
> formulario, que ahora manda solo lo que D define. Documentado en formatos.md §4.3.
> ⚠ El contrato ⑦ frenó el import: ahora `settings` e `ioutils` son **hojas comprobadas** (el test
> verifica que no importan ningún dominio) en vez de excepciones concedidas a mano.
> ⚠ **Verificado, no razonado**: **123 tests** (+1: la vuelta lista→guardar, y que un campo de
> verdad desconocido **sigue dando 400**), `npm run build` limpio, y el flujo pulsado con
> Playwright **en la instancia del usuario** (clic en la fila → Guardar → 200), con **control**:
> con el código anterior la misma fila devolvía 400. La receta temporal creada se **borró**.
> **2026-07-28 (3) — EDITAR UNA RECETA ERA IMPOSIBLE DESDE LA UI (409 «elige otro nombre»).**
> Reportado por el usuario al intentar cambiarle un número a `mejorada`. Los dos stores aceptan
> `overwrite` **desde el día 1** y **ninguna pantalla lo enviaba nunca**, así que abrir una
> definición guardada y guardarla acababa siempre en `recipe_exists` con un consejo absurdo
> («elige otro nombre, o edita esa» — *estaba* editando esa). Lo que hay que saber:
> 1. **Regla nueva `U5.11`** (79 reglas), el reverso explícito de U5.8: **un run no se sobrescribe;
>    una red y una receta son fuente y se editan**. Dos acciones distintas — «Guardar» con nombre
>    nuevo, **«Actualizar» + confirmación** con uno que ya existe — para no confundir el accidente
>    con la intención. `overwrite` es bandera de la petición, **nunca campo del objeto** (test).
> 2. **La confirmación dice qué NO cambia y qué SÍ**: los runs y recorridos hechos copiaron los
>    valores; los **estudios que la fijan por nombre** re-resuelven en su próximo `advance`. Por eso
>    `GET /recipes` sirve **`used_by`** (`{receta: [estudios]}`) — **mapa aparte**, no dentro del
>    objeto: un campo mezclado ahí vuelve en el siguiente guardado como «desconocido» (la lección
>    del sobre, dos notas más abajo). Verificado con `corta`, que la fijan **5 estudios**.
> 3. **Redes tenía el mismo agujero** y el mismo arreglo (sin `used_by`: un recorrido congela
>    `base_network_value` al crearse, así que nadie fija una C por nombre para el futuro).
> ⚠ **Verificado, no razonado**: **127 tests** (+2), `verify_spec --live` **79 reglas, 65 ok, 0
> violadas, 82 %**, `verify_ui.py` 12 pantallas limpias, y la edición hecha **en la pantalla** sobre
> una receta temporal (3 → 12 épocas en disco, `overwrite` no guardado) que se **borró** después.
> La receta `mejorada` del usuario y las versionadas **no se tocaron**: de `corta` solo se abrió la
> confirmación —para leer los 5 estudios— y se canceló.

> **2026-07-28 (2) — UN OBJETIVO NO ES UN MONITOR, Y EL `<select>` LO ENSEÑABA MAL.**
> Descubierto al verificar lo anterior y arreglado a petición del usuario. El `<select>` de
> `monitor` se llenaba con los **objetivos** (`f1`, `loss`, `pos_err_px`); el default de la receta
> es `val_loss`, que no está en esa lista, y **un `<select>` cuyo value no está entre sus opciones
> dibuja la primera y calla** → enseñaba `f1` guardando `val_loss`. Lo grave no era la pantalla:
> 1. ⚠ **Elegir `f1` ahí habría corrompido `best.pt` en silencio.** `monitor_key("f1")` encuentra
>    el valor, pero la dirección vivía en un `frozenset({"val_f1"})` aparte → `f1` caía en
>    «menor es mejor» y el checkpoint se habría quedado con la **peor** época. Ni un aviso.
>    **Ningún artefacto está afectado**: los 708 `monitor` en disco dicen `val_loss` (medido).
> 2. **El mismo dato en dos sitios, otra vez**: `MONITOR_HIGHER_IS_BETTER` y `OBJECTIVES` eran la
>    misma tabla escrita dos veces, y las mitades no se conocían. Ahora **`fv.metrics.VAL_METRICS`
>    manda** (`MONITORS = val_ + cada métrica`) y `OBJECTIVES = dict(VAL_METRICS)`.
> 3. **Tres puertas dicen que no** con el mismo código `unknown_monitor`: la receta (guardar **y**
>    leer un yaml editado a mano), el eje `monitor` de un recorrido (`check_sweep`) y el de un
>    estudio (`validate_plan`). Antes **ninguna** validaba el valor.
> 4. **El API sirve el vocabulario** (`GET /recipes` → `vocabulary.monitor`) desde la constante
>    contra la que valida la puerta; y un valor que no pertenece se dibuja **`(no reconocido)`**,
>    nunca sustituido por uno plausible.
> 5. **Regla nueva `U5.10`** (78 reglas) con **verbo nuevo** `select_matches_served_vocabulary`:
>    el DOM no conserva rastro de la mentira, así que la comprobación viene de fuera (las opciones
>    **son** las servidas; lo mostrado **es** lo guardado).
> ⚠ **Verificado, no razonado**: **125 tests** (+2), `npm run build` limpio y el flujo pulsado en
> la instancia del usuario — opciones = vocabulario servido, enseña `val_loss`, un `f1` recordado
> sale `f1 (no reconocido)` y guardarlo devuelve `[unknown_monitor]` con razón y arreglo. Y
> `verify_spec --live`: **78 reglas, 64 ok, 0 violadas, 82 %** (U5.10 medida por su propio verbo),
> más `verify_ui.py` con las 12 pantallas sin un error de consola. Para correr `--live` hubo que
> **parar el vite del usuario** (con su permiso) y **se le devolvió levantado**.
> ⚠ La primera corrida volvió a medir al validador, no al código: `U5.10` reventó (`source:
> "GET /recipes"` se pasaba entero como ruta) y `U7.11` cazó que un **`data-testid` calculado no
> existe** para el escáner, que lee literales del fuente. Los tres selects lo llevan literal ahora.
> ⚠ **Corrección de la nota anterior**: en la primera verificación la semilla de `localStorage`
> usaba la clave sin el prefijo `fv.ui.`, así que **esa mitad no probó nada** (la del clic en la
> fila sí). Rehecha con la clave real: pasa. **Lección: una aserción que no puede fallar es peor
> que ninguna** — verificar el efecto (el fichero escrito), no solo la ausencia de error.
> **Queda abierto**: F16 sigue sin decidir (los enums `optimizer`/`scheduler` siguen copiados en
> `Recipes.tsx`; `monitor` ya no).

> **2026-07-28 — LISTAR NO ES VERIFICAR: LA REGLA U1.6 Y EL PLAN DE UN ESTUDIO.**
> El usuario reportó que **los parámetros de un estudio no vuelven a verse una vez creado**. Se
> documentó primero (a petición suya) y se implementó después. Lo que hay que saber:
>
> 1. **Regla nueva `U1.6`** en [docs/ui/1-estructura.md](docs/ui/1-estructura.md) — *un objeto
>    enseña entera la definición con que se creó, y la enseña en su detalle*. Va en el tipo 1 junto
>    a U1.5 («verificar un objeto no exige entrenar») porque es la misma exigencia. Fija cuatro
>    cosas: se lee **del objeto guardado**, nunca del formulario recordado (U7.3); **definición y
>    progreso separados** en pantalla como ya lo están en disco; los valores compuestos **completos**
>    (el rango de un eje es su lista, no su longitud) y el presupuesto **con unidad**; ausente se
>    dibuja como ausente. **77 reglas** ahora (78 desde U5.10, ver la nota de arriba).
> 2. **Estudios lo cumple**: bloque `study-plan` (B, receta D, objetivo, semillas, presupuesto) +
>    `study-axes` (la escalera con el rango literal y `hecho`/`en curso`/`pendiente` por eje, sacado
>    del progreso vivo); debajo, «El progreso (lo que ha pasado)». **`channels[i]` reconoce sus
>    propios sub-pasos** (`channels[0..L-1]`) — la única parte no trivial.
> 3. ⚠ **El bloque no enumera los campos que conoce y calla el resto**: lo que `plan.json` traiga y
>    la pantalla no nombre se pinta igual bajo su clave. Un campo añadido en Python **no puede
>    volverse invisible** aquí. Escribir la regla ya destapó uno: **`budget` no estaba descrito en
>    [docs/formatos.md](docs/formatos.md) §4.7** aunque lo guarda `POST /studies` y lo lee
>    `driver.advance`. Documentado.
> 4. **Y un fallo del propio validador**: `sibling_required` **dormía 300 ms fijos** esperando al
>    DOM — justo lo que `validador.md` §8 prohíbe por escrito tras habérselo cobrado ya una vez.
>    Ahora espera al selector; si no, el check de U1.6 habría sido intermitente.
>
> ⚠ **Verificado, no razonado**: `verify_spec --live` **63 ok, 0 violadas, 81 %** (tipo 1 al 100 %),
> `verify_ui.py` con las 12 pantallas y las aserciones nuevas sobre **los 5 estudios**, 122 tests,
> `npm run build` limpio. Los estados que ningún estudio real ejercitaba (`auto`, `channels[i]`
> expandido, un eje en cola) se probaron con un estudio temporal creado por HTTP y **borrado**
> después (arrastre comprobado antes: 0 recorridos). Para correr `--live` hubo que **parar el vite
> del usuario** (con su permiso) y **se le devolvió levantado**; su backend de `:8010` no se tocó.

> **2026-07-27 — LA ESPECIFICACIÓN DE LA UI, EN OCHO TIPOS, Y UN VALIDADOR QUE LA COMPRUEBA.**
> A petición del usuario se sintetizaron los **tipos** de especificación que rigen la UI y luego se
> construyó lo que los hace cumplir. Lo que hay que saber:
>
> 1. **`docs/ui.md` es ahora un índice**; las reglas viven en **`docs/ui/`, una por tipo** (1
>    estructura · 2 vistas · 3 representación · 4 datos · 5 invariantes · 6 números · 7 operación ·
>    8 léxico), **76 reglas numeradas y citables** (`U4.2`, `U6.7`…). El contenido se **movió**, no
>    se copió: dejarlo en los dos sitios era el modo de fallo que este proyecto tiene registrado.
> 2. **`scripts/verify_spec.py` valida esa especificación y se alimenta de ella**: cada regla lleva
>    pegado un bloque ` ```check ` (opción **A2**: el markdown manda). Motor híbrido (**C3**): 12
>    verbos declarativos + handlers nombrados; lo que no encaja se declara `substrate: none` **con
>    su razón** y sale `no_verificable`, nunca `ok`. Informe **por regla en cuatro estados**; salida
>    ≠ 0 solo con `violada`. **Diseño y lecciones: [docs/ui/validador.md](docs/ui/validador.md).**
> 3. **Medido: 81 % de cobertura mecánica** (`--live`: 62 ok, **0 violadas**, 5 no verificables, 9
>    no aplicables) y 39 % en estático. La cobertura **la calcula la herramienta**; no se mantiene a
>    mano en ningún documento.
> 4. **Lo que encontró, y estaba vivo**: los **estados de run escritos cuatro veces**, con una copia
>    esperando un estado `failed` **que no existe** y **ninguna** conociendo `interrupted` — un run
>    interrumpido no se marcaba terminal y **su curva se re-pedía en cada sondeo, para siempre**;
>    una **lista de objetivos** duplicada en `Recipes.tsx`; el **umbral `n<100` en dos sitios**
>    (ahora el veredicto `small_sample` viaja con el número); cuatro **colores literales** que eran
>    segundas definiciones de tokens —y al quitarlos apareció que el mapa secuencial **no seguía al
>    tema**—; y un **400 documentado en api.md que el código no daba**. Todo arreglado.
> 5. **`npm run validate:palette` existe** (era deuda declarada desde el día 1). Una implementación,
>    dos entradas: el validador de Python **ejecuta el mismo script**. Paleta medida: claro ΔE 9,1
>    (protan) y suelo de visión normal 19,6; oscuro 8,4 / 19,3.
> 6. ⚠ **Dos avisos de la herramienta sobre sí misma.** (a) Un check con `DELETE` **borró un dataset
>    real** antes de que existiera la guarda; se recuperó y se regeneró **bit-idéntico**, y ahora
>    todo lo que puede escribir está **bloqueado por defecto**. (b) De las 11 primeras «violadas»,
>    **10 eran checks mal escritos**: la primera corrida de un validador mide al validador.
> 7. **Cuatro preguntas abiertas anotadas, no decididas**: **F16–F19** en decisiones.md (si el API
>    debe servir estados y enums de receta; qué runs se ofrecen en Diagnóstico/Predecir; el alcance
>    del relieve del WARN de contraste; y si se anotan los componentes con `data-*` — hoy sí).
>
> ⚠ **Verificado, no razonado**: 122 tests en verde, `verify_spec --live` en 0 con backend y front
> propios (arrancados y **parados**), `verify_ui.py` con las **12 pantallas sin un error de
> consola**, `npm run build` limpio y `npm run validate:palette` en verde.

> **2026-07-26 — EL PROXY DE VENTANA VALE TAMBIÉN PARA C, Y LA PERIFERIA NO ESTÁ APORTANDO.**
> Cerradas las **Fases 3b y 4-código** de metrica-de-tarea.md; la **3 queda aplazada por decisión
> del usuario**. Lo que hay que saber, por orden de importancia:
>
> 1. **Fase 3b ✅ (§2 bis del doc).** Recorrido `proxy-c-d` (eje **`d`**, dominio C: 6 valores × 5
>    semillas, 20 épocas, **68 min** de CPU). **Spearman agregado +1,000**, mismo ganador (`d=2`)
>    por ventana y por tarea, dentro de la frontera δ. **`OBJECTIVES` NO cambia y §5.5 no se
>    ejecuta**: el ranking barato se queda. El criterio estaba escrito antes de mirar y es
>    **comprobable** — las constantes viven en `scripts/proxy_vs_task.py`, commiteado en `7dd34ad`,
>    antes de que el recorrido terminara. Reservas: el eje separa poco (amplitud 0,028; δ se come
>    3 de los 6 puntos), n=6, y es **un solo** eje de C.
> 2. ⚠ **LA PERIFERIA NO ESTÁ APORTANDO DE FORMA MEDIBLE** (§2 bis.1) — es media respuesta a *la*
>    pregunta del proyecto (protocolo.md §6) **sin construir el control que F12 bloquea**. El
>    máximo está en `d=2` (**4 px** de periferia); `d=1` —casi sin contexto— queda **segundo**; y
>    `d=6` (12 px) de los últimos. El coste no lo explica: 7,0–8,8 s/época en todos. Honestidad:
>    mejor−peor son **1,43 SE** y `d=5` rompe la tendencia, así que se afirma «no ayuda de forma
>    medible», **no** «estorba».
> 3. **El cuello de botella es la red, y es de DETECCIÓN.** §9.1: con esquinas perfectas la
>    reconstrucción actual da **0,97** (19/20 imágenes perfectas; el NMS no suprime nada), contra
>    0,6448 del mejor modelo real → los 0,33 que faltan son **todos** de detectar esquinas, no de
>    `_reconstruct`. §9.3: ni aflojando el IoU a 0,3 se pasa de **0,66** — un tercio de los párrafos
>    no se detecta en absoluto. **Ahí está el trabajo.**
> 4. **Ocho de las diez pruebas de §9, hechas — y dos corrigen al documento.** §9.4: la sd por
>    imagen **sube a 0,4148** (se suponía que bajaría de 0,372: es máxima con modelos
>    *intermedios*, y el 0,372 promediaba modelos de F1 0,10) → hoy **±0,093** por run, tabla de
>    §4.1 rehecha. §9.2: **los tres defaults de F están mal** y el óptimo es **el mismo** en tres
>    runs muy distintos (`threshold≈0,3`, `stride n/4`, `nms 3n/4`), acotados por dentro; deja
>    +0,065/+0,187/+0,261 sobre la mesa. §9.5: macro≈micro, pero **7 de 20 imágenes cargan casi
>    todo el fallo** y la 60 falla en 20/20 réplicas (con techo 1,000 → es la red). §9.7: `f1` es el
>    mejor proxy con diferencia (`loss` +0,780, `pos_err_px` +0,544, **y eligen otro ganador**).
>    §9.6: el `sem` aguanta el bootstrap (0,973×) → **el bloqueo es la `n`, no la fórmula**.
> 5. **Cuatro decisiones cerradas por el usuario** (decisiones.md): **F11 — no se regenera el dato
>    por ahora** (la métrica de tarea es *informe del ganador*, nunca criterio entre puntos; se
>    conserva la comparabilidad); **F13** aparcada con ella; **F15 — los knobs de F no se tocan**
>    pese a §9.2; **F14 — sí se registra que el holdout se tocó**, y está construido.
> 6. **Fase 4: todo el código, hecho y probado; falta la fuente** (depende de F11). Selectores de
>    dataset/split en el detalle de un run con el aviso de «se toca una vez»; y **F14**: cada
>    medición contra un holdout anexa una línea a `runs/<run>/holdout.jsonl` **también cuando el
>    número sale de caché** —ese era el vistazo invisible—, con `holdout_touches` en el payload y
>    en ámbar en la UI. Append-only: **registra miradas, nunca bloquea una**. Qué cuenta como
>    holdout lo dice **una sola función** (`is_holdout_source`): el campo `"holdout"` del
>    `dataset.json` manda **en los dos sentidos** y el nombre `-holdout` es el respaldo.
> 7. **Piezas nuevas reutilizables**: `fv.metrics.spearman` (empates a rango medio, **None** —nunca
>    0— si una serie es constante); `scripts/proxy_vs_task.py` (no calcula ninguna métrica por su
>    cuenta y **descuenta diciéndolo** los runs sin checkpoint); `sweep_trials`/`suggest_winner`
>    aceptan `objective=` para **re-leer** un recorrido con otro proxy sin tocar el spec, y lo
>    **declaran** (`objective_overridden`); `loader.source_meta` unifica los dos lectores que había
>    de `dataset.json`.
>
> ⚠ **Verificado, no razonado**: **122 tests en verde** (+15), 12 pantallas Playwright con el
> backend **reiniciado** (estaba stale de ayer), el camino de holdout probado **por HTTP en los dos
> sentidos** (200 con otra fuente / 400 `holdout_shares_source` con la misma), y el flujo del README
> **ejecutado de punta a punta con un holdout real** — dos miradas, dos líneas, la segunda marcada
> `from_cache`. Esas dos líneas se dejan en el repo a propósito.
> **Lo que sigue teniendo más valor**: (a) por qué la red no detecta un tercio de las esquinas
> (punto 3); (b) las 7 imágenes que fallan siempre, con Diagnóstico/Predecir; (c) si «la periferia
> no aporta» se sostiene con otra fóvea o con otro dataset. Ninguna necesita regenerar nada.
> **Queda sin hacer de §9**: 9.8 (vectorizar `build_view` — *no hasta que duela*) y 9.9 (F12).
>
> **2026-07-26 — LA MÉTRICA DE TAREA, CABLEADA (Fase 2 de metrica-de-tarea.md).** `paragraph_f1`
> ya no es una función que no llama nadie: hay **módulo nuevo `fv.task`** (contrato **⑬ E×A vía
> F**, escrito en organizacion.md §2) que puntúa un run **por imagen** contra los párrafos de la
> **fuente** — B guarda las imágenes pero no los párrafos verdaderos, así que la costura es
> `manifest["source_id"]` y sin fuente se falla (`task_needs_source`), nunca se puntúa contra las
> etiquetas de ventana. Es **caché, no entidad** (como el diagnóstico E×B), con los **knobs de F
> dentro de la clave** (cambiarlos obliga a re-inferir; el `threshold` del diagnóstico no entra en
> la suya porque allí solo se re-leen scores). Se puntúa **`best.pt`**, el fichero que sobrevive.
> Superficies: `GET /runs/{name}/task-score`, bloque en el detalle de un run, botón «medir la
> tarea del ganador» en el veredicto de Recorridos (solo sugerido + mejor), y `fv-oat/fv-study
> --task-score` (**apagado por defecto**: un recorrido nocturno no paga inferencia que nadie
> pidió). Todo número sale con `sem` y n de imágenes, y **con el aviso cuando n < 100**.
> **NO entra en `OBJECTIVES` y no se calcula por época** — §2 del doc lo desaconseja con datos.
> ⚠ **La comprobación que vale**: el código nuevo reproduce **exactamente** el número de la Fase 1
> (0,5353 de media en las 5 semillas del ganador de `fast-lr-s0-lr`, 1,9 s). **107 tests en verde**
> (+8: los 7 de §3.9 y el ⑬), 12 pantallas Playwright con los botones nuevos pulsados, los dos
> CLIs de punta a punta bajo cp1252. Se adelantó de la Fase 4 el parámetro `window_dataset` +
> `holdout_shares_source` (puntuar contra otro B); **falta el holdout en sí**.
> **Pendiente de este doc, por orden**: 3b (validar el proxy en un eje de **C** — el más barato y
> el que más puede cambiar el plan), 3 (regenerar el dato: **F11, decisión del usuario**), 4 (el
> holdout). **Las tres están especificadas al detalle** en metrica-de-tarea.md §§4-6 para que otra
> sesión las ejecute en frío: comandos reales con banderas verificadas, costes **medidos** (el
> recorrido de 3b son 30 runs × 140 s ≈ 70 min; los 6 valores de `d` pasan `check_run`), la pieza
> que falta (`spearman` en `fv.metrics` — **no hay scipy**, con números dorados para su test), y
> **todo lo que habría que tocar** si el proxy no valiera para C (§5.5: meter `paragraph_f1` en
> `OBJECTIVES` **no basta** — `sweep_trials` lee el `val` por época, que no tiene esa clave, y el
> ranking se quedaría en `None` sin avisar). Además, §9 lista **10 pruebas que valdría la pena
> hacer**, cuatro de ellas **sin entrenar nada** (el techo de la reconstrucción con esquinas
> perfectas; barrer los knobs de F, que nadie ha mirado nunca; la curva F1-vs-IoU; re-medir la sd).
> ⚠ **Y una corrección que manda sobre la Fase 3**: el dato real sale del **generador hermano +
> un resize que este repo NO tiene portado** — `make_synth_source.py` hace otro problema (barras de
> juguete). Nuevas decisiones registradas: **F13** (¿portar el resize?) y **F14** (¿registrar que
> el holdout se tocó?).
>
> **2026-07-26 — CÓMO SE ELIGE UN GANADOR: CUATRO ARREGLOS ENCADENADOS.** El usuario reportó que
> «el gráfico se detiene antes de terminar las épocas». No era el entrenamiento (en disco no
> faltaba ni una época: `patience: 0`, `stopped_early: false` en los 134 runs) sino la UI — y al
> tirar del hilo aparecieron tres cosas que sí decidían el ganador:
> 1. **La curva de un run terminal se congelaba** (`Sweeps.tsx`): se cacheaba en cuanto el estado
>    leía `done` Y había algo cacheado, pero ese algo venía del sondeo ANTERIOR, tomado mientras
>    el run entrenaba. Se perdían las épocas entrenadas en esa ventana (3 s normalmente; hasta un
>    minuto con la pestaña de fondo, que el navegador estrangula; más tras hibernar). Ahora un run
>    solo se «settle» al traerlo CON el estado ya terminal. **Verificado en vivo con control**:
>    4 runs × 6 épocas mirados sin recargar → 6/6 con el arreglo, **5/6 con el guard viejo**.
> 2. **El ranking medía la última época, no el checkpoint.** `best.pt` se elige por `monitor` y es
>    lo que cargan Diagnóstico/Predecir y lo que arrastra un estudio; el ranking usaba `m[-1]`.
>    Eran épocas distintas en el **63%** de los runs de `fast-lr-2-s0-lr`. Ahora `sweep_trials`
>    puntúa la época que guardó `best.pt` (`fv.metrics.checkpoint_record`, **la misma regla que usa
>    el bucle** — verificada contra el `best_epoch` de los 134 runs, 134/134). Sin checkpoint (el
>    monitor nunca midió) → `value: null` + razón, nunca la última época de consuelo. **Cambia el
>    ganador** de `fast-lr-2-s0-lr` (lr 0.0014 → 0.00168).
> 3. **No había regla de empate.** `select_winner` cogía `scored[0]` aunque `aggregate_seeds` ya
>    calculara la banda. protocolo.md §1.5 dice lo contrario. Ahora `δ` por defecto = 1-SE de las
>    semillas del mejor punto (`tie_delta`), con `tie`/`tie_reason` en palabras. Veredicto sobre
>    los recorridos reales: `batch_size` gana de verdad; `fast-lr-s0-lr` empata sus dos primeros;
>    **`fast-lr-2-s0-lr` empata los SEIS** — 30 runs que no distinguen nada. ⚠ `fv-study --delta`
>    tenía default `0.0`, que habría pisado la regla justo en el camino desatendido: ahora es None.
> 4. **La banda del gráfico se estrechaba sola**: promediaba las réplicas presentes en cada época,
>    así que un grupo con réplicas a distinta altura fingía converger al final. Se corta donde
>    faltan réplicas y se dice dónde.
>
> **2026-07-26 — LA MÉTRICA DE TAREA: EL PROXY VALIDADO (PARA D) Y EL RESTO ESPECIFICADO.**
> `paragraph_f1` existía desde el día 1 sin que la llamara nadie. Se ejecutó el **paso obligado**
> de protocolo.md §2 usando **runs ya entrenados** (cero entrenamiento, 40 s de inferencia):
> Spearman ventana↔tarea **+0,736** por run y **+0,956** agregado, sobre los 65 de `fast-lr-s0-lr`,
> con el **mismo ganador**. ⇒ **`OBJECTIVES` no cambia**: el proxy barato ordena igual que el caro.
> ⚠ Medido solo sobre un eje de **D**; para ejes de **C** (que cambian la vista) sigue sin medirse.
> El mismo trabajo destapó el límite real: sd entre imágenes 0,372 sobre **20 imágenes de val** →
> **±0,083** por run, más ruido que las diferencias a distinguir. **El bloqueo no es código, es el
> tamaño del val.** Todo lo pendiente —cablear `task_score` (módulo nuevo `fv.task`, contrato ⑬),
> validar el proxy en el eje `d`, dimensionar el dato, el holdout— está especificado al detalle en
> **[docs/metrica-de-tarea.md](docs/metrica-de-tarea.md)**, con firmas, payloads, claves de caché,
> tests y costes medidos. Dos decisiones quedaron **abiertas y son del usuario**: **F11** (regenerar
> el dato invalida la comparabilidad con los 130 runs actuales) y **F12** (qué es la «CNN plana
> equivalente» del primer experimento, que hoy no se puede construir por `no_periphery`).
>
> **2026-07-26 — EL PATRÓN DETRÁS DE CASI TODOS LOS FALLOS: «el mismo dato en dos sitios».**
> A petición del usuario se analizó el historial de arreglos y sale un modo de fallo dominante:
> un hecho representado dos veces (escritor↔lector, fuente↔caché, productor↔consumidor,
> puerta↔puerta, cabecera↔celda) donde solo una copia se actualizó. Tres propiedades lo hacen
> caro: **el conflicto se resuelve por precedencia y no por detección** (nadie lanza), **el
> resultado parece correcto** (un número plausible, no un crash), y **lo encuentra el usuario, no
> los tests** (un test unitario prueba UN lado de la costura). Se concentra en fronteras **que
> nadie numeró** — los contratos ①–⑫ protegen bien lo que cubren.
> Barrido a fondo: se eliminaron las **cuatro copias vivas** que el front tenía (defaults de C,
> ejes de geometría, objetivos, orden de esquinas); **dos ya habían divergido**. Ahora el API
> sirve cada vocabulario desde su única definición (`/networks` → `full_config({})`, `/sweeps/axes`
> → objetivos + `loss_weight_params` + `window_size_fields`, y `corner_order` viaja en todo payload
> indexado por él). Al servir los objetivos apareció un **hueco real**: `validate_plan` era **más
> laxa** que `check_sweep` para el contrato ⑨ (objetivo `loss` + eje de peso de la pérdida) — el
> `<select>` de Estudios lo tapaba no ofreciendo `loss`. Cerrado en la puerta. También la columna
> «siguiente» de Estudios decía el dataset porque `next_axis` solo existía en el detalle: ahora
> ambos salen de `fv.studies.driver.summarize`. **99 tests en verde** (+4 de costura).
> ⚠ **Regla de trabajo que se deriva de esto:** antes de cambiar la forma o el significado de un
> campo compartido, buscar **todos** sus lectores; y todo dato derivado, una definición y dos
> lectores — nunca dos definiciones.
>
> **Regresión propia detectada por el usuario y arreglada el mismo día:** cambiar `studies.delta`
> de número a cadena dejaba **Estudios en blanco** al abrir un estudio con un paso sin confirmar
> (el 0 recordado del navegador llegaba a `delta.trim()`). Dos capas: `usePersistedState` **rechaza
> un valor recordado cuyo tipo no encaje con el default** (deriva de esquema, no preferencia — mata
> toda la familia), y hay **error boundary por ruta**: una pantalla que revienta ya no borra la app,
> muestra la razón y ofrece olvidar las preferencias. `verify_ui.py` siembra el valor viejo y hace
> click en TODOS los estudios. **Lección: una página en blanco es el fallo silencioso definitivo —
> verificar en la UI, no solo con tests.**
>
> UI nueva: columnas «época»/«última» en el ranking, aviso cuando `monitor != objective` (el caso
> de los tres recorridos vivos), y componente `WinnerVerdict` — **Recorridos estrena veredicto**,
> antes esa pantalla no mostraba ganador ninguno. Los CLIs (`fv-oat`, `fv-sweep`, `fv-study`) lo
> imprimen también; `tie_reason` es ASCII a propósito (llevaba una δ griega, **que no existe en
> cp1252** y habría matado un estudio nocturno en la última línea — reproducido y arreglado).
> **95 tests en verde** (+7), 12 pantallas Playwright sin errores, `fv-oat` y `fv-study --auto`
> ejecutados de punta a punta (también bajo `PYTHONIOENCODING=cp1252`).
>
> **2026-07-25 — BORRAR UN ESTUDIO ARRASTRA SUS RECORRIDOS (antes los dejaba huérfanos).**
> `StudyStore.delete` borraba solo el estudio (plan+progress), no los recorridos que generó
> (`{estudio}-s{i}-{eje}`). Al borrar y **recrear un estudio con el mismo nombre**, el recorrido de
> la versión anterior quedaba huérfano y el siguiente `advance` chocaba con `sweep_exists` (nunca
> sobrescribe — correcto) sin salida. Ahora `fv.studies.driver.delete_study` **arrastra** a los
> recorridos del estudio (por `spec.study`, vía `SweepStore.used_by_study`) y estos a sus runs
> (reusa `delete_sweep`); rechaza ANTES de borrar nada si algún recorrido/run está `queued`/`running`
> (`study_has_live_sweeps`, 409 — R4). El endpoint `DELETE /studies/{name}` usa el arrastre. **87
> tests en verde** (+2 de contrato: ciclo borrar→recrear→avanzar sin colisión; guarda de vivo).
> ⚠ Espejo del arrastre recorrido→runs; simétrico con la auditoría CRUD del 2026-07-25.
>
> **2026-07-25 — SEMILLAS DEL ESTUDIO: N SEMILLAS EN CADA PUNTO (antes se ignoraban).** El `seeds`
> del plan de un estudio se **validaba y guardaba pero nunca generaba runs**: cada punto del eje se
> entrenaba con una sola semilla (la del recipe base), pidieras 1, 3 o 5. Por decisión del usuario
> se implementó **N semillas en cada punto**: el generador (`fv.sweeps.generate.build_generated_spec`)
> añade un **segundo eje `seed=[s0..s0+N-1]`** junto al eje real cuando `seeds>1`, así el barrido
> produce `len(rango)·N` runs (uno por `(valor, seed)`) y la banda existe en todo el eje. Cableado:
> `studies.advance` pasa `plan["seeds"]`; también `fv-oat --seeds` y `POST /sweeps/generate {seeds}`.
> El nombre del run lleva la semilla (`…-d2_seed1 .. _seed3`, `point_run_name` seed-aware). El
> **ganador agrega por valor de eje** (`fv.sweeps.winner.aggregate_seeds`): rankea la **media** de
> las semillas + banda min–max + `n_seeds`, nunca la réplica con suerte (recipe.py: seed es el eje
> RÉPLICA). `seeds=1` = sondeo rápido de antes (sin eje de semilla). Divergencia registrada en
> [barrido-por-ejes.md](docs/barrido-por-ejes.md) §11.1: el esquema D-M1 «confirmación N-en-frontera»
> queda como modo alternativo NO construido. **85 tests en verde** (+5 de contrato); verificado en
> vivo con `fv-oat --seeds 2` sobre `synth-b16` (2 runs, seeds 1/2 distintas en disco, ganador
> agregado). ⚠ **El estudio `batchSize_fast-80px-5seeds` que ya corría se generó ANTES del arreglo:
> su recorrido no tiene eje de semilla (sigue en 5 runs, seed=1). Para obtener las 5 semillas hay
> que borrarlo y recrearlo.**
>
> **2026-07-25 — CURVAS DEL RECORRIDO: OVERLAY MULTI-RUN EN RECORRIDOS.** Al elegir un recorrido,
> la pantalla **Recorridos** superpone las curvas de val (loss / f1 / pos_err_px) de **todos sus
> runs** en tres small-multiples (la misma vista V14 de RunDetail, pero N líneas). Dos modos, por
> decisión del usuario: **líneas por run** (una por run, leyenda de casillas para ocultar/mostrar
> cada uno o todos, énfasis al pasar el ratón) y **media ± banda** (agrupa runs que comparten
> config salvo `seed`: línea media + sombra min–max; con 1 semilla la banda es degenerada, a
> propósito y anunciado en el subtítulo). El color **sigue a la entidad** (índice de trial / orden
> de grupo), nunca al rank, así ocultar/ordenar no repinta a los demás. La paleta de 8 hues vive en
> `tokens.css` (`--series-1..8`, claro/oscuro), validada CVD para las dos superficies; identidad
> siempre por leyenda+etiqueta, nunca por color solo. Los datos ya existían: `/sweeps/{n}/trials` +
> `/runs/{r}/metrics` (fan-out en el poll de 3 s; runs terminales se traen una vez). **Cero rutas
> nuevas de backend.** Verificado: `web` typecheck+build limpio; Playwright sobre el recorrido vivo
> `batchSize_fast-80px-s0-batch_size` (5 runs) — ambos modos, toggle y ocultar/mostrar todo, **sin
> errores de consola**. `LineChart` extendido (banda + serie atenuada + leyenda opcional) sin
> romper RunDetail. `SweepCurves.tsx` es nuevo. ~~⚠ `scripts\verify_ui.py` sigue apuntando a
> `test4-s0-n_layers` (borrado)~~ — **RESUELTO 2026-07-26**: reapuntado a `fast-lr-2-s0-lr` y
> ampliado (época, aviso de monitor, veredicto, los dos modos de curva). El «runs terminales se
> traen una vez» de arriba era el bug del gráfico congelado: ver la nota del 2026-07-26.
>
> **2026-07-25 — AUDITORÍA CRUD CROSS-PÁGINA + REFRESCO DE LISTAS EN VIVO.** Se revisó el CRUD de
> las 12 pantallas buscando «borrar aquí rompe allá». Arreglos:
> - **Diagnóstico/Predecir** ya no piden un run recordado en `localStorage` que fue borrado o
>   renombrado (la migración de sufijo de eje dejó nombres huérfanos): **gatean por pertenencia a la
>   lista, no por verdad**, y la doomed-request ya no revienta ni pisa la carga válida. Además
>   **refrescan la lista de runs en vivo** (sondeo 3 s, como Runs/Recorridos/Estudios), gateando la
>   carga pesada con un booleano `runReady`/`sourceReady` estable para no re-computar en cada pasada.
> - **Borrado cruzado por-nombre (R4):** borrar un **dataset B** que fija un **recorrido**
>   (`spec.window_dataset`) o un **estudio** (`plan.window_dataset`), o una **receta D** que fija un
>   estudio (`plan.base_recipe`), devolvía 200 y rompía la otra pantalla *dentro del job* al
>   reanudar/avanzar. Ahora se rechaza en la puerta con **409 + razón+arreglo** nombrando al referente
>   (`SweepStore/StudyStore.used_by_dataset`, `StudyStore.used_by_recipe`; código `recipe_in_use`). La
>   asimetría es deliberada: las referencias **instantánea** (run/recorrido copian los VALORES de C/D
>   inline) SÍ permiten borrar C/D sin romper el run ni su diagnóstico.
> - **Verificado:** `tests/test_crud_integration.py` fija el grafo (instantánea no obliga, por-nombre
>   sí; borrar el recorrido de un estudio lo deja legible). **80 tests en verde**; guards probados en
>   vivo por HTTP; ciclo crear→borrar y refusal-en-UI por Playwright, 12 rutas sin errores de consola.
> - **Artefactos:** por decisión del usuario **no se borró nada**. Todos los runs/recorridos/estudios
>   actuales **cargan** — la nota de §13 sobre checkpoints incompatibles (`fov-run-*`, `cli-run-1`)
>   quedó **OBSOLETA**: hoy cargan y se diagnostican/predicen sin error.
>
> **2026-07-24 — EJES DE BARRIDO: `N`/`c_frac` REHUSADOS + BUDGET NO COLAPSA `epochs` +
> VERIFICADOR DE TODOS LOS EJES.** Un recorrido generado por un estudio (`test2-s0-N`) quedaba en
> `done 0/3` sin razón visible: barría el eje `N` con `c_frac` fijo, y como `center_out =
> round_to_even(N·c_frac)` está atado por ①a al `window_size` del dataset, **cada punto** daba una
> fóvea != la ventana. `expand_points` solo validaba geometría (`check_network`), no la costura con
> el dataset (`check_run`), así que los 3 puntos pasaban como "válidos" y morían con
> `window_size_mismatch` dentro del job (`RunError`→`continue`) — la trampa que R4 prohíbe. Arreglos:
> - **`N` y `c_frac` no son ejes** (`WINDOW_SIZE_FIELDS`): se rechazan en las **dos puertas** —
>   `check_sweep` (H) y `validate_plan` (I)— con `axis_breaks_window_size` (razón + arreglo: barre
>   `d`, o usa un dataset con esa ventana), antes de reservar nada. Como ningún otro eje toca
>   `center_out`, `check_network` por punto sigue bastando.
> - **Budget no colapsa `epochs`:** el runner solo aplica `budget.epochs` si el punto no barre
>   `epochs` (antes lo pisaba siempre → el eje no hacía nada, en silencio).
> - **`scripts\verify_axes.py`:** corre un recorrido real por CADA eje de C y D. Verificado hoy:
>   **26/26 ejes** (11 red + 13 receta + `N`/`c_frac` rehusados), 0 fallos. Ejes probados listados
>   en el README (§«Qué ejes se pueden barrer»).
>
> **74 tests en verde**. Artefactos muertos del usuario borrados (`studies/test2`,
> `sweeps/test2-s0-N`, `sweeps/test1-s0-N`). Fresco y cargable sigue siendo `fov-16-param` +
> `oat-d-demo`.
>
> **2026-07-24 — PARADA DE RECORRIDOS: CORTE EN VUELO + RECONCILIACIÓN DE ESTADO MUERTO.** Dos
> arreglos sobre la parada de recorridos (H):
> - **Corte del punto en vuelo (feat 1):** `train` acepta `should_stop`; el runner le pasa
>   `lambda: store.stop_requested(name)`, así una parada pedida al recorrido corta el punto **en
>   marcha** en su siguiente frontera de epoch, no solo entre puntos. La cooperación seguía siendo
>   entre puntos: un run largo ignoraba la parada hasta acabar.
> - **Reconciliación de `running` muerto (feat 2):** quien marca un recorrido/run `running` graba
>   su **PID dueño** (`fv.proc.pid_alive`, portable Win/POSIX sin psutil). `SweepStore.reconcile`
>   y `RunStore.reconcile` sanan `running`→`interrupted` cuando el proceso dueño ya no existe
>   (caída/reinicio del API/hibernación) — se llama al leer (`GET /sweeps`, `GET /sweeps/{name}`,
>   `sweep_trials`). Cierra la trampa heredada "un crash queda running para siempre". Erra seguro:
>   dueño vivo o sin PID (legacy) → intacto. El runner ahora redó todo lo no-(done|cancelled), así
>   que un punto `interrupted` se rehace al reanudar. `interrupted` es terminal: borrable y
>   reanudable; badge ámbar en la UI. **69 tests en verde**, 12 pantallas Playwright sin errores,
>   reconciliación verificada por HTTP. Causa raíz de `test1-s0-n_layers` pegado en running: su job
>   murió y nadie leía el `stop.json`.
>
> **2026-07-24 — BARRIDO POR EJES (OAT) IMPLEMENTADO Y VERIFICADO.** Se construyeron las cinco
> piezas de [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) §14, respetando las decisiones
> cerradas de su §13:
> - **Builder paramétrico (C)**: `fv.models.builder` honra `n_layers` y `channels` por capa
>   (D-C3), stride solo en la 1ª capa (D-S1); no-regresión bit-exacta para `n_layers=2` con
>   `channels=[16,32]`. Lee `ch1/ch2` viejo, escribe siempre `channels`.
> - **Derivador de base (G/C)**: `fv.models.derive` — de `window_size` deriva `N`/geometría
>   (①a), defaults estáticos, ganadores arrastrados, corrección de inválidos con razón, `N` mínimo
>   (D-G2), afloje de `c_frac` con razón (D-G3).
> - **Base inline + generador P1 (H)**: `fv.sweeps.generate` + CLI **`fv-oat`** + `POST
>   /sweeps/generate`. `base_network=null` + `base_label` + `derivation` (D-H2). Barrer `n_layers`
>   redimensiona `channels` a `[16]*L` (§6.1).
> - **Arrastre del ganador (I/H)**: `fv.sweeps.winner` — regla coste/calidad con δ (D-W1),
>   sugiere y el usuario confirma; `GET /sweeps/{name}/winner`.
> - **Estudio OAT (I, dominio nuevo `studies/`)**: `fv.studies` (plan.json comiteable +
>   progress.json vivo), CLI **`fv-study`**, endpoints `/studies/*`, pantalla **Estudios**. Guía
>   y no ejecuta; expande `channels[i]` al fijar `n_layers`.
>
> **65 tests en verde** (~25 s) incluyendo el contrato ⑫ (estudio↔recorrido) y **las 12
> pantallas con Playwright sin un solo error de consola** (`scripts\verify_ui.py`). README con
> comandos **ejecutados** (`fv-oat`, `fv-study` verificados de punta a punta).
>
> ⚠ **Consecuencia de §13 (deuda de pesos = 0):** el builder paramétrico renombró los módulos
> conv (`center_conv1` → `center_convs.0`), así que **los checkpoints previos ya no cargan**.
> `load_model` lo rechaza limpio (`checkpoint_incompatible`, 400) en vez de un 500. Los runs de
> ejemplo antiguos (`fov-run-*`, `cli-*`, `dirty-*`, `rec-d`, `demo-seeds`) quedan **no cargables
> por diseño**: reentrénalos o descártalos. Fresco y cargable: run **`fov-16-param`** + recorrido
> inline **`oat-d-demo`** (base `ws16-p2-d2-L2`). Diagnóstico/Predecir usan `fov-16-param`.
>
> ---
>
> **2026-07-21 — IMPLEMENTACIÓN BASE COMPLETA Y VERIFICADA.** El sistema entero está construido y
> probado de punta a punta en esta máquina: paquete `fv` (fovea/datasets/windows/models/
> training/inference/diagnostics/sweeps/studies/validation/metrics/matrixview), API FastAPI, front
> Vite+React con las **diez pantallas**, CLIs (`fv-extract`, `fv-train`, `fv-sweep`, `fv-oat`,
> `fv-study`, `fv-api`) — un test por contrato más el muestreo foveado
> contra los números de la spec. Verificado además: el flujo completo por HTTP (extract →
> train → diagnóstico → predict → sweep), los CLIs (bit-idénticos al API con la misma
> semilla), las negativas con razón+arreglo antes de reservar nombre. El README lleva los comandos
> **ejecutados**, no razonados.
>
> **Decisiones cerradas en la implementación** (registradas en decisiones.md §4): F1=C9
> (cabezas de esquina), **F1b=C10: las esquinas se etiquetan SOLO sobre la fóvea**
> (`center_out == window_size` de B, contrato ①a; la periferia es contexto), C11 (relleno
> `pad_mode: edge` + máscara de cobertura solo para depurar — F0), C12 (anillo por pooling
> anisótropo por zonas, co-registrado: el código §5 de la spec no tipa para d>1, ver
> decisiones.md).
>
> **Datos de ejemplo vivos en el repo**: fuente `local/synth-01` (60 img 96×72, regenerable
> con `scripts\make_synth_source.py`), dataset `synth-b16`, red `fov-16` (migrada a `channels:
> [16,32]`), recetas `corta`/`media`. **Cargables con el builder actual**: run `fov-16-param` y
> recorrido inline `oat-d-demo`. Los runs/recorridos anteriores (`fov-run-*`, `cli-*`, `dirty-*`,
> `rec-d`, `demo-seeds`) conservan su metadata comiteada pero **sus checkpoints no cargan** (§13):
> son historia, no artefactos vivos.
>
> **Pendiente, por orden de valor**: (1) el primer experimento real (protocolo.md §6:
> ¿fóvea+periferia gana a una CNN plana de coste equivalente? — control con `d=1`/`c_frac`→1
> o red plana equivalente, N semillas, criterio escrito antes); (2) el holdout y el dato de
> verdad (fuente del generador reducida con resize — el resize aún NO está portado, decisión
> al llegar); (3) V16/occlusion pre-muestreo (diseño en [docs/ui/2-vistas.md](docs/ui/2-vistas.md)); (4) poda en el runner de
> recorridos (hoy corre todos los puntos); (5) pantalla Entrenar: el estimador solo usa runs
> comparables (hecho) pero no hay curva de coste por punto del recorrido.
>
> **Servidores dev**: ~~al cerrar esta sesión quedaron corriendo backend (:8010) y vite (:5173)~~
> — **OBSOLETO (2026-07-27)**: se pararon, y la regla ahora es cerrarlos siempre al terminar
> (ver «Convenciones»).
>
> Nada de lo documentado está construido ni verificado. Cuando un documento cita código
> (`loop.py:166`, `extract.py:127`), habla del **proyecto hermano** — es la evidencia que motivó
> el diseño, no código de este repo.

**Al terminar una fase, actualiza estas líneas.** Es lo único que le dice a la siguiente sesión
dónde está.

---

## Regla permanente: la organización por dominios manda

**[docs/organizacion.md](docs/organizacion.md) es la fuente de verdad sobre cómo se organiza
este sistema. Léelo antes de cualquier cambio y respeta sus fronteras.** Aplica a todo cambio,
por pequeño que parezca — un campo nuevo en un config es exactamente donde las fronteras se
rompen.

Los demás documentos, en orden de lectura:

| | |
|---|---|
| [instructionsNewNN.md](instructionsNewNN.md) | **La red.** Geometría foveada, parámetros, rangos calculados, código de referencia |
| [docs/organizacion.md](docs/organizacion.md) | **La raíz.** Dominios (A–I, X, G) y contratos ①–⑫ donde se tocan |
| [docs/herencia.md](docs/herencia.md) | Qué viene de `image-text-finder`, qué se adapta y qué se descarta |
| [docs/protocolo.md](docs/protocolo.md) | Cuándo un resultado es creíble. **Léelo antes de sacar conclusiones de un entrenamiento** |
| [docs/api.md](docs/api.md) · [docs/ui.md](docs/ui.md) | La organización proyectada sobre HTTP y sobre pantallas. **`ui.md` es el índice**: las reglas de UI viven en `docs/ui/`, una por **tipo de especificación** — [1 estructura](docs/ui/1-estructura.md) · [2 vistas](docs/ui/2-vistas.md) · [3 representación](docs/ui/3-representacion.md) · [4 datos](docs/ui/4-datos.md) · [5 invariantes](docs/ui/5-invariantes.md) · [6 números](docs/ui/6-numeros.md) · [7 operación](docs/ui/7-operacion.md) · [8 léxico](docs/ui/8-lexico.md) |
| [docs/plan.md](docs/plan.md) | El plan de ejecución, por fases verticales |
| [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) | **IMPLEMENTADO (2026-07-24).** Barrido OAT (un eje a la vez) con base derivada del problema, defaults estáticos, arrastre del ganador y estudio (dominio I). Ver `fv.models.derive`, `fv.sweeps.generate/winner`, `fv.studies`, CLIs `fv-oat`/`fv-study` |
| [docs/metrica-de-tarea.md](docs/metrica-de-tarea.md) | **FASES 1, 2 y 3b HECHAS (2026-07-26); la 3 aplazada (F11), la 4 con el código hecho.** La métrica que manda (párrafo por imagen): el proxy de ventana ordena igual en ejes de **D** (+0,956) **y de C** (+1,000, §2 bis) → `OBJECTIVES` no cambia. `task_score` cableada (`fv.task`, contrato ⑬) con registro de holdout. 8 de las 10 pruebas de §9, medidas. **Léelo antes de tocar métricas de ranking** |
| [docs/formatos.md](docs/formatos.md) · [docs/tests.md](docs/tests.md) | Los artefactos en disco; qué se testea |
| [docs/decisiones.md](docs/decisiones.md) | Lo que sigue sin decidir, y qué bloquea. **No tomes tú una decisión que esté ahí: pregunta** |
| [docs/glosario.md](docs/glosario.md) | Las palabras que significan dos cosas |

Reglas que estos documentos fijan y que se citan aquí porque se incumplen solas:

- **Ausente ≠ cero** (formatos.md §2): un lector que necesita un campo ausente **falla con la
  razón**; nunca lo inventa ni lo rellena.
- **Toda restricción se valida antes, con razón y arreglo** (api.md R4): un `400` al entrar vale
  mil veces más que un stack trace dentro del hilo del job media hora después.
- **Toda puerta que entrene pregunta al mismo validador** antes de reservar el nombre. Dos
  comprobaciones separadas se desincronizan, y la puerta más laxa es por la que entra un
  recorrido automático.
- **Un run no se sobrescribe jamás** (409 con la razón).
- **Un contrato sin test es un comentario** (tests.md): los contratos van a
  `tests/test_contracts.py`, los no implementados en `xfail(strict=True)`.
- **Un resultado sin N semillas es una anécdota** (protocolo.md).

### Los dominios (resumen; el detalle está en organizacion.md)

| | Dominio | Es | Vive en |
|---|---|---|---|
| **A** | Fuente | Imágenes + geometría de párrafos (proyecto externo, solo-lectura) | `src/fv/datasets/` |
| **B** | Dataset de ventanas | Lo que se etiqueta: imágenes completas + etiquetas por ventana. **La vista foveada NO se hornea aquí: se construye en el dataloader** | `src/fv/windows/`, `data/window-datasets/` |
| **C** | Red foveada | `N`, fracciones, kernels/strides por rama, fusión. Config puro, cero datos | `src/fv/models/`, `configs/networks/` |
| **D** | Receta | Hiperparámetros de entrenamiento que definen el resultado | `src/fv/training/`, `configs/recipes/` |
| **E** | Run | Modelo entrenado: pesos + métricas + procedencia | `runs/<name>/` |
| **H** | Recorrido | Un espacio sobre **C y/o D** con B fijo → muchos E, sin intervención humana | `src/fv/sweeps/`, `sweeps/` |
| **I** | Estudio (OAT) | Un plan ordenado de ejes sobre **H** con B fijo → muchos recorridos; guía, **no ejecuta** | `src/fv/studies/`, `studies/` |
| **F** | Inferencia | Aplicar un E a una imagen completa (ventana foveada deslizante) | `src/fv/inference/` |
| **G** | Geometría foveada | `derive_dims`, `build_foveated_input`, `build_masks`, rangos calculados. **Un solo módulo, todos lo importan** | `src/fv/fovea/` |
| **X** | Ejecución | `device`, `num_workers`, concurrencia. **Cuesta tiempo, no cambia el resultado** | `src/fv/api/jobs.py` |

### Antes de tocar nada, pregúntate a qué dominio pertenece

El criterio, en orden:

1. ¿Cambia **la forma del modelo o de su entrada**? → **C** (`N`, `c_frac`, `d`, `pen_frac`,
   kernels, strides, `merge`, `pool_mode`, `dropout` son C — *incluida la geometría del muestreo
   foveado*, aunque suene a datos: es la red quien define qué vista consume).
2. ¿Cambia **los pesos resultantes** sin cambiar la forma? → **D** (`lr`, `batch_size`, pesos de
   la pérdida).
3. ¿Solo cambia **cuánto tarda**? → **X**. Nunca dentro de la identidad de D. **`batch_size` es
   D, no X** — subirlo al pasar a GPU invalida la comparación con lo entrenado en CPU (contrato ⑩).
4. ¿Se ajusta **sin reentrenar**, sobre un modelo ya hecho? → **F** (`threshold`, stride de
   inferencia, NMS). Barrer esto no cuesta horas; no lo metas en D.

Si un cambio necesita tocar dos dominios, eso es un **contrato**: está numerado en
organizacion.md §2. Respétalo explícitamente o actualiza el doc.

---

## Contexto de trabajo

- **Hoy solo CPU (esta máquina). Habrá un server con GPU** para los recorridos largos. Por eso X
  está separado de D **desde el diseño**: si no, lo entrenado en CPU queda incomparable con lo
  de GPU. Y por eso `environment` (python/torch/plataforma/device) va en la procedencia de cada
  run.
- **El flujo objetivo son recorridos secuenciales desatendidos**: una receta de recorrido (H)
  nombra el espacio y el presupuesto, se lanza, y corre puntos de uno en uno guardando runs de
  primera clase. Primero versiones cortas aquí (pocas épocas, dataset pequeño) para validar el
  instrumento; el mismo spec, con más presupuesto, en la GPU.
- En CPU, **el límite de workers concurrentes es 1**: torch ya usa todos los núcleos. En GPU se
  reevalúa (es X: no cambia resultados).
- **El espacio de geometría foveada es pequeño y discreto por construcción** (los rangos los
  calculan las funciones de instructionsNewNN.md §3: con N=20, ~3·2·2·1·varios puntos) →
  **grid exhaustivo**. Optuna se reserva para lo continuo: `lr`, canales, dropout
  (instructionsNewNN.md §9).

## Convenciones

- **Idioma**: el usuario se comunica en español; documentación de alto nivel en español. El
  código (identificadores, docstrings) en inglés.
- **Commits**: cada tarea terminada acaba en un commit descriptivo. Además, **cada cambio
  solicitado por el usuario, una vez completado, se cierra con su propio commit descriptivo.**
- **Stack**: Python 3.12 (PyTorch no tiene wheels para 3.14) + PyTorch + FastAPI + Vite/React.
  En Windows el intérprete será `.\.venv\Scripts\python.exe`. Paquete `fv`, layout `src/`.
- **Tests**: `.\.venv\Scripts\python -m pytest -q` desde la raíz, antes de commitear código.
- **README verificado**: antes de decir que un comando documentado funciona, **ejecútalo** en
  PowerShell tal como está escrito (regla global del usuario). Nunca presentar una instrucción
  no probada como verificada.
- **Probar ejecutando, sí; dejarlo corriendo, no.** Lanzar procesos del proyecto para verificar
  (backend `fv-api`, `npm run dev`, entrenamientos, Playwright) está **siempre permitido** y no
  hace falta pedirlo. Pero **al terminar la tarea se cierran todos** y se comprueba que los
  puertos (`:8010`, `:5173`) quedan libres: el usuario prueba a mano después, y un server viejo
  vivo le ocupa el puerto o le contesta con rutas obsoletas. Matar hijos antes que padres (vite
  antes de `npm run dev`; el `fv-api` que escucha antes de su lanzador) y filtrar por ruta —
  en esta máquina hay pythons ajenos al proyecto.
- `data/`, `runs/` y `sweeps/` son artefactos: **se versiona la descripción (configs, métricas,
  manifests, specs), se ignora la carga (`.npz`, `.pt`, `optuna.db`)** — formatos.md §5.
- **Enlaces a ficheros en las respuestas**: siempre en formato markdown `[texto](ruta)` con la
  ruta **relativa a la raíz del workspace** (nunca backticks ni ruta pelada), para que sean
  clickeables en la extensión de VSCode. **No envuelvas el enlace entre paréntesis** ni pegues
  puntuación al `)` de cierre: `(... [x](ruta) ...)` rompe la detección del enlace y deja de ser
  clickeable. Déjalo suelto o sepáralo con `—`, dos puntos, o una coma con espacio.
  **Los enlaces solo abren ficheros de texto (código fuente), no imágenes** — verificado
  2026-07-23: un `.png` no abre al clicar aunque esté rastreado por git (no es el git-ignore, es
  el tipo binario). Para una imagen (capturas de `data/ui-shots/`, etc.) NO ofrezcas un link
  markdown que no abre: da la ruta para `Ctrl+P`/Go-to-File, o muéstrala inline con la tool Read.

## Observaciones de esta máquina (medidas en el proyecto hermano — no re-aprenderlas)

- **La máquina HIBERNA en entrenamientos nocturnos largos**: suspende el proceso. Para un
  recorrido desatendido, desactivar la suspensión (`powercfg`) o contar con que se pausa.
- **Throttling térmico**: en carga sostenida los runs se ralentizan ~5×. Los presupuestos de un
  recorrido nocturno deben contarlo.
- **La consola de Windows es cp1252**: los CLIs imprimen ASCII (un `→` en un `--help` revienta
  con `UnicodeEncodeError`).
- **Los JSON de estado se escriben con temporal + `os.replace`, y en Windows con reintento en
  los dos lados** (escritor y lector): Windows no reemplaza un fichero con un handle abierto.
  Detalle en formatos.md §4.2.
- **Hay Playwright y Chromium en esta máquina: la UI SE PUEDE ver.** Los navegadores están en
  `%LOCALAPPDATA%\ms-playwright\`; hace falta `pip install playwright` en el venv del proyecto
  (los navegadores ya están, no hace falta `playwright install`). No entregar UI diciendo «no
  puedo verlo» sin haber mirado.
- **Al verificar UI: reinicia el backend** — un server stale da 404 engañosos sobre rutas nuevas.

## Trampas heredadas: no las reproduzcas

Medidas en `image-text-finder` (lista completa y razonada en
[docs/herencia.md](docs/herencia.md) §4 y organizacion.md §3). **Casi todas eran *defaults***:
nadie las eligió, aparecieron por no elegir. Construir desde cero no protege de ellas — las
invita:

- **SGD sin momentum** si solo pasas `lr` y `weight_decay` → cualquier comparación de
  optimizadores queda sesgada a favor de Adam.
- **Un hilo por job sin límite** → un recorrido de 20 puntos son 20 entrenamientos peleándose
  por los mismos núcleos. En CPU el límite es 1.
- **Sobrescritura silenciosa de runs** (`mkdir(exist_ok=True)` + truncar métricas) — quien la
  pisa es justo un recorrido que autogenera nombres.
- **Un dataset sin val** elige `best.pt` por train loss sin avisar → se niega, no se degrada.
- **Estado de run deducido del disco** → un crash queda «running» para siempre. `status.json`
  explícito.
- **Augmentation con flips/rotaciones sin reetiquetar** → enseña basura en silencio (las
  etiquetas de posición/región son direccionales).
- **Definir un número dos veces** (una métrica calculada en dos sitios) → módulo único
  `fv.metrics`, y un test que afirma la costura, no la función.
- **Lógica de dominio dentro de `app.py`** → si una función no menciona HTTP, no es del API.
- **Medir con un val diminuto**: los ejemplos de una misma imagen están correlacionados; el
  tamaño de muestra efectivo lo dan las **imágenes**, no las ventanas. El dato es sintético:
  generar más es gratis.
- **Optimizar un proxy sin validarlo**: la métrica que manda es la de la tarea real (párrafo
  bien reconocido por imagen), no la de ventana — protocolo.md §2.
