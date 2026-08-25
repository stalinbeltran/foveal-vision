# Decisiones abiertas

Lo que está **sin decidir** y bloquea algo. Índice, no archivo histórico. Una decisión que no se
ve **se acaba tomando sola, por defecto y sin pensar** — así nacieron las trampas del hermano
(CORS abierto, val diminuto, runs sin versionar).

**Ciclo de vida**: al decidirse, la decisión se escribe en el documento que le corresponde y
aquí queda una línea en §4 apuntando allí. **Claude no toma solo una decisión de esta lista:
pregunta.**

---

## 1. Bloquean la fase 0 → 1

### F1 — ✅ CERRADA (2026-07-21): la red conserva las cabezas de esquina

**Decidido por el usuario**: la red predice, por ventana etiquetada, **una cabeza por tipo de
esquina (TL, TR, BR, BL) con `[exists, x, y]`** — la `CornerHead` del hermano sobre el backbone
foveado de dos ramas. La spec de instructionsNewNN.md **no lo impide**: su clasificador
(`num_classes=10`) era un placeholder de referencia; lo que la spec define es el muestreo y el
backbone. Dos consecuencias técnicas, registradas en herencia.md §2:

- El `adaptive_avg_pool2d(feat, 1)` del `forward` de referencia **se sustituye** (un pooling
  global destruye la posición); la cabeza consume el `feat` fusionado aplanado.
- Con la cabeza vuelven **enteros**: la pérdida `BCE + λ·smoothL1` con su catálogo
  (`lambda_pos`, `pos_weight`, `smooth_l1_beta` explícito), el contrato ⑨ en su forma original,
  `corner_evidence`/V18, y la reconstrucción TL→BR como punto de partida de F.
- La progresión párrafos → líneas → palabras entra como `target_kinds` de B (qué bloques de A
  se etiquetan), no como cambio de arquitectura.

Registrada en §4 (C9). **Lo que queda abierto es F1b:**

### F1b — ✅ CERRADA (2026-07-21): las esquinas se etiquetan SOLO sobre la fóvea

**Decidido por el usuario**: la ventana etiquetada es el **centro** (`center_out`); la
periferia está ahí solo como apoyo (contexto, posiblemente útil). Registrada en §4 (C10);
implementada: contrato ①a = `center_out(C) == window_size(B)`, `corner_evidence` congelada
contra la fóvea, `label_window: "center"` en el manifest. *(Lo que sigue es la entrada
original que fijó la pregunta.)*

### (histórico) F1b — ¿centro o campo completo?

**En juego**: la entrada compuesta cubre `original_size = fovea_px + 2·border_px` px de
imagen, pero solo el centro va a resolución completa. Dos opciones:

1. **Ventana etiquetada = el centro (`center_out`)**: la periferia es **contexto**, como el
   `context_40` del P4 del hermano. La posición se predice donde hay resolución completa;
   `corner_evidence` se congela contra el centro; el contrato ①a queda limpio.
2. **Ventana etiquetada = el campo completo (`original_size`)**: más esquinas por ventana, pero
   la posición en la periferia solo resuelve a ±d px (cada píxel compuesto son d reales), la
   escala del `err_px` se mezcla entre zonas, y mover `d` en un recorrido **cambia la población
   etiquetada** — exactamente el error metodológico que el P4 del hermano señaló en su
   baseline barato («mueve la ventana etiquetada ⇒ otra población»).

**Recomiendo la 1**, con fuerza: mantiene comparable el recorrido (la ventana etiquetada no
depende de la geometría barrida), la posición queda a resolución completa, y replica el montaje
que allí ya funcionó (etiqueta fija + campo de visión barrible).

**Dónde vivirá**: organizacion.md §1-B y contrato ①a, formatos.md §4.1, glosario.

### F2 — ¿Qué fuente(s) y a qué resolución arranca el proyecto?

**En juego**: la escala. La geometría foveada necesita margen alrededor de la ventana
(`original_size` hasta 2N); con las fuentes de 80×60 del hermano, un `N=20` con `d=2` ya pide
40×40 de original — cabe, pero condiciona qué `N` son explorables. Decide también el presupuesto
de `images` en B.
**Recomiendo**: la misma familia de fuentes del hermano (`image-text-sample-generator`),
generando el **holdout primero** (protocolo.md §3), y elegir resolución mirando el tamaño
mediano del párrafo respecto a `center_out` (allí midieron 32,4 × 12,9 px — el párrafo no cabía
en 20; aquí esa relación es un eje de diseño, no un accidente).
**Dónde vivirá**: protocolo.md §3, formatos.md §4.1.

## 2. Pueden esperar (se responden al llegar a su fase)

| | Decisión | Recomiendo | Fase |
|---|---|---|---|
| **F3** | `merge: sum` vs `concat` como default (instructionsNewNN.md §7) | `concat` si se barren strides por rama (más libertad); el validador ya rechaza `sum` con strides desiguales | 3 |
| **F4** | `avg_pool2d` vs `max_pool2d` para reducir la periferia (§5 de la spec: trazos finos) | Es un eje a barrer (`pool_mode`), no una constante; default `avg`, medir pronto | 3 |
| **F5** | ¿la geometría del borde es fija por aplicación o barrible? (§11 de la spec) | **CERRADA (2026-08-25)** por la reparametrización (C14): `border_px`, `border_reduce`, `overlap_fovea_px` y `overlap_border_px` son **ejes de primera clase**, con rango calculado o explícito. Antes la pregunta era sobre `c_frac`/`pen_frac`, que no podían ser ejes porque entre `N` y `c_frac` se repartía la fóvea | cerrada |
| **F6** | Política de redondeo/paridad al derivar la fóvea (§11) | **DISUELTA (2026-08-25)**: la fóvea ya no se deriva de nada, se declara (`fovea_px`, par, = `window_size` de B). `round_to_even` sólo sobrevive leyendo la ortografía vieja. Lo que queda de paridad es `border_px % border_reduce == 0`, rechazado con razón | cerrada |
| **F7** | Relleno de bordes de imagen en la vista: ¿valor del relleno + máscara, o solo máscara? | Media enmascarada (el tono del relleno no entra en el número) + máscara de cobertura — la lección medida del hermano | 3 |
| **F8** | ¿Kernels periféricos con forma distinta o sparsity? (§11) | Aplazar: primero medir la forma básica | investigación |
| **F9** | ¿Integración con glimpses secuenciales tipo RAM? (§11) | Aplazar; la arquitectura por ventana no lo bloquea | investigación |
| **F10** | Presupuesto de recorrido en épocas o en segundos (organizacion.md §1-H) | Declararlo en el spec; para espacios sobre C, **segundos** con el coste por punto registrado | 7 |
| **D2** | ¿Extraer las librerías compartidas del hermano (`matrixview`, cola de jobs, registro de runs) a `claude-libs/`, o copiar? | Extraer `matrixview` seguro (cero deps, dos consumidores reales); la cola y el registro, decidir en su fase mirando cuánto divergen | 1/4/5 |
| **F11** | ✅ **CERRADA (2026-07-26): NO se regenera por ahora.** Decidido por el usuario: se sigue con 20 imágenes de val y la métrica de tarea queda como **informe del ganador**, nunca para decidir entre puntos — se conserva la comparabilidad con los 164 runs, 5 recorridos y 4 estudios. Reabrir cuando el ruido estorbe de verdad. *(La pregunta original, con sus números, se conserva a continuación.)* **¿Se regenera el dato para que la métrica de tarea pueda decidir?** Medido y **re-medido** (2026-07-26): sd del F1 de párrafo entre imágenes = **0,4148** sobre **20 imágenes de val** → **±0,093** por run, más ruido que las diferencias a distinguir. Llegar a ±0,029 pide ~200 imágenes de val (~2000 en total, lo que protocolo.md §3 ya pedía). **⚠ La primera estimación (0,372 → ±0,083) se dio por conservadora suponiendo que con modelos mejores la sd bajaría; se midió sobre los 20 runs ganadores y SUBE** (metrica-de-tarea.md §9.4): la sd es máxima con modelos intermedios. **El argumento de esta decisión se refuerza.** Dato de apoyo (§9.5): **7 de 20 imágenes cargan casi todo el fallo** y una falla en 20/20 réplicas, así que cambiar una sola imagen del val mueve el F1 en ~0,05. Y (§9.1) el techo de la reconstrucción con esquinas perfectas es **0,97**: lo que falta lo tiene que dar la red, así que sí hay margen real que medir | **No la tomo yo.** Regenerar **invalida la comparabilidad** con los 130 runs, 4 recorridos y 4 estudios actuales (otro fingerprint de B). Alternativa: seguir con 20 y usar la métrica de tarea **solo como informe del ganador**, nunca para elegir entre puntos. Detalle y aritmética en [metrica-de-tarea.md](metrica-de-tarea.md) §4. **⚠ Corregido 2026-07-26:** regenerar «más de lo mismo» **no** es una corrida de `make_synth_source.py` (ese script hace otro problema: barras de juguete, no texto renderizado) — pide el generador hermano **más el resize de F13**. La decisión, por tanto, incluye **por qué ruta** (§4.2) | 9 |
| **F12** | **¿Qué es exactamente «la CNN plana de coste equivalente»** del primer experimento (protocolo.md §6)? **Construible desde 2026-08-09** (`regions: single`) y desde 2026-08-25 se declara literal: `border_px: 0`. Falta **medir** la familia de 6 controles de plan-cnn-plana.md §3 | Aplazada hasta que exista la métrica de tarea. Cuando se abra: definir si el control es una arquitectura nueva en un registro (`arch`) o una degeneración permitida de la foveada, y **escribir el criterio antes de medir**. Nota (2026-07-26): el eje `d` de metrica-de-tarea.md §5 da **media respuesta sin construir el control** — si el F1 de tarea no se mueve entre `d=1` y `d=6`, la periferia no está aportando | investigación |
| **F13** | ✅ **CERRADA (2026-08-13): SÍ se porta, y está implementada.** Decidido por el usuario, que además encargó regenerar el dato (1000 imágenes 640×480, receta `dirty`, en el proyecto hermano) — con lo que el resize deja de ser de un solo uso y cae en la rama «más de una vez» de la respuesta. Vive en `fv.datasets.resize` (dominio A′) con CLI `fv-resize`, y **conserva las dos reglas**: la escala se mide de la salida (`scale: [sx, sy]`, dos escalas independientes por el redondeo) y las máscaras se remuestrean con NEAREST. Añadidas dos garantías que el original no documentaba: el reescalado de coordenadas es **recursivo por nombre de campo** (`box`/`quad` a cualquier profundidad, así `lines[]`/`words[]` no se quedan atrás) y los tamaños mezclados se rechazan (`mixed_source_sizes`) en vez de escribir un `derived` que mentiría. 13 tests en `tests/test_resize.py`. **Queda fuera**: la ruta `POST /sources/{id}/resize` + pantalla, que api.md sigue listando como prevista. *(La pregunta original se conserva a continuación.)* ⏸ **APARCADA con F11 (2026-07-26)**: sin regenerar el dato no hay resize que hacer; se reabre con F11 y la respuesta de la derecha sigue en pie. **¿Se porta el `resize` de fuentes a `fv` (dominio A), o se hace fuera y solo entra el resultado?** Descubierto al detallar la Fase 3 (2026-07-26): **todo el dato real** (`dirty-paragraphs-*`, 80×60) sale del generador hermano **reducido con un resize que este repo NO tiene** (`docs/plan.md` ya lo decía; `POST /sources/{id}/resize` está en api.md como previsto). Sin resolverlo, «regenerar el dato» de F11 no se puede hacer sin salir del proyecto | Si el resize se va a hacer **una vez**, un script de un solo uso + declarar el bloque `derived` en el `dataset.json` (como ya hacen las fuentes actuales). Si se va a hacer **más de una vez**, portarlo de `image-text-finder/src/itf/datasets/resize.py` conservando sus dos reglas: la escala se **mide de la salida** (dos escalas, x e y) y las máscaras se remuestrean con **NEAREST**. Detalle en [metrica-de-tarea.md](metrica-de-tarea.md) §4.3 | 9 (con F11) |
| **F14** | ✅ **CERRADA (2026-07-26): SÍ, y está implementada.** Decidido por el usuario. `fv.task.record_holdout_touch` anexa una línea a `runs/<run>/holdout.jsonl` por cada medición contra un B de holdout —**incluso cuando el número sale de caché**, que era justo el vistazo que no dejaba rastro— con `when/window_dataset/source/split/images/f1/sem/knobs/checkpoint/from_cache`. El payload devuelve `holdout_touches` y la UI lo enseña en ámbar. Qué cuenta como holdout lo dice **una sola función** (`is_holdout_source`): el campo `"holdout"` del `dataset.json` manda **en los dos sentidos** y el convenio de nombre `-holdout` es el respaldo. Append-only: **registra miradas, nunca bloquea una**. Tests: `test_holdout_touch_is_recorded`, `test_holdout_is_recognised_by_flag_over_name`. *(La pregunta original se conserva a continuación.)* **¿Se registra en disco que el holdout se tocó?** protocolo.md §3 dice «una sola vez, al final, y solo el ganador», pero hoy nada lo recuerda — y la caché hace que la segunda mirada sea gratis e **invisible** | Propuesta: anexar una línea a `runs/<run>/holdout.jsonl` al puntuar contra un B de holdout (cuándo, qué B, qué split, el número y sus knobs). **No lo hago solo**: convierte una caché pura en algo que escribe en el artefacto del run. Detalle en [metrica-de-tarea.md](metrica-de-tarea.md) §6.4 | 9 (Fase 4) |
| **F15** | ✅ **CERRADA (2026-07-26): NO se cambian.** Decidido por el usuario: mover un knob movería todos los números reportados sin que la caché avise, y los knobs «buenos» **comprimen** la separación entre modelos (0,343 → 0,147) con el `sem` quieto en ~0,08 — mejoran el número y empeoran su poder de distinguir. La medición queda en metrica-de-tarea.md §9.2 para cuando se reabra. *(La pregunta original se conserva a continuación.)* **¿Se cambian los defaults de los knobs de F** (`threshold=0.5`, `stride=n/2`, `nms_radius=n/2`)? Medido (2026-07-26, metrica-de-tarea.md §9.2): el óptimo es **el mismo en tres runs de calidad muy distinta** — `threshold≈0,3`, `stride=n/4`, `nms_radius=3n/4` — y los tres son **óptimos interiores** (la rejilla se extendió hasta acotarlos). El default deja en la mesa **+0,065** en el mejor run, **+0,187** en uno medio y **+0,261** en uno malo | **No la tomo yo**, y §3.7 ya lo decía: cambiarlos **mueve todos los números que el proyecto ha reportado** (la tabla de la Fase 1 incluida) y **la caché no avisa** — los knobs entran en su clave, así que simplemente se recalcula otra cosa con el mismo aspecto. Además las ganancias son **desiguales** (el run malo gana 4× más que el bueno), así que los knobs buenos **comprimen** la separación entre modelos de 0,343 a 0,147 con el `sem` quieto en ~0,08: mejora el número y **empeora** su poder de distinguir. Si se cambian: declararlo, re-medir §2, y decidir si los viejos se conservan como «knobs de la Fase 1» | 9 |

### Abiertas por el validador de especificación (2026-07-27)

*(Salieron construyendo `scriptserify_spec.py`. Ninguna bloquea nada hoy: el sistema funciona
con la respuesta provisional que se indica, y por eso se anotan en vez de detener el trabajo.)*

| | Pregunta | Provisional (lo que hay hoy) | Por qué es tuya |
|---|---|---|---|
| **F16** | ¿El **API** debe servir el vocabulario de **estados** (`queued/running/done/error/cancelled/interrupted`) y las **enumeraciones de receta** (`optimizer`, `scheduler`), como ya sirve los objetivos en `/sweeps/axes`? | Los estados se declaran **una vez** en `web/src/api.ts` (`TERMINAL_STATES`/`ACTIVE_STATES`) y los enums de receta **una vez** en `Recipes.tsx`. Una sola definición, sí — **pero en el dominio equivocado**: son vocabulario del backend | Añadir una ruta (o ampliar `/sweeps/axes`) es una decisión de API (R1: un recurso por sustantivo). El coste de no hacerlo ya se cobró: había **cuatro copias** de los estados y una esperaba un estado inexistente (`failed`) mientras ignoraba `interrupted` |
| **F17** | En Diagnóstico y Predecir, ¿qué runs se ofrecen: solo `done`/`cancelled` (como antes) o **todos los terminales**, incluidos `error` e `interrupted`? | **Todos los terminales**. Un run interrumpido tiene `best.pt` y es diagnosticable; si no lo tiene, el API se niega con su razón (`run_has_no_checkpoint`), que es la regla del proyecto | Cambia lo que el usuario ve en dos pantallas. La alternativa —filtrar por «tiene checkpoint»— exigiría que la lista de runs lo declare, que es un cambio de payload |
| **F18** | ¿Hasta dónde llega el «relieve» que hace legal el **WARN de contraste** en claro? | Se admite porque U3.6 (tabla de números) y U3.8 (leyenda con etiqueta) están mandadas por la especificación. La política está escrita en el bloque `check`, no en el código | Si algún día se quiere subir las cuatro series a 3:1, cambia la paleta y con ella todas las capturas y comparaciones visuales |
| **F19** | ¿Se anotan los componentes con `data-domain` / `data-view` / `data-fixes,-varies,-measures` (fase 4, sustrato B6)? | **Sí, se anotan** — es lo único que hace verificables los tipos 1, 2 y 6, que hoy son los que se degradan en silencio | Mete atributos de especificación en el DOM de producción. Son inertes (no cambian estilos ni comportamiento) y pesan bytes, pero es una decisión de estilo de código |
| **F20** | ¿**δ = 1-SE del mejor punto** sigue siendo la regla que declara un ganador, o se sustituye/acompaña por una **prueba de diferencia** (`fv.metrics.permutation_test`)? Medido el 2026-08-10 (plan-lr-L4.md §7, R4): sobre los **mismos 20 números**, `suggest_winner` imprime *«el mejor punto despega del resto»* con δ = 0,0020 mientras la permutación exacta da **p = 0,341** | Se queda δ. Es la regla de **empate** de protocolo.md §1.5 y para eso funciona; el problema medido es que (a) δ solo mira la dispersión **del mejor punto** e ignora la del rival —aquí el doble—, (b) 1 SE es una banda de ~68 % sobre *una* media, no una prueba entre dos, y (c) **la frase que imprime afirma más de lo que el número aguanta**, y es la que lee un estudio OAT al arrastrar un ganador | Cambia **cómo se elige un ganador en todo el proyecto**: `suggest_winner` la usan los recorridos, los estudios OAT (`advance` arrastra por ella) y los dos CLIs. Endurecerla haría que estudios que hoy avanzan **declaren empate y se paren**. Los veredictos ya publicados no se caen (`n_layers` L4 vs L2 son 12× δ y p = 0,032), pero **todo margen cercano a δ habría que releerlo**. Mínimo intermedio, si no se quiere tocar la selección: **imprimir la p al lado de δ** y suavizar la frase |

## 3. Abiertas por diseño (heredadas, siguen abiertas aquí)

- **Reanudar dentro de un run** (contrato ⑪, formatos.md §4.2.2): esperar y **medir el ahorro**
  antes de pagar el formato de `last.pt` bit-exacto. Reanudar el recorrido sí se construye.
- **Sondas multicanal** (occlusion/deconvolución sobre entrada compuesta): el hermano lo dejó
  como «diseño real, no un fix». Aquí V4 se diseña ocluyendo **en la imagen original**, pre-
  muestreo ([ui/2-vistas.md](ui/2-vistas.md), V4), cuando llegue su fase.

## 4. Cerradas

| | Decisión | Dónde quedó escrita |
|---|---|---|
| C1 | La vista foveada se construye en el dataloader; B guarda `images` + etiquetas (lección D23 del hermano) | organizacion.md §1-B, formatos.md §4.1 |
| C2 | Enmascarar **antes** de convolucionar (opción A de instructionsNewNN.md §7) | organizacion.md §3, instructionsNewNN.md §11 |
| C3 | H cubre C y/o D desde el diseño (la D22 del hermano, resuelta aquí por definición) | organizacion.md §1-H |
| C4 | Los rangos de búsqueda son funciones de `N` (`fv.fovea`), nunca constantes en un spec | instructionsNewNN.md §3, organizacion.md §1-C |
| C5 | `N` fijo por experimento (no se mezclan escalas en una corrida) | instructionsNewNN.md §11 |
| C6 | Métricas de ranking en píxeles de la imagen original (⑨-extensión) | organizacion.md §2-⑨, protocolo.md §2 |
| C7 | Allowlist de raíces + CORS cerrado (el API acabará en un server) | api.md §3 |
| C8 | Se versiona la descripción, se ignora la carga | formatos.md §5 |
| C9 | **F1**: cabezas de esquina `4×[exists, x, y]` sobre el backbone foveado (2026-07-21, decisión del usuario). El clasificador de la spec era placeholder; el avg-pool global se sustituye | herencia.md §2, organizacion.md §1-C, formatos.md §4.1 |
| C10 | **F1b**: la ventana etiquetada es la **fóvea** (2026-07-21, decisión del usuario): `center_out(C) == window_size(B)` es el contrato ①a; la periferia es solo contexto; `corner_evidence` se congela contra la fóvea | organizacion.md §2-①, formatos.md §4.1, `fv/validation` |
| C11 | Relleno de bordes en la vista: `pad_mode: edge` (replicar borde; nunca ceros a secas — «no hay texto» enseña una regla falsa), con **máscara de cobertura** calculada y enseñada en F0 para depurar. Un canal de máscara como entrada de la red queda como trabajo futuro (F7 sigue abierta para esa mitad) | `fv/fovea.build_view`, [ui/2-vistas.md](ui/2-vistas.md) F0 |
| C12 | El anillo periférico se construye con **pooling anisótropo por zonas** (celdas de esquina d×d, bandas d×1/1×d, centro 1×1 copiado exacto), co-registrado columna a columna con la fóvea. El código de referencia de instructionsNewNN.md §5 (`avg_pool2d` de la imagen entera + pegar bordes) **no tipa para d>1** (el pooled mide original/d ≠ N); esta construcción reproduce exactamente la tabla de coordenadas de §4, que es la intención del documento. Implementado con `np.add.reduceat` (C-speed, sin bucle Python); tests contra la tabla | `fv/fovea.build_foveated_input`, tests/test_fovea.py |
| C13 | **Barrido por ejes (OAT)** — 12 decisiones de **diseño** cerradas con el usuario (2026-07-23), **código pendiente en otra sesión**: builder paramétrico (`channels` lista, stride en 1ª capa ⇒ `n_layers` fuera de `stride_range`, `n_layers` único), derivador de base desde `window_size` (expone `d`/`c_frac`, desempate `N` mínimo, sin `N` → aflojar `c_frac` con razón), schedule como **dominio nuevo `studies/`** no ejecutor, base inline (`base_label` + `derivation`), ganador **sugerido** coste/calidad (`δ`, usuario confirma), semillas de confirmación configurables (default 3). Runs de ejemplo **descartados** (sin migración de checkpoints) | barrido-por-ejes.md §13; organizacion.md §1-I y ⑫; formatos.md §4.7 |
| C14 | **Geometría en px reales** (2026-08-25, decisión del usuario). La fóvea y el borde difuso se declaran como longitudes independientes (`fovea_px`, `border_px`), y **cómo se reduce el borde** (`border_reduce`) queda separado de **cuánto borde hay** — para que el método de reducción pueda cambiar sin tocar ninguna de las dos definiciones. `N` pasa a **derivado**. Los dos solapes se declaran por separado (`overlap_fovea_px`, `overlap_border_px`) y ambos admiten 0. `d` se **renombra** a `border_reduce` porque cambió de significado: un eje `d` viejo se rechaza con `axis_renamed` en vez de entrenar otra red en silencio. **Ningún peso cambia** (verificado: checkpoint viejo carga `strict=True`, 168.652 params, misma salida) | instructionsNewNN.md §2.1; organizacion.md §1-C; barrido-por-ejes.md §5; glosario.md |
