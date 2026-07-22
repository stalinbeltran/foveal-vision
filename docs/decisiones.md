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

**En juego**: la entrada compuesta cubre `original_size = center_out + 2·periph_out·d` px de
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
| **F5** | ¿`c_frac` y `pen_frac` fijos por aplicación o barribles? (§11 de la spec) | Barribles **en grid aparte** con descarte declarado de puntos inválidos (los asserts matan muchas combinaciones); no mezclarlos con el barrido de kernels al principio | 7 |
| **F6** | Política de redondeo/paridad al derivar `center_out` de `N` impar o `c_frac` rara (§11) | `round_to_even` documentado en `derive_dims` con test; rechazar `N` impar de entrada | 2 |
| **F7** | Relleno de bordes de imagen en la vista: ¿valor del relleno + máscara, o solo máscara? | Media enmascarada (el tono del relleno no entra en el número) + máscara de cobertura — la lección medida del hermano | 3 |
| **F8** | ¿Kernels periféricos con forma distinta o sparsity? (§11) | Aplazar: primero medir la forma básica | investigación |
| **F9** | ¿Integración con glimpses secuenciales tipo RAM? (§11) | Aplazar; la arquitectura por ventana no lo bloquea | investigación |
| **F10** | Presupuesto de recorrido en épocas o en segundos (organizacion.md §1-H) | Declararlo en el spec; para espacios sobre C, **segundos** con el coste por punto registrado | 7 |
| **D2** | ¿Extraer las librerías compartidas del hermano (`matrixview`, cola de jobs, registro de runs) a `claude-libs/`, o copiar? | Extraer `matrixview` seguro (cero deps, dos consumidores reales); la cola y el registro, decidir en su fase mirando cuánto divergen | 1/4/5 |

## 3. Abiertas por diseño (heredadas, siguen abiertas aquí)

- **Reanudar dentro de un run** (contrato ⑪, formatos.md §4.2.2): esperar y **medir el ahorro**
  antes de pagar el formato de `last.pt` bit-exacto. Reanudar el recorrido sí se construye.
- **Sondas multicanal** (occlusion/deconvolución sobre entrada compuesta): el hermano lo dejó
  como «diseño real, no un fix». Aquí V4 se diseña ocluyendo **en la imagen original**, pre-
  muestreo (ui.md §3), cuando llegue su fase.

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
| C11 | Relleno de bordes en la vista: `pad_mode: edge` (replicar borde; nunca ceros a secas — «no hay texto» enseña una regla falsa), con **máscara de cobertura** calculada y enseñada en F0 para depurar. Un canal de máscara como entrada de la red queda como trabajo futuro (F7 sigue abierta para esa mitad) | `fv/fovea.build_view`, ui.md F0 |
| C12 | El anillo periférico se construye con **pooling anisótropo por zonas** (celdas de esquina d×d, bandas d×1/1×d, centro 1×1 copiado exacto), co-registrado columna a columna con la fóvea. El código de referencia de instructionsNewNN.md §5 (`avg_pool2d` de la imagen entera + pegar bordes) **no tipa para d>1** (el pooled mide original/d ≠ N); esta construcción reproduce exactamente la tabla de coordenadas de §4, que es la intención del documento. Implementado con `np.add.reduceat` (C-speed, sin bucle Python); tests contra la tabla | `fv/fovea.build_foveated_input`, tests/test_fovea.py |
