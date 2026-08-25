# UI · Tipo 1 — Especificación estructural (dominio → pantalla)

> **Qué decide**: cuántas pantallas existen, a qué dominio responde cada una, y qué cabe dentro.
> **Qué NO decide**: qué significa una vista de análisis (→ [2-vistas.md](2-vistas.md)), cómo se
> pinta (→ [3-representacion.md](3-representacion.md)).
> **De dónde deriva**: [organizacion.md](../organizacion.md) §1 — **ese documento manda**. Esta
> especificación es su proyección sobre pantallas; si divergen, el error está aquí.
> **Cómo se hace cumplir**: **ejecutable desde 2026-07-27** (fase 4 del validador): cada pantalla
> declara su dominio con `data-domain` y el mapa de abajo se casa contra las rutas y las
> etiquetas reales del nav. Lo que sigue siendo humano: si el reparto de dominios es el
> *correcto*. Ver [validador.md](validador.md).

---

## Las reglas

**U1.1 — Una pantalla, un dominio.** Cada sustantivo del dominio (A, B, C, D, E, H, I) tiene su
pantalla: listar, crear, nombrar, borrar. **Un formulario que mezcla dos dominios es un bug de
organización**, no una decisión de diseño, y se arregla partiendo la pantalla — nunca añadiendo
una pestaña.

```check U1.1
substrate: dom
kind: dom_query
scope: "*"
args:
  selector: "h2[data-domain]"
  min_count: 1
strength: strong
```

**U1.2 — F es un verbo, no una entidad.** La inferencia no se lista ni se guarda: es un panel de
resultados sobre un E. Lo mismo vale para los **cruces** (E×B diagnóstico, E×A métrica de tarea):
son cachés, no entidades, y **viven dentro de la pantalla del dominio dueño** — el bloque de tarea
está en el detalle de un run, no en una pantalla «Métricas».

```check U1.2
substrate: doc
kind: catalog_match
args:
  left: doc_screen_routes
  right: app_routes
strength: strong
```

**U1.3 — Sin números de paso.** No hay wizard ni pipeline numerado: en investigación no se recorre
un flujo, se **itera sobre un punto y se vuelve**. Los grupos del nav (Datos / Modelo / Entrenar /
Analizar) ordenan por dependencia de dominio, no por secuencia temporal.

```check U1.3
substrate: dom
kind: dom_absent_text
scope: "/sources"
args:
  words: ["paso 1", "paso 2", "siguiente paso"]
strength: weak
```

**U1.4 — Una pantalla nueva se justifica por un dominio nuevo.** Si algo pide pantalla y no es un
sustantivo de organizacion.md §1, la pregunta correcta es de qué dominio es un panel. El dominio
**I** (estudios) apareció así: primero fue dominio, después pantalla.

```check U1.4
substrate: doc
kind: catalog_match
args:
  left: doc_screen_labels
  right: nav_labels
strength: strong
```

**U1.5 — Verificar un objeto no exige entrenar.** Requisito literal del usuario: *en todo momento
debe ser posible verificar los objetos creados — fuentes, datasets, redes, runs, recorridos,
análisis*. De ahí que Ventanas enseñe el recorte crudo sin modelo, y que Redes enseñe geometría
sin pesos.

```check U1.5
substrate: dom
kind: dom_query
scope: "/window-datasets/:name"
args:
  selector: "[data-testid=window-grid] canvas"
  min_count: 1
strength: strong
```

**U1.6 — Un objeto enseña entera la definición con que se creó, y la enseña en su detalle.**
Listar no es verificar: la tabla resume (nombre y dos o tres columnas), y **al seleccionar una fila
aparece la definición completa** — todos los parámetros que ese objeto fija, con su valor literal,
no un conteo. Un parámetro que solo existió en el formulario de creación es un parámetro que ya no
se puede comprobar, y U1.5 pide justo lo contrario. Cómo se lee:

- **Del objeto guardado, nunca del formulario recordado.** La fuente es lo que devuelve el API
  (`GET /studies/{name}` → `plan`), no el `localStorage` del navegador: lo recordado son defaults,
  jamás verdad ([7-operacion.md](7-operacion.md) U7.3). El payload ya lo trae; lo que se especifica
  aquí es enseñarlo.
- **Definición y estado se separan, y se dice cuál es cuál.** Lo comiteable (`plan.json`,
  `spec.json`, el config de C o de D) es la **definición**: lo que el usuario pidió. Lo vivo y
  regenerable (`progress.json`, `state`, la cola pendiente, los ganadores confirmados) es
  **progreso**. Mezclados en un mismo bloque no se puede distinguir lo pedido de lo ocurrido; el
  disco ya los tiene separados ([formatos.md](../formatos.md) §4.7: `plan.json` comiteable,
  `progress.json` estado vivo) y la pantalla lo respeta.
- **Los valores compuestos se enseñan completos**: el rango de un eje es su lista de valores (con
  su cardinalidad al lado, no en su lugar), y el presupuesto va **con su unidad declarada**.
- Un campo ausente se dibuja como ausente ([5-invariantes.md](5-invariantes.md) U5.3) y las
  etiquetas son **las mismas palabras** con que el formulario lo pidió
  ([8-lexico.md](8-lexico.md) U8.4).

Caso vivo (2026-07-28, reportado por el usuario): **Estudios era la única pantalla que no lo
cumplía**. La lista daba dataset, nº de ejes y nº de pasos; el detalle daba los pasos y los
ganadores arrastrados — pero `objective`, `seeds`, `base_recipe`, `budget.epochs` y **el rango de
cada eje** no volvían a aparecer en ningún sitio después de crear el estudio: vivían solo en
`plan.json`. Recorridos sí cumple, desde su bloque `base-nn` (red base, receta base, ejes barridos,
dims derivadas).

```check U1.6
substrate: dom
kind: dom_query
scope: "/studies"
args:
  selector: "[data-testid=studies-table] tbody tr"
  sibling_required: "[data-testid=study-plan]"
strength: strong
```

## El mapa de pantallas

| Grupo | Pantalla | Ruta | Dominio |
|---|---|---|---|
| **Datos** | Fuentes | `/sources` | A (+ derivadas y resize) |
| | Ventanas | `/window-datasets` · `/window-datasets/:name` | B (+ detalle paginado con los recortes crudos) |
| **Modelo** | Redes | `/networks` | C |
| | Recetas | `/recipes` | D |
| **Entrenar** | Entrenar | `/train` | B×C×D + X → E |
| | Recorridos | `/sweeps` | H |
| | Estudios | `/studies` | I (estudio OAT; genera recorridos H, guía paso a paso) |
| | Runs | `/runs` · `/runs/:name` | E (lista + detalle) |
| **Analizar** | Diagnóstico | `/diagnostics` | E×B |
| | Predecir | `/predict` | F |

**Cumplimiento (verificado 2026-07-27)**: las 12 rutas existen en
[web/src/App.tsx](../../web/src/App.tsx#L114-L128) con esas etiquetas de nav.
**U1.6 (2026-07-28)**: se escribió `violada` a propósito y se implementó el mismo día en
[web/src/screens/Studies.tsx](../../web/src/screens/Studies.tsx) — bloque `study-plan` con los
campos escalares del plan y `study-axes` con la escalera; ambos ya en el inventario de
[7-operacion.md](7-operacion.md) U7.11. Verificado clicando los **5 estudios** con Playwright (sin
errores de consola) y, para los estados que ningún estudio real ejercitaba (`auto`, `channels[i]`
expandido, un eje aún en cola), con un estudio temporal creado y **borrado** después. El bloque
del plan **no enumera los campos que conoce y calla el resto**: lo que `plan.json` traiga y esta
pantalla no nombre se pinta igual bajo su clave — un campo añadido en Python no puede volverse
invisible aquí, que es exactamente cómo divergen las dos copias.
⚠ [ui.md](../ui.md) llamaba a la pantalla de estudios **«Barrido por ejes»**; la UI dice
**«Estudios»**. Manda la UI: el nombre en pantalla es *Estudios* y el doc de diseño es
[barrido-por-ejes.md](../barrido-por-ejes.md), que describe el **método**, no la pantalla.

## Lo propio de cada pantalla

*(Lo que no se menciona funciona como en el proyecto hermano: tablas, galerías paginadas, jobs con
polling, negativas con `hint` visible.)*

- **Fuentes (A)**: solo lectura + resize. Columna de procedencia para derivadas
  (`← padre ×escala`; ausente = original, pintado como tal). El visor con el `quad` dibujado sobre
  los píxeles es la herramienta de verificación del resize.
- **Ventanas (B)**: metadatos + desbalance de clases (del manifest) + **ver el dato crudo**: vista
  previa al azar en la tarjeta y detalle paginado con el recorte y su etiqueta — **sin necesitar un
  run**. Es donde se decide la ventana etiquetada, y donde nace el contrato ①.
- **Redes (C)** — la pantalla más importante de este proyecto y la más distinta del hermano. Al
  editar `fovea_px`, `border_px`, `border_reduce` y los dos solapes:
  - enseña **en vivo** las dimensiones derivadas (`N`, `border_cells`, `center_band`,
    `original_size`) — ver U5.5;
  - enseña los **rangos calculados** de `k_center`/`k_periph`/`s_center`/`s_periph`/`d` para ese
    `N` — son los mismos que un recorrido usará con `"auto"`;
  - dibuja **el diagrama de zonas** (anillo externo / banda de penetración / núcleo) y la
    correspondencia original→input — la vista FG1 de [2-vistas.md](2-vistas.md);
  - los asserts violados se enseñan con su razón (`penetration_too_large`…), en el momento.
- **Recetas (D)**: el catálogo con **cada definición en línea**. Un hiperparámetro sin definición
  no está terminado. `device` no está aquí (contrato ⑩).
- **Entrenar**: elegir B+C+D **por nombre**, `device` aparte; enseña si casan (①) y **estima el
  coste** con los `seconds` de runs comparables — si no hay comparables, lo dice (U6.9).
- **Recorridos (H)**: fijar B; construir el espacio sobre C y/o D — los ejes de geometría se ofrecen
  **desde los rangos calculados** (marcar cuáles entran, restringirlos, o `auto`); estrategia,
  objetivo (con el bloqueo del ⑨ activo en el formulario), presupuesto **con su unidad declarada**;
  tabla de puntos ordenada por objetivo; parar/reanudar; veredicto de ganador. La **lista** ofrece
  las facetas de Runs que aplican y se **parte por estado**: Activos arriba, Terminados debajo
  (plegable). Muestra el límite de workers y por qué en CPU es 1.
- **Estudios (I)** — `(fija: B + receta D; varía: un eje a la vez; mide: el objetivo con la regla
  coste/calidad)`: deriva la base del problema (`window_size` → `N`/geometría), muestra el config
  base con el **origen por campo** (`default | winner | user`), lleva la **escalera** de ejes
  ordenados y, por paso, **propone** el ganador que el usuario **confirma**. Reutiliza la tabla de
  ranking de Recorridos. Al **seleccionar un estudio en la tabla**, el detalle abre con su **plan
  completo** (U1.6) antes que con el progreso: dataset B, receta base D, objetivo, semillas,
  presupuesto **con unidad** (`epochs`), y la **escalera entera** — cada eje con su rango literal
  (o `auto`), en orden de barrido, marcando cuál está hecho, cuál corre y cuáles quedan. Los pasos,
  los ganadores arrastrados y la cola son **progreso**, y van debajo, separados.
- **Runs (E)**: lista y detalle. La lista **agrupa por jerarquía de dominio B → C → D** (el árbol
  colapsa solo los niveles con un único valor tras filtrar), ofrece por fila **renombrar y borrar**,
  y expone **facetas** —B, C, D, recorrido (con «sin recorrido»), estado, monitor, búsqueda— cuyas
  opciones salen de los valores presentes, **nunca ofrecen vacío**. Filtrar hasta una sola
  combinación degrada el árbol a tabla plana. El detalle: procedencia entera, `execution`, curvas en
  small multiples, todas las épocas, y el bloque de métrica de tarea (U6.8).
- **Diagnóstico (E×B)**: elegir run y split; las vistas leen la tabla-caché por ventana. La galería
  va peor-primero y filtra por resultado al umbral puesto — **mover el umbral no recalcula nada**.
- **Predecir (F)**: run + imagen → **todas las etapas** superpuestas y conmutables; knobs como
  sliders con repintado en vivo y **acuse de espera** (una respuesta lenta sin acuse se lee como un
  clic perdido).
