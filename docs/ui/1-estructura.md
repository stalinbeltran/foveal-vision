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
  editar `N`, `c_frac`, `d`, `pen_frac`:
  - enseña **en vivo** las dimensiones derivadas (`center_out`, `periph_out`, `penetration`,
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
  ranking de Recorridos.
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
