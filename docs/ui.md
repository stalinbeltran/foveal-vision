# Organización de la UI

Cómo se estructura la interfaz aplicando los dominios de [organizacion.md](organizacion.md) —
**ese documento manda**. Proyecto de investigación: la UI no es un panel de control, es el
**instrumento de medida**. Las pantallas de análisis son el producto.

Requisito del usuario, literal: *en todo momento debe ser posible verificar los objetos creados
— fuentes, datasets, redes, runs, recorridos, análisis — y revisar cada nn, grupos de
parámetros, y probar los resultados.*

---

## 0. Las dos reglas (heredadas, probadas)

**Regla 1 — una pantalla, un dominio.** Cada sustantivo (A, B, C, D, E, H) tiene su pantalla:
listar, crear, nombrar, borrar. Un formulario que mezcla dos dominios es un bug de organización.

**Regla 2 — toda vista de análisis declara `(qué fija, qué varía, qué mide)`.** Un recorrido
fija B, varía C/D y mide el objetivo. Un mapa de activaciones fija E y la ventana, y varía la
capa. Si no puedes decir qué fija una vista, la vista no sabe lo que enseña.

### Librerías y color

- **Observable Plot** para gráficas con ejes/leyendas (curvas, scatter, paralelas); **canvas a
  mano** para matrices densas (kernels, feature maps, vistas de entrada) — reutilizando el
  patrón `MatrixCanvas`/`LayerMaps` del hermano vía `matrixview`; **HTML/CSS** para meters y
  overlays.
- **La paleta vive en `web/src/theme/tokens.css` y solo ahí, y se valida con script**
  (`npm run validate:palette`, portado del hermano), nunca a ojo.
- Reglas de color heredadas: datos con signo (pesos, gradientes) → **divergente centrada en 0**
  (esconder el signo esconde qué excita y qué inhibe); magnitudes → secuencial de una tinta;
  **jamás doble eje** (loss, accuracy y px son escalas distintas: small multiples con eje x
  alineado); **el trabajo de color lo declara el payload**, el cliente no lo adivina; todo mapa
  de calor tiene su **tabla de números** (gemela accesible y la mejor vista de depuración).

## 1. El mapa de pantallas

| Grupo | Pantalla | Dominio |
|---|---|---|
| **Datos** | Fuentes | A (+ derivadas y resize) |
| | Ventanas | B (+ detalle paginado con los recortes crudos) |
| **Modelo** | Redes | C |
| | Recetas | D |
| **Entrenar** | Entrenar | B×C×D + X → E |
| | Recorridos | H |
| | Runs | E (lista + detalle) |
| **Analizar** | Diagnóstico | E×B |
| | Predecir | F |

Sin números de paso: en investigación no se recorre un pipeline, se **itera sobre un punto** y
se vuelve.

**Estado de UI recordado (defaults, no fuente de verdad).** Los filtros y los valores de
formulario de cada pantalla se recuerdan en `localStorage` (per-browser, listo para multi-usuario
sin cambios) vía el hook `usePersistedState`. El nav ofrece **Guardar / Cargar sesión**: Guardar
vuelca todo a un JSON comiteable (`state/ui-state.json`, `PUT /ui-state`) para que una sesión de
trabajo viaje al server con GPU; Cargar lo trae y recarga. Es solo conveniencia: las pantallas
siguen leyendo el A–H real del API, y un nombre de run/recorrido **no** se recuerda (es de un solo
uso). Un valor recordado que ya no existe (un run borrado) cae al primero disponible, no rompe.

## 2. Pantallas de dominio — lo propio de este proyecto

*(Lo que no se menciona funciona como en el hermano: tablas, galerías paginadas, jobs con
polling, negativas con `hint` visible.)*

- **Fuentes (A)**: solo lectura + resize. Columna de procedencia para derivadas
  (`← padre ×escala`; ausente = original, pintado como tal). El visor con el `quad` dibujado
  sobre los píxeles es la herramienta de verificación del resize.
- **Ventanas (B)**: metadatos + desbalance de clases (del manifest) + **ver el dato crudo**:
  vista previa al azar en la tarjeta y detalle paginado (`/window-datasets/{name}`) con el
  recorte y su etiqueta — sin necesitar un run. Es donde se decide la ventana etiquetada, y
  donde nace el contrato ①.
- **Redes (C)** — la pantalla más importante de este proyecto y la más distinta del hermano.
  Al editar `N`, `c_frac`, `d`, `pen_frac`:
  - enseña **en vivo** las dimensiones derivadas (`center_out`, `periph_out`, `penetration`,
    `original_size`) vía `POST /networks/validate` — el usuario ve qué implica cada fracción
    antes de guardar;
  - enseña los **rangos calculados** de `k_center`/`k_periph`/`s_center`/`s_periph`/`d` para
    ese `N` — son los mismos que un recorrido usará con `"auto"`;
  - dibuja **el diagrama de zonas** (anillo externo / banda de penetración / núcleo) y la
    correspondencia original→input (las tablas de instructionsNewNN.md §4 y §6, dibujadas);
  - los asserts violados se enseñan con su razón (`penetration_too_large`…), en el momento.
- **Recetas (D)**: el catálogo con cada definición en línea. `device` no está aquí.
- **Entrenar**: elegir B+C+D por nombre, `device` aparte; enseña si casan (①) y **estima el
  coste** con los `seconds` de runs comparables (misma huella de B, misma red) — si no hay
  comparables, lo dice, no inventa.
- **Recorridos (H)**: fijar B; construir el espacio sobre C y/o D — los ejes de geometría se
  ofrecen **desde los rangos calculados** (marcar cuáles entran, restringirlos, o `auto`);
  estrategia, objetivo (con el bloqueo del ⑨ activo en el formulario), presupuesto **con su
  unidad declarada** (épocas o segundos); tabla de puntos ordenada por objetivo; parar/reanudar.
  Muestra el límite de workers y por qué en CPU es 1. La **lista** ofrece las mismas facetas que
  Runs que aplican (B, C, D, objetivo, búsqueda) y se **parte por estado**: Activos arriba,
  Terminados debajo (plegable). **Borrar un recorrido borra en cascada sus runs hijos** (un run
  hijo no se borra solo: sus puntos se comparan juntos) — se confirma antes y se niega si algo
  está en marcha, para no dejar huérfanos.
- **Runs (E)**: lista y detalle. La lista **agrupa por jerarquía de dominio B → C → D** (el árbol
  colapsa solo los niveles con un único valor tras filtrar), ofrece por fila **renombrar y borrar**
  (borrar se niega con su razón si el run pertenece a un recorrido), y expone **facetas** —B, C, D,
  recorrido (con "sin recorrido"), estado, monitor y búsqueda por nombre— que podan el conjunto;
  las opciones de cada faceta salen de los valores presentes, nunca ofrecen vacío. Filtrar hasta
  una sola combinación degrada el árbol a tabla plana. Una fila: estado, recorrido padre, monitor,
  best, s/época. El detalle: procedencia entera, `execution`, curvas en small multiples, todas
  las épocas. Renombrar navega a la URL nueva.
- **Diagnóstico (E×B)**: elegir run y split; las vistas leen la tabla-caché por ventana. La
  galería va peor-primero y filtra por resultado al umbral puesto — mover el umbral no recalcula
  nada.
- **Predecir (F)**: run + imagen → todas las etapas superpuestas y conmutables; knobs como
  sliders con repintado en vivo y acuse de espera (una respuesta lenta sin acuse se lee como un
  clic perdido).

## 3. Catálogo de vistas

Numeración del hermano donde la vista se hereda; se marca lo que cambia. Cada una declara
(fija / varía / mide).

| | Vista | Fija | Varía | Mide | Notas foveadas |
|---|---|---|---|---|---|
| V19→**F0** | **Vista de entrada, canal a canal** | E (o C), ventana | el canal/zona | la entrada compuesta: centro, periferia reducida, **máscaras de rama**, cobertura del relleno | **Pasa de sonda a vista fundamental**: aquí la entrada es una composición no trivial y depurarla es depurar el proyecto. La máscara siempre junto a su canal, con la cobertura como número y filtro por cobertura |
| V1 | Kernels | E | la rama | los pesos | **Por rama** (centro/periferia), divergente ±0. `in_channels=1` por rama ⇒ la capa 1 es exacta e interpretable en ambas |
| V2 | Feature maps | E, ventana | capa × rama | activación | Secuencial (o divergente si la activación tiene signo — mirar `spec.activation`, no asumir). **La banda de penetración es lo interesante**: dónde se suman las dos ramas |
| V3 | Predicción de la ventana | E, ventana | — | 4×`[p, x, y]` | **4 meters** contra el umbral + overlay con el error dibujado (anillo = verdad, punto = predicción), categórica ×4 — heredada tal cual |
| V4 | Occlusion | E, ventana | posición de la máscara | caída de p | Diseñar para entrada compuesta: ocluir en la imagen original **antes** del muestreo foveado (ocluir la vista mezclaría escalas). El hermano dejó esto explícitamente como diseño pendiente |
| V5 | Scrubber | E, imagen | el recorte | predicción y estabilidad a ±1 px | La estabilidad fija el stride de inferencia y la fusión |
| V6 | Galería peor-primero | E, split | la ventana | error | |
| V7 | Error por posición | E, split | posición real | error | La resolución del mapa es un control (el moteado parece estructura); enseñar nº de muestras por celda |
| V8 | Scores + PR | E, split | threshold | precision/recall | El barrido gratis: scores guardados |
| V12/V13 | Pareto / paralelas | B | el punto (C/D) | el objetivo | **Los ejes de geometría entran**: `d`, `k_center`… en paralelas; color por el eje barrido (magnitud continua → rampa secuencial) |
| V14 | Curvas | B,C,D | la época | loss y métricas | Small multiples, eje x alineado |
| V16 | Deconvolución | E, ventana | el filtro | qué píxeles lo activaron | Gradiente puro, siempre divergente ±0; gana valor de la capa 2 en adelante; `silent` con palabras cuando un filtro no dispara |
| V18 | Evidencia disponible | E, split, umbral | cuánto del párrafo cabe en la ventana | detección y posición **por separado**, por banda de evidencia | Vuelve con las cabezas de esquina (C9). `corner_evidence` congelada contra la ventana etiquetada (F1b). **Es el criterio de éxito del primer experimento**: ¿la periferia baja el `err_px` de la banda ciega sin dañar la visible? |
| V11 | Etapas del pipeline | E, imagen | la etapa | qué se pierde y dónde | Crudas pre-NMS / esquinas post-NMS / cajas TL→BR, conmutables — sin la cruda, «el párrafo salió mal» no es diagnosticable |

**Nuevas, propias de la geometría foveada:**

- **FG1 — El diagrama de zonas de una red** (en Redes, sin pesos): anillo / penetración /
  núcleo, dimensiones derivadas, y la huella sobre la imagen original (`original_size` dibujado
  sobre una muestra real de A a escala). Es lo único que una red sin entrenar puede enseñar de
  sí misma, y aquí es mucho.
- **FG2 — Contribución por rama**: para una ventana, `‖c‖` vs `‖p‖` por zona (¿la periferia
  aporta, o la fusión la apaga?). Fija E y ventana; varía la zona; mide la norma de cada rama.
  Es la vista que contesta la pregunta de investigación del proyecto.
- **FG3 — Comparador de vistas foveadas** (en Redes o Ventanas): la misma ventana original
  construida con dos configs de geometría lado a lado. Barato (no hay modelo) y es como se elige
  qué rangos merecen recorrido.

## 4. Prioridad

1. **F0 (entrada), FG1 (zonas), V3, V14** — verificar el dato, la geometría y el run: el
   mínimo para confiar en el primer entrenamiento.
2. **V6, V7, V8** — la tabla por ventana y lo que se lee de ella.
3. **V12/V13** — cuando exista H.
4. **V1, V2, FG2** — mirar por dentro, por rama.
5. **V4, V5, V16** — sondas finas (V4 exige el diseño de oclusión pre-muestreo).
