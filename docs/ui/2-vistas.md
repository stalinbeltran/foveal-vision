# UI · Tipo 2 — Especificación epistémica (qué significa una vista)

> **Qué decide**: qué afirma cada vista de análisis, y en qué orden se construyen.
> **Qué NO decide**: en qué pantalla vive (→ [1-estructura.md](1-estructura.md)), con qué colores
> se dibuja (→ [3-representacion.md](3-representacion.md)), con qué salvedades se enseña el número
> (→ [6-numeros.md](6-numeros.md)).
> **Cómo se hace cumplir**: **parcialmente ejecutable desde 2026-07-27**: cada vista declara su
> tripleta en el DOM (`data-view` + `data-fixes/-varies/-measures`) y el validador comprueba que
> **está declarada** y que el id existe en este catálogo. Que la tripleta declarada sea la
> **cierta** sigue siendo juicio humano (U2.2), y por eso este tipo conserva el peor ratio
> daño/detectabilidad de los ocho.

---

## Las reglas

**U2.1 — Toda vista de análisis declara `(qué fija, qué varía, qué mide)`.** Un recorrido fija B,
varía C/D y mide el objetivo. Un mapa de activaciones fija E y la ventana, y varía la capa.

```check U2.1
substrate: dom
kind: dom_query
scope: "*"
args:
  selector: "[data-view]"
  optional: true
  must_have_attr: ["data-fixes", "data-varies", "data-measures"]
strength: strong
```

**U2.2 — Si no puedes decir qué fija una vista, la vista no sabe lo que enseña** — y no se
construye. La tripleta es un requisito de diseño, no documentación posterior.

```check U2.2
substrate: none
reason: "juicio de diseno: ninguna herramienta decide si una vista sabe lo que ensena.
  B6 comprueba que alguien lo declaro, no que la declaracion sea cierta"
```

**U2.3 — La tripleta viaja a la pantalla.** Donde quepa, va como subtítulo de la vista (p. ej. el
subtítulo de las curvas de un recorrido dice qué agrupa y qué banda dibuja). El usuario no debería
tener que abrir este documento para saber qué está mirando.

```check U2.3
substrate: same_as
target: U2.1
reason: "declarar la tripleta y declararla EN PANTALLA son el mismo atributo"
```

**U2.4 — La vista es del proyecto, no de la pantalla.** Una vista heredada del hermano **conserva
su número** (`V1`…`V19`) aunque cambie de sitio o de forma: es lo que permite leer su evidencia
original. Las propias de la geometría foveada se numeran `FG#`. `V19` se renombró a **`F0`** porque
aquí dejó de ser una sonda y pasó a ser vista fundamental.

```check U2.4
substrate: doc
kind: spec_lint
args:
  assert: view_ids_unique
strength: strong
```

**U2.5 — Una vista sin (fija, varía, mide) escrito aquí no se implementa**, y una implementada que
no está en el catálogo es deuda: o entra, o se quita.

```check U2.5
substrate: same_as
target: U2.1
reason: "el catalogo se casa en los dos sentidos: la misma comprobacion cubre la vista
  huerfana y la vista sin implementar"
```

## El catálogo

Numeración del hermano donde la vista se hereda; se marca lo que cambia.

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
| V11 | Etapas del pipeline | E, imagen | la etapa | qué se pierde y dónde | Crudas pre-NMS / esquinas post-NMS / cajas TL→BR, conmutables — sin la cruda, «el párrafo salió mal» no es diagnosticable. El **tamaño** de cada punto es su `score` en escala **absoluta 0→1** (relativizarla al min/max de la imagen haría que mover el umbral repintara todo sin cambiar el dato), y las cuatro **ranuras se filtran** desde la leyenda. ⚠ Con una red entrenada los scores se saturan (mediana 0,998 *medido 2026-08-31*) y todos los círculos salen iguales: por eso el rango por ranura va en la tabla de números y el valor exacto en el tooltip |
| V12/V13 | Pareto / paralelas | B | el punto (C/D) | el objetivo | **Los ejes de geometría entran**: `d`, `k_center`… en paralelas; color por el eje barrido (magnitud continua → rampa secuencial) |
| V14 | Curvas | B,C,D | la época | loss y métricas | Small multiples, eje x alineado. En Recorridos, N runs superpuestos (líneas por run / media ± banda) |
| V16 | Deconvolución | E, ventana | el filtro | qué píxeles lo activaron | Gradiente puro, siempre divergente ±0; gana valor de la capa 2 en adelante; `silent` con palabras cuando un filtro no dispara |
| V18 | Evidencia disponible | E, split, umbral | cuánto del párrafo cabe en la ventana | detección y posición **por separado**, por banda de evidencia | Vuelve con las cabezas de esquina (C9). `corner_evidence` congelada contra la ventana etiquetada (F1b). **Es el criterio de éxito del primer experimento**: ¿la periferia baja el `err_px` de la banda ciega sin dañar la visible? |

- **FR1 — Revisión a ojo de un split** (pantalla Revisar) · *fija: el DATASET de ventanas y el
  split · varía: la imagen · mide: qué detecta y qué se le escapa*: las imágenes enteras del split
  en miniaturas, con las cajas de párrafo encima si se elige un run. Es la pregunta que la
  **métrica de tarea no contesta**: aquélla dice *cuánto* acierta, ésta dice *qué* falla. Se
  distingue de **V6** (galería peor-primero) en que la unidad es la **imagen**, no la ventana, y de
  **V11** en que fija un split entero en vez de una imagen.

  ⚠ **Lo que fija es el DATASET, no el run** — y empezó al revés, lo que la hizo inservible en el
  server real: con el run como selector principal el `select` traía **859 opciones** (medido el
  2026-08-29 en 157.230.221.199). La lista de datasets es corta a propósito: sólo los que traen
  `windows.npz`, o sea los que de verdad viajaron por git (**2 de 18** ese día). Los runs los
  filtra el servidor a los de ese dataset (8), y se **marcan** los que no tienen `best.pt` en vez
  de esconderlos.

  ⚠ **El run es OPCIONAL, y sin él se ven las imágenes igual.** Los pesos de un run no viajan por
  git (`*.pt` está en el `.gitignore` del repo de datos), así que exigir un modelo sería no poder
  mirar nunca el dataset que sí viajó. La única excepción son los runs **`demo-*`**, que sí
  commitean su `best.pt` para que una máquina nueva pueda inferir (desde el 2026-08-30). Las imágenes salen entonces del propio `windows.npz`
  (`/window-datasets/<b>/samples/<i>/image`), que guarda los píxeles **verbatim**. Lo que falta
  —el modelo, la verdad— se **dice en la pantalla**: una rejilla sin cajas y sin aviso se lee como
  «la red no detecta nada», que es la conclusión equivocada.

  ⚠ **Lo que la hace distinta de un panel de caché**: deja **estado propio y commiteado** — qué
  rangos se miraron ya y qué imágenes quedaron marcadas—, así que es una entidad con su artefacto,
  no un cruce recalculable (por eso tiene pantalla, como `Diagnóstico`). El registro lo escribe el
  servidor **al inferir**, no un botón de guardar.
  ⚠ **El detalle ensena las ESQUINAS, la rejilla no** — y es la misma asimetria: alli la unidad
  es la imagen entera para triar, aqui es una imagen sola y grande para diagnosticar. Los puntos de
  la inferencia (post-NMS, y la nube cruda si se pide) van con el **tamano por `score`** en escala
  absoluta 0→1 y se **filtran por ranura**; el rango de score por ranura va en la tabla de numeros,
  porque con una red entrenada los scores se saturan (mediana 0,998 *medido 2026-08-31*) y todos
  los circulos salen iguales. Viajan **a peticion** (`with_detections`), que es lo que evita
  mandarle a un movil las decenas de puntos por imagen de un lote de 60.

  ⚠ En la rejilla la inferencia sale **sola** al elegir el split; en el detalle va **a botón**, y
  la asimetría es deliberada: allí se pueden cambiar el run y el umbral, así que repetir significa
  algo. La imagen se pinta antes y aparte, de modo que un fallo del modelo (sin `best.pt`, por
  ejemplo) deja la página **con la imagen**, no en blanco.

**Nuevas, propias de la geometría foveada:**

- **FG1 — El diagrama de zonas de una red** (en Redes, sin pesos) · *fija: C · varía: —  · mide: la
  geometría*: anillo / penetración / núcleo, dimensiones derivadas, y la huella sobre la imagen
  original (`original_size` dibujado sobre una muestra real de A a escala). Es lo único que una red
  **sin entrenar** puede enseñar de sí misma, y aquí es mucho.
- **FG2 — Contribución por rama** · *fija: E y ventana · varía: la zona · mide: la norma de cada
  rama*: `‖c‖` vs `‖p‖` por zona (¿la periferia aporta, o la fusión la apaga?). **Es la vista que
  contesta la pregunta de investigación del proyecto** — y hoy hay media respuesta sin ella
  ([protocolo.md](../protocolo.md) §2), lo que sube su valor, no lo baja.
- **FG3 — Comparador de vistas foveadas** (en Redes o Ventanas) · *fija: la ventana original ·
  varía: la config de geometría · mide: la vista resultante*: dos configs lado a lado. Barato (no
  hay modelo) y es como se elige qué rangos merecen recorrido.

## Prioridad

1. **F0 (entrada), FG1 (zonas), V3, V14** — verificar el dato, la geometría y el run: el mínimo
   para confiar en el primer entrenamiento.
2. **V6, V7, V8** — la tabla por ventana y lo que se lee de ella.
3. **V12/V13** — cuando exista H.
4. **V1, V2, FG2** — mirar por dentro, por rama.
5. **V4, V5, V16** — sondas finas (V4 exige el diseño de oclusión pre-muestreo, aún sin decidir).

**El estado de construcción de cada vista no se lleva aquí** — lo lleva el estado de
[CLAUDE.md](../../CLAUDE.md). Este catálogo dice qué afirma cada vista, no cuál existe: si también
dijera lo segundo, sería una segunda copia que envejece sola.
