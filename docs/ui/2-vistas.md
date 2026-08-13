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
| V11 | Etapas del pipeline | E, imagen | la etapa | qué se pierde y dónde | Crudas pre-NMS / esquinas post-NMS / cajas TL→BR, conmutables — sin la cruda, «el párrafo salió mal» no es diagnosticable |
| V12/V13 | Pareto / paralelas | B | el punto (C/D) | el objetivo | **Los ejes de geometría entran**: `d`, `k_center`… en paralelas; color por el eje barrido (magnitud continua → rampa secuencial) |
| V14 | Curvas | B,C,D | la época | loss y métricas | Small multiples, eje x alineado. En Recorridos, N runs superpuestos (líneas por run / media ± banda) |
| V16 | Deconvolución | E, ventana | el filtro | qué píxeles lo activaron | Gradiente puro, siempre divergente ±0; gana valor de la capa 2 en adelante; `silent` con palabras cuando un filtro no dispara |
| V18 | Evidencia disponible | E, split, umbral | cuánto del párrafo cabe en la ventana | detección y posición **por separado**, por banda de evidencia | Vuelve con las cabezas de esquina (C9). `corner_evidence` congelada contra la ventana etiquetada (F1b). **Es el criterio de éxito del primer experimento**: ¿la periferia baja el `err_px` de la banda ciega sin dañar la visible? |

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
