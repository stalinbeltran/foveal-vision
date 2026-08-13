# Organización de la UI — índice

Cómo se estructura la interfaz aplicando los dominios de [organizacion.md](organizacion.md) —
**ese documento manda**. Proyecto de investigación: la UI no es un panel de control, es el
**instrumento de medida**. Las pantallas de análisis son el producto.

Requisito del usuario, literal: *en todo momento debe ser posible verificar los objetos creados
— fuentes, datasets, redes, runs, recorridos, análisis — y revisar cada nn, grupos de
parámetros, y probar los resultados.*

> **Este fichero es el índice, no la especificación.** Las reglas viven en `docs/ui/`, **una por
> tipo**, y cada una en un solo sitio: repetirlas aquí sería exactamente el modo de fallo que este
> proyecto tiene registrado (*el mismo dato en dos sitios*, CLAUDE.md).

---

## Los ocho tipos de especificación

| # | Tipo | Qué fija | Cómo se hace cumplir |
|---|---|---|---|
| 1 | [Estructural](ui/1-estructura.md) | Cuántas pantallas hay, de qué dominio es cada una, qué cabe dentro | prosa |
| 2 | [Epistémica (vistas)](ui/2-vistas.md) | La tripleta `(fija, varía, mide)` y el catálogo de vistas | prosa |
| 3 | [De representación](ui/3-representacion.md) | Color, ejes, forma: el mapeo dato → píxel | validador (⚠ sin portar) |
| 4 | [De contrato de datos](ui/4-datos.md) | Qué pide y qué recibe la UI, y qué **no** puede saber por su cuenta | parcial (HTTP + tests) |
| 5 | [De invariante de dominio](ui/5-invariantes.md) | Qué bloquea, avisa o deriva un formulario | ejecutable (`fv.validation`) |
| 6 | [Metodológica](ui/6-numeros.md) | Qué número tiene derecho a enseñarse, y con qué salvedades | prosa (+ tests parciales) |
| 7 | [Operativa](ui/7-operacion.md) | Arranque, estado recordado, resistencia, «verificado» | ejecutable (Playwright) |
| 8 | [Léxica](ui/8-lexico.md) | Las palabras en pantalla | prosa |

**Precedencia**: organizacion.md → api.md / protocolo.md / formatos.md / glosario.md → `docs/ui/`.
Una regla de UI que contradiga a su fuente es un error de la regla de UI.

**Cómo se comprueba que estas reglas se cumplen**: [ui/validador.md](ui/validador.md) — el spec se
extrae de estos mismos ficheros (bloques `check` junto a cada regla) y `scripts\verify_spec.py` las
evalúa e informa **por regla en cuatro estados**. Una regla sin bloque `check` es un fallo del lint,
nunca un aprobado.

**Las dos reglas fundacionales**, que estos ocho tipos desarrollan:

- **Una pantalla, un dominio** → [1-estructura.md](ui/1-estructura.md) U1.1.
- **Toda vista de análisis declara `(qué fija, qué varía, qué mide)`** →
  [2-vistas.md](ui/2-vistas.md) U2.1.

## Dónde mirar según lo que vayas a tocar

| Vas a… | Lee |
|---|---|
| añadir una pantalla o mover algo de sitio | 1 |
| añadir una gráfica, un mapa o una sonda | 2, luego 3 |
| pintar algo nuevo (color, escala, leyenda) | 3 |
| añadir un campo, una ruta o un poll | 4 |
| añadir un formulario, un botón que crea o borra | 5 |
| enseñar un número nuevo | 6 |
| tocar arranque, `localStorage` o el verificador | 7 |
| escribir cualquier texto visible | 8 |

**Los tres tipos que solo viven en prosa (1, 2 y 6)** son los que se degradan sin que nada se ponga
rojo, y coinciden con los que producen el fallo característico de este dominio: *un número plausible
que mide otra cosa*.

## Dónde fue lo que había aquí

| Antes en `ui.md` | Ahora |
|---|---|
| §0 Regla 1 · §1 mapa de pantallas · §2 pantalla a pantalla | [ui/1-estructura.md](ui/1-estructura.md) |
| §0 Regla 2 · §3 catálogo de vistas · §4 prioridad | [ui/2-vistas.md](ui/2-vistas.md) |
| §0 «Librerías y color» | [ui/3-representacion.md](ui/3-representacion.md) |
| §1 «Estado de UI recordado» | [ui/7-operacion.md](ui/7-operacion.md) |

Las vistas conservan su nombre (`F0`, `V3`, `FG1`…): una referencia externa a «ui.md V19» se
resuelve hoy en [ui/2-vistas.md](ui/2-vistas.md).
