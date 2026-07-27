# UI · Tipo 3 — Especificación de representación (dato → píxel)

> **Qué decide**: la codificación visual — color, ejes, forma, y con qué se dibuja cada cosa.
> **Qué NO decide**: qué afirma la vista (→ [2-vistas.md](2-vistas.md)); quién decide el tipo de
> escala, que **viaja en el payload** (→ [4-datos.md](4-datos.md) U4.1).
> **Cómo se hace cumplir**: la paleta, con **validador ejecutable**; el resto, prosa + Playwright
> (que ve errores de consola, no un doble eje). Ver «Cumplimiento» abajo: hoy el validador **no
> está portado**.

---

## Las reglas

**U3.1 — La paleta vive en [web/src/theme/tokens.css](../../web/src/theme/tokens.css) y solo ahí, y
se valida con script, nunca a ojo.** Un color escrito en un componente es una segunda definición.

```check U3.1
substrate: fs
kind: no_match_outside
scope: "web/src/**/*.{ts,tsx,css}"
args:
  pattern: "#[0-9a-fA-F]{3,8}\\b"
  allow: ["web/src/theme/tokens.css"]
strength: strong
```

```check U3.1
substrate: fs
kind: json_path
args:
  file: "web/package.json"
  path: "scripts.validate:palette"
strength: strong
```

**U3.2 — Datos con signo → rampa divergente centrada en 0.** Pesos, gradientes, deconvolución.
Esconder el signo esconde qué excita y qué inhibe.

```check U3.2
substrate: http
kind: http_shape
scope: "/runs/{run}/kernels"
args:
  requires: ["color_work"]
  when: {signed: true, color_work: "diverging"}
strength: strong
```

**U3.3 — Magnitudes → secuencial de una tinta.** Activaciones sin signo, conteos, errores.

```check U3.3
substrate: http
kind: http_shape
scope: "/runs/{run}/feature-maps"
args:
  requires: ["color_work"]
  when: {signed: false, color_work: "sequential"}
strength: strong
```

**U3.4 — Jamás doble eje.** `loss`, `f1` y píxeles son escalas distintas: **small multiples con eje
x alineado**. Un doble eje deja poner cualquier par de curvas a coincidir.

```check U3.4
substrate: ast
kind: ast_query
scope: "web/src/**/*.tsx"
args:
  forbid_identifiers: ["y2", "yRight", "secondaryAxis", "rightAxis"]
strength: weak
```

**U3.5 — El trabajo de color lo declara el payload; el cliente no lo adivina.** El front no puede
saber si mira un peso con signo o una activación: lo dice `matrixview`
(`sequential | diverging`). Adivinarlo por el rango observado falla justo cuando todos los pesos
salen positivos por casualidad.

```check U3.5
substrate: http
kind: http_shape
scope: "matrixview:*"
args:
  requires: ["color_work", "min", "max", "mean", "truncated"]
strength: strong
```

**U3.6 — Todo mapa de calor tiene su tabla de números.** Gemela accesible y, en la práctica, la
mejor vista de depuración que existe: un mapa bonito no dice si el centro vale 0,0 o 0,0001.

```check U3.6
substrate: dom
kind: dom_query
scope: "*"
args:
  selector: "canvas[data-matrix]"
  sibling_required: "[data-numbers-twin]"
strength: strong
```

**U3.7 — El color sigue a la entidad, nunca al rank.** En overlays multi-run el color se asigna por
índice de trial u orden de grupo, así ocultar o reordenar **no repinta a los demás**. Colorear por
posición hace que el gráfico cambie de significado cuando cambia el ranking.

```check U3.7
substrate: dom
kind: color_follows_entity
scope: "/sweeps"
args:
  toggle: "[data-testid=sweep-legend] input"
  assert: "los colores de las series restantes no cambian"
strength: strong
```

**U3.8 — La identidad nunca es solo color.** Siempre leyenda + etiqueta (y énfasis al pasar el
ratón). La rampa categórica tiene 8 tintas en orden fijo validado para daltonismo; a partir de 8
**cicla**, y por eso el color no puede ser el único portador de identidad.

```check U3.8
substrate: dom
kind: dom_query
scope: "/sweeps"
args:
  selector: "[data-testid=sweep-legend] label"
  min_count: 1
  assert_text_not_empty: true
strength: strong
```

**U3.9 — Las ranuras de esquina son entidades fijas**: el mismo color de TL en **toda** vista
(`--corner-tl`…`--corner-bl`), y las etiquetas siempre en tinta de texto. El orden de esquinas lo
sirve el API (U4.2), no el front.

```check U3.9
substrate: css
kind: css_tokens
args:
  required: ["--corner-tl", "--corner-tr", "--corner-br", "--corner-bl"]
  both_themes: true
strength: strong
```

**U3.10 — Una banda de agregación se corta donde faltan réplicas, y se dice dónde.** Promediar las
réplicas presentes en cada época hace que un grupo con réplicas a distinta altura **finja
converger** al final. Si la banda se acorta, la vista lo declara (`band-cut`).

```check U3.10
substrate: dom
kind: dom_query
scope: "/sweeps"
args:
  selector: "[data-testid=band-cut]"
  when: "hay replicas desiguales; si no las hay, no_aplicable"
strength: strong
```

**U3.11 — Nada se pinta como cero por no tener dato.** Un hueco se dibuja como hueco: la regla
`ausente ≠ cero` es de datos ([formatos.md](../formatos.md) §1) pero se cobra en píxeles — un 0
pintado en una rampa es indistinguible de un 0 medido. Ver [5-invariantes.md](5-invariantes.md)
U5.3.

```check U3.11
substrate: http
kind: null_not_zero
scope: "/runs/{run}/task-score"
args:
  fields: ["mean_iou"]
strength: strong
```

**U3.12 — Cada cosa con su herramienta**: gráficas con ejes y leyendas (curvas, scatter,
paralelas) → librería de gráficas; **matrices densas** (kernels, feature maps, vistas de entrada) →
**canvas a mano**, reutilizando el patrón `MatrixCanvas`/`LayerMaps` del hermano vía `matrixview`;
meters, overlays y badges → **HTML/CSS**. Un heatmap de 10⁴ celdas en SVG es un fallo de
rendimiento con forma de decisión estética.

```check U3.12
substrate: ast
kind: ast_query
scope: "web/src/components/*.tsx"
args:
  assert: "las matrices densas se pintan en canvas, no en <rect> por celda"
strength: weak
```

**U3.13 — Claro y oscuro son dos superficies, no una con filtro.** Cada token tiene su valor en
`prefers-color-scheme: dark`, y la validación de contraste/CVD se corre en las dos.

```check U3.13
substrate: css
kind: css_tokens
args:
  theme_parity: true
strength: strong
```

## Cumplimiento (verificado 2026-07-27)

| Regla | Estado |
|---|---|
| U3.1 tokens únicos | ✅ `tokens.css` tiene la paleta completa, claro + oscuro, con las 8 series y las 4 ranuras de esquina |
| U3.1 **validador** | ⚠ **NO EXISTE**. `web/package.json` solo tiene `dev`, `build`, `preview`: **`npm run validate:palette` no está portado** del hermano, pese a que [ui.md](../ui.md) lo daba por hecho y [tests.md](../tests.md) §5 lo cita como *la única* comprobación ejecutable de la UI. Hoy U3.1 es prosa como las demás |
| U3.12 librería de gráficas | ⚠ **No hay Observable Plot** (ni ninguna dependencia de gráficas): [LineChart.tsx](../../web/src/components/LineChart.tsx) es **SVG a mano**, y `MatrixCanvas`/`WindowCanvas` son canvas. La regla se cumple en el reparto (SVG para ejes, canvas para matrices); lo que no se cumple es la librería nombrada. **Manda el reparto**, no el nombre |
| U3.7/U3.8 | ✅ implementadas en [SweepCurves.tsx](../../web/src/components/SweepCurves.tsx) (color por identidad, leyenda de casillas) |
| U3.10 | ✅ `data-testid=band-cut` |

**Deuda declarada**: portar `validate:palette`. Mientras no exista, la afirmación «la paleta se
valida, no se mira» es incomprobable — exactamente el tipo de promesa que este proyecto exige
convertir en rastro (ver la decisión F14 como precedente).
