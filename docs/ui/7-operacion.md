# UI · Tipo 7 — Especificación operativa (cómo corre, qué recuerda, cuándo está entregada)

> **Qué decide**: arranque y puertos, estado recordado en el cliente, resistencia a fallos de
> pantalla, y el criterio de «verificado».
> **Qué NO decide**: nada de contenido. Es el único de los ocho tipos que no habla de qué se
> enseña.
> **Cómo se hace cumplir**: **ejecutable casi entero** — `scripts\verify_ui.py` (Playwright), el
> esquema de `launcher.json`, y el chequeo de puertos al terminar.

---

## Arranque

**U7.1 — El registro en el launcher tiene esquema exacto y cerrado.** Está en
[launcher.md](../launcher.md) — no se inventan campos. Backend `:8010` (evita el `:8000` del
hermano), front Vite `:5173`.

```check U7.1
substrate: fs
kind: json_path
args:
  file: "launcher.json"
  path: "processes.0.cmd"
strength: strong
```

**U7.2 — El front no adivina la URL del backend**: el proxy de `/api` lee `FV_API_URL`.
⚠ `vite.config.ts` fija `port: 5173` con `strictPort: true` y **5173 está en la allowlist de CORS
del backend**: cambiar el puerto del front sin tocar la allowlist da un fallo que parece de red y
es de configuración.

```check U7.2
substrate: fs
kind: must_match
scope: "web/vite.config.ts"
args:
  pattern: "FV_API_URL"
  min: 1
strength: strong
```

## Estado recordado

**U7.3 — Lo recordado son defaults, nunca fuente de verdad.** Filtros y valores de formulario se
recuerdan por navegador (`localStorage`, namespace `fv.ui.`, hook `usePersistedState`). Las
pantallas siguen leyendo el A–I real del API.

```check U7.3
substrate: ast
kind: ast_query
scope: "web/src/**/*"
args:
  string_only_in:
    value: "fv.ui."
    files: ["web/src/uiState.ts"]
  require_any_identifier: []
strength: weak
```

**U7.4 — Un valor recordado cuyo tipo no encaja con el default se descarta.** No es una preferencia:
es **deriva de esquema**. El caso medido: `studies.delta` pasó de número a cadena, el `0` recordado
llegó a `delta.trim()` y **dejó Estudios en blanco**. El default es además la migración correcta
(el valor viejo *era* el default viejo).

```check U7.4
substrate: dom
kind: dom_query
scope: "/studies"
args:
  seed_localstorage: {"studies.delta": 0}
  assert_no: "[data-testid=screen-error]"
strength: strong
```

**U7.5 — Un nombre recordado se usa solo si sigue existiendo.** ⚠ **Corrección de
[ui.md](../ui.md)**, que decía *«un nombre de run/recorrido no se recuerda (es de un solo uso)»*: sí
se recuerda (`diag.run`, `predict.run`), y lo que lo hace seguro es el gateo por pertenencia a la
lista viva ([5-invariantes.md](5-invariantes.md) U5.7). Manda el código; la frase vieja describía un
diseño que no se construyó.

```check U7.5
substrate: same_as
target: U5.7
reason: "una regla, un check: U5.7 es la comprobacion; esta es su correccion historica"
```

**U7.6 — Guardar / Cargar sesión hace viajar una sesión de trabajo.** Vuelca lo recordado a un JSON
**comiteable** (`state/ui-state.json`, `PUT /ui-state`) para llevarlo al server con GPU, y lo trae
de vuelta. Es conveniencia, con tope de tamaño (U4.8).

```check U7.6
substrate: http
kind: http_shape
scope: "GET /ui-state"
args:
  requires: []
strength: weak
```

## Resistencia

**U7.7 — Error boundary por ruta.** Una pantalla que revienta **no borra la app**: muestra la razón
y ofrece olvidar las preferencias. **Una página en blanco es el fallo silencioso definitivo** —
ningún test unitario la ve.

```check U7.7
substrate: dom
kind: dom_query
scope: "*"
args:
  assert_no: "[data-testid=screen-error]"
strength: strong
```

**U7.8 — La UI sobrevive a lo que el backend no controla**: hibernación, pestaña de fondo
estrangulada, reinicio del API. De ahí U4.5 (settle con estado terminal) y la reconciliación de
`running` muerto en el servidor.

```check U7.8
substrate: none
reason: "no se provoca una hibernacion en una corrida; la mitad del servidor (reconcile)
  la cubren los tests de contrato"
```

## Verificación y cierre

**U7.9 — La UI se verifica mirándola, no razonándola.** `scripts\verify_ui.py` recorre las **14
pantallas/interacciones** con Playwright/Chromium, **falla ante cualquier error de consola o de
página**, hace clicks reales (no solo `goto`) y deja capturas en `data/ui-shots/`. Hay Chromium en
esta máquina: **no se entrega UI diciendo «no puedo verlo»**.

```check U7.9
substrate: fs
kind: catalog_match
args:
  left: app_routes
  right: verify_ui_routes
  normalize: route
  mode: covers
strength: strong
```

**U7.10 — Antes de verificar, se reinicia el backend.** Un server viejo da 404 engañosos sobre
rutas nuevas y hace perder la tarde persiguiendo un bug de front que no existe.

```check U7.10
substrate: none
reason: "procedimiento, no propiedad: lo cumple el propio runner en --live"
```

**U7.11 — Los `data-testid` son contrato con el verificador**, no adorno: renombrar uno rompe la
verificación en silencio (el selector expira por timeout y parece lentitud). Los vivos hoy:
`sources-table`, `wds-table`, `window-grid`, `networks-table`, `validate-panel`, `zone-diagram`,
`recipes-table`, `monitor-select`, `optimizer-select`, `scheduler-select`, `update-btn`,
`overwrite-confirm`, `compat`, `sweeps-table`, `trials-table`, `sweep-curves`, `sweep-legend`,
`band-cut`, `sweep-winner`, `winner-verdict`, `winner-table`, `monitor-mismatch`, `studies-table`,
`study-detail`, `study-plan`, `study-axes`, `axis-select`, `base-nn`, `advance-btn`, `confirm-box`,
`task-score`, `task-row`,
`task-split`, `task-dataset`, `task-small-sample`, `task-holdout-warn`, `task-holdout-touches`,
`diag-summary`, `gallery`, `nine-block`, `probes`, `predict-stage`, `predict-numbers`,
`runs-table`, `session-bar`, `screen-error`, `review-grid`, `review-detail`.

```check U7.11
substrate: fs
kind: catalog_match
args:
  left: doc_testids
  right: code_testids
strength: strong
```

**U7.12 — No se testea el render píxel a píxel** ([tests.md](../tests.md) §5) — salvo la paleta, que
debería tener validador ejecutable. ⚠ Hoy **no lo tiene**: ver
[3-representacion.md](3-representacion.md), «Cumplimiento».

```check U7.12
substrate: fs
kind: no_match_outside
scope: "tests/**/*.py"
args:
  pattern: "screenshot.*assert|image_snapshot|toMatchImageSnapshot"
  allow: []
strength: weak
```

**U7.13 — Probar ejecutando, sí; dejarlo corriendo, no.** Lanzar `fv-api`, `npm run dev`,
entrenamientos o Playwright para verificar está **siempre permitido** y no hace falta pedirlo. Al
terminar la tarea **se cierran todos** y se comprueba que `:8010` y `:5173` quedan libres: el usuario
prueba a mano después, y un server viejo vivo le ocupa el puerto o le contesta con rutas obsoletas.
Matar hijos antes que padres (vite antes de `npm run dev`; el `fv-api` que escucha antes de su
lanzador) y filtrar por ruta — en esta máquina hay pythons ajenos al proyecto.
```check U7.13
substrate: mixed
kind: ports_free
args:
  ports: [8010, 5173]
  when: "al terminar una corrida --live que arranco servidores"
strength: strong
```

