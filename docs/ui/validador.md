# El validador de especificación (`verify_spec`)

Cómo se comprueba, **en código y de forma automática, que las especificaciones de `docs/ui/` se
cumplen**. No es una suite de tests: un test afirma que un comportamiento es correcto; esto afirma
que **una regla declarada se cumple en todo el sistema**, y se alimenta de la regla como dato.

Decisiones tomadas por el usuario (2026-07-27): **A2** (el markdown manda, el spec se extrae de
él) · **C3** (motor híbrido: verbos declarativos + handlers nombrados, y lo que no encaja se
declara) · informe **por regla en cuatro estados** · alcance **completo** (los siete sustratos).

> **Precedente del proyecto**: `scripts\verify_axes.py` (26/26 ejes) y `scripts\verify_ui.py` (12
> pantallas) ya son validadores, no tests. Lo que añade este documento es sacar la regla del código
> del validador y convertirla en su entrada.

---

## 1. Dónde vive la especificación (A2)

**La regla y su comprobación no se pueden separar**: viven en el mismo fichero, pegadas. El motor
parsea `docs/ui/*.md` y extrae los bloques.

````markdown
**U3.1 — La paleta vive en tokens.css y solo ahí, y se valida con script, nunca a ojo.** …

```check U3.1
substrate: fs
kind: no_match_outside
scope: "web/src/**/*.{ts,tsx,css}"
args:
  pattern: "#[0-9a-fA-F]{3,8}\\b"
  allow: ["web/src/theme/tokens.css"]
strength: strong
```
````

**Formato**: bloque fenced con lenguaje `check`, **id de la regla en la línea de apertura**, cuerpo
**YAML** (`pyyaml 6.0.3` verificado en el venv; YAML es JSON válido, así que un bloque puede
escribirse en JSON puro si se prefiere).

### Campos

| Campo | Obligatorio | Qué |
|---|---|---|
| `substrate` | sí | `doc \| css \| ast \| http \| dom \| mixed \| none \| delegated \| same_as` |
| `kind` | si hay sustrato mecánico | verbo declarativo o nombre de handler (§2) |
| `scope` | según el verbo | glob de ficheros, ruta HTTP o ruta de pantalla |
| `args` | según el verbo | parámetros; **aquí es donde la especificación se vuelve más detallada** que la prosa (umbrales, listas, tablas) |
| `strength` | sí | `strong` (afirma la regla) · `weak` (heurística: caza el caso conocido, no todos) |
| `reason` | si `substrate: none` | por qué ninguna herramienta puede decidirlo |
| `target` | si `delegated`/`same_as` | el test que ya lo cubre, o la regla que tiene el check |

### Reglas del propio formato

1. **Toda regla `U*.*` tiene exactamente un bloque `check`** — incluidas las que no se pueden
   comprobar, que llevan `substrate: none` + `reason`. Un id **sin bloque es un fallo del lint**,
   nunca un aprobado: es `ausente ≠ cero` aplicado al validador.
2. Una regla puede llevar **varios `kind`** (lista en `checks:`); su estado es **el peor** de ellos.
3. `same_as` evita duplicar una comprobación entre dos reglas que comprueban lo mismo (caso real:
   U7.5 y U5.7). Dos ids no pueden tener el mismo check literal.
4. `delegated` señala que la regla ya la hace cumplir un test de contrato; el validador comprueba
   **que ese test existe y se llama así**, no la vuelve a ejecutar.
5. El bloque **no repite la regla en palabras**. Si el bloque y la prosa dicen cosas distintas,
   manda la prosa y el bloque es el bug.

---

## 2. El motor (C3)

### Verbos declarativos

Cubren lo repetitivo; añadir una regla de este tipo **no toca código**.

| Verbo | Sustrato | Afirma |
|---|---|---|
| `file_exists` | fs | un fichero/script existe |
| `json_path` | fs | un campo de un JSON vale/existe (`package.json`, `launcher.json`) |
| `must_match` | fs | un patrón aparece ≥ N veces en el ámbito |
| `no_match_outside` | fs/ast | un patrón **solo** aparece en los ficheros permitidos |
| `ast_query` | ast | consulta estructural TS: imports, atributos JSX, llamadas, literales |
| `css_tokens` | css | tokens declarados, paridad claro/oscuro, presencia de familias |
| `http_shape` | http | forma de una respuesta: campos obligatorios, tipos, nullables, tope de filas |
| `http_refuses` | http | una petición imposible devuelve `code` esperado + `hint` no vacío |
| `dom_query` | dom | selector / atributo / texto presente en una ruta viva |
| `dom_absent_text` | dom | palabras prohibidas ausentes del texto visible |
| `catalog_match` | mixed | **una tabla del documento == lo que declara el código** (bidireccional: sin huérfanos por ningún lado) |
| `single_definition` | ast | un vocabulario compartido tiene **una sola** definición literal |

`catalog_match` y `single_definition` son los dos que más compran: el primero convierte las tablas
que ya están escritas en `docs/ui/*.md` en especificación ejecutable; el segundo ataca el modo de
fallo dominante del proyecto (*el mismo dato en dos sitios*).

### Handlers nombrados

Lo que no se deja parametrizar. Se registran por nombre y el bloque los invoca igual (`kind:
palette_contrast`):

`palette_contrast` · `palette_cvd_delta_e` · `number_has_uncertainty` · `null_not_zero` ·
`settle_guard` · `error_hint_propagated` · `color_follows_entity` · `testid_inventory` ·
`spec_lint` · `ports_free`.

### La cláusula que evita el verde falso

- Una regla sin verbo ni handler se informa **`no_verificable`**, jamás `ok`.
- Un `scope` que no casa con ningún fichero, o una ruta HTTP que no existe, es **violación de
  configuración**, no `ok`. (Es la forma más común de que un validador mienta.)
- El validador **no reescribe la regla para que encaje**. Si no encaja, se declara `none` con su
  razón. La especificación manda sobre su herramienta.

---

## 3. El informe: cuatro estados

| Estado | Significa | Cuenta para cobertura |
|---|---|---|
| `ok` | comprobada y se cumple | sí |
| `violada` | comprobada y se incumple, **con fichero:línea, ruta o selector** | sí |
| `no_verificable` | ninguna herramienta puede decidirlo (`substrate: none`) — juicio humano | no |
| `no_aplicable` | el sustrato no estaba disponible en esta corrida (modo estático, backend apagado) **o** el dato que hace falta no existe todavía | no |

Salida a consola en **ASCII** (U8.6: la consola de esta máquina es cp1252 y una `δ` ya mató un
estudio nocturno), con el rollup por tipo:

```
tipo 3 representacion : 11 ok  1 violada  0 no-verif  1 no-aplic   (cobertura 92%)
...
TOTAL 76 reglas : 64 ok  2 violadas  6 no-verificables  4 no-aplicables
cobertura mecanica 87% (fuertes 71%, debiles 16%)
```

- **Código de salida ≠ 0 solo si hay `violada`.** `no_verificable` no es un fallo: es el mapa de lo
  que sigue dependiendo de una persona.
- La cobertura se **calcula**, nunca se mantiene a mano — ni en este documento ni en CLAUDE.md.
- `--json <ruta>` opcional. **El informe no se commitea**: es recomputable, luego es caché
  ([formatos.md](../formatos.md) §3).

---

## 4. Los siete sustratos y qué hace falta para cada uno

| | Sustrato | Herramienta | Disponible |
|---|---|---|---|
| B1 | El propio documento | parser markdown propio | ✅ |
| B2 | Tokens CSS | parser CSS mínimo propio | ✅ |
| B3 | AST del front | **side-car Node** que emite hechos JSON (imports, atributos JSX, literales con posición) usando el compilador de TypeScript | ✅ node v24.17 + `typescript` ya en `web/node_modules` |
| B4 | Forma de las respuestas | `httpx` contra `:8010` | ✅ |
| B5 | DOM vivo | Playwright/Chromium | ✅ (ya lo usa `verify_ui.py`) |
| B6 | Anotaciones `data-*` | B3 + B5 | requiere **tocar los componentes una vez** |
| B7 | Inventario de costuras | B3 + fichero de costuras declarado en el doc | ✅ |

**Nota sobre B3**: el AST se saca con el compilador de TypeScript (Node), no con expresiones
regulares, porque las reglas interesantes son estructurales («este componente define su propia lista
de objetivos»). El side-car emite JSON y Python decide: así el motor sigue siendo uno.

**Nota sobre B6**: es el multiplicador del alcance completo. Los componentes declaran a qué
obedecen (`data-domain="C"`, `data-view="V14"`, `data-fixes/-varies/-measures`,
`data-color-work="diverging"`), y el validador comprueba tres cosas hoy imposibles: que la
declaración exista, que **coincida con el catálogo del documento**, y que el catálogo no tenga
vistas huérfanas. Es lo que hace verificables los tipos 1, 2 y 6.

---

## 5. Dónde vive el código, y sus fronteras

```
tools/speccheck/           el motor (paquete)
  extract.py               docs/ui/*.md  ->  reglas + bloques check
  engine.py                despacho verbo/handler, estados, informe
  verbs/                   un modulo por verbo
  handlers/                un modulo por handler nombrado
  tsfacts.mjs              side-car Node: AST -> hechos JSON
scripts/verify_spec.py     entrada delgada (junto a verify_axes.py / verify_ui.py)
```

- **No importa `fv`.** Lee ficheros y habla HTTP, como `verify_ui.py`. Mantiene limpia la dirección
  de dependencias del contrato ⑦ y permite que el validador corra contra un backend remoto.
- **No toca `data/`, `runs/` ni `sweeps/`** y no entrena nada. Si una comprobación necesita un
  artefacto que no existe, el estado es `no_aplicable` con la razón.
- **Se autoaplica**: `ports_free` comprueba que lo que arrancó quedó cerrado (U7.13), y su salida es
  ASCII (U8.6).

## 6. Modos de corrida

| Modo | Sustratos | Coste esperado | Para |
|---|---|---|---|
| `--static` (por defecto) | B1, B2, B3, B7 | segundos, sin servidores | antes de cada commit |
| `--live` | + B4, B5, B6 | minutos (arranca backend + vite) | antes de cerrar una tarea de UI |
| `--rule U4.2` / `--type 3` | el que toque | — | iterar sobre una regla |
| `--coverage` | ninguno | instantáneo | el cuadro, sin ejecutar comprobaciones |

En `--static`, todo lo que necesite servidores sale **`no_aplicable`**, nunca `ok`.

---

## 7. Fases de construcción

Cada fase deja la herramienta funcionando y **el número de cobertura más alto que la anterior** —
medido por ella misma, no estimado.

| Fase | Qué | Deja |
|---|---|---|
| **0 ✅ (2026-07-27)** | `extract` + `engine` + los cuatro estados + informe + `spec_lint` (B1) + los **76 bloques `check`** + los verbos que no piden dependencias nuevas: `file_exists`, `json_path`, `must_match`, `no_match_outside` y **`catalog_match` con extractores de fichero** | La **cobertura real medida** (§8 bis) y los ocho documentos protegidos contra su propia deriva |
| **1** | B2: `css_tokens`, `palette_contrast`, `palette_cvd_delta_e` | Cierra la deuda declarada (`validate:palette`, que hoy **no existe**) |
| **2** | B3 + B7: `ast_query`, `no_match_outside`, `single_definition` | La regla que ya costó cuatro copias vivas (U4.2) pasa a ser mecánica |
| **3** | B4: `http_shape`, `http_refuses` | Los `code` del backend y los que la UI conoce se casan (U5.1, U5.5) |
| **4** | B6 + B5: anotaciones, `dom_query`, `dom_absent_text`, `catalog_match` completo | Los tipos 1, 2 y 6 dejan de ser solo prosa |

---

## 8. Triaje de las 76 reglas

Instantánea del **2026-07-27**, hecha regla a regla. **No se mantiene**: en cuanto existan los
bloques `check`, la fuente de verdad son ellos y el número lo calcula `--coverage`. Está aquí para
justificar el plan con reparto real, no con una estimación.

`F` = comprobación fuerte · `d` = heurística débil (caza el caso conocido, no todos).

| Tipo | Reglas | Verbo/handler principal | F | d | `none` | `no_aplic.` |
|---|---|---|---|---|---|---|
| 1 estructura | U1.1–U1.5 | `catalog_match` (rutas↔dominios), `ast_query` | 4 | 1 | 0 | 0 |
| 2 vistas | U2.1–U2.5 | `catalog_match` (catálogo↔`data-view`), `dom_query` | 3 | 1 | 1 | 0 |
| 3 representación | U3.1–U3.13 | `css_tokens`, `no_match_outside`, `color_follows_entity` | 9 | 3 | 0 | 1 |
| 4 datos | U4.1–U4.10 | `http_shape`, `single_definition`, `settle_guard` | 7 | 3 | 0 | 0 |
| 5 invariantes | U5.1–U5.9 | `http_refuses`, `catalog_match` (códigos), `null_not_zero` | 7 | 1 | 0 | 0 (1 `delegated`) |
| 6 números | U6.1–U6.13 | `http_shape`, `dom_query`, `number_has_uncertainty` | 8 | 2 | 2 | 1 |
| 7 operación | U7.1–U7.13 | `json_path`, `testid_inventory`, `ports_free` | 7 | 2 | 2 | 1 (1 `same_as`) |
| 8 léxico | U8.1–U8.8 | `dom_absent_text`, `catalog_match` (nav↔tabla) | 6 | 2 | 0 | 0 |
| **Total** | **76** | | **51** | **15** | **5** | **3** |

**Cobertura mecánica prevista: 87 %** (67 % fuerte + 20 % débil). Es más de lo que estimé antes
(80 %), y la diferencia viene casi entera de dos ideas: **provocar los errores por HTTP** en vez de
leer el código que los produce, y las **anotaciones `data-*`** (B6).

Las cinco `none` y las tres `no_aplicable`, nombradas —porque una lista de excepciones sin nombres
es una promesa:

| Regla | Estado | Por qué |
|---|---|---|
| U2.2 «si no puedes decir qué fija, no se construye» | `none` | juicio de diseño; B6 comprueba que *alguien lo declaró*, no que sea cierto |
| U6.12 «una estimación razonada no es una medición» | `none` | epistemología, no sintaxis |
| U6.6 val del ganador sesgado | `none` | comprobable solo como etiqueta obligatoria; se deja a juicio para no fingir rigor |
| U7.10 «reiniciar el backend antes de verificar» | `none` | procedimiento; lo cumple el propio runner en `--live` |
| U7.12 «no se testea el render píxel a píxel» | `none` | regla negativa sobre lo que *no* se hace |
| U6.7 rastro del holdout | `no_aplicable` | **no existe todavía la fuente holdout** (bloqueada por F11) |
| U3.10 banda cortada | `no_aplicable` | necesita un recorrido con réplicas desiguales; si no hay, no se finge |
| U7.8 supervivencia a hibernación | `no_aplicable` | no se provoca una hibernación en una corrida |

Y las dos reglas que **espero que salgan `violada` en la primera corrida**, porque ya sé que se
incumplen: **U3.1** (no existe `validate:palette`) y **U7.9** (`verify_ui.py` recorre las pantallas
con una lista propia, que nadie casa contra las rutas reales de `App.tsx`).

---

## 8 bis. Lo medido en la fase 0 (2026-07-27)

Primera corrida real, `scriptserify_spec.py` en modo estatico:

```
TOTAL  76 reglas : 15 ok  1 violada  5 no-verificables  55 no-aplicables   cobertura 21%
pendientes por construir -> fase 1: 2 | fase 2: 11 | fase 3: 23 | fase 4: 19
```

**21 % hoy, 87 % previsto** cuando estén las cuatro fases: el `no_aplicable` mayoritario es *«el
verbo aún no existe»*, y el informe lo dice con su fase para que el número no se lea como techo.

Lo que la fase 0 ya afirma de verdad (15 reglas, todas con sustancia): las **12 rutas** del
documento casan con `App.tsx` (U1.2); las **10 etiquetas de nav** casan con la tabla (U1.4, y U8.3
por `same_as` — es la comprobación que habría cazado «Barrido por ejes» contra «Estudios»); los
**16 ids de vista** del catálogo no se repiten (U2.4); el front **no recalcula geometría** (U5.4);
`verify_ui.py` **cubre las 12 rutas** (U7.9); ningún `print` con no-ASCII en `fv` (U8.6); ningún
número de resultado cableado en la UI (U6.13); y el contrato ① **delegado** a su test, que existe
(U5.9).

**Dos hallazgos en la primera corrida** — el objetivo de construir esto:

1. **U3.1 `violada`** (predicha): cuatro colores literales fuera de `tokens.css`
   (`MatrixCanvas.tsx:44,45`, `WindowCanvas.tsx:39,41` — fallbacks que duplican `--div-neg`,
   `--div-pos`) y **`npm run validate:palette` sin portar**. Se deja violada a propósito: es la
   deuda declarada, y ahora tiene código de salida.
2. **U7.11 `violada` y ya arreglada**: el inventario de `data-testid` del documento **no tenía
   `runs-table`**. La lista se armó con un `grep` que trató `Runs.tsx` como binario y lo saltó. Es
   justo el modo de fallo del proyecto —*el mismo dato en dos sitios, y solo una copia se
   actualizó*— cazado por la herramienta el día que nació.

**Corrección a §8**: predije que U7.9 saldría `violada`. **No lo está** — `verify_ui.py` sí recorre
las 12 rutas. La predicción estaba razonada, no medida.

**Control (obligatorio, y ejecutado)**: al quitar el bloque `check` de U8.6, el lint aborta con
`U8.6 (docs/ui/8-lexico.md:87): sin bloque check` y salida **2**, sin evaluar nada. Un spec
malformado no produce un verde. El control se cobró solo, además, en vivo: un `git checkout` de mi
parte borró los ocho bloques del fichero léxico y **el lint lo dijo en la corrida siguiente**.

## 9. Lo que este validador no puede afirmar

- Que una vista **sepa** lo que enseña (U2.2): comprueba que la tripleta está declarada y es
  coherente con el catálogo — no que sea la tripleta correcta.
- Que un número sea **correcto**: comprueba que viaja con su `sem` y su `n`, no que la media esté
  bien calculada. Eso es `fv.metrics` y sus tests.
- Que la interfaz sea **buena**. Ninguna de las 76 reglas habla de eso.
- Que una regla **débil** no tenga falsos negativos: por definición caza el caso conocido. Por eso
  `strength` se informa aparte y no se suma a la cobertura fuerte.

## 10. El riesgo que hay que vigilar

**Que el validador se convierta en la especificación.** El día que una regla incómoda se reescriba
para que encaje en un verbo, este documento habrá empeorado el proyecto en vez de mejorarlo. Las
tres protecciones: el markdown manda (A2), `no_verificable` es un estado respetable, y el bloque
`check` tiene prohibido repetir la regla en palabras.
