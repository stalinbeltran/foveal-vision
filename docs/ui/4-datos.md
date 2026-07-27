# UI · Tipo 4 — Especificación de contrato de datos (qué pide y qué recibe la UI)

> **Qué decide**: qué puede pedir el front, qué le llega, y —sobre todo— **qué no tiene derecho a
> saber por su cuenta**.
> **Qué NO decide**: la forma del API, que es de [api.md](../api.md) R1–R7. Aquí solo vive la
> **obligación del cliente**; si una regla se contradice con api.md, manda api.md.
> **Cómo se hace cumplir**: parcialmente ejecutable — los errores tipados se prueban por HTTP y en
> `tests/test_contracts.py`; la ausencia de copias vivas en el front, **no**: se audita a mano.

---

## Las reglas

**U4.1 — El cliente no adivina: el payload lo declara.** Todo lo que el front necesita para
interpretar unos números viaja con ellos:

| Qué | Por qué no se puede inferir |
|---|---|
| `sequential \| diverging` | el signo puede no aparecer en la muestra visible |
| `branch: center \| periph`, `region` | pintar un filtro periférico como si viera la imagen original **miente sobre la escala** |
| `corner_order` | el orden de las 4 esquinas es del backend; asumirlo desalinea etiquetas sin fallar |
| los `knobs` con que se calculó | con sliders en vivo **las respuestas llegan desordenadas** |
| `min/max/mean` y `truncated` | una matriz recortada dibujada como completa es una mentira silenciosa |

**U4.2 — Una definición y dos lectores; jamás dos definiciones.** El front **no define
vocabularios**: los defaults de C, los ejes barribles, los objetivos, el orden de esquinas, los
campos que fijan `window_size` — todos salen del API (`/networks`, `/sweeps/axes`, …), que los sirve
desde su única definición. Es la regla que costó cuatro copias vivas, **dos de ellas ya
divergidas** (2026-07-26). Corolario operativo: **antes de cambiar la forma o el significado de un
campo compartido, buscar todos sus lectores.**

**U4.3 — Los agregados se calculan en el servidor.** El navegador nunca recibe 10⁵ filas; las
tablas van filtradas y paginadas con `limit` acotado por la ruta. Una media calculada en el front
es una segunda definición de una métrica (U4.2).

**U4.4 — Polling incremental, nunca historial completo.** Las métricas se piden con `?since=N`. Las
listas vivas se refrescan cada ~3 s; la carga pesada se gatea con un booleano estable para no
recomputar en cada pasada.

**U4.5 — Un dato solo se cachea cuando llega junto a su estado terminal.** La trampa medida
(2026-07-26): se cacheaba la curva de un run en cuanto el estado leía `done` **y había algo
cacheado**, pero ese algo venía del sondeo **anterior**, tomado mientras el run entrenaba — se
perdían las épocas de esa ventana (3 s normalmente, hasta un minuto con la pestaña de fondo, más
tras hibernar). Un run se «settle» **solo** al traerlo *con* el estado ya terminal.

**U4.6 — El `code` del error es contrato; el `hint` es parte de la respuesta, no adorno.** La UI
reacciona al `code` (no al texto), y muestra `message` + `hint`. Tragarse el `hint` convierte una
negativa útil en un callejón — ver [5-invariantes.md](5-invariantes.md) U5.2.

**U4.7 — Lo que cuesta, se pide a mano.** Nada que dispare inferencia de imagen completa entra en un
sondeo periódico: la métrica de tarea se pide **con un botón** (U6.8). Un poll de 3 s sobre algo que
cuesta 0,6 s de CPU por run es un bucle de calor, no una UI viva.

**U4.8 — `/ui-state` es un blob opaco, no una segunda fuente de verdad.** Las pantallas siguen
leyendo el A–I real del API. Ver [7-operacion.md](7-operacion.md) U7.2–U7.3.

**U4.9 — Las rutas se resuelven dentro del dominio.** No existe `GET /image?path=`: allowlist de
raíces (403 fuera) y CORS cerrado al origen del front. **No es teórico**: este API acabará
corriendo en un server con GPU.

**U4.10 — Si el número tiene una procedencia rara, el payload la declara y la UI la enseña.**
Ejemplos vivos: `objective_overridden` (un recorrido releído con otro proxy), `from_cache`,
`holdout_touches`, `n_seeds`, el conteo de puntos descartados por geometría inválida. La regla
general está en [6-numeros.md](6-numeros.md); aquí lo que importa es que **viaja en el payload**,
no que el front lo deduzca.

## Cumplimiento (verificado 2026-07-27)

- ✅ [web/src/api.ts](../../web/src/api.ts) tipa `{code, message, hint}` y `ErrorBox` pinta el
  `hint` como línea de arreglo (`→ …`).
- ✅ U4.2: tras el barrido del 2026-07-26 el front no conserva copias de los vocabularios de C/H.
  **No hay test que lo impida volver** — es auditoría manual, y por eso la regla se escribe aquí.
- ✅ U4.5: implementado en `Sweeps.tsx`, verificado en vivo con control (6/6 épocas con el arreglo,
  5/6 con el guard viejo).
