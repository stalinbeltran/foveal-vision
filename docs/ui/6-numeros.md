# UI · Tipo 6 — Especificación metodológica (qué número tiene derecho a enseñarse)

> **Qué decide**: con qué incertidumbre, qué procedencia y qué salvedades sale a pantalla cada
> número — y cuál no sale.
> **Qué NO decide**: cómo se calcula (eso es `fv.metrics`, `fv.task`, `fv.sweeps.winner`) ni cuándo
> un resultado es creíble (→ [protocolo.md](../protocolo.md), que **manda** sobre este documento).
> **Cómo se hace cumplir**: **prosa**, con apoyo parcial de tests (el `sem`, el `tie_reason`, el
> `n`). Es el tipo que produce el fallo característico del proyecto: **un número plausible que mide
> otra cosa** — no revienta, no sale en consola, y lo encuentra el usuario.

---

## Las reglas

**U6.1 — Nunca un número suelto.** Todo número que se compara sale con su dispersión y su tamaño de
muestra: media ± `sd`/`sem` y `n`. Un run aislado es una anécdota.

```check U6.1
substrate: http
kind: http_shape
scope: "/runs/{run}/task-score"
args:
  requires: ["macro", "macro.sem", "macro.sd", "n_images"]
strength: strong
```

**U6.2 — Cuando la muestra es pequeña, se dice en la pantalla.** Regla única (`n < 100` imágenes),
en **las tres superficies** (UI y los dos CLIs), desde una sola definición. Hoy es el límite real
del proyecto: sd por imagen 0,4148 sobre 20 imágenes de val → **±0,093 por run**, más ruido que las
diferencias que se quieren distinguir.

```check U6.2
substrate: ast
kind: ast_query
scope: "web/src/**/*"
args:
  forbid_numeric_comparison: ["images\\s*<\\s*100", "n_images\\s*<\\s*100"]
strength: strong
```

**U6.3 — El empate se declara; no se rompe en silencio.** Si varios puntos caben en la banda
(δ = 1-SE de las semillas del mejor), la vista dice **«empatados»** y por qué (`tie`, `tie_reason`
en palabras). *«Ganó el primero»* cuando los seis empatan es la forma más cara de mentir con una
tabla ordenada.

```check U6.3
substrate: http
kind: http_shape
scope: "/sweeps/{name}/winner"
args:
  requires: ["tie", "tie_reason", "tie_delta"]
strength: strong
```

**U6.4 — Se enseña el número del checkpoint, y se dice de qué época sale.** `best.pt` es lo que
sobrevive y lo que cargan Diagnóstico y Predecir; rankear por la última época era otra métrica
(épocas distintas en el **63 %** de los runs de un recorrido real, y **cambiaba el ganador**). La
columna «última» va aparte, nunca en su lugar.

```check U6.4
substrate: http
kind: http_shape
scope: "/sweeps/{name}/trials"
args:
  requires: ["epoch", "last_epoch"]
strength: strong
```

**U6.5 — Si el `monitor` no es el `objective`, se avisa.** Son dos preguntas distintas —qué eligió
`best.pt` y con qué se rankea— y coincidir es una coincidencia, no un invariante.

```check U6.5
substrate: dom
kind: dom_query
scope: "/sweeps"
args:
  selector: "[data-testid=monitor-mismatch]"
  when: "monitor != objective"
strength: strong
```

**U6.6 — El val del ganador está sesgado al alza y no se reporta como resultado.** El val hace dos
trabajos (elegir `best.pt` y rankear el recorrido). El número que se reporta es el del **holdout**,
una vez, al final, y solo del ganador.

```check U6.6
substrate: none
reason: "que el val del ganador no se reporte como resultado es una decision de quien
  escribe el informe, no una propiedad de la pantalla"
```

**U6.7 — Mirar el holdout deja rastro visible.** Cada medición contra un holdout anexa una línea a
`runs/<run>/holdout.jsonl` **también cuando el número sale de caché** —ese era el vistazo
invisible— y la UI enseña `holdout_touches` **en ámbar** sobre el bloque. Append-only: **registra
miradas, nunca bloquea una**. Sin esto, «el holdout se mira una vez» es una promesa incomprobable.

```check U6.7
substrate: http
kind: http_shape
scope: "/runs/{run}/task-score?window_dataset={holdout}"
args:
  requires: ["holdout_touches"]
  assert: "dos llamadas -> dos lineas en holdout.jsonl, la segunda from_cache"
strength: strong
```

**U6.8 — La métrica de tarea informa del ganador; no elige entre puntos.** No entra en `OBJECTIVES`,
no se calcula por época, no la dispara ningún sondeo: **botón explícito**, y solo para el punto
sugerido y el mejor. El proxy de ventana ordena igual en ejes de D (+0,956) y de C (+1,000).

```check U6.8
substrate: same_as
target: U4.7
reason: "es la misma comprobacion: la metrica de tarea no entra en ningun sondeo"
```

**U6.9 — Una estimación dice de qué se estimó, o no se da.** Entrenar estima el coste con los
`seconds` de runs **comparables** (misma huella de B, misma red); si no hay comparables, **lo dice**
— no inventa un número.

```check U6.9
substrate: dom
kind: dom_query
scope: "/train"
args:
  selector: "[data-testid=compat]"
  assert_text_when_no_comparables: "sin runs comparables"
strength: strong
```

**U6.10 — Todo número declara su procedencia cuando no es la de siempre**: `objective_overridden`
cuando un recorrido se releyó con otro proxy, `from_cache`, `n_seeds`, los puntos descartados por
geometría inválida, los runs sin checkpoint descontados. El payload lo trae ([4-datos.md](4-datos.md)
U4.10); la UI lo enseña.

```check U6.10
substrate: http
kind: http_shape
scope: "/sweeps/{name}/trials"
args:
  requires_when_present: ["objective_overridden", "discarded_points"]
strength: strong
```

**U6.11 — La unidad se declara siempre.** El presupuesto de un recorrido, en épocas **o** en
segundos (con redes de coste distinto, la poda por tiempo favorece a las pequeñas sin decirlo); los
errores de posición, en **píxeles de la imagen original**, nunca en celdas de la vista — o barrer la
geometría cambia la regla de medir a la vez que el modelo.

```check U6.11
substrate: http
kind: http_shape
scope: "/sweeps/{name}"
args:
  requires: ["budget.unit"]
strength: strong
```

**U6.12 — Una estimación razonada no es una medición, y la UI no debe hacerla pasar por tal.**
Precedente: la sd «conservadora» de 0,372 **subió** a 0,4148 al medirla. Si un número de pantalla
sale de un supuesto, se etiqueta como supuesto.

```check U6.12
substrate: none
reason: "distinguir una estimacion razonada de una medicion es epistemologia, no sintaxis"
```

**U6.13 — Los resultados de investigación no viven en la UI ni en los tests.** Van fechados y con
sus salvedades a [protocolo.md](../protocolo.md) §5. La pantalla enseña la medición de hoy; el
documento guarda la conclusión y su contexto.
```check U6.13
substrate: fs
kind: no_match_outside
scope: "web/src/**/*.{ts,tsx}"
args:
  pattern: "0\\.6448|0\\.4148|0\\.5353|0\\.956"
  allow: []
strength: weak
```

