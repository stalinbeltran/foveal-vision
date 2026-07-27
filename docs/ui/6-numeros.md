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

**U6.2 — Cuando la muestra es pequeña, se dice en la pantalla.** Regla única (`n < 100` imágenes),
en **las tres superficies** (UI y los dos CLIs), desde una sola definición. Hoy es el límite real
del proyecto: sd por imagen 0,4148 sobre 20 imágenes de val → **±0,093 por run**, más ruido que las
diferencias que se quieren distinguir.

**U6.3 — El empate se declara; no se rompe en silencio.** Si varios puntos caben en la banda
(δ = 1-SE de las semillas del mejor), la vista dice **«empatados»** y por qué (`tie`, `tie_reason`
en palabras). *«Ganó el primero»* cuando los seis empatan es la forma más cara de mentir con una
tabla ordenada.

**U6.4 — Se enseña el número del checkpoint, y se dice de qué época sale.** `best.pt` es lo que
sobrevive y lo que cargan Diagnóstico y Predecir; rankear por la última época era otra métrica
(épocas distintas en el **63 %** de los runs de un recorrido real, y **cambiaba el ganador**). La
columna «última» va aparte, nunca en su lugar.

**U6.5 — Si el `monitor` no es el `objective`, se avisa.** Son dos preguntas distintas —qué eligió
`best.pt` y con qué se rankea— y coincidir es una coincidencia, no un invariante.

**U6.6 — El val del ganador está sesgado al alza y no se reporta como resultado.** El val hace dos
trabajos (elegir `best.pt` y rankear el recorrido). El número que se reporta es el del **holdout**,
una vez, al final, y solo del ganador.

**U6.7 — Mirar el holdout deja rastro visible.** Cada medición contra un holdout anexa una línea a
`runs/<run>/holdout.jsonl` **también cuando el número sale de caché** —ese era el vistazo
invisible— y la UI enseña `holdout_touches` **en ámbar** sobre el bloque. Append-only: **registra
miradas, nunca bloquea una**. Sin esto, «el holdout se mira una vez» es una promesa incomprobable.

**U6.8 — La métrica de tarea informa del ganador; no elige entre puntos.** No entra en `OBJECTIVES`,
no se calcula por época, no la dispara ningún sondeo: **botón explícito**, y solo para el punto
sugerido y el mejor. El proxy de ventana ordena igual en ejes de D (+0,956) y de C (+1,000).

**U6.9 — Una estimación dice de qué se estimó, o no se da.** Entrenar estima el coste con los
`seconds` de runs **comparables** (misma huella de B, misma red); si no hay comparables, **lo dice**
— no inventa un número.

**U6.10 — Todo número declara su procedencia cuando no es la de siempre**: `objective_overridden`
cuando un recorrido se releyó con otro proxy, `from_cache`, `n_seeds`, los puntos descartados por
geometría inválida, los runs sin checkpoint descontados. El payload lo trae ([4-datos.md](4-datos.md)
U4.10); la UI lo enseña.

**U6.11 — La unidad se declara siempre.** El presupuesto de un recorrido, en épocas **o** en
segundos (con redes de coste distinto, la poda por tiempo favorece a las pequeñas sin decirlo); los
errores de posición, en **píxeles de la imagen original**, nunca en celdas de la vista — o barrer la
geometría cambia la regla de medir a la vez que el modelo.

**U6.12 — Una estimación razonada no es una medición, y la UI no debe hacerla pasar por tal.**
Precedente: la sd «conservadora» de 0,372 **subió** a 0,4148 al medirla. Si un número de pantalla
sale de un supuesto, se etiqueta como supuesto.

**U6.13 — Los resultados de investigación no viven en la UI ni en los tests.** Van fechados y con
sus salvedades a [protocolo.md](../protocolo.md) §5. La pantalla enseña la medición de hoy; el
documento guarda la conclusión y su contexto.
