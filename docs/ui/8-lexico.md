# UI · Tipo 8 — Especificación léxica (las palabras en pantalla)

> **Qué decide**: qué palabra se escribe en una etiqueta, un encabezado, un botón o un mensaje —
> y cuál está prohibida.
> **Qué NO decide**: el significado de los términos. Eso es [glosario.md](../glosario.md), donde
> **cada entrada ya causó un error una vez**. Aquí solo está la obligación de la pantalla.
> **Cómo se hace cumplir**: **ejecutable desde 2026-07-27**: el validador recorre las 12 pantallas y
> falla si una palabra prohibida aparece en el texto visible, y casa las etiquetas del nav contra la
> tabla del documento (es la comprobación que habría cazado «Barrido por ejes» contra «Estudios»).

---

## Las reglas

**U8.1 — Una palabra con dos significados se cualifica siempre.** En prosa y **en la UI**. No es
estilo: es la diferencia entre «980 ejemplos de val» y «20 imágenes correlacionadas», que es el
malentendido que ya se pagó.

```check U8.1
substrate: same_as
target: U8.2
reason: "cualificar se comprueba por su contrapositivo: la palabra a secas no aparece"
```

**U8.2 — Las palabras ambiguas no entran al vocabulario visible.** Ni en rutas, ni en encabezados,
ni en botones:

| Nunca se escribe | Se escribe | Por qué |
|---|---|---|
| «muestra», «samples» | **imagen** o **ventana** | `sample` es una imagen; el ejemplo de esta red es la ventana |
| «modelo» a secas | **red** (C, sin pesos) o **run** (E, entrenado) | por eso el nav dice *Redes* y *Runs* |
| «dataset» a secas | **fuente** (A) o **dataset de ventanas** (B) | por eso el nav dice *Fuentes* y *Ventanas* |
| «stride» a secas | **stride de extracción** (B) · **stride de inferencia** (F) · **`s_center`/`s_periph`** (C) | son tres cosas distintas, dos de ellas barribles |
| «semilla» a secas | **semilla del split** (B) · **semilla de réplica** (D) | confundirlas hace medir el ruido del split |
| «kernel» a secas | **`kernel_size`** (tamaño) · **filtros** (cuántos, `channels[i]`) · **kernel** (el tensor) | tres sentidos, uno por vista |
| «trial» en la UI | **punto** (del recorrido) o **run** | `trial` es vocabulario de optuna; el **job** es la ejecución en cola |
| «test» como sinónimo de val | **val** o **holdout** | el test se toca una vez, al final, y solo el ganador |

```check U8.2
substrate: dom
kind: dom_absent_text
scope: "*"
args:
  words: ["muestras", "samples", "trial"]
  allow_qualified: ["dataset de ventanas", "muestras de la fuente"]
strength: strong
```

**U8.3 — El nombre en pantalla manda sobre el nombre en el documento.** Cuando divergen, se corrige
el documento. Caso vivo: la pantalla del dominio I se llama **Estudios**; «Barrido por ejes» es el
**método** ([barrido-por-ejes.md](../barrido-por-ejes.md)), no la pantalla.

```check U8.3
substrate: same_as
target: U1.4
reason: "el nav contra la tabla del documento es la misma comprobacion; habria cazado
  'Barrido por ejes' contra 'Estudios'"
```

**U8.4 — Las etiquetas humanas viven en un solo sitio.** El vocabulario de dominio (ejes,
objetivos, defaults, estados) lo sirve el API ([4-datos.md](4-datos.md) U4.2); su traducción a
palabras visibles se escribe una vez en el front. Dos listas de etiquetas divergen igual que dos
listas de valores.

```check U8.4
substrate: ast
kind: single_definition
args:
  seams:
    - name: status_labels
      owner: "web/src/components/ui.tsx"
      markers: ["en cola", "corriendo", "terminado", "interrumpido"]
      min_markers: 2
    - name: axis_labels
      owner: "web/src/screens/Sweeps.tsx"
      markers: ["eje", "objetivo", "presupuesto"]
      min_markers: 3
strength: weak
```

**U8.5 — Los estados se dicen con la palabra exacta del dominio**, no con un sinónimo cómodo:
`queued`, `running`, `done`, `error`, `cancelled`, `interrupted`. **`interrupted` es terminal**
(borrable y reanudable, badge ámbar) y no significa «falló»: significa que su proceso dueño murió y
alguien lo reconcilió.

```check U8.5
substrate: same_as
target: U4.2
reason: "los estados salen de su unica definicion; la costura run_states es esa comprobacion"
```

**U8.6 — Todo texto que pueda acabar en una consola es ASCII.** La consola de Windows es cp1252: una
`δ` griega en un `tie_reason` **mató un estudio nocturno en su última línea** — reproducido y
arreglado. El texto que solo vive en el navegador puede llevar acentos y símbolos; el que viaja a
un CLI, no. Si una cadena viaja a los dos sitios, gana la restricción del CLI.

```check U8.6
substrate: fs
kind: no_match_outside
scope: "src/fv/**/*.py"
args:
  pattern: "print\\([^)]*[^\\x00-\\x7F]"
  allow: []
strength: weak
```

**U8.7 — El idioma: pantalla y documentación en español; código en inglés.** Identificadores,
docstrings, `code` de error y claves de payload en inglés (son contrato); lo que lee el usuario, en
español. Un `code` traducido deja de ser contrato ([4-datos.md](4-datos.md) U4.6).

```check U8.7
substrate: fs
kind: no_match_outside
scope: "src/fv/**/*.py"
args:
  pattern: "code=\"[^\"]*[A-Z ][^\"]*\""
  allow: []
strength: weak
```

**U8.8 — Una negativa se escribe como razón + arreglo, en ese orden**, y en segunda persona del
plural impersonal («no se puede…», «barre `d`, o usa un dataset con esa ventana»). Un mensaje que
solo dice qué falló obliga al usuario a adivinar la mitad que importa.

```check U8.8
substrate: http
kind: http_shape
scope: "POST /networks/validate"
args:
  body: {N: 21, c_frac: 0.8, d: 2, pen_frac: 0.1}
  expect_json: {valid: false}
  requires: ["problems.code", "problems.message", "problems.hint"]
strength: strong
```

