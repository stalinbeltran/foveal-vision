# UI · Tipo 8 — Especificación léxica (las palabras en pantalla)

> **Qué decide**: qué palabra se escribe en una etiqueta, un encabezado, un botón o un mensaje —
> y cuál está prohibida.
> **Qué NO decide**: el significado de los términos. Eso es [glosario.md](../glosario.md), donde
> **cada entrada ya causó un error una vez**. Aquí solo está la obligación de la pantalla.
> **Cómo se hace cumplir**: prosa. Aplica a los ocho tipos: una etiqueta ambigua estropea una vista
> correcta, y no hay forma automática de detectarla.

---

## Las reglas

**U8.1 — Una palabra con dos significados se cualifica siempre.** En prosa y **en la UI**. No es
estilo: es la diferencia entre «980 ejemplos de val» y «20 imágenes correlacionadas», que es el
malentendido que ya se pagó.

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

**U8.3 — El nombre en pantalla manda sobre el nombre en el documento.** Cuando divergen, se corrige
el documento. Caso vivo: la pantalla del dominio I se llama **Estudios**; «Barrido por ejes» es el
**método** ([barrido-por-ejes.md](../barrido-por-ejes.md)), no la pantalla.

**U8.4 — Las etiquetas humanas viven en un solo sitio.** El vocabulario de dominio (ejes,
objetivos, defaults, estados) lo sirve el API ([4-datos.md](4-datos.md) U4.2); su traducción a
palabras visibles se escribe una vez en el front. Dos listas de etiquetas divergen igual que dos
listas de valores.

**U8.5 — Los estados se dicen con la palabra exacta del dominio**, no con un sinónimo cómodo:
`queued`, `running`, `done`, `error`, `cancelled`, `interrupted`. **`interrupted` es terminal**
(borrable y reanudable, badge ámbar) y no significa «falló»: significa que su proceso dueño murió y
alguien lo reconcilió.

**U8.6 — Todo texto que pueda acabar en una consola es ASCII.** La consola de Windows es cp1252: una
`δ` griega en un `tie_reason` **mató un estudio nocturno en su última línea** — reproducido y
arreglado. El texto que solo vive en el navegador puede llevar acentos y símbolos; el que viaja a
un CLI, no. Si una cadena viaja a los dos sitios, gana la restricción del CLI.

**U8.7 — El idioma: pantalla y documentación en español; código en inglés.** Identificadores,
docstrings, `code` de error y claves de payload en inglés (son contrato); lo que lee el usuario, en
español. Un `code` traducido deja de ser contrato ([4-datos.md](4-datos.md) U4.6).

**U8.8 — Una negativa se escribe como razón + arreglo, en ese orden**, y en segunda persona del
plural impersonal («no se puede…», «barre `d`, o usa un dataset con esa ventana»). Un mensaje que
solo dice qué falló obliga al usuario a adivinar la mitad que importa.
