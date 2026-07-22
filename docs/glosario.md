# Glosario

Las palabras de este proyecto que **significan dos cosas**, y cuál usar. Heredado del hermano
(donde cada entrada ya causó un error) más las colisiones nuevas de la geometría foveada.

Regla: cuando una palabra tiene dos significados, **en prosa y en la UI se cualifica siempre.**

---

## 1. Colisiones heredadas (ya hicieron daño una vez)

- **`sample` — una imagen, no un ejemplo de entrenamiento.** El ejemplo de esta red es la
  **ventana**. `num_samples: 200` son 200 imágenes. Es la raíz del malentendido de medir con
  «980 ejemplos de val» que eran 20 imágenes correlacionadas. En prosa: **imagen** y
  **ventana**; nunca «muestra» a secas.
- **`model` — no se usa a secas, nunca.** Es **red** (C, config sin pesos) o **run** (E,
  entrenado). Por eso los recursos son `/networks` y `/runs`.
- **`dataset` — fuente (A) o dataset de ventanas (B).** Siempre cualificado; los recursos son
  `/sources` y `/window-datasets`.
- **`stride` — ¡tres aquí!** (1) el de **extracción/muestreo de B** (identidad de B), (2) el de
  **inferencia** (knob de F, gratis de barrer), y (3) **`s_center`/`s_periph`: los strides de
  las convoluciones por rama** (C, barribles con rango calculado). En prosa, siempre
  cualificado.
- **`seed` — el de B fija el split (fijo en un recorrido); el de D es el eje de réplica.**
  Confundirlos hace que cada punto se evalúe sobre un split distinto: mides el ruido del split.
- **`kernel` — el tamaño (`k_center: 3`), el tensor de pesos («los kernels de la capa 1»), o
  cuántos hay (`ch1: 32`).** `kernel_size` para el tamaño, **filtros** para el número, kernel a
  secas solo para el tensor.
- **`run` / `job` / `trial` / punto** — el **run** es el artefacto en disco; el **job** es la
  ejecución en la cola; el **trial** es vocabulario de optuna; el **punto** es un elemento del
  espacio de un recorrido. **Un trial no es un run**: lanza uno y guarda su nombre.
- **`best`** — `best.pt` (mejor monitor dentro de un run) vs el **ganador** de un recorrido
  (mejor objetivo). Criterios distintos.
- **`config`** — de extracción (B), de red (C), del run (congelado). Siempre cualificado.

## 2. Un concepto, dos nombres (peor: parecen dos cosas)

- **La ventana etiquetada de B == lo que la cabeza de C predice** (contrato ①a). Dos
  declaraciones independientes que deben cuadrar; el validador es quien lo sabe.
- **`periph_real = periph_out · d`** — «cuánto ve la periferia» no es un parámetro: es el
  **producto** de dos buscables. Escribirlo como si fuera un parámetro propio es la trampa que
  instructionsNewNN.md §2 desmontó (antes era un 4 fijo).

## 3. Términos propios de la geometría foveada

| | |
|---|---|
| **corner / esquina** | Uno de los cuatro tipos: `TL, TR, BR, BL`, orden fijo (`corner_order`), horario desde arriba-izquierda. La salida es `(4, 3)` = `[exists, x, y]` (C9) |
| **exists / score** | `p(hay una esquina de este tipo en la ventana)`; score = `sigmoid(exists)` |
| **esquina ciega** | Esquina con `corner_evidence` < 0,05: su párrafo cae fuera de la ventana etiquetada. No significa que no se detecte; significa que su **posición** no es deducible de esos píxeles. Es la población que la periferia foveada existe para arreglar (medido en el hermano: 2,30→1,14 px) |
| **entrada compuesta** | El tensor N×N que consume la red: centro a resolución completa + anillo periférico reducido |
| **ventana original** | El recorte `original_size × original_size` de la imagen del que se construye la entrada compuesta. `original_size = center_out + 2·periph_out·d` |
| **centro / fóvea** | Los `center_out × center_out` px centrales, sin reducir |
| **anillo periférico** | El marco de `periph_out` px de la entrada compuesta; procede de `periph_real = periph_out·d` px reales reducidos ÷d |
| **penetración** | Las `penetration` filas/columnas donde el kernel periférico entra en el centro. **Contributiva**: ambas ramas se suman ahí |
| **banda periférica** (`periph_band`) | `periph_out + penetration`: la banda útil del kernel externo, la que acota su rango |
| **rama** | Cada uno de los dos caminos convolucionales (central / periférico), con kernels y strides propios |
| **máscara de rama** | La que delimita dónde contribuye cada rama sobre la entrada N×N. Se aplica **antes** de convolucionar (opción A, decidida) |
| **muestreo excluyente vs solape contributivo** | El armado de la entrada es excluyente (cada píxel un origen); el procesamiento es contributivo (las ramas se suman en la penetración). No se contradicen — instructionsNewNN.md §8 |
| **máscara de cobertura** | La fracción real por celda del relleno en bordes de imagen. Acompaña al canal que puede tener relleno; jamás binarizada, jamás blanco |
| **rangos calculados** | `kernel_range`/`stride_range`/`downsample_range`/`build_search_space`: los espacios de búsqueda como funciones de `N`, nunca constantes |
| **recorrido** | H: un espacio sobre C y/o D con B fijo, con nombre, reanudable, desatendido («receta de recorrido») |
| **huella / fingerprint** | Hash del contenido de un B; distingue un dataset reconstruido bajo el mismo nombre (contrato ⑧) |
| **knob barato** | Parámetro de F ajustable post-hoc sin reentrenar |
| **suelo de ruido** | La diferencia mínima creíble, medida con N semillas; por debajo, empate |
| **holdout** | Imágenes apartadas una vez, fuente propia, que ninguna configuración de B toca; se mide una vez, al final, solo el ganador |
