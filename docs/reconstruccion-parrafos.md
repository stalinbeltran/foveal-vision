# Cómo se arman los párrafos con las esquinas (knob `reconstruct`)

La red no predice párrafos: predice **esquinas**, cuatro tipos, ventana a ventana. Quien
convierte esas esquinas en rectángulos es `_reconstruct` en `src/fv/inference/predict.py`,
que es **F**, post-hoc, y no toca la red.

**Medido el 2026-09-02: ahí se pierde la mayor parte de los párrafos.**

## 1. La pregunta que hay que contestar primero: ¿contamina esto al `f1` que valida?

**No.** Son dos `f1` distintos y sólo uno pasa por aquí:

| Número | Cómo se calcula | ¿Pasa por la reconstrucción? |
|---|---|---|
| **`val_f1`** — el que monitoriza el entrenamiento, elige `best.pt` y rankea recorridos | `detection_counts(corner_scores(logits), y[:,:,0])` en `training/loop.py:68` — por ventana y por esquina | **NO.** No llama a `predict_image`, ni a NMS, ni a `_reconstruct` |
| **f1 de tarea** — `fv.task.task_score`, «la métrica que importa» (protocolo.md §2) | imagen completa → NMS → `_reconstruct` → `paragraph_f1` | **SÍ, entera** |

Y `OBJECTIVES` (`sweeps/spec.py:50`) = `VAL_METRICS` = `{f1, pos_err_px, loss}`: la métrica de
tarea **no** es objetivo de ningún recorrido y no se calcula por época. Aparece sólo como
informe al final de `fv-oat` y `fv-study`, detrás de `--task-score`, y su propio módulo lo dice:
*«this is an informative report at the end of a sweep, never a gate»*.

**Conclusión: ninguna red ha sido mal calificada en su validación, ni en la selección de
`best.pt`, ni en el ranking de ningún barrido.** Todo eso vive en el nivel de ventana.

⚠ **Pero la métrica de tarea sí estaba contaminada, y llegaba a REORDENAR las redes.** Medido
sobre las 987 imágenes con verdad completa de `dirty1000-80px-16px-r20260827`:

| | orden de las tres redes aprobadas |
|---|---|
| ventana (`val_f1`) | mask 0,9543 > optimo 0,9475 > edge 0,9473 *(las dos últimas, empate)* |
| tarea con `tlbr` | mask 0,7666 > **optimo 0,7560 > edge 0,6823** |
| tarea con `quad` | mask 0,9773 > **edge 0,9577 > optimo 0,9385** |

O sea que el heredado metía ruido **independiente de la red** y con él se daba la vuelta a un par.
Cualquier informe que use un número de métrica de tarea anterior al 2026-09-02 lo tiene dentro.

## 2. La avería, en una frase

**La red predice cuatro tipos de esquina y `_reconstruct` usa dos.** `TR` y `BL` se calculan,
pasan el NMS, viajan en `corners`… y se tiran. La única prueba de que un TL y un BR son del
mismo párrafo es que el BR esté abajo a la derecha y que los dos tengan score alto — o sea
**confianza**, que no dice nada sobre pertenecer al mismo bloque.

Se ve a ojo: las cajas unen el TL de un párrafo con el BR de **otro**.

Y hace falta poco para dispararlo: basta con que el BR equivocado sea **el más confiado**. Con los
scores empatados el heredado acierta por el orden en que genera los candidatos, que es otra forma
de decir que no está decidiendo nada.

## 3. `quad`: usar las cuatro

Un TL y un BR son del mismo párrafo si la red vio **también** el TR en `(x1, y0)` y el BL en
`(x0, y1)`. Es la misma prueba geométrica con la que `fv.fallidos` recompone la verdad desde las
etiquetas de ventana (allí sale exacta en 989 de 1000), aplicada a detecciones y por tanto con
tolerancia.

Tres decisiones, y las tres se midieron:

1. **Ordena por apoyo (4 > 3 > 2) antes que por score.** Una caja respaldada por sus cuatro
   esquinas se las queda antes de que una de dos se las lleve.
2. **A igual apoyo, gana LA MÁS PEQUEÑA** — y esto no es un desempate cosmético, es la evidencia
   que resuelve el caso que el apoyo no resuelve. Con párrafos **alineados** (una rejilla, dos
   columnas), el TL de uno y el BR del de al lado forman un rectángulo cuyo TR y BL **también
   existen**, porque son de los dos párrafos verdaderos: esa caja falsa tiene apoyo 4 y scores de
   1,00 igual que las buenas. Lo que sí la distingue es que un *span* es **estrictamente mayor**
   que sus partes y deja huérfanas las esquinas de ambas. Tomar la pequeña primero deja sitio a
   que las demás se formen; al revés, no. El riesgo simétrico —una caja pequeña espuria *dentro*
   de un párrafo grande— necesitaría que dos bloques se solaparan, y los párrafos no se anidan.
3. **Degrada, no exige.** Una caja con sólo TL+BR sigue valiendo, la última. Sin eso se cambiaría
   un fallo de precisión por uno de recall, que no es arreglar.

### Lo que se probó y perdió, para que no se vuelva a probar

Antes del área, el desempate era el **residuo** (cuánto se desvía cada esquina de donde el
rectángulo dice que debería estar). Es plausible y funciona menos:

| orden | optimo | edge | mask | media |
|---|---:|---:|---:|---:|
| `tlbr` (heredado) | 0,7560 | 0,6823 | 0,7666 | 0,7350 |
| apoyo · **residuo** · score | 0,9324 | 0,9535 | 0,9716 | 0,9525 |
| apoyo · **área** · score | **0,9385** | **0,9577** | **0,9773** | **0,9578** |

El área gana en las **tres** redes. La razón se ve en la img 151: el residuo separa 1,5 px de
1,6 px, y el área separa 490 px² de 936 px². Añadir el residuo *después* del área no cambia nada
(es un empate que ya no ocurre), así que no está.

## 4. Lo que gana, medido

987 imágenes con verdad completa de `dirty1000-80px-16px-r20260827`, `corner_tol` por defecto:

| red | `tlbr` | `quad` | Δ | errores | perfectas |
|---|---|---|---|---|---|
| `demo-fov16-optimo` | 0,7560 ± 0,0108 | **0,9385 ± 0,0044** | **+0,1826** (17 SEM) | 1153 → 293 | 560 → 765 |
| `fov16-edge-p20` | 0,6823 ± 0,0120 | **0,9577 ± 0,0038** | **+0,2754** (23 SEM) | 1501 → 212 | 506 → 837 |
| `fov16-mask-p20` | 0,7666 ± 0,0111 | **0,9773 ± 0,0030** | **+0,2106** (19 SEM) | 1120 → 123 | 641 → 916 |

Por imagen, con `fov16-mask-p20`: **mejora 312, empeora 14, igual 661**. Y el 98 % de las cajas
salen respaldadas por sus **cuatro** esquinas (2388 de 2405), que es por qué esto funciona.

⚠ **El error estándar cae a menos de la mitad** (0,0111 → 0,0030). El emparejado no sólo bajaba el
número: lo hacía ruidoso, y ese ruido es el que obliga a `metrica-de-tarea.md` §2 a avisar de que
con 20 imágenes la banda es ±0,083.

### La tolerancia (`corner_tol`) no es delicada

Cuándo una esquina detectada «está donde debería». Por defecto = `nms_radius` (una sola escala
para «dos detecciones son la misma esquina», en vez de dos números que se pueden desincronizar).

| tol px | 1,0 | 2,0 | 3,0 | 4,0 | 6,0 | **8,0** | 10,0 | 12,0 | 16,0 | 24,0 |
|---|---|---|---|---|---|---|---|---|---|---|
| media de las 3 | 0,9379 | 0,9543 | 0,9571 | 0,9590 | 0,9595 | **0,9578** | 0,9539 | 0,9492 | 0,9332 | 0,9026 |

Meseta ancha de **3 a 10 px**, y el defecto (8) cae dentro. No se coge el argmax (6,0) a propósito:
sería ajustar un knob a este dataset, y la diferencia con el defecto (0,0017) está muy por debajo
del error estándar (0,003-0,004).

## 5. ⚠ El defecto NO ha cambiado, y es una decisión pendiente

`RECONSTRUCT_DEFAULT = "tlbr"`, congelado por un test. Cambiarlo **movería todos los números de
métrica de tarea publicados**, y eso es del dueño, no de este fichero.

Lo que sí está hecho para que el cambio sea seguro el día que se tome:

- `reconstruct` y `corner_tol` **entran en la clave de caché** de `fv.task` (era el fallo obvio:
  cambiar el defecto habría servido números cacheados con la otra reconstrucción, bajo el mismo
  nombre y sin decirlo). Hay test.
- Los dos knobs viajan en `knobs` de cada payload de `predict_image`, y por el cuerpo de
  `POST /runs/{name}/predict` y del endpoint de revisión.
- `scripts/dataset_fallidos.py --reconstruir quad`.

**Mi recomendación, con lo medido delante:** cambiarlo. La ganancia es de 17 a 23 errores estándar,
mejora 25 imágenes por cada una que empeora, y baja el ruido de la métrica a menos de la mitad.
Lo que hay que hacer al mismo tiempo es **anotar la fecha del cambio** en `reportes/README.md` del
repo central, porque a partir de ahí un f1 de tarea deja de ser comparable con los de antes.

## 6. Lo que sigue sin estar arreglado

- **La asignación sigue siendo VORAZ.** Coge el mejor candidato y consume sus esquinas. Lo
  correcto sería el reparto **global** que explica más esquinas (un emparejamiento, no un bucle):
  ahí el caso de la rejilla se resuelve por construcción y no por el prior del área. No se hizo
  porque el área ya lo resuelve en la práctica y un cambio grande pide su propia medición.
- **La caja se construye con TL y BR aunque haya cuatro esquinas.** Promediar las dos estimaciones
  de cada lado (`x0` de TL y de BL, `y0` de TL y de TR…) es plausible que suba el IoU. Es **otro**
  cambio: mezclarlo habría hecho que la mejora medida no se pudiera atribuir a ninguno de los dos.
- **Nada de esto mira los píxeles.** Un span entre dos párrafos cruza un hueco de fondo, y eso es
  evidencia que ninguna de las dos estrategias usa.
