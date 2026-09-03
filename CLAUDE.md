# foveal-vision — instrucciones para Claude

**El mismo problema que `image-text-finder`, con otra red y con recorridos automáticos.** La
tarea es idéntica al proyecto hermano: detección de esquinas de párrafo por ventana (`TL, TR,
BR, BL` con `[exists, x, y]`) y reconstrucción de párrafos en la imagen — más adelante líneas,
y luego palabras. Lo que cambia: **la red** es una NN con **muestreo foveado y ramas por
región**, totalmente parametrizada (el centro a resolución completa; la periferia abarca más
área a menor resolución; dos ramas convolucionales que se suman en la banda de penetración), y
**los recorridos (sweeps) barren automáticamente las configuraciones de esa red** además de las
recetas.

**La especificación de la red es [instructionsNewNN.md](instructionsNewNN.md)** — ese documento
manda sobre todo lo que toque la arquitectura. Su principio rector gobierna el proyecto entero:
*todo dato es un parámetro*; las dimensiones y los **rangos de búsqueda se calculan** a partir de
unas pocas longitudes en píxeles reales, nunca se escriben a mano.

El objetivo operativo: poder **preparar series de runs secuenciales** —pruebas cortas en esta
máquina (CPU), luego largas en un server con GPU— que recorran configuraciones de red y
parámetros **sin intervención humana** (recetas de recorrido), y poder **verificar cada objeto
creado** (fuente, dataset, red, run, recorrido, análisis) desde una web app.

---

## Resumen ejecutivo — qué se pidió el 2026-08-31, qué se puede y qué no

**Léelo antes que nada.** Los bloques largos de abajo son el detalle; esto es el estado.

| # | Lo pedido | Estado | Dónde |
|---|---|---|---|
| 1 | Entradas que digan a la red si hay **borde de imagen**, conectadas a la **cabeza**, no a las convs | ✅ hecho · ⚠ **NO medido** | [instructionsNewNN.md §6bis](instructionsNewNN.md) |
| 2 | **Sólo las nn aprobadas** se guardan, y sólo ésas infieren en la app | ✅ hecho. Hoy la lista es **una**: `demo-fov16-optimo` | [inferencia.md](docs/inferencia.md) |
| 3 | **Carpeta temporal** en dev para los pesos mientras entrena | ✅ `data/inferencia/<run>/`, fuera de git | ídem |
| 4 | **Endpoint** que reciba esos pesos | ✅ `PUT /api/inference/staging/<run>/<best\|last>.pt` | [api.md](docs/api.md) |
| 5 | Al terminar, **copiar temporal → repo de datos** | ✅ `POST …/promote` — copia **y aprueba**, es una decisión | ídem |
| 6 | **Entrenar** una nn | ⏸ **no lanzado**: faltan 5 decisiones del dueño (§ abajo) | [entrenar.md](docs/entrenar.md) |
| 7 | Que el relleno del borde **no desvíe** el entrenamiento | ✅ `mask_channel` hecho · ⚠ **NO medido** · entrenando `fov16-mask-p20` | [plan-mask-channel](docs/plan-mask-channel-2026-09-01.md) |
| 8 | Las **medidas de seguridad** de todo entrenamiento para inferir | ✅ las siete, con su comprobación y su lista | [entrenar-para-inferencia.md](docs/entrenar-para-inferencia.md) |

## ⚠ Si vas a entrenar una red que se va a USAR, lee esto primero

**No es lo mismo entrenar un punto de un barrido que entrenar un modelo.** En el
primero lo que se conserva es el número y perder la máquina cuesta un punto; en el
segundo lo que se conserva es el `.pt`, y perder la máquina cuesta el encargo
entero. Las siete obligaciones —y por qué cada una, con lo que costó— están en
**[docs/entrenar-para-inferencia.md](docs/entrenar-para-inferencia.md)**. Resumen
operativo:

```bash
# se lanza SIEMPRE así: unidad de systemd (padre PID 1), nunca hijo de la sesión
"$COORD_HOME/scripts/desacoplar-persistente.sh" entrenar-<run> \
  /bin/bash -lc "cd ~/src/foveal-vision && scripts/entrenar_para_inferencia.sh <run> <red> \
                 --horas-max 6 --presupuesto 1.50 --cada 60 --max-cambios 4"

# y un celador que avisa a Telegram cuando el veredicto CAMBIA
"$COORD_HOME/scripts/desacoplar-persistente.sh" celador-<run> \
  /bin/bash -lc "cd ~/src/foveal-vision && scripts/celador.sh <run> 300 300"

# ¿va bien, y SE ESTAN GUARDANDO los pesos?  (o /use vigilar-entrenamiento)
.venv/bin/python scripts/vigilar_entrenamiento.py --name <run> --max-edad 300
```

⚠ **`entrenar_para_inferencia.sh` y no `entrenar_vast.py` a pelo**: la unidad lleva
`Restart=on-failure`, y un reintento a ciegas es **peor** que ninguno — sin
`--continuar` entra en bucle de alquilar-fallar, y con `--continuar` alquila **otra**
máquina aunque la de antes siga viva. El wrapper mira el estado y elige entre
**adoptar / continuar / arrancar**.

### ❌ Lo que NO se puede (no es una limitación de esta implementación)

- **Recuperar los pesos de los runs previos.** Hay **862 runs** y **1 con pesos**
  *(medido 2026-08-31)*. Los otros 861 **no los tienen en ninguna parte** — no se guardaron
  nunca. La única vía es **reentrenar** el que interese.
- **Predecir sobre imágenes de una fuente**: hay **0 fuentes publicadas** (`GET /api/sources`
  devuelve `[]`). **Revisar sí funciona**, porque lee las imágenes del `windows.npz`. Publicar
  una fuente es trabajo aparte y **no trivial** (ver abajo).

### ⚠ Lo que NO conviene, y por qué

| Idea que parece obvia | Por qué no |
|---|---|
| Guardar los pesos de todos los runs | 862 × 2,7 MB = **~2,3 GB** en un repo de **49 MB**, y git guarda **todas** las versiones commiteadas |
| Empujar (PUSH) los pesos al endpoint **desde una máquina de Vast** | obliga a mandarle el token del API y a exponer el puerto del dev. El **PULL por ssh** de `entrenar_vast.py` ya funciona y por diseño *«ahí no viaja ningún secreto»* |
| Re-renderizar la fuente para publicarla | está **medido** que un re-render **no da el mismo dato**, y `fv.datasets.publish` **aborta** si no cuadra pixel a pixel contra el `windows.npz` |
| Entrenar en este droplet si hay prisa | 2 vCPU → **142–176 s/época** *(medido)*. 40 épocas ≈ 1 h 35 – 2 h |
| Exigir aprobación también para **introspeccionar** (`/kernels`, `/feature-maps`) | rompería el flujo local: `fv-train` deja `best.pt` en el directorio del run, así que habría que commitear 2,7 MB sólo para mirar qué salió |

### 🔶 Riesgo abierto, si se entrena en Vast

**`entrenar_vast.py` NO está en la lista `TRABAJOS` de `cerrable.mjs`** (`fv-train` y `fv-continue`
sí). La máquina alquilada sí se vería —el freno consulta la API de Vast—, pero el proceso local que
la vigila y la **destruye** no se contaría. Si se va por Vast, **esa línea va en el mismo commit**.

### Las 5 decisiones que faltan para entrenar

1. **Qué red**: `fov16-optimo` (la mejor que la evidencia respalda; sus dos ejes nunca se midieron
   juntos) · `fov16-vigente` (comparable con las tablas publicadas) · una con `edge_inputs` (⚠
   mezclaría un mecanismo sin medir con un modelo que se va a usar).
2. **Dónde**: este droplet (gratis, lento) o Vast (~0,05 $/h, rápido → ver el riesgo de arriba).
3. **Cuántas épocas** (`demo-fov16-optimo` hizo 74).
4. **Qué se guarda**: ¿se aprueba para inferencia toda nn entrenada a mano, o sólo la que se ordene?
5. **¿Hace falta que Predecir funcione**, o basta Revisar? Lo primero pide publicar una fuente.

---

## Estado actual — léelo primero

> **⏳ 2026-09-02 — SONDA L1: implementada entera y probada; la rejilla NO se ha lanzado.**
> El encargo ([`instruccioneslargas.md`](instruccioneslargas.md)) pregunta si los kernels de L1
> podrían aprender filtros genéricos **cuando sí hay presión sobre ellos** — en
> `fov16-optimo-mask` no lo hicieron (energía en el subespacio clásico **0,688** contra **0,667**
> de un kernel aleatorio: **1,03×**, o sea nada). La sonda es un autoencoder de **una capa por
> lado** sobre la MISMA vista 20×20; el modelo *son* los kernels.
> 1. **Dónde vive:** módulo aislado **`src/fv/probe/`** (modelo · datos · gabor · métricas · run ·
>    figuras · tabla) y la CLI en `scripts/sonda_l1.py`. Desde Telegram, `/use sonda-l1`.
>    ⚠ **`fv.probe` NO importa `fv.models`**, y de `fv.fovea` sólo `build_view`/`dims_of` (el
>    cargador de ventanas). Es el §6 del encargo y **tiene test**: un import de más ata el
>    experimento a la red que estudia y no rompe nada visible.
> 2. **El criterio está escrito ANTES de mirar** en
>    [`docs/plan-sonda-l1-2026-09-02.md`](docs/plan-sonda-l1-2026-09-02.md), y sus umbrales están
>    **PENDIENTES de que el dueño los confirme**. Ahí están también las tres enmiendas que las
>    medidas previas obligan a proponer.
> 3. ⚠⚠ **La rejilla de λ del encargo `{0 · 0,03 · 0,1 · 0,3}` NO alcanza la banda de activación
>    5-15 % que el propio criterio exige.** *Medido el 2026-09-02* (k=7, K=16, 6 épocas,
>    `--limite 8000`): λ=0,3 deja la activación en **39,9 %**, y la banda vive en **λ ≈ 6-40**,
>    o sea **20× a 130×** más arriba. Con la rejilla tal cual, **el éxito es inalcanzable por
>    construcción**, no por el resultado.
> 4. ⚠ **El nulo del ajuste a Gabor en 3×3 es 0,879** *(medido: mediana del R² sobre 64 kernels
>    aleatorios)*, así que el techo de `Gabor Δ` ahí es **0,121** — **menos** que el umbral de
>    éxito propuesto (0,25). El ancla k=3 se lee por el **enriquecimiento**, no por el Gabor.
>    Un Gabor tiene 7 parámetros libres: en 3×3 ajusta cualquier cosa. **La métrica se lee
>    siempre como diferencia contra su nulo, nunca en absoluto.**
> 5. **Coste MEDIDO**, no estimado *(2026-09-02, este droplet, 2 vCPU)*: la combinación más cara
>    (k=9, K=32) va a **101,3 s/época**; la rejilla de 48 runs × 30 épocas son **~12,0 h**, o
>    **~4,0 h** con `--limite 20000` (34,0 s/época). **No alquila nada**, y `sonda_l1.py` ya está
>    en la lista `TRABAJOS` del freno (`telegram-coordinator/scripts/cerrable.mjs`): comprobado
>    en vivo, el veredicto dice `🔴 NO CERRAR — 1 trabajo(s) vivo(s): sonda_l1.py`.
> 7. ⚠ **El aviso ya NO decide el código de salida del trabajo desacoplado.** `/use sonda-l1` con
>    `--rejilla`/`--solo`/`--repetir-mejores` levanta una unidad de systemd con
>    `Restart=on-failure`, y el `notify.mjs` iba al final de la tubería: *medido el 2026-09-02*,
>    la sonda **terminó bien**, el aviso falló y la unidad se quedó **reiniciándose cada 30 s** —
>    12 h de rejilla relanzadas por un aviso. Y al revés, un trabajo que reventara salía como
>    `success` y no se reintentaba. Ahora manda el código del **trabajo**, en
>    `scripts/sonda_l1_desacoplada.sh` (un fichero, no una línea escapada dentro de un JSON),
>    **con un test por cada dirección del fallo**. La unidad corre en su propio cgroup
>    (`/system.slice/sonda-l1.service`, comprobado), o sea que sí sobrevive al reinicio del bot.
> 6. **La fase 2** (congelar el codificador ganador como L1 de la rama central y reentrenar
>    contra el f1 0,954 de `fov16-mask-p20`) **no está implementada, a propósito**: depende de
>    qué `k` y `K` ganen.
> **67 tests** en `tests/test_sonda_l1.py`; suite **574 pasan** (eran 533).
>
> ### ⚠ La revisión del dueño del 2026-09-02 cambió el diseño. Lee esto antes de tocar la sonda
>
> Contestó las cuatro preguntas en `instruccioneslargas.md` (`7959c558d`) y el criterio quedó
> **congelado** en el plan, con la rejilla todavía sin correr. Cinco cosas:
> 1. **λ ya NO es un eje: se CALIBRA por celda** (`src/fv/probe/calibrate.py`) hasta activación
>    10 % ± 3, y la λ resultante se guarda como dato del run. Su motivo, que yo no había visto:
>    mi mapa λ→activación salía de **un** punto (k=7, K=16), así que una rejilla fija más el
>    filtro por banda dejaba celdas enteras sin combinación admisible.
> 2. ⚠⚠ **Lo que asienta la activación son los PASOS del optimizador, no las épocas ni el
>    tamaño del dataset** — y medirlo mal invierte la conclusión. La primera calibración
>    evaluaba con «2 épocas sobre 8.000», o sea **64 pasos**, y *medido con las MISMAS 8.000
>    ventanas*: 64 pasos → **24,3 %**, 256 → 4,3 %, 640 → 3,6 %; el run real (329 pasos en su
>    primera época) dio 4,1 %. El proxy **sobreestimaba seis veces**, y por eso declaró k=3
>    como *«satura, no puede esparcirse»* cuando en realidad se esparce de sobra. Con el
>    presupuesto correcto (400 pasos) **las cuatro celdas llegan a la banda y ninguna satura**.
>    ⚠ Si alguna vez una celda «satura», **sospecha primero del presupuesto de pasos**.
>    La lección: **una calibración cuyo proxy no transfiere es peor que no calibrar**, porque
>    hace que la esparsidad *parezca* igualada cuando no lo está.
> 3. **El criterio se normaliza**: manda `Δ > p95 de la mediana de K kernels aleatorios`
>    (bootstrap, sin unidades) y la magnitud es `Δ/(1−nulo)`. Un 0,25 **absoluto** es el 52 %
>    del margen alcanzable en k=5 y el 32 % en k=9 — tres exigencias escritas como una.
> 4. ⚠⚠ **`enriq` está por DEBAJO de su nulo en toda la sonda (0,47-0,61) y ya se sabe por qué.**
>    `classic_basis` es de **baja** frecuencia y la normalización de contraste del §2 deja los
>    kernels en **alta**. *Medido en la misma celda*: con normalización `enriq` 0,47, sin ella
>    **1,01**. Y la consecuencia grande: **producción NO normaliza** (`build_view` da la vista
>    cruda), así que el **0,688** de `fov16-mask-p20` y el `enriq` de la sonda **nunca fueron
>    comparables** — ni en k=3, que era donde se apoyaba el ancla. `enriq` pasa a diagnóstico.
> 5. **Dos métricas sin plantilla** (`src/fv/probe/spectrum.py`), de la FFT 2D con rejilla común:
>    `conc_orient` (concentración a doble ángulo) y `conc_banda`. Tapan el hueco del Gabor: en
>    3×3 su nulo es 0,238 (techo 0,762) contra el 0,879 del Gabor (techo 0,121). ⚠ `conc_banda`
>    **no** sirve en 3×3: un soporte 3×3 no puede ser de banda estrecha (incertidumbre).
>
> ⚠ **Una propuesta suya que NO se aplicó, y por qué**: dijo que los 101 s/época son sobrecarga,
> «casi seguro la normalización recalculada por época». **No lo es** — se calcula una vez y se
> cachea en disco. *Medido*: 1.045 GFLOP/época, 14,3 GFLOP/s en 2 vCPU, y el `conv2d` de torch
> **a pelo** ya cuesta 178 de los 222 ms del paso. **El 80 % del tiempo está dentro de la
> convolución**; renormalizar es el 0,0 %. No hay un 100× esperando.
> Lo que sí se aplicó entero: **`--limite` es variable de confusión JUSTO sobre la métrica
> principal** (menos ventanas → kernels más ruidosos → Gabor más bajo), así que sesga hacia el
> fracaso. El tanteo corre con el train entero.

> **🔒 2026-08-31 — REGLA DEL DUEÑO: los pesos de un run NO se guardan por defecto.**
> Sólo se conservan —y **sólo ésas puede usar la web app para inferir**— las redes que
> **el dueño aprueba una a una**. **Hoy la lista es exactamente una: `demo-fov16-optimo`**
> (`best.pt` + `last.pt`). Las demás se añaden **cuando él lo ordene** («guarda esta nn para
> inferencia», «guarda sus pesos») — **no** porque hayan salido bien, **no** por iniciativa
> propia.
> 1. **La lista es un DATO**, no código: `inferencia.json` en la raíz del repo de datos. Ahí y
>    no en el repo de código porque gobierna unos ficheros que viven ahí: la lista y los pesos
>    que nombra tienen que viajar y revertirse juntos.
> 2. ⚠ **Un `.pt` que esté en disco pero no en la lista NO se usa.** Es algo que se coló, y
>    servirlo haría inferir con una red que nadie eligió. Tiene test.
> 3. **Por qué la regla:** 862 runs × 2,7 MB de pesos = ~2,3 GB en un repo de 49 MB *(medido
>    2026-08-31)*, y git guarda **todas** las versiones que se commitean. La mayoría de esos
>    runs son puntos de un barrido: lo que se lee de ellos es su número, no el modelo.
>    ⚠ Y la otra cara también está medida: hasta el 2026-08-30 no se guardaba **ninguno**, y eso
>    costó `fov-optimo-p20` entero (~1 h 40 min de reentrenamiento). La regla no es «no
>    guardes»: es **«guarda lo que se va a usar, y dilo»**.
> 4. **La ANTESALA** (`data/inferencia/<run>/`, **fuera de git**) recibe los pesos **mientras se
>    entrena**, por `PUT /api/inference/staging/<run>/<best|last>.pt`. **Gana al definitivo** al
>    inferir: durante un entrenamiento la buena es la que acaba de bajar. ⚠ Estar en la antesala
>    **no es** estar aprobada.
> 5. **Al terminar, `POST /api/inference/staging/<run>/promote`** copia los dos al repo de datos
>    **y aprueba**: copiar y aprobar son **una** decisión. No commitea — devuelve el comando,
>    como `fv-train`.
> ⚠ **La antesala NO es `data/cache/`**: la caché se borra sin perder nada, esto guarda los
> únicos pesos de un entrenamiento en curso.
> **El detalle entero, la puerta del endpoint y por qué el PULL de `entrenar_vast.py` sigue
> siendo el camino recomendado: [docs/inferencia.md](docs/inferencia.md).** 24 tests en
> `tests/test_inferencia.py`.

> **🔧 2026-08-31 — `edge_inputs`: la red ya puede saber DÓNDE SE ACABA LA IMAGEN.
> Implementado y probado; NO medido.**
> Pedido por el usuario, y con el sitio elegido por él: **las entradas nuevas van directas a la
> cabeza final, no a las capas convolucionales.**
> 1. **El problema es un límite del muestreo, no de capacidad.** `pad_mode: edge` replica la fila
>    del borde al salirse de la imagen (C11), y esa réplica es **por construcción**
>    indistinguible de imagen real parecida. Un párrafo **pegado** al borde superior y uno
>    **cortado** por el borde de la vista producen casi la misma entrada con etiquetas opuestas
>    (en el primero TL/TR existen; en el segundo están fuera, más arriba). Ninguna arquitectura
>    puede separarlas: **la información no está en la entrada**.
> 2. **`edge_inputs` ∈ {`off` · `pad` · `dist`}** añade **4 escalares** por ventana (orden
>    `EDGE_SIDES = L, T, R, B`, `0` = no hay borde por ahí, `1` = está aquí **en los dos modos**)
>    concatenados a las features **justo antes de la `Linear`**, fuera del ReLU y fuera del
>    dropout. **+48 pesos, +0,03 %** *(medido con `network_trace`)*. `off` es el default y es
>    **bit-idéntico** a la red anterior: los checkpoints en disco cargan `strict`.
> 3. ⚠ **Por qué a la cabeza y no como canal** (que es lo que F7 dejó abierto desde C11): la
>    señal **no es espacial** —como canal sería una constante pintada en N×N celdas— y sobre todo
>    **las ramas están enmascaradas por región**, así que una señal que entrara por el input sería
>    invisible justo para la rama central, que es la que predice las esquinas. F7 queda **cerrada
>    a medias**, por el lado barato (C15).
> 4. ⚠ **Lo que ya se sabe SIN gastar nada** *(medido el 2026-08-31 sobre
>    `dirty1000-80px-16px-r20260827`, 140.000 ventanas)*: `pad` se enciende en el **31,4 %** de las
>    ventanas (30,4 % de las esquinas positivas) y `dist` en el **91,4 %**. Las esquinas
>    etiquetadas a ≤1 px del borde son **3,02 %**. Dos lecturas que reordenan lo obvio: **`pad` no
>    es «demasiado corto»** (el borde es una fracción grande de una imagen de 80×60), y **`dist`
>    encendida en el 91 % es ambigua** — en 5×3,75 fóveas, «cerca del borde» y «dónde estoy en la
>    página» son casi la misma variable, así que un `dist` ganador **no se podría citar como que
>    el borde importa** sin una medida más.
> 5. ⚠ **NO se ha barrido: que exista y entrene no es que mejore.** El criterio está escrito
>    **antes de mirar** en [docs/plan-edge-inputs-2026-08-31.md](docs/plan-edge-inputs-2026-08-31.md)
>    (tanteo `ei-t`, 6 runs, ≈0,4 $ estimado), y va **detrás de `do-v`**, que sigue siendo lo
>    pendiente.
> **Dónde quedó escrito**: [instructionsNewNN.md §6bis](instructionsNewNN.md) (la spec manda sobre
> la arquitectura) · decisiones.md (F7 cerrada a medias, C15) · barrido-por-ejes.md (tabla de
> defaults) · organizacion.md · glosario.md (⚠ la colisión **`edge` ≠ `border`**, que el español
> fabrica). **18 tests** en `tests/test_edge_inputs.py`; suite **382 pasan** (eran 364).

> **✅ 2026-08-26 — LAS DOS SEMILLAS, DOCUMENTADAS: EL SPLIT ES POR IMAGEN Y ES DE B.**
> El usuario pidió verificar dónde se separan las muestras, sospechando que se repartieran las
> **ventanas** ya recortadas (lo que filtraría la misma imagen a train y val). **No es el caso, y
> está comprobado ejecutando, no razonando.** Lo que hay que saber:
> 1. **El reparto es POR IMAGEN y ocurre ANTES de recortar.**
>    [`_assign_splits`](src/fv/windows/extract.py#L53-L64) recibe el número de **imágenes**, baraja
>    esos índices y devuelve `split_by_image`; dentro del doble bucle de posiciones cada ventana
>    sólo **hereda** `split_by_image[si]` ([extract.py:127](src/fv/windows/extract.py#L127)). No hay
>    ninguna decisión aleatoria a nivel de ventana. `grep` de `_assign_splits`/`split_by_image` en
>    `src`, `tests` y `scripts`: **una sola ruta**. El resto de la cadena sólo **filtra** por esa
>    etiqueta (`dataset.py:20`, `loop.py:120-122`).
> 2. ⚠ **HAY DOS `seed` Y NO SE COMUNICAN — es la confusión que el glosario ya avisaba, ahora con
>    su consecuencia escrita.** `ExtractConfig.seed` (**B**) baraja las imágenes **una sola vez**, al
>    extraer, y el reparto queda **congelado en disco** (`split.json` + el array `split` del `.npz`).
>    `Recipe.seed` (**D**) sólo fija pesos iniciales y orden de lotes. **Correr N semillas mueve sólo
>    el segundo**: los N runs comparten val.
> 3. ⚠⚠ **Consecuencia que NO estaba documentada y cambia cómo se citan los números.** La banda de
>    N semillas mide **la varianza de reinicializar**, no la de *haber elegido otro val*. A favor:
>    dos puntos de un eje se comparan contra el mismo val, y eso es lo que legitima
>    `permutation_test`. En contra: si el `seed` de extracción dejó en val 200 imágenes atípicamente
>    fáciles, los N runs lo heredan y **ninguna banda lo delata**. El error es **común a todos los
>    puntos**, así que **el orden se sostiene** (que es lo que decide un ganador) pero **el nivel
>    absoluto no está acotado**. Todos los f1 del inventario — el 0,9574 de `red-fov` incluido — son
>    **comparables entre sí, no absolutos**.
> 4. **Es un nivel por encima del ± de metrica-de-tarea.md §4.1**: aquel `sem` sale de la dispersión
>    entre imágenes **dentro** de ese val; la incertidumbre de **cuál** val **no se ha medido nunca
>    aquí**. Exigiría re-extraer B con otras semillas de split y repetir el recorrido entero — y
>    **F11** decidió no regenerar el dato, justo para conservar la comparabilidad histórica. Queda
>    como **pregunta abierta, no como defecto**.
> **Dónde quedó escrito**: protocolo.md §1 regla 4 (el aviso donde se declara un ganador) y §3 Paso 1
> (la tabla de las dos semillas y las dos caras de la reserva) · formatos.md §4.1 (el reparto se
> congela, y `split.json` y el `.npz` dicen lo mismo) · glosario.md (entrada `seed`, ampliada).
> ⚠ **Verificado, no razonado**: extracción real de una fuente sintética temporal (30 img, 2.640
> ventanas) → **0 imágenes en más de un split**, y los conjuntos del `.npz` coinciden con
> `split.json` en los tres; los **10 `split.json` del repo** tienen los tres conjuntos disjuntos (sus
> `.npz` no están en disco: son artefactos ignorados). **192 tests** (el contrato ⑧ ampliado con dos
> aserciones nuevas: `split.json` ≡ array del `.npz`, y que el extractor **no importa `Recipe`**),
> y **las dos se probaron rompiéndolas a propósito**: fallan cuando deben. La fuente y el dataset
> temporales se **borraron**.

> **✅ 2026-08-26 — LA GEOMETRÍA NUEVA YA DIO RESULTADOS, Y SON LOS MÁS GRANDES DEL INVENTARIO.**
> Los ocho recorridos de prioridad que corrió la flota, **leídos** en esta sesión desde los
> artefactos del repo con `scripts/estudio_informe.py` (aplica R1–R6, escritas antes). Detalle
> completo en
> [el reporte de la geometría nueva, en el repo central](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/sintesis/2026/08-agosto/2026-08-26-geometria-nueva-primeros-resultados.md).
> ⚠ **Nada de esto lo midió esta sesión**: los runs los produjo la flota.
> 1. ⚠⚠ **`red-fov` es el resultado grande: a igual área, ver el borde SIN COMPRIMIR gana.** Con
>    `border_px` fijo en 8, `border_reduce` 1 → **0,9574** contra 0,9472 (reduce 2) y 0,9408
>    (reduce 4). Monótono, **bandas min–max disjuntas**, y +0,0102 sobre el siguiente con
>    **p = 0,008 — el mínimo alcanzable** con 5 semillas. Contra el vigente del proyecto son
>    **+0,0233**, la mejora más grande medida nunca aquí. **Es la pregunta que la reparametrización
>    hizo formulable** y antes no se podía ni escribir.
>    ⚠ **NO es gratis**: N pasa de 20 a **32**, la cabeza crece **+156 %** y el reloj **1,9×**
>    (69,8 s/época contra 36,8). Citarlo sin esto es citarlo mal.
> 2. **`borde-ancho` CIERRA el eje del ancho, que llevaba dos recorridos abierto por la derecha.**
>    A coste constante (anillo atado a 2 celdas, N=20 en los cinco puntos): 4 → 0,9341 · **8 →
>    0,9408** · 10 → 0,9385 · 12 → 0,9376 · 16 → **0,9321, peor que el vigente**. Óptimo
>    **interior**, por fin. Y **cae justo donde el análisis físico predijo**: a 16 px el 26 % del
>    anillo es relleno replicado (instructionsNewNN.md §2.2). ⚠ R4 ❌: 8 px da p = 0,063, así que
>    **el vigente no se mueve** por la regla escrita antes.
> 3. ⚠ **`ov-fov` NO declara nada y no hay que citarlo como si lo hiciera.** Incompleto (16/20, 3
>    runs excluidos por morir a medias) y el punto de 4 px tiene **2 semillas**: con 2 contra 5 sólo
>    hay 15 arreglos, o sea **p mínimo alcanzable 0,133** — R4 no puede declarar al 5 % pase lo que
>    pase. Lo único decible: la tendencia es monótona a favor del solape, y **`0` (ramas disjuntas)
>    es el peor de los cuatro**. Si se sostiene al completarlo, sería la primera evidencia directa a
>    favor del solape contributivo de la spec §7. **Terminarlo antes de usarlo.**
> 4. **Cuatro ejes se cierran EN CONTRA, con 5 semillas** — y eso también es información:
>    `pos_weight` (4,0 pierde −0,0204 y 8,0 −0,0561, **ambos p = 0,008**), `k_center` (5 y 7 pierden
>    con p = 0,024 y 0,008), `scheduler` (`cosine` −0,0012, **p = 0,857**: no hace nada) y `monitor`
>    (`val_f1` +0,0059 con p = 0,214, no cruza). ⚠ **`pos_weight` duele**: era *la* hipótesis
>    plausible del inventario para atacar el cuello de botella de detección, y **empeora**.
> 5. ⚠ **El plano `(border_px, border_reduce)` NO está barrido**, sólo sus dos rectas por el punto
>    vigente. El mejor punto medido (8 px sin comprimir) es **la esquina de las dos**, no un óptimo
>    demostrado: con `border_reduce=1`, el `border_px` óptimo podría estar **por debajo** de 8 y
>    salir más barato. Es lo siguiente que vale la pena preguntar.
> 6. ⚠ **Todo esto es f1 de VENTANA, el proxy.** Ninguno se ha llevado a la métrica de tarea, y el
>    proxy ya exageró una vez por un factor de dos en `n_layers`. Y si se adopta N=32, **la
>    comparación con la plana hay que rehacerla**: `plana-24-single` se eligió para igualar
>    parámetros con la foveada de N=20, contra una de N=32 ya no es coste equivalente.
> **Arreglo hecho aquí**: `estudio_informe.py` tenía `--eje` con default **`"lr"` cableado**. Sin
> pasar la bandera no fallaba: imprimía la tabla con el eje entero a `None` y un ganador llamado
> `None` — creíble y falsa, peor que un error. Ahora **se deriva del `space` del recorrido**, y si
> no hay un único eje se niega diciéndolo.

> **✅ 2026-08-26 (15:12 UTC) — LA FLOTA SE APAGÓ A MANO. NO QUEDA NADA CORRIENDO.**
> El usuario apagó la instancia de Vast (`estudio-c11` y las demás de su flota) mientras el
> relanzamiento de los cuatro recorridos a medias estaba en marcha. Estado comprobado en el repo
> justo después:
> - **`sch-fov` 10/10 y `pw-fov` 20/20 se cerraron enteros** antes del apagado, y se leen en el
>   punto 4 del bloque de arriba. Los dos dejan el vigente donde estaba.
> - **`ch-fov` quedó en 19/20** (falta `channels`=32 semilla 5; la lectura no depende de ella) y
>   **`ov-fov` en 16/20** (faltan 4, y ahí la falta **sí** bloquea el veredicto — punto 3 de arriba).
> - **Cuatro runs murieron a mitad** (épocas 43–57). Al pasar `estudio_informe.py` después del
>   apagado se detectaron solos y quedaron marcados **`interrupted`** con el motivo escrito, así que
>   **están excluidos de todas las tablas**. No son medidas: pararon por la muerte de la máquina, no
>   por `patience`, que es R1.
> - ⚠ **El `flota.json` de ese relanzamiento NO llegó a escribirse**, porque lo escribe
>   `estudio_flota.py` al terminar y la flota se apagó desde fuera. **Coste e instancias de esa
>   corrida son irrecuperables** desde el repo; los `flota.json` que hay en esos cuatro directorios
>   son de la corrida **anterior** (06:53 UTC) y atribuírselos sería contar dos veces 101 instancias
>   y 3,2996 $. Reporte con el detalle:
>   [el reporte #13, en el repo central](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/08-agosto/2026-08-26-prioridad2-relanzamiento.md).
> - ⚠ **El vigilante horario sigue existiendo y PUEDE VOLVER A ALQUILAR MÁQUINAS** si está armado:
>   mira si a un estudio le faltan puntos y relanza la flota sólo para lo que falte — y a `ov-fov` y
>   `ch-fov` **les faltan**. Se para con `/use vigilante` → `parar`. Comprueba en qué estado está
>   antes de dar por hecho que no va a gastar nada.

> **⚠ 2026-08-26 — CÓMO LLEGÓ AQUÍ ESE TRABAJO (contexto; ya NO está en vuelo — ver el bloque de arriba).**
> Al hacer merge a `main` apareció que `origin/main` **ya tenía** la reparametrización de abajo y
> **había seguido construyendo encima** desde otra máquina (la flota). Esto NO lo midió aquella
> sesión; se registra aquí porque el aviso de «TODO PARADO» de más abajo **quedó obsoleto** y la
> siguiente sesión lo leería como si nada corriera. Lo que hay, comprobado en el repo:
> 1. **Mecanismo nuevo `couple` («ataduras»)** en `fv.sweeps.spec`: un campo que se mueve **con**
>    el eje en vez de multiplicarse contra él — la **diagonal**, no el producto cartesiano. Existe
>    exactamente para poder preguntar *«¿más área a coste constante?»*: barrer `border_px`
>    [4,8,10,12,16] con `border_reduce` atado [2,4,5,6,8], de modo que el anillo se queda en **2
>    celdas y N=20 en los cinco puntos**. Mismo tensor, mismos parámetros, mismo coste por época.
> 2. **`docs/plan-prioridades-2026-08-25.md`**: cuatro estudios (E1 `borde-ancho`, E2 afinado de la
>    plana, E3 foveada vs plana por métrica de tarea, E4 los knobs de F) **con el criterio escrito
>    antes de mirar**, como manda protocolo.md §1.
> 3. **18 recorridos en `queued`** en `sweeps/` (`borde-ancho`, `pw-fov`, `sch-fov`, `red-fov`,
>    `ch-fov`, `kc-fov`, `mon-fov`, `ov-fov`, los `pl-t-*`…). ~~Ninguno tiene resultados todavía~~
>    **— desactualizado: los resultados llegaron después** (ver los dos bloques de arriba).
>    ⚠ **Y el `state.json` de esos directorios sigue diciendo `queued` aunque los runs estén
>    hechos**, porque lo escribe quien lanza el recorrido y aquí los runs llegaron por el libro de a
>    bordo desde la flota. **No uses `state.json` para saber si un recorrido tiene datos**: cuenta
>    `runs/<recorrido>-*/status.json`, o pasa `scripts/estudio_informe.py`, que además excluye los
>    que murieron a medias.
> 4. ⚠ **Hay automatización que ALQUILA MÁQUINAS**: el ejecutor `telegram/executors/vigilante.json`
>    corre `scripts/vigilante_prioridades.py` cada hora, mira si a un estudio le faltan puntos y
>    **relanza la flota sólo para lo que falte**. Se para con `/use vigilante -> parar`, y eso **no**
>    detiene una flota ya viva (para eso está `/use apagar-vast`). Si vas a tocar `spec.py`,
>    `runner.py` o la geometría, cuenta con que puede haber runs entrando mientras tanto.
>    **Sigue vigente tras el apagado del 26-ago**, y a `ov-fov` y `ch-fov` les faltan puntos: si el
>    vigilante está armado, volverá a alquilar.
> 5. **Tres arreglos suyos que conviene conocer**: un run cortado a mitad entraba en la tabla como
>    si fuera una medida; el pozo de máquinas se pedía una sola vez y tres estudios se quedaban a
>    cero; y `estudio-informe` se rompía cuando el eje no era un número (`channels`).
> **Esta sesión sólo integró su commit de verificación encima y comprobó que la suite sigue en
> verde con todo junto.** Los resultados de esos estudios, cuando los haya, no están aquí.

> **✅ 2026-08-25 — LA GEOMETRÍA SE DECLARA EN PÍXELES REALES. NINGUNA RED CAMBIÓ.**
> Reparametrización pedida por el usuario y decidida con él (decisión **C14**). La geometría se
> declaraba desde el lado equivocado: `N` y `c_frac` fijaban la fóvea **entre los dos**, así que
> ninguno podía ser eje (cada uno *solo* rompía ①a) y **una parte legal del espacio era
> inalcanzable por barrido** — el motor es OAT, un eje cada vez. Ahora:
>
> | | |
> |---|---|
> | `fovea_px` | la ventana etiquetada de B (contrato ①a: se **toma**, no se busca) |
> | `border_px` | el borde difuso, por lado, en **px reales** |
> | `border_reduce` | px reales por celda de borde — **sólo el método de reducción** |
> | `overlap_fovea_px` | px de **fóvea** que ve también la rama del borde |
> | `overlap_border_px` | px de **borde** que ve también la rama de la fóvea (**nuevo**) |
>
> 1. **`N` pasa a derivado** (`fovea_px + 2·border_px/border_reduce`) y no lo escribe nadie. El
>    objetivo del usuario era separar **cuánto borde hay** de **cómo se comprime**, para que el
>    método de reducción pueda cambiar mañana sin tocar ninguna de las dos definiciones.
> 2. **Los dos solapes son un grado de libertad nuevo.** Antes sólo existía la penetración hacia
>    dentro, y con **suelo de 1 px**: que la fóvea saliera sobre el borde no era expresable, y la
>    ausencia de solape tampoco. Hoy `overlap_fovea_px=0` es el **control** de la elección de
>    solape contributivo de instructionsNewNN.md §7.
> 3. ⚠ **NO cambia ni un peso, y está verificado, no razonado**: un checkpoint guardado con la
>    ortografía vieja carga `strict=True` en una red construida con la nueva, con los **168.652
>    params** documentados de la L4 vigente, y las dos dan **la misma salida bit a bit** (test
>    permanente en `test_contracts.py`). Los **478 runs en disco** siguen resolviendo su geometría
>    (otro test). Los `.pt` no se pudieron cargar de verdad porque **esta máquina se rehizo y no
>    quedan** (son artefactos ignorados por git) — la equivalencia se probó construyendo y
>    guardando, que es la misma afirmación.
> 4. ⚠ **`d` se RENOMBRÓ a `border_reduce` porque cambió de significado**, no sólo de nombre: antes
>    agrandaba el contexto (`borde = celdas·d`), hoy sólo comprime un borde de tamaño fijo. Un spec
>    viejo con eje `d` **para con `axis_renamed`** en las dos puertas en vez de entrenar otra red en
>    silencio. Misma idea con una config que trae **las dos ortografías**: `geometry_double_spec`,
>    rechazo — nunca precedencia (es «el mismo dato en dos sitios»).
> 5. **`derive_geometry` muere.** Existía sólo para buscar el `N` cuya fóvea cayera en `W` y para
>    aflojar `c_frac` con una razón cuando no lo encontraba (D-G2/D-G3, ahora obsoletas). Queda
>    `legacy_border_px` como shim para leer los planes escritos antes (`studies/plana-*`).
> 6. ⚠⚠ **HALLAZGO QUE SALE GRATIS DE LA TRADUCCIÓN, y vale más que el refactor.** Los recorridos
>    `proxy-c-d` y `d5-L4` barrían `d` con `N` fijo, o sea movían **`border_px` y `border_reduce`
>    juntos** dejando el anillo en 2 celdas: lo que midieron fue **`border_px` = 2, 4, 6, 8 px a
>    coste constante**. Y sube monótono — 0,9310 · 0,9341 · 0,9362 · **0,9408** — con el ganador en
>    **el borde del rango** (p = 0,063) y siendo **el punto más barato**. Es decir: **ensanchar el
>    borde sin pagar nada estaba ganando cuando se cortó el rango.** Eso es ahora el estudio nº 1 del
>    inventario, y el rango recomendado (`border_px ∈ [8,10,12,16]` con el anillo fijo en 2 celdas)
>    para **justo antes** de que el relleno domine: a 16 px el recorte es 48×48 sobre imágenes de
>    60×80 y el **26 % del anillo ya es `pad_mode: edge`**, no imagen.
> 7. ⚠ **Corrige un titular anterior**: «la periferia no está aportando» (nota del 2026-07-26) vale
>    para *aquel* dataset y *aquella* red (L2, 20 épocas). `d5-L4`, sobre el dataset del 24-ago y la
>    L4 vigente, sale **al revés**. Cítese con las dos fechas.
> 8. **La pregunta simétrica sigue sin medir**: *a igual área de contexto, ¿ayuda verla con más
>    resolución?* — `border_reduce` con `border_px` fijo. **No es cost-neutral**: la cabeza es el
>    97 % de los parámetros y crece con `N²` (+44 % al pasar de 2 a 4 celdas, +156 % a 8).
>
> ⚠ **Verificado, no razonado**:
> - **183 tests en verde** (+33) y `npm run build` limpio.
> - **`verify_axes.py`: 28/28 ejes, 0 fallos** — los **cuatro ejes geométricos nuevos entrenan
>   runs de verdad** (`border_px`, `border_reduce`, `overlap_fovea_px`, `overlap_border_px`), y los
>   **cinco rechazos** salen en **las dos puertas** (`fovea_px` → `axis_breaks_window_size`;
>   `N`/`c_frac`/`pen_frac`/`d` → `axis_renamed`).
> - **`verify_spec.py --live`: 0 violadas**, 44 ok. ⚠ La cobertura sale **55 %** y no el 82 %
>   documentado: **no es una regresión**, son 30 reglas «no aplicables» porque en esta máquina no
>   hay recorridos ni runs con carga que ejercitarlas (ver el punto siguiente).
> - **UI en la instancia real**: 12 pantallas **sin un error de consola**; Redes enseña los cinco
>   campos nuevos y **ya no enseña `c_frac`/`pen_frac`**; el panel «lo que implica» reacciona en
>   vivo (borde 4 px → N=20, recorte 24; borde 8 px → N=24, recorte 32) y un `border_px=5` con
>   `border_reduce=2` sale como **`[border_not_divisible]` con razón y arreglo**. Backend y vite
>   **se cerraron** al terminar.
> - Los **5 configs de red migrados** a `format_version: 2`, con la geometría comprobada idéntica
>   uno a uno.
>
> ⚠ **Esta máquina se había rehecho**: no había `.venv`, ni Python 3.12, ni `node_modules`, ni
> **ningún `.npz` ni `.pt`** (son artefactos ignorados por git). Se instaló Python 3.12, se recreó
> el entorno (torch 2.13.0+cpu), se hizo `npm install`, y para `verify_axes` se regeneró una fuente
> sintética temporal que **se borró después**. Consecuencia a tener presente: **no se pudo cargar
> ningún checkpoint histórico** porque no queda ninguno — la equivalencia de pesos se probó
> construyendo y guardando (test permanente), que afirma lo mismo, pero no es lo mismo que abrir un
> `best.pt` de agosto. **Cero entrenamiento, cero artefactos del usuario borrados.**
>
> **Lo siguiente, por valor**: (a) el estudio nº 1 de arriba, que es barato y tiene evidencia
> directa; (b) la comparación foveada vs plana, que sigue siendo *la* pregunta del proyecto y sigue
> sin contestar; (c) los knobs de F, gratis y sin aplicar (decisión F15 del usuario).

> **✅ 2026-08-11 — TODO PARADO** *(obsoleto: ver la nota del 2026-08-26 arriba — hay 18
> recorridos en cola y un vigilante horario que puede alquilar máquinas)*.
> `p40-lr-L4` terminó 20/20 (analizado en [docs/plan-lr-L4.md](docs/plan-lr-L4.md) §7: *el eje es
> plano, `lr`=0,0014 se queda*), y **la cadena de la CNN plana corrió sola y completa** — arrancó
> **23 min** después de que cerrara el recorrido, 41 runs, **22,5 h**. El watchdog
> `fv-plana-watchdog` se **desregistró el 2026-08-12** a petición del usuario: **no queda ninguna
> tarea programada `fv-*`** ni ningún proceso del proyecto vivo. Para rearmar la cadena (o correr
> otra), `scripts/plan_plana_watchdog.ps1` y el registro de la tarea están en
> [docs/plan-plana.md](docs/plan-plana.md) §5.
>
> **⚠ 2026-08-11 — LA PLANA: EL NÚMERO QUE SALE ES UN ARTEFACTO DE PROMEDIAR FALLOS.**
> Resultado completo en [docs/plan-plana.md](docs/plan-plana.md) §6. Respuesta nominal
> `n_layers=4, lr=0,0009`, y hay que leerla con cuidado:
> 1. **La profundidad ≥5 no es peor: NO ARRANCA.** L5 colapsa **1 de 5** semillas y L6 **2 de 5**,
>    con f1 exactamente **0,0000** (no entrenan peor: no despegan). **Entre las que sí arrancan, L5
>    (0,8612) y L6 (0,8606) GANAN a L4 (0,8491)**. `suggest_winner` corona L4 **por fiabilidad, no
>    por calidad**: la media de una mezcla bimodal no mide calidad.
> 2. **Es la misma firma que la foveada L5** (plan-40h §3), así que **no es de la arquitectura
>    foveada**: es de inicialización/optimización a profundidad ≥5 con esta cabeza.
> 3. ⚠ **Las dos métricas se contradicen EN EL SIGNO** sobre las semillas vivas: por ventana gana
>    L5 (+0,0121, **p=0,024** — cruza el umbral); **por tarea gana L4** (+0,0236, p=0,079 — no lo
>    cruza). **La única diferencia declarable del eje apunta donde la métrica que manda no
>    respalda.** Se publica el desacuerdo de signo, no un ganador. Segundo caso, y más nítido, de
>    la «reserva del proxy en ejes de profundidad».
> 4. ⚠ **Con los colapsos dentro, `proxy_vs_task.py` da Spearman +1,000** — concordancia perfecta
>    **por los ceros compartidos**. Quitarlos la invierte. Cuidado con correlaciones sobre mezclas.
> 5. ⚠ **Bug del ejecutor, arreglado**: `derive_base` aplica los `overrides` **después** de los
>    `winners`, así que un campo fijado en `base_network` **anula el arrastre del estudio en
>    silencio**. Fijar `n_layers` midió el paso de `lr` a L4 cuando el cribado había coronado L5.
>    No cambió la respuesta final, **por suerte**. Hay guarda nueva que se niega a arrancar.
> 6. **`lr` es plano también aquí** (0,0009 y 0,0014 empatan, δ=0,0020): misma meseta que la foveada.
> **Lo siguiente, y es decisión del usuario**: la comparación foveada vs plana con la familia de 6
> controles de [docs/plan-cnn-plana.md](docs/plan-cnn-plana.md) §3. Nada de eso se ha medido.
>
> **2026-08-09 — LA CNN PLANA YA SE PUEDE CONSTRUIR: `regions: single`, y F12 CERRADA.**
> El control del §6 de protocolo.md existía como pregunta desde el día 1. **Medido antes de tocar
> nada: poner la periferia a cero NO da una CNN plana** — `penetration = max(1, round(N·pen_frac))`
> nunca puede ser 0, así que con `periph_out=0` la máscara periférica se vuelve **un anillo sobre el
> borde de la propia ventana** (112/256 px con `pen_frac=0,1`; **60/256 incluso con `pen_frac=0`**).
> Había **dos grados de libertad confundidos en uno**: si la periferia comprime (`d`) y si hay dos
> ramas enmascaradas o una sola sobre todo el input (**ninguno**). Detalle en
> [docs/plan-cnn-plana.md](docs/plan-cnn-plana.md), con la familia de 6 controles y qué aísla cada uno.
> - **`build_masks` NO se toca**: en `single` no se llama. Ausente = `split`, y los nombres de módulo
>   no se mueven → **ningún artefacto cambia de significado y los checkpoints siguen cargando**
>   (test de identidad bit a bit + números dorados de la red que entrena ahora: 168.652 params).
> - ⚠ `derive_geometry` tenía cableado el suelo de «≥1 de periferia»: pedir la base plana devolvía
>   `ws16-p1-d1-L4`, una red **con anillo**, **sin que nada se negara**. Y un estudio no podía
>   declarar sobre qué red base corre → campos nuevos `base_network` y `c_frac` en el plan.
> - `fv.fovea.dims_of` unifica **los seis sitios** que derivaban geometría a mano.
> ⚠ **Verificado**: **150 tests** (+18), `npm run build` limpio, y la plana entrenada, un recorrido
> con base plana corrido y un **estudio completo** ejecutados de punta a punta.
>
> **2026-08-10 — EL EJE `lr` ES PLANO SOBRE L4: EL VIGENTE SE QUEDA, Y EL EJE SE CIERRA.**
> Recorrido `p40-lr-L4` terminado (20/20, **36,9 h** de cómputo contra 33,6 estimadas). Criterio
> escrito antes en [docs/plan-lr-L4.md](docs/plan-lr-L4.md) y comiteado sin un solo run (`68df83b`);
> el resultado y las cinco reglas aplicadas, en su §7. Lo que hay que saber:
> 1. **R1 ✅ el recorrido es válido**: los **20 runs** pararon por `patience` (32–71 épocas), ninguno
>    cerca del tope de 150. El tope alto era caro y era lo correcto.
> 2. **R3 ✅ el óptimo QUEDA ACOTADO**: gana un valor **interior** (`0,0006`, ventana 0,9293) y el
>    extremo izquierdo `0,00035` es **el peor** de los cuatro. La anomalía que motivó todo esto —un
>    ganador pegado al borde— **está cerrada**.
> 3. ⚠ **R4 ❌ pero no le gana al vigente: `lr` SIGUE SIENDO 0,0014.** `0,0006` da +0,0049 de
>    ventana con **p = 0,341** y +0,0066 de tarea con **p = 0,651**; el umbral escrito antes era
>    p ≤ 0,05. No porque 0,0006 sea peor: porque **no hay con qué distinguirlos**.
> 4. ⚠⚠ **HALLAZGO QUE VALE MÁS QUE EL RECORRIDO: δ y la permutación no dicen lo mismo, y δ es la
>    optimista.** Sobre los mismos 20 números, `suggest_winner` imprime *«el mejor punto despega del
>    resto»* (δ = 0,0020) mientras la permutación exacta da **p = 0,341**. δ es **1-SE de las
>    semillas del mejor punto y solo de ese**: ignora la dispersión del rival (aquí el doble) y 1 SE
>    no es una prueba de diferencia. Sirve como criterio de **empate** (protocolo.md §1.5), pero **la
>    frase que imprime afirma más de lo que el número aguanta** — y es la que lee un estudio OAT al
>    arrastrar un ganador. Los veredictos publicados **no se caen** (`n_layers` L4 vs L2 son 12× δ y
>    p = 0,032), pero **todo margen cercano a δ hay que releerlo**. **Pregunta abierta, no arreglada:
>    cambiar la regla de selección toca todos los estudios — decisión del usuario.**
> 5. **R5: la tarea ordena al revés (0,00035 el mejor) y no distingue nada** — p = 0,817 / 0,341 /
>    0,302. Por eso el Spearman de **−0,200** **no dice nada del proxy**: es ruido ordenando ruido.
>    `proxy_vs_task.py` lo declara solo ahora — guarda nueva que devuelve **`no_concluyente`** cuando
>    ningún par de tarea baja de p = 0,05, en vez de un «NO» que se leería como *el proxy falla*.
> 6. **La conclusión**: entre 0,00035 y 0,0014 el `lr` **no mueve la aguja** (amplitud 0,0062 de
>    ventana, 0,0096 de tarea). Con `d1000-lr-1`, que sí midió degradación **por encima**, queda una
>    **meseta ancha**: aquel estudio encontró **el borde derecho**, no un óptimo. **El eje se cierra
>    y no merece más cómputo.**
> **Artefactos**: recorrido `p40-lr-L4` (20 runs), `data/p40-lr-L4-task.json`. La tarea
> `fv-lrL4-watchdog` **ya está desregistrada**.

> **2026-08-08 — LA PROFUNDIDAD GANA: `n_layers=4` CONTRA LAS 2 DE HOY, SIN SOLAPAMIENTO.**
> Plan desatendido de ~37 h (30,4 h de cómputo, 24 runs) especificado **antes** en
> [docs/plan-40h.md](docs/plan-40h.md) y comiteado antes de que existiera un solo run (`b8545db`).
> Lo que hay que saber:
> 1. **`n_layers` 2 → 4 sube el f1 de ventana de 0,8756 a 0,9244**, 5 semillas cada uno, y **las
>    bandas son disjuntas**: la peor semilla de L4 (0,9105) gana a la mejor de L2 (0,8804). `f1` y
>    `loss` coinciden en el orden y ninguno declara empate (δ=0,0041). Cuesta 106 s/época contra 60.
> 2. ⚠ **`n_layers` NO es un eje de capacidad, es de campo receptivo.** El **97 % de los parámetros
>    está en la cabeza** (153.612 de 158.572); una capa más añade 4.640 (+3 %). Por eso **doblar los
>    canales no sirve** (`ch[32,32]`: +0,0046, dentro de δ, con 2× los parámetros y **más** coste) y
>    **agrandar el kernel tampoco** (`k_center=5`: −0,0063, peor que la base). Lo que gana es
>    **apilar capas**, no ensancharlas.
> 3. **El óptimo está acotado por los dos lados**: L3 = 0,9093 y L5 = 0,8832. Pero L5 **no es peor,
>    es inestable**: sus semillas son **bimodales** — 0,8471 / 0,8581 / 0,8620 contra 0,9279 /
>    0,9209. O arranca o no arranca, sin valores intermedios; sem 0,0170 contra 0,0041 de L4. Para
>    pasar de 4 capas hay que cambiar **algo más que el número** (residuales, otra inicialización).
> 4. ⚠ **El presupuesto de 20 épocas de los estudios `d1000-*` medía velocidad, no calidad.** Los 70
>    runs tienen su mejor época ≥16, y 65 seguían mejorando entre la 15 y la 20 (caída media 0,0127
>    de `val_loss`, **más de la mitad de la amplitud completa del eje de `lr`**). Con `patience=10`
>    la misma base pasa de f1 0,8437 a 0,8789. **`patience` mínimo seguro = 8**: la racha más larga
>    sin mejorar seguida de una mejora es de 6 épocas (medido sobre esos 70 runs).
> 5. ⚠ **`patience` mete varianza por la puerta de atrás**: cada semilla para donde quiere (32–71
>    épocas) y **la que entrena más lejos aterriza más abajo**. Las bandas de este recorrido son más
>    anchas que las de los estudios de época fija, y no es ruido de medición, es el criterio de
>    parada.
> 6. **Dos bugs del propio plan, encontrados corriendo** y anotados en plan-40h.md §7: la guarda de
>    presupuesto recortaba el rango **a los 3 valores más baratos, dejando fuera al ganador**; y
>    arrastrar la config ganadora **menos el eje** no basta — hay que soltar también **lo acoplado**
>    (`channels` con `n_layers`), o `check_run` rechaza la base. Es otra vez «el mismo dato en dos
>    sitios»: la profundidad vive en `n_layers` **y** en `len(channels)`.
> 7. ⚠ **El micro-benchmark de coste miente bajo carga**: medido con un entrenamiento ocupando los
>    núcleos daba 34 h para el mismo recorrido que, con la máquina libre, sale en 22 h.
> **Artefactos**: `p40-screen-{base,depth,width,kernel}` (cribado, 1 semilla) + recorrido
> **`p40-confirm-n_layers`** (20 runs, 4 valores × 5 semillas). Receta nueva `plan40`.
> **Lo que sigue teniendo más valor**: ~~(a) la métrica de tarea~~ **hecha, ver la nota de abajo**;
> (b) probar residuales para desbloquear >4 capas; (c) re-barrer `lr`/`batch_size` sobre L4, porque
> se fijaron sobre L2 y con 20 épocas — y el `lr` ganador quedó **pegado al borde izquierdo** de su
> rango, sin acotar.

> **2026-08-08 (2) — LA PROFUNDIDAD GANA TAMBIÉN POR TAREA, PERO LA MITAD Y SIN BANDAS DISJUNTAS.**
> Paso (a) de la nota anterior, hecho sobre los **mismos 20 runs**, sin reentrenar nada (5,4 s de
> inferencia por run). Detalle en [docs/metrica-de-tarea.md](docs/metrica-de-tarea.md) §2 ter y en
> [docs/plan-40h.md](docs/plan-40h.md) §8. Lo que hay que saber:
> 1. **`n_layers=4` gana con las dos métricas**: tarea **0,7796** contra **0,7572** de L2, y la
>    diferencia sobrevive a una **permutación exacta** de las semillas (**p = 0,032**, 252 arreglos).
>    L3 y L5 **no** se separan de L4 (p = 0,135 y 0,167). El split de este dataset son **200
>    imágenes** de val, así que el aviso de muestra pequeña ya no salta (`sem` por run ±0,023).
> 2. ⚠ **Corrige el titular de la nota anterior**: la ganancia se encoge a la mitad (+0,0488 →
>    **+0,0224**) y **las bandas se solapan** en tarea (peor L4 = 0,7532 < mejor L2 = 0,7689).
>    «Bandas disjuntas» era una propiedad **del proxy**, no del resultado. Cítese siempre así.
> 3. ⚠ **El f1 de ventana EXAGERA el hundimiento de L5.** Su bimodalidad —amplitud 0,081 en
>    ventana— es de **0,029** en tarea, y los dos grupos **se entrelazan**: una red que «no arranca»
>    según la ventana sigue reconstruyendo párrafos casi igual de bien.
> 4. **El criterio de §5.4 sale NO** (Spearman agregado **+0,800** < 0,90) y aun así **§5.5 no se
>    ejecuta**: el fallo es **un solo intercambio** entre L3 y L5, dos puntos que la tarea **tampoco
>    distingue** (0,0010 de diferencia, **p = 0,897**). No hay orden que acertar. Queda anotado como
>    reserva del proxy en ejes de profundidad, no como su refutación.
> 5. ⚠ **El cribado de 1 semilla no habría visto nada por tarea**: base 0,7523 vs depth 0,7532
>    (+0,0009 contra ±0,023). Que el plan funcionara fue suerte del proxy. Y **`k_center=5`, el peor
>    por ventana, es el mejor de los cuatro por tarea** (0,7594) — con 1 semilla no afirma nada,
>    pero **no queda descartado**: es el candidato barato si se vuelve a barrer estructura.
> 6. **Pieza nueva `fv.metrics.permutation_test`**: exacta hasta C(n+m,n) ≤ 200.000 y **se niega**
>    por encima en vez de pasar a muestreo en silencio (un p que cambia entre corridas no decide
>    nada). La imprime `proxy_vs_task.py` para todo recorrido con varias semillas — porque la
>    correlación dice si las métricas *coinciden*, nunca si la diferencia es *real*.
> ⚠ **Verificado, no razonado**: **132 tests** (+5), los **dos comandos del README reproducen sus
> números documentados** (+0,737 / +0,956 y el veredicto OK de `fast-lr-s0-lr`: sin regresión), y el
> comando nuevo ejecutado en PowerShell tal como está escrito. Detalle comiteado en
> `data/p40-n_layers-task.json`. **Cero entrenamiento, cero artefactos borrados.**

> **2026-07-28 — EL SOBRE DEL FICHERO SE ESCAPÓ POR LA LISTA: GUARDAR UNA RECETA DABA 400.**
> El usuario reportó `[unknown_recipe_fields] campos desconocidos: ['format_version']` en
> **Recetas**. Otra vez **el mismo dato en dos sitios**, y otra vez en una costura sin numerar:
> `RecipeStore.get()` quitaba el sobre del fichero (`name`, `format_version`) y **`list()` no**;
> la pantalla rellena su formulario con **una fila de la lista** y lo devuelve tal cual, así que
> `save()` rechazaba como «campo desconocido» **lo que el propio API acababa de servir**. Peor:
> el formulario se recuerda (`localStorage`), así que **un clic en cualquier fila envenenaba el
> guardado para siempre**, también con un nombre nuevo. Arreglo en la única definición posible —
> `fv.ioutils.strip_envelope` / `with_envelope` — aplicada a **las tres puertas** de las **dos**
> tiendas de config (D y C: `NetworkStore.list()` filtraba igual, sin dar error todavía) y al
> formulario, que ahora manda solo lo que D define. Documentado en formatos.md §4.3.
> ⚠ El contrato ⑦ frenó el import: ahora `settings` e `ioutils` son **hojas comprobadas** (el test
> verifica que no importan ningún dominio) en vez de excepciones concedidas a mano.
> ⚠ **Verificado, no razonado**: **123 tests** (+1: la vuelta lista→guardar, y que un campo de
> verdad desconocido **sigue dando 400**), `npm run build` limpio, y el flujo pulsado con
> Playwright **en la instancia del usuario** (clic en la fila → Guardar → 200), con **control**:
> con el código anterior la misma fila devolvía 400. La receta temporal creada se **borró**.
> **2026-07-28 (3) — EDITAR UNA RECETA ERA IMPOSIBLE DESDE LA UI (409 «elige otro nombre»).**
> Reportado por el usuario al intentar cambiarle un número a `mejorada`. Los dos stores aceptan
> `overwrite` **desde el día 1** y **ninguna pantalla lo enviaba nunca**, así que abrir una
> definición guardada y guardarla acababa siempre en `recipe_exists` con un consejo absurdo
> («elige otro nombre, o edita esa» — *estaba* editando esa). Lo que hay que saber:
> 1. **Regla nueva `U5.11`** (79 reglas), el reverso explícito de U5.8: **un run no se sobrescribe;
>    una red y una receta son fuente y se editan**. Dos acciones distintas — «Guardar» con nombre
>    nuevo, **«Actualizar» + confirmación** con uno que ya existe — para no confundir el accidente
>    con la intención. `overwrite` es bandera de la petición, **nunca campo del objeto** (test).
> 2. **La confirmación dice qué NO cambia y qué SÍ**: los runs y recorridos hechos copiaron los
>    valores; los **estudios que la fijan por nombre** re-resuelven en su próximo `advance`. Por eso
>    `GET /recipes` sirve **`used_by`** (`{receta: [estudios]}`) — **mapa aparte**, no dentro del
>    objeto: un campo mezclado ahí vuelve en el siguiente guardado como «desconocido» (la lección
>    del sobre, dos notas más abajo). Verificado con `corta`, que la fijan **5 estudios**.
> 3. **Redes tenía el mismo agujero** y el mismo arreglo (sin `used_by`: un recorrido congela
>    `base_network_value` al crearse, así que nadie fija una C por nombre para el futuro).
> ⚠ **Verificado, no razonado**: **127 tests** (+2), `verify_spec --live` **79 reglas, 65 ok, 0
> violadas, 82 %**, `verify_ui.py` 12 pantallas limpias, y la edición hecha **en la pantalla** sobre
> una receta temporal (3 → 12 épocas en disco, `overwrite` no guardado) que se **borró** después.
> La receta `mejorada` del usuario y las versionadas **no se tocaron**: de `corta` solo se abrió la
> confirmación —para leer los 5 estudios— y se canceló.

> **2026-07-28 (2) — UN OBJETIVO NO ES UN MONITOR, Y EL `<select>` LO ENSEÑABA MAL.**
> Descubierto al verificar lo anterior y arreglado a petición del usuario. El `<select>` de
> `monitor` se llenaba con los **objetivos** (`f1`, `loss`, `pos_err_px`); el default de la receta
> es `val_loss`, que no está en esa lista, y **un `<select>` cuyo value no está entre sus opciones
> dibuja la primera y calla** → enseñaba `f1` guardando `val_loss`. Lo grave no era la pantalla:
> 1. ⚠ **Elegir `f1` ahí habría corrompido `best.pt` en silencio.** `monitor_key("f1")` encuentra
>    el valor, pero la dirección vivía en un `frozenset({"val_f1"})` aparte → `f1` caía en
>    «menor es mejor» y el checkpoint se habría quedado con la **peor** época. Ni un aviso.
>    **Ningún artefacto está afectado**: los 708 `monitor` en disco dicen `val_loss` (medido).
> 2. **El mismo dato en dos sitios, otra vez**: `MONITOR_HIGHER_IS_BETTER` y `OBJECTIVES` eran la
>    misma tabla escrita dos veces, y las mitades no se conocían. Ahora **`fv.metrics.VAL_METRICS`
>    manda** (`MONITORS = val_ + cada métrica`) y `OBJECTIVES = dict(VAL_METRICS)`.
> 3. **Tres puertas dicen que no** con el mismo código `unknown_monitor`: la receta (guardar **y**
>    leer un yaml editado a mano), el eje `monitor` de un recorrido (`check_sweep`) y el de un
>    estudio (`validate_plan`). Antes **ninguna** validaba el valor.
> 4. **El API sirve el vocabulario** (`GET /recipes` → `vocabulary.monitor`) desde la constante
>    contra la que valida la puerta; y un valor que no pertenece se dibuja **`(no reconocido)`**,
>    nunca sustituido por uno plausible.
> 5. **Regla nueva `U5.10`** (78 reglas) con **verbo nuevo** `select_matches_served_vocabulary`:
>    el DOM no conserva rastro de la mentira, así que la comprobación viene de fuera (las opciones
>    **son** las servidas; lo mostrado **es** lo guardado).
> ⚠ **Verificado, no razonado**: **125 tests** (+2), `npm run build` limpio y el flujo pulsado en
> la instancia del usuario — opciones = vocabulario servido, enseña `val_loss`, un `f1` recordado
> sale `f1 (no reconocido)` y guardarlo devuelve `[unknown_monitor]` con razón y arreglo. Y
> `verify_spec --live`: **78 reglas, 64 ok, 0 violadas, 82 %** (U5.10 medida por su propio verbo),
> más `verify_ui.py` con las 12 pantallas sin un error de consola. Para correr `--live` hubo que
> **parar el vite del usuario** (con su permiso) y **se le devolvió levantado**.
> ⚠ La primera corrida volvió a medir al validador, no al código: `U5.10` reventó (`source:
> "GET /recipes"` se pasaba entero como ruta) y `U7.11` cazó que un **`data-testid` calculado no
> existe** para el escáner, que lee literales del fuente. Los tres selects lo llevan literal ahora.
> ⚠ **Corrección de la nota anterior**: en la primera verificación la semilla de `localStorage`
> usaba la clave sin el prefijo `fv.ui.`, así que **esa mitad no probó nada** (la del clic en la
> fila sí). Rehecha con la clave real: pasa. **Lección: una aserción que no puede fallar es peor
> que ninguna** — verificar el efecto (el fichero escrito), no solo la ausencia de error.
> **Queda abierto**: F16 sigue sin decidir (los enums `optimizer`/`scheduler` siguen copiados en
> `Recipes.tsx`; `monitor` ya no).

> **2026-07-28 — LISTAR NO ES VERIFICAR: LA REGLA U1.6 Y EL PLAN DE UN ESTUDIO.**
> El usuario reportó que **los parámetros de un estudio no vuelven a verse una vez creado**. Se
> documentó primero (a petición suya) y se implementó después. Lo que hay que saber:
>
> 1. **Regla nueva `U1.6`** en [docs/ui/1-estructura.md](docs/ui/1-estructura.md) — *un objeto
>    enseña entera la definición con que se creó, y la enseña en su detalle*. Va en el tipo 1 junto
>    a U1.5 («verificar un objeto no exige entrenar») porque es la misma exigencia. Fija cuatro
>    cosas: se lee **del objeto guardado**, nunca del formulario recordado (U7.3); **definición y
>    progreso separados** en pantalla como ya lo están en disco; los valores compuestos **completos**
>    (el rango de un eje es su lista, no su longitud) y el presupuesto **con unidad**; ausente se
>    dibuja como ausente. **77 reglas** ahora (78 desde U5.10, ver la nota de arriba).
> 2. **Estudios lo cumple**: bloque `study-plan` (B, receta D, objetivo, semillas, presupuesto) +
>    `study-axes` (la escalera con el rango literal y `hecho`/`en curso`/`pendiente` por eje, sacado
>    del progreso vivo); debajo, «El progreso (lo que ha pasado)». **`channels[i]` reconoce sus
>    propios sub-pasos** (`channels[0..L-1]`) — la única parte no trivial.
> 3. ⚠ **El bloque no enumera los campos que conoce y calla el resto**: lo que `plan.json` traiga y
>    la pantalla no nombre se pinta igual bajo su clave. Un campo añadido en Python **no puede
>    volverse invisible** aquí. Escribir la regla ya destapó uno: **`budget` no estaba descrito en
>    [docs/formatos.md](docs/formatos.md) §4.7** aunque lo guarda `POST /studies` y lo lee
>    `driver.advance`. Documentado.
> 4. **Y un fallo del propio validador**: `sibling_required` **dormía 300 ms fijos** esperando al
>    DOM — justo lo que `validador.md` §8 prohíbe por escrito tras habérselo cobrado ya una vez.
>    Ahora espera al selector; si no, el check de U1.6 habría sido intermitente.
>
> ⚠ **Verificado, no razonado**: `verify_spec --live` **63 ok, 0 violadas, 81 %** (tipo 1 al 100 %),
> `verify_ui.py` con las 12 pantallas y las aserciones nuevas sobre **los 5 estudios**, 122 tests,
> `npm run build` limpio. Los estados que ningún estudio real ejercitaba (`auto`, `channels[i]`
> expandido, un eje en cola) se probaron con un estudio temporal creado por HTTP y **borrado**
> después (arrastre comprobado antes: 0 recorridos). Para correr `--live` hubo que **parar el vite
> del usuario** (con su permiso) y **se le devolvió levantado**; su backend de `:8010` no se tocó.

> **2026-07-27 — LA ESPECIFICACIÓN DE LA UI, EN OCHO TIPOS, Y UN VALIDADOR QUE LA COMPRUEBA.**
> A petición del usuario se sintetizaron los **tipos** de especificación que rigen la UI y luego se
> construyó lo que los hace cumplir. Lo que hay que saber:
>
> 1. **`docs/ui.md` es ahora un índice**; las reglas viven en **`docs/ui/`, una por tipo** (1
>    estructura · 2 vistas · 3 representación · 4 datos · 5 invariantes · 6 números · 7 operación ·
>    8 léxico), **76 reglas numeradas y citables** (`U4.2`, `U6.7`…). El contenido se **movió**, no
>    se copió: dejarlo en los dos sitios era el modo de fallo que este proyecto tiene registrado.
> 2. **`scripts/verify_spec.py` valida esa especificación y se alimenta de ella**: cada regla lleva
>    pegado un bloque ` ```check ` (opción **A2**: el markdown manda). Motor híbrido (**C3**): 12
>    verbos declarativos + handlers nombrados; lo que no encaja se declara `substrate: none` **con
>    su razón** y sale `no_verificable`, nunca `ok`. Informe **por regla en cuatro estados**; salida
>    ≠ 0 solo con `violada`. **Diseño y lecciones: [docs/ui/validador.md](docs/ui/validador.md).**
> 3. **Medido: 81 % de cobertura mecánica** (`--live`: 62 ok, **0 violadas**, 5 no verificables, 9
>    no aplicables) y 39 % en estático. La cobertura **la calcula la herramienta**; no se mantiene a
>    mano en ningún documento.
> 4. **Lo que encontró, y estaba vivo**: los **estados de run escritos cuatro veces**, con una copia
>    esperando un estado `failed` **que no existe** y **ninguna** conociendo `interrupted` — un run
>    interrumpido no se marcaba terminal y **su curva se re-pedía en cada sondeo, para siempre**;
>    una **lista de objetivos** duplicada en `Recipes.tsx`; el **umbral `n<100` en dos sitios**
>    (ahora el veredicto `small_sample` viaja con el número); cuatro **colores literales** que eran
>    segundas definiciones de tokens —y al quitarlos apareció que el mapa secuencial **no seguía al
>    tema**—; y un **400 documentado en api.md que el código no daba**. Todo arreglado.
> 5. **`npm run validate:palette` existe** (era deuda declarada desde el día 1). Una implementación,
>    dos entradas: el validador de Python **ejecuta el mismo script**. Paleta medida: claro ΔE 9,1
>    (protan) y suelo de visión normal 19,6; oscuro 8,4 / 19,3.
> 6. ⚠ **Dos avisos de la herramienta sobre sí misma.** (a) Un check con `DELETE` **borró un dataset
>    real** antes de que existiera la guarda; se recuperó y se regeneró **bit-idéntico**, y ahora
>    todo lo que puede escribir está **bloqueado por defecto**. (b) De las 11 primeras «violadas»,
>    **10 eran checks mal escritos**: la primera corrida de un validador mide al validador.
> 7. **Cuatro preguntas abiertas anotadas, no decididas**: **F16–F19** en decisiones.md (si el API
>    debe servir estados y enums de receta; qué runs se ofrecen en Diagnóstico/Predecir; el alcance
>    del relieve del WARN de contraste; y si se anotan los componentes con `data-*` — hoy sí).
>
> ⚠ **Verificado, no razonado**: 122 tests en verde, `verify_spec --live` en 0 con backend y front
> propios (arrancados y **parados**), `verify_ui.py` con las **12 pantallas sin un error de
> consola**, `npm run build` limpio y `npm run validate:palette` en verde.

> **2026-07-26 — EL PROXY DE VENTANA VALE TAMBIÉN PARA C, Y LA PERIFERIA NO ESTÁ APORTANDO.**
> Cerradas las **Fases 3b y 4-código** de metrica-de-tarea.md; la **3 queda aplazada por decisión
> del usuario**. Lo que hay que saber, por orden de importancia:
>
> 1. **Fase 3b ✅ (§2 bis del doc).** Recorrido `proxy-c-d` (eje **`d`**, dominio C: 6 valores × 5
>    semillas, 20 épocas, **68 min** de CPU). **Spearman agregado +1,000**, mismo ganador (`d=2`)
>    por ventana y por tarea, dentro de la frontera δ. **`OBJECTIVES` NO cambia y §5.5 no se
>    ejecuta**: el ranking barato se queda. El criterio estaba escrito antes de mirar y es
>    **comprobable** — las constantes viven en `scripts/proxy_vs_task.py`, commiteado en `7dd34ad`,
>    antes de que el recorrido terminara. Reservas: el eje separa poco (amplitud 0,028; δ se come
>    3 de los 6 puntos), n=6, y es **un solo** eje de C.
> 2. ⚠ **LA PERIFERIA NO ESTÁ APORTANDO DE FORMA MEDIBLE** (§2 bis.1) — es media respuesta a *la*
>    pregunta del proyecto (protocolo.md §6) **sin construir el control que F12 bloquea**. El
>    máximo está en `d=2` (**4 px** de periferia); `d=1` —casi sin contexto— queda **segundo**; y
>    `d=6` (12 px) de los últimos. El coste no lo explica: 7,0–8,8 s/época en todos. Honestidad:
>    mejor−peor son **1,43 SE** y `d=5` rompe la tendencia, así que se afirma «no ayuda de forma
>    medible», **no** «estorba».
> 3. **El cuello de botella es la red, y es de DETECCIÓN.** §9.1: con esquinas perfectas la
>    reconstrucción actual da **0,97** (19/20 imágenes perfectas; el NMS no suprime nada), contra
>    0,6448 del mejor modelo real → los 0,33 que faltan son **todos** de detectar esquinas, no de
>    `_reconstruct`. §9.3: ni aflojando el IoU a 0,3 se pasa de **0,66** — un tercio de los párrafos
>    no se detecta en absoluto. **Ahí está el trabajo.**
> 4. **Ocho de las diez pruebas de §9, hechas — y dos corrigen al documento.** §9.4: la sd por
>    imagen **sube a 0,4148** (se suponía que bajaría de 0,372: es máxima con modelos
>    *intermedios*, y el 0,372 promediaba modelos de F1 0,10) → hoy **±0,093** por run, tabla de
>    §4.1 rehecha. §9.2: **los tres defaults de F están mal** y el óptimo es **el mismo** en tres
>    runs muy distintos (`threshold≈0,3`, `stride n/4`, `nms 3n/4`), acotados por dentro; deja
>    +0,065/+0,187/+0,261 sobre la mesa. §9.5: macro≈micro, pero **7 de 20 imágenes cargan casi
>    todo el fallo** y la 60 falla en 20/20 réplicas (con techo 1,000 → es la red). §9.7: `f1` es el
>    mejor proxy con diferencia (`loss` +0,780, `pos_err_px` +0,544, **y eligen otro ganador**).
>    §9.6: el `sem` aguanta el bootstrap (0,973×) → **el bloqueo es la `n`, no la fórmula**.
> 5. **Cuatro decisiones cerradas por el usuario** (decisiones.md): **F11 — no se regenera el dato
>    por ahora** (la métrica de tarea es *informe del ganador*, nunca criterio entre puntos; se
>    conserva la comparabilidad); **F13** aparcada con ella; **F15 — los knobs de F no se tocan**
>    pese a §9.2; **F14 — sí se registra que el holdout se tocó**, y está construido.
> 6. **Fase 4: todo el código, hecho y probado; falta la fuente** (depende de F11). Selectores de
>    dataset/split en el detalle de un run con el aviso de «se toca una vez»; y **F14**: cada
>    medición contra un holdout anexa una línea a `runs/<run>/holdout.jsonl` **también cuando el
>    número sale de caché** —ese era el vistazo invisible—, con `holdout_touches` en el payload y
>    en ámbar en la UI. Append-only: **registra miradas, nunca bloquea una**. Qué cuenta como
>    holdout lo dice **una sola función** (`is_holdout_source`): el campo `"holdout"` del
>    `dataset.json` manda **en los dos sentidos** y el nombre `-holdout` es el respaldo.
> 7. **Piezas nuevas reutilizables**: `fv.metrics.spearman` (empates a rango medio, **None** —nunca
>    0— si una serie es constante); `scripts/proxy_vs_task.py` (no calcula ninguna métrica por su
>    cuenta y **descuenta diciéndolo** los runs sin checkpoint); `sweep_trials`/`suggest_winner`
>    aceptan `objective=` para **re-leer** un recorrido con otro proxy sin tocar el spec, y lo
>    **declaran** (`objective_overridden`); `loader.source_meta` unifica los dos lectores que había
>    de `dataset.json`.
>
> ⚠ **Verificado, no razonado**: **122 tests en verde** (+15), 12 pantallas Playwright con el
> backend **reiniciado** (estaba stale de ayer), el camino de holdout probado **por HTTP en los dos
> sentidos** (200 con otra fuente / 400 `holdout_shares_source` con la misma), y el flujo del README
> **ejecutado de punta a punta con un holdout real** — dos miradas, dos líneas, la segunda marcada
> `from_cache`. Esas dos líneas se dejan en el repo a propósito.
> **Lo que sigue teniendo más valor**: (a) por qué la red no detecta un tercio de las esquinas
> (punto 3); (b) las 7 imágenes que fallan siempre, con Diagnóstico/Predecir; (c) si «la periferia
> no aporta» se sostiene con otra fóvea o con otro dataset. Ninguna necesita regenerar nada.
> **Queda sin hacer de §9**: 9.8 (vectorizar `build_view` — *no hasta que duela*) y 9.9 (F12).
>
> **2026-07-26 — LA MÉTRICA DE TAREA, CABLEADA (Fase 2 de metrica-de-tarea.md).** `paragraph_f1`
> ya no es una función que no llama nadie: hay **módulo nuevo `fv.task`** (contrato **⑬ E×A vía
> F**, escrito en organizacion.md §2) que puntúa un run **por imagen** contra los párrafos de la
> **fuente** — B guarda las imágenes pero no los párrafos verdaderos, así que la costura es
> `manifest["source_id"]` y sin fuente se falla (`task_needs_source`), nunca se puntúa contra las
> etiquetas de ventana. Es **caché, no entidad** (como el diagnóstico E×B), con los **knobs de F
> dentro de la clave** (cambiarlos obliga a re-inferir; el `threshold` del diagnóstico no entra en
> la suya porque allí solo se re-leen scores). Se puntúa **`best.pt`**, el fichero que sobrevive.
> Superficies: `GET /runs/{name}/task-score`, bloque en el detalle de un run, botón «medir la
> tarea del ganador» en el veredicto de Recorridos (solo sugerido + mejor), y `fv-oat/fv-study
> --task-score` (**apagado por defecto**: un recorrido nocturno no paga inferencia que nadie
> pidió). Todo número sale con `sem` y n de imágenes, y **con el aviso cuando n < 100**.
> **NO entra en `OBJECTIVES` y no se calcula por época** — §2 del doc lo desaconseja con datos.
> ⚠ **La comprobación que vale**: el código nuevo reproduce **exactamente** el número de la Fase 1
> (0,5353 de media en las 5 semillas del ganador de `fast-lr-s0-lr`, 1,9 s). **107 tests en verde**
> (+8: los 7 de §3.9 y el ⑬), 12 pantallas Playwright con los botones nuevos pulsados, los dos
> CLIs de punta a punta bajo cp1252. Se adelantó de la Fase 4 el parámetro `window_dataset` +
> `holdout_shares_source` (puntuar contra otro B); **falta el holdout en sí**.
> **Pendiente de este doc, por orden**: 3b (validar el proxy en un eje de **C** — el más barato y
> el que más puede cambiar el plan), 3 (regenerar el dato: **F11, decisión del usuario**), 4 (el
> holdout). **Las tres están especificadas al detalle** en metrica-de-tarea.md §§4-6 para que otra
> sesión las ejecute en frío: comandos reales con banderas verificadas, costes **medidos** (el
> recorrido de 3b son 30 runs × 140 s ≈ 70 min; los 6 valores de `d` pasan `check_run`), la pieza
> que falta (`spearman` en `fv.metrics` — **no hay scipy**, con números dorados para su test), y
> **todo lo que habría que tocar** si el proxy no valiera para C (§5.5: meter `paragraph_f1` en
> `OBJECTIVES` **no basta** — `sweep_trials` lee el `val` por época, que no tiene esa clave, y el
> ranking se quedaría en `None` sin avisar). Además, §9 lista **10 pruebas que valdría la pena
> hacer**, cuatro de ellas **sin entrenar nada** (el techo de la reconstrucción con esquinas
> perfectas; barrer los knobs de F, que nadie ha mirado nunca; la curva F1-vs-IoU; re-medir la sd).
> ⚠ **Y una corrección que manda sobre la Fase 3**: el dato real sale del **generador hermano +
> un resize que este repo NO tiene portado** — `make_synth_source.py` hace otro problema (barras de
> juguete). Nuevas decisiones registradas: **F13** (¿portar el resize?) y **F14** (¿registrar que
> el holdout se tocó?).
>
> **2026-07-26 — CÓMO SE ELIGE UN GANADOR: CUATRO ARREGLOS ENCADENADOS.** El usuario reportó que
> «el gráfico se detiene antes de terminar las épocas». No era el entrenamiento (en disco no
> faltaba ni una época: `patience: 0`, `stopped_early: false` en los 134 runs) sino la UI — y al
> tirar del hilo aparecieron tres cosas que sí decidían el ganador:
> 1. **La curva de un run terminal se congelaba** (`Sweeps.tsx`): se cacheaba en cuanto el estado
>    leía `done` Y había algo cacheado, pero ese algo venía del sondeo ANTERIOR, tomado mientras
>    el run entrenaba. Se perdían las épocas entrenadas en esa ventana (3 s normalmente; hasta un
>    minuto con la pestaña de fondo, que el navegador estrangula; más tras hibernar). Ahora un run
>    solo se «settle» al traerlo CON el estado ya terminal. **Verificado en vivo con control**:
>    4 runs × 6 épocas mirados sin recargar → 6/6 con el arreglo, **5/6 con el guard viejo**.
> 2. **El ranking medía la última época, no el checkpoint.** `best.pt` se elige por `monitor` y es
>    lo que cargan Diagnóstico/Predecir y lo que arrastra un estudio; el ranking usaba `m[-1]`.
>    Eran épocas distintas en el **63%** de los runs de `fast-lr-2-s0-lr`. Ahora `sweep_trials`
>    puntúa la época que guardó `best.pt` (`fv.metrics.checkpoint_record`, **la misma regla que usa
>    el bucle** — verificada contra el `best_epoch` de los 134 runs, 134/134). Sin checkpoint (el
>    monitor nunca midió) → `value: null` + razón, nunca la última época de consuelo. **Cambia el
>    ganador** de `fast-lr-2-s0-lr` (lr 0.0014 → 0.00168).
> 3. **No había regla de empate.** `select_winner` cogía `scored[0]` aunque `aggregate_seeds` ya
>    calculara la banda. protocolo.md §1.5 dice lo contrario. Ahora `δ` por defecto = 1-SE de las
>    semillas del mejor punto (`tie_delta`), con `tie`/`tie_reason` en palabras. Veredicto sobre
>    los recorridos reales: `batch_size` gana de verdad; `fast-lr-s0-lr` empata sus dos primeros;
>    **`fast-lr-2-s0-lr` empata los SEIS** — 30 runs que no distinguen nada. ⚠ `fv-study --delta`
>    tenía default `0.0`, que habría pisado la regla justo en el camino desatendido: ahora es None.
> 4. **La banda del gráfico se estrechaba sola**: promediaba las réplicas presentes en cada época,
>    así que un grupo con réplicas a distinta altura fingía converger al final. Se corta donde
>    faltan réplicas y se dice dónde.
>
> **2026-07-26 — LA MÉTRICA DE TAREA: EL PROXY VALIDADO (PARA D) Y EL RESTO ESPECIFICADO.**
> `paragraph_f1` existía desde el día 1 sin que la llamara nadie. Se ejecutó el **paso obligado**
> de protocolo.md §2 usando **runs ya entrenados** (cero entrenamiento, 40 s de inferencia):
> Spearman ventana↔tarea **+0,736** por run y **+0,956** agregado, sobre los 65 de `fast-lr-s0-lr`,
> con el **mismo ganador**. ⇒ **`OBJECTIVES` no cambia**: el proxy barato ordena igual que el caro.
> ⚠ Medido solo sobre un eje de **D**; para ejes de **C** (que cambian la vista) sigue sin medirse.
> El mismo trabajo destapó el límite real: sd entre imágenes 0,372 sobre **20 imágenes de val** →
> **±0,083** por run, más ruido que las diferencias a distinguir. **El bloqueo no es código, es el
> tamaño del val.** Todo lo pendiente —cablear `task_score` (módulo nuevo `fv.task`, contrato ⑬),
> validar el proxy en el eje `d`, dimensionar el dato, el holdout— está especificado al detalle en
> **[docs/metrica-de-tarea.md](docs/metrica-de-tarea.md)**, con firmas, payloads, claves de caché,
> tests y costes medidos. Dos decisiones quedaron **abiertas y son del usuario**: **F11** (regenerar
> el dato invalida la comparabilidad con los 130 runs actuales) y **F12** (qué es la «CNN plana
> equivalente» del primer experimento, que hoy no se puede construir por `no_periphery`).
>
> **2026-07-26 — EL PATRÓN DETRÁS DE CASI TODOS LOS FALLOS: «el mismo dato en dos sitios».**
> A petición del usuario se analizó el historial de arreglos y sale un modo de fallo dominante:
> un hecho representado dos veces (escritor↔lector, fuente↔caché, productor↔consumidor,
> puerta↔puerta, cabecera↔celda) donde solo una copia se actualizó. Tres propiedades lo hacen
> caro: **el conflicto se resuelve por precedencia y no por detección** (nadie lanza), **el
> resultado parece correcto** (un número plausible, no un crash), y **lo encuentra el usuario, no
> los tests** (un test unitario prueba UN lado de la costura). Se concentra en fronteras **que
> nadie numeró** — los contratos ①–⑫ protegen bien lo que cubren.
> Barrido a fondo: se eliminaron las **cuatro copias vivas** que el front tenía (defaults de C,
> ejes de geometría, objetivos, orden de esquinas); **dos ya habían divergido**. Ahora el API
> sirve cada vocabulario desde su única definición (`/networks` → `full_config({})`, `/sweeps/axes`
> → objetivos + `loss_weight_params` + `window_size_fields`, y `corner_order` viaja en todo payload
> indexado por él). Al servir los objetivos apareció un **hueco real**: `validate_plan` era **más
> laxa** que `check_sweep` para el contrato ⑨ (objetivo `loss` + eje de peso de la pérdida) — el
> `<select>` de Estudios lo tapaba no ofreciendo `loss`. Cerrado en la puerta. También la columna
> «siguiente» de Estudios decía el dataset porque `next_axis` solo existía en el detalle: ahora
> ambos salen de `fv.studies.driver.summarize`. **99 tests en verde** (+4 de costura).
> ⚠ **Regla de trabajo que se deriva de esto:** antes de cambiar la forma o el significado de un
> campo compartido, buscar **todos** sus lectores; y todo dato derivado, una definición y dos
> lectores — nunca dos definiciones.
>
> **Regresión propia detectada por el usuario y arreglada el mismo día:** cambiar `studies.delta`
> de número a cadena dejaba **Estudios en blanco** al abrir un estudio con un paso sin confirmar
> (el 0 recordado del navegador llegaba a `delta.trim()`). Dos capas: `usePersistedState` **rechaza
> un valor recordado cuyo tipo no encaje con el default** (deriva de esquema, no preferencia — mata
> toda la familia), y hay **error boundary por ruta**: una pantalla que revienta ya no borra la app,
> muestra la razón y ofrece olvidar las preferencias. `verify_ui.py` siembra el valor viejo y hace
> click en TODOS los estudios. **Lección: una página en blanco es el fallo silencioso definitivo —
> verificar en la UI, no solo con tests.**
>
> UI nueva: columnas «época»/«última» en el ranking, aviso cuando `monitor != objective` (el caso
> de los tres recorridos vivos), y componente `WinnerVerdict` — **Recorridos estrena veredicto**,
> antes esa pantalla no mostraba ganador ninguno. Los CLIs (`fv-oat`, `fv-sweep`, `fv-study`) lo
> imprimen también; `tie_reason` es ASCII a propósito (llevaba una δ griega, **que no existe en
> cp1252** y habría matado un estudio nocturno en la última línea — reproducido y arreglado).
> **95 tests en verde** (+7), 12 pantallas Playwright sin errores, `fv-oat` y `fv-study --auto`
> ejecutados de punta a punta (también bajo `PYTHONIOENCODING=cp1252`).
>
> **2026-07-25 — BORRAR UN ESTUDIO ARRASTRA SUS RECORRIDOS (antes los dejaba huérfanos).**
> `StudyStore.delete` borraba solo el estudio (plan+progress), no los recorridos que generó
> (`{estudio}-s{i}-{eje}`). Al borrar y **recrear un estudio con el mismo nombre**, el recorrido de
> la versión anterior quedaba huérfano y el siguiente `advance` chocaba con `sweep_exists` (nunca
> sobrescribe — correcto) sin salida. Ahora `fv.studies.driver.delete_study` **arrastra** a los
> recorridos del estudio (por `spec.study`, vía `SweepStore.used_by_study`) y estos a sus runs
> (reusa `delete_sweep`); rechaza ANTES de borrar nada si algún recorrido/run está `queued`/`running`
> (`study_has_live_sweeps`, 409 — R4). El endpoint `DELETE /studies/{name}` usa el arrastre. **87
> tests en verde** (+2 de contrato: ciclo borrar→recrear→avanzar sin colisión; guarda de vivo).
> ⚠ Espejo del arrastre recorrido→runs; simétrico con la auditoría CRUD del 2026-07-25.
>
> **2026-07-25 — SEMILLAS DEL ESTUDIO: N SEMILLAS EN CADA PUNTO (antes se ignoraban).** El `seeds`
> del plan de un estudio se **validaba y guardaba pero nunca generaba runs**: cada punto del eje se
> entrenaba con una sola semilla (la del recipe base), pidieras 1, 3 o 5. Por decisión del usuario
> se implementó **N semillas en cada punto**: el generador (`fv.sweeps.generate.build_generated_spec`)
> añade un **segundo eje `seed=[s0..s0+N-1]`** junto al eje real cuando `seeds>1`, así el barrido
> produce `len(rango)·N` runs (uno por `(valor, seed)`) y la banda existe en todo el eje. Cableado:
> `studies.advance` pasa `plan["seeds"]`; también `fv-oat --seeds` y `POST /sweeps/generate {seeds}`.
> El nombre del run lleva la semilla (`…-d2_seed1 .. _seed3`, `point_run_name` seed-aware). El
> **ganador agrega por valor de eje** (`fv.sweeps.winner.aggregate_seeds`): rankea la **media** de
> las semillas + banda min–max + `n_seeds`, nunca la réplica con suerte (recipe.py: seed es el eje
> RÉPLICA). `seeds=1` = sondeo rápido de antes (sin eje de semilla). Divergencia registrada en
> [barrido-por-ejes.md](docs/barrido-por-ejes.md) §11.1: el esquema D-M1 «confirmación N-en-frontera»
> queda como modo alternativo NO construido. **85 tests en verde** (+5 de contrato); verificado en
> vivo con `fv-oat --seeds 2` sobre `synth-b16` (2 runs, seeds 1/2 distintas en disco, ganador
> agregado). ⚠ **El estudio `batchSize_fast-80px-5seeds` que ya corría se generó ANTES del arreglo:
> su recorrido no tiene eje de semilla (sigue en 5 runs, seed=1). Para obtener las 5 semillas hay
> que borrarlo y recrearlo.**
>
> **2026-07-25 — CURVAS DEL RECORRIDO: OVERLAY MULTI-RUN EN RECORRIDOS.** Al elegir un recorrido,
> la pantalla **Recorridos** superpone las curvas de val (loss / f1 / pos_err_px) de **todos sus
> runs** en tres small-multiples (la misma vista V14 de RunDetail, pero N líneas). Dos modos, por
> decisión del usuario: **líneas por run** (una por run, leyenda de casillas para ocultar/mostrar
> cada uno o todos, énfasis al pasar el ratón) y **media ± banda** (agrupa runs que comparten
> config salvo `seed`: línea media + sombra min–max; con 1 semilla la banda es degenerada, a
> propósito y anunciado en el subtítulo). El color **sigue a la entidad** (índice de trial / orden
> de grupo), nunca al rank, así ocultar/ordenar no repinta a los demás. La paleta de 8 hues vive en
> `tokens.css` (`--series-1..8`, claro/oscuro), validada CVD para las dos superficies; identidad
> siempre por leyenda+etiqueta, nunca por color solo. Los datos ya existían: `/sweeps/{n}/trials` +
> `/runs/{r}/metrics` (fan-out en el poll de 3 s; runs terminales se traen una vez). **Cero rutas
> nuevas de backend.** Verificado: `web` typecheck+build limpio; Playwright sobre el recorrido vivo
> `batchSize_fast-80px-s0-batch_size` (5 runs) — ambos modos, toggle y ocultar/mostrar todo, **sin
> errores de consola**. `LineChart` extendido (banda + serie atenuada + leyenda opcional) sin
> romper RunDetail. `SweepCurves.tsx` es nuevo. ~~⚠ `scripts\verify_ui.py` sigue apuntando a
> `test4-s0-n_layers` (borrado)~~ — **RESUELTO 2026-07-26**: reapuntado a `fast-lr-2-s0-lr` y
> ampliado (época, aviso de monitor, veredicto, los dos modos de curva). El «runs terminales se
> traen una vez» de arriba era el bug del gráfico congelado: ver la nota del 2026-07-26.
>
> **2026-07-25 — AUDITORÍA CRUD CROSS-PÁGINA + REFRESCO DE LISTAS EN VIVO.** Se revisó el CRUD de
> las 12 pantallas buscando «borrar aquí rompe allá». Arreglos:
> - **Diagnóstico/Predecir** ya no piden un run recordado en `localStorage` que fue borrado o
>   renombrado (la migración de sufijo de eje dejó nombres huérfanos): **gatean por pertenencia a la
>   lista, no por verdad**, y la doomed-request ya no revienta ni pisa la carga válida. Además
>   **refrescan la lista de runs en vivo** (sondeo 3 s, como Runs/Recorridos/Estudios), gateando la
>   carga pesada con un booleano `runReady`/`sourceReady` estable para no re-computar en cada pasada.
> - **Borrado cruzado por-nombre (R4):** borrar un **dataset B** que fija un **recorrido**
>   (`spec.window_dataset`) o un **estudio** (`plan.window_dataset`), o una **receta D** que fija un
>   estudio (`plan.base_recipe`), devolvía 200 y rompía la otra pantalla *dentro del job* al
>   reanudar/avanzar. Ahora se rechaza en la puerta con **409 + razón+arreglo** nombrando al referente
>   (`SweepStore/StudyStore.used_by_dataset`, `StudyStore.used_by_recipe`; código `recipe_in_use`). La
>   asimetría es deliberada: las referencias **instantánea** (run/recorrido copian los VALORES de C/D
>   inline) SÍ permiten borrar C/D sin romper el run ni su diagnóstico.
> - **Verificado:** `tests/test_crud_integration.py` fija el grafo (instantánea no obliga, por-nombre
>   sí; borrar el recorrido de un estudio lo deja legible). **80 tests en verde**; guards probados en
>   vivo por HTTP; ciclo crear→borrar y refusal-en-UI por Playwright, 12 rutas sin errores de consola.
> - **Artefactos:** por decisión del usuario **no se borró nada**. Todos los runs/recorridos/estudios
>   actuales **cargan** — la nota de §13 sobre checkpoints incompatibles (`fov-run-*`, `cli-run-1`)
>   quedó **OBSOLETA**: hoy cargan y se diagnostican/predicen sin error.
>
> **2026-07-24 — EJES DE BARRIDO: `N`/`c_frac` REHUSADOS + BUDGET NO COLAPSA `epochs` +
> VERIFICADOR DE TODOS LOS EJES.** Un recorrido generado por un estudio (`test2-s0-N`) quedaba en
> `done 0/3` sin razón visible: barría el eje `N` con `c_frac` fijo, y como `center_out =
> round_to_even(N·c_frac)` está atado por ①a al `window_size` del dataset, **cada punto** daba una
> fóvea != la ventana. `expand_points` solo validaba geometría (`check_network`), no la costura con
> el dataset (`check_run`), así que los 3 puntos pasaban como "válidos" y morían con
> `window_size_mismatch` dentro del job (`RunError`→`continue`) — la trampa que R4 prohíbe. Arreglos:
> - **`N` y `c_frac` no son ejes** (`WINDOW_SIZE_FIELDS`): se rechazan en las **dos puertas** —
>   `check_sweep` (H) y `validate_plan` (I)— con `axis_breaks_window_size` (razón + arreglo: barre
>   `d`, o usa un dataset con esa ventana), antes de reservar nada. Como ningún otro eje toca
>   `center_out`, `check_network` por punto sigue bastando.
> - **Budget no colapsa `epochs`:** el runner solo aplica `budget.epochs` si el punto no barre
>   `epochs` (antes lo pisaba siempre → el eje no hacía nada, en silencio).
> - **`scripts\verify_axes.py`:** corre un recorrido real por CADA eje de C y D. Verificado hoy:
>   **26/26 ejes** (11 red + 13 receta + `N`/`c_frac` rehusados), 0 fallos. Ejes probados listados
>   en el README (§«Qué ejes se pueden barrer»).
>
> **74 tests en verde**. Artefactos muertos del usuario borrados (`studies/test2`,
> `sweeps/test2-s0-N`, `sweeps/test1-s0-N`). Fresco y cargable sigue siendo `fov-16-param` +
> `oat-d-demo`.
>
> **2026-07-24 — PARADA DE RECORRIDOS: CORTE EN VUELO + RECONCILIACIÓN DE ESTADO MUERTO.** Dos
> arreglos sobre la parada de recorridos (H):
> - **Corte del punto en vuelo (feat 1):** `train` acepta `should_stop`; el runner le pasa
>   `lambda: store.stop_requested(name)`, así una parada pedida al recorrido corta el punto **en
>   marcha** en su siguiente frontera de epoch, no solo entre puntos. La cooperación seguía siendo
>   entre puntos: un run largo ignoraba la parada hasta acabar.
> - **Reconciliación de `running` muerto (feat 2):** quien marca un recorrido/run `running` graba
>   su **PID dueño** (`fv.proc.pid_alive`, portable Win/POSIX sin psutil). `SweepStore.reconcile`
>   y `RunStore.reconcile` sanan `running`→`interrupted` cuando el proceso dueño ya no existe
>   (caída/reinicio del API/hibernación) — se llama al leer (`GET /sweeps`, `GET /sweeps/{name}`,
>   `sweep_trials`). Cierra la trampa heredada "un crash queda running para siempre". Erra seguro:
>   dueño vivo o sin PID (legacy) → intacto. El runner ahora redó todo lo no-(done|cancelled), así
>   que un punto `interrupted` se rehace al reanudar. `interrupted` es terminal: borrable y
>   reanudable; badge ámbar en la UI. **69 tests en verde**, 12 pantallas Playwright sin errores,
>   reconciliación verificada por HTTP. Causa raíz de `test1-s0-n_layers` pegado en running: su job
>   murió y nadie leía el `stop.json`.
>
> **2026-07-24 — BARRIDO POR EJES (OAT) IMPLEMENTADO Y VERIFICADO.** Se construyeron las cinco
> piezas de [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) §14, respetando las decisiones
> cerradas de su §13:
> - **Builder paramétrico (C)**: `fv.models.builder` honra `n_layers` y `channels` por capa
>   (D-C3), stride solo en la 1ª capa (D-S1); no-regresión bit-exacta para `n_layers=2` con
>   `channels=[16,32]`. Lee `ch1/ch2` viejo, escribe siempre `channels`.
> - **Derivador de base (G/C)**: `fv.models.derive` — de `window_size` deriva `N`/geometría
>   (①a), defaults estáticos, ganadores arrastrados, corrección de inválidos con razón, `N` mínimo
>   (D-G2), afloje de `c_frac` con razón (D-G3).
> - **Base inline + generador P1 (H)**: `fv.sweeps.generate` + CLI **`fv-oat`** + `POST
>   /sweeps/generate`. `base_network=null` + `base_label` + `derivation` (D-H2). Barrer `n_layers`
>   redimensiona `channels` a `[16]*L` (§6.1).
> - **Arrastre del ganador (I/H)**: `fv.sweeps.winner` — regla coste/calidad con δ (D-W1),
>   sugiere y el usuario confirma; `GET /sweeps/{name}/winner`.
> - **Estudio OAT (I, dominio nuevo `studies/`)**: `fv.studies` (plan.json comiteable +
>   progress.json vivo), CLI **`fv-study`**, endpoints `/studies/*`, pantalla **Estudios**. Guía
>   y no ejecuta; expande `channels[i]` al fijar `n_layers`.
>
> **65 tests en verde** (~25 s) incluyendo el contrato ⑫ (estudio↔recorrido) y **las 12
> pantallas con Playwright sin un solo error de consola** (`scripts\verify_ui.py`). README con
> comandos **ejecutados** (`fv-oat`, `fv-study` verificados de punta a punta).
>
> ⚠ **Consecuencia de §13 (deuda de pesos = 0):** el builder paramétrico renombró los módulos
> conv (`center_conv1` → `center_convs.0`), así que **los checkpoints previos ya no cargan**.
> `load_model` lo rechaza limpio (`checkpoint_incompatible`, 400) en vez de un 500. Los runs de
> ejemplo antiguos (`fov-run-*`, `cli-*`, `dirty-*`, `rec-d`, `demo-seeds`) quedan **no cargables
> por diseño**: reentrénalos o descártalos. Fresco y cargable: run **`fov-16-param`** + recorrido
> inline **`oat-d-demo`** (base `ws16-p2-d2-L2`). Diagnóstico/Predecir usan `fov-16-param`.
>
> ---
>
> **2026-07-21 — IMPLEMENTACIÓN BASE COMPLETA Y VERIFICADA.** El sistema entero está construido y
> probado de punta a punta en esta máquina: paquete `fv` (fovea/datasets/windows/models/
> training/inference/diagnostics/sweeps/studies/validation/metrics/matrixview), API FastAPI, front
> Vite+React con las **diez pantallas**, CLIs (`fv-extract`, `fv-train`, `fv-sweep`, `fv-oat`,
> `fv-study`, `fv-api`) — un test por contrato más el muestreo foveado
> contra los números de la spec. Verificado además: el flujo completo por HTTP (extract →
> train → diagnóstico → predict → sweep), los CLIs (bit-idénticos al API con la misma
> semilla), las negativas con razón+arreglo antes de reservar nombre. El README lleva los comandos
> **ejecutados**, no razonados.
>
> **Decisiones cerradas en la implementación** (registradas en decisiones.md §4): F1=C9
> (cabezas de esquina), **F1b=C10: las esquinas se etiquetan SOLO sobre la fóvea**
> (`center_out == window_size` de B, contrato ①a; la periferia es contexto), C11 (relleno
> `pad_mode: edge` + máscara de cobertura solo para depurar — F0), C12 (anillo por pooling
> anisótropo por zonas, co-registrado: el código §5 de la spec no tipa para d>1, ver
> decisiones.md).
>
> **Datos de ejemplo vivos en el repo**: fuente `local/synth-01` (60 img 96×72, regenerable
> con `scripts\make_synth_source.py`), dataset `synth-b16`, red `fov-16` (migrada a `channels:
> [16,32]`), recetas `corta`/`media`. **Cargables con el builder actual**: run `fov-16-param` y
> recorrido inline `oat-d-demo`. Los runs/recorridos anteriores (`fov-run-*`, `cli-*`, `dirty-*`,
> `rec-d`, `demo-seeds`) conservan su metadata comiteada pero **sus checkpoints no cargan** (§13):
> son historia, no artefactos vivos.
>
> **Pendiente, por orden de valor**: (1) el primer experimento real (protocolo.md §6:
> ¿fóvea+periferia gana a una CNN plana de coste equivalente? — control con `d=1`/`c_frac`→1
> o red plana equivalente, N semillas, criterio escrito antes); (2) el holdout y el dato de
> verdad (fuente del generador reducida con resize — el resize aún NO está portado, decisión
> al llegar); (3) V16/occlusion pre-muestreo (diseño en [docs/ui/2-vistas.md](docs/ui/2-vistas.md)); (4) poda en el runner de
> recorridos (hoy corre todos los puntos); (5) pantalla Entrenar: el estimador solo usa runs
> comparables (hecho) pero no hay curva de coste por punto del recorrido.
>
> **Servidores dev**: ~~al cerrar esta sesión quedaron corriendo backend (:8010) y vite (:5173)~~
> — **OBSOLETO (2026-07-27)**: se pararon, y la regla ahora es cerrarlos siempre al terminar
> (ver «Convenciones»).
>
> Nada de lo documentado está construido ni verificado. Cuando un documento cita código
> (`loop.py:166`, `extract.py:127`), habla del **proyecto hermano** — es la evidencia que motivó
> el diseño, no código de este repo.

**Al terminar una fase, actualiza estas líneas.** Es lo único que le dice a la siguiente sesión
dónde está.

---

## La web app corre como SERVICIO, y el ajuste sobrevive a rehacer la máquina

**Desde el 2026-08-29.** Antes la app eran dos terminales (`fv-api` + `npm run dev`)
que alguien tenía que abrir a mano después de cada `lanzar launch dev`; en un server
desechable eso significa que **no está** casi nunca. Ahora es una unidad de systemd,
`foveal-vision-web`, y la app entera vive en **un** proceso.

```bash
python3 scripts/web_app.py preparar | estado | url | abrir | cerrar | parar | arrancar | log
```

Desde Telegram: `/use fvweb`. Sin argumentos da el estado.

### Las cinco decisiones que hay que respetar si se toca

1. **UN proceso, no dos, y la razón de peso es del lanzador**: `selected_services` en
   `do_droplet.py` **rechaza** dos servicios del mismo repo, porque comparten directorio
   y `.env`. Además el servidor de vite es una herramienta de desarrollo. Así que
   `fv.api --web` sirve `web/dist` en `/` y monta el API en `/api`.

2. ⚠ **El API va bajo `/api` y NO en la raíz** — y esto es lo que muerde: las rutas del
   front y los recursos del API **colisionan** (`/runs`, `/sweeps`, `/studies`,
   `/networks`, `/recipes` son cada uno una **pantalla** y un **recurso**). Servido en la
   raíz, abrir «Runs» en el navegador devuelve el JSON del API. `/api` es además el
   prefijo que `web/src/api.ts` ya manda y el que el proxy de vite quita en desarrollo:
   **el front no cambia**. Tiene test.

3. **Expuesta pide token, y se niega a arrancar sin él.** El API **borra** datasets,
   runs, recorridos y estudios sin preguntar, así que `fv.api` con un `--host` que no sea
   local y sin token **no arranca** (R2 b: o degrada con un defecto declarado, o falla
   *antes de empezar*). Vale también para `--host 0.0.0.0` **sin** `--web`: ahí también
   se le instala la puerta, porque sigue siendo el API entero publicado.

4. ⚠ **Loopback entra SIN token, y eso sólo vale mientras nada haga de proxy delante.**
   Sin proxy, `request.client.host` es el par real, así que `127.0.0.1` significa «ya está
   dentro de la máquina» — y es como `cerrable.mjs` pregunta si hay un entrenamiento vivo,
   y como no se rompe el flujo de desarrollo. **Si algún día se pone un reverse proxy, esta
   regla se va con él**: detrás de un proxy todas las peticiones parecen locales y la
   puerta quedaría abierta para todos. Escrito también en
   [src/fv/api/web.py](src/fv/api/web.py).

5. **`web/dist` no se commitea** (`dist/` está en `.gitignore`, y aquí se versiona la
   descripción, no la carga). Lo construye `preparar` con `npm ci` —no `install`— para que
   al server llegue exactamente el lockfile medido.

### El token: de dónde sale, y por qué en ese orden

1. `FV_WEB_TOKEN` del entorno. 2. `<repo>/.env`, que es lo que escribe el lanzador con
`env_prefix: "FVW_"` — **el único camino por el que un token sobrevive a rehacer el dev**,
así que con `FVW_WEB_TOKEN` en el `.env` del lanzador la URL se puede marcar en el móvil
una vez y ya. 3. `~/.config/fv-web.env` (modo 600), generado en la máquina la primera vez:
efímero como ella, y se pregunta con `/use fvweb` → `url`.

⚠ **El token viaja en `?t=` la primera vez** porque es la única forma de dárselo a un
navegador de móvil. Queda en el log de acceso de este proceso; en cuanto entra, se guarda
en cookie y se **rebota a la misma ruta sin él**, para que deje de estar en marcadores,
enlaces y `Referer`.

### El freno va con el acelerador (R11), y aquí no era el obvio

Esto **no** gasta dinero nuevo: el droplet factura por existir. Lo que introduce es una
forma nueva de **perder trabajo**: un entrenamiento lanzado desde el navegador vive en un
**hilo** de `fv.api` (`JobQueue`, `max_workers=1`), no en un proceso propio — así que
`cerrable.mjs`, que casa **líneas de comando**, no podía verlo, y un barrido lanzado desde
el móvil se habría perdido con el veredicto en 🟢. Ahora `cerrable.mjs` le pregunta a
`/api/jobs` por loopback; seis tests en `telegram-coordinator/tests/cerrable-webapp.test.mjs`.

El otro freno es la **exposición**: `web_app.py cerrar` cierra el puerto en `ufw` sin parar
el proceso — lo que se quita es que se llegue desde fuera, no el trabajo que haya dentro.

### Dónde vive cada pieza, y por qué ahí (R7)

| Pieza | Dónde | Por qué |
|---|---|---|
| cómo se sirve, se prepara y se arranca | **aquí** (`src/fv/api/web.py`, `scripts/web_app.py`) | es de quien produce la app, no de quien la transporta |
| que exista como unidad de systemd | lanzador (`services/foveal-vision-web.json`) | es el mecanismo declarado del lanzador para un proceso de larga vida, y da gratis `service logs`, `update` y el reinicio al hacer `git pull` |
| que **toda** máquina `dev` la traiga | lanzador (`types/dev.json`) | si hay que acordarse de un `--service`, tarde o temprano no se pide |
| instalarla en una máquina YA viva | lanzador (`do_droplet.py install-service`) | `update` traía código y reiniciaba lo instalado, pero una unidad **nueva** sólo la escribía `provision`: declararla llegaba a las máquinas futuras y no a ésta |

⚠ **El puerto lo abre `preparar` en `ufw`, no `cloud-init`**: cloud-init vale para **todas**
las máquinas y este puerto sólo tiene sentido donde corre este servicio.

⚠ **Y el `:8010` puede estar ocupado por un `fv-api` lanzado a mano.** Pasó el 2026-08-29
con uno de `~/ws/tema-2` puesto con `setsid nohup`, huérfano de la sesión que lo arrancó.
Un servicio con `Restart=always` contra un puerto ocupado no falla de una vez: se reinicia
en bucle con un «address already in use» que parece un fallo suyo. Por eso `estado` dice
**de quién es** el puerto (pid y **cwd**, que es lo que identifica al dueño con varios
workspaces, misma lección que `cerrable.mjs`).

## Los datos de los estudios van a `foveal-vision-data`, no aquí

**Desde 2026-08-27, por decisión del usuario: todo dato generado por un estudio se guarda en el
repositorio hermano [`foveal-vision-data`](https://github.com/stalinbeltran/foveal-vision-data).**
Este repo es **el código que mide**; aquel es **lo medido**. El objetivo es mantener los dos
limpios: 3.256 ficheros JSON de resultados ahogaban el diff de cualquier cambio de código.

**Qué va allá** — los artefactos de los dominios **E, H, I**: los runs (`runs/<name>/`: `config.json`,
`metrics.jsonl`, `status.json`, `summary.json`), los recorridos (`sweeps/<name>/`: `spec.json`,
`state.json`, `informe.json`, `flota.json`) y los estudios (`studies/<name>/plan.json`).

**Qué se queda aquí** — todo lo que es código o criterio, no medida:
- `configs/networks/` y `configs/recipes/` (**C** y **D**): son **fuente**, se editan a mano y
  definen el experimento; no los genera un estudio.
- `docs/plan-*.md`: se escriben **antes** de medir. Son criterio, no resultado (protocolo.md §1).
- `reportes/`: **ya no vive aquí.** Desde el 2026-08-29 los reportes van al repo central
  [`estudios-redes-neuronales`](https://github.com/stalinbeltran/estudios-redes-neuronales); aquí sólo se queda el **dato crudo** que producen
  (`reportes/2026/08-agosto/datos/*.json`).
- `benchmarks/`: caracterizan **la máquina**, no un estudio.

**La estructura de allá es `<año>/<NN>-<mes>/`** (`2026/08-agosto/`), fechada por la **fecha de
generación leída del propio JSON** (`status.json.updated_at`) — **nunca por el mtime**, que en un
clon limpio es la fecha del checkout y fecharía todo el mismo día. Dentro:
`sweeps/<recorrido>/runs/<run>/` — **un run vive dentro de su recorrido**, para que la relación
sea estructura de directorios y no un prefijo en el nombre que se pierde al renombrar. Un
recorrido **no se parte por el mes**: sus runs heredan su fecha. `index.json` mapea
run → ruta/recorrido/fecha y estudio → sus recorridos. El detalle y los criterios, en el README
de aquel repo; el migrador que lo produjo, en su `scripts/migrar_data.py`.

⚠ **Lo que esto NO cambia todavía, y hay que saberlo antes de tocar código:**
1. **La migración fue una COPIA: `runs/`, `sweeps/` y `studies/` siguen en este repo**, y el
   código sigue leyéndolos de aquí. Quitarlos es un segundo paso pendiente de hacer.
2. **El paquete `fv` está limpio**: todas las rutas pasan por
   [`src/fv/settings.py`](src/fv/settings.py), que ya tiene indirección por `FV_ROOT`. Separar de
   verdad la lectura es añadir ahí una raíz de datos, **en un solo sitio**.
3. ⚠ **9 scripts cablean las rutas a mano** (16 ocurrencias de `ROOT / "runs"` y similares) sin
   pasar por `settings.py`: `estudio_informe.py`, `estudio_flota.py`, `estudio_cierre.py`,
   `estudio_comparar.py`, `estudio_prioridades.py`, `estudio_progreso.py`, `comparar_repro.py`,
   `knobs_f.py` y `vigilante_avance.py`. Cada uno hay que tocarlo, o romperá.
4. ⚠⚠ **`vigilante_avance.py` PUEDE ALQUILAR MÁQUINAS** y decide qué relanzar mirando `runs/`. Si
   se separa la data sin ajustarlo, verá los recorridos vacíos y **relanzará flota para puntos que
   ya están medidos** — cuesta dinero. Es lo primero que hay que arreglar en ese segundo paso.

**Mientras tanto, para todo dato nuevo**: cuando un estudio produzca resultados, su sitio final es
`foveal-vision-data`, en el mes en que se generó, y **se commitea y se empuja allí** — un resultado
que sólo existe en una máquina desaparece con ella (la regla de «servidores efímeros» de
Convenciones aplica igual, y con más motivo).

---

## Regla permanente: la organización por dominios manda

**[docs/organizacion.md](docs/organizacion.md) es la fuente de verdad sobre cómo se organiza
este sistema. Léelo antes de cualquier cambio y respeta sus fronteras.** Aplica a todo cambio,
por pequeño que parezca — un campo nuevo en un config es exactamente donde las fronteras se
rompen.

Los demás documentos, en orden de lectura:

| | |
|---|---|
| [instructionsNewNN.md](instructionsNewNN.md) | **La red.** Geometría foveada, parámetros, rangos calculados, código de referencia |
| [docs/organizacion.md](docs/organizacion.md) | **La raíz.** Dominios (A–I, X, G) y contratos ①–⑫ donde se tocan |
| [docs/herencia.md](docs/herencia.md) | Qué viene de `image-text-finder`, qué se adapta y qué se descarta |
| [docs/protocolo.md](docs/protocolo.md) | Cuándo un resultado es creíble. **Léelo antes de sacar conclusiones de un entrenamiento** |
| [docs/api.md](docs/api.md) · [docs/ui.md](docs/ui.md) | La organización proyectada sobre HTTP y sobre pantallas. **`ui.md` es el índice**: las reglas de UI viven en `docs/ui/`, una por **tipo de especificación** — [1 estructura](docs/ui/1-estructura.md) · [2 vistas](docs/ui/2-vistas.md) · [3 representación](docs/ui/3-representacion.md) · [4 datos](docs/ui/4-datos.md) · [5 invariantes](docs/ui/5-invariantes.md) · [6 números](docs/ui/6-numeros.md) · [7 operación](docs/ui/7-operacion.md) · [8 léxico](docs/ui/8-lexico.md) |
| [docs/plan.md](docs/plan.md) | El plan de ejecución, por fases verticales |
| [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) | **IMPLEMENTADO (2026-07-24).** Barrido OAT (un eje a la vez) con base derivada del problema, defaults estáticos, arrastre del ganador y estudio (dominio I). Ver `fv.models.derive`, `fv.sweeps.generate/winner`, `fv.studies`, CLIs `fv-oat`/`fv-study` |
| [docs/barrido-stride.md](docs/barrido-stride.md) · [docs/plan-stride-2026-08-27.md](docs/plan-stride-2026-08-27.md) | **El stride de EXTRACCION (dominio B), que no es eje de `space` y por que.** Un dataset por valor, rejilla de evaluacion FIJA y presupuesto de pasos igualado: sin esas dos cosas la tabla mide el examen y el cómputo, no la densidad. El mecanismo en el primero, el criterio escrito antes de mirar en el segundo |
| [docs/metrica-de-tarea.md](docs/metrica-de-tarea.md) | **FASES 1, 2 y 3b HECHAS (2026-07-26); la 3 aplazada (F11), la 4 con el código hecho.** La métrica que manda (párrafo por imagen): el proxy de ventana ordena igual en ejes de **D** (+0,956) **y de C** (+1,000, §2 bis) → `OBJECTIVES` no cambia. `task_score` cableada (`fv.task`, contrato ⑬) con registro de holdout. 8 de las 10 pruebas de §9, medidas. **Léelo antes de tocar métricas de ranking** |
| [docs/formatos.md](docs/formatos.md) · [docs/tests.md](docs/tests.md) | Los artefactos en disco; qué se testea |
| [docs/decisiones.md](docs/decisiones.md) | Lo que sigue sin decidir, y qué bloquea. **No tomes tú una decisión que esté ahí: pregunta** |
| [docs/glosario.md](docs/glosario.md) | Las palabras que significan dos cosas |

Reglas que estos documentos fijan y que se citan aquí porque se incumplen solas:

- **Ausente ≠ cero** (formatos.md §2): un lector que necesita un campo ausente **falla con la
  razón**; nunca lo inventa ni lo rellena.
- **Toda restricción se valida antes, con razón y arreglo** (api.md R4): un `400` al entrar vale
  mil veces más que un stack trace dentro del hilo del job media hora después.
- **Toda puerta que entrene pregunta al mismo validador** antes de reservar el nombre. Dos
  comprobaciones separadas se desincronizan, y la puerta más laxa es por la que entra un
  recorrido automático.
- **Un run no se sobrescribe jamás** (409 con la razón).
- **Un contrato sin test es un comentario** (tests.md): los contratos van a
  `tests/test_contracts.py`, los no implementados en `xfail(strict=True)`.
- **Un resultado sin N semillas es una anécdota** (protocolo.md).

### Los dominios (resumen; el detalle está en organizacion.md)

| | Dominio | Es | Vive en |
|---|---|---|---|
| **A** | Fuente | Imágenes + geometría de párrafos (proyecto externo, solo-lectura) | `src/fv/datasets/` |
| **B** | Dataset de ventanas | Lo que se etiqueta: imágenes completas + etiquetas por ventana. **La vista foveada NO se hornea aquí: se construye en el dataloader** | `src/fv/windows/`, `data/window-datasets/` |
| **C** | Red foveada | Fóvea y borde en px reales, kernels/strides por rama, fusión. Config puro, cero datos | `src/fv/models/`, `configs/networks/` |
| **D** | Receta | Hiperparámetros de entrenamiento que definen el resultado | `src/fv/training/`, `configs/recipes/` |
| **E** | Run | Modelo entrenado: pesos + métricas + procedencia | `runs/<name>/` |
| **H** | Recorrido | Un espacio sobre **C y/o D** con B fijo → muchos E, sin intervención humana | `src/fv/sweeps/`, `sweeps/` |
| **I** | Estudio (OAT) | Un plan ordenado de ejes sobre **H** con B fijo → muchos recorridos; guía, **no ejecuta** | `src/fv/studies/`, `studies/` |
| **F** | Inferencia | Aplicar un E a una imagen completa (ventana foveada deslizante) | `src/fv/inference/` |
| **G** | Geometría foveada | `normalize_geometry`, `derive_dims`, `build_foveated_input`, `build_masks`, rangos calculados. **Un solo módulo, todos lo importan** | `src/fv/fovea/` |
| **X** | Ejecución | `device`, `num_workers`, concurrencia. **Cuesta tiempo, no cambia el resultado** | `src/fv/api/jobs.py` |

### Antes de tocar nada, pregúntate a qué dominio pertenece

El criterio, en orden:

1. ¿Cambia **la forma del modelo o de su entrada**? → **C** (`fovea_px`, `border_px`,
   `border_reduce`, `overlap_fovea_px`, `overlap_border_px`, kernels, strides, `merge`,
   `pool_mode`, `dropout` son C — *incluida la geometría del muestreo foveado*, aunque suene a
   datos: es la red quien define qué vista consume). `N` **no es un parámetro: se deriva**.
2. ¿Cambia **los pesos resultantes** sin cambiar la forma? → **D** (`lr`, `batch_size`, pesos de
   la pérdida).
3. ¿Solo cambia **cuánto tarda**? → **X**. Nunca dentro de la identidad de D. **`batch_size` es
   D, no X** — subirlo al pasar a GPU invalida la comparación con lo entrenado en CPU (contrato ⑩).
4. ¿Se ajusta **sin reentrenar**, sobre un modelo ya hecho? → **F** (`threshold`, stride de
   inferencia, NMS). Barrer esto no cuesta horas; no lo metas en D.

Si un cambio necesita tocar dos dominios, eso es un **contrato**: está numerado en
organizacion.md §2. Respétalo explícitamente o actualiza el doc.

---

## Dónde caen los datos de un estudio: **en `foveal-vision-data`**

**Aplicado el 2026-08-27.** Los artefactos de estudio —runs, recorridos, estudios— se escriben en
el repo hermano. Este repo es **el código que mide**; aquel es **lo medido**.

```bash
.venv/bin/python scripts/prueba_destino_datos.py     # 0 = van al repo de datos · 1 = siguen aquí
```

Corre un recorrido real de 1 punto y 2 épocas con el **mismo `run_sweep` que usa la flota** (no un
mock: lo que se comprueba es la ruta que elige el código de verdad) y dice dónde aterrizó cada
fichero. Es local, no alquila nada, y se limpia con `--limpiar`.

### La indirección está en UN sitio

[`src/fv/settings.py`](src/fv/settings.py) → **`data_root()`**, y `runs_root()`, `sweeps_root()` y
`studies_root()` cuelgan de ella. Orden: `FV_DATA_ROOT` > el hermano `foveal-vision-data` si está
clonado > **este repo**.

⚠ **Ese último caso es deliberado y no es un fallo**: sin el repo de datos clonado todo sigue
funcionando como antes. Una separación que rompe al que no ha clonado nada es una separación que
nadie adopta.

⚠⚠ **Pero «como antes» ya no incluye «y se commitea».** Desde que se vació el legado, `runs/`,
`sweeps/` y `studies/` están en el `.gitignore` de **este** repo. Así que el fallback escribe en
un sitio que git ignora: los datos existen en disco y desaparecen con el servidor, sin un solo
error por el camino. Por eso `estudio_flota.py --git` **aborta antes de alquilar nada** si
`data_root()` resuelve a este repo, y dice el `git clone` que lo arregla. Medido el 2026-08-27:
la máquina apareció recién rehecha, sin el repo de datos clonado, y la separación llevaba así
desde el commit que la aplicó.

### El **dataset de ventanas** también va allí — y su `windows.npz` **se commitea**

**Aplicado el 2026-08-27.** `data/window-datasets/` salió de este repo: los 16 datasets
(manifest, `split.json` **y el `windows.npz`**) viven en `foveal-vision-data/window-datasets/`.
La indirección es **`settings.window_datasets_root()`**, que cuelga de `data_root()`.

**Por qué se guarda la carga, si la regla de siempre es que la carga no entra en git.** Porque la
regla tenía una premisa —«son artefactos regenerables»— y **está medido que es falsa**. `repro-chk`,
el 2026-08-26: mismo punto, misma semilla, misma familia de CPU (donde el entrenamiento sale
idéntico bit a bit), y las curvas salieron **distintas**. Veredicto escrito antes de mirar: *es otro
dataset*. Por eso el de hoy se llama `r20260826` y no `r20260824`.

Lo que costó tratarlo como regenerable: al rehacer la máquina, el `r20260824` **desapareció** —no
estaba en ningún git— y con él la comparabilidad de **20 runs ya pagados**, que hubo que volver a
medir enteros (#14). Un dato que no se puede re-derivar y no se guarda, se pierde; y se descubre
cuando ya no hay remedio.

Cuesta poco: **~3-6 MB por dataset**, y sólo se añaden — dato nuevo = nombre nuevo, nunca se
reescribe uno.

#### Un B puede salir de otro B: los datasets de **fallos** (2026-09-02)

`scripts/dataset_fallidos.py` pasa una red por un dataset de ventanas, puntúa cada imagen a nivel
de **párrafo**, y escribe un B nuevo con las peores — entrenable con `fv-train` sin tocar nada.
Hay tres, con `--verdad ventanas`: `optimo-fallidos` (427 img), `edge-fallidos` (481) y
`mask-fallidos` (346). Desde Telegram, `/use fallidos`.

⚠ **Dos cosas que hay que saber antes de usarlos**, y las dos están medidas el 2026-09-02:

1. **La mayor parte de lo que parece «la red falla» es el EMPAREJADO, no la red** — y **ya está
   arreglado**, ver el apartado de abajo. Cada imagen lleva su `solo_emparejado` para no confundir
   las dos averías, que piden arreglos opuestos. Los tres datasets se construyeron con la
   reconstrucción heredada, que sigue siendo el defecto.
2. **Su verdad está RECOMPUESTA desde las etiquetas de ventana**, no leída de la fuente — la de
   `dirty-1000-80px` se perdió con la máquina anterior. Es la excepción declarada al contrato ⑬ y
   pierde los párrafos cortados por el borde (13 de 1000). Queda escrito en cada `manifest.json`.

El detalle, el criterio escrito antes de mirar y las tolerancias medidas, en
[docs/dataset-fallidos.md](docs/dataset-fallidos.md).

#### ⚠ La reconstrucción de párrafos estaba rota, y la métrica de tarea con ella (2026-09-02)

**La red predice CUATRO tipos de esquina y `_reconstruct` usaba DOS**: `TR` y `BL` se calculan,
pasan el NMS y se tiran, así que la única prueba de que un TL y un BR eran del mismo párrafo era
la confianza. Resultado, visto a ojo: cajas que unen el TL de un párrafo con el BR de otro.

**Lo primero, porque es lo que asusta: `val_f1` NO depende de esto.** El f1 que monitoriza el
entrenamiento, elige `best.pt` y rankea recorridos sale de `detection_counts` por ventana
(`training/loop.py:68`) y no llama a `predict_image` ni a la reconstrucción. `OBJECTIVES` sólo
tiene métricas de ventana. **Ninguna red ha sido mal calificada en su validación ni en ningún
barrido.**

**Lo que sí estaba contaminado es la métrica de TAREA**, que es la que el proyecto llama «la que
importa» — y llegaba a **reordenar** redes. Usar las cuatro esquinas (`reconstruct="quad"`) sobre
las 987 imágenes con verdad completa de `dirty1000-80px-16px-r20260827`:

| red | `tlbr` | `quad` | Δ |
|---|---|---|---|
| `demo-fov16-optimo` | 0,7560 ± 0,0108 | 0,9385 ± 0,0044 | **+0,1826** (17 SEM) |
| `fov16-edge-p20` | 0,6823 ± 0,0120 | 0,9577 ± 0,0038 | **+0,2754** (23 SEM) |
| `fov16-mask-p20` | 0,7666 ± 0,0111 | 0,9773 ± 0,0030 | **+0,2106** (19 SEM) |

⚠ **El defecto SIGUE siendo `tlbr`, y cambiarlo es decisión del dueño**: movería todos los números
de métrica de tarea publicados. Lo que ya está hecho para que ese cambio sea seguro es que los dos
knobs nuevos **entran en la clave de caché** de `fv.task` — si no, cambiar el defecto habría
servido números viejos bajo el mismo nombre y en silencio.

El porqué de cada decisión, lo que se probó y perdió (el residuo), la meseta de la tolerancia y lo
que sigue sin arreglar, en [docs/reconstruccion-parrafos.md](docs/reconstruccion-parrafos.md).

⚠ **El fallback aquí no es cosmético: es el contrato con la máquina alquilada.** Sin repo de datos,
`window_datasets_root()` cae a `<código>/data/window-datasets`, que es **exactamente** donde
`construir_payload()` mete los datasets en el tar y donde `bench_fleet.py` los copia por `scp`. Las
máquinas de Vast y los droplets de medición **no tienen ni deben tener** el repo de datos: reciben
el dato hecho, no lo buscan. Origen y destino del tar son distintos **a propósito** — igualarlos
(que es la simplificación que parece obvia) rompe uno de los dos lados, y se descubre con la flota
alquilada y facturando. Tiene test: `test_sin_repo_de_datos_el_dataset_cae_donde_lo_deja_el_payload`.

⚠ **Estar en disco no es estar guardado.** `estudio_flota.py` pregunta antes de alquilar si cada
dataset está **commiteado** (`datasets_sin_guardar()`), avisa si no, y **con `--git` aborta**: pedir
`--git` es decir «esto tiene que sobrevivir». Es la misma regla que el libro de a bordo, aplicada al
dato en vez de a lo medido. Sin repo de datos, **ninguno** cuenta como guardado — ante la duda, el
fallo ruidoso.

Y por lo mismo `bench_dataset.py` (build/publish) resuelve por la indirección: si escribiera en
`ROOT/data` mientras el resto lee del repo de datos, saldrían las dos mitades que divergen. `install`
**sí** sigue escribiendo en `ROOT/data`, y es correcto: corre en el droplet de medición, que es
justo el caso del fallback.

**Los tres caminos por los que una máquina creada desde aquí recibe el dato**, y los tres salen de
git:

| Máquina | Cómo lo recibe |
|---|---|
| **Vast** (`estudio_flota.py`) | dentro del **payload tar**, leído de `DATASETS` |
| **Droplet de medición** (`bench_fleet.py`) | `preparar_dataset()` **publica desde git** a una etapa temporal y copia de ahí |
| **Droplet nuevo** (`lanzar launch dev`) | `types/dev.json` **clona `foveal-vision-data`** |

⚠ En `bench_fleet.py` el orden es **git primero, volumen de respaldo**, y no al revés: el volumen es
una copia que alguien publicó alguna vez, git es la fuente. Si divergen —y no hay nada que lo
impida— vale la commiteada, que es la única que puede reproducir un tercero. `--reap`, que es el
freno de gasto, no pasa por ahí: sigue funcionando sin dataset.

### Leer y escribir NO son lo mismo, y por eso hay dos métodos

Lo ya medido está repartido en **tres formas** que no coinciden, así que
[`src/fv/artefactos.py`](src/fv/artefactos.py) resuelve en cascada:

| # | forma | qué es |
|---|---|---|
| 1 | `<data>/runs/<run>/` | la forma **plana**. ⚠ Ya **no** se escribe aquí (desde 2026-08-28); se sigue **leyendo** mientras quede algo sin recoger |
| 2 | `<data>/<año>/<mes>/sweeps/<recorrido>/runs/<run>/` | el **archivo fechado**: lo que dejó la migración **y lo que se escribe de ahora en adelante**. `index.json` es el mapa de lo migrado |
| 3 | `<foveal-vision>/runs/<run>/` | el **legado** — ⚠ **ya vaciado** (2026-08-27), pero la cascada lo sigue mirando por si un proceso lo recrea |

- **`path(nombre)`** busca en ese orden — 1 → 2 → **lo agrupado hoy** → 3. Si (3) se mirara antes,
  un run migrado se leería de la copia vieja y no de la buena.
- **`destino(...)`** devuelve dónde se CREA, y **siempre fecha** (ver abajo). `create()` usa
  `destino()`, **nunca `path()`**: si un run ya estuviera archivado, `path()` devolvería el archivo
  y se escribiría dentro de él.
- ⚠ **Y `path()` parte de la forma PLANA, no de `destino()`.** Desde que `destino()` siempre fecha,
  pasárselo a `path()` haría invisible de golpe todo lo que ya está escrito en la raíz plana. Fijado
  por `test_what_is_already_flat_stays_visible`, **probado rompiéndolo**.

### El mes lo elige EL ESTUDIO, y lo hereda todo lo suyo

Decisión del usuario, y es el motivo de que exista el agrupamiento: **no ver un mismo estudio
disperso en varias carpetas de mes** sólo porque unos recorridos corrieron al día siguiente.

- Un **estudio** estrena su carpeta de mes: es quien la elige.
- Un **recorrido** hereda el mes de **su estudio** (`spec.study`), no el de hoy. Uno lanzado el día
  1 del mes siguiente **se queda con su estudio**. Si su estudio no tiene mes todavía, este
  recorrido lo **estrena**, y los siguientes del mismo estudio lo heredarán de él.
- Un **run** vive **dentro** de su recorrido (`<mes>/sweeps/<rec>/runs/<run>`), vía
  `provenance.sweep` — así la relación es estructura de directorios y no un prefijo en el nombre.
- Un **run suelto** (un benchmark, sin recorrido) no se inventa un padre, pero sí va a
  `<mes>/runs/`: *«un huérfano no se inventa un padre»* es sobre el **recorrido**, no sobre la fecha.

⚠ **El mes AGRUPA para poder leer el directorio; no fecha cada run.** Fijado por
`test_a_study_keeps_its_sweeps_and_runs_in_one_month`, **probado rompiéndolo**.

#### ⚠ El agujero que dejó todo un estudio en la raíz plana (medido 2026-08-28)

**Un estudio de este proyecto casi nunca es un directorio `studies/<nombre>/`.** Sólo el motor OAT
del API lo crea; los `scripts/estudio_*.py` —que es cómo se ha lanzado **todo** lo que se ha medido
aquí— nombran su estudio en el `spec.json` de cada recorrido y **no crean el artefacto nunca**.

Como el mes se buscaba **únicamente** por ese directorio, `mes_del_estudio()` devolvía `None` para
todos ellos, `destino_agrupado()` devolvía `None` detrás, y el recorrido y sus runs caían en
`<data>/sweeps/` y `<data>/runs/`. **Sin fecha y sin un solo aviso.** Le pasó al tanteo `do-t` de
`dropout-2026-08-28` mientras corría.

El razonamiento que lo permitía estaba escrito y era *«devolver `None` es la respuesta honesta, en
vez de inventar una carpeta de mes que separaría lo que debería ir junto»*. **Tenía un agujero: la
alternativa a la carpeta de mes no era «no separar», era la RAÍZ PLANA** — un tercer sitio, sin
fecha, donde el estudio queda igual de separado de los demás y además sin decirlo. Lo que hay que
conservar es *un estudio en UN mes*, y eso se conserva **heredando** el mes, no renunciando a tenerlo.

Así que ahora:

- **`mes_del_estudio()` también mira los recorridos**: si algún `spec.json` ya archivado nombra ese
  estudio, su mes es el del más antiguo (criterio 3 del README del repo de datos).
- **`destino_agrupado()` sólo devuelve `None` en el único caso en que el mes de verdad separaría**:
  un run cuyo recorrido está plano. Ese run se queda con su recorrido — peor sitio, pero no *otro*.

Fijado por `test_nothing_new_is_ever_written_to_the_flat_root`,
`test_a_study_without_an_artifact_still_keeps_one_month` y
`test_a_loose_run_gets_a_month_but_never_an_invented_sweep`, **los tres probados rompiéndolos**.

#### Y lo que ya quedó plano: `scripts/recoger_planos.py`

Dejar de escribir mal no mueve lo ya escrito. Eso lo hace `recoger_planos.py`, que **simula por
defecto** y sólo mueve con `--aplicar` (`/use recoger-planos` desde Telegram):

```bash
.venv/bin/python scripts/recoger_planos.py             # dice qué haría
.venv/bin/python scripts/recoger_planos.py --aplicar   # lo hace
```

- **Fecha por el JSON del propio artefacto**, nunca por el mtime — en un clon limpio el mtime es la
  fecha del checkout y movería el archivo entero al mes en que alguien clonó.
- **No toca `index.json`**: es el mapa de la migración de agosto, y lo movido se encuentra
  recorriendo las carpetas de mes. (`migrar_data.py` del repo de datos **no sirve** para esto: aquel
  *copia* —lo plano seguiría estando— y reescribe `index.json` entero, cargándose el mapa de los 851
  artefactos ya migrados.)
- ⚠ **Se niega si hay algo vivo**, por dos vías porque ninguna ve lo de la otra: un `estudio_flota.py`
  cuyo `/proc/<pid>/cwd` esté en este workspace, y cualquier run plano en estado no terminal (que es
  lo que se ve cuando quien entrena es una máquina alquilada). Mover un directorio bajo los pies de
  quien escribe deja los runs a medias en el sitio viejo y al escritor apuntando adonde ya no lee
  nadie: datos ya pagados, perdidos sin un solo error.
- ⚠ Y **excluye su propio pid y los de sus padres** del `pgrep`: `pgrep -f` casa con la línea de
  comando entera, así que un shell que mencione `estudio_flota.py` se contaba como flota viva y
  bloqueaba la recogida para siempre. Es la otra cara de la trampa del `pkill -f` del coordinador.

⚠ La cascada es **una escalera para migrar sin parar el mundo, no un diseño permanente**. El legado
ya está vacío; cuando se confirme que nada lo recrea, `legado()` se borra y quedan dos escalones.

⚠ El archivo fechado **no tiene la forma plana** `runs/<name>/` que usan los almacenes: un run vive
dentro de su recorrido y de su mes. Por eso se lee por `index.json` y no intentando que una sola
raíz sirva para las dos formas.

### Dos trampas que costaron encontrarlas

1. **Los tests aislaban parcheando `ROOT`, y dejó de valer.** `tests/test_stride.py` apuntaba
   `F.ROOT` a un tmpdir para construir el payload; al salir los recorridos de `ROOT`, el test
   empezó a leer del repo de datos real. Por eso `estudio_flota.py` tiene ahora `SWEEPS` y `RUNS`
   **a nivel de módulo**: es lo que un test puede parchear. El dataset (`data/window-datasets/`)
   **sigue bajo `ROOT`** y así debe seguir — no es un artefacto de estudio.
2. **Un `rglob("*<nombre>*")` filtrado por `is_file()` da CERO y se lee como «no hay nada».** Los
   artefactos se llaman `config.json`, `metrics.jsonl`…; el nombre del estudio está en el
   **directorio**. La primera versión de la comprobación decía «los datos siguen aquí» cuando ya
   estaban allá — la conclusión contraria a la verdadera.

### Tres trampas más, encontradas al integrar (2026-08-27)

3. ⚠ **`lru_cache` sin argumentos sobre `index.json`**: el primer repo mirado se quedaba pegado, así
   que un test apuntando `FV_DATA_ROOT` a un temporal **seguía resolviendo contra el repo REAL**.
   La caché va **por raíz**. Lo fija `test_the_archive_index_is_cached_per_root`.
4. ⚠ **`used_by_study` miraba sólo la raíz plana**: con el recorrido agrupado bajo el mes de su
   estudio, borrar el estudio **dejaba huérfanos sus recorridos** — justo el bug que la cascada de
   borrado existe para evitar. Los tres listados (`list`, `used_by_dataset`, `used_by_study`) y
   `RunStore.list` pasan ahora por `artefactos.nombres()`.
5. ⚠ **Los tests podían escribir en el repo de datos real.** Un almacén sin `root=` explícito
   resuelve al hermano de verdad. Hay un fixture **`autouse`** en `tests/conftest.py` que apunta
   `FV_DATA_ROOT` a un temporal en **todos** los tests — global, porque los que no usan `world` son
   justo los que construyen almacenes a pelo.

### El libro de a bordo commitea allá (2026-08-27)

`estudio_flota.py --git` se trae de cada máquina, en cada sonda, los ficheros pequeños de cada
run y los commitea. Ahora contra `foveal-vision-data`, y con tres cosas que hay que respetar:

1. **Se COLOCA, no se extrae.** El tar viene en la forma plana `runs/<run>/`, pero un run vive
   dentro de su recorrido y del mes de su estudio. `_colocar_runs()` lee el `provenance.sweep`
   del `config.json` que viaja en el mismo tar y pregunta a `RunStore.destino()`. Extraer tal
   cual dejaría los ficheros donde `path()` no mira: **medidos, en disco, y contados como
   pendientes** — o sea, máquinas alquiladas otra vez para repetir puntos ya pagados.
2. **El `rc` del `git add` se mira.** Cuando no se miraba, un `add` fallido dejaba el índice
   vacío y el `git diff --cached --quiet` de la línea siguiente lo leía como *«nada que
   commitear, y no es un fallo»*. Es exactamente cómo esto pasó desapercibido: `git add --
   runs sweeps` contra este repo devolvía `fatal: pathspec 'sweeps' did not match any files`,
   y el libro se quedaba mudo. Y por eso se estadea con `-A` y no con una lista de directorios:
   el repo de datos no contiene otra cosa que artefactos, y desde el agrupamiento las rutas son
   `<año>/<mes>/…` — un `-- runs sweeps` cableado es justo la suposición que se rompió.
3. **El freno va antes del acelerador.** `--git` sin un repo de datos donde commitear **aborta
   con código 2 antes de alquilar**, no avisa a mitad. Un libro que no commitea no se nota hasta
   que se rehace la máquina, que es cuando ya no hay remedio.

Los tres, con test en `tests/test_stride.py` (`test_el_libro_deja_cada_run_dentro_de_su_recorrido`,
`test_el_libro_se_niega_si_los_datos_caen_en_el_repo_de_codigo`,
`test_un_git_add_que_falla_no_se_lee_como_nada_que_commitear`), y los dos primeros **probados
rompiéndolos**.

### Lo que queda pendiente

- ✅ ~~Vaciar el legado~~ **hecho (2026-08-27)**: `runs/`, `sweeps/` y `studies/` ya no están en este
  repo, y el `.gitignore` los ignora para que no vuelvan a entrar si un proceso los recrea.
  Comprobado que los 851 runs, 61 recorridos y 8 estudios **siguen resolviendo sin el legado**.
- ✅ ~~Commitear lo nuevo en el repo de datos~~ **hecho (2026-08-27)**: el libro de a bordo de
  `estudio_flota.py` coloca y commitea en `foveal-vision-data`. Ver «El libro de a bordo commitea
  allá» abajo.
- **Los 5 JSON sueltos de `data/`** (`p40-*-task.json`, `proxy-c-d-3b.json`,
  `stride-*-informe.json`): por criterio irían al repo de datos. `data/window-datasets/` **se queda
  aquí**: es dominio B, entrada del experimento y no resultado.

## Varias sesiones a la vez: comprueba en qué copia estás

Desde 2026-08-27 hay **más de una sesión de Claude** trabajando sobre copias distintas de
este repo (`~/ws/<linea>/foveal-vision/`), cada una en su rama. **La regla y la estructura
completas están en el coordinador**, que es quien descubre los repos y por eso es donde se
dispara:
[`telegram-coordinator/CLAUDE.md` § «Varias sesiones a la vez»](https://github.com/stalinbeltran/telegram-coordinator/blob/main/CLAUDE.md#varias-sesiones-a-la-vez-un-workspace-por-línea-de-trabajo).

Aquí queda sólo lo que se dispara **en este repo**, que es lo que cuesta dinero:

1. **`--prefijo <pfx>` en TODA flota, siempre.** Es lo único que separa tus máquinas de las
   de otra sesión en una cuenta que es una sola. `vigilante_avance.py` sólo toca instancias
   cuya etiqueta reconoce y dice *«ajena: no la toco»* del resto — **medido el 2026-08-27**,
   con `st-` y `estudio-` conviviendo sin pisarse. Sin prefijos distintos, un vigilante
   destruye las máquinas de otro creyéndolas huérfanas suyas.

2. **No mates flotas por nombre.** `pkill -f estudio_flota` mata las de **todos** los
   workspaces: casa por cadena de comando, no por ruta. Saca el PID por `/proc/<pid>/cwd` y
   mata **ese PID**. Desde el coordinador, `node scripts/workspace.mjs` te dice cuáles son
   tuyos y cuáles no.

3. **⚠ `flota_viva()` no distingue copias, y eso silencia relanzamientos.**
   `vigilante_avance.py:362` usa `pgrep -f "estudio_flota.py"` sin filtrar por ruta, así que
   el vigilante de un workspace ve la flota de otro y aplica su regla 4 («hay una flota viva:
   no se relanza»). Consecuencia: **los puntos que le faltan no se relanzan nunca y nadie
   avisa** — un barrido incompleto que parece terminado, que es justo lo que el índice de
   reportes existe para evitar. Arreglo: mirar `/proc/<pid>/cwd` de cada PID y quedarse con los que
   cuelguen de `ROOT`. ⚠ **No** filtrar por la línea de `ps`: la flota se lanza con ruta
   relativa (`.venv/bin/python scripts/estudio_flota.py`), así que el workspace no aparece
   en ella y el filtro daría por ajenos **tus propios procesos**.

4. **Los repos hermanos se resuelven por `ROOT.parent`** (`bench_dataset.py:46`,
   `estudio_flota.py:179`, `vigilante_avance.py:106`). Copiar este repo **solo** deja esas
   referencias apuntando a un padre que puede no tener el generador ni el lanzador: no falla
   al empezar, falla a mitad. Se copia el workspace entero o no se copia.

5. **Un recorrido con el mismo nombre en dos copias alquila dos veces.** Los cerrojos de
   `estudio_flota.py` son `threading.Lock`, o sea **dentro del proceso**: dos procesos no se
   ven, y cada copia tiene su propio `runs/`, así que ninguna de las dos sabe que la otra ya
   pagó esos puntos. Antes de lanzar, mira que nadie más esté corriendo ese recorrido.

## Contexto de trabajo

- **Hoy solo CPU (esta máquina). Habrá un server con GPU** para los recorridos largos. Por eso X
  está separado de D **desde el diseño**: si no, lo entrenado en CPU queda incomparable con lo
  de GPU. Y por eso `environment` (python/torch/plataforma/device) va en la procedencia de cada
  run.
- **El flujo objetivo son recorridos secuenciales desatendidos**: una receta de recorrido (H)
  nombra el espacio y el presupuesto, se lanza, y corre puntos de uno en uno guardando runs de
  primera clase. Primero versiones cortas aquí (pocas épocas, dataset pequeño) para validar el
  instrumento; el mismo spec, con más presupuesto, en la GPU.
- En CPU, **el límite de workers concurrentes es 1**: torch ya usa todos los núcleos. En GPU se
  reevalúa (es X: no cambia resultados).
- **El espacio de geometría foveada es pequeño y discreto por construcción** (los rangos los
  calculan las funciones de instructionsNewNN.md §3: con N=20, ~3·2·2·1·varios puntos) →
  **grid exhaustivo**. Optuna se reserva para lo continuo: `lr`, canales, dropout
  (instructionsNewNN.md §9).

## Convenciones

- **Idioma**: el usuario se comunica en español; documentación de alto nivel en español. El
  código (identificadores, docstrings) en inglés.
- **Commits**: cada tarea terminada acaba en un commit descriptivo. Además, **cada cambio
  solicitado por el usuario, una vez completado, se cierra con su propio commit descriptivo.**
- **Rama: TODO va a `main`. NO se usa `dev`.** *(Regla vigente desde 2026-08-26, por decisión
  del usuario; invierte la convención anterior.)* Cada cambio que pida el usuario se commitea
  **en `main`** y se **empuja a `main`** (`git push origin main`) en cuanto queda terminado y
  probado. **No crees ramas `dev` ni empujes a `origin/dev`**, y no propongas un merge: no hay
  paso intermedio que esperar.
  **El porqué**: la máquina se rehace sin aviso y **un clon limpio saca `main`** — lo que se
  quedaba en `dev` era invisible para la máquina siguiente, para la flota y para los ejecutores
  de Telegram (que se descubren por `git pull` a `main`). La rama de desarrollo sólo añadía una
  deuda de merge que ya se cobró: ver el bloque de «Servidores efímeros» de abajo, y el
  desfase de **74 commits** que tenía `origin/dev` el 2026-08-26.
  ⚠ **`origin/dev` sigue existiendo** y quedó al día ese día. Es historia: no se le empuja más.
  Misma regla en el proyecto hermano `image-text-sample-generator` — que **no se actualizó hasta
  el 2026-09-01**, o sea que durante cinco semanas los dos repos se contradijeron por escrito.
  ⚠ **La ÚNICA excepción, y este bloque no la decía**: los **trabajos paralelos** —otras sesiones
  de Claude, con sus propias conversaciones, en **workspaces separados** del mismo dev— usan la
  rama de su workspace, para que dos líneas que tocan los mismos ficheros no se pisen. **Si no
  estás dentro de `~/ws/<algo>`, no es tu caso: vas a `main`.** El mecanismo, en
  [`telegram-coordinator/CLAUDE.md` § «Varias sesiones a la vez»](https://github.com/stalinbeltran/telegram-coordinator/blob/main/CLAUDE.md).
  Y esa rama **tiene que acabar en `main`**: mientras no llegue, su trabajo es invisible para el
  server siguiente — el mismo fallo con otro nombre de rama.
- **Servidores efímeros: lo que no está empujado, no existe.** La máquina se rehace sin
  aviso, así que **todo cambio y toda documentación se empuja en cuanto queda terminado**,
  no al final del encargo. **Un clon limpio saca `main`**, y por eso hoy se trabaja
  directamente ahí (ver la regla de rama de arriba): lo que se quedaba en `dev` era
  invisible para la máquina siguiente. Medido el 2026-08-14: el droplet apareció restaurado y sin datos, y el
  procedimiento para reconstruir la fuente del benchmark estaba empujado **solo a `dev`**
  del generador; se dio por imposible lo que sí estaba escrito y se midió sobre la fuente
  equivocada. Aplica igual a los artefactos que sí se versionan (reportes de
  `benchmarks/`, manifests): commitearlos es lo único que los salva de la reconstrucción.
- **Los comandos de este repo para el bot de Telegram van en `telegram/executors/*.json`.**
  El coordinador los **descubre** ahí con sólo estar el repo clonado (su `data/fuentes.json`
  escanea `~/src/*/telegram`): no hay que copiarlos a ninguna parte ni reiniciar nada, y
  llegan con `git pull` **a `main`** — que es donde se commitea todo (regla de rama). Se
  escriben como si estuvieras dentro de este repo (el cwd ya es su raíz: nada de
  `cd ~/src/foveal-vision &&`), con `descripcion` y `ejemplos` en el mismo fichero, y
  `$COORD_HOME` para llamar a algo del coordinador (`notify.mjs`, `desacoplar.sh`). Son
  varios; `bench` y `fvweb` son los dos que hay que conocer.
  ⚠ **Terminan en `; true`** (o no fallan nunca): el coordinador lee cualquier código ≠ 0
  como «el ejecutor falló» y entonces **no corre los encargados**, o sea que la respuesta no
  llega a Telegram. El veredicto va en el **texto**, que es donde se lee — la misma razón
  por la que `cerrable.mjs` se invoca con `--exit0`. El porqué está en
  [`telegram-coordinator/docs/ejecutores-federados.md`](https://github.com/stalinbeltran/telegram-coordinator/blob/main/docs/ejecutores-federados.md).
- **Stack**: Python 3.12 (PyTorch no tiene wheels para 3.14) + PyTorch + FastAPI + Vite/React.
  En Windows el intérprete será `.\.venv\Scripts\python.exe`. Paquete `fv`, layout `src/`.
- **Tests**: `.\.venv\Scripts\python -m pytest -q` desde la raíz, antes de commitear código.
- **README verificado**: antes de decir que un comando documentado funciona, **ejecútalo** en
  PowerShell tal como está escrito (regla global del usuario). Nunca presentar una instrucción
  no probada como verificada.
- **Probar ejecutando, sí; dejarlo corriendo, no.** Lanzar procesos del proyecto para verificar
  (backend `fv-api`, `npm run dev`, entrenamientos, Playwright) está **siempre permitido** y no
  hace falta pedirlo. Pero **al terminar la tarea se cierran todos** y se comprueba que los
  puertos (`:8010`, `:5173`) quedan libres: el usuario prueba a mano después, y un server viejo
  vivo le ocupa el puerto o le contesta con rutas obsoletas. Matar hijos antes que padres (vite
  antes de `npm run dev`; el `fv-api` que escucha antes de su lanzador) y filtrar por ruta —
  en esta máquina hay pythons ajenos al proyecto.
- **Los reportes van al repo central [`estudios-redes-neuronales`](https://github.com/stalinbeltran/estudios-redes-neuronales)**, no aquí — desde el
  2026-08-29, y sea cual sea el repo que lanzó el trabajo o produjo el dato. Van a
  `reportes/<tipo>/<año>/<mes>/`, p. ej. `reportes/estudios/2026/08-agosto/`; el **tipo** sale de
  dos preguntas mecánicas (*¿se corrió algo?* y *¿el sujeto medido es la red, la máquina, o el
  sistema?*) y está escrito en [su `reportes/README.md`](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/README.md).
  Un *reporte* es todo informe dirigido al usuario que resume, compara
  o prioriza (inventarios de parámetros, análisis de resultados, comparativas, informes de
  estado); **no** lo son los planes de estudio, que **siguen aquí**, en `docs/plan-*.md`, porque se
  escriben **antes** de medir y son criterio, no informe — esa parte no se movió y no debe moverse.
  El mes va **con su número delante** para que
  ordenen solos, y el nombre en español y en minúscula (`01-enero` … `12-diciembre`). Fichero por
  reporte, nombre en kebab-case que diga de qué va, y **se commitean y se empujan**: un reporte
  que sólo existe en esta máquina desaparece con ella.
- `data/`, `runs/` y `sweeps/` son artefactos: **se versiona la descripción (configs, métricas,
  manifests, specs), se ignora la carga (`.npz`, `.pt`, `optuna.db`)** — formatos.md §5.
  ⚠ **Y desde 2026-08-27 la descripción de runs, recorridos y estudios se versiona en el repo
  hermano `foveal-vision-data`, no aquí** — ver la sección «Los datos de los estudios van a
  `foveal-vision-data`». La regla de qué se ignora (la carga) no cambia: sigue fuera de git en
  los dos repos.
- **Enlaces a ficheros en las respuestas**: siempre en formato markdown `[texto](ruta)` con la
  ruta **relativa a la raíz del workspace** (nunca backticks ni ruta pelada), para que sean
  clickeables en la extensión de VSCode. **No envuelvas el enlace entre paréntesis** ni pegues
  puntuación al `)` de cierre: `(... [x](ruta) ...)` rompe la detección del enlace y deja de ser
  clickeable. Déjalo suelto o sepáralo con `—`, dos puntos, o una coma con espacio.
  **Los enlaces solo abren ficheros de texto (código fuente), no imágenes** — verificado
  2026-07-23: un `.png` no abre al clicar aunque esté rastreado por git (no es el git-ignore, es
  el tipo binario). Para una imagen (capturas de `data/ui-shots/`, etc.) NO ofrezcas un link
  markdown que no abre: da la ruta para `Ctrl+P`/Go-to-File, o muéstrala inline con la tool Read.

## Observaciones de esta máquina (medidas en el proyecto hermano — no re-aprenderlas)

- **La máquina HIBERNA en entrenamientos nocturnos largos**: suspende el proceso. Para un
  recorrido desatendido, desactivar la suspensión (`powercfg`) o contar con que se pausa.
- **Throttling térmico**: en carga sostenida los runs se ralentizan ~5×. Los presupuestos de un
  recorrido nocturno deben contarlo.
- **La consola de Windows es cp1252**: los CLIs imprimen ASCII (un `→` en un `--help` revienta
  con `UnicodeEncodeError`).
- **Los JSON de estado se escriben con temporal + `os.replace`, y en Windows con reintento en
  los dos lados** (escritor y lector): Windows no reemplaza un fichero con un handle abierto.
  Detalle en formatos.md §4.2.
- **Hay Playwright y Chromium en esta máquina: la UI SE PUEDE ver.** Los navegadores están en
  `%LOCALAPPDATA%\ms-playwright\`; hace falta `pip install playwright` en el venv del proyecto
  (los navegadores ya están, no hace falta `playwright install`). No entregar UI diciendo «no
  puedo verlo» sin haber mirado.
- **Al verificar UI: reinicia el backend** — un server stale da 404 engañosos sobre rutas nuevas.

## Trampas heredadas: no las reproduzcas

Medidas en `image-text-finder` (lista completa y razonada en
[docs/herencia.md](docs/herencia.md) §4 y organizacion.md §3). **Casi todas eran *defaults***:
nadie las eligió, aparecieron por no elegir. Construir desde cero no protege de ellas — las
invita:

- **SGD sin momentum** si solo pasas `lr` y `weight_decay` → cualquier comparación de
  optimizadores queda sesgada a favor de Adam.
- **Un hilo por job sin límite** → un recorrido de 20 puntos son 20 entrenamientos peleándose
  por los mismos núcleos. En CPU el límite es 1.
- **Sobrescritura silenciosa de runs** (`mkdir(exist_ok=True)` + truncar métricas) — quien la
  pisa es justo un recorrido que autogenera nombres.
- **Un dataset sin val** elige `best.pt` por train loss sin avisar → se niega, no se degrada.
- **Estado de run deducido del disco** → un crash queda «running» para siempre. `status.json`
  explícito.
- **Augmentation con flips/rotaciones sin reetiquetar** → enseña basura en silencio (las
  etiquetas de posición/región son direccionales).
- **Definir un número dos veces** (una métrica calculada en dos sitios) → módulo único
  `fv.metrics`, y un test que afirma la costura, no la función.
- **Lógica de dominio dentro de `app.py`** → si una función no menciona HTTP, no es del API.
- **Medir con un val diminuto**: los ejemplos de una misma imagen están correlacionados; el
  tamaño de muestra efectivo lo dan las **imágenes**, no las ventanas. El dato es sintético:
  generar más es gratis.
- **Optimizar un proxy sin validarlo**: la métrica que manda es la de la tarea real (párrafo
  bien reconocido por imagen), no la de ventana — protocolo.md §2.
