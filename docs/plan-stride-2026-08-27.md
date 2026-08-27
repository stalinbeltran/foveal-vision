# Plan del estudio de `stride` de extracción — 2026-08-27

**Pregunta:** ¿cuánto mejora la calidad de predicción de la red foveada al muestrear las imágenes
de entrenamiento con una rejilla más densa —de 16 px de paso a 1 px— y **dónde deja de mejorar**?

El mecanismo (qué cambia en el código, con qué contratos y qué tests) está en
[`barrido-stride.md`](barrido-stride.md). Aquí va **el criterio, escrito antes de mirar ningún
número**, el coste y cómo se opera.

---

## 0. Las reglas de lectura, que no cambian

1. **Todo número lleva su procedencia**: medido (con fecha y comando) o estimado (marcado).
2. **Un tanteo no declara ganador.** El `p` mínimo alcanzable con n contra n manda: 5v5 → 0,0079;
   2v2 → 0,333.
3. **El criterio se escribe antes de mirar.** Lo de abajo (§3) queda congelado al commitear este
   documento; el informe lo aplica, no lo reinventa.
4. **Un hueco no es un cero.** Si un brazo no termina, se dice qué le falta.

---

## 1. La pregunta, acotada

### 1.1 Qué se mide

El **stride de extracción** (`ExtractConfig.stride`, dominio B): cada cuántos píxeles se corta la
siguiente ventana etiquetada de la imagen fuente. Con ventana de 16 px:

- **stride 1** = rejilla máximamente densa; ventanas contiguas comparten 15/16 de su superficie.
- **stride 16** = teselado exacto, sin solape y sin hueco. Es el **techo natural** del eje: por
  encima quedarían píxeles que no entran en ninguna ventana, y eso ya no es «menos denso» sino
  «incompleto».

### 1.2 Qué NO se mide, y por qué está dicho aquí

- **El uso final de la red.** Es petición explícita del encargo: esto mide **calidad de
  predicción** (f1 por ventana, la métrica proxy), no la métrica de tarea sobre la página
  completa. La razón técnica está en [`barrido-stride.md`](barrido-stride.md) §0.2: la métrica de
  tarea mezcla el stride de **inferencia** con el de extracción y el número dejaría de decir cuál
  de los dos lo movió.
- **El stride de inferencia** (knob de F) ni **`s_center`/`s_periph`** (dominio C). Son los otros
  dos `stride` del glosario, y son otros estudios.

### 1.3 Por qué la respuesta no es obvia

Lo obvio es «más denso, mejor», y por eso **el estudio se diseña para que esa no sea la respuesta
por construcción**. A épocas iguales, stride 1 recibiría **146,2×** más pasos de gradiente que
stride 16 y la tabla mediría el presupuesto, no la densidad. Con el presupuesto igualado
(§2.3) la pregunta se vuelve real y tiene tres respuestas posibles, todas informativas:

- **satura pronto** (stride 8 ≈ stride 1) → se puede extraer 27× menos dato sin perder nada, y eso
  es un ahorro directo en cada estudio futuro;
- **satura tarde** (sólo stride 1 y 2 empatan) → la densidad es un parámetro caro que hay que
  presupuestar;
- **no separa** → el ruido entre semillas se come el efecto, y hay que decirlo así y no como «da
  igual».

---

## 2. El diseño

### 2.1 Los cinco brazos

Espaciado logarítmico, porque el número de ventanas va como ~1/s²: un espaciado lineal gastaría
brazos en la zona barata y dejaría el salto grande sin cubrir.

| brazo (`sweep`) | `window-dataset` | stride | ventanas de train | pool relativo |
|---|---|---|---:|---:|
| `stride-01` | `dirty1000-80px-16px-st01` | 1 | 1.755.000 | 146,2× |
| `stride-02` | `dirty1000-80px-16px-st02` | 2 | 455.400 | 38,0× |
| `stride-04` | `dirty1000-80px-16px-st04` | 4 | 122.400 | 10,2× |
| `stride-08` | `dirty1000-80px-16px-st08` | 8 | 37.800 | 3,2× |
| `stride-16` | `dirty1000-80px-16px-st16` | 16 | 12.000 | 1,0× |

Conteos MEDIDOS el 2026-08-27 con `fv.windows.extract._positions`; la aritmética reproduce el
manifest real de `dirty1000-80px-16px-r20260826` (140.000 / 84.000 en stride 5). Detalle y
comprobación en [`barrido-stride.md`](barrido-stride.md) §3.

### 2.2 Lo que se mantiene idéntico entre brazos

| Qué | Valor | Por qué |
|---|---|---|
| Fuente | `local/dirty-1000-80px` (1000 imágenes 60×80) | es el dato real del proyecto; usar otro rompería la comparabilidad con todo lo medido |
| `window_size` | 16 | contrato ①a: la ventana etiquetada **es** la fóvea |
| `seed` de B | 1 | `_assign_splits` sólo depende de `(n, val_frac, test_frac, seed)` ⇒ **las mismas imágenes** en train/val/test en los cinco brazos. Cambiarlo mediría el ruido del split |
| `val_frac` / `test_frac` | 0,2 / 0,2 | los de producción |
| **`eval_stride`** | **5** | la rejilla de evaluación es **fija**: 28.000 ventanas de val idénticas en los cinco brazos. Sin esto cada brazo se examinaría de otra cosa (§2.1 del mecanismo) |
| Red base | `ws16-p2-d2-L4` (167.852 params) | la vigente, la misma de `ov-r26` |
| Receta base | `plan40` (lr 0,0014, batch 85, `patience` 10, monitor `val_loss`) | la vigente |
| **`windows_per_epoch`** | **84.000** | presupuesto igualado: 988 pasos/época en **todos** los brazos, el mismo que todo estudio anterior |
| Objetivo | `f1` | la métrica proxy |
| CPU | `E5-26` | dentro de esa familia el entrenamiento sale bit a bit idéntico entre máquinas: convierte el ruido de máquina en cero |

### 2.3 Las semillas: 5 por brazo, de entrada

**5 semillas y no un tanteo de 2**, contra la costumbre de este repo, por una razón de coste: el
estudio entero son **25 runs ≈ 1,35 $** (§4). Un tanteo ahorraría ~0,80 $ y costaría otra vuelta
de reloj; el techo del contraste con 5v5 es `p` = 0,0079, suficiente para declarar.

**25 runs** = 5 brazos × 5 semillas (`seed` ∈ {1..5}, el eje réplica).

---

## 3. Criterio, escrito antes de mirar

`δ` = la banda de ruido que **este** estudio mida, por la regla de 1 SE que el proyecto ya usa
(`tie_delta` sobre las 5 semillas de cada brazo). No se fija a mano aquí para no elegir el umbral
después de ver la tabla.

**R1 — Saturación (la pregunta principal).** El **punto de saturación** es el stride **más grande**
cuya media de f1 quede **dentro de δ** del mejor brazo. Ése, y no el mejor, es la recomendación
práctica: es el dato más barato que no pierde calidad.
*Si el punto de saturación es 16* → la densidad no compra nada en este rango y hay que decirlo así.
*Si es 1* → el eje **no queda cerrado por arriba**: satura fuera de lo medible con ventana 16, y la
frase correcta es «gana el extremo, no sabemos dónde satura».

**R2 — Significación.** Se contrasta el **mejor brazo contra `stride-16`** con `permutation_test`
(exacto con 5v5).
- `p` < 0,05 **y** diferencia > δ → **la densidad de la rejilla mueve la calidad de predicción**,
  con su tamaño de efecto escrito.
- `p` ≥ 0,05 → se declara **«con 5 semillas la densidad no separa»**. **No** es «da igual»: es que
  el efecto, si lo hay, cabe dentro del ruido de reinicialización de este dataset.

**R3 — Monotonía.** Se comprueba si la media es no-decreciente al bajar el stride. Una
**no-monotonía mayor que δ** no se suaviza ni se explica a posteriori: se reporta como tal y
**abre sospecha sobre el control R4**, porque a igual cómputo no hay mecanismo obvio que haga que
más posiciones distintas empeoren.

**R4 — El control de coste (esto es un control, no un resultado).** Con `windows_per_epoch` igual,
los cinco brazos hacen los mismos pasos por época, así que **`seconds_per_epoch` tiene que salir
igual entre brazos** dentro del ruido de máquina. Si un brazo se desvía más del 15 % de la mediana
de los cinco, **el igualado de presupuesto ha fallado** y la tabla está midiendo cómputo, no
densidad: el estudio **no declara nada** hasta explicarlo.

El 15 % sale de lo ya medido: con la CPU fijada a `E5-26` la dispersión entre máquinas del mismo
catálogo es la que la criba de velocidad recorta, y el rango observado en corridas anteriores es
36–53 s/época (factor 1,47) **sin** criba. Con criba y misma familia, un 15 % es holgado.

⚠ **Lo que este estudio NO contesta, escrito aquí para que no se lea de más:** si el stride de
extracción ayuda a la **tarea** (párrafos sobre la página). Todo es f1 de ventana.

---

## 4. Coste

| Concepto | Cantidad | Procedencia |
|---|---|---|
| Runs | 25 | 5 brazos × 5 semillas |
| $/run | 0,054 | **MEDIDO** el 2026-08-25 (usado igual en `plan-cierre-2026-08-26.md` §2) |
| **Alquiler estimado** | **≈ 1,35 $** | 25 × 0,054 |
| Máquinas | 25 con `--reparto seed` | una por (brazo × semilla) |
| Reloj estimado | ≈ 45 min | peaje 8,4 min/máquina + ~52 épocas × 40 s/época, en paralelo |

**Estimado, no medido** — se apoya en coeficientes medidos (`estudio_estimar.py` §1-§7) pero la
composición es predicción. Se contrasta contra la factura real en el reporte.

**Contraste con el estimador del proyecto** (`--dry-run` del 2026-08-27, 02:26 UTC, con los cinco
recorridos ya creados): optimista **1,12 $** · central **1,21 $** · pesimista **1,53 $**, reloj
0,6–0,8 h, 25 máquinas. Cae junto a los 1,35 $ de la fila de arriba, que se calcularon por otro
camino ($/run medido). Dos estimaciones independientes que coinciden no la convierten en medida,
pero descartan el error de bulto.

⚠ **El estimador no sabe de tamaños de dataset.** Comprobado el 2026-08-27: `estudio_estimar.py`
no menciona `num_windows` ni el dataset en ninguna línea; su `S_EPOCA_REF` está atado
implícitamente a un train de 84.000 ventanas. Para este estudio eso **es correcto por
construcción** —`windows_per_epoch` = 84.000 en todos los brazos—, pero deja de serlo en cuanto
alguien barra el stride sin igualar el presupuesto. Queda anotado como límite conocido del
estimador, no como que esté bien.

### El coste que no es alquiler: extraer los datasets

Los cinco `windows.npz` se extraen **una vez, en local**, y viajan en el payload.

**MEDIDO el 2026-08-27** en este droplet (2 vCPU, 3,8 GB), extrayendo de grande a pequeño:

| brazo | ventanas | `windows.npz` | tiempo |
|---|---:|---:|---:|
| st16 | 68.000 | 2,2 MB | < 6 s |
| st08 | 93.800 | 2,3 MB | 0,1 min |
| st04 | 178.400 | 2,5 MB | 0,1 min |
| st02 | 511.400 | 3,1 MB | 0,1 min |
| **st01** | **1.811.000** | **6,6 MB** | **0,4 min** |

Los cinco: **28.000 de val y 28.000 de test**, la rejilla fija funcionando. Total en disco 16,6 MB.

Era el riesgo que había que descartar y quedó descartado: `extract_windows` acumula las ventanas
en listas de Python antes de `np.stack`, así que el brazo de stride 1 era el candidato a quedarse
sin memoria — el proceso llegó a ~460 MB de RSS (observado con `ps`, no instrumentado) sobre 3,8
GB y terminó en 24 s. El `npz` comprime muy bien porque `y` es casi todo ceros, así que el payload
tampoco es un problema: **6,7 MB** el tar con los dos datasets de la validación.

---

## 5. Operación

### 5.0 Lo que tiene que estar antes (y no está en git)

**La fuente `local/dirty-1000-80px` vive en `data/sources/`, que está en `.gitignore`.** Un clon
limpio **no la trae**, y sin ella no hay datasets que extraer. Se reconstruye —es reproducible,
los specs del generador están congelados con semilla 1— o se copia de una máquina que ya la tenga:

```bash
.venv/bin/python scripts/bench_dataset.py build      # ~15-20 min de renders
```

`estudio_stride.py` lo comprueba **antes de nada** y falla con ese comando al lado. Es la regla de
preflight de este proyecto: lo que se descubre a mitad se resuelve improvisando, y así es como se
da por imposible algo que sí se puede hacer.

Los cinco `windows.npz` tampoco están en git (`/data/window-datasets/*/windows.npz`); sí lo están
sus `manifest.json` y `split.json`, así que se puede comprobar que lo reconstruido es lo mismo.

### 5.1 Orden de los pasos

```bash
# 1. Datasets + recorridos (no alquila nada, no gasta)
.venv/bin/python scripts/estudio_stride.py --fuente local/dirty-1000-80px

# 2. Qué va a costar (no toca Vast)
.venv/bin/python scripts/estudio_estimar.py --sweep stride-01 --sweep stride-02 \
    --sweep stride-04 --sweep stride-08 --sweep stride-16

# 3. La flota (ALQUILA: esto es lo que factura)
scripts/desacoplar.sh .venv/bin/python scripts/estudio_flota.py \
    --sweep stride-01 --sweep stride-02 --sweep stride-04 --sweep stride-08 \
    --sweep stride-16 --cpu E5-26 --criba 2 --git --horas-max 6 --yes

# 4. El veredicto (cuando termine)
.venv/bin/python scripts/estudio_stride_informe.py --estudio stride-2026-08-27
```

**Una sola flota para los cinco brazos**, que es lo que obliga a que el payload lleve varios
datasets ([`barrido-stride.md`](barrido-stride.md) §4.3). No es comodidad: con cinco flotas los
monitores no funcionan (§5.3).

### 5.2 Los monitores

| Qué | Comando | Cuesta |
|---|---|---|
| Cómo va | `/use estudio-progreso` → `--sweep stride-01 … --tabla` | nada |
| Vigilancia máquina a máquina | `scripts/desacoplar.sh .venv/bin/python scripts/vigilante_avance.py --sweep stride-01 … --cada 600` | nada |
| **Freno de emergencia** | `/use apagar-vast` | — |

El vigilante destruye la máquina que no avanza: alquilada y muda (>25 min), congelada (>20 min sin
latido) o huérfana (su lote `done` y sigue facturando). Umbrales medidos el 2026-08-26; son de
tiempo, no de dataset, así que valen aquí sin tocar.

### 5.3 ⚠ Condición de arranque: este estudio NO es el único de la cuenta

Comprobado el 2026-08-27 a las 01:5x UTC: había **8 máquinas vivas** (`estudio-c3` … `estudio-c19`,
0,5159 $/h) de otro estudio, con **su `vigilante_avance.py` corriendo** sobre 14 recorridos. Eso
destapa dos fallos que hay que arreglar **antes** de lanzar nada, y están en el mismo commit que
este plan:

1. **La etiqueta `estudio-` está cableada** (`V.alquilar(oferta, f"estudio-{etiqueta}", …)`), así
   que dos estudios a la vez comparten espacio de nombres en la cuenta.
2. **La rama de «sobrantes» del vigilante no respeta su propio criterio de propiedad.** `juzgar()`
   devuelve `ajena` («no sé de qué recorrido es; no la toco») para lo que no reconoce — pero
   cuando sus recorridos terminan, `una_vuelta` destruye
   `sobrantes = [i for i in mias if i not in danadas]`, y `mias` es **toda** instancia
   `estudio-*`. O sea que al acabar un estudio, su vigilante **mata las máquinas del otro**, que
   momentos antes había declarado ajenas.

   El síntoma sería de los malos: runs cortados a media época, sin error propio, indistinguibles
   de una máquina que se murió sola.

3. **`flota_viva()` preguntaba por CUALQUIER `estudio_flota.py`.** La regla 4 del vigilante —no
   relanzar si ya hay flota— es correcta, pero le faltaba «*sobre mis puntos*». Con la flota del
   otro estudio viva en esta misma máquina, el vigilante de éste habría visto «hay una flota viva»
   en cada vuelta y **no habría relanzado nunca**.

**Los tres arreglos**, en el mismo commit que el código: un **prefijo de etiqueta por estudio**
usado por los dos lados (`estudio_flota.py` lo pone, `vigilante_avance.py` lo filtra), que
`sobrantes` **respete el veredicto de `juzgar`**, y que `flota_viva(sweeps)` sólo cuente la flota
que menciona alguno de sus recorridos. Son la misma pregunta en tres capas —de quién es la
máquina, de quién es la flota— y hacen falta las tres.

**Comprobado en la validación del 2026-08-27**: las dos máquinas nacieron como
`st-stride-h01-s1` y `st-stride-h16-s1`, fuera del espacio de nombres `estudio-*` del otro
estudio, que en ese momento tenía 5-8 máquinas vivas.

---

## 6. Lo que este plan deja explícitamente abierto

- **El efecto sobre la métrica de tarea.** Otro estudio; necesita fijar antes el stride de
  inferencia.
- **La interacción stride × solape (`overlap_fovea_px`).** Podría ser que con más contexto el
  muestreo denso rinda menos: es un factorial 5×7 que no cabe en este presupuesto.
- **Strides no potencia de 2** (3, 5, 6…). Si R1 sitúa la saturación entre dos brazos, el
  refinamiento es un segundo barrido corto en ese hueco.
- **Otras fuentes.** Todo esto es `dirty-1000-80px`, 1000 imágenes de 60×80. La densidad óptima
  puede depender del tamaño de imagen y de la densidad de párrafos.
- **`windows_per_epoch` como eje propio.** Este estudio lo fija a 84.000 para igualar; cuánto
  cambia el resultado subirlo o bajarlo es una pregunta distinta y legítima.
