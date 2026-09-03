# Plan — sonda L1: ¿pueden los kernels de la primera capa aprender filtros genéricos? (2026-09-02)

> **Estado: criterio CONGELADO el 2026-09-02, antes de mirar ningún resultado.** El dueño
> contestó las cuatro preguntas del §6 en [`instruccioneslargas.md`](../instruccioneslargas.md)
> (commit `7959c558d`), y sus respuestas cambiaron el diseño lo bastante como para que el §4
> de aquí se reescribiera **entero** — con la rejilla todavía sin correr, que es lo que hace
> que siga siendo un criterio y no una explicación de lo que salió.
>
> **Lo que corre ahora: el tanteo del eje `k`** (8 runs, §4.4). La rejilla grande sigue sin
> lanzarse, y el §4.5 dice qué tiene que pasar para que valga la pena lanzarla.
>
> Los números se pegan abajo cuando lleguen; **este documento no se reescribe para que cuadre**.

## 0. El encargo, en una línea

*Ver si los 16 kernels de L1, que en `fov16-optimo-mask` no aprendieron nada parecido a un
filtro genérico, podrían aprenderlo **cuando sí hay presión sobre ellos**.*

El encargo completo está en [`instruccioneslargas.md`](../instruccioneslargas.md) de este mismo
repo. Este documento es el §7 de aquél: el criterio, más lo que se ha podido medir **sin
gastar** y que cambia tres de sus umbrales.

---

## 1. La premisa, en números

Medido sobre `best.pt` del run `fov16-mask-p20`:

| Qué | Valor | Su nulo | Lectura |
|---|---:|---:|---|
| energía en el subespacio clásico 6D | **0,688** | 6/9 = **0,667** | **1,03×**: indistinguible de un kernel aleatorio |
| dimensión efectiva (PCA) | 5–6 de 9 | — | el espacio 3×3 está casi lleno |
| par duplicado k5/k7 | coseno **+0,96** | — | los dos DC negativo puro |

**La hipótesis: L1 no está bajo presión.** No hay reducción tras ella, y detrás hay una cabeza
de 153.660 parámetros que puede extraer las esquinas de casi cualquier proyección. Además 3×3
tiene 9 dimensiones y el subespacio clásico ya ocupa 6: no hay sitio para que emerja estructura.

**La sonda quita esa red de seguridad**: un autoencoder de una capa por lado, decodificador
lineal sin sesgo, nada en medio. El modelo *son* los kernels.

---

## 2. Qué está implementado, y dónde

Todo el §1–§6 del encargo. El módulo aislado vive en **`src/fv/probe/`** y la CLI en
**`scripts/sonda_l1.py`**.

| § | Qué | Dónde |
|---|---|---|
| 1 | `Conv(1→K)+ReLU → z → ConvTranspose(K→1)` sin sesgo, stride 1, padding `k//2` | `probe/model.py` |
| 2 | normalización de contraste local (σ=2 px, ε **medido** del train) | `probe/data.py` |
| 3 | `mse/var + λ·mean(|z|)`, `var` **fija del train**, renormalización L2=1 tras **cada** paso | `probe/run.py`, `probe/model.py` |
| 4 | rejilla completa `k × K × λ`, `--cronometrar`, `--repetir-mejores` | `scripts/sonda_l1.py` |
| 5 | las **ocho** métricas, cada una con su nulo | `probe/metrics.py`, `probe/gabor.py` |
| 6 | `config.json`, `metrics.jsonl`, `checkpoint.pt`, kernels `.npy`, hojas de contactos, mapas `z`, tabla | `probe/run.py`, `probe/figures.py`, `probe/table.py` |
| — | el lanzamiento desacoplado (unidad de systemd, cgroup propio) | `scripts/sonda_l1_desacoplada.sh` |

**50 tests** en `tests/test_sonda_l1.py`; suite del repo **557 verde** *(medido 2026-09-02)*.

⚠ **El §7 dice «párate antes de escribir el entrenamiento» y esa parada NO se respetó**, porque
el entrenamiento ya estaba escrito cuando el encargo llegó al repo (commit `31ad236b`, 22:07
UTC; `instruccioneslargas.md` entró a las 22:16). La parada se respeta donde queda algo que
parar: **la rejilla no se lanza** hasta el §6 de aquí.

### 2.1 Tres decisiones cuya rotura sería silenciosa

1. **La renormalización del decodificador a L2=1 tras cada paso.** Sin ella el modelo escala el
   codificador por 0,01 y el decodificador por 100: misma reconstrucción, penalización cien
   veces menor. Aprendería a hacer `z` **pequeño** en vez de **disperso**, y el barrido en λ no
   mediría nada. El codificador queda libre a propósito: es el que tiene que moverse.
2. **`var(x)` es constante FIJA del train** (0,3066, *medido*), no la del lote. Si cambiara por
   lote, λ significaría algo distinto en cada paso.
3. **Cada métrica con nulo se lee CONTRA su nulo**, nunca en crudo. Vale para las dos que lo
   tienen: el subespacio clásico (§3.2 de abajo) y el Gabor (§3.3).

### 2.2 Dónde caen los artefactos, y qué entra en git

`foveal-vision-data/sondas/l1/<run>/`. El `.gitignore` de aquel repo tira todo `.pt` salvo
`runs/*/best.pt`, así que:

| Artefacto | ¿Entra en git? | Por qué |
|---|---|---|
| `kernels_enc.npy`, `kernels_dec.npy` | **sí** | **son el entregable** (§1: «el modelo *son* los kernels») |
| `config.json`, `metrics.jsonl`, `summary.json`, `tabla.*`, `*.png` | **sí** | descripción |
| `checkpoint.pt` | **no** | el dueño fijó el 2026-08-31 que los pesos de un run no se guardan por defecto. Aquí no hace falta la excepción: **los kernels ya viajan en `.npy`**, así que el experimento sigue siendo reproducible desde git |

---

## 3. Lo que ya está medido SIN gastar, y que cambia el criterio

Las seis cosas de aquí abajo se midieron **antes** de lanzar nada. Tres de ellas afectan a los
umbrales que el §7 del encargo propone, y por eso este documento pide confirmarlos en vez de
darlos por buenos.

### 3.1 El coste: **12,0 h**, y no alquila nada

*Medido el 2026-09-02 en este droplet (2 vCPU), con `--cronometrar`:* la combinación más cara
(k=9, K=32) va a **101,3 s/época** sobre las 84.000 ventanas de train.

| Qué | Reloj | Cómo |
|---|---:|---|
| rejilla de **48** runs × 30 épocas | **~12,0 h** | extrapolado por `K·k²` desde la más cara |
| + las 3 mejores × 3 semillas | ~1,5 h | |
| lo mismo con `--limite 20000` | **~4,0 h** | *medido el 2026-09-02: 34,0 s/época* |

**No alquila nada**: corre en esta máquina y satura sus 2 vCPU. **Lo que se pierde al apagar es
trabajo, no dinero** — y son 12 h de él, así que el freno cuenta igual. Está puesto y
**comprobado en vivo el 2026-09-02**: con la sonda corriendo, `cerrable.mjs` dijo
`🔴 NO CERRAR — 1 trabajo(s) vivo(s): sonda_l1.py`.

**Y el lanzamiento desacoplado también está comprobado en vivo** (2026-09-02, por el ejecutor de
Telegram): la unidad `sonda-l1` quedó `active` con cgroup **`/system.slice/sonda-l1.service`** —
el suyo, **no** el del bot— y el python colgando de ella. O sea que sobrevive a un
`systemctl restart telegram-coordinator`, que es la garantía que el mensaje promete.

⚠ **Y esa misma comprobación encontró un fallo, ya arreglado.** `desacoplar-persistente.sh`
registra la unidad con `Restart=on-failure`, y el aviso iba al final de la tubería: **el código
de salida lo decidía `notify.mjs`**. Medido: la sonda terminó bien, el aviso falló (el arnés no
pasa `BOT_TOKEN`) y la unidad se quedó en `Result=exit-code` **reiniciándose cada 30 s** — o sea
12 h de rejilla relanzadas por un aviso. Y al revés también: un trabajo que reventara habría
salido como `success` y **no** se habría reintentado, que es justo cuando el reintento sirve.
Ahora manda el código del **trabajo** y el aviso no puede cambiarlo; vive en
`scripts/sonda_l1_desacoplada.sh` —un fichero, no una línea escapada dentro de un JSON— y tiene
**dos tests**, uno por cada dirección del fallo.

### 3.2 El nulo del subespacio clásico cambia con `k`, y por eso no se compara en crudo

`6/k²` vale **0,667** en 3×3 pero **0,074** en 9×9. Comparar fracciones crudas entre columnas de
`k` distinto es comparar tres escalas. La tabla reporta `energia_6d / (6/k²)`, que vale **1**
cuando el kernel es indistinguible de uno aleatorio — y es lo que convierte el 0,688 de la
premisa en «1,03×».

### 3.3 ⚠ El nulo del **Gabor** es enorme en 3×3, y hace **imposible** el umbral propuesto ahí

La métrica principal ajusta un Gabor 2D a cada kernel. El encargo ya avisa de que *«un Gabor
ajusta ruido mejor de lo que uno espera»*; **cuánto mejor, medido** (mediana del R² sobre 64
kernels **aleatorios** del mismo tamaño, 2026-09-02):

| `k` | nulo (R² de ruido) | techo de `Gabor Δ` = 1 − nulo | ¿alcanza el umbral propuesto de **0,25**? |
|---:|---:|---:|---|
| **3** | **0,879** | **0,121** | ❌ **imposible por construcción** |
| 5 | 0,515 | 0,485 | sí, pero hay que gastar la mitad del margen |
| 7 | 0,337 | 0,663 | sí |
| 9 | 0,228 | 0,772 | sí |

Con 7 parámetros libres sobre 9 muestras, un Gabor ajusta **cualquier** 3×3.

#### Qué NO implica pasar la prueba del Gabor — medido mientras corría el tanteo

⚠ **Esto no cambia el criterio del §4.2, que está congelado.** Es una caracterización del
instrumento, y va aquí porque hace falta para leer la tabla sin sobreinterpretarla.

Al ver que en los dos primeros runs el Gabor superaba el p95 mientras `conc_orient` no, la
sospecha fue que el ajuste se estuviera satisfaciendo de forma **degenerada** — un Gabor con
frecuencia ≈0 es una mancha gaussiana, no una onda orientada. *Medido el 2026-09-02*
extrayendo los parámetros ajustados (ciclos de la sinusoide dentro de la envolvente):

| | R² mediano | ciclos medianos | % con < 0,5 ciclos |
|---|---:|---:|---:|
| `k3-K16-λ0` | 0,958 | 0,71 | **19 %** |
| `k3-K16-λcal` | 0,923 | 0,62 | 38 % |
| **aleatorio k=3** | 0,879 | 0,75 | **38 %** |
| aleatorio k=5 | 0,515 | 0,93 | 31 % |
| aleatorio k=9 | 0,228 | 1,30 | 25 % |

**La sospecha era falsa**: los kernels aprendidos **no** son más degenerados que los aleatorios —
el de λ=0 lo es la mitad. O sea que el ajuste mejora de verdad.

**Pero la lectura correcta es otra, y es la que importa:** a k=3 el mejor Gabor tiene **0,7
ciclos dentro de su envolvente**, o sea que apenas oscila. Una «Gabor» de menos de un ciclo es
una forma suave genérica —un bulto, un borde— cuyo espectro es ancho, no un lóbulo orientado.
Por eso un kernel puede **ajustar bien un Gabor sin tener energía concentrada en una
orientación**: las dos métricas no miden lo mismo, y en 3×3 la familia Gabor es tan flexible que
ajustar bien **no es evidencia de estructura orientada**.

Los ciclos suben con `k` (0,75 → 0,93 → 1,30 en el nulo), así que este solapamiento **se afloja
solo** al subir el eje: es en k=7 y k=9 donde «ajusta un Gabor» empieza a significar «oscila».
**Las columnas informativas del tanteo son ésas**, y el ancla k=3 —donde el plan ya dice que
manda `conc_orient`— es la que menos peso tiene aquí. **Esto no invalida
el ancla k=3** —sigue siendo el único brazo comparable con el 0,688 de la premisa por la métrica
5— pero sí dice que **el ancla no puede juzgarse con la métrica 4**, y que un umbral único para
los cuatro `k` no significa lo mismo en cada uno.

### 3.4 ⚠⚠ La rejilla de λ del encargo **no llega** a la banda de activación que el criterio exige

El §3 del encargo fija el objetivo de activación en **5–15 %** y el §7 lo mete en el criterio de
éxito. *Medido el 2026-09-02* (k=7, K=16, 6 épocas, `--limite 8000`, semilla 1 — un tanteo de
rango, **no** un resultado):

| λ | activa % | kernels muertos | R² rec int | Gabor Δ | enriq |
|---:|---:|---:|---:|---:|---:|
| **0,0** *(control)* | 45,6 | 0 | 0,975 | +0,070 | 0,47 |
| **0,03** | 44,9 | 0 | 0,975 | +0,072 | 0,48 |
| **0,1** | 43,4 | 0 | 0,974 | +0,072 | 0,48 |
| **0,3** *(tope de la rejilla)* | **39,9** | 0 | 0,974 | +0,074 | 0,48 |
| 1,0 | 31,7 | 0 | 0,973 | +0,065 | 0,49 |
| 3,0 | 22,3 | 0 | 0,969 | +0,078 | 0,52 |
| **6,0** | **15,9** | 0 | 0,961 | +0,069 | 0,55 |
| **10,0** | **12,9** | 0 | 0,949 | +0,056 | 0,57 |
| **20,0** | **10,4** | 0 | 0,919 | +0,004 | 0,61 |
| **40,0** | **8,0** | 0 | 0,845 | +0,019 | 0,60 |

**Tres cosas que salen de aquí:**

1. **El tope de la rejilla (λ=0,3) deja la activación en 39,9 %**, o sea por encima del 30 % que
   el propio encargo llama «λ es baja». **Los cuatro puntos de la rejilla caen en la misma
   zona**: 45,6 % → 39,9 % es todo el recorrido que compran. Con la rejilla tal cual, **la
   cláusula de activación del criterio de éxito no la puede cumplir ninguna combinación**, y
   entonces el éxito es inalcanzable por construcción, no por el resultado.
2. **La banda 5–15 % vive en λ ≈ 6–40**, o sea **20× a 130×** por encima del tope propuesto.
3. **Y ahí el Gabor Δ NO sube: baja** (+0,069 a λ=6, +0,004 a λ=20). Si eso aguanta a 30 épocas,
   la respuesta a *«¿aporta la esparsidad?»* es **no**, y es un resultado válido — pero conviene
   medirlo **en el rango donde la esparsidad existe de verdad**, no en uno donde ni siquiera se
   ha encendido.

⚠ **Lo que este tanteo NO dice.** Son 6 épocas, 8.000 ventanas, **un** punto de (k, K) y **una**
semilla; las cifras se moverán con 30 épocas sobre 84.000. Lo que sí es estructural es el
**orden de magnitud** del desajuste: 20–130× no lo explica el presupuesto de épocas.

### 3.7 ⚠⚠ `frac_activa` es una MEDIA que esconde una distribución bimodal

Salió al mirar por qué ningún run cumple la cláusula de «cero kernels muertos». *Medido el
2026-09-02*, activación **por canal** sobre validación, ordenada:

| run | activación de cada uno de los 16 canales (%) |
|---|---|
| `k5-K16-λ0` | `0 0 0 0 0 0,01 0,10` ┃ `99,91 99,94 99,95 99,96 99,97 99,97 99,97 99,97 99,97` |
| `k3-K16-λ0` | `0 0 0,01 0,03` ┃ `24,2 24,2 24,2` ┃ `75,8 75,8` ┃ `100 100 100 100 100 100 100` |
| `k3-K16-λcal` | `0 0 0` ┃ `3,4 4,3 4,9 4,9 5,4 5,4 9,1 9,2 10,5 16,2 17,0 17,5 22,2` |

**Con λ=0 el modelo se parte en dos: canales muertos y canales encendidos el 99,97 % del
tiempo.** Y un canal que nunca se apaga es, a efectos prácticos, **lineal**: su ReLU no recorta
nunca, así que no aporta ni no-linealidad ni selectividad. Es exactamente el desenlace que el
encargo predijo — *«sin este término el autoencoder converge a algo equivalente a PCA: base
válida pero difusa, sin estructura local»* — y explica de paso el `R² rec int` de **1,000**.

**Tres consecuencias, y ninguna es cosmética:**

1. **`frac_activa` = 56,2 % no describe a ningún canal.** No hay un solo canal cerca del 56 %:
   es la media de un montón de ceros y un montón de cienes. La banda 5-15 % del encargo supone
   una distribución unimodal, y aquí no lo es. **Hace falta reportar la distribución**, no su
   media — un diagnóstico que promedia dos poblaciones no diagnostica nada.
2. **El brazo λ=0 no es «el mismo modelo sin esparsidad»: es un autoencoder casi LINEAL.** Sigue
   siendo el control que el encargo pide, pero hay que leerlo como tal.
3. ⚠ **La calibración sí produce distribuciones sanas** (`λcal`: 3 muertos y el resto entre 3 %
   y 22 %), o sea que el problema no es la esparsidad: es su ausencia.

⚠ **Y esto pone en tensión dos cláusulas del criterio del encargo**, que hay que resolver antes
de leer el resultado: pide **activación 5-15 %** *y* **cero kernels muertos**. Empujar la
activación hacia abajo con L1 **mata unidades** — es el mecanismo de la L1, no un defecto — así
que las dos cláusulas tiran en direcciones opuestas. **Ningún run de los tres primeros cumple la
de cero muertos** (4, 3 y 7 de 16). Es la misma clase de problema que la rejilla de λ del §3.4:
una cláusula que nada puede satisfacer convierte el «fracaso» en aritmética en vez de en
resultado. **Es decisión del dueño**, y va al §6.

⚠ El diagnóstico por canal **no se añade al código con el tanteo en vuelo**: dejaría media
tanda instrumentada y media no. Se añade al terminar, y se recalcula para los 8 runs desde sus
checkpoints, que para eso se guardan.

### 3.5 `ConvTranspose2d` no admite `padding_mode='replicate'`

*Comprobado con torch 2.14*: lanza `ValueError`. El codificador replica el borde (igual que
`pad_mode: edge` de producción) y el decodificador **no puede**, así que reconstruye el anillo
exterior de `k//2` píxeles viendo ceros. Con k=9 son 4 de cada 10 píxeles por lado, o sea **no
es un detalle**: por eso se reporta también `err_rec_int` / `r2_rec_int` sobre el interior, que
es la cifra limpia. Hay un test que **falla** si una versión futura de torch lo admite, para que
la decisión se revise en vez de quedarse.

---

### 3.6 ⚠⚠ `enriq` está por DEBAJO de su nulo en toda la sonda — y se sabe por qué

Lo encontró el dueño revisando `metrics.py`: el enriquecimiento va de **0,47 a 0,61** en todo el
tanteo, cuando **1,0 es «indistinguible de aleatorio»**. Los kernels aprendidos tienen **menos**
energía en el subespacio clásico que kernels al azar.

Su hipótesis era que `classic_basis` construye los filtros a k>3 con suavizado binomial —o sea
plantillas de baja frecuencia— y que la normalización de contraste obligatoria del §2 quita DC y
las bajas frecuencias de la **entrada**, dejando los kernels aprendidos en alta frecuencia y casi
ortogonales a esa base. **Comprobada el 2026-09-02**, misma celda (k=7, K=16, 6 épocas,
`--limite 8000`), lo único que cambia es la normalización:

| | `enriq` | Gabor Δ | activa % | R² rec int |
|---|---:|---:|---:|---:|
| **CON** normalización, λ=0 | **0,47** | +0,070 | 45,6 | 0,975 |
| **CON** normalización, λ=3 | 0,52 | +0,083 | 22,3 | 0,969 |
| **SIN** normalización, λ=0 | **1,01** | −0,041 | 30,7 | 0,154 |
| **SIN** normalización, λ=3 | 1,01 | −0,041 | 30,7 | 0,129 |

Y el espectro radial medio lo explica del todo (potencia normalizada a su máximo, `r`=0 es DC):

| | r=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| entrada **cruda** | **1,000** | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| entrada **normalizada** | 0,673 | 0,554 | 0,759 | 0,870 | 0,858 | 0,670 | 0,877 | **1,000** | 0,867 | 0,573 | 0,298 |
| **base clásica** k=7 | **1,000** | 0,448 | 0,056 | 0,025 | | | | | | | |

Una vista de texto cruda tiene **prácticamente toda su potencia en DC** — que es exactamente por
qué el §2 obliga a normalizar, y lo confirma el `R² rec int` de 0,154 sin normalizar: sin ella la
sonda ni siquiera reconstruye. Pero normalizada, la entrada hace pico en `r`=7 y la base clásica
vive en `r` ∈ {0, 1}. Base de baja frecuencia contra kernels de alta frecuencia.

**Consecuencia, y es más grande de lo que él planteó:** *producción no normaliza el contraste* —
`fv.fovea.build_view` entrega la vista cruda en [0,1] y nada la toca después *(comprobado
2026-09-02)*. O sea que el **0,688** de `fov16-mask-p20` se midió sobre kernels entrenados con
entrada cruda, y el `enriq` de la sonda sobre entrada normalizada: **nunca fueron comparables**,
ni siquiera en k=3. Y el plan anterior apoyaba ahí toda la lectura del ancla.

Por eso `enriq` pasa a ser **diagnóstico, no criterio**, y entran las dos métricas sin plantilla
del §4.2. Tiene test: si alguien cambia `classic_basis` y deja de ser de baja frecuencia, falla y
hay que revisar la lectura en vez de silenciarlo.

---

## 4. El criterio, congelado ANTES de mirar

### 4.1 Lo que proponía el encargo (§7), y por qué no se queda

- **Éxito** si alguna configuración logra a la vez: mediana de R² Gabor **≥ 0,25 por encima de
  su línea base**, R² de reconstrucción **≥ 0,80**, activación **entre 5 % y 15 %**, y **cero
  kernels muertos**.
- **Fracaso** si ninguna separa el Gabor de su base por más de **0,10**.
- Si el mejor λ=0 **iguala** al mejor λ>0 en Gabor, **la esparsidad no aportó nada y hay que
  decirlo**.

Los umbrales absolutos **no se quedan**, y no por gusto: con los nulos medidos del §3.3, un
0,25 absoluto pide explicar el **52 % del margen alcanzable en k=5, el 38 % en k=7 y el 32 % en
k=9**. Es tres exigencias distintas escritas como si fueran una. La tercera frase —«si λ=0
iguala a λ>0, dilo»— **sí se queda tal cual**: es la única que ya era comparable.

### 4.2 El criterio que manda, desde el 2026-09-02

**Prueba (¿hay señal?)** — sin unidades, por celda:

> la mediana del run supera el **p95 de la mediana de K kernels aleatorios** del mismo tamaño
> (bootstrap de 2.000 remuestreos, `fv/probe/spectrum.py:bootstrap_p95`).

El estadístico que se compara es una mediana sobre K, así que el nulo tiene que ser la
distribución **de esa mediana**, no la de un kernel suelto. Se aplica a las tres métricas de
forma: `gabor_supera_p95`, `conc_orient_supera_p95`, `conc_banda_supera_p95`.

**Magnitud (¿cuánta?)** — normalizada por lo que es alcanzable en esa celda:

> `Δ / (1 − nulo) ≥ 0,40`

El **0,40 es del dueño y es negociable**; queda escrito aquí antes de mirar para que se sepa que
no se eligió después.

**Éxito** = alguna configuración pasa **las dos** en `conc_orient` **o** en Gabor, con
`r2_rec_int ≥ 0,80` y **cero kernels muertos**.

#### Qué puede detectar cada prueba, medido (2026-09-02, K=16)

Las dos condiciones son complementarias, y esta tabla dice por qué hacen falta **las dos**:

| `k` | nulo Gabor | p95(mediana de 16) | **Δ absoluto mínimo para pasar** | ...y eso es, del margen |
|---:|---:|---:|---:|---:|
| **3** | 0,888 | 0,928 | **0,0402** | **35,7 %** |
| 5 | 0,523 | 0,553 | 0,0296 | 6,2 % |
| 7 | 0,327 | 0,351 | 0,0237 | 3,5 % |
| 9 | 0,226 | 0,240 | **0,0141** | **1,8 %** |

⚠ **Lo contrario de lo que se esperaba, y por eso está medido.** Parecía que en 3×3, con el nulo
pegado al techo, la prueba se volvería hipersensible y dispararía con cualquier ruido. Es al
revés: **en k=3 es la MÁS estricta de las cuatro**. Con sólo 9 números por kernel la calidad del
ajuste varía mucho entre kernels aleatorios, así que la distribución del nulo es **ancha** en
términos absolutos (0,0402 contra 0,0141 en k=9) y pasar exige el **35,7 % del margen**.

**La consecuencia para leer la tabla:** la prueba del p95 es un test al 5 % en los cuatro `k`
—eso es lo que la hace correcta— pero el **tamaño de efecto** que detecta va de 1,8 % a 35,7 %
del margen. O sea que en k=9 dispararía con efectos triviales, y por eso **la condición de
magnitud (`Δ/(1−nulo) ≥ 0,40`) no es redundante**: es la que impide leer un 1,8 % como hallazgo.
Y en k=3 pasar la prueba ya implica un efecto grande de por sí.
**Fracaso** = ninguna pasa la prueba en ninguna de las tres métricas de forma. Es un resultado
válido: la reconstrucción tampoco produce filtros genéricos en este dominio.

### 4.3 λ deja de ser un eje: se **calibra** por celda

`λ` no significa lo mismo en cada celda —el mapa λ→activación del §3.4 sale de **un** punto—, y
una rejilla fija más el filtro por banda deja celdas enteras sin ninguna combinación admisible,
justo en el eje que lleva la premisa. Así que se bisecta en log(λ) hasta **activación 10 % ± 3**
(`fv/probe/calibrate.py`), la λ resultante se guarda como **dato del run**, y λ pasa a ser
`{0 = control, calibrada}`. La esparsidad queda constante entre celdas y el barrido mide lo que
dice medir.

⚠⚠ **Y la calibración se midió MAL la primera vez, de una forma que invirtió su propia
conclusión.** Queda escrito porque la lección vale más que el número:

La primera versión evaluaba cada λ con «2 épocas sobre un subconjunto de 8.000», o sea **64
pasos del optimizador**. Lo que asienta la activación son los **pasos**, no las épocas ni el
tamaño del dataset — *medido el 2026-09-02 en k=3/K=16, λ=80, con las MISMAS 8.000 ventanas*:

| pasos | 64 | 256 | 640 |
|---|---:|---:|---:|
| activación | **24,3 %** | 4,3 % | 3,6 % |

...y el run de verdad (84.000 ventanas, 329 pasos en su primera época) dio **4,1 % en la época 1**
y 1,8 % en la 30. O sea que el proxy **sobreestimaba la activación asentada seis veces**.

**Lo que eso produjo fue peor que imprecisión: fue la conclusión al revés.** La calibración
declaró k=3/K=8 y k=3/K=16 como `saturado=True, en_banda=False` — *«esta celda no puede
esparcirse hasta la banda»* — cuando la verdad es lo contrario: se esparcen de sobra. Con el
presupuesto de pasos correcto (400, `fv/probe/calibrate.py:PASOS`):

| celda | λ calibrada | activación | ¿en banda? | ¿satura? |
|---|---:|---:|---|---|
| k=3, K=8 | 28,3 | **9,6 %** | ✅ | no |
| k=3, K=16 | 28,3 | **7,5 %** | ✅ | no |
| k=5, K=16 | 28,3 | 8,7 % | ✅ | no |
| k=9, K=16 | 28,3 | 8,7 % | ✅ | no |

**No hay ninguna celda medida que sature.** El freno de saturación se conserva —un suelo real es
posible, y protege del λ absurdo— pero con una nota: si alguna celda satura, **sospecha primero
del presupuesto de pasos**.

⚠ Y un segundo fallo del mismo día, más pequeño y de la misma familia: el desempate «entre λ que
empatan gana la menor» usaba una tolerancia (1 punto) **más ancha que la banda misma**, así que
cambiaba una λ *dentro* de banda por una *fuera* — en k=3/K=16 elegía λ=10 (13,4 %, fuera) en vez
de λ=28 (7,5 %, dentro). Un desempate no puede tirar el criterio. Ahora se desempata **dentro de
la banda primero**. Los dos fallos tienen test.

**La lección, que es la que hay que llevarse:** una calibración cuyo proxy **no transfiere** es
peor que no calibrar, porque hace que la esparsidad *parezca* igualada entre celdas cuando no lo
está — y ese es exactamente el fallo silencioso que calibrar venía a evitar.

### 4.4 ⚠ Y el ancla ya no se puede leer por el enriquecimiento

El §3.6 mide que `enriq` está **por debajo de su nulo** en toda la sonda por la normalización de
contraste, y que **producción no normaliza**. O sea que el 0,688 de `fov16-mask-p20` y el
`enriq` de la sonda **nunca fueron comparables**, ni siquiera en k=3 — que era justo donde el
plan anterior apoyaba la lectura del ancla.

**El ancla se lee ahora por `conc_orient`**, cuyo nulo en 3×3 es 0,238 (techo 0,762) frente al
0,879 del Gabor (techo 0,121). ⚠ `conc_banda` **no** sirve en 3×3: un kernel de soporte 3×3 no
puede ser de banda estrecha (principio de incertidumbre), y su valor cae por debajo del nulo
incluso para un Gabor sintético. Eso no es un defecto de la métrica: es la premisa
—*«en 3×3 no cabe la estructura»*— saliendo por otro lado.

### 4.5 El tanteo del eje `k` va PRIMERO, y puede cerrar el estudio

**8 runs: k ∈ {3,5,7,9} × λ ∈ {0, calibrada}, K=16, train entero, 30 épocas.** ~1,7 h
*extrapolado de los 101,3 s/época medidos*. Es el punto 3 de la revisión del dueño, y su razón
es exacta: **todo lo medido hasta ahora cubre UN punto del eje `k`, y el eje `k` es la premisa
entera del experimento.**

⚠ Corre con el **train entero**, no con `--limite`: menos ventanas dan kernels más ruidosos y el
ajuste Gabor baja, así que `--limite` es una variable de confusión **justo sobre la métrica
principal**, y sesga hacia el fracaso.

**Qué decide, escrito antes:**

- **Si ninguna de las 8 pasa la prueba del §4.2 en ninguna métrica de forma**, el estudio se
  cierra aquí: la respuesta a *«¿salen filtros genéricos?»* es **no**, y la rejilla de 12-15 h
  no se lanza. Ya hay un indicio en esa dirección —a k=7 el Δ del Gabor es +0,07, o sea el
  **10,6 % del margen disponible**, plano entre λ=0 y λ=3 y luego bajando— pero un indicio en
  un punto no es un resultado en el eje.
- **Si alguna pasa**, la rejilla se lanza **sólo sobre los `k` que pasaron**, con K ∈ {8,16,32}
  y λ ∈ {0, calibrada}.

### 4.6 Lo que se reporta pase lo que pase

La **tabla de todas las combinaciones con sus métricas y sus nulos**, la **hoja de contactos** de
cada run, los **mapas `z`** de las 3 mejores, y el apartado de **«lo que quedó pendiente»**. Va
al repo central `estudios-redes-neuronales/reportes/estudios/2026/09-septiembre/`, con inicio y
fin en UTC, **máquinas: 0** (no alquila) y **coste: 0 $** — aquí el coste es **reloj**.

---

## 5. Lo que este estudio NO contesta

1. **Que un 9×9 entrenado aquí sea genérico NO dice que el 3×3 de producción pudiera serlo.**
   Son dos preguntas —*¿cabe la estructura?* y *¿hay presión?*— y la sonda mueve **las dos a la
   vez**. Por eso está el ancla k=3: es el único brazo que separa una de la otra.
2. **Que un kernel sea «genérico» no dice que sirva para la tarea.** Eso lo contesta la fase 2
   (§7 de aquí), y sólo se paga si la fase 1 sale bien.
3. **La reconstrucción no es la tarea.** Un código óptimo para reconstruir puede tirar justo lo
   que distingue una esquina. Es el desenlace que la fase 2 detectaría, y es barato.
4. **`patience` y el presupuesto de épocas.** Aquí son 30 épocas fijas para todos: no hay parada
   temprana, así que ninguna combinación recibe más entrenamiento que otra. Es una simplificación
   deliberada — y significa que una configuración lenta puede salir peor por no haber terminado.

---

## 6. Las cuatro preguntas, CONTESTADAS el 2026-09-02

| # | Pregunta | Respuesta del dueño | Dónde vive ahora |
|---|---|---|---|
| 1 | ¿E1 (λ hasta 30)? | **No en esa forma**: λ deja de ser eje y se **calibra por celda** | §4.3 · `fv/probe/calibrate.py` |
| 2 | ¿E2 y E3? | **Sí**, pero E3 se queda corta: el umbral se normaliza por el margen | §4.2 · §4.4 |
| 3 | ¿Rejilla entera o `--limite`? | **Ninguna todavía**: primero el tanteo del eje `k`, 8 runs | §4.5 |
| 4 | ¿Se quedan 0,25 / 0,10? | **No** | §4.2 |

Y una quinta cosa que él encontró revisando el código, que resultó ser la más importante: el
**§3.6** de arriba.

### 6.2 ⏳ ABIERTA: las dos cláusulas del criterio que tiran en direcciones opuestas

**Pendiente de decisión suya, y sale de medir (§3.7), no de opinar.** El encargo pide para el
éxito, a la vez, **activación 5-15 %** y **cero kernels muertos**. Bajar la activación con L1
**mata unidades** — es el mecanismo de la L1 — así que las dos cláusulas se estorban. En los
tres primeros runs mueren 4, 3 y 7 de 16, así que **tal cual, nada puede pasar**.

Tres salidas, sin recomendación todavía porque faltan k=7 y k=9:

1. **Aflojar a «menos del X % de kernels muertos»** (p. ej. 25 %). Reconoce que la L1 mata
   unidades y sigue rechazando el caso degenerado.
2. **Contarlo sobre los canales VIVOS**: pedir que los vivos estén en banda y reportar el K
   efectivo aparte. Es lo que de verdad se quiere saber — *«¿cuántos filtros útiles salen?»*.
3. **Dejarlo literal** y aceptar que el veredicto sea «fracaso» por esta cláusula. Es una opción
   legítima, pero entonces hay que decir en el reporte que el fracaso es por ahí y no por la
   forma de los kernels.


### 6.1 Una cosa suya que NO se aplicó, y por qué

En el punto 3(b) propuso que los 101 s/época *«en un modelo de ~1.500 parámetros no es cómputo,
es sobrecarga, casi seguro la normalización de contraste recalculada por época»*, y que
cachearla lo llevaría a segundos. **Medido, y no es así:**

- `local_contrast_norm` se llama **una vez** en `prepare()` y el resultado se cachea en disco; el
  bucle de entrenamiento sólo toca el tensor ya normalizado. No hay nada que cachear.
- Son **1.045 GFLOP por época** (84.000 ventanas × 400 px × 32 canales × 81 taps × 2 lados ×
  3 pases), o sea **14,3 GFLOP/s efectivos** en 2 vCPU — una cifra razonable para convolución
  float32 en CPU.
- El `conv2d` de torch **a pelo**, sin nada del proyecto en medio, ya cuesta **178 ms** de los
  **222 ms** del paso completo: **el 80 % del tiempo está dentro de la convolución**, y
  renormalizar el decodificador es el **0,0 %**.

O sea: no hay un 100× esperando. Lo que **sí** se aplicó entero es su punto 3(a) —`--limite` es
una variable de confusión sobre la métrica principal— y por eso el tanteo corre con el train
entero aunque cueste más.

---

## 7. Fase 2 — sólo si la fase 1 tiene éxito

Congelar el codificador de la mejor configuración como L1 de la **rama central** de
`fov16-optimo-mask`, con `k_center` y `channels[0]` iguales a sus `k` y `K`. La rama periférica
se entrena normal: la sonda se entrenó con 1 canal y la periferia recibe 2. Entrenar el resto
con la receta `plan40` y comparar contra el **f1 0,954** y el error de posición **1,05 px** de
`fov16-mask-p20`.

- Si el f1 **aguanta** con L1 congelada, los kernels son transferibles.
- Si se **hunde**, aprendieron a reconstruir y nada más — también es un resultado, y sale barato.

⚠ Ese modelo **cambia la forma de L2** y **no debe intentar cargar checkpoints anteriores**.

⚠ **La fase 2 no está implementada**, y es deliberado: depende de qué `k` y `K` gane, y escribir
hoy el cableado de una geometría que aún no se conoce es escribir la mitad que habrá que
rehacer.
