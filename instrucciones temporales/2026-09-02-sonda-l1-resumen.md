# Resumen de lo dicho sobre el encargo de `instruccioneslargas.md` (sonda L1)

**Fecha:** 2026-09-02 · **Última actualización:** tras tu revisión y sus consecuencias
(`c19cb745b`). Commits en `main` de `foveal-vision`: `65dd66024` → `a9095531c` → `ad4a4b8c6` →
`c19cb745b`, todos empujados.

> Esto es **un resumen de lo que te conté**, no una fuente nueva. Lo que manda es:
> el encargo en [`instruccioneslargas.md`](../instruccioneslargas.md), el criterio en
> [`docs/plan-sonda-l1-2026-09-02.md`](../docs/plan-sonda-l1-2026-09-02.md), y el bloque
> `⏳ 2026-09-02 — SONDA L1` al principio de [`CLAUDE.md`](../CLAUDE.md).
> Si algo de aquí choca con alguno de esos tres, **gana el otro**.

---

## 1. Dónde está esto ahora mismo

**El encargo está implementado entero, el criterio está congelado, y corre el tanteo del eje `k`**
(8 runs, ~1,7 h, no alquila nada). La rejilla grande de 12-15 h **sigue sin lanzarse**, y el
criterio de cierre —escrito antes de ver los números— dice que puede no lanzarse nunca.

El camino fue: implementar el encargo → tú contestaste las cuatro preguntas y encontraste un
quinto problema → rediseño → lanzar → **el propio tanteo destapó un fallo mío** → parar,
arreglar, relanzar.

---

## 2. El encargo, y lo que faltaba

De las siete secciones, al empezar faltaban cinco. Lo implementado:

| § | Qué | Dónde |
|---|---|---|
| 1 | el autoencoder de una capa por lado | `src/fv/probe/model.py` |
| 2 | normalización de contraste local | `src/fv/probe/data.py` |
| 3 | esparsidad + renormalización L2 tras **cada** paso | `probe/run.py`, `probe/model.py` |
| 4 | rejilla, `--cronometrar`, `--repetir-mejores`, `--tanteo-k` | `scripts/sonda_l1.py` |
| 5 | las ocho métricas, cada una con su nulo | `probe/metrics.py`, `probe/gabor.py` |
| 5+ | **dos métricas sin plantilla** (tu punto 5) | `probe/spectrum.py` |
| 6 | artefactos, hojas de contactos, mapas `z`, tabla | `probe/run.py`, `figures.py`, `table.py` |
| — | **λ calibrada por celda** (tu punto 1) | `probe/calibrate.py` |
| — | el lanzamiento desacoplado, con su código de salida bien puesto | `scripts/sonda_l1_desacoplada.sh` |

**§6, el aislamiento, tiene test**: `fv.probe` no importa `fv.models`, y de `fv.fovea` sólo
`build_view`/`dims_of`. Un import de más no rompe nada visible — ata el experimento a la red que
estudia — así que se comprueba leyendo el código, no la documentación.

---

## 3. Tu revisión: qué acepté y qué no

### 3.1 Punto 1 — λ deja de ser eje, se **calibra** por celda ✅

Tu crítica era correcta y yo no la había visto: mi mapa λ→activación salía de **un** punto
(k=7, K=16), y con la rejilla fija más el filtro por banda, celdas enteras se quedaban sin
combinación admisible — justo en el eje que lleva la premisa. Ahora se bisecta en log(λ) hasta
activación 10 % ± 3 y la λ resultante se guarda como dato del run.

### 3.2 Punto 2 — el umbral normalizado ✅

Tienes razón en que E3 se quedaba corta. Manda `Δ > p95 de la mediana de K kernels aleatorios`
(bootstrap, sin unidades) y la magnitud es `Δ/(1−nulo)`. Tu 0,40 queda escrito **como tuyo y
negociable**.

**Y midiendo qué puede detectar cada prueba salió lo contrario de lo que yo esperaba** (K=16):

| `k` | nulo Gabor | Δ absoluto mínimo para pasar | ...del margen |
|---:|---:|---:|---:|
| **3** | 0,888 | **0,0402** | **35,7 %** |
| 5 | 0,523 | 0,0296 | 6,2 % |
| 7 | 0,327 | 0,0237 | 3,5 % |
| **9** | 0,226 | **0,0141** | **1,8 %** |

Yo esperaba que en 3×3, con el nulo pegado al techo, la prueba fuese hipersensible. Es **la más
estricta de las cuatro**: con sólo 9 números por kernel el ajuste varía mucho entre kernels
aleatorios, así que el nulo es **ancho**. Y de paso demuestra por qué hacen falta **las dos**
condiciones: en k=9 el p95 dispararía con efectos triviales (1,8 %), y la de magnitud es la que
impide leer eso como hallazgo.

### 3.3 Punto 3 — el tanteo del eje `k` ✅, pero tu diagnóstico del coste ❌

**Acepté (a) entero**: `--limite` es variable de confusión justo sobre la métrica principal
—menos ventanas → kernels más ruidosos → Gabor más bajo— así que sesga hacia el fracaso. El
tanteo corre con el train entero aunque cueste más.

**Acepté (c) entero**, y es tu punto más fuerte: todo lo medido cubría **un** punto del eje `k`,
y el eje `k` es la premisa entera. Es lo que está corriendo.

**(b) no se sostiene, y lo digo con la medida.** Propusiste que 101 s/época es sobrecarga, «casi
seguro la normalización de contraste recalculada por época»:

- `local_contrast_norm` se llama **una vez** en `prepare()` y se cachea en disco. El bucle sólo
  toca el tensor ya normalizado. No hay nada que cachear.
- Son **1.045 GFLOP por época**, o **14,3 GFLOP/s** efectivos en 2 vCPU.
- El `conv2d` de torch **a pelo** ya cuesta **178 ms de los 222 ms** del paso completo: **el 80 %
  del tiempo está dentro de la convolución**, y renormalizar es el **0,0 %**.

No hay un 100× esperando.

### 3.4 Punto 4 — 0,25 / 0,10 fuera ✅

### 3.5 Punto 5 — confirmado, y con el mecanismo medido ✅

Tu hipótesis era exacta. Misma celda, lo único que cambia es la normalización:

| | `enriq` | R² rec int |
|---|---:|---:|
| **CON** normalización | **0,47** | 0,975 |
| **SIN** normalización | **1,01** | 0,154 |

El espectro radial lo cierra: la entrada **cruda** tiene prácticamente toda su potencia en DC
(1,000 en r=0, 0,000 en el resto); la **normalizada** hace pico en r=7; y la **base clásica** a
k=7 vive en r ∈ {0, 1} y luego ~0. Base de baja frecuencia contra kernels de alta frecuencia.

⚠⚠ **Y hay una consecuencia mayor que la que planteaste:** *producción no normaliza el contraste*
— `fv.fovea.build_view` entrega la vista cruda en [0,1] y nada la toca después. Así que el
**0,688** de `fov16-mask-p20` y el `enriq` de la sonda **nunca fueron comparables**, ni siquiera
en k=3 — que era justo donde E3 apoyaba la lectura del ancla. **El ancla se lee ahora por
`conc_orient`**, cuyo nulo en 3×3 es 0,238 (techo 0,762) frente al 0,879 del Gabor (techo 0,121).

**Las dos métricas que pediste**, de la FFT 2D con rejilla de frecuencia común por relleno a
ceros. Gabor sintético a k=9: `conc_orient` **0,923** contra nulo **0,112**.
⚠ `conc_banda` **no** sirve en 3×3: un soporte 3×3 no puede ser de banda estrecha (principio de
incertidumbre), y cae por debajo del nulo incluso para un Gabor sintético. No es defecto de la
métrica: es la premisa del experimento saliendo por otro lado.

---

## 4. ⚠ Lo que retiré: el "suelo de activación en k=3" NO existe

**Lo llegué a escribir en tres sitios y era falso.** Lo destapó el propio tanteo: el run
`k3-K16` con la λ calibrada predijo **24,3 %** de activación y dio **1,9 %**.

Separé las dos causas posibles con las **mismas** 8.000 ventanas, cambiando sólo los pasos:

| pasos | 64 *(lo que medía la calibración)* | 256 | 640 |
|---|---:|---:|---:|
| activación | **24,3 %** | 4,3 % | 3,6 % |

Y el run real (84.000 ventanas, 329 pasos en su primera época) dio **4,1 % en la época 1**.
**Lo que asienta la activación son los pasos del optimizador**, no las épocas ni el tamaño del
dataset. «2 épocas sobre un subconjunto» son 64 pasos, y **sobreestiman seis veces**.

Lo grave no fue la imprecisión sino que **invirtió la conclusión**: declaraba «esta celda no
puede esparcirse hasta la banda» justo donde se esparce de sobra. Con 400 pasos:

| celda | λ | activación | ¿en banda? | ¿satura? |
|---|---:|---:|---|---|
| k=3, K=8 | 28,3 | 9,6 % | ✅ | no |
| k=3, K=16 | 28,3 | 7,5 % | ✅ | no |
| k=5, K=16 | 28,3 | 8,7 % | ✅ | no |
| k=9, K=16 | 28,3 | 8,7 % | ✅ | no |

**Ninguna celda medida satura.**

Y un segundo fallo del mismo día, de la misma familia: el desempate «entre λ que empatan gana la
menor» usaba una tolerancia (1 punto) **más ancha que la banda**, así que cambiaba una λ *dentro*
de banda por una *fuera*. Un desempate no puede tirar el criterio que lo precede.

**La lección, que es lo que se lleva el plan:** una calibración cuyo proxy **no transfiere** es
peor que no calibrar, porque hace que la esparsidad *parezca* igualada entre celdas cuando no lo
está — justo el fallo silencioso que calibrar venía a evitar.

---

## 5. Un resultado parcial, y con cautela

Del primer intento sobreviven dos runs de **λ=0** (no usan calibración, así que siguen siendo
válidos). En `k3-K16-λ=0`:

| | valor | ¿pasa? |
|---|---:|---|
| Gabor Δ/margen | **+0,651** | **sí, supera el p95** |
| `conc_orient` Δ | −0,023 | no |
| `conc_banda` Δ | −0,055 | no |

**Es un conflicto entre métricas, y no saco conclusiones de un run.** Si el patrón «Gabor sí,
espectrales no» aguanta a lo largo del eje, hay que decidir cuál de las dos lecturas manda — y
esa decisión es tuya. El plan hoy dice `conc_orient` **o** Gabor, y este caso es justo donde esa
disyunción se vuelve importante.

---

## 6. El criterio, congelado antes de ver nada

**Prueba:** la mediana del run supera el **p95 de la mediana de K kernels aleatorios** (bootstrap
de 2.000 remuestreos). **Magnitud:** `Δ/(1−nulo) ≥ 0,40`.

**Éxito** = alguna configuración pasa las dos en `conc_orient` **o** en Gabor, con
`r2_rec_int ≥ 0,80` y cero kernels muertos.
**Fracaso** = ninguna pasa la prueba en ninguna métrica de forma. Es un resultado válido.

**Y el tanteo puede cerrar el estudio:** si ninguno de los 8 runs pasa, **la rejilla de 12-15 h
no se lanza**. Si alguno pasa, se lanza **sólo sobre los `k` que pasaron**.

---

## 7. Coste, medido

| Qué | Reloj |
|---|---:|
| combinación más cara (k=9, K=32), por época | **101,3 s** |
| **el tanteo del eje `k`** (8 runs, K=16, 30 épocas) | **~1,7 h** ← corriendo |
| la rejilla completa, si llega a lanzarse | ~12-15 h |

**No alquila nada.** El coste es **reloj** de esta máquina, y el freno lo cuenta:
`🔴 NO CERRAR — 1 trabajo(s) vivo(s): sonda_l1.py`.

---

## 8. Lo verificado ejecutando

| Qué | Resultado |
|---|---|
| suite de `foveal-vision` | **577 pasan**, 3 skip (eran 533) |
| tests de la sonda | **70** (eran 26) |
| los tests de cada arreglo | fallan con el código anterior, comprobado con `git stash` |
| la unidad de systemd | `active`, cgroup **propio** `/system.slice/sonda-l1.service` |
| el freno | nombra la sonda con ella corriendo |

⚠ Y una cosa que salió al verificar y ya está arreglada: `desacoplar-persistente.sh` usa
`Restart=on-failure`, y el aviso iba al final de la tubería — **el código de salida lo decidía
`notify.mjs`**. Medido: la sonda terminó bien, el aviso falló, y la unidad se quedó
**reiniciándose cada 30 s**. Y al revés, un trabajo que reventara salía como `success` y no se
reintentaba. Ahora manda el código del **trabajo**, con un test por cada dirección.

---

## 9. Qué falta

1. **Que termine el tanteo** (~1,7 h). Resultados en `foveal-vision-data/sondas/l1/`
   (`tabla.md`, `resumen.json`), log en `/tmp/sonda-l1.log`.
2. **Leerlo contra el criterio del §6** y decidir si la rejilla se lanza o el estudio se cierra.
3. **Si el conflicto Gabor/espectrales aguanta**, decidir cuál manda (§5).
4. **El reporte** al repo central `estudios-redes-neuronales`, en
   `reportes/estudios/2026/09-septiembre/`, con inicio y fin UTC, **máquinas: 0** y
   **coste: 0 $** — aquí el coste es reloj.
5. **La fase 2 (§8 del encargo) no está implementada, a propósito**: depende de qué `k` y `K`
   ganen, y sólo se paga si la fase 1 sale bien.

---

<sub>**Fuera de este encargo, en la misma sesión:** pediste el merge de `tema-2` a `main`. No
había nada que fusionar —en los cinco repos `tema-2` es ancestro de `origin/main`—, pero al
comprobarlo salió un falso positivo del freno: `git-pendiente.mjs` se comparaba contra
`origin/<rama>` en vez de contra todos los remotos, y una rama remota vieja hacía que `main`
entero se leyera como «se perdería». Arreglado y empujado en `telegram-coordinator`
(`efbb033`), con tres tests, dos de los cuales fallan con el código anterior.</sub>
