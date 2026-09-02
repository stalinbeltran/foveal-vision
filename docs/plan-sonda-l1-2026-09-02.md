# Plan — sonda L1: ¿pueden los kernels de la primera capa aprender filtros genéricos? (2026-09-02)

> **Estado: criterio escrito ANTES de mirar ningún número, y PENDIENTE de que el dueño
> confirme los umbrales.** Es lo que pide el §7 del encargo y el protocolo de este proyecto:
> quien decide qué cuenta como «gana» tiene que hacerlo sin ver el resultado, o el rango y el
> umbral acaban ajustándose a lo que salió. Los números se pegan abajo cuando lleguen; **este
> documento no se reescribe para que cuadre**.
>
> ⚠ **Lo que NO se ha hecho todavía: lanzar la rejilla.** Son 12,0 h *medidas* de esta máquina
> (§3.1). El código está entero y probado; el gasto espera a la confirmación del §6.

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

Las cuatro cosas de aquí abajo se midieron **antes** de lanzar nada. Tres de ellas afectan a los
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

Con 7 parámetros libres sobre 9 muestras, un Gabor ajusta **cualquier** 3×3. **Esto no invalida
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

### 3.5 `ConvTranspose2d` no admite `padding_mode='replicate'`

*Comprobado con torch 2.14*: lanza `ValueError`. El codificador replica el borde (igual que
`pad_mode: edge` de producción) y el decodificador **no puede**, así que reconstruye el anillo
exterior de `k//2` píxeles viendo ceros. Con k=9 son 4 de cada 10 píxeles por lado, o sea **no
es un detalle**: por eso se reporta también `err_rec_int` / `r2_rec_int` sobre el interior, que
es la cifra limpia. Hay un test que **falla** si una versión futura de torch lo admite, para que
la decisión se revise en vez de quedarse.

---

## 4. El criterio, escrito ANTES de mirar

### 4.1 Tal como lo propone el encargo (§7)

- **Éxito** si alguna configuración logra **a la vez**: mediana de R² Gabor **≥ 0,25 por encima
  de su línea base aleatoria**, R² de reconstrucción **≥ 0,80**, activación **entre 5 % y 15 %**,
  y **cero kernels muertos**.
- **Fracaso** si **ninguna** separa el Gabor de su línea base por más de **0,10**. Sería un
  resultado válido: la reconstrucción tampoco produce filtros genéricos en este dominio.
- Si el mejor λ=0 **iguala** al mejor λ>0 en la métrica Gabor, **la esparsidad no aportó nada y
  hay que decirlo**.

### 4.2 Las tres enmiendas que las medidas del §3 obligan a proponer

Se proponen; **no se aplican sin confirmación** (§6). Cada una nace de un número, no de un gusto:

| # | Enmienda | Por qué | Si NO se aplica |
|---|---|---|---|
| **E1** | **λ ∈ {0 · 0,3 · 3 · 10 · 30}** en vez de `{0 · 0,03 · 0,1 · 0,3}` | §3.4: la rejilla propuesta no sale de la zona 40-45 % de activación | el criterio de éxito es **inalcanzable por construcción**, y el estudio no puede contestar «¿aporta la esparsidad?» |
| **E2** | La cláusula de activación (5-15 %) pasa de **requisito de éxito** a **filtro de admisión**: sólo se juzgan por Gabor las combinaciones que caen en banda; las demás se reportan con su activación al lado | así la métrica principal se lee entre configuraciones **comparables**; el encargo ya llama a la activación «un diagnóstico, no va en la pérdida» | una configuración con Gabor Δ alto y 45 % de activación se descarta sin que nadie sepa si la esparsidad tenía algo que ver |
| **E3** | El umbral de Gabor Δ (0,25 / 0,10) **se juzga sólo en k ∈ {5,7,9}**; **k=3 se lee por la métrica 5** (enriquecimiento) | §3.3: en 3×3 el techo de Δ es 0,121, o sea **menos** que el propio umbral de éxito | el ancla «fracasa» siempre, por aritmética, y ese falso fracaso contamina la lectura de las otras columnas |

⚠ **E1 cuesta lo mismo**: 5 valores de λ en vez de 4 son 60 runs en lugar de 48, o sea **~15,0 h
en vez de 12,0**. Si el reloj aprieta, `--limite 20000` lo baja a ~5,0 h *(medido: 34,0 s/época)* — a costa de entrenar
sobre 20.000 ventanas en vez de 84.000, que es una decisión aparte y también hay que tomarla.

### 4.3 Lo que se reporta pase lo que pase

Éxito, fracaso o punto medio, el reporte lleva: la **tabla de las 48/60 combinaciones con las
ocho métricas**, la **hoja de contactos** de cada run, los **mapas `z`** de las 3 mejores, y el
apartado de **«lo que quedó pendiente»**. Va al repo central
`estudios-redes-neuronales/reportes/estudios/2026/09-septiembre/`, con inicio y fin en UTC,
máquinas (0: no alquila) y coste real (0 $: es CPU propia, el coste es **reloj**).

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

## 6. Lo que hace falta confirmar antes de gastar las 12 h

1. **¿Se aplica E1 (λ hasta 30)?** Es la única que cambia lo que se mide; sin ella el criterio de
   éxito no lo puede cumplir nada.
2. **¿Se aplican E2 y E3?** Cambian cómo se **lee**, no qué se corre. Se pueden decidir después
   de ver los números sin contaminar nada, porque están escritas aquí antes.
3. **¿Rejilla entera (84.000 ventanas, ~12-15 h) o `--limite 20000` (~4-5 h)?**
4. **¿Los umbrales 0,25 / 0,10 se quedan como están?** Con los nulos del §3.3 medidos, 0,25 en
   k=9 es exigir explicar el 45 % del margen disponible.

Mientras tanto **no se lanza nada**. Un `--cronometrar` y los dos tanteos del §3.4 es todo lo
que se ha gastado, y son minutos.

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
