# Rehacer los tres ejes que quedaban, con 5 semillas — criterio escrito ANTES

> **Este documento se commitea antes de que exista un solo run de estos recorridos.**
> Su valor entero está en eso: las reglas de abajo se pueden comprobar contra el commit.
> Si se cambian después de ver resultados, el plan deja de decidir nada
> ([protocolo.md](protocolo.md) §1). Lo que se cambie igual va en un apartado aparte, con
> la hora, como en [plan-40h.md](plan-40h.md) §7 y [plan-lr-alto.md](plan-lr-alto.md) §6.

Fecha: 2026-08-24. Recorridos: **`bs5-L4`** (`batch_size`), **`nl5-L4`** (`n_layers`),
**`d5-L4`** (`d`). Dataset B: **`dirty1000-80px-16px-r20260824`**, que es nuevo y §3 dice por qué.

## 0. Qué pregunta responde, y por qué estos tres ejes y no otros

`lr` acaba de cerrarse por los dos lados ([plan-lr-L4.md](plan-lr-L4.md) por la izquierda,
[plan-lr-alto.md](plan-lr-alto.md) §6 por la derecha). Quedan los **demás ejes que este
proyecto llegó a estudiar de verdad**, y son exactamente tres. La lista no es una opinión:
un estudio de este proyecto es un recorrido con **N semillas** (*«un resultado sin N
semillas es una anécdota»*, protocolo.md), y sólo cuatro ejes lo tuvieron alguna vez:

| eje | dónde se estudió | semillas | ¿se rehace? |
|---|---|---|---|
| `lr` | `fast-lr`, `fast-lr-2`, `d1000-lr-1`, `p40-lr-L4`, `lr-alto-L4` | 5 / 5 / 5 / 5 / 3 | no — cerrado el 2026-08-23 |
| **`batch_size`** | `batch_size-1`, `batch_size-2`, `d1000-batch_size-1` | 5 | **sí**, §2.1 |
| **`n_layers`** | `p40-confirm-n_layers`, `plana-confirm-s0-n_layers` | 5 | **sí**, §2.2 |
| **`d`** | `proxy-c-d` | 5 | **sí**, §2.3 |

⚠ **Lo que NO entra, y por qué se dice:** `k_center`, `k_periph`, `s_center`, `s_periph`
y `channels` se barrieron alguna vez, pero **con una sola semilla**
(`sweeps/dirty-80px-fast_kcenter`, `_kperiph_1`, `_s_center_1`, `_s_periph_1`, borrados
desde entonces; y `p40-screen-width` / `p40-screen-kernel`, que el propio plan-40h.md §2
declara *«un cribado de una semilla no declara ningún ganador»*). Por la regla del proyecto
eso no es un estudio, así que aquí no hay nada que *rehacer*: habría que **hacerlo por
primera vez**, que es otra decisión y no ésta. Queda apuntado como lo que está sin medir.

## 1. Constantes fijadas antes (las tres iguales — es OAT)

| | valor | de dónde sale |
|---|---|---|
| base de red | **`ws16-p2-d2-L4`** (`n_layers=4`, `channels=[16,16,16,16]`) | el vigente; el mismo `base_label` que `p40-lr-L4` y `lr-alto-L4`. Verificado al derivarlo |
| receta | **`plan40`** (`lr` 0,0014, adam, `batch_size` 85, `patience` 10, sin scheduler) | los ganadores vigentes; sólo se mueve **un** eje por recorrido |
| tope de épocas | **150** | alto a propósito: **`patience` tiene que ser quien pare**, no el tope (R1). Es el defecto que invalida los estudios de `batch_size` de julio, §2.1 |
| `seeds` | **5** (1..5) | lo que pidió el usuario, y la misma N que `p40-lr-L4` y `p40-confirm-n_layers`, para que las bandas sean comparables |
| métrica de ranking | `val_f1` **del checkpoint** | [metrica-de-tarea.md](metrica-de-tarea.md) §9.7 |
| monitor de `best.pt` | `val_loss` (el de `plan40`) | ⚠ monitor y objetivo no coinciden: el ranking describe un checkpoint elegido por **otro** criterio. Es legal y ya pasaba en los estudios anteriores, pero el lector tiene que saberlo (`sweep_trials` lo declara en `monitor_matches_objective`) |
| δ (banda de ruido) | **1-SE de las semillas del mejor punto**, calculada al cerrar | `suggest_winner`, la regla del proyecto |
| device | `cpu`, con **8 hilos de torch en todas las máquinas** | máquinas distintas no deben entrenar distinto por tener más núcleos |
| familia de CPU | **`--cpu 'E5-26'`** | MEDIDO (plan-lr-alto §7.4): dentro de Xeon E5-26xx v3/v4 el entrenamiento sale idéntico bit a bit; al cruzar de familia diverge hasta 0,0457 en f1. Y es lo que hace **inocua** la criba de §4.1 |

**El orden de cada rango es la mitigación** (misma idea que plan-40h.md §7.3): con el
reparto por semilla, cada máquina entrena los valores **en el orden de la lista**, así que
si algo se corta lo que sobrevive es el **vigente y su vecindario**, y lo que falta son los
extremos. El ranking agrega por valor, no por orden: **esto no cambia ningún resultado**,
sólo qué se pierde si algo se corta.

## 2. Los rangos, y de qué medida sale cada uno

### 2.1 `batch_size` — `[85 · 57 · 128 · 38 · 192]` (recorrido `bs5-L4`)

**Por qué se rehace, y es el caso más claro de los tres.** Los tres estudios anteriores
comparten un defecto que los invalida para decidir hoy: **los 105 runs pararon por el tope
de 20 épocas, ninguno por `patience`.** Medido, releyendo sus `summary.json`: `stopped_early`
es `false` en los 35 de `batch_size-1`, los 35 de `batch_size-2` y los 35 de
`d1000-batch_size-1`. Es el mismo defecto que plan-40h.md §0 denunció: *«aquellos recorridos
midieron en parte velocidad de convergencia, no calidad»*.

Y con `batch_size` ese defecto no es neutral, **está sesgado en una dirección conocida**: a
épocas fijas, un batch grande da menos actualizaciones, así que llega menos lejos. Un tope
que no se agota penaliza sistemáticamente a los valores altos. Se ve en los números:

| estudio | red | receta | rango | ganador | ¿tope? |
|---|---|---|---|---|---|
| `batch_size-1` | L2 | `corta` | 100…1000 | **100** (el más pequeño del rango) | 35/35 |
| `batch_size-2` | L2 | `corta` | 10…100 | **25** | 35/35 |
| `d1000-batch_size-1` | L2 | `mejorada` | 10…100 | **85** | 35/35 |

Tres estudios y tres respuestas distintas. El vigente (85) sale del tercero — y **nunca se ha
medido sobre la red vigente (L4) dejando que `patience` decida**.

**El rango:** log-espaciado con factor **1,5** alrededor del vigente, cinco valores:
`38 · 57 · 85 · 128 · 192`. Cubre 5× y deja el vigente en el centro, acotado por los dos
lados — que es la condición que plan-lr-alto §0 dejó escrita como definición de «óptimo»:
*«un óptimo en el borde no es un óptimo: es el final de la regla»*.

- **Sube hasta 2,26×.** Es la dirección que el tope de 20 épocas penalizaba, así que es
  donde puede haber algo que los estudios viejos no pudieron ver.
- **Baja sólo hasta 0,45×**, y no hasta el 25 que ganó `batch_size-2`. Dos razones: aquel
  ganador es el más contaminado por el defecto (a épocas fijas, batch pequeño = más pasos =
  más lejos), y bajar es **la dirección cara** — MEDIDO en `d1000-batch_size-1`, batch 10
  costó 109,2 s/época contra 62,8 de batch 100. Con 38 ya se prueba ese lado.
- **No se extiende a 250+**, aunque `batch_size-1` lo midiera: allí el f1 se hundía a 0,49
  y peor, y confirmar lo evidente gasta presupuesto que hace falta para los otros dos ejes.

### 2.2 `n_layers` — `[4 · 3 · 5 · 2]` (recorrido `nl5-L4`)

**Por qué se rehace.** `p40-confirm-n_layers` (2026-08-07) sí fue un estudio correcto: 5
semillas, receta `plan40`, y **ninguno de los 20 runs paró por el tope** — paró `patience`.
Su respuesta fue clara y aguantó también por métrica de tarea (plan-40h.md §8): gana **4**,
y contra L2 la diferencia sobrevive a una permutación exacta (p = 0,032).

Lo que ha cambiado desde entonces es **el dato**: aquel estudio corrió sobre
`dirty1000-80px-16px`, cuyo `windows.npz` ya no se puede reconstruir con la misma huella
(§3, y plan-lr-alto §2 lo documentó primero). O sea que la pregunta que este recorrido
contesta no es *«¿cuál gana?»* sino **«¿sigue ganando el 4 sobre el dato de hoy?»** — una
replicación, que es lo que el usuario pidió y lo único honesto que se puede pedir cuando el
dato de abajo se ha movido.

**El rango es el mismo, `[2,3,4,5]`, a propósito.** Una replicación que cambia el rango deja
de ser una replicación. Y el rango ya está acotado por los dos lados: ganó el 4, con el 5
(0,8832) y el 2 (0,8756) por debajo. No se añade el 6 porque es el punto **más caro de todos**
(≈1,44× el coste de L4 por época, extrapolando la serie medida) y sólo confirmaría una caída
que el 5 ya muestra.

⚠ Un aviso que viaja con este eje: el 5 tuvo `sem` 0,0170 frente a 0,0041 del 4 —
**cuatro veces más dispersión**. Con 5 semillas se volverá a ver; no es ruido de medición,
es que a esa profundidad el entrenamiento es inestable.

### 2.3 `d` — `[2 · 1 · 3 · 4]` (recorrido `d5-L4`)

**Por qué se rehace.** `proxy-c-d` (5 semillas, rango `auto` = `[1..6]`) midió el eje **plano**:

| `d` | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| f1 | 0,6213 | **0,6244** | 0,6089 | 0,5965 | 0,6112 | 0,6063 |
| sem | 0,0219 | 0,0153 | 0,0125 | 0,0131 | 0,0104 | 0,0112 |

El ganador (2) le saca **0,0031** al segundo, con un `sem` de 0,0153: cinco veces menos que
la banda. Ese estudio no distingue 1 de 2, y él mismo lo dice. Además arrastra los mismos
tres defectos: **L2** (no la red vigente), **tope de 20 épocas** en los 30 runs, y el dataset
`dirty-paragraphs-fast-80px`, que no es el del proyecto.

**El rango: `[1,2,3,4]`,** recortado del `auto` `[1..6]`. Se quitan el 5 y el 6 porque el
estudio anterior los midió por debajo del 2 y porque, si el eje vuelve a salir plano, gastar
dos puntos × 5 semillas en la cola derecha es presupuesto que no compra información. El
vigente (2) queda en el interior, acotado por los dos lados.

## 3. ⚠ El dato es NUEVO, y por tercera vez hay que decirlo

El `windows.npz` de este estudio se ha reconstruido en esta máquina (efímera, recién hecha)
desde los specs congelados del generador. **Su huella no coincide con ninguna de las
anteriores**, y las tres huellas conocidas de la misma cadena tampoco coinciden entre sí:

```
manifest de git (2026-07-27, dirty1000-80px-16px)    : sha256:4327325b8e30…
reconstrucción del 2026-08-23 (…-r20260823)          : sha256:13786b8649…
reconstrucción de hoy          (…-r20260824)         : sha256:3df67624f5…
```

**Y esta vez la diferencia se ve en el contenido resumido, no sólo en la huella.** El
2026-08-23 coincidían campo a campo `num_windows`, `windows_per_split` y **los cuatro
positivos por esquina**; hoy los positivos **no** coinciden:

| | TL | TR | BR | BL |
|---|---|---|---|---|
| 23-ago (`r20260823`) | 17.121 | 17.613 | 19.357 | 18.852 |
| **hoy** (`r20260824`) | **17.043** | **17.564** | **19.198** | **18.575** |
| diferencia | −78 | −49 | −159 | −277 |

Es entre un 0,3 % y un 1,5 % de las esquinas positivas. `num_windows` (140.000),
`windows_per_split` (84.000/28.000/28.000), `images.shape`, `label_window` y `corner_order`
sí coinciden. O sea que la rejilla de ventanas es la misma y **lo que se movió son los
bordes de los párrafos**: exactamente lo que produce rasterizar con otro binario de Chromium.
Es la confirmación de la causa que se conjeturaba arriba, y no una sospecha nueva.

Y esta vez **se sabe por qué**, que es lo que las dos veces anteriores no se pudo decir: el
generador rasteriza con Chromium, y su propio README ya avisaba de que *«la rasterización de
Chromium entre SO»* es la salvedad de la reproducibilidad. En esta máquina **el Chromium que
Playwright fija no se puede descargar** (la CDN devuelve 403 desde este proveedor:
*«this service is not available in your location»*), así que se rasterizó con
**google-chrome-stable** instalado desde `dl.google.com`. Es otro binario, y por tanto otros
píxeles.

**La consecuencia, dicha entera: los tres recorridos de este estudio son comparables entre
sí y con el `lr` de agosto sólo con reservas.** Por eso el dataset lleva **nombre nuevo** en
vez de reutilizar el de ayer — es la regla que el propio descriptor tiene escrita: *«Si algún
día cambia, cambia también el nombre en vez de reutilizarlo»*. Reutilizarlo haría que dos
datasets distintos compartieran nombre, que es la forma de que la no-comparabilidad no se
note nunca.

**Qué se salva y qué no:**

- **Se salvan los tres estudios enteros.** Los 65 runs entrenan sobre **el mismo
  `windows.npz`**, extraído una vez aquí y enviado hecho a todas las máquinas (§4). Todas
  las comparaciones *dentro* de estos recorridos —que son las que contestan las preguntas—
  son válidas.
- **No se salva la comparación directa** con los números de julio ni con los del 23 de
  agosto. Se reportarán como medidas nuevas, no como continuidad.
- **Hay un ancla, y es gratis.** Los tres recorridos contienen el **vigente** en su rango
  (`batch_size` 85, `n_layers` 4, `d` 2), y los tres vigentes juntos **son la configuración
  de `lr-alto-L4` a `lr` 0,0014**, que midió `val_f1` = 0,9246 ± 0,0003. O sea que este
  estudio re-mide ese punto **tres veces por semilla**, y la distancia con 0,9246 es una
  medida de cuánto movió el dato. Se publica ese número en §7.

## 4. Cómo se corre: una máquina por RECORRIDO × SEMILLA (15 máquinas)

`scripts/estudio_flota.py` con `--reparto seed` y los tres recorridos en **una sola flota**:
15 lotes, uno por (recorrido, semilla), cada uno con su máquina. Es literalmente *«cada
parámetro y cada semilla en servidores independientes»*.

**Se reparte por semilla y no por valor del eje, y es la decisión que más importa.** La
semilla es el eje réplica, no la pregunta: repartiendo así, cada máquina mide **todos** los
valores de su eje, de modo que si una máquina es más lenta o más rara, esa rareza entra por
igual en todos los valores que se comparan — es un bloque y se cancela. Repartir por valor
haría lo contrario: la máquina quedaría **confundida con la respuesta**. Eso no se arregla
después con estadística, y por eso el script no lo ofrece.

**Los tres recorridos en una flota y no en tres.** Tres procesos consultando el catálogo por
su cuenta eligen por precio, o sea que eligen **las mismas** ofertas, y el segundo en
alquilar se encuentra la máquina ocupada. Con un pozo único bajo cerrojo eso no puede pasar.
Y hay un límite duro detrás: **medido hoy, el catálogo de Vast devuelve 29 máquinas
distintas** con `--cpu 'E5-26'` (la API corta en 64 ofertas y el filtro se lleva el resto).
15 + criba + repuestos cabe; 65 (una por run) no cabría.

Tres detalles que sostienen la comparabilidad, y por qué:

- **El dataset se extrae UNA vez aquí y viaja hecho.** Las 15 máquinas leen el mismo fichero
  byte a byte. Extraerlo en cada una sería pedir que 15 extracciones coincidan: promesa más
  fuerte, ganancia ninguna.
- **8 hilos de torch en todas**, fijados por `OMP_NUM_THREADS`.
- **Los nombres de los runs los pone el índice global del punto** dentro de su recorrido, así
  que los runs de las 15 máquinas se juntan en un solo `runs/` y forman los tres recorridos
  enteros sin renombrar nada.

### 4.1 La criba: se alquilan 4 de más y se quedan las rápidas

MEDIDO el 2026-08-23 (plan-lr-alto §6.3): entre tres máquinas del mismo catálogo el s/época
fue **36,3 · 50,5 · 53,3** — un factor **1,47** — y la de **más** vCPU (16) fue la **más
lenta**. La conclusión de aquel informe fue literal: *«la elección de oferta filtra hoy por
número de núcleos y precio, que resulta ser un mal criterio para este trabajo»*.

Así que se alquilan **19** y se entrenan **15**. A cada una, nada más instalar, se le piden
unos segundos de entrenamiento **de verdad** —el modelo, la receta, el batch y el
`windows.npz` del recorrido, no un micro-benchmark sintético— con
`scripts/sonda_velocidad.py`. **El criterio, escrito antes:**

1. se descarta toda máquina que tarde por paso **más que la mediana de la cohorte dividida
   por 0,75** — «significativamente más lenta» es esto y no una impresión;
2. de las que sobreviven se quedan las **15 más rápidas**;
3. si sobreviven menos de 15, se completan con las mejores descartadas y **se dice en voz
   alta**: una máquina lenta es peor que ninguna sólo hasta que la alternativa es un punto
   sin medir;
4. **la más rápida al lote más largo**, porque el reloj lo marca la cadena más larga.

Las descartadas se destruyen ahí mismo. Cuestan su peaje (≈3,5 min cada una) y ese gasto se
suma al del estudio en `flota.json` — no es un extra que se olvida.

⚠ **Por qué cribar por velocidad no contamina el resultado, y cuándo sí lo haría.** Se
selecciona sobre una variable —la velocidad— que dentro de la familia E5-26xx v3/v4 **no
mueve la respuesta**: allí el entrenamiento sale idéntico bit a bit (plan-lr-alto §7.4).
Sin `--cpu`, esa garantía no existe y la criba pasaría a elegir sobre algo que sí puede
moverla; el script avisa si se usa `--criba` sin `--cpu`. **Esto es una dependencia real
entre las dos opciones y hay que respetarla.**

⚠ Y hasta dónde llega lo medido: la igualdad bit a bit está comprobada en **tres pares** de
runs, todos E5-2673 v3 / E5-2680 v4. `E5-26` es una subcadena y dejaría pasar una v2 (sin
AVX2), que por el propio razonamiento debería divergir — no comprobado. En el catálogo de
hoy no había ninguna v2, pero eso puede cambiar mañana.

### 4.2 La vigilancia: la velocidad de una máquina alquilada cambia sola

Una máquina rápida a las 10:00 puede tener otro inquilino a las 11:00. El vigilante **no
vuelve a correr la sonda**, y es una decisión: repetirla robaría núcleos justo a lo que se
está midiendo, y falsearía la medida que se quiere proteger. Lee en su lugar los tiempos por
época que el propio entrenamiento ya escribe en `metrics.jsonl`, que no cuestan nada:

```
base     = mediana de las 3 primeras épocas de este run
reciente = mediana de las 3 últimas
degradada  si  reciente / base > 1,35  en DOS sondas seguidas
```

Dos sondas seguidas y no una: una época lenta suelta es ruido normal. Se comparan épocas
**del mismo run** porque cambiar de punto cambia legítimamente el coste (otro batch, otra
profundidad).

Este estudio corre con `--degradado avisar`: se dice y se sigue. Abandonar la máquina
(`--degradado abandonar`) está implementado y es barato gracias a §4.3, pero **cambiaría de
máquina a mitad de un lote**, y con el reparto por bloques eso rompe justo la propiedad que
el reparto compra. Se prefiere un lote lento y homogéneo a uno rápido y partido.

⚠ Una máquina degradada **no** se apunta en la lista negra. Un inquilino ajeno se va; un host
roto no. Bloquear por lentitud vaciaría el catálogo sin arreglar nada.

### 4.2 bis ⚠ El primer lanzamiento se abortó: la API dio el mismo SSH a dos máquinas

**Medido el 2026-08-24 17:12.** La flota arrancó, alquiló 19 máquinas, y en el log apareció
esto:

```
17:12:00  [c7] SSH listo en ssh4.vast.ai:21482 (0.7 min), subiendo 2.5 MB
17:12:00  [c6] SSH listo en ssh4.vast.ai:21482 (0.7 min), subiendo 2.5 MB
```

**El mismo `ssh_host:puerto` para dos instancias distintas** (48581482 y 48581483), publicado
por la API de Vast mientras las dos arrancaban. Las dos hebras subieron el payload a la
**misma** máquina; la instalación de una borró el `payload.tar.gz` de la otra
(`INSTALL` hace `rm -f`), y el fallo se leyó como *«el payload subió pero no se pudo
desempaquetar»*. Se destruyeron las dos máquinas y **se apuntó en la lista negra a dos hosts
que no habían hecho nada** (27568 y 137844, desbloqueados después).

**Lo que se rompió de verdad no es eso.** Ese camino fue ruidoso y se detuvo solo. El
peligro está en la carrera que sale al revés: si las dos instalaciones hubieran terminado,
**dos lotes habrían entrenado en la misma máquina**, compartiendo `/root/bench/runs/`,
peleándose por los núcleos —lo que además corrompe el `s/época` que la §4.2 vigila— y con
otra máquina alquilada sin hacer nada y facturando. Eso no habría dado un error: **habría
dado números.**

Se paró la flota a los dos minutos (nada había entrenado todavía), se destruyó todo
—verificado: *«No hay ninguna instancia viva»*— y costó **0,05 $**.

**La corrección, en dos capas, porque una no basta:**

1. **Registro de destinos.** Ningún destino SSH puede estar reclamado por dos lotes. Si la
   API da uno ya reclamado, se le vuelve a preguntar (el dato se corrige solo en segundos,
   porque describe una instancia que aún está arrancando). Es la defensa barata.
2. **Un sello en la máquina.** Antes de subir nada se escribe un nonce en
   `/root/.duenno-estudio` y se relee; y se **vuelve a comprobar justo antes de arrancar el
   entrenamiento**. Esta es la que cierra el agujero: el registro sólo sabe lo que este
   proceso ha repartido, así que no distingue *«soy el primero»* de *«soy el equivocado»*.
   El sello no se cree a nadie — si dos hebras acaban en la misma máquina, la segunda pisa
   el fichero y la primera lo nota.

**Y el sello resultó ser también la puerta de entrada, lo que obligó a un segundo arreglo.**
MEDIDO en el segundo lanzamiento (2026-08-24 17:16): sin reintentos, **3 de las 5 primeras
máquinas fallaron el sello con `rc=255`** y acabaron en la lista negra sin haber hecho nada.
La causa es la que plan-lr-alto §6.4 dejó apuntada como sospecha sin poder medirla: **el
banner de `sshd` llega antes que la clave.** `esperar_ssh` comprueba el banner, que no es lo
mismo que «SSH funciona»; el sello es el primer comando que necesita autenticarse de verdad,
así que se come esa carrera entera. Aquella nota decía que si volvía a pasar habría que
arreglar *la espera* y no culpar al host — y eso es lo que se hizo.

La asimetría del reintento es deliberada:

- **`rc != 0` es transporte**: la máquina aún no acepta la clave. Se reintenta (8 veces, 15 s)
  y sólo entonces se declara fallo suyo.
- **un sello que se lee y no coincide es una suplantación**: eso no mejora esperando, así que
  falla a la primera.

⚠ **Y una suplantación NO va a la lista negra.** El host no ha hecho nada; se equivocó el
catálogo. Bloquearlo sería castigar al inocente y, peor, ir vaciando un catálogo que ya sólo
tiene 22 máquinas para este filtro. Es la misma regla que ya estaba escrita para la lentitud
sobrevenida, aplicada a un caso nuevo.

**La lección general, indexada por la acción que la dispara y no por su primera víctima:**
*al identificar una máquina alquilada por lo que dice el catálogo, comprobarlo contra la
propia máquina antes de darle trabajo.* El identificador que un proveedor publica sobre un
recurso que todavía está arrancando puede ser de otro, y el síntoma no es un error: es un
resultado.

### 4.3 El libro de a bordo: cada época, en git

Con `--git`, en cada sonda (`--cada 60`, y las épocas duran 40-60 s: en la práctica **una
entrada por época**) el vigilante se trae de cada máquina los ficheros pequeños de sus runs
—`metrics.jsonl`, `status.json`, `config.json`, `summary.json`— y un hilo aparte los
**commitea y los empuja**.

Los pesos (`*.pt`) **no** van a git, y no es un olvido: `.gitignore` lo dice desde siempre
(`/runs/*/*.pt`) porque son ~700 KB por run y por época y el repo se comería gigabytes por
estudio. Lo que va es **el resultado**, que es lo que se lee y lo que rankea.

Qué compra eso, exactamente:

- **Nada de lo terminado se pierde.** Antes, una máquina que se caía en el cuarto de sus
  cinco runs se llevaba también los tres que ya había hecho: los runs sólo bajaban al final,
  en un tar. Ahora bajan según se escriben.
- **Relanzar continúa.** Al arrancar, la flota mira `runs/` y **salta todo punto cuyo
  `status.json` diga `done`**. Un lote sin pendientes ni siquiera alquila máquina. Es lo que
  convierte una caída en un rearranque barato en vez de en repetirlo todo.
- **Sobrevive a esta máquina.** El droplet de control es efímero y lo que no está empujado no
  existe. El libro en el remoto es la única copia que aguanta que se rehaga el servidor.

⚠ **Hasta dónde llega, dicho antes de que haga falta: se reanuda por PUNTO, no por época.**
Un run cortado a mitad se repite entero. Reanudar a media época pediría llevarse los pesos y
el estado de Adam, y además **cambiaría el experimento**: el flujo de números aleatorios del
dataloader no se retoma igual en otra máquina, así que el run reanudado ya no sería bit a bit
el que se pidió — y plan-lr-alto §7.4 midió que esa diferencia importa. Se prefiere repetir
un run a publicar uno que nadie puede reproducir. Lo que el libro garantiza es que **la
unidad que se pierde es un run, nunca un lote**.

⚠ Un `git push` que falla **no** para la flota: se apunta, se sigue y se reintenta en la
vuelta siguiente. Que el estudio se cayera porque la red parpadeó sería cambiar un problema
pequeño por uno grande. `flota.json` guarda cuántos commits hubo y cuántos push fallaron.

## 5. Cómo se lee el resultado (escrito antes)

Se aplican a **los tres recorridos por igual**, y `scripts/estudio_informe.py` los calcula
con las funciones del proyecto (`sweep_trials`, `aggregate_seeds`, `suggest_winner`,
`permutation_test`) en vez de re-implementarlos.

**R1 — validez antes que ranking.** Un punto cuyos runs paran **por el tope de 150** y no por
`patience` mide presupuesto, no calidad. Si el ganador es uno de ésos, **no se declara
ganador**: se reporta `budget-limited`. Es exactamente el defecto que invalida los estudios
de `batch_size` anteriores (§2.1), así que aquí la regla no es un adorno.

**R2 — el ganador.** Media de las semillas del `val_f1` del checkpoint, δ = 1-SE de las
semillas del mejor punto, exactamente `suggest_winner`. Sin regla nueva.

**R3 — ¿queda acotado por los dos lados?** El óptimo está acotado si y sólo si el ganador
**no es un extremo del rango**. Si gana un extremo, la respuesta es *sigue sin acotar por ese
lado*, y **eso se publica como tal** — no es excusa para lanzar otro recorrido en silencio.

**R4 — ¿alguno le gana al vigente?** El ganador contra el vigente (`batch_size` 85,
`n_layers` 4, `d` 2) **medido en este mismo recorrido**, con permutación exacta de las
semillas. Con 5 contra 5 hay 252 arreglos, así que el p mínimo alcanzable es **1/126 ≈
0,0079**: con este tamaño R4 **sí** puede declarar significación al 5 %. Se dice ahora,
antes de mirar, porque es lo que distingue este estudio del de 3 semillas de agosto.

**Y la regla de qué se mueve, escrita antes: el vigente sólo cambia si (a) otro valor le gana
con p < 0,05 por permutación exacta Y (b) la diferencia supera δ.** Si sólo se cumple una,
se reporta y **el vigente se queda**.

**R5 — nada se arrastra sin pasar por la métrica de tarea.** `val_f1` es un **proxy**. Si
algún resultado de aquí fuera a arrastrarse como nuevo vigente, antes se mide con
`scripts/proxy_vs_task.py` ([metrica-de-tarea.md](metrica-de-tarea.md) §2 ter). Está medido
que el proxy exagera: en `p40-confirm-n_layers` la ganancia de L4 sobre L2 pasó de +0,0488 en
ventana a **+0,0224** en tarea, y las bandas dejaron de ser disjuntas.

**R6 — el eje sale plano y hay que poder decirlo.** Si ningún punto se separa del vigente más
que δ, el veredicto es **«eje plano, el vigente se queda»**, y ése es un resultado, no un
fallo. Es lo que pasó con `d` en `proxy-c-d` y lo que aquel estudio no llegó a escribir
claramente.

## 6. Coste y tiempo estimados ANTES de alquilar

Calculado por `scripts/estudio_estimar.py`, cuyos coeficientes están **todos medidos** y con
su procedencia escrita al lado (el único estimado, marcado como tal, es cuánto alargan las
épocas al subir el batch). Se imprime una **franja**, no un número: los extremos salen del
s/época mejor y peor medidos en Vast (36,3 y 53,3).

### 6.1 Los números, calculados el 2026-08-24 antes de alquilar

```
MAQUINAS: 15   (una por recorrido x semilla)
optimista: RELOJ  2,6 h  ·  maquina-horas  32,8  ·  2,17 $
central  : RELOJ  2,9 h  ·  maquina-horas  36,0  ·  2,37 $
pesimista: RELOJ  3,8 h  ·  maquina-horas  47,2  ·  3,11 $
(peaje incluido: 8,4 min x 15 maquinas = 126 min. Recargo por catalogo: x1,10)
```

Por recorrido, la máquina más cargada (la que marca el reloj de su eje):

| recorrido | runs | máquinas | la más cargada |
|---|---|---|---|
| `bs5-L4` | 25 | 5 | **~164 min** ← marca el reloj de todo el estudio |
| `d5-L4` | 20 | 5 | ~126 min |
| `nl5-L4` | 20 | 5 | ~115 min |

**El reloj lo marca `batch_size`**, y dentro de él el punto `batch_size=38` (~58 s/época
estimados, contra 32 de `batch_size=192`). Es el precio de probar el lado barato-en-calidad
y caro-en-tiempo del eje, y se acepta a sabiendas: son ~40 min de reloj sobre los otros dos.

⚠ **Dónde puede fallar esta predicción, dicho antes:** el coeficiente que menos apoyo tiene es
cuántas épocas alarga un batch grande (§5 de `estudio_estimar.py`, el único **estimado**). Si
`batch_size=192` necesitara 90 épocas en vez de las 58 previstas, ese punto pasaría de 31 a
~48 min y el reloj del estudio subiría ~15 min. No cambia ninguna decisión, pero si ocurre se
apunta en §7 y se corrige el coeficiente **en el script**, no en la interpretación.

⚠ El modelo **no** diferencia coste por `d`: `proxy-c-d` midió el s/época plano en todo el eje
(7,0 a 8,8 s con L2). Si resulta que sobre L4 no lo es, se apunta.

**Comparación con lo que costaría en serie**, para tener la referencia: 65 runs a ~35 min de
media son ~38 h en una sola máquina. El reparto los deja en ~2,9 h. Es el mismo orden de
mejora que midió plan-lr-alto §6.3 (36,9 h → 2 h 19 min).

**Coste, que es lo único irreversible aquí.** Las máquinas facturan por segundo mientras
existan. Se destruyen en un `finally` que no es opcional. Si algo se corta a mitad:

```bash
python3 ~/src/digital-ocean-dropplet-auto-launching/scripts/vast_instance.py list
python3 ~/src/digital-ocean-dropplet-auto-launching/scripts/vast_instance.py destroy <id> --yes
```

…y desde Telegram, que es desde donde se opera cuando no hay portátil delante: **`/use
apagar-vast`**.

## 7. RESULTADO

*(Se rellena al cerrar, con los mismos apartados R1..R6 y la comparación del ancla de §3.)*
