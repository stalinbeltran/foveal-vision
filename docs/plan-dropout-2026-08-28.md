# Plan: `dropout` — ¿regulariza algo, y con qué valor? (2026-08-28)

> **Estado: criterio escrito ANTES de mirar ningún número.** Es el requisito del protocolo de
> este proyecto: quien decide qué cuenta como «gana» tiene que hacerlo sin ver el resultado, o
> el rango y el umbral acaban ajustándose a lo que salió. Los números se pegan abajo cuando
> lleguen; **este documento no se reescribe para que cuadre**.

## 0. El encargo, en una línea

*Verificar si aplicar `dropout` incrementa la capacidad predictiva de la red foveada, y
determinar el valor más apropiado.* Primero un **tanteo** que acota el rango; después el
**estudio completo** de 5 semillas sobre el rango que el tanteo señale.

---

## 1. Qué es y por qué se mira ahora

`dropout` apaga al azar una fracción de las activaciones **en cada paso de entrenamiento**, y
está desactivado en evaluación. Es el mando de **regularización dentro de la red** (C), el
hermano de `weight_decay`, que regulariza desde la receta (D).

En esta red **va sobre las features aplanadas, justo antes de la cabeza**
([`builder.py`](../src/fv/models/builder.py) L124-L134), que es donde vive el **97 % de los
parámetros**. No está entre capas conv a propósito: allí hay ~3 % de los parámetros y apagar
activaciones enteras de un mapa espacial pequeño sobre todo añadiría ruido a una cabeza que
tiene que decir **dónde** está una esquina.

**El motivo medido**, del [inventario de parámetros](../reportes/2026/08-agosto/parametros-y-prioridad-de-estudios.md) §1:
sobre los **612 runs con curvas en disco**, la brecha `val_loss` contra `train_loss` en la época
del checkpoint es de **+28 % mediana**, y **390 de los 612** pasan del 20 %. Además, entre el
mejor punto y la parada por `patience`, la `val_loss` **vuelve a subir** (mediana +0,0026)
mientras la de train sigue bajando. Es la firma de un **sobreajuste leve pero sistemático**, y
es exactamente lo que ataca la regularización.

### ⚠ La evidencia EN CONTRA, que hay que leer antes de gastar

El plan de prioridades puso `weight_decay` (**10 ter**) **antes** que `dropout` (**10 quater**)
con esta frase: *«si `weight_decay` no mueve nada, implementar `dropout` es mucho menos
prometedor»*. **`weight_decay` ya se midió** (barrido
[#14](https://github.com/stalinbeltran/telegram-coordinator/blob/main/reportes/README.md),
recorridos `wd-t` + `wd-v`, 5 semillas) y el resultado fue el peor posible para esta hipótesis:

> **`weight_decay` = 0 GANA y subirlo HUNDE** (0,001 → 0,8731 contra ~0,93 del vigente).

O sea: **la puerta barata de la regularización ya se probó y se cerró.** Este estudio se lanza
sabiéndolo.

### Por qué se lanza igualmente: dos razones, y no son «por si acaso»

1. **No es el mismo mecanismo, ni actúa en el mismo sitio.** `weight_decay` penaliza la norma de
   **todos** los parámetros por igual. `dropout` aquí actúa **sólo sobre el vector de features
   que lee la cabeza**, que es donde está el 97 % de la capacidad. Un resultado sobre el
   primero no predice el segundo.

2. **Y hay una razón concreta para desconfiar de aquel resultado como prueba de que «la
   regularización no sirve aquí»:** la receta vigente es `optimizer: adam`, y
   [`loop.py`](../src/fv/training/loop.py) L39 pasa `weight_decay` directamente a
   `torch.optim.Adam`. Eso es **L2 acoplado al escalado adaptativo**, no *weight decay*
   desacoplado — es justo la forma que se sabe que se porta mal con Adam (es la razón de existir
   de AdamW). Que **esa** implementación hunda el f1 es compatible con «la regularización no
   ayuda» **y también** con «esa forma de regularizar está mal condicionada». Las dos hipótesis
   siguen vivas, y `dropout` las separa.

   ⚠ **Lo que esto NO dice:** que `dropout` vaya a ganar. `patience` ya está recogiendo casi todo
   el daño del sobreajuste (para cerca del mejor punto), así que el margen que queda es el hueco
   entre *«paro a tiempo»* y *«generalizo mejor»*, y puede ser pequeño o nulo. **Un resultado
   negativo aquí es un resultado**: cierra el último eje de regularización del inventario.

## 2. Lo que ya está comprobado, y no hay que volver a comprobar

| Qué | Cómo se sabe |
|---|---|
| `dropout` **existe en el código** y es eje de C | `NETWORK_DEFAULTS["dropout"] = 0.0` (medido hoy). Antes del 2026-08-27 estaba en tres documentos y en ningún dict: un eje habría entrenado N veces la misma red **sin avisar** |
| **`0.0` es identidad bit a bit** | `test_dropout_off_is_the_net_that_was_already_on_disk` — los checkpoints previos cargan `strict` y la salida no se mueve |
| **Actúa en train y NUNCA en eval** | `test_dropout_on_acts_in_train_and_never_in_eval` |
| **Fuera de `[0, 1)` se rechaza en la puerta** | `test_dropout_out_of_range_is_refused_at_the_gate` (`dropout_out_of_range`) |
| **Es eje barrible de verdad** | `test_dropout_is_a_sweepable_axis_of_c` + `verify_axes.py` lo entrena end-to-end |
| **Es COST-NEUTRAL** | **Medido el 2026-08-28**: `167.852` parámetros en `{0,0 · 0,05 · 0,1 · 0,2 · 0,3 · 0,5}`, idéntico al vigente. `nn.Dropout` no tiene parámetros, y el s/época tampoco debería moverse. **No hay coste que sopesar contra la ganancia** — a diferencia de `border_reduce` |

Los cuatro tests pasan hoy en esta máquina (`pytest -k dropout` → 4 passed).

## 3. El montaje

| Qué | Valor | Por qué |
|---|---|---|
| **Red base** | **la vigente**, `ws16-p2-d2-L4` (`border_px` 4, `border_reduce` 2, `overlap_fovea_px` 2, `n_layers` 4, `channels` [16]×4, `merge` concat) · 167.852 parámetros | Es la base de **todos** los estudios de la tabla, y la disciplina OAT de este proyecto es mover **un** eje sobre una base fija. ⚠ Ver la reserva de §6 |
| **Receta** | `plan40`, `epochs` 150 | La misma con la que se midieron los demás ejes |
| **Dataset** | **`dirty1000-80px-16px-r20260827`** | Es **el único dataset de estudio con `windows.npz` en git** hoy. ⚠ Ver §6: es un dataset **nuevo**, distinto del `r20260826` de #14 |
| **Objetivo** | `f1` de ventana (proxy) | El de siempre. ⚠ Ningún eje ha pasado por la métrica de tarea (R5), y éste tampoco |
| **CPU** | `--cpu E5-26` | La familia donde está medido que el entrenamiento sale **idéntico bit a bit** entre máquinas — es lo que hace legítima la criba por velocidad |

## 4. Fase 1 — el TANTEO (`do-t`)

**2 semillas. ACOTA, no declara.** Con 2 contra 2 el `p` mínimo alcanzable es **0,333**: este
recorrido no puede mover un vigente ni aunque quiera, y no se le va a pedir que lo haga.

| recorrido | eje | rango | runs | por qué ESE rango |
|---|---|---|---:|---|
| **`do-t`** | `dropout` | **{0,0 · 0,1 · 0,25 · 0,5}** | **8** | `0,0` es el **ancla**: es el vigente, y su media sobre este dataset dice además cuánto movió el dato respecto de los estudios anteriores. `0,1` y `0,25` son los dos valores que el inventario propuso como «10 quater». `0,5` es el default clásico de la literatura de dropout **y acota el eje por la derecha**: si todo baja monótono desde 0,0, se sabe con 8 runs |

**Coste estimado: ≈0,43 $** (8 runs a los 0,054 $/run medidos el 25-ago).

## 5. Fase 2 — el ESTUDIO COMPLETO (`do-v`), y su rango decidido AHORA

**El rango de la fase 2 no se elige después de mirar: se elige aquí, con una tabla que cubre
todos los resultados posibles del tanteo.** Ésa es la única forma de que «el rango lo dijo el
tanteo» signifique algo; si se decidiera al ver los números, el rango estaría ajustado al
resultado y el estudio no probaría nada que no supiera ya.

Sea **`p*`** el punto con mejor f1 medio en `do-t`:

| Si el tanteo deja `p*` = | el estudio completo barre | por qué |
|---|---|---|
| **0,0** *(dropout sólo hace daño)* | **{0,0 · 0,05 · 0,1 · 0,2}** | El paso más pequeño del tanteo es 0,1; que 0,1 ya haga daño **no descarta** una ganancia en 0,05. Cerrar «no ayuda» exige haber mirado ahí, y es donde un dropout ligero sobre 97 % de los parámetros tendría su oportunidad |
| **0,1** | **{0,0 · 0,05 · 0,1 · 0,2}** | Encierra al ganador por los dos lados: hay un punto por debajo y otro por encima |
| **0,25** | **{0,0 · 0,1 · 0,25 · 0,4}** | Ídem, encierra el 0,25 |
| **0,5** *(gana el extremo)* | **{0,0 · 0,25 · 0,5 · 0,7}** | Se **extiende más allá del extremo**, como hizo `borde-ancho`: un ganador en el borde del rango no está acotado, y hay que ir a buscar dónde baja. `0,7` es legal (`[0, 1)`) |

**5 semillas por punto, 4 puntos = 20 runs. Coste estimado: ≈1,08 $.** Con 5 contra 5 el `p`
mínimo alcanzable es **0,0079**, o sea que **R4 sí puede declarar significación al 5 %**.

⚠ **Las 5 semillas se corren enteras en `do-v`, sin reaprovechar las 2 de `do-t`.** Sumarlas
sería legítimo (mismo dato, misma red, misma receta — es lo que hizo el bloque D de #14), pero
`estudio_informe.py` trabaja **sobre un recorrido**, así que la tabla de 5 semillas habría que
componerla a mano. Este proyecto ya tiene escrito lo que pasa con los números que se calculan en
dos sitios: acaban divergiendo. La redundancia cuesta **≈0,43 $** y compra un veredicto que sale
entero de una herramienta.

### Cómo se lee el resultado — las reglas, aplicadas por `estudio_informe.py`

- **R1 (validez).** Un punto que para **por el tope de épocas** mide presupuesto, no calidad. Si
  hay truncados, no se declara ganador. *(Medido sobre 630 runs: la época más alta observada es
  130 y ninguno llegó a 150, así que no debería atar — pero se comprueba.)*
- **R2 (ganador por media).** Con su banda δ y `suggest_winner`, que dentro de δ se queda con la
  **más barata**.
- **R4 (contra el vigente `0.0`).** Permutación exacta, 5 contra 5.
- **Y el vigente sólo se mueve si `p` < 0,05 **y** la diferencia supera δ.** Un ganador nominal
  con `p` = 0,063 no mueve nada — es la regla que ya dejó `border_px` = 8 sin aplicar durante dos
  estudios.

### R5-bis: el mecanismo, que es gratis y dice más que el f1

`dropout` no se lanza para subir el f1 y ya: se lanza contra una **brecha val/train medida de
+28 %**. Así que además del ranking se mide, **sobre los `metrics.jsonl` que los runs ya
escriben y sin alquilar nada**, la brecha val/train por valor del eje. Hay tres desenlaces y los
tres son informativos:

| brecha | f1 | qué significa |
|---|---|---|
| **baja** al subir el dropout | **sube** | funciona por el motivo que se creía |
| **baja** | **no sube** o baja | el sobreajuste **no era el cuello de botella**: `patience` ya lo estaba recogiendo. Cierra la hipótesis de regularización entera, y con una razón, no con un encogimiento de hombros |
| **no baja** | — | el dropout **no está regularizando** donde se puso, y el sitio (antes de la cabeza) es la variable a revisar, no la idea |

## 6. Las tres reservas, escritas antes y no después

1. ⚠ **Este estudio NO es comparable, punto por punto, con los números de #14.** Aquél corrió
   sobre `dirty1000-80px-16px-r20260826`, que **se perdió y no vuelve** (reconstruirlo da otro
   `.npz`, comprobado tres veces). Éste corre sobre `r20260827`, renderizado el 2026-08-27 y
   **commiteado en git**. Cada estudio lleva su propia ancla `0,0`, así que **el contraste
   interno es válido**; lo que no se puede es restar el 0,9302 de allí del 0,93x de aquí.
2. ⚠ **La base es la vigente, no la mejor conocida.** #14 dejó `border_px` 4→8 y
   `overlap_fovea_px` 2→7 **medidos y sin aplicar**. Sobre una red con más contexto la brecha
   val/train podría ser otra, y con ella el óptimo de dropout. Se usa la vigente **porque es lo
   que hace este número comparable con los otros 16 barridos**, y cambiar dos cosas a la vez es
   justo lo que OAT existe para no hacer.
3. ⚠ **Es f1 de VENTANA, un proxy** que está medido que **exagera** (en `n_layers` la ganancia
   real fue la mitad). Ningún eje de este proyecto ha pasado por la métrica de tarea, y éste
   tampoco.

## 7. Operación

```bash
cd ~/src/foveal-vision

# 1. Crear los recorridos (no alquila, no gasta)
.venv/bin/python scripts/estudio_dropout.py --dataset dirty1000-80px-16px-r20260827 --fase tanteo

# 2. Qué va a costar (no toca Vast)
.venv/bin/python scripts/estudio_estimar.py --sweep do-t

# 3. La flota (ESTO ES LO QUE FACTURA)
#    ⚠ `desacoplar.sh` vive en el COORDINADOR, no en este repo. Con la ruta
#    relativa que usan otros planes de aquí el comando no corre. Y los secretos
#    NO viajan por `desacoplar.sh` (a propósito: `sudo` los escribiría en claro
#    en el journal), así que los carga el propio comando desde disco — son DOS
#    ficheros, `.env` y `~/.config/dev-secrets.env`, y el token de Vast está en
#    el segundo.
"$COORD_HOME/scripts/desacoplar.sh" sh -c '
set -a
[ -f "$COORD_HOME/.env" ] && . "$COORD_HOME/.env"
[ -f "$HOME/.config/dev-secrets.env" ] && . "$HOME/.config/dev-secrets.env"
set +a
.venv/bin/python scripts/estudio_flota.py --sweep do-t --cpu E5-26 --criba 2 \
    --git --horas-max 6 --prefijo dr- --yes > /tmp/estudio-dropout-tanteo.log 2>&1
node "$COORD_HOME/scripts/notify.mjs" "tanteo de dropout terminado"
' &

# 4. Leerlo
.venv/bin/python scripts/estudio_informe.py --sweep do-t --vigente 0.0

# 5. Fase 2. El rango NO se teclea: el script deriva el pico de los runs de
#    `do-t` con las mismas funciones que el informe, y la tabla de §5 (que vive
#    en TABLA_PICO) dice qué rango le toca. Se niega si el tanteo está a medias.
.venv/bin/python scripts/estudio_dropout.py --dataset dirty1000-80px-16px-r20260827 \
    --fase completo
# (misma envoltura que el paso 3, cambiando do-t por do-v y horas-max a 8)
.venv/bin/python scripts/estudio_informe.py --sweep do-v --vigente 0.0
```

| Qué | Comando | Cuesta |
|---|---|---|
| Cómo va | `/use estudio-progreso` → `--sweep do-t --tabla` | nada |
| Vigilancia máquina a máquina | `/use vigilante-avance` → `--sweep do-t --cada 600` | nada |
| **Freno de emergencia** | `/use apagar-vast` | — |

⚠ **`--prefijo dr-` no es opcional.** Es el espacio de nombres de las instancias en la cuenta, y
es por lo que `vigilante_avance.py` distingue *sus* máquinas de las de otra sesión. Sale del
`WORKSPACE.json` del workspace (creado hoy en `~/src`: `nombre` "src", `prefijo` `dr-`). Sin él,
dos estudios a la vez se creen dueños de las máquinas del otro.

---

## 8. Resultados

*(Se pegan aquí cuando lleguen. El criterio de arriba no se toca.)*

### 8.1 Tanteo `do-t` — TERMINADO

**8/8 runs**, dataset `dirty1000-80px-16px-r20260827`, base `ws16-p2-d2-L4`, receta `plan40`.
Corrido el **2026-08-28 de 01:30:11 a 04:51:22 UTC** (201,2 min de reloj), **5 máquinas
alquiladas** (2 entrenaron; 3 se fueron en fallos de alquiler y criba), **0,3626 $**.

| `dropout` | f1 (media) | sem | min | max | épocas | s/época |
|---:|---:|---:|---:|---:|---:|---:|
| **0,0** | **0,9315** | 0,0010 | 0,9305 | 0,9326 | 47 · 54 | 35,7 |
| 0,25 | 0,9282 | 0,0022 | 0,9260 | 0,9304 | 46 · 59 | 56,9 |
| 0,5 | 0,9274 | 0,0025 | 0,9250 | 0,9299 | 73 · 82 | 63,2 |
| **0,1** | **0,9129** | 0,0020 | 0,9108 | 0,9149 | 34 · 35 | 53,3 |

- **R1 ✅** — recorrido válido: los 8 runs pararon por `patience`, entre 34 y 82 épocas, ninguno
  cerca del tope de 150.
- **R2** — gana **`dropout` = 0,0**, el vigente. δ = 0,0010 (1-SE del mejor punto), así que
  **ningún otro valor entra en la banda**.
- **R4** — los tres contrastes dan `p` = 0,333, que es **el suelo alcanzable con 2 contra 2**
  (6 arreglos). Como estaba escrito: **un tanteo no declara**. Lo que sí hace es acotar, y
  la amplitud es **0,0186**, casi el doble del umbral de 0,010 — o sea que el eje **sí mueve
  la aguja**, sólo que hacia abajo.

⚠ **El 0,1 es el PEOR, por debajo de 0,25 y de 0,5: el eje no es monótono.** Y no parece ruido:
sus dos semillas caen juntas (0,9108 · 0,9149) y son las que **antes paran** (34 y 35 épocas,
contra 73-82 de `0,5`). Ver la nota de §8.4.

### 8.1 bis — R5-bis: el mecanismo. **Esto es lo que este estudio ha medido de verdad**

La brecha `val_loss` contra `train_loss` en la época del checkpoint, run a run:

| `dropout` | seed | épocas | mejor | train_loss | val_loss | **brecha** | val_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,0 | 1 | 54 | 44 | 0,07930 | 0,10611 | **+33,8 %** | 0,9305 |
| 0,0 | 2 | 47 | 37 | 0,08035 | 0,10052 | **+25,1 %** | 0,9326 |
| 0,1 | 1 | 34 | 24 | 0,14799 | 0,14336 | −3,1 % | 0,9108 |
| 0,1 | 2 | 35 | 25 | 0,14123 | 0,13839 | −2,0 % | 0,9149 |
| 0,25 | 1 | 46 | 36 | 0,10762 | 0,10446 | −2,9 % | 0,9260 |
| 0,25 | 2 | 59 | 49 | 0,10521 | 0,09904 | −5,9 % | 0,9304 |
| 0,5 | 1 | 82 | 72 | 0,11534 | 0,09809 | −15,0 % | 0,9299 |
| 0,5 | 2 | 73 | 63 | 0,15090 | 0,12619 | −16,4 % | 0,9250 |

| `dropout` | brecha media | f1 medio |
|---:|---:|---:|
| **0,0** | **+29,5 %** | 0,9315 |
| 0,1 | −2,6 % | 0,9129 |
| 0,25 | −4,4 % | 0,9282 |
| 0,5 | −15,7 % | 0,9274 |

**Dos cosas, y las dos importan:**

1. **La brecha de +28 % que motivó todo esto queda CONFIRMADA de forma independiente.** Aquel
   número salió de 612 runs viejos sobre datasets que ya no existen; aquí, sobre `r20260827` y
   con dos semillas limpias, el `dropout` = 0,0 da **+29,5 %**. La premisa del estudio era buena.

2. **`dropout` cierra esa brecha ENTERA, y ya con 0,1** (+29,5 % → −2,6 %), y con 0,5 la invierte
   (−15,7 %: val mejor que train, que es lo que se espera cuando el ruido sólo actúa en train).
   **O sea que regulariza exactamente como se diseñó.** Y aun así **el f1 no sube: baja en los
   tres valores.**

Ése es el segundo renglón de la tabla que este plan escribió en §5 **antes de mirar**:

> *la brecha baja y el f1 no sube → el sobreajuste **no era el cuello de botella**: `patience`
> ya lo estaba recogiendo. Cierra la hipótesis de regularización entera, y con una razón, no con
> un encogimiento de hombros.*

**Y hay un tercer dato que lo refuerza y que no estaba previsto: el `train_loss` casi se dobla**
(0,0793 → 0,1480 con `dropout` = 0,1). El modelo no está redistribuyendo capacidad: está
**perdiéndola**. Con 167.852 parámetros sobre este dato, la red no sobra — le falta.

### 8.2 Estudio completo `do-v` — CREADO, **SIN LANZAR**

El pico del tanteo es **`0,0`**, y la tabla de §5 (escrita antes de mirar, y que vive en
`TABLA_PICO` dentro de `estudio_dropout.py`) le asigna el rango:

> **`dropout` ∈ {0,0 · 0,05 · 0,1 · 0,2}** — *«el paso más pequeño del tanteo era 0,1; que 0,1 ya
> duela NO descarta una ganancia en 0,05: cerrar "no ayuda" exige haber mirado ahí»*.

El recorrido **está creado y commiteado** (`sweeps/do-v/spec.json`, 4 valores × 5 semillas =
**20 runs**). **No se ha lanzado**: el server se destruye. Estimado **≈1,1 $** y ~3,5 h de reloj
al ritmo real medido en el tanteo (53 s/época, no los 40 estimados).

Qué añade sobre lo que ya se sabe, que no es poco:

- **`0,05`, que nadie ha mirado.** Es el único punto donde podría quedar una ganancia.
- **5 semillas**, que bajan el `p` mínimo alcanzable de 0,333 a **0,0079**: es lo que convierte
  «el vigente gana» en **una declaración al 5 %** en vez de en una impresión.

### 8.3 Veredicto — **PROVISIONAL** (falta `do-v`)

**Con lo medido hasta ahora, `dropout` no mejora la capacidad predictiva de esta red, y el
vigente `0,0` se queda.** Los tres valores probados pierden, y el mejor de ellos (0,25) pierde
0,0033 — dentro de lo que un tanteo no puede declarar, pero sin ningún indicio en la otra
dirección.

Lo importante no es el «no», es **por qué**: el sobreajuste existía y estaba bien medido, y
`dropout` lo elimina por completo — **y el f1 no mejora**. Junto con el resultado de
`weight_decay` en el #14, eso cierra **los dos** mandos de regularización del inventario con la
misma conclusión, y ahora con un mecanismo medido detrás en vez de una conjetura:
**la brecha val/train de esta red no es el cuello de botella; `patience` ya la estaba
recogiendo.**

⚠ **No está cerrado del todo hasta que corra `do-v`**, por el punto `0,05` y por las 5 semillas.

### 8.4 Lo que quedó pendiente, y una pista que vale más que el veredicto

1. **Lanzar `do-v`.** Está creado y commiteado. Un comando.
2. ⚠ **`patience` = 10 NO es neutral a lo largo de este eje, y eso confunde parte de lo medido.**
   Las épocas van de **34-35** (`dropout` 0,1) a **73-82** (`dropout` 0,5): un factor **2,4**.
   `dropout` mete ruido en el entrenamiento, la `val_loss` mejora a tirones, y una `patience` fija
   corta antes. Así que parte de lo que mide este eje es *«cómo le sienta a `patience` = 10 este
   nivel de ruido»*, no sólo *«cuánto regulariza»*. **Es la explicación más plausible de la
   no-monotonía** (0,1 el peor y el que antes para; 0,5 el que más entrena y casi alcanza al
   vigente). Comprobarlo es un estudio propio: `dropout` × `patience`, o `dropout` con `patience`
   escalada. **No se ha hecho, y sin él el «no» de arriba tiene esa reserva.**
3. **Sigue sin pasar por la métrica de tarea (R5)**, como todos los ejes del proyecto.
4. **El s/época NO es comparable entre valores de este recorrido.** Va de 35,7 a 63,2, pero los
   cuatro valores se corrieron **en orden en la misma máquina**, así que está confundido con el
   momento del alquiler (las máquinas de Vast se frenan cuando entra otro inquilino: el log
   registra la de `s1` pasando de ~36 a 53,9 s/época). **`dropout` es cost-neutral en parámetros
   —167.852 en todo el rango, medido— y no hay razón para que cueste más.** Si el coste importa,
   se mide aparte.
