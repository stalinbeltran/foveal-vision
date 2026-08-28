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

### 8.1 Tanteo `do-t`

*pendiente*

### 8.2 Estudio completo `do-v`

*pendiente*

### 8.3 Veredicto

*pendiente*
