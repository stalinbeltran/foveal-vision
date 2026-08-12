# Optimizar la CNN plana — el criterio, escrito antes de medir

**Estado: el criterio y el ejecutor. Ni un run entrenado.**
Encadenado detrás del recorrido `p40-lr-L4`: **no arranca hasta que ese cierre**, y corre
**sin supervisión** (decisión del usuario, 2026-08-09: *«no voy a estar presente»*).

Depende de [plan-cnn-plana.md](plan-cnn-plana.md), que construyó el control (`regions: single`).
Aquí **no se compara nada todavía**: se lleva la plana a **su** óptimo, porque comparar una red
tuneada (la foveada: `n_layers=4` de un plan de 40 h, `lr` del recorrido de ahora) contra una que
nunca se ajustó **regala el resultado**, y sería la primera objeción a cualquier conclusión.

---

## 1. La base

La CNN plana de control **C** de plan-cnn-plana.md §3: una rama sobre la ventana, sin contexto.

| | |
|---|---|
| dataset (B) | `dirty1000-80px-16px` — el mismo del recorrido foveado, ventana 16 px |
| receta (D) | `plan40` (`patience=10`, `monitor=val_loss`, `batch_size=85`) |
| red (C) | `regions: single`, `c_frac=1,0`, `d=1` → `N=16`, `periph_out=0`, base `ws16-p0-d1-L4` |
| tope | **150 épocas**, alto a propósito: tiene que parar `patience`, no el tope |
| objetivo | `f1` de ventana (el proxy; la métrica de tarea es informe, nunca criterio — F11) |

## 2. Los dos estudios

**Cribado con 1 semilla, confirmación con 5** — el patrón del plan de 40 h.

### `plana-screen` (1 semilla)
| eje | rango | por qué |
|---|---|---|
| `n_layers` | **2 · 3 · 4 · 5 · 6 · 8** | En la plana `n_layers` sigue siendo **campo receptivo**, no capacidad. Con kernel 3×3 el campo tras L capas es `1+2L`: **L=8 es la primera que cubre los 16 px** de la ventana. Por eso el rango llega más lejos que el de la foveada ([2..5]), donde el anillo ya aportaba contexto |
| `lr` | **0,00035 · 0,0009 · 0,0014 · 0,0028 · 0,005** | Cubre el rango de la foveada **y lo extiende hacia arriba**: la plana tiene ~1/3 de los parámetros, y una red más pequeña suele tolerar un `lr` mayor. El de la foveada quedó pegado al borde izquierdo sin acotar; aquí se acota por los dos lados o se dice |

El estudio **arrastra el ganador** de `n_layers` al paso de `lr` (es lo que hace un estudio OAT).

### `plana-confirm` (5 semillas)
Los mismos dos ejes, con el rango **estrechado al ganador del cribado y sus dos vecinos**. Se
confirma con banda de dispersión y `n_seeds=5`.

## 3. El criterio, antes de mirar

1. **⚠ El cribado de 1 semilla no decide nada, solo descarta.** Medido el 2026-08-08: por métrica
   de tarea el cribado del plan de 40 h **no habría visto nada** (+0,0009 contra ±0,023) — que
   funcionara fue **suerte del proxy**. Aquí se usa igual, porque descartar barato sigue valiendo,
   pero **ningún número del cribado entra en una conclusión**: solo elige qué confirmar.
2. **Nada se afirma sin las 5 semillas** y su banda (protocolo.md).
3. **El empate lo declara la regla**, no el ojo: δ = 1-SE de las semillas del mejor punto
   (`tie_delta`). Si el rango entero empata, **eso es el resultado**, como pasó con
   `fast-lr-2-s0-lr`.
4. **Si gana un extremo del rango, se publica «sigue sin acotar»** — la regla R3 de
   [plan-lr-L4.md](plan-lr-L4.md), que existe justamente porque el `lr` foveado quedó en el borde.
5. **Ninguna comparación con la foveada sale de aquí.** Esto produce «la plana en su óptimo», y
   nada más. La comparación es el paso siguiente y **la decide el usuario** con estos datos en la
   mano.

## 4. Presupuesto y guardas (§2.1)

Coste **estimado, no medido**: la plana es ~1/3 de la foveada (una rama, 16×16 contra dos y 20×20),
y la foveada va a 103 s/época → **~35 s/época**. Con ~66 épocas hasta `patience`:

| | runs | estimado |
|---|---|---|
| `plana-screen` | 6 + 5 = **11** | ~7 h |
| `plana-confirm` | (3 + 3) × 5 = **30** | ~19 h |
| **total** | **41** | **~26 h** |

⚠ Es una estimación aritmética. **El ejecutor mide el coste real en el primer punto y rehace la
proyección**; si la proyección supera **40 h** (el umbral de §2.1), **para y lo dice** en vez de
quemar días en silencio. La máquina se apaga sola y hace throttling térmico, así que el calendario
será bastante más largo que el cómputo.

## 5. Cómo corre sin supervisión

- **`scripts/plan_plana.py`** es el ejecutor. **Reanudable**: salta lo hecho y sigue.
- **La guarda**: si `sweeps/p40-lr-L4/state.json` no dice `done`, **no entrena nada** y sale. Se
  lee el estado del propio recorrido, no un fichero aparte.
- **`fv-plana-watchdog`** (tarea de usuario, cada 10 min) lo relanza si no está corriendo — el
  mismo patrón, con la misma sonda de permisos, que `fv-lrL4-watchdog`. Cubre las dos formas en que
  esto ya murió antes: la sesión que cierra y se lleva al hijo, y el corte de luz.
- **Al terminar**: `Unregister-ScheduledTask -TaskName "fv-plana-watchdog" -Confirm:$false`.

---

## 6. RESULTADO (2026-08-11 21:04) — **el número que sale es un artefacto de promediar fallos**

La cadena corrió **sola y entera**: `p40-lr-L4` cerró el 2026-08-10 a las 22:12 y el watchdog
arrancó el cribado **23 minutos después**, a las 22:35. **41 runs, 22,5 h** de cómputo (estimadas
26). La proyección del presupuesto llegó a tocar **37,3 h** contra el techo de 40 — no saltó la
guarda, pero por poco.

**Respuesta nominal: `n_layers = 4`, `lr = 0,0009`. Y hay que leerla con mucho cuidado.**

### 6.1 La profundidad ≥5 no es peor: NO ARRANCA

| `n_layers` | media de las 5 | colapsan | **media de las que arrancan** | tarea (las que arrancan) |
|---|---|---|---|---|
| **4** | 0,8491 | **0/5** | 0,8491 ± 0,0029 (n=5) | **0,7755** ± 0,0076 |
| 5 | 0,6890 | **1/5** | **0,8612** ± 0,0017 (n=4) | 0,7520 ± 0,0083 |
| 6 | 0,5164 | **2/5** | **0,8606** ± 0,0026 (n=3) | 0,7572 ± 0,0163 |

Las semillas que fallan dan **f1 exactamente 0,0000**: no entrenan peor, **no despegan**. La media
de una mezcla bimodal no mide calidad — mide *probabilidad de arrancar × calidad de las que
arrancan*. Por eso `suggest_winner` corona a L4: **gana por fiabilidad, no por calidad**.

Es **la misma firma** que plan-40h.md §3 documentó para la foveada L5 («o arranca o no arranca,
sin valores intermedios»). Que aparezca **también en la red plana** dice que no es de la
arquitectura foveada: es de **inicialización/optimización a profundidad ≥5** con esta cabeza. Y
refuerza lo que aquel plan ya concluyó: para pasar de 4 capas hay que cambiar **algo más que el
número** (residuales, otra inicialización).

⚠ En el cribado, `n_layers` **6 y 8 dieron 0,0000 con su única semilla**. Con 1 semilla este modo
de fallo es indistinguible de «es malo». Que la cadena acabara mirando [4,5,6] fue **suerte**: si
la semilla de L5 hubiera colapsado, el cribado habría coronado L4 y nadie habría visto nada.

### 6.2 Y las dos métricas se contradicen EN EL SIGNO

Sobre **solo las semillas que arrancan**:

- **por ventana (el proxy que la cadena optimizó)**: L5 y L6 **ganan** a L4 por ~0,012, con `sem`
  de 0,002–0,003.
- **por tarea (la métrica que manda)**: **L4 gana** a L5 por **+0,0236** y a L6 por **+0,0184**.

**Ninguna diferencia de tarea cruza el umbral**: L4 vs L5 da **p = 0,079** (126 arreglos) y L4 vs
L6, **p = 0,321**. Así que **no se afirma que L4 sea mejor por tarea** — lo que se publica es que
**el proxy y la tarea ordenan al revés** en este eje, que es exactamente la reserva anotada el
2026-08-08 («reserva del proxy en ejes de profundidad»), ahora con un segundo caso y más nítido.

⚠ Con los colapsos **dentro**, `proxy_vs_task.py` da Spearman agregado **+1,000** y «mismo
ganador»: la concordancia perfecta es un **artefacto de los ceros compartidos**. Quitarlos la
invierte. Una correlación calculada sobre una mezcla bimodal no dice nada del proxy.

### 6.3 `lr`: plano otra vez

Con L4 fijo, `0,0009` (0,8499 ± 0,0020) y `0,0014` (0,8491 ± 0,0029) son **empate técnico**
(δ = 0,0020, la regla lo declara sola). `0,0028` ya colapsa 1 de 5. Mismo dibujo que la foveada
(§7.1 de [plan-lr-L4.md](plan-lr-L4.md)): **una meseta ancha que se rompe hacia arriba**.

### 6.4 ⚠ Un bug del ejecutor: `BASE_NETWORK` pisaba al ganador arrastrado

`derive_base` aplica los `overrides` **después** de los `winners`, así que un campo fijado en
`base_network` **anula el arrastre del estudio, en silencio**. `plan_plana.py` fijaba
`n_layers: 4` «como punto de partida» → el paso de `lr` del cribado se midió a **L4** aunque el
estudio había coronado **L5** (`winners` lo dice; `base_network_value` del recorrido dice 4).

**No cambió la respuesta final** —la confirmación coronó L4, que es justo lo que estaba clavado—
pero fue **suerte**. Arreglado: `BASE_NETWORK` ya no fija `n_layers`, y el ejecutor **se niega a
arrancar** si fija un campo que además es eje. El rastro existía (`field_origin` decía `user` en
vez de `winner`); lo que faltaba era que alguien lo mirara.

### 6.5 Qué NO dice esto

**Nada sobre la foveada.** No se ha comparado. Para eso está la familia de 6 controles de
[plan-cnn-plana.md](plan-cnn-plana.md) §3, y **la decide el usuario**. Lo que este trabajo deja
listo es la plana en un óptimo **defendible y con sus reservas escritas**: L4, `lr` en la meseta,
y el aviso de que por encima de 4 capas el problema es de arranque, no de capacidad.
