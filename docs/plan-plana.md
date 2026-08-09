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
