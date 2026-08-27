# Validación en Vast del barrido de stride — 2026-08-27

**Qué se corrió**: no el estudio, sino la **cadena entera del estudio** sobre máquinas de verdad,
antes de gastar en las 25 que pide `docs/plan-stride-2026-08-27.md`. Recorridos `stride-h01` y
`stride-h16` (modo `--humo`: nombres propios, estudio aparte, 1 semilla, 3 épocas), sobre los
mismos `window-datasets` que usará el estudio.

| | |
|---|---|
| **Inicio (UTC)** | 2026-08-27 **02:29:43** (corrida 1) |
| **Fin (UTC)** | 2026-08-27 **02:42:39** (c1) · **02:58:49** (c2) · c3 en curso al escribir |
| **Instancias ALQUILADAS** | **7** (4 en c1 + 3 en c2), de las que **1** entrenó hasta el final |
| **Coste real** | **0,0300 $** (0,0180 + 0,0120) |
| **Reloj** | 12,8 min (c1) + 14,3 min (c2) |
| Recorridos | `stride-h01`, `stride-h16` · estudio `stride-2026-08-27-humo` |
| Prefijo de etiqueta | `st-` (a propósito: ver §3) |
| Logs | `/tmp/stride-humo.log`, `/tmp/stride-humo2.log`, `/tmp/stride-humo3.log` — **`/tmp` no sobrevive a rehacer la máquina**; lo que importa está aquí |
| Informe de flota | `sweeps/stride-h01/flota.json`, `sweeps/stride-h16/flota.json` |

⚠ **«Instancias» son las alquiladas, no las que trabajaron.** 7 alquiladas, 1 entrenó. Las otras 6
también facturaron, y por eso están en la cuenta de arriba.

---

## 1. Qué quedó demostrado

El brazo `stride-h16` completó sus 3 épocas en una máquina de Vast
(`Xeon E5-2680 v4`, 9,3 vCPU, 0,0502 $/h) y con eso quedan comprobadas, **en producción y no en
local**, las cinco cosas que el mecanismo prometía:

1. **El payload lleva VARIOS datasets.** 6,7 MB con `…-st01` y `…-st16` dentro, subido a la
   máquina y desempaquetado. Antes la flota moría con «lanza una flota por dataset».
2. **Cada recorrido entrena con el SUYO.** El `config.json` que la máquina devolvió declara
   `window_dataset: dirty1000-80px-16px-st16` con su huella
   `sha256:5b7a7737…`, no la del otro brazo.
3. **El presupuesto igualado viaja.** El mismo `config.json` trae
   `windows_per_epoch: 84000`, así que la máquina entrenó 84.000 ventanas por época sobre un
   pool de 12.000 — 7 pasadas — y no las 12.000 del pool.
4. **La rejilla de evaluación fija funciona.** 28.000 ventanas de val, idénticas a las que tendrá
   cualquier otro brazo.
5. **El run es un run normal.** f1 **0,7054 → 0,8093 → 0,8724**, `loss` 0,3102 → 0,1872,
   `pos_err_px` 2,316 → 1,909, a 39-40 s/época. La máquina se destruyó sola al terminar el lote
   (vivió 4,4 min, 0,0037 $) y el libro de a bordo commiteó y **empujó** desde el proceso
   desacoplado.

Y la ruta de **un solo dataset** también quedó ejercitada en producción (corridas 2 y 3): el log
dice «1 recorrido(s) sobre dirty1000-80px-16px-st01», no «N datasets». Era la regresión que había
que no cometer, porque este script gasta dinero.

## 2. Lo que la validación encontró (y que en local no se veía)

| # | Qué | Cómo se vio |
|---|---|---|
| 1 | **La sonda proyectaba s/época desde el POOL, no desde la época.** Anunció «~392 s/época» para un brazo que midió 39,4. Con `windows_per_epoch` una época ya no es el pool entero: 20,9× de más. La **criba no se equivocaba** —ordena por ms/paso— pero el número del log sí | el log de c1 contra el `summary.json` del run |
| 2 | **`flota_viva()` preguntaba por CUALQUIER flota.** Con la flota del otro estudio viva en esta misma máquina, el vigilante de éste no habría relanzado nunca: un estudio que parece vigilado y no avanza | leyendo el código con las dos flotas delante |
| 3 | **El informe se contradecía con un brazo.** Con una sola medida, `min == max` y R1 decía a la vez «la densidad no compra nada» y «el eje no queda cerrado por arriba». Y es el caso NORMAL: es lo que sale al mirar mientras la flota corre | corriendo `estudio_stride_informe.py` sobre los datos reales |
| 4 | **El informe moría después de darlo.** `relative_to` lanza si el `--json` cae fuera del repo, y estaba en la última línea: informe entero y código de salida 1 | lo encontró su propio test |

Las cuatro van arregladas, con test, en la rama.

## 3. Dos estudios en una cuenta: por qué el prefijo

Al preparar esto había **8 máquinas vivas de otro estudio** (`estudio-c3` … `estudio-c19`,
0,5159 $/h) con su `vigilante_avance.py` corriendo. Eso destapó que la etiqueta `estudio-` estaba
cableada y que **la rama de «sobrantes» del vigilante destruye toda instancia con el prefijo**
cuando sus recorridos terminan — incluidas las que `juzgar()` acababa de declarar ajenas.

O sea: al acabar el otro estudio, su vigilante habría matado estas máquinas. El síntoma habría
sido de los peores —runs cortados a media época, sin error propio, indistinguibles de una máquina
que se muere sola—, así que la validación se lanzó con `--prefijo st-`. Las máquinas nacieron
`st-stride-h01-s1` y `st-stride-h16-s1`, fuera de `estudio-*`. Comprobado en el listado de la
cuenta.

## 4. Lo que quedó pendiente

**El brazo de stride 1 NO se ha validado en una máquina.** Es el que importa comprobar (1.755.000
ventanas de train, el único con riesgo de memoria), y falló **dos veces seguidas por
indisponibilidad de Vast, no por el código**:

```
02:49:05  no utilizable (intento 1/4): SSH no llego a aceptar la clave en ssh2.vast.ai:12174
          tras 12 intentos (4 min): rc=255
02:54:03  no utilizable (intento 2/4): … ssh9.vast.ai:12538 …
02:58:49  no utilizable (intento 3/4): … ssh5.vast.ai:12890 …
02:58:49  FALLO (intento 4/4): ApiError HTTP 400 … no_such_ask (la oferta se la llevó otro)
```

Seis máquinas alquiladas y ninguna llegó a aceptar la clave. Con `--cpu E5-26` el catálogo se
quedaba en 12 ofertas (9-12 más saltadas por lista negra o por familia), así que el pozo se agotaba.

**La tercera corrida (10:30 UTC) se lanzó sin `--cpu`**: para una prueba de humo la familia de CPU
es irrelevante —no se compara nada entre máquinas— y sube el catálogo a 14 ofertas. Su resultado se
añade abajo.

Lo que **sí** está descartado sin máquina, porque se midió en local: el dataset de stride 1 se
extrae en 24 s, ocupa 6,6 MB de `npz` y su pico de RSS fue ~460 MB sobre 3,8 GB.

### Y lo que esta validación no es

No es el estudio. No hay veredicto de stride: son 3 épocas y 1 semilla, y el propio informe se
niega a leer un eje con un solo brazo. El estudio de verdad son 25 runs y su criterio está escrito
en `docs/plan-stride-2026-08-27.md` §3, **antes de mirar**.

## 5. Resultado de la tercera corrida

*(pendiente al escribir esto — se rellena al terminar; log en `/tmp/stride-humo3.log`, y la fuente
de verdad es `runs/stride-h01-0000-seed1/` y `sweeps/stride-h01/flota.json`)*
