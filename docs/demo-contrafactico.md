# `demo_contrafactico.py` — por qué «el mínimo cambio que hace acertar a la red» no es una mejora

**Qué contesta**, y nació de una pregunta del dueño el 2026-09-01: si se define *entrada
mejorada* como **la entrada más parecida a la original que hace que la red dé la salida
correcta**, ¿qué sale de ahí?

Esa definición es, escrita en fórmula, `min ‖δ‖ s.a. f(x+δ) = y*` — la formulación canónica
de un **ejemplo adversario dirigido**. Este script lo enseña en vez de argumentarlo: resuelve
el **mismo** problema tres veces sobre la **misma** ventana, cambiando sólo **en qué espacio
puede vivir el cambio**, y mide la norma de cada uno.

| | grados de libertad | qué es |
|---|---:|---|
| **(a) libre** | 1024 | cualquier cosa en los px del recorte |
| **(b) suave** | 25 | un campo 5×5 interpolado: iluminación / fondo |
| **(c) semántico** | 1 | cuánto se aplica una limpieza de fondo |

## Cómo se corre

Necesita un venv con `torch` (el de `foveal-vision`), unos pesos **aprobados** y un
`windows.npz`. No alquila nada, no toca red: son minutos de CPU.

```bash
# la tabla de geometría sola (no necesita dataset)
.venv/bin/python scripts/demo_contrafactico.py --geometria \
    --ckpt ~/src/foveal-vision-data/2026/08-agosto/runs/demo-fov16-optimo/best.pt

# la demostración entera (~4 min en un droplet de 2 vCPU, medido 2026-09-01)
"$COORD_HOME/scripts/desacoplar-persistente.sh" demo-contrafactico \
  /bin/bash -lc "cd ~/src/foveal-vision && .venv/bin/python -u scripts/demo_contrafactico.py \
    --ckpt ~/src/foveal-vision-data/2026/08-agosto/runs/demo-fov16-optimo/best.pt \
    --dataset ~/src/foveal-vision-data/window-datasets/dirty1000-80px-16px-r20260827/windows.npz \
    --out data/demo-contrafactico --n 3"
```

⚠ **Se lanza desacoplado y no con el `run_in_background` del harness**, aunque dure sólo
cuatro minutos. Esta demostración es justamente lo que lo dejó medido: murió dos veces con el
final del turno, con el directorio de salida vacío. El porqué está en
[`telegram-coordinator/CLAUDE.md` § «Un mensaje = un proceso que muere al responder»](https://github.com/stalinbeltran/telegram-coordinator/blob/main/CLAUDE.md).
⚠ Y **`-u`**: sin él el log se queda en blanco hasta que el proceso sale, y eso se lee como
«no ha arrancado».

## Qué salió — medido el 2026-09-01

Red `demo-fov16-optimo` (la única aprobada entonces), las **6.000** primeras ventanas
interiores del split val de `dirty1000-80px-16px-r20260827` → **352 con alguna esquina mal**.
Se eligen las **3 de margen más negativo** (fallos con convicción, no del filo: ordenar por
`|margen|` elige las que cualquier empujón da la vuelta, y ahí las tres rutas cuestan lo mismo
y no se ve nada).

| ventana | (a) libre | (b) suave | (c) semántico | px tocados por (a) | max\|δ\| de (a) |
|---|---|---|---|---|---|
| 10840 | **0,937** ✅ | 0,802 ✅ | 1,193 ❌ | 233/1024 (23 %) | 40/255 |
| 6948 | **0,892** ✅ | 2,984 ✅ | 0,176 ❌ | 767/1024 (75 %) | 45/255 |
| 13894 | **0,656** ✅ | 1,164 ✅ | 0,381 ❌ | 432/1024 (42 %) | 21/255 |

**Lo que se ve en los PNG, que es el punto:** el δ libre es **ruido sin estructura**, ningún
píxel movido más de 45/255, y la imagen resultante es indistinguible de la original — y
arregla las 4 esquinas en las tres ventanas. El δ semántico es **grande, en bloques y
visible** — y aplicado al 100 % **no arregla ninguna**.

## Los tres límites, para que no se lea de más

1. **3 ventanas, 1 red, 1 dataset.** Es una ilustración, no un estudio: no tiene semillas, no
   tiene criterio escrito antes de mirar y no declara nada.
2. **(a) y (b) son COTAS SUPERIORES.** Se resuelven con Adam sobre `‖δ‖² + c·hinge` y una
   rejilla de `c`; eso no garantiza el mínimo. La prueba está en la propia tabla: en la ventana
   10840 el espacio **suave** —que está *contenido* en el libre— sale **más barato** (0,802 <
   0,937), lo cual es imposible para un mínimo de verdad. El mínimo real es **menor o igual**,
   nunca mayor, así que el argumento no se debilita — se refuerza.
3. **(c) es UNA limpieza concreta** (dilatación en gris con k=9, `limpiar_fondo`). Que ésta no
   arregle la red **no dice** que ninguna lo haga.

## Las dos decisiones de implementación que hay que respetar si se toca

1. **El δ se optimiza sobre los PÍXELES del recorte, nunca sobre la vista.** En la periferia
   una celda de la vista es la media de varios px reales (`--geometria` lo imprime: con la
   geometría vigente, 4 px en la cruz y 16 en las esquinas del anillo), así que puedes mover
   esos píxeles conservando la media **sin que la red vea nada** — y al revés, mover la celda
   exige mover todos. Un mínimo en el espacio de la vista **no corresponde a ninguna imagen**.
   En el centro sí es 1:1 (256 de las 400 celdas, el 64 % de las entradas).

2. **El gradiente hasta el píxel es exacto, y por eso no hace falta un resampler
   diferenciable.** `build_foveated_input` con `pool_mode='avg'` es un avg-pool separable de
   bins desiguales (`_axis_edges` + `add.reduceat`), o sea un **operador lineal**: la vista es
   `P @ recorte @ P.T`. El script comprueba esa costura contra la ruta de numpy que usan el
   dataloader y la inferencia — **`max|dif| = 6,45e-08`** *(medido 2026-09-01)*. Si alguien
   cambia el muestreo, ese número es lo que avisa.

⚠ **Y el criterio de «acierta» es sólo `exists` en las 4 esquinas.** Es la decisión binaria que
luego mueve o no mueve un párrafo, pero **no** es la métrica que manda: ésa es `paragraph_f1`
de imagen, después de umbral / stride / NMS / `min_size`. Arreglar una ventana no mueve
necesariamente ni un párrafo.
