# El criterio, **escrito antes de correr nada**

> Congelado el 2026-09-04 a las 00:30 UTC, **antes** de lanzar la primera época. Lo que se mide
> después no puede cambiar esto: un criterio que se escribe al ver el número no es un criterio.

## La predicción que se pone a prueba

El README de [`cnn-plana-2k7-sinpadding`](../../2026-09-03-cnn-plana-2k7-sinpadding/README.md) §3
observó que **cada vez que se parte el número de features por la mitad se pierden ~0,09 de f1**, y
lo marcó explícitamente como una coincidencia sin explicación establecida.

Este run cae en **196 features**, otra mitad exacta. Si la tendencia es real:

> **f1 esperado ≈ 0,656 − 0,09 ≈ 0,57.**

## El ruido, medido antes (no estimado)

Banda de oscilación de f1 en las últimas 9 épocas de cada uno de los cuatro runs ya hechos
(`metrics.jsonl` de sus `nn/pesos/`, medido el 2026-09-04):

| run | banda de f1 |
|---|---:|
| 4k7 `zeros` | 0,019 |
| 4k7 `replicate` | 0,020 |
| 2k7 `zeros` | 0,025 |
| 2k7 sin relleno | 0,039 |

Se toma **0,04** como banda (el peor de los cuatro, que además es el del gemelo directo de este
run). ⚠ Es oscilación **dentro de un run**, no dispersión entre semillas: con `n = 1` no hay forma
de medir la segunda desde aquí, y la primera es un **suelo** de la segunda, no un sustituto.

## Los tres desenlaces, y qué concluye cada uno

| si el mejor f1 cae en | veredicto | qué significa |
|---|---|---|
| **0,53 – 0,61** (0,57 ± 0,04) | **la tendencia CONTINÚA** | cuatro puntos en la misma recta, y el tercero de ellos ya no confunde tamaño con relleno. Refuerza la lectura «lo que duele es el número de features»; **no la demuestra**, porque sigue sin haber un run que separe los dos ejes a la vez |
| **< 0,53** | **la tendencia SE ROMPE por abajo** | un solo kernel 7×7 no basta: con 99 parámetros en L1 la capa no puede abarcar lo que hace falta, y las caídas anteriores no eran «una recta» sino el principio de un desplome. **Sería el resultado más informativo**, porque acota por dónde está el suelo |
| **> 0,61** | **la tendencia SE ROMPE por arriba** | la cabeza estaba sobrada y las caídas anteriores medían otra cosa. Obligaría a releer el §3 del gemelo |

⚠ **No hay desenlace que «mueva» nada de producción**, y es a propósito: esto mide una plana de
una capa, no la foveada. La decisión que este experimento puede informar es si merece la pena
seguir bajando por este eje, no qué red se despliega.

## Lo que este run NO puede contestar, dicho antes

1. **Una semilla.** Acota, no declara. El `p` mínimo alcanzable con `n = 1` no existe.
2. **Sigue sin haber un run con `padding=0` y la cabeza compensada**, que es lo único que
   separaría del todo «relleno» de «tamaño». Este punto ayuda porque no mueve el relleno, pero no
   sustituye a aquél.
3. **Con un solo kernel no hay reparto que medir.** Toda la sección de concentración entre kernels
   —el 4,3× / 2,2× / 3,4× de los otros— aquí es literalmente inaplicable, y su ausencia no es un
   resultado.
