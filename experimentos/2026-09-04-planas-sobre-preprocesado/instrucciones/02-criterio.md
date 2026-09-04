# El criterio, **escrito antes de correr nada**

> Congelado el 2026-09-04, antes de que existiera un solo run de este experimento. No hay ninguna
> época de ningún brazo. Lo que se escribe abajo no puede cambiarse al ver los números.
>
> ⚠ **Lo que sí se sabe, y se dice para no fingir un ciego perfecto:** el experimento
> [detenido](../../2026-09-04-preproceso-kernel-congelado/) midió *la misma función* con redes de
> 4 capas y 69.340 parámetros, y dio +0,186/+0,309 sobre sus anclas a la ép. 3 que se desplomaron
> a +0,030/+0,055 a la ép. 11. **Aquí la red es 242× más pequeña**, así que esos números no son
> una predicción — pero el *patrón* (ventaja inicial que se evapora) sí es la hipótesis a batir.

## La pregunta

¿Sirve de algo alimentar una CNN plana con la salida de un kernel **ya aprendido y congelado**, en
vez de con los píxeles crudos — **en el régimen más barato posible**?

## Qué se corre

Tres brazos, idénticos salvo por el tamaño de su entrada. Receta **`plan40`** (lr 0,0014 ·
`batch_size` 85 · semilla 1), la misma de los siete gemelos. Stops **3 · 11 · 24 · 37**, los
mismos. `--patience 0`, para que «misma época» sea cierto por construcción.

| brazo | entrada | features | parámetros | ancla iso-features |
|---|---|---:|---:|---|
| `pre-1k3` | (1, 18, 18) | **18** | 286 | **`pre-1k5`** (mismo ancho, por construcción) |
| `pre-1k5` | (1, 16, 16) | **18** | 286 | **`pre-1k3`** |
| `pre-1k7` | (1, 14, 14) | 8 | 166 | ninguna |

## ⚠⚠ Lo que este experimento puede y NO puede contestar — y es la parte que importa

**No puede compararse con los siete gemelos.** Aquéllos son planas de 1 capa, 16 canales,
sin stride, con 2.511–19.656 parámetros. Éstos tienen 2 capas, 2 canales, stride 2 y **286**.
Poner sus f1 en la misma tabla sería comparar dos cosas distintas; los gemelos entran sólo como
**contexto de escala**, nunca como ancla.

**Lo que sí puede contestar** es la comparación **interna**: `pre-1k3` contra `pre-1k5`, que por
construcción tienen **exactamente 18 features** y redes bit a bit idénticas en forma. Ahí lo único
que cambia es **con qué kernel se preprocesó**, que es justo la pregunta. Es la comparación más
limpia de toda la serie, y sale gratis.

**El `pre-1k7` no tiene ancla** (8 features contra 18) y su lectura es sólo descriptiva.

## Los desenlaces, a época fija

Banda de ruido **0,04**, la misma que fijaron los criterios de los gemelos.

| si `pre-1k3` y `pre-1k5` difieren en | veredicto |
|---|---|
| **≤ 0,04** | **el tamaño del kernel de preproceso NO importa** a este régimen. Con 18 features iguales y la misma red, un empate dice que da igual resumir con 3×3 o con 5×5 |
| **> 0,04 a favor del `1k3`** | **preprocesar con el kernel PEQUEÑO conserva más**, coherente con que recorta menos la imagen antes de resumirla |
| **> 0,04 a favor del `1k5`** | **el campo receptivo mayor del preproceso aporta**, pese a llegar con la misma anchura a la cabeza |

⚠ **Ningún desenlace declara que «el preproceso sirve o no sirve»**, porque no hay brazo crudo en
este régimen. Para eso haría falta una cuarta red idéntica sobre la vista 20×20 sin preprocesar, y
**no está corrida**. Se dice ahora para que no se lea de más después.

## Cuándo se puede leer

⚠⚠ **A 3 épocas NO se declara nada.** Medido dos veces en este proyecto: en los gemelos el orden a
la ép. 3 sale **invertido** respecto al final (`1k3` es el peor a la 3 y el mejor a la 37), y en el
experimento detenido la ventaja se dividió por 6 entre la 3 y la 11.

1. **Ép. 3** — sólo «arrancó / no arrancó», el reloj y la forma de la curva.
2. **Ép. 11** — primera lectura con veredicto, **provisional**.
3. **Ép. 24 y 37** — lectura firme. Sólo la de 37 entraría en un reporte central.

## Lo que este experimento NO puede contestar, dicho antes

1. **Una semilla por brazo.** Acota, no declara.
2. **⚠ La red puede ser demasiado pequeña para la tarea.** 286 parámetros y 18 features hacia una
   cabeza de 12 salidas. Si los tres brazos salen mal a la vez, el desenlace más probable **no** es
   «el preproceso no sirve» sino «este régimen no da para esta tarea», y hay que decirlo así. Un f1
   bajo en los tres **no es evidencia contra el preproceso**.
3. **Sin los escalares de borde** (`n_edge=0`): el dataset no los trae. Están medidos como flojos
   —cerraban <1/5 del hueco del borde (0,608 → 0,674)— pero la foveada sí los lleva.
4. **El canal de relleno se perdió** al preprocesar: el kernel consume `(vista, relleno)` y devuelve
   un mapa. El reporte #19 midió que ese canal vale mucho (0,608 → 0,974). Es una tercera cosa que
   se mueve y no está controlada.
5. **Nada de esto mueve producción.** Instrucción del dueño (2026-09-03): un experimento no cambia
   el código de producción hasta que el número lo respalde.
