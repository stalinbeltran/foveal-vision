# `experimentos/` — una carpeta por experimento, con TODO dentro

Cada experimento vive en `experimentos/<fecha>-<nombre>/` y es **autocontenido**: el encargo, el
criterio, el código que lo produjo, los resultados y **la red con sus pesos**. La idea es poder
abrir la carpeta dentro de un año y saber qué se hizo, por qué, y volver a cargar el modelo sin
depender de que el resto del repo siga teniendo la misma forma.

| | |
|---|---|
| [`2026-09-03-cnn-plana-4k7/`](2026-09-03-cnn-plana-4k7/) | CNN **plana** de 1 capa y **4** kernels 7×7, misma entrada y salida que la foveada con máscara. 37 épocas (24,0 min) · f1 **0,840** |
| [`2026-09-03-cnn-plana-2k7/`](2026-09-03-cnn-plana-2k7/) | **El gemelo con 2 kernels.** Mismos stops, misma semilla, mismas 10 ventanas. 37 épocas (24,7 min) · f1 **0,739** — **quitar dos kernels cuesta 0,10 de f1** |
| [`comun/`](comun/) | el evaluador, el set de 10 ventanas y las entradas que **comparten los dos gemelos**. Dos copias derivarían y la comparación se volvería una ilusión |
| [`2026-09-03-sonda-l1/`](2026-09-03-sonda-l1/) | ¿Pueden los kernels de la primera capa aprender filtros genéricos si SÍ hay presión sobre ellos? **Respuesta: no.** 8 redes entrenadas, 2,1 h, 0 $ |

⚠ **Dos experimentos gemelos comparten su medida, no la copian.** `cnn-plana-4k7` y
`cnn-plana-2k7` sólo se diferencian en `channels`, y sus stops caen en las mismas épocas; para
que ponerlos uno al lado del otro signifique algo, el evaluador y las 10 ventanas viven en
[`comun/`](comun/) y no dentro de ninguno de los dos.

## La forma

```
<fecha>-<nombre>/
  README.md            qué se preguntó, cómo, qué salió, y cómo repetirlo
  instrucciones/       el encargo tal como llegó, las respuestas del dueño, el criterio
  nn/
    modelo.py          la red AUTOCONTENIDA: no importa nada del repo
    pesos/*.pt         un checkpoint por run
  resultados/          por run: config, métricas por época, resumen, kernels .npy, figuras
  codigo/              SNAPSHOT congelado del código que lo produjo
```

## Cuatro reglas, y las cuatro tienen un motivo

1. **`nn/modelo.py` no importa nada del repo.** Es lo único que garantiza que los pesos se
   puedan cargar más adelante. La copia **viva** vive en `src/fv/probe/` y va a seguir
   cambiando; ésta se congela con los pesos que produjo. Corre `python nn/modelo.py` y te lo
   comprueba contra los `.npy` guardados.

2. **`codigo/` es un SNAPSHOT, no una copia de trabajo.** No se edita nunca. Dos copias vivas del
   mismo código divergen y nadie se entera — por eso ésta está muerta a propósito, y el README
   del experimento dice de qué commit salió.

3. **Los pesos SÍ entran en git aquí, y es una excepción con motivo.** La regla general de este
   proyecto es que los pesos de un run **no se guardan por defecto** y viven en
   `foveal-vision-data`, porque 862 runs × 2,7 MB son ~2,3 GB. Estas redes son **de 304 a 2.608
   parámetros**: los ocho pesos ocupan **80 KB**. La razón de la regla no aplica, y el dueño los
   pidió explícitamente el 2026-09-03.
   ⚠ **Eso no cambia la regla para la red de producción.** Un `best.pt` de `fov16-optimo-mask`
   son 665 KB y sigue yendo al repo de datos, aprobado uno a uno por `inferencia.json`.

4. **Lo que no se puede regenerar, se guarda; lo que sí, se enlaza.** Los pesos, las métricas y
   las figuras se copian. El dato de entrada no: es un `windows.npz` de
   `foveal-vision-data/window-datasets/`, y el README de cada experimento dice **cuál**.

## Cómo se añade uno

Copia la forma de arriba. Lo mínimo que no puede faltar: **qué se preguntó**, **el criterio
escrito antes de mirar**, **qué salió**, y **cómo volver a correrlo**. Un experimento sin
criterio previo no es un experimento: es una anécdota con números.
