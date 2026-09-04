# `experimentos/` — una carpeta por experimento, con TODO dentro

Cada experimento vive en `experimentos/<fecha>-<nombre>/` y es **autocontenido**: el encargo, el
criterio, el código que lo produjo, los resultados y **la red con sus pesos**. La idea es poder
abrir la carpeta dentro de un año y saber qué se hizo, por qué, y volver a cargar el modelo sin
depender de que el resto del repo siga teniendo la misma forma.

| | |
|---|---|
| [`2026-09-03-cnn-plana-4k7/`](2026-09-03-cnn-plana-4k7/) | CNN **plana** de 1 capa y **4** kernels 7×7, misma entrada y salida que la foveada con máscara. 37 épocas (24,0 min) · f1 **0,840** |
| [`2026-09-03-cnn-plana-2k7/`](2026-09-03-cnn-plana-2k7/) | **El gemelo con 2 kernels.** Mismos stops, misma semilla, mismas 10 ventanas. 37 épocas (24,7 min) · f1 **0,739** — **quitar dos kernels cuesta 0,10 de f1** |
| [`2026-09-03-cnn-plana-4k7-replicate/`](2026-09-03-cnn-plana-4k7-replicate/) | **El control del anillo de padding**: mismo 4k7 con `conv_pad_mode: replicate`. El anillo baja de 9,43× a 2,17× — y el f1 **baja** de 0,840 a 0,820. **El arreglo obvio no era una mejora** |
| [`2026-09-03-cnn-plana-2k7-sinpadding/`](2026-09-03-cnn-plana-2k7-sinpadding/) | El 2k7 con `padding=0` **de verdad**: sin anillo porque no hay relleno. Cae a 2,15× — **el mismo residuo que `replicate`, por otro camino**. f1 **0,656**, y arrastra el confound de que la cabeza baja a 392 features. **Código local**, sin tocar `src/fv/` |
| [`2026-09-04-cnn-plana-1k7-sinpadding/`](2026-09-04-cnn-plana-1k7-sinpadding/) | **UN kernel 7×7**, sin relleno. 196 features · f1 **0,618**. La tendencia de «~0,09 de f1 por mitad de features» **no continúa**: esta mitad cuesta 0,038, dentro del ruido. Y el único kernel resulta ser un **promediador** (24 % de su energía en el DC) |
| [`2026-09-04-cnn-plana-1k5-sinpadding/`](2026-09-04-cnn-plana-1k5-sinpadding/) | **UN kernel 5×5**, sin relleno. 256 features · f1 **0,642** — el más rápido de los seis. Bajar el campo receptivo de 7 a 5 px **no cuesta nada medible**: manda el tamaño de la cabeza |
| [`2026-09-04-cnn-plana-1k3-sinpadding/`](2026-09-04-cnn-plana-1k3-sinpadding/) | **UN kernel 3×3**, sin relleno — el valor que usa la foveada de producción. 324 features · f1 **0,680**, el mejor de los tres de un kernel. ⚠ **Su README y su lectura del criterio están PENDIENTES**: el run terminó (37 épocas) y sus stops están, pero nadie ha escrito qué salió contra lo que `instrucciones/02-criterio.md` pedía |
| [`2026-09-04-preproceso-kernel-congelado/`](2026-09-04-preproceso-kernel-congelado/) | ⛔ **DETENIDO en la ép. 11 por el dueño**: pedía 3 planas sobre 3 datasets preprocesados y esto son 3 redes de DOS convoluciones sobre el dataset de siempre (el preproceso quedó DENTRO del modelo, así que la entrada siguió siendo 20×20). Se conserva como el **brazo CON ReLU** del nuevo. La L1 ya entrenada de `1k3`/`1k5`/`1k7`, **congelada**, como preproceso de una plana nueva. Va por la **época 11 de 37**, primer veredicto y **provisional**: la ventaja sobre su ancla iso-features **se desploma** de +0,186/+0,309 (ép. 3) a **+0,030/+0,055** (ép. 11). O sea que el preproceso da **velocidad de convergencia, no calidad** — justo lo que el criterio escribió antes de mirar |
| [`2026-09-04-planas-sobre-preprocesado/`](2026-09-04-planas-sobre-preprocesado/) | **Los 3 datasets ya CONSTRUIDOS y COMMITEADOS** en `foveal-vision-data/preprocesado/` (18² · 16² · 14²; 98 MB comprimidos, 436 MB en crudo — ⚠ caben sólo porque se comprimen: GitHub rechaza >100 MB por fichero), con su muestra de las mismas 10 ventanas en PNG. Y las **3 estructuras de red**: planas de 4 capas con los ÓPTIMOS medidos de la foveada (`n_layers` 4 · `k` 3 · `channels` [16]×4), **idénticas salvo la cabeza** — las convs son 7.120 params exactos en las tres. Falta el entrenamiento. Lo que el detenido de arriba debía haber sido: los 3 datasets preprocesados se **escriben a disco antes** de entrenar (18² · 16² · 14²) y encima entrena una plana **de verdad** (UNA convolución). ⚠ Hay una decisión abierta que decide si el estudio puede salir a favor: si el mapa se guarda **con signo** la plana colapsa a una sola conv atada y sólo puede empatar o perder |
| [`comun/`](comun/) | el evaluador, el set de 10 ventanas y las entradas que **comparten los siete**, más `serie.py`, `concentracion.py`, `cargar_pesos.py` y [`preproceso.py`](comun/preproceso.py). Dos copias derivarían y la comparación se volvería una ilusión |
| [`2026-09-03-sonda-l1/`](2026-09-03-sonda-l1/) | ¿Pueden los kernels de la primera capa aprender filtros genéricos si SÍ hay presión sobre ellos? **Respuesta: no.** 8 redes entrenadas, 2,1 h, 0 $ |

⚠ **El código de producción NO se toca para probar una idea.** Instrucción del dueño
(2026-09-03): *«estos son experimentos, nada tienen que ver con las redes previas… Si hay que
hacer cambios al código tendremos que copiarlo localmente (pero si vale la pena, y eso depende de
nuestras pruebas en estos experimentos)»*. Cuando un experimento necesita un cambio que no cabe
en una config, vive en su `nn/` — ver `cnn-plana-2k7-sinpadding`, que parchea **en memoria** el
único punto donde `fv.training.loop` construye el modelo y deja `src/fv/` intacto (lo comprueba
al terminar cada corrida).

⚠ **Los experimentos gemelos comparten su medida, no la copian.** Los seis de la serie plana
caen en las **mismas épocas** (0, 3, 11, 24, 37), arrancan de la **misma semilla** y ven las
**mismas 10 ventanas**; para que ponerlos uno al lado del otro signifique algo, el evaluador y el
set viven en [`comun/`](comun/) y no dentro de ninguno. La tabla de los seis la imprime
[`comun/serie.py`](comun/serie.py), leída de los `metrics.jsonl` commiteados — transcribirla a
mano es como nacen los números que nadie puede auditar.

## ⏳ PENDIENTE: medir el efecto del relleno **en la FOVEADA**

**Anotado el 2026-09-04 a petición del dueño.** Toda la evidencia sobre el relleno de la
convolución que hay en este repo es de **redes planas**. En la foveada —la red de producción—
**nadie lo ha medido nunca**.

### Lo que sí está medido (y por qué no se puede extrapolar)

Misma receta en los cuatro: se resta a cada mapa el nivel mediano de su interior y se compara el
anillo de `k//2` px contra el interior ([`porque_el_anillo.py`](2026-09-03-cnn-plana-4k7/nn/porque_el_anillo.py)).

| | anillo / interior | f1 |
|---|---:|---:|
| `4k7` **`zeros`** *(lo que hace producción)* | **9,43×** | **0,840** |
| `4k7` `replicate` | 2,17× | **0,820** |
| `2k7` **sin relleno** | 2,15× | 0,656 ⚠ *confundido: la cabeza cae a 392 features* |

⚠⚠ **El arreglo obvio NO era una mejora**: `replicate` baja el anillo 4,3× y el f1 **también
baja**. Eso es exactamente lo que hace que este eje merezca medirse en vez de darse por sabido.

### Por qué la foveada puede comportarse distinto — no es una suposición gratuita

El [reporte #19](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/09-septiembre/2026-09-01-canal-de-relleno.md)
midió que el problema del borde **no es qué píxeles inventas, es que la red no sabe que son
inventados**: `pad_mode` sale plano, y `mask_channel` (decírselo) mueve el recall de la esquina de
**0,608 a 0,974**. La foveada tiene ese canal y las planas de la serie no, así que el anillo puede
dejar de doler cuando la red sabe que está ahí. **Es una hipótesis, no un resultado.**

### ⚠ Lo que hay que saber antes de lanzarlo

1. **El tamaño del relleno NO es un dato hoy.** `builder.py:145` hace `pc, pp = kc // 2, kp // 2`:
   es una expresión, no un campo de config. Barrerlo pide **tocar producción** o parchear en local
   como hacen los experimentos de esta carpeta. `conv_pad_mode` (`zeros`/`replicate`) **sí** es
   dato desde `efb8d5d05`.
2. **Sin relleno la cabeza encoge, y en esta serie manda la cabeza.** Es el confound que arrastró
   el `2k7-sinpadding` y que haría ilegible una comparación ingenua: hay que igualar features o
   declarar el confound, como en el resto de la serie.
3. **Son DOS ejes, no uno**: el *tamaño* (`k//2` contra 0) y el *modo* (`zeros` contra
   `replicate`). El primero cambia la geometría; el segundo no.

Estado en el repo central:
[`ESTADO.md`](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/ESTADO.md),
fila «relleno de la convolución» — **sin medir en la foveada**.

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
   `foveal-vision-data`, porque 862 runs × 2,7 MB son ~2,3 GB. Estas redes son pequeñas —de 304 a
   19.656 parámetros— y **los doce checkpoints ocupan 1,1 MB**. La razón de la regla no aplica, y
   el dueño los pidió explícitamente el 2026-09-03 (*«guarda los pesos de la nn en el
   experimento»*). Van con su `config.json`, su `metrics.jsonl` y su `summary.json`: el registro
   completo, no sólo el tensor.
   ⚠ **Se comprueba que CARGAN, no que están.** Un `.pt` que no carga ocupa sitio y parece un
   respaldo. [`comun/cargar_pesos.py`](comun/cargar_pesos.py) los abre y contrasta la norma de sus
   kernels contra la que quedó escrita en el último stop.
   ⚠ **Eso no cambia la regla para la red de producción.** Un `best.pt` de `fov16-optimo-mask`
   son 665 KB y sigue yendo al repo de datos, aprobado uno a uno por `inferencia.json`.

4. **Lo que no se puede regenerar, se guarda; lo que sí, se enlaza.** Los pesos, las métricas y
   las figuras se copian. El dato de entrada no: es un `windows.npz` de
   `foveal-vision-data/window-datasets/`, y el README de cada experimento dice **cuál**.

## Cómo se añade uno

Copia la forma de arriba. Lo mínimo que no puede faltar: **qué se preguntó**, **el criterio
escrito antes de mirar**, **qué salió**, y **cómo volver a correrlo**. Un experimento sin
criterio previo no es un experimento: es una anécdota con números.

## Usar un kernel entrenado como PREPROCESO: `aplicaKernel`

**Encargo del dueño, 2026-09-04:** *«de estos 3 experimentos toma sus kernels, y para cada uno
crea una función `aplicaKernel` que tome una entrada cualquiera (imagen, como las que empleamos
en nuestros entrenamientos) y le aplique este kernel sin padding. La salida de esta función será
luego empleada (opcionalmente) como pre-procesador de las imágenes de entrada»*.

```python
from preproceso import aplicaKernel_1k3, aplicaKernel_1k5, aplicaKernel_1k7

y = aplicaKernel_1k5(imagen)      # (20,20) -> (1, 16,16);  (B,2,H,W) -> (B,1,H-4,W-4)
```

…o desde dentro de un experimento, con su kernel ya puesto:

```python
from aplica_kernel import aplicaKernel     # experimentos/<el que sea>/nn/aplica_kernel.py
```

**Es literalmente la capa L1 de esa red, no algo parecido.** Las tres son `regions: single` y
`n_layers: 1`, así que su única convolución ve la entrada entera **sin máscara** y su mapa se
queda **sin activar** — la última capa no lleva ReLU a propósito (`builder.py:197`). Y eso no se
afirma: `python experimentos/comun/preproceso.py --comprobar` lo contrasta contra el
`stop-04/mapas.npy` que cada experimento dejó escrito, calculado en su momento con el modelo
vivo. *Medido el 2026-09-04: diferencia máxima **0,0** en los tres.*

| | k | entrada | salida | features |
|---|---|---|---|---|
| `aplicaKernel_1k3` | 3×3 | 20×20 | **18×18** | 324 |
| `aplicaKernel_1k5` | 5×5 | 20×20 | **16×16** | 256 |
| `aplicaKernel_1k7` | 7×7 | 20×20 | **14×14** | 196 |

**Las cuatro decisiones que hay que respetar si se toca**, y las cuatro tienen test
([`tests/test_preproceso.py`](../tests/test_preproceso.py), 37):

1. **No depende de `fv` ni de `red_local.py`.** El tensor se saca del `state_dict` a pelo, así que
   la carpeta del experimento se puede abrir dentro de un año con poco más. El precio es que
   `center_convs.0.weight` pasa a ser un contrato, y hay un test que lo fija contra la red
   construida — si el builder renombra esa capa, se rompe **ahí** y no en un entrenamiento seis
   meses después.
2. **La entrada trae DOS canales** (la vista y el **relleno**, `1 - coverage`). Una imagen suelta
   sólo trae uno, y el segundo se rellena con el **defecto declarado** `relleno=0` = *«todo píxel
   es real»*, que es lo que vale en el interior de una página. No es un detalle: con `relleno=1`
   la salida es otra, y hay test de las dos.
3. **Un float que se sale de [0,1] se NIEGA.** Es *el* fallo caro de un preprocesador: el kernel se
   entrenó sobre vistas en [0,1] (`build_view` divide por 255), y una entrada en niveles 0..255 no
   revienta — sale 255× y entrena algo. `uint8` sí se divide solo; para lo demás, `escala='0-255'`
   explícito. Igual con un 3-D ambiguo: `(C,H,W)` y `(B,H,W)` no se distinguen por la forma, así
   que se niega en vez de elegir.
4. **Vive en `comun/` porque los tres son gemelos.** Si sus salidas van a compararse como
   preproceso, la operación tiene que ser **la misma**; tres copias derivarían y la comparación
   sería una ilusión sin que nada fallara. Los `nn/aplica_kernel.py` son atajos de veinte líneas
   que sólo le atan su kernel.

⚠ **Es OPCIONAL y no toca producción.** `src/fv/` sigue intacto: la instrucción del dueño es que un
experimento no cambia el código de producción hasta que el número lo respalde. Esto es el material
con el que decidirlo, no la decisión.

⚠ **Y nadie ha medido todavía que preprocesar así sirva de algo.** Que la función exista y
reproduzca la capa L1 no dice nada sobre si una red entrenada sobre su salida va mejor. Eso es un
experimento, con su criterio escrito antes de mirar, y **no se ha hecho**.
