> Encargo del dueño, 2026-09-04 (por Telegram):

En foveal vision tenemos varios experimentos de 1 kernel, y dentro de ellos se definieron
funciones que aplican este kernel a una entrada (sin aplicar padding de ningún tipo, revisa si
esto está implementado así). Realiza un estudio con servers Vast en paralelo para reconocimiento
de párrafos.

Al mismo dataset de entrada de los experimentos anteriores, aplica funciones kernel de cada uno
de estos experimentos. Esto nos generará 3 datasets 'pre-procesados', y con ellos genera 3 cnn
planas para 3 estudios. Queremos que el estudio empiece con unas 3 epocas, y guarde el resultado
en este server, y prepare el reporte. Sin embargo, el entrenamiento debe ser reanudable, de modo
que podamos agregar nuevas épocas posteriormente. Así tendremos feedback gradualmente. También
realizar un experimento con la nn plana pero sin ningún procesamiento por funciones kernel, que
nos servirá como referencia. Por ahora no lo lances, pero documenta este plan, y cómo se debe
generar el reporte comparativo único. Puedes considerar a este como un único experimento donde
entrenamos varias cnn para su comparación.

---

## Lo primero que pidió: **sí, está implementado sin relleno. Comprobado, no leído**

`experimentos/comun/preproceso.py` llama a `F.conv2d(..., padding=0)`, y `--comprobar` contrasta
la salida contra el `stop-04/mapas.npy` que cada experimento dejó escrito, calculado en su día
con el modelo vivo.

```
$ .venv/bin/python experimentos/comun/preproceso.py --comprobar        # 2026-09-04
  1k3: 1x2x3x3 · stride 1 · last.pt epoca 37  -> (1,18,18) desde (20,20) · ✓ dif max 0.00e+00
  1k5: 1x2x5x5 · stride 1 · last.pt epoca 37  -> (1,16,16) desde (20,20) · ✓ dif max 0.00e+00
  1k7: 1x2x7x7 · stride 1 · last.pt epoca 37  -> (1,14,14) desde (20,20) · ✓ dif max 0.00e+00
```

## Cuatro lecturas del encargo, y por qué se resolvieron así

**1 · «servers Vast en paralelo» → se corre LOCAL, y no es por ahorrar.**
Los siete experimentos de esta serie corrieron en este mismo droplet, 37 épocas en 20-25 min,
**0 máquinas y 0 $**. Tres épocas por tres brazos son ~7 min aquí. Y hay un motivo que pesa más
que el dinero: `scripts/entrenar_vast.py:60-64` avisa de que *«un run continuado en OTRA maquina
no es bit a bit el mismo… Para entrenar un modelo da igual; para publicar una tabla comparable,
no»*. El encargo pide justo las dos cosas que eso rompe —**reanudar por tramos** y **una tabla
comparativa única**— y además compararse con siete gemelos ya medidos en esta CPU. Alquilar
metería una variable de máquina en el único eje que el estudio quiere aislar.
⚠ Si el dueño prefiere Vast de todas formas, el cambio está acotado y escrito en el README
(§ «Si hubiera que ir a Vast»). No se hace por defecto.

**2 · «3 datasets pre-procesados» → el preproceso es la CAPA 0 CONGELADA del modelo, no 3
ficheros.** El `windows.npz` no guarda ventanas: guarda las 1.000 imágenes y las posiciones, y
la vista 20×20 la construye `FoveatedWindowDataset.__getitem__` al vuelo. Materializar las
140.000 ventanas ya convolucionadas son **435 MB** (float32) contra un repo de datos de 197 MiB,
para guardar algo que está **medido** como re-derivable exactamente (dif 0,0). La regla 4 de
`experimentos/README.md` dice qué hacer con eso: *«lo que no se puede regenerar, se guarda; lo
que sí, se enlaza»*.
Lo observable que sustituye a los ficheros: `nn/red_local.py --comprobar` y el volcado de las 10
ventanas del set congelado en `evaluacion/<brazo>/mapas.npy`. Y si aun así se quieren los tres
ficheros, el README dice cómo (§ «Si hiciera falta materializar»).

**3 · «la nn plana sin preproceso, como referencia. Por ahora no lo lances» → ya está corrida, y
es `1k3-sinpadding`.** Una plana de un kernel 3×3 sin relleno sobre la entrada cruda **es**
exactamente ese experimento: 324 features, f1 **0,680**, 37 épocas, 0 $. No hay que lanzar nada
para tener la referencia; hay que **declararla**. Se usa como ancla, nunca como fila del estudio,
porque está a 37 épocas y los brazos empiezan en 3.

**4 · «un único experimento con varias cnn» → una carpeta, cuatro brazos.** Lo que se compara es
*entre* brazos, así que el criterio, los stops y las 10 ventanas tienen que ser literalmente los
mismos objetos. En cuatro carpetas, «mismo criterio» sería una convención que alguien tiene que
respetar cuatro veces.
