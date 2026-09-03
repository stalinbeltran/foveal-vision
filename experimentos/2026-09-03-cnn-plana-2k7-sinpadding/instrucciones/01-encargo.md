> Encargo del dueño, 2026-09-03 (por Telegram):

Ok. Termina este experimento. Luego añade una copia del experimento plana-2k7 'sin padding'.

Y recuerda q estos son experimentos, nada tienen que ver con las redes previas, ellas serán
modificadas posteriormente. Si hay que hacer cambios al código tendremos que copiarlo localmente
(pero si vale la pena, y eso depende de nuestras pruebas en estos experimentos)

---

**Lectura:** «sin padding» aquí es **literal**, a diferencia del control de `replicate`:
`padding=0`, convolución *valid*, **no hay anillo porque no hay relleno**. La salida baja de
20×20 a 14×14 y la cabeza de 800 a 392 features — es inevitable al no rellenar, y hay que tenerlo
delante al comparar.

**Y el código va LOCAL**: `builder.py` calcula el relleno como `k//2` y no es un dato, así que
ponerlo a 0 desde una config habría pedido tocar producción. Se hace en
[`nn/red_local.py`](../nn/red_local.py) y [`nn/entrenar_local.py`](../nn/entrenar_local.py).
