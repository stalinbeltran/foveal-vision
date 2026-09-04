> Encargo del dueño, 2026-09-04 (por Telegram):

Ahora realiza un estudio similar al
https://github.com/stalinbeltran/foveal-vision/tree/main/experimentos/2026-09-03-cnn-plana-2k7-sinpadding
, pero con un unico kernel

---

**Lectura:** «similar al 2k7-sinpadding» fija todo menos una línea. Se hereda **entero** el
montaje de aquél —`padding=0` de verdad (convolución *valid*), código local sin tocar `src/fv/`,
receta `plan40`, semilla 1, dataset `dirty1000-80px-16px-r20260827`, las mismas 10 ventanas de
`../comun/` y los mismos stops (0, 3, 11, 24, 37 épocas)— y **sólo** cambia `channels: [2] → [1]`.

**Por qué este punto vale la pena, y no es «uno más».** El `2k7-sinpadding` dejó escrito en su §3
que la caída de f1 encajaba en una tendencia de **~0,09 por cada mitad de features**, y que eso
era *«una coincidencia numérica compatible con dos historias»*: «lo que duele es el tamaño» y «lo
que duele es quitar el relleno». Los tres puntos que había mezclaban los dos ejes:

| red | features | relleno | f1 |
|---|---:|---|---:|
| 4k7 `zeros` | 1.600 | ceros | 0,840 |
| 2k7 `zeros` | 800 | ceros | 0,739 |
| 2k7 sin relleno | 392 | **ninguno** | 0,656 |

Este run añade **196 features sin volver a tocar el relleno**: es la primera mitad exacta que se
mide **dentro** del régimen sin relleno, así que separa los dos ejes en vez de moverlos juntos.
