# Plan — `edge_inputs`: decirle a la cabeza dónde se acaba la imagen (2026-08-31)

**Estado: el mecanismo está implementado y probado; NO se ha medido nada.** Este
documento fija **qué se mide y cómo se lee, antes de mirar**, que es la regla de
[protocolo.md](protocolo.md) §1. Escribirlo después convierte cualquier resultado
en la confirmación de lo que ya se creía.

---

## 1. La hipótesis, y por qué no es una corazonada

No sale de «probemos entradas nuevas». Sale de un **límite del muestreo** que se
puede escribir sin entrenar nada:

`pad_mode: edge` replica la fila/columna del borde cuando el recorte se sale de la
imagen (decisión C11 — nunca ceros a secas, porque cero significa «no hay tinta» y
enseña una regla falsa). Esa réplica es, **por construcción**, indistinguible de
imagen real que se parece a más de lo mismo. Entonces:

| en la imagen | lo que la red ve | lo que la etiqueta dice |
|---|---|---|
| párrafo **pegado** al borde superior | banda replicada arriba | TL/TR **existen** ahí |
| párrafo **cortado** por el borde de la vista | banda de contexto arriba | TL/TR están fuera, más arriba |

Dos entradas casi idénticas con etiquetas opuestas. **Ninguna arquitectura puede
separarlas**, porque la información no está en la entrada. La hipótesis es que
darle esa información a la cabeza recupera parte de ese error.

⚠ **Es una hipótesis sobre un límite, no sobre capacidad.** Si falla, no falla
«porque hacían falta más parámetros»: falla porque el error que ataca era más
pequeño de lo que parece, o porque la red ya lo estaba compensando por otra vía
(el `pos_weight` que perdió en el #14 apuntaba al mismo cuello de botella y
empeoró).

## 2. Qué se añadió, exactamente

`edge_inputs` ∈ {`off` · `pad` · `dist`}, en `NETWORK_DEFAULTS` (o sea, **eje de C
barrible** — la trampa que costó `dropout`: `full_config` filtra por ese dict, y un
campo que no esté ahí hace que el barrido entrene N veces la misma red sin avisar).

- **4 escalares por ventana**, en orden `EDGE_SIDES = (L, T, R, B)`, en [0, 1],
  orientados igual en los dos modos: **0 = no hay borde por este lado, 1 = está aquí**.
- Van **concatenados a las features justo antes de la `Linear`**, fuera del ReLU y
  fuera del dropout. No tocan ninguna convolución. El porqué, en
  [instructionsNewNN.md §6bis](../instructionsNewNN.md).
- Coste: **+48 pesos** (4 × 12), +0,03 % sobre los 159.372 de la base vigente.
  *(medido 2026-08-31 con `fv.models.network_trace`)*
- `off` es el default y es **bit-idéntico** a la red anterior: los checkpoints en
  disco cargan `strict` y el forward no se mueve. 18 tests en
  `tests/test_edge_inputs.py`.

## 3. Lo que ya está medido sin gastar nada — y acota el resultado

Sobre `dirty1000-80px-16px-r20260827` (1000 imágenes 80×60, fóvea 16, stride 8,
140.000 ventanas, 72.380 esquinas positivas). *Medido el 2026-08-31 recorriendo el
`.npz` con `fv.fovea.edge_features`; no se entrenó nada.*

| | ventanas con la señal encendida | esquinas positivas dentro |
|---|---:|---:|
| `pad` | 44.000 = **31,4 %** | 22.022 = **30,4 %** |
| `dist` | 128.000 = **91,4 %** | 63.343 = **87,5 %** |

**Esquinas etiquetadas a ≤ 1 px del borde de la imagen: 2.183 = 3,02 %** de las
positivas. Ése es el caso literal («el párrafo está pegado al borde»).

Tres cosas que esto fija **antes** de medir:

1. **El efecto no está limitado al 3 %.** La ambigüedad la sufre **toda** ventana
   con relleno (31,4 %), tenga o no una esquina pegada: en todas ellas la red está
   viendo píxeles inventados y no lo sabe.
2. ⚠ **`dist` está encendida en el 91 % de las ventanas.** En una imagen de 5×3,75
   fóveas, «cerca del borde» y «en qué parte de la página estoy» son casi la misma
   variable. **Si `dist` gana, no se podrá decir por cuál de las dos.** `pad` no
   tiene esa ambigüedad.
3. La proporción es **idéntica en train, val y test** (31,4 % los tres, al 0,1 %),
   porque el reparto es por imagen. No hay sesgo de split que explique nada.

## 4. El tanteo: `ei-t`

**`edge_inputs` ∈ {`off` · `pad` · `dist`}**, 2 semillas = **6 runs**.
Estimado **≈0,4 $ y ~2,5 h** *(estimado, no medido: 3 × 2 × ~50 épocas a los 53 s/época
del tanteo de dropout, que es la misma base `ws16-p2-d2-L4` sobre el mismo dataset)*.

- Base y receta: **las vigentes**, sin tocar nada más. Es un eje cost-neutral en
  reloj (+48 pesos no mueven el s/época de forma medible), a diferencia de
  `patience` o `border_reduce`.
- Dataset: **`dirty1000-80px-16px-r20260827`**, el mismo de los números de §3.
- `epochs`: el tope actual basta — este eje no alarga los runs.

```bash
cd ~/src/foveal-vision
.venv/bin/python scripts/estudio_progreso.py --sweep ei-t --tabla   # ¿queda algo?
"$COORD_HOME/scripts/desacoplar.sh" sh -c '
set -a; [ -f "$COORD_HOME/.env" ] && . "$COORD_HOME/.env"
[ -f "$HOME/.config/dev-secrets.env" ] && . "$HOME/.config/dev-secrets.env"; set +a
.venv/bin/python scripts/estudio_flota.py --sweep ei-t --cpu E5-26 --criba 2 \
    --git --horas-max 6 --prefijo ei- --yes > /tmp/estudio-edge-tanteo.log 2>&1
node "$COORD_HOME/scripts/notify.mjs" "tanteo de edge_inputs (ei-t) terminado"' &
```

⚠ **Va DETRÁS de `do-v`**, que es lo que estaba pendiente antes de esto y ya tiene
su plan escrito (ver el `CLAUDE.md` del coordinador). Este documento no lo adelanta.

## 5. El criterio, escrito ANTES de mirar

**`edge_inputs` se queda en `off` y el estudio se cierra en el tanteo** si se
cumple **cualquiera**:

1. la amplitud entre los tres puntos **no llega a 0,010** (el doble del ruido
   típico entre semillas; el mismo umbral que el bloque B del #14 y que el tanteo
   de `patience`) — entonces el límite del muestreo existe pero **no es el cuello
   de botella**, igual que pasó con la brecha val/train y `dropout`; **o**
2. **ningún modo supera a `off`**. El eje se cierra en contra, que también es un
   resultado: diría que la red ya estaba compensando la ambigüedad por otra vía.

**Sólo se sube a 5 semillas** (`ei-v`) si se cumplen **las dos**: algún modo supera
a `off` por **más de 1 SE** *y* la amplitud pasa de 0,010.

⚠ **Y si el que gana es `dist`, el estudio NO termina ahí.** Por el punto 2 de §3,
un `dist` ganador es ambiguo entre «sabe dónde acaba la imagen» y «sabe dónde está
en la página». Desambiguarlo pide **una medida más**: repetir el punto ganador
sobre una fuente de imágenes **más grandes** (donde las dos variables se separan),
o barajar la señal (permutar los 4 valores entre ventanas de la misma imagen) y
comprobar que el f1 cae. Sin eso, el resultado no se puede citar como «el borde
importa».

Si gana **`pad`**, esa ambigüedad no existe y el resultado se lee directo.

## 6. Lo que este estudio NO contesta

- **No mide la métrica de tarea**, sólo el f1 de ventana. Y el proxy ya exageró una
  vez por un factor de dos (`n_layers`). Un ganador aquí es candidato, no adopción.
- **No dice si el canal de máscara `N×N` habría sido mejor.** F7 se cierra por el
  lado barato (48 pesos); si `edge_inputs` gana claramente, la pregunta de si un
  canal ganaría *más* vuelve a estar sobre la mesa — pero ahora con un suelo
  medido que superar en vez de con una intuición.
- **No toca `pad_mode`.** Si la red sabe dónde está el relleno, quizá el relleno
  pueda volver a ser `zero` o `mean` sin enseñar la regla falsa que C11 temía. Es
  un plano (`edge_inputs` × `pad_mode`), no una recta, y es otro estudio.

## 7. Al terminar

Reporte en
`estudios-redes-neuronales/reportes/estudios/2026/<mes>/<fecha>-edge-inputs-tanteo.md`
con inicio y fin en UTC, **instancias alquiladas** (no las que trabajaron), coste
real y el apartado de «lo que quedó pendiente». Y su fila al final de la tabla de
`reportes/README.md`, sin tocar las anteriores.
