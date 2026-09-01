# Plan — `mask_channel`: decirle a la red qué píxeles se ha inventado (2026-09-01)

**Qué contesta este documento:** qué se ha implementado, por qué así y no de otra
forma, **qué red se entrena en Vast, con qué medidas de seguridad, y con qué
criterio se leerá el resultado — escrito antes de mirarlo**
([protocolo.md](protocolo.md) §1). Escribirlo después convierte cualquier número
en la confirmación de lo que ya se creía.

**Encargado por el dueño el 2026-09-01**, después de que la revisión a ojo en la
pantalla Revisar encontrara que el relleno del borde *«hace parecer que el párrafo
se extiende»*.

---

## 1. El fallo, medido

`pad_mode: edge` replica la fila/columna del borde cuando el recorte se sale de la
imagen. Esa réplica es, **por construcción**, indistinguible de imagen real que
sigue. Entonces dos situaciones producen casi la misma entrada con etiquetas
opuestas:

| en la imagen | lo que la red ve | lo que la etiqueta dice |
|---|---|---|
| párrafo **pegado** al borde de la página | banda replicada | la esquina **existe** aquí |
| párrafo **cortado** por el borde de la vista | banda de contexto real | la esquina está fuera, más allá |

Hasta hoy eso era una hipótesis escrita
([plan-edge-inputs](plan-edge-inputs-2026-08-31.md) §1). **Ahora está medido.**

*Medido el 2026-09-01 sobre el split val de `dirty1000-80px-16px-r20260827`
(28.000 ventanas, 14.724 esquinas positivas), con `demo-fov16-optimo`. La
medición reproduce exactamente el `val f1` y el `pos_err_px` que el propio run
registró (0,9475 y 1,214 px), así que el desglose sale de la misma tubería que la
tabla oficial.*

| esquina a … del borde de la imagen | esquinas | recall | err px |
|---|---:|---:|---:|
| **0–1 px** | 380 | **0,608** | **2,30** |
| 1–2 px | 393 | 0,977 | 1,28 |
| 2–4 px | 663 | 0,974 | 1,12 |
| 4–8 px | 2.158 | 0,962 | 1,11 |
| > 8 px | 11.130 | 0,939 | 1,20 |

**El recall se hunde 33 puntos en el último píxel, y sólo ahí.**

Dos cosas que este desglose fija y que cambian el planteamiento:

1. ⚠ **No es que «las ventanas con relleno sean malas».** Su f1 es **0,9490**
   contra 0,9453 de las que no llevan relleno. El relleno como tal no molesta:
   molesta **la esquina que cae dentro de él**.
2. **El alcance está acotado**: 3,02 % de las esquinas positivas están a ≤ 1 px del
   borde. Pero **el 15,6 % de las imágenes** tienen tinta tocando el borde, y una
   esquina perdida no empeora una caja — **borra el párrafo entero**. El f1 de
   ventana infravalora esto por construcción.

## 2. Lo que ya se descartó, y con qué

**No se prueba `pad_mode` otra vez.** Ya se barrió (`pad-t`, 2026-08-27, 3 modos ×
2 semillas sobre `r20260826`): `edge` 0,9271 · `mean` 0,9258 · `zero` 0,9303,
**amplitud 0,0045** — por debajo del umbral de 0,010 del proyecto, y con la
dispersión *dentro* de `zero` (0,0128) mayor que la que hay *entre* modos.
Veredicto ya publicado: *«tanteado, sin señal»*
([cierre de parámetros](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/08-agosto/2026-08-26-cierre-parametros.md) §2.4).

**Y eso encaja con el diagnóstico**: si el problema fuera *qué* píxeles inventas,
el eje no saldría plano. Sale plano porque el problema es que **la red no sabe que
son inventados**.

⚠ **Y hay un cabo suelto que este plan no resuelve**: en estos datos el papel es
claro (mediana 244; el 87 % de los píxeles ≥ 224), así que `pad_mode: zero` pinta
un marco **negro** — tinta pura. El comentario de la decisión C11 dice que no se
usan ceros *«porque cero significa no hay tinta»*, que es lo contrario de lo que
hay en disco. O el convenio cambió, o la nota está invertida. **No se toca aquí**;
queda anotado.

## 3. Lo implementado: `mask_channel` ∈ {`off` · `coverage`}

La información que falta —*qué celdas son inventadas*— **ya se calcula en cada
paso y se tiraba**: `build_view` devuelve `(vista, cobertura)` y todos los caminos
hacían `view, _cov = ...`. Sólo la usaba la vista de depuración F0.

- **Un segundo canal de entrada** con el **relleno** (`1 − cobertura`), celda a
  celda.
- **Va sólo a la rama periférica.** ⚠ Y esto no es una economía: *medido el
  2026-09-01 con las dos geometrías vivas (borde 4/solape 2 y borde 8/solape 7),
  bajo la máscara del **centro** la cobertura es 1,000 en todas las celdas,
  también en la ventana (0,0)* — las 76 celdas con relleno caen enteras bajo la
  periferia. Es por construcción: la fóvea está dentro de la imagen. Dárselo al
  centro serían 144 pesos **equivalentes a un término de sesgo**.
- **Coste: +144 pesos** (168.700 → 168.844 con la config de esta red, +0,085 %).
- **`off` es el defecto y es bit-idéntico** a la red anterior: mismo `state_dict`
  salvo la forma de `periph_convs.0.weight`, checkpoints cargan `strict`.
- Es **eje barrible de C** (está en `NETWORK_DEFAULTS`, la trampa que costó
  `dropout`), con **puerta**: `mask_channel` desconocido y `border_px = 0` se
  rechazan *antes* de entrenar. **11 tests** en `tests/test_mask_channel.py`.

**El dibujo**, generado del propio código (`scripts/diagrama_red.py`, así que no
puede quedarse desfasado): [`red-fov16-optimo-mask.svg`](red-fov16-optimo-mask.svg)
— y el mismo sin el canal, para comparar: [`red-fov16-optimo.svg`](red-fov16-optimo.svg).

### Por qué un canal y no sólo los 4 escalares de `edge_inputs`

`edge_inputs` resume la misma información en 4 números y la entrega a la
**cabeza**, después de las convoluciones. El canal la entrega **dentro** de la
convolución, que es donde nace la ambigüedad.

⚠ **Esto contradice en parte un comentario del propio modelo**
(`builder.py`, sobre `edge_inputs`), que argumenta que el canal es mala idea
porque *«los ramales gastarían kernels re-derivando una constante»* y *«la rama
del centro no ve el anillo»*. **Las dos objeciones son ciertas y las dos están
contestadas**: la primera es exactamente por lo que el canal **no** va al centro
(allí sí sería constante); la segunda deja de ser una objeción cuando la
información se le da a la rama que **sí** ve el anillo, y la cabeza lee las dos
ramas concatenadas.

⚠ **Y la evidencia de que hace falta algo más que la cabeza es débil, pero
existe:** `fov16-edge-p20` (`edge_inputs: pad`, ya entrenada) sube el recall del
borde de **0,608 → 0,674** con el f1 global idéntico — se mueve donde debe y **no
cierra ni una quinta parte** del hueco. *Una semilla cada una y ni siquiera un par
controlado (103 épocas contra 74): es un indicio, no un resultado.*

## 4. Lo que se entrena, y qué NO es

**`fov16-mask-p20`**: `fov16-optimo` + `edge_inputs: pad` + `mask_channel: coverage`,
sobre `dirty1000-80px-16px-r20260827`, receta `plan40` (patience 20), en Vast.

⚠ **Es UNA red entrenada para USARLA, no un estudio.** Una semilla no declara
nada — la regla del proyecto es que con 2 semillas el `p` mínimo ya es 0,333. Lo
que sí es: **el tercer punto de una familia comparable**, porque las otras dos
(`demo-fov16-optimo` con `off`, `fov16-edge-p20` con `pad`) están sobre el mismo
dataset y la misma base.

## 5. El criterio, escrito ANTES de mirar

Al terminar se corre el **mismo desglose por distancia al borde** de §1 sobre las
tres redes, y se lee así:

1. **El canal cumple** si el recall del tramo **0–1 px** sube por encima de
   **0,75** *(a mitad de camino entre el 0,608 de `off` y el 0,939 de las
   ventanas interiores)* **y** el f1 global no baja más de **0,005**.
2. **El canal no aporta** si el recall del tramo 0–1 px se queda **por debajo de
   0,70**, o sea sin separarse del 0,674 que ya daba `edge_inputs` solo. Entonces
   la conclusión es que **la cabeza ya recibía lo que hacía falta** y el problema
   no es dónde entra la señal.
3. **Entre 0,70 y 0,75: no concluye.** Con una semilla no se distingue de ruido y
   haría falta el tanteo (`mk-t`, 2 modos × 2 semillas) para decir algo.

⚠ **Y nada de esto mueve el vigente.** Cualquiera de los tres desenlaces es
material para un tanteo, no una adopción: `demo-fov16-optimo` sigue siendo la red
aprobada hasta que un estudio con semillas diga otra cosa.

## 6. Lo que este entrenamiento NO contesta

- **No mide la métrica de tarea**, sólo el f1 de ventana y el desglose por
  distancia. Y el f1 de ventana **infravalora** este fallo (§1.2). La medida que
  lo cerraría es el f1 de **párrafo** sobre imágenes con tinta en el borde, y pide
  una fuente con verdad.
- **No separa el canal de `edge_inputs`**, porque la red lleva los dos. Si gana,
  no se podrá decir cuál de los dos lo hizo. Es deliberado: se entrena para usarla,
  y para separar están los tanteos.
- **No toca el generador.** Quitar los párrafos del borde
  (`placement.area`, un campo de la receta) es la otra salida y es **una hipótesis
  sobre el dominio**: sólo vale si las imágenes reales tampoco los tienen. No se
  hace aquí.

## 7. Al terminar

Reporte en
`estudios-redes-neuronales/reportes/estudios/2026/09-septiembre/` con inicio y fin
en UTC, **instancias alquiladas** (no las que trabajaron), coste real y el
apartado de «lo que quedó pendiente». Y su fila al final de la tabla de
`reportes/README.md`.
