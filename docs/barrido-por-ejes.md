# Barrido por ejes (OAT) con base derivada del problema

> **Estado: DISEÑO, decisiones cerradas — sin implementar.** Este documento especifica una
> funcionalidad cuyo código se generará en otra sesión. **Las decisiones abiertas se resolvieron
> con el usuario el 2026-07-23; los valores fijados están en §13 y mandan.** Los marcadores
> `[DECIDIDO D-xx]` del cuerpo llevan el valor elegido; lo no listado en §13 se sigue preguntando,
> no se inventa.
>
> Cuando la funcionalidad se construya y verifique, actualícese el bloque «Estado actual» de
> [CLAUDE.md](../CLAUDE.md).

## 0. Qué es esto y por qué

El flujo real de trabajo hasta hoy ha sido **manual y lento**: para probar una variante de red el
usuario escribe a mano una definición C completa (~14 campos: `N, c_frac, d, pen_frac, n_layers,
k_center, k_periph, s_center, s_periph, channels, merge, pool_mode, pad_mode`), la nombra, lanza
un recorrido de un solo eje, mira el ranking, y repite. La evidencia está en
[sweeps/](../sweeps/): `dirty-80px-fast_kcenter`, `_kperiph_1`, `_s_center_1`, `_s_periph_1`,
`_14-k_center_1`… cada uno es un barrido de **un eje** con todo lo demás fijo. Eso es **descenso
por coordenadas** (one-factor-at-a-time, OAT) hecho a mano.

Esta funcionalidad **automatiza al humano, no inventa un camino nuevo**. Un generador escribe la
receta de recorrido que el usuario habría escrito, reutilizando **todos** los instrumentos ya
construidos (runs, recorridos, diagnósticos, matrixview, pantallas de la web app, validadores).
El único ingreso manual pasa a ser: **dataset + qué eje escanear + su rango**.

### Alcance (lo que ESTE documento cubre y lo que no)

- **Cubre (P1):** el generador escribe una **receta de recorrido (H)** con una **base inline
  derivada del problema** + **un solo eje** con su rango. Las estructuras bajo prueba siguen
  siendo **puntos del recorrido** (overrides sobre la base), no redes con nombre. Se reutiliza el
  recorrido tal cual, con una extensión mínima para aceptar la base inline.
- **Cubre (dependencia obligatoria):** el **builder paramétrico** (dominio C). Generar un config
  con `n_layers=3` o canales por capa **no sirve de nada** si el modelo sigue construyendo dos
  capas fijas. Esta pieza es prerequisito y se especifica en §3.
- **Cubre:** el **arrastre del ganador** (carry-forward) y el **schedule OAT** (orden de ejes con
  dependencias) para procesos largos.
- **No cubre (P2):** generar N **redes con nombre** en disco, una por estructura. Se descartó
  porque obliga a añadir «red como eje» o un contenedor de lote nuevo — reintroduce el código que
  P1 evita. Si en el futuro se quiere navegar cada estructura por nombre en la pantalla Redes,
  es otra decisión.
- **No cubre:** generar **código de modelo** (un `nn.Module` por estructura). Contradice el
  principio rector «todo dato es un parámetro» y multiplica lo que hay que testear. La red es
  **una sola, parametrizada**; el generador solo rellena parámetros.

### Decisiones ya tomadas por el usuario (no reabrir al codificar)

| # | Decisión | Consecuencia |
|---|---|---|
| U1 | **Descenso por coordenadas (OAT):** un eje a la vez, se fija el ganador, se pasa al siguiente | El coste es **suma** de ejes, no producto. Sin explosión por construcción |
| U2 | **Defaults estáticos y documentados** para los ejes que NO se barren (§4) | Contexto de referencia estable ⇒ el efecto de cada eje es comparable |
| U3 | **Arrastrar el ganador** como base del siguiente eje (§7) | La base evoluciona a lo largo de la cadena |
| U4 | **Base inline** en la receta de recorrido (no red con nombre) | Extensión mínima de H para aceptar base inline (§8) |
| U5 | **El usuario define el rango del eje** (además de `"auto"`) | Ya soportado por `check_sweep`; el generador lo respeta |
| U6 | **El usuario define el orden de barrido** de los ejes para procesos largos (§6) | El schedule es un plan ordenado con dependencias |

---

## 1. Principio rector aplicado

- **Todo dato es un parámetro.** La geometría se **calcula** de `N` y unas fracciones; los rangos
  de búsqueda los calculan las funciones de [instructionsNewNN.md](../instructionsNewNN.md) §3,
  ya presentes en [`fv.fovea`](../src/fv/fovea/__init__.py). El generador no escribe dimensiones a
  mano; las deriva.
- **La organización por dominios manda** ([docs/organizacion.md](organizacion.md)). Esta
  funcionalidad toca C (builder paramétrico), G (derivador y rangos), H (base inline + generador)
  y la UI. Cada cambio se ubica en su dominio; los cruces son contratos (§9).
- **Toda puerta que entrena pregunta al mismo validador** ANTES de reservar nada. El generador
  **no** es una puerta más laxa: valida la base y cada punto con el mismo `check_run` /
  `check_network` de hoy (§10).
- **Un resultado sin N semillas es una anécdota** ([docs/protocolo.md](protocolo.md)). El
  criterio de «ganador» y la disciplina de medición se fijan en §11.

---

## 2. OAT evita el problema del espacio anidado

Un barrido general de estructura es un **espacio condicional**: el rango auto de `k_center`
depende de `center_out`, que depende de `N` y `c_frac`; los canales por capa dependen de
`n_layers`. Expandir eso exige un producto **anidado**, no el `itertools.product` plano de hoy
([`expand_points`](../src/fv/sweeps/spec.py#L73-L108)).

**OAT lo evita**: al barrer **un solo eje**, todo lo demás está fijo, así que:

- El rango auto del eje barrido se calcula desde **una** geometría (la base actual) — exactamente
  lo que [`build_search_space`](../src/fv/fovea/__init__.py#L142-L157) ya hace.
- No hay ejes ragged simultáneos: si el eje es `n_layers`, se barre `n_layers` con canales fijos;
  los canales por capa se barren **después**, en su propio paso, con `n_layers` ya fijado (§6).

Por eso P1 **no necesita** el expansor anidado. Es la simplificación grande.

---

## 3. Dependencia obligatoria: el builder paramétrico (dominio C)

Sin esto, los ejes de grafo (`n_layers`, canales) no se pueden barrer. Hoy
[`FoveatedRegionalNN`](../src/fv/models/builder.py#L41-L90) hardcodea exactamente dos capas conv
por rama (`conv1`, `conv2`) y `n_layers` es **fantasma** (está en el config y solo alimenta
`stride_range`; el modelo no lo lee).

El builder paramétrico debe honrar `n_layers` y canales por capa **sin cambiar el comportamiento
actual para el caso `n_layers=2`** (compatibilidad con los runs existentes).

### 3.1 Contrato de construcción que el nuevo builder debe cumplir

Para cada rama (centro y periferia), con `L = n_layers` capas conv apiladas:

- **Kernel por capa:** el mismo kernel de la rama en todas sus capas (`k_center` para todas las
  capas del centro; `k_periph` para todas las de la periferia), con `padding = k // 2` (como hoy).
- **Canales por capa:** un vector de longitud `L`. Entrada de la capa 1 = 1 canal (imagen
  compuesta enmascarada); la capa `i` produce `channels[i]`.
- **Stride por capa: [DECIDIDO D-S1]** — *el stride de la rama (`s_center` / `s_periph`) se aplica
  en la **primera** capa; las siguientes van con stride 1.* Preserva el comportamiento actual.
  **Consecuencia:** el submuestreo total es `s`, independiente de `L`, así que **`n_layers` se
  saca de `stride_range`** — el rango de stride no depende de la profundidad.
- **Fusión (`merge`):** igual que hoy — `concat` aplana ambas ramas y concatena; `sum` suma
  mapas alineados (el validador exige strides iguales por rama para `sum`).
- **Cabeza:** sin cambios — `Linear(flat_features, 12)` → `view(-1, 4, 3)` (cabezas de esquina,
  decisión C9). `flat_features` se infiere con el mismo mecanismo de hoy
  ([`_infer_flat_features`](../src/fv/models/builder.py#L74-L80)) tras construir las `L` capas.
- **Profundidad por rama: [DECIDIDO D-S2]** — `n_layers` **único y simétrico** (ambas ramas igual
  de profundas). Profundidad por rama queda como eje futuro.

### 3.2 Canales: regla del vector por defecto

Los canales son el eje que más infla el espacio y su rango `"auto"` continuo pertenece a
Optuna (instructionsNewNN.md §9) — **[DECIDIDO D-C1: aplazado]**, porque el usuario dará rangos
explícitos (U5).

El **vector de canales por defecto** para una `L` dada (parte de los defaults estáticos de §4,
porque cuando se barre otro eje los canales están fijos):

- **[DECIDIDO D-C2]** **Constante `16` en todas las capas** (`channels = [16] * L`). ⚠ Esto
  **cambia** el default de hoy (`[16, 32]`): el modelo derivado por defecto ya no coincide en pesos
  ni forma con `fov-16`. No rompe nada —los runs de ejemplo se descartan (§13) y `fov-16` migra su
  `[16, 32]` explícito tal cual—, pero el test de no-regresión se redacta con `channels=[16,32]`
  **explícito**, no con el default (§12, §13).

### 3.3 Introspección (V1/V2) tras el cambio

Los kernels de la **primera** capa siguen siendo exactos e interpretables (entrada de 1 canal por
rama), así que `kernels()` y `feature_maps()`
([builder.py](../src/fv/models/builder.py#L95-L110)) siguen valiendo para la capa 1. Para capas
`> 1` la introspección es opcional en esta fase. **No romper V1/V2 para `n_layers=2`.**

### 3.4 Config: forma del campo de canales

**[DECIDIDO D-C3]** El config C pasa de `ch1`, `ch2` escalares a **lista explícita**
`channels: [16, 32, 64]` (longitud = `n_layers`), documentado en formatos.md. Se eligió sobre
«base + regla» por transparencia y por encajar con «barrer la dimensión de la capa i» (§6), que
necesita direccionar cada capa individualmente. **Compatibilidad hacia atrás:** se **lee** el
`ch1/ch2` viejo (`ch1=16, ch2=32` → `channels=[16,32]`) y se **escribe** siempre `channels`.

---

## 4. Defaults estáticos y documentados (decisión U2)

Cuando se barre un eje, **todos los demás** toman un valor fijo. Para que el efecto medido de cada
eje sea **comparable entre barridos**, ese contexto debe ser **estable y documentado en un solo
sitio** (organizacion.md avisa contra «definir un número dos veces»).

- **Fuente única:** los defaults viven en **un** lugar canónico
  (recomendado: [`NETWORK_DEFAULTS`](../src/fv/models/builder.py#L27-L32) como fuente, citado
  desde la doc), y todo lo demás lo importa. La doc no repite los valores: los referencia.
- **`N` NO es un default estático.** Se **deriva** del `window_size` del problema (§5). Es el
  único valor que el problema obliga.

Tabla de defaults (valores actuales de `NETWORK_DEFAULTS`, salvo `N`):

| Campo | Default estático | Nota |
|---|---|---|
| `c_frac` | 0.8 | fija junto con `window_size` la geometría; ver §5 |
| `pen_frac` | 0.1 | penetración |
| `d` | 2 | **validar contra `downsample_range`**; si inválido, caer al máximo válido con razón (§10) |
| `n_layers` | 2 | preserva el modelo actual |
| `k_center`, `k_periph` | 3 | kernels mínimos válidos |
| `s_center`, `s_periph` | 1 | sin submuestreo por stride |
| `channels` | `[16]*L` (constante 16) | D-C2; ⚠ cambia el default de hoy `[16,32]` (§3.2 / §3.4) |
| `merge` | `concat` | `sum` exige strides iguales |
| `pool_mode` | `avg` | trazos finos |
| `pad_mode` | `edge` | decisión C10/C11 |

**Contradicción aparente U2 ↔ U3, documentada:** los defaults dan el contexto de **arranque**;
el arrastre del ganador (U3) **muta** ese contexto a medida que se desciende. Por tanto **el
efecto que se mide de cada eje es condicional a los ganadores previos**, no absoluto. Esto es OAT
y es correcto, pero obliga a §7.2 (registrar qué ejes están en ganador vs en default).

---

## 5. El derivador de base desde el problema (dominio G/C)

El contrato ①a fija `center_out(C) == window_size(B)`. El dataset B ya da el tamaño de la fóvea
(`window_size` del manifest, ver [extract.py](../src/fv/windows/extract.py#L153-L162)). A partir
de ahí la geometría se calcula.

### 5.1 Entradas

- `window_size` `W`: del manifest del dataset B seleccionado.
- Fracciones de forma con default estático (§4): `c_frac`, `pen_frac`, `d`.
  - **[DECIDIDO D-G1]** El derivador **expone `d` y `c_frac`**; `pen_frac` queda **fijo** (0.1).
    (OAT igual puede barrer cualquiera como eje de rango explícito — F5.)
- El **eje a barrer** + su **rango** (`"auto"` o lista explícita, U5).
- **Ganadores arrastrados** (§7): valores ya fijados de ejes decididos en pasos previos.

### 5.2 Algoritmo (determinista, sin números mágicos)

1. **Requisito previo:** `W` par (lo exige `round_to_even` en
   [`derive_dims`](../src/fv/fovea/__init__.py#L97-L111)). Si `W` impar, rechazar con razón+arreglo.
2. **Geometría:** con `c_frac` default, `N` es el par tal que
   `round_to_even(N * c_frac) == W` y `(N - W)/2 = periph_out ≥ 1`. Derivar `dims` con
   `derive_dims(N, c_frac, d, pen_frac)`.
   **[DECIDIDO D-G2]** Si varios `N` cumplen, elegir **el `N` más pequeño** (determinista y el más
   barato en cómputo).
   **[DECIDIDO D-G3]** Si **ningún `N` par** cumple ①a con el `c_frac` default (y `periph_out ≥ 1`),
   **aflojar `c_frac`** dentro de una tolerancia hasta que exista `N`, y **registrar el `c_frac`
   efectivo con su razón** (como el paso 4; nunca a ciegas). `W` no se toca — viene de B.
3. **Tunables:** rellenar todos los campos NO barridos con los defaults estáticos (§4), aplicando
   sobre ellos los **ganadores arrastrados** (§7).
4. **Validar `d` y kernels contra la geometría derivada:** un default estático puede ser inválido
   para este `W` (p. ej. `d=2` desborda `downsample_range`, `k_periph=3` excede la banda). Cae al
   valor válido más cercano **con su razón** (§10), nunca a ciegas.
5. **Fijar el espacio:** `space = { <eje>: <rango> }`, con `<rango>` = `"auto"` (rango calculado
   desde `dims`) o la lista explícita del usuario.
6. **Validar la base** con `check_run(manifest, base)` ANTES de escribir la receta. Si hay
   problema, devolver `code/message/hint`, no crear nada.

### 5.3 Salida

Una **receta de recorrido** (§8), lista para el recorrido existente. No se materializa ninguna red
con nombre (P1, U4).

---

## 6. El schedule OAT: orden con dependencias (decisión U6)

Para procesos largos el usuario define **el orden de barrido**. El ejemplo del usuario:
1. número de capas (`n_layers`, fijando lo demás),
2. dimensión de la capa `i` (canales, fijando las otras dimensiones),
3. …

### 6.1 El orden NO es una permutación plana: es un grafo de dependencias

- **Ejes que se desbloquean:** «dimensión de la capa `i`» **no existe** hasta fijar `n_layers`.
  Con `n_layers=3` aparecen 3 sub-pasos de canal (`channels[0]`, `channels[1]`, `channels[2]`);
  con `n_layers=2`, 2. El schedule **se expande según el ganador**.
- **Longitud dinámica:** el número total de pasos **no se conoce hasta correr la cadena** (depende
  de los ganadores). El presupuesto del proceso largo debe contarlo.
- **Rangos que dependen de un ganador previo:** `stride_range` recibe `n_layers`; por eso barrer
  strides **después** de fijar `n_layers` usa el rango correcto.

### 6.2 Representación propuesta del schedule

**[DECIDIDO D-H1]** El schedule es un **objeto de primera clase comiteable, en un dominio nuevo
(`studies/`), que NO ejecuta.** Es un plan (JSON/YAML), coherente con «se versiona la descripción»
(formatos.md §5): describe el orden y las dependencias, y la herramienta guía al usuario paso a
paso (pre-rellena el siguiente recorrido con la base derivada + ganadores arrastrados). **Hay que
registrar en organizacion.md el dominio nuevo `studies/`** y su frontera con H.

Esquema propuesto (ilustrativo, nombres a confirmar):

```yaml
# schedule OAT — plan, no ejecutor
name: estudio-estructura-01
window_dataset: dirty-paragraphs-fast-80px   # B fijo
base_recipe: corta                            # D fija (o barrible en su propio paso)
objective: f1
seeds: 3                                       # N semillas de confirmación (§11)
axes:                                          # orden = orden de barrido
  - axis: n_layers
    range: [1, 2, 3]                           # explícito (U5)
  - axis: channels[i]                          # se expande a channels[0..n_layers-1]
    range: [8, 16, 32, 64]
    depends_on: n_layers                       # desbloqueado por el ganador anterior
  - axis: k_center
    range: auto
  - axis: d
    range: [1, 2, 3]                           # d=1 ⇒ control ~plano (§11.3)
```

### 6.3 Sensibilidad al orden

OAT es **greedy y ciego a interacciones** (los ejes de esta red interactúan: `n_layers`↔strides,
canales↔kernels, `d`↔`k_periph`). El schedule debería:

- ordenar **geometría/estructura gruesa primero** (acota rangos de los finos),
- y ofrecer una **segunda pasada de confirmación** opcional: re-barrer 1–2 ejes iniciales con los
  ganadores finales fijados. Si el ganador no se mueve, la interacción era débil. Seguro barato.

---

## 7. Arrastre del ganador (decisión U3)

### 7.1 Mecanismo

Cada punto de un recorrido es un **run de primera clase** cuyo `config.json` guarda el config C
completo en su procedencia (ver cualquier `runs/*/config.json`). «Arrastrar el ganador» =

1. leer el **punto ganador** del ranking del recorrido terminado
   ([`sweep_trials`](../src/fv/sweeps/runner.py#L128-L158)),
2. tomar el valor del eje barrido en ese punto,
3. **fijarlo en la base derivada** del siguiente eje (paso 3 de §5.2),
4. generar la receta del siguiente eje.

Es un `clona-y-varía`: base = ganador anterior; espacio = siguiente eje. Sin tecleo manual.

### 7.2 Procedencia: qué está en ganador vs en default

**Obligatorio** para reconstruir la cadena: la receta generada y la procedencia de cada run deben
registrar, por campo de la base, **su origen**: `default` | `winner(from=<recorrido/paso>)` |
`user`. Sin esto, mirando un `base_network_value` inline no se puede saber por qué `n_layers=3`
estaba ahí (¿elegido, arrastrado o default). Es barato y salva la auditoría del estudio.

### 7.3 Criterio de «ganador» — coste/calidad con significancia

El usuario lo enunció así: *«una nn de 3 capas más costosa que no supera significativamente a la
de 2 capas»* debe **perder**. Es un criterio **coste-ajustado con umbral**, no «mejor objetivo a
secas». Ingredientes ya disponibles:

- **Calidad:** el objetivo del ranking (`f1` / `pos_err_px`), de las métricas de val.
- **Coste:** `seconds_per_epoch` (de `summary.json`) y `num_params`
  (de [`network_trace`](../src/fv/models/builder.py#L117-L132)).
- **Significancia:** exige **N semillas** por punto (§11); sin varias semillas «no supera
  significativamente» no está definido.

**[DECIDIDO D-W1]** La regla, **sugerida (el usuario confirma)**: la herramienta propone *el valor
más barato cuya calidad no sea peor que la del mejor por más de un margen `δ`* (estilo «1-SE /
Pareto»), con `δ` y la métrica de coste (tiempo vs params) **a la vista**, y el usuario aprieta el
gatillo antes de arrastrar el ganador (coherente con un schedule que guía, no ejecuta). `δ` y la
métrica de coste son entradas del estudio; la confirmación exige las N semillas en **ambos**
candidatos de la frontera (§11.1), no solo el provisional.

---

## 8. La receta de recorrido generada y la extensión de H (base inline)

### 8.1 Estado actual de H que se reutiliza

El recorrido de hoy ya hace «una base + espacio de un eje»: ver
[`create_sweep`](../src/fv/api/app.py#L417-L449), [`prepare_sweep`](../src/fv/sweeps/runner.py#L20-L40)
y el formato de spec en cualquier `sweeps/*/spec.json`. Campos actuales:
`window_dataset, base_network, base_network_value, base_recipe, base_recipe_value, space,
strategy, objective, budget{points,epochs}, device, seed, name, points, discarded`.

### 8.2 Extensión mínima: aceptar base inline

Hoy `create_sweep` **resuelve un nombre**: `net = nstore.get(body["base_network"])`. Para base
inline (U4) hay que permitir **pasar el config directamente** en vez de un nombre.

**[DECIDIDO D-H2]** Forma de la extensión:

- `base_network` (nombre) pasa a **opcional**. Si falta, se usa `base_network_value` tal cual
  (config inline ya resuelto por el derivador).
- Añadir una **etiqueta sintética** legible para la UI (el filtro por nombre de red muere sin
  nombre — ver [Sweeps.tsx](../web/src/screens/Sweeps.tsx#L123)): `base_label: "ws16-p2-d2-L2"`
  **con separador guion** (seguro como filtro y nombre de fichero; el middot se descartó), derivada
  de la geometría, para filtrar y leer de un vistazo.
- Añadir el bloque de **derivación** (procedencia del generador), para reconstruir la cadena:

```jsonc
// añadido propuesto al spec del recorrido
{
  "base_network": null,                 // inline: sin nombre
  "base_label": "ws16-p2-d2-L2",        // etiqueta sintética para UI/filtros
  "base_network_value": { /* config C completo derivado */ },
  "derivation": {                        // §5, §7.2 — cómo se llegó a esta base
    "window_size": 16,
    "fractions": { "c_frac": 0.8, "pen_frac": 0.1, "d": 2 },
    "field_origin": {                    // por campo: default | winner | user
      "n_layers": { "origin": "winner", "from": "estudio-01/paso-1" },
      "k_center": { "origin": "default" },
      "channels": { "origin": "user" }
    }
  },
  "space": { "k_center": [3, 5, 7] }     // el ÚNICO eje barrido (U5)
}
```

Todo lo demás del recorrido (validación previa, expansión, descarte con razón, ejecución
secuencial worker=1, resume contando runs terminados, ranking, borrado en cascada) **se reutiliza
sin cambios**.

### 8.3 Contrato ⑨ preservado

El bloque ⑨ ya vive en [`check_sweep`](../src/fv/sweeps/spec.py#L50-L55): si el objetivo es `loss`
y el espacio barre un peso de la pérdida, se rechaza. El generador no lo elude: pasa por el mismo
`check_sweep`.

---

## 9. Ubicación por dominios y contratos tocados

| Pieza | Dominio | Archivos previsibles | Contrato |
|---|---|---|---|
| Builder paramétrico (`n_layers`, canales) | **C** | `src/fv/models/builder.py`, `configs/networks/*` | — |
| Rango auto de `n_layers`/canales | **G** | `src/fv/fovea/__init__.py` | (calculado, no a mano) |
| Derivador de base desde `window_size` | **G/C** | nuevo módulo (p. ej. `src/fv/models/derive.py`) | ①a (`center_out==window_size`) |
| Base inline en el recorrido | **H** | `src/fv/sweeps/spec.py`, `runner.py`, `api/app.py` | ③ (nombre/valor), ⑧ (B fijo), ⑨ |
| Generador de receta (P1) | **H** | nuevo (p. ej. `src/fv/sweeps/generate.py` + CLI `fv-oat`) | — |
| Schedule OAT (plan) | **dominio nuevo `studies/` (I)** | plan comiteable, guía al recorrido | ⑫ (I ↔ H) |
| Arrastre del ganador | **I/H** | `src/fv/sweeps/runner.py` (lee ganador), generador del estudio | ⑫ |
| Etiqueta sintética + filtros | **UI** | `web/src/screens/Sweeps.tsx` + pantalla nueva del estudio | — |

**Ya escrito en [organizacion.md](organizacion.md)** (2026-07-23, prerequisito de §9 cumplido):
(a) H con **base inline** (§1-H) y su cobertura por el contrato **③**; (b) el **dominio nuevo `I —
Estudio (`studies/`)`** (§1-I) y su frontera con H, el contrato **⑫**. El formato en disco está en
[formatos.md](formatos.md) §4.7 (`plan.json`/`progress.json`) y §4.3 (`channels`).

**Relación con decisiones abiertas existentes** ([decisiones.md](decisiones.md)):
- **F5** (¿`c_frac`/`pen_frac` barribles?): OAT los alcanza como ejes con **rango explícito**, un
  eje a la vez, con descarte declarado — encaja con la recomendación de F5 («en grid aparte»).
- **§9 de instructionsNewNN.md / D-C1** (rango auto de canales, Optuna): **aplazable**, porque U5
  da rangos explícitos.
- **F3/F4** (`merge`/`pool_mode` como ejes): ya son campos barribles; OAT los cubre sin trabajo
  extra.

---

## 10. Validación: la misma puerta, antes de escribir

El mayor riesgo de un generador es **materializar un artefacto inválido**. Reglas:

1. **La base derivada pasa `check_run(manifest, base)`** (compatibilidad ①/② + medibilidad) ANTES
   de escribir la receta. Si falla, `code/message/hint`, sin crear nada.
2. **Un default estático inválido para este `W`** (p. ej. `d`, `k_periph`) **cae al valor válido
   más cercano con su razón** — nunca a ciegas (§5.2 paso 4).
3. **Cada punto expandido** (base + valor del eje) pasa `check_network`; los geométricamente
   inválidos van a `discarded` **con su razón** — mecanismo que ya existe
   ([expand_points](../src/fv/sweeps/spec.py#L96-L108)). La UI ya muestra el conteo de descartes.
4. **El generador reutiliza `check_sweep`** para el bloque ⑨ y la forma del espacio.

El generador **no es una puerta más laxa**: usa exactamente los validadores de hoy.

---

## 11. Disciplina de medición

### 11.1 N semillas (dos niveles)

Un ganador de una sola semilla es una anécdota (protocolo.md). Pero N semillas × varios ejes ×
varios valores infla el proceso largo **en una máquina que hiberna y hace throttling** (ver
CLAUDE.md, «Observaciones de esta máquina»). Política de **dos niveles**:

- **Sondeo rápido:** 1 semilla sobre un dataset B **pequeño** (ya seleccionable — U del enunciado
  original) para **rankear y descartar** dentro de cada eje.
- **Confirmación:** N semillas (default 3) para **confirmar el ganador** antes de arrastrarlo (§7)
  — corriéndolas en **ambos candidatos de la frontera** coste-calidad, no solo el provisional.

**[DECIDIDO D-M1]** El `N` de confirmación es **configurable por estudio, default 3** (sondeo 1 +
confirmación 3). Multiplica el schedule, por eso es perilla: pocas en CPU corta, más en GPU larga.
La confirmación corre las N semillas en **ambos candidatos de la frontera** coste-calidad, no solo
en el provisional — si no, «no supera significativamente» no está definido.

### 11.2 Dataset pequeño: válido para rankear, no para concluir

Cualquier `window_size`/tamaño de B es seleccionable (ya disponible). Pero el tamaño de muestra
efectivo lo dan las **imágenes**, no las ventanas (protocolo.md). Un dataset chico sirve para
**candidatos y ranking rápido**; la decisión final necesita imágenes suficientes.

### 11.3 El control plano debe ser alcanzable

La pregunta que justifica toda la red (protocolo.md §6: ¿fóvea+periferia gana a una CNN plana de
coste equivalente?) **no se mide sola**. Debe ser un eje del schedule: p. ej. `d ∈ {1, 2, 3}`
(con `d=1` ≈ periferia sin submuestreo) o un toggle de periferia. Si el schedule nunca lo escanea,
se optimiza la fóvea sin comprobar que la fóvea sirve.

---

## 12. Expectativas de test (tests.md)

«Un contrato sin test es un comentario». Al codificar, como mínimo:

- **Builder paramétrico:** `n_layers=2` con `channels=[16,32]` produce un modelo **idéntico**
  (misma forma, mismo `num_params`) al actual — prueba de no-regresión. Y `n_layers=3` construye y
  hace forward con salida `(-1, 4, 3)`.
- **Derivador:** para un `window_size` dado, `derive_dims` del config derivado cumple
  `center_out == window_size` (contrato ①a). `W` impar se rechaza con razón.
- **Default inválido:** un `W` que invalida `d=2` produce una base con `d` corregido y **razón
  registrada**, no un descarte total.
- **Generador:** la receta generada, pasada por `check_sweep`/`prepare_sweep`, produce puntos
  válidos y descartes con razón; el bloque ⑨ se dispara si el objetivo es `loss` y el eje es un
  peso de la pérdida.
- **Base inline:** un recorrido sin `base_network` (solo `base_network_value`) se prepara, corre y
  rankea igual que uno con red nombrada — mismo gate.
- **Arrastre:** dado un recorrido terminado, el generador del siguiente eje fija correctamente el
  ganador y registra `field_origin`.
- **CLI ↔ API** bit-idénticos con la misma semilla (como el resto del sistema).

---

## 13. Decisiones cerradas (2026-07-23)

Resueltas con el usuario. **Estos valores mandan** sobre cualquier recomendación provisional del
cuerpo. Lo no listado aquí se sigue preguntando; no se inventa.

| ID | Tema | **Decisión** |
|---|---|---|
| D-S1 | Dónde caen los strides con `L` capas | **Stride de rama en la 1ª capa, resto stride 1.** Consecuencia: **`n_layers` se saca de `stride_range`** (el submuestreo total es `s`, independiente de `L`) |
| D-S2 | `n_layers` único o por rama | **Único y simétrico** (ambas ramas igual de profundas). Profundidad por rama = eje futuro |
| D-C1 | Rango **auto** de canales (Optuna, §9) | **Aplazado** (U5 da rangos explícitos) |
| D-C2 | Vector de canales **por defecto** para `L` | **Constante `16`** en todas las capas (`[16]*L`). ⚠ Cambia el default de hoy (`[16,32]`); ver nota de no-regresión abajo |
| D-C3 | Forma del campo de canales en el config | **Lista explícita `channels: [...]`**; lee `ch1/ch2` viejo, escribe `channels` |
| D-G1 | Qué fracciones expone el derivador | **Exponer `d` y `c_frac`; `pen_frac` fijo** (0.1) |
| D-G2 | Desempate si varios `N` cumplen ①a | **El `N` más pequeño** (determinista, el más barato) |
| D-G3 | **(nuevo)** Ningún `N` par factible para ese `W` | **Aflojar `c_frac`** dentro de tolerancia hasta que exista `N`, **registrando el `c_frac` efectivo con su razón** (nunca a ciegas). `W` no se toca (viene de B) |
| D-H1 | Schedule: objeto nuevo o metadatos | **Objeto de primera clase comiteable, dominio nuevo (`studies/`), NO ejecutor** — describe orden y dependencias, guía paso a paso |
| D-H2 | Forma exacta de la base inline en el spec | **Forma completa, separador guion**: `base_network=null` + `base_label:"ws16-p2-d2-L2"` + bloque `derivation{window_size, fractions, field_origin}` |
| D-W1 | Regla de «ganador» coste/calidad | **Sugerida, el usuario confirma.** La herramienta propone «el más barato cuya calidad ≥ best−`δ`» (con `δ` y métrica de coste a la vista); el usuario aprieta el gatillo antes de arrastrar |
| D-M1 | `N` de semillas de confirmación | **Configurable por estudio, default 3** (sondeo 1 + confirmación 3). La confirmación corre las N semillas en **ambos candidatos de la frontera**, no solo el provisional |

**Checkpoints (hueco de la review):** los runs de ejemplo (`fov-run-1/2`, `cli-run-1`) **se
descartan** — eran pruebas. El builder paramétrico nace sin deuda de migración de `state_dict`;
no se escribe código de compatibilidad de pesos.

**No-regresión con el nuevo default (D-C2):** como el default pasa a `[16,16]`, el test de §12 se
redacta «con `channels=[16,32]` **dado explícitamente** se reproduce la forma/param-count de hoy»,
no «el default reproduce hoy». El `fov-16` existente migra su `[16,32]` explícito tal cual.

---

## 14. Secuencia de construcción recomendada

Las piezas son independientes y entregables por separado:

1. **Builder paramétrico (C)** — prerequisito de todo; sin él los ejes de grafo no existen.
   No-regresión para `n_layers=2` es el criterio de aceptación.
2. **Derivador de base + defaults validados (G/C)** — elimina la autoría de los 14 campos; alivia
   ya para los ejes de geometría que **ya** construyen.
3. **Base inline en H + generador P1 (H)** — el «script que hace lo que tú harías».
4. **Arrastre del ganador (H)** — cierra la cadena OAT sin tecleo.
5. **Schedule OAT (plan) + UI** — para procesos largos ordenados.
