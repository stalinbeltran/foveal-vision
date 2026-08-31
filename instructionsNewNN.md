# CLAUDE.md — Red neuronal con muestreo foveado y ramas por región (totalmente parametrizada)

Documento de contexto para que Claude retome este proyecto sin perder las
decisiones de diseño. Describe una red neuronal (no una CNN pura) que procesa
una entrada con **resolución variable por región** (centro en alta resolución,
periferia en baja resolución pero con mayor campo visual) y con **kernels
independientes por región** que se **suman** en las zonas de solape.

**Principio rector:** *todo dato es un parámetro*. Nada está hardcodeado a
20×20 ni a downsample=2. Las dimensiones concretas y los **rangos de búsqueda**
(kernels, strides, reducción del borde) se **calculan** a partir de unas pocas
longitudes en **píxeles reales de la imagen**.

> ⚠ **2026-08-25 — reparametrización.** Este documento decía «se calculan en
> función de un lado de entrada `N` y unas pocas fracciones». Ya no: **`N` se
> deriva**, y lo que se declara son longitudes. El porqué está en §2.1. **Ninguna
> red cambia**: es un cambio de ortografía, verificado bit a bit — un checkpoint
> guardado con la ortografía vieja carga `strict=True` en una red construida con
> la nueva y las dos dan la misma salida.

---

## 1. Idea central

Inspiración biológica: visión foveada. El centro de la entrada se ve con
detalle (sin reducir); la periferia se ve "borrosa" pero abarca más área del
entorno. Dos objetivos combinados, que son **etapas separadas**:

1. **Muestreo foveado** (cómo se *arma* la entrada): el centro va a resolución
   completa; la periferia se toma de una región más amplia de la imagen
   original y se reduce (downsampling) antes de colocarla en el borde.
2. **Ramas convolucionales por región** (cómo se *procesa* esa entrada): una
   rama de kernels para el centro y otra para la periferia; en la banda de
   penetración **ambas contribuyen y se suman** (la posición del píxel importa
   aunque pertenezca a dos regiones).

> El muestreo foveado es **excluyente** por naturaleza (un píxel de la original
> va al centro *o* al borde). El solapamiento **contributivo** vive en el
> procesamiento por las dos ramas, *después* de construida la entrada. No se
> contradicen.

---

## 2. Parámetros fundamentales (todo lo demás se deriva)

Estos son los únicos grados de libertad reales, y **todos son longitudes en
píxeles de la imagen original** salvo `border_reduce`, que pertenece al *método*
de reducción y no es una longitud:

```python
fovea_px           # lado de la fovea. ES la ventana etiquetada de B (contrato (1)a)
border_px          # grosor del borde difuso, por lado, en px REALES
border_reduce      # px reales que se condensan en UNA celda de borde (antes `d`)
overlap_fovea_px   # px de FOVEA que ve tambien la rama del borde
overlap_border_px  # px de BORDE que ve tambien la rama de la fovea
n_layers           # n de capas conv por rama (para acotar strides; antes 2)
```

Dimensiones derivadas:

```python
border_cells         = border_px // border_reduce      # celdas de anillo, por lado
N                    = fovea_px + 2 * border_cells     # <- DERIVADO: la entrada de la NN
overlap_border_cells = overlap_border_px // border_reduce
center_band          = fovea_px + 2 * overlap_border_cells   # banda del kernel central
periph_band          = border_cells + overlap_fovea_px       # banda del kernel externo
original_size        = fovea_px + 2 * border_px        # imagen original necesaria
```

Validaciones de consistencia obligatorias:

```python
assert fovea_px % 2 == 0                        # el borde reparte simetrico
assert border_px % border_reduce == 0           # el borde cae en celdas enteras
assert overlap_border_px % border_reduce == 0   # el solape, tambien
assert overlap_fovea_px < fovea_px // 2         # el nucleo de la fovea no desaparece
assert overlap_border_px < border_px            # el borde exclusivo tampoco
```

### 2.1 Por qué se declara así (y no con `N` y fracciones)

La ortografía anterior era `N`, `c_frac`, `d`, `pen_frac`, y tenía tres
problemas que se pagaban:

1. **La fóvea estaba repartida entre dos números.** `center_out =
   round_to_even(N·c_frac)`, y el contrato ①a la ata al `window_size` de B. Así
   que **ni `N` ni `c_frac` podían ser eje** — mover cualquiera *solo* rompía
   ①a. Para pasar de 2 a 4 celdas de anillo había que mover **los dos a la vez**,
   y el motor es OAT (un eje cada vez): esa parte del espacio era legal pero
   **inalcanzable por barrido**. Hoy `border_px` es un eje de primera clase.
2. **«Cuánto contexto» y «cómo se comprime» eran el mismo mando.** El contexto
   real era `periph_out · d`, un producto: `d` movía las dos cosas a la vez. Hoy
   `border_px` es la longitud y `border_reduce` sólo el método, así que se pueden
   separar dos preguntas que son experimentos distintos — *¿ayuda más área?* y
   *¿ayuda verla con más resolución?*. Y si mañana el borde se reduce de otra
   forma (no un promedio de bloques), **la definición del borde no se toca**:
   cambia el método.
3. **El solape sólo existía hacia dentro, y nunca podía ser 0.** `penetration =
   max(1, round(N·pen_frac))` describía cuánto entra el borde en la fóvea; que la
   fóvea saliera sobre el borde **no era expresable**, y tampoco lo era la
   ausencia de solape. Hoy son dos parámetros independientes y barribles, y
   `overlap_fovea_px = 0` es el **control** de la elección de solape contributivo
   de §7.

Y una consecuencia práctica: `derive_geometry` —la función que buscaba el menor
`N` par cuya fóvea cayera exactamente en `W`, aflojando `c_frac` con una razón
cuando no lo encontraba— **desaparece**. Existía sólo porque la geometría se
declaraba desde el lado equivocado.

### 2.2 Coste: `border_px` no es gratis, `border_reduce` casi sí

`N` crece con el borde **en celdas**, y la cabeza es `Linear(2·C·N², 12)` — el
97 % de los parámetros (medido, plan-40h §2). Con la fóvea de 16 y C=16:

| `border_px` | `border_reduce` | celdas | N | params de la cabeza |
|---:|---:|---:|---:|---:|
| 4 (vigente) | 2 | 2 | 20 | 153.612 |
| 8 | 4 | 2 | 20 | 153.612 |
| 8 | 2 | 4 | 24 | 221.196 (+44 %) |
| 16 | 2 | 8 | 32 | 393.228 (+156 %) |

Es decir: **más área con la misma resolución es casi gratis** (sube
`border_reduce` a la par que `border_px` y `N` no se mueve); **más resolución
sobre la misma área cuesta N²**. Un estudio que barra `border_px` con
`border_reduce` fijo mueve las dos cosas, y hay que escribirlo en el plan o el
confound se lee como señal.

Y hay un techo físico que G no conoce: `original_size = fovea_px + 2·border_px`
tiene que caber en la imagen. Sobre las de 60×80 px de este proyecto, con ventana
16 y stride 8, la fracción del anillo que es **relleno replicado**
(`pad_mode: edge`) y no contexto sube así — medido:

| `border_px` | recorte | % ventanas con relleno | % del anillo que es relleno |
|---:|---:|---:|---:|
| 4 | 24×24 | 35 % | 11,5 % |
| 8 | 32×32 | 48 % | 15,3 % |
| 12 | 40×40 | 72 % | 21,4 % |
| 16 | 48×48 | 82 % | 26,4 % |
| 22 | 60×60 | 100 % | 34,9 % |

Pasados los ~8–12 px se mide el relleno, no la comprensión de la imagen. La
**máscara de cobertura** que devuelve `build_view` es exactamente lo que lo
cuantifica, celda por celda.

---

## 3. Rangos de búsqueda como FUNCIONES (no constantes)

Las reglas geométricas deducidas se vuelven fórmulas que escalan con `N`.

### 3.1 Kernels (impares, sin exceder su región)

```python
def kernel_range(region_size):
    """Kernels impares desde 3 hasta ~region/2, sin desbordar la región."""
    k_max = region_size // 2
    if k_max % 2 == 0:
        k_max -= 1                    # forzar impar
    return [k for k in range(3, max(3, k_max) + 1, 2)]

# centro=16 → [3,5,7];  centro=32 → [3,5,7,9,11,13,15]
k_center_options = kernel_range(center_band)   # la fovea + lo que sale sobre el borde
k_periph_options = kernel_range(periph_band)   # banda fina → [3] o [3,5]
```

### 3.2 Strides (acotados por el tamaño de cada región)

Heurística: el **producto acumulado de strides no debe colapsar la región**
(≤ región/4).

```python
def stride_range(region_size, n_layers=2):
    import math
    max_cumulative = max(1, region_size // 4)
    s_max = max(1, int(round(max_cumulative ** (1 / n_layers))))
    return list(range(1, s_max + 1))

s_center_options = stride_range(center_band, n_layers)  # centro 16 → [1,2]
s_periph_options = stride_range(periph_band, n_layers)  # banda fina → [1]
```

La periferia, por delgada, casi siempre devuelve `[1]` — ahora emerge del
cálculo en vez de ser una regla escrita a mano.

### 3.3 Padding (derivado, NUNCA buscado)

```python
padding = k // 2      # por rama; conserva 20×20 espacial con stride=1
```
Fuerza `kernel_size` impar (padding entero). Un kernel par desalinea máscaras.

### 3.4 El borde y su reducción (dos rangos, no uno)

Antes había un solo rango, `downsample_range`, y estaba acotado por «que la
original no explote» — porque `d` era también el mando del área. Ya no lo es: el
recorte real es `fovea_px + 2·border_px` y **no se mueve con la reducción**. Así
que son dos rangos independientes:

```python
def reduce_range(border_px):
    """Los factores que un borde de este tamaño admite: sus divisores."""
    if border_px <= 0:
        return [1]
    return [r for r in range(1, border_px + 1) if border_px % r == 0]

def border_range(fovea_px, border_reduce=1, max_original=None):
    """Anchos de borde (px) sobre la rejilla de celdas, con el recorte acotado.
    max_original por defecto = 3*fovea_px (un borde tan ancho como la fovea).
    Es una cota de BUSQUEDA: el limite duro es la imagen, y solo B sabe cuanto
    mide."""
    r = max(1, border_reduce)
    cap = max_original or 3 * fovea_px
    return [b for b in range(r, max(0, (cap - fovea_px) // 2) + 1, r)]
```

### 3.5 Los dos solapes (uno por región)

```python
def overlap_fovea_range(fovea_px):
    """Cuanto entra la rama del borde en la fovea: 0 .. fovea/2 - 1.
    El 0 es legal y es NUEVO: hace las dos ramas disjuntas, que es el control
    de la eleccion de solape contributivo de la seccion 7."""
    return list(range(0, max(1, fovea_px // 2)))

def overlap_border_range(border_px, border_reduce=1):
    """Cuanto sale la rama de la fovea sobre el borde: 0 .. border - reduce.
    Acotado una celda antes del borde entero: espejo de la regla de la fovea,
    la rama del borde conserva algo exclusivo."""
    r = max(1, border_reduce)
    return [b for b in range(0, max(0, border_px - r) + 1, r)]
```

### 3.6 Ensamblado del espacio de búsqueda

```python
def build_search_space(geom, n_layers=2, max_original=None):
    dims = derive_dims(normalize_geometry(geom))
    return {
        "k_center":          kernel_range(dims.center_band),
        "k_periph":          kernel_range(dims.periph_band),
        "s_center":          stride_range(dims.center_band, n_layers),
        "s_periph":          stride_range(dims.periph_band, n_layers),
        "border_px":         border_range(dims.fovea_px, dims.border_reduce, max_original),
        "border_reduce":     reduce_range(dims.border_px),
        "overlap_fovea_px":  overlap_fovea_range(dims.fovea_px),
        "overlap_border_px": overlap_border_range(dims.border_px, dims.border_reduce),
        # derivados no-buscables:
        "_fovea_px": dims.fovea_px,
        "_border_cells": dims.border_cells,
        "_N": dims.N,
    }
```

Las longitudes definen **todo**: dimensiones y los siete rangos buscables. Una
fóvea de 28 (MNIST), 32 o 64 recalcula los rangos sola — y ahora también los
recalcula cambiar el borde, que antes no era una pregunta que se pudiera hacer.

⚠ **`k_center` sale de `center_band`, no de la fóvea.** Con
`overlap_border_px > 0` la rama central ve más que la fóvea, y su kernel puede
crecer con ella. Con solape 0 los dos coinciden, así que nada cambia respecto de
lo que se midió antes.

---

## 4. Geometría del muestreo foveado (construcción de la entrada)

Descomposición de la imagen original de `original_size` (ejemplo con los
valores vigentes: fóvea 16 px, borde 4 px, `border_reduce`=2 → N=20, original
24×24):

```
Imagen original: original_size × original_size       (ej. 24×24)
├── Fovea fovea_px×fovea_px  → se toma TAL CUAL      → ocupa fovea_px celdas centrales
│   (px border_px .. border_px+fovea_px-1)
└── Borde de border_px px    → se reduce /border_reduce
                             → border_cells celdas   → ocupa el anillo del input

Resultado: entrada compuesta N×N  →  border_cells + fovea_px + border_cells = N
```

Correspondencia de coordenadas (ejemplo: fóvea 16, borde 4 px, reduce 2 → N=20,
original 24):

| Zona          | Original (24px) | Reducción | Input (20px)   |
|---------------|-----------------|-----------|----------------|
| Borde difuso  | px 0–3 / 20–23  | ÷2        | px 0–1 / 18–19 |
| Fóvea         | px 4–19         | ×1        | px 2–17        |

Los mismos 4 px de borde con `border_reduce=1` darían **4 celdas** por lado
(N=24) y el mismo recorte de 24×24: **misma información, más resolución, cabeza
más grande** (§2.2). Ésa es la separación que la ortografía nueva permite
expresar y la vieja no.

---

## 5. Construcción de la entrada (código de referencia)

El "**lienzo con relleno cero**" es un tensor `N×N` inicializado en ceros que
sirve de superficie donde se "pegan" las piezas. Aquí la asignación es
**excluyente** (cada píxel del input tiene un único origen: centro *o* borde),
lo cual es correcto para el muestreo foveado.

> ⚠ **Código de referencia del diseño original, no el implementado.** Conserva
> los nombres viejos (`center_out`, `periph_out`, `d`) y el lienzo de ceros. La
> implementación viva es `fv.fovea.build_foveated_input` / `build_view`: usa la
> ortografía en px, hace el muestreo con `reduceat` (no con un doble bucle) y
> rellena con `pad_mode: edge` en vez de ceros — un cero significa «no hay
> tinta» y enseñaría una regla falsa (decisión C11). La traducción es directa:
> `center_out`→`fovea_px`, `periph_out`→`border_cells`, `periph_out*d`→`border_px`,
> `d`→`border_reduce`.

```python
import torch
import torch.nn.functional as F

def build_foveated_input(img, center_out, periph_out, d):
    """
    img: tensor (B, C, original_size, original_size)
    original_size = center_out + 2*periph_out*d
    return: entrada compuesta (B, C, N, N) con N = center_out + 2*periph_out
    """
    B, C, H, W = img.shape
    m = periph_out * d                         # margen real en la original
    N = center_out + 2 * periph_out

    # 1. Centro: recorte directo, SIN reducir
    center = img[:, :, m:m+center_out, m:m+center_out]

    # 2. Periferia: reducir la imagen COMPLETA; su borde es el anillo reducido
    periph_full = F.avg_pool2d(img, kernel_size=d)     # (B,C, original/d, original/d)

    # 3. Lienzo N×N (relleno cero)
    out = torch.zeros(B, C, N, N, device=img.device, dtype=img.dtype)

    # 3a. Centro sin tocar, en el medio (offset = periph_out)
    o = periph_out
    out[:, :, o:o+center_out, o:o+center_out] = center

    # 3b. Anillo reducido de periph_out px alrededor (bordes de periph_full)
    out[:, :, :o,     :]  = periph_full[:, :, :o,     :]
    out[:, :, -o:,    :]  = periph_full[:, :, -o:,    :]
    out[:, :, o:-o,  :o]  = periph_full[:, :, o:-o,  :o]
    out[:, :, o:-o, -o:]  = periph_full[:, :, o:-o, -o:]

    return out
```

**Nota (downsampling):** para trazos finos tipo EMNIST, `avg_pool2d` puede
difuminar demasiado la periferia. Evaluar `max_pool2d`. Decisión abierta.

---

## 6. Ramas por región y máscaras contributivas (procesamiento)

Sobre la entrada `N×N` ya construida actúan **dos ramas convolucionales
independientes**. En la **zona de penetración ambas contribuyen y se suman**
(no se sobrescriben). Se implementa con **máscaras solapadas**.

El solape tiene **dos lados y son independientes** (reparametrización
2026-08-25). Antes sólo existía el primero, y con suelo de 1 px:

- `overlap_fovea_px` — cuánto entra la rama **del borde** hacia la fóvea.
- `overlap_border_px` — cuánto sale la rama **de la fóvea** sobre el borde.

```
Zonas dentro de la entrada N×N
(ejemplo: border_cells=2, overlap_fovea_px=2, overlap_border_px=0):
├── Anillo externo:  px 0-1 y 18-19   -> solo kernel periferico
├── Zona compartida: px 2-3 y 16-17   -> AMBOS kernels (se suman)
└── Nucleo central:  px 4-15          -> solo kernel central

Con overlap_border_px=2 (una celda, si border_reduce=2) la rama central
crece hacia fuera y la celda 1 / 18 pasa a ser tambien compartida.
Con overlap_fovea_px=0 no hay zona compartida: las ramas son DISJUNTAS
(el control de la eleccion de la seccion 8, que antes no era expresable).
```

```python
def build_masks(dims):
    N, po = dims.N, dims.border_cells
    pen, ob = dims.overlap_fovea_px, dims.overlap_border_cells
    center_mask = torch.zeros(1, 1, N, N)
    periph_mask = torch.zeros(1, 1, N, N)

    # Rama de la fovea: la fovea, CRECIDA hacia fuera ob celdas
    lo, hi = po - ob, N - po + ob
    center_mask[..., lo:hi, lo:hi] = 1

    # Rama del borde: todo menos el nucleo exclusivo de la fovea
    inner_lo = po + pen
    inner_hi = N - po - pen
    periph_mask[...] = 1
    periph_mask[..., inner_lo:inner_hi, inner_lo:inner_hi] = 0

    # Donde AMBAS valen 1 -> se suman.
    return center_mask, periph_mask
```

```python
import torch.nn as nn

class FoveatedRegionalNN(nn.Module):
    # referencia del diseño original; el builder vivo (fv.models.builder) recibe
    # el config entero y deriva la geometría con fv.fovea.dims_of
    def __init__(self, N, center_out, periph_out, penetration,
                 k_center=3, k_periph=3, s_center=1, s_periph=1,
                 ch1=32, ch2=64, num_classes=10):
        super().__init__()
        pc, pp = k_center // 2, k_periph // 2

        self.center_conv1 = nn.Conv2d(1,   ch1, k_center, stride=s_center, padding=pc)
        self.center_conv2 = nn.Conv2d(ch1, ch2, k_center, stride=1,        padding=pc)
        self.periph_conv1 = nn.Conv2d(1,   ch1, k_periph, stride=s_periph, padding=pp)
        self.periph_conv2 = nn.Conv2d(ch1, ch2, k_periph, stride=1,        padding=pp)

        cm, pm = build_masks(N, periph_out, center_out, penetration)
        self.register_buffer('center_mask', cm)
        self.register_buffer('periph_mask', pm)
        self.classifier = nn.Linear(ch2, num_classes)

    def forward(self, x):                      # x: (B,1,N,N) ya foveado
        c = self.center_conv2(F.relu(self.center_conv1(x)))
        p = self.periph_conv2(F.relu(self.periph_conv1(x)))
        # Si s_center != s_periph, c y p difieren en tamaño: ver §7 (suma vs concat).
        c = c * F.interpolate(self.center_mask, size=c.shape[-2:], mode='nearest')
        p = p * F.interpolate(self.periph_mask, size=p.shape[-2:], mode='nearest')
        feat = c + p                           # suma contributiva (si alinean)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return self.classifier(feat)
```

> **Nota — builder paramétrico (decidido 2026-07-23).** Este `__init__` es **referencia
> ilustrativa**: fija dos capas y `ch1/ch2` escalares. El builder real las **parametriza** y ese
> código **no manda** sobre el número de capas ni la forma de los canales — igual que la cabeza
> (C9 sustituye `classifier` + `adaptive_avg_pool2d`, que destruye la posición, por
> `4×[exists,x,y]`). Lo que manda: [docs/barrido-por-ejes.md](docs/barrido-por-ejes.md) §3 y §13.
> En concreto: `channels` es una **lista por capa** de longitud `n_layers` (D-C3, lee `ch1/ch2`
> viejo), el stride de rama se aplica **solo en la 1ª capa** (D-S1, por lo que `n_layers` **sale**
> de `stride_range`), y `n_layers` es **único y simétrico** para ambas ramas (D-S2).

---

## 6bis. Entradas de borde: lo que la vista NO puede decir (`edge_inputs`)

**Añadido el 2026-08-31.** Cuatro números por ventana que van **directos a la
cabeza**, sin pasar por ninguna convolución.

### El problema que resuelve

`pad_mode: edge` replica la fila/columna del borde al salirse de la imagen
(decisión C11: nunca ceros a secas, porque cero significa «no hay tinta» y
enseña una regla falsa). **Esa réplica es, por construcción, indistinguible de
imagen real que casualmente se parece a más de lo mismo.** Consecuencia:

| Lo que hay en la imagen | Lo que la red ve | Lo que la etiqueta dice |
|---|---|---|
| párrafo **pegado** al borde superior | banda replicada arriba | TL/TR **existen** ahí |
| párrafo **cortado** por el borde de la vista | banda de contexto arriba | TL/TR están **más arriba**, fuera |

Las dos entradas son (casi) la misma y las dos etiquetas son opuestas. Ninguna
combinación de kernels puede separarlas, porque **la información no está en la
entrada**: está en dónde cae la ventana dentro de la imagen. Es un límite del
muestreo, no de la capacidad de la red.

⚠ **`edge` (borde de la IMAGEN) no es `border` (el anillo periférico de la
vista).** El español dice «borde» para los dos y por eso el código no: `border_px`
es un ancho de anillo, `edge_inputs` es quedarse sin imagen. Confundirlos hace
leer `border_px: 0` como «no hay borde de imagen», que es falso para **todas** las
ventanas de un control plano.

### Los modos (`fv.fovea.EDGE_MODES`)

Los cuatro valores van en el orden de `EDGE_SIDES = (L, T, R, B)`, y los dos
modos se orientan igual: **0 = no hay borde de imagen por este lado, 1 = el borde
está justo aquí.**

| valor | qué mide, por lado | alcance |
|---|---|---|
| `off` *(default)* | nada. Cero entradas extra | — |
| `pad` | qué **fracción del margen** de esta vista es relleno en vez de imagen | se apaga a `border_px` del borde |
| `dist` | a qué **distancia** está el borde de la imagen, en fóveas, saturado a 1 | una fóvea |

#### Cuánto alcanza cada uno — **medido, no supuesto**

Sobre `dirty1000-80px-16px-r20260827` (1000 imágenes de 80×60, fóvea 16, stride 8,
140.000 ventanas), *medido el 2026-08-31 recorriendo el `.npz` con
`fv.fovea.edge_features`*:

| | ventanas donde la señal se enciende | esquinas positivas dentro de ellas |
|---|---:|---:|
| `pad` (`border_px` = 4) | **31,4 %** | **30,4 %** |
| `dist` (satura a 1 fóvea) | **91,4 %** | **87,5 %** |

*(idéntico en train, val y test al 0,1 %: el reparto es por imagen, así que la
proporción de ventanas de borde no se mueve entre splits)*

Y el caso literal —**esquinas etiquetadas a ≤ 1 px del borde de la imagen**— son
**2.183, el 3,02 %** de las 72.380 positivas.

⚠ **Estos números reordenan lo que parecía obvio, y en las dos direcciones:**

1. **`pad` NO es «demasiado corto».** Con `border_px` = 4 sobre imágenes de 80×60
   alcanza a **casi un tercio** de las ventanas: el borde es una fracción grande de
   una imagen pequeña. La intuición contraria (que 4 px sobre una ventana de 16
   apenas roza nada) es correcta por ventana y falsa por dataset.
2. ⚠ **`dist` está encendida en el 91 % de las ventanas, y eso es un problema de
   interpretación, no de potencia.** En una imagen de 5×3,75 fóveas, «cerca de un
   borde» y «en qué parte de la página estoy» son casi la misma variable. Si
   `dist` gana, **no se podrá decir por cuál de las dos**. `pad` no tiene esa
   ambigüedad: se enciende sólo cuando la vista **realmente** contiene relleno.

**Por eso `pad` es el modo que contesta la pregunta y `dist` el que la desborda.**
Con imágenes grandes la relación se invertiría, y por eso son dos modos y no una
constante: el alcance útil depende del dataset, así que es un dato. Además `dist`
funciona con `border_px = 0` (la CNN plana), donde `pad` sería una constante 0 —
esa combinación **se rechaza en la puerta** en vez de entrenar una entrada muerta.

### Por qué a la cabeza y no como canal de entrada

Esto es una decisión, no una comodidad. Un quinto canal `N×N` con la máscara era
la otra opción (F7 la dejó abierta desde 2026-07-27) y aquí se descarta por dos
motivos, el segundo decisivo:

1. **No es una señal espacial.** «Arriba no hay más imagen» es *un* número sobre
   toda la vista; como canal sería el mismo valor pintado en `N×N` celdas, y las
   ramas gastarían kernels en volver a derivar una constante.
2. **Las ramas están ENMASCARADAS por región (§6).** La rama central no ve el
   anillo en absoluto, así que una señal que entrase por el input sería invisible
   justo para la rama que predice las esquinas.

Coste: `n_layers`, `channels` y la geometría **no se mueven**; la cabeza pasa de
`Linear(flat, 12)` a `Linear(flat + 4, 12)`, o sea **+48 pesos** — +0,03 % sobre
los 159.372 de la base vigente *(medido 2026-08-31 con `network_trace`)*. Con
`off` son `flat + 0`: la red es **bit-idéntica** a la de antes y los checkpoints
cargan `strict` (tiene test).

El vector se concatena **después** del ReLU y **después** del dropout:

- fuera del ReLU porque la cabeza debe leer el número tal cual. Hoy es ≥ 0 y el
  ReLU sería la identidad — pero un modo futuro con signo perdería medio rango
  contra un clamp que nadie recuerda que está ahí.
- fuera del dropout porque es una **medición**, no una activación aprendida.
  Apagarla al azar no regulariza 4 entradas: le dice a la cabeza «no hay borde»
  en una ventana que sí lo tiene. Ruido con forma de hecho.

**No barrido todavía.** El campo existe, es un eje de C y entrena; que **mejore**
el f1 no está medido. Ver `docs/plan-edge-inputs-2026-08-31.md`.

---

## 7. Suma vs. concatenación cuando los strides difieren

Con `stride > 1`, la salida deja de ser `N×N`:
`salida = floor((N + 2·padding − k) / stride) + 1`.

Enmascarar tiene dos opciones; usar la **A** en este diseño:

- **A — enmascarar ANTES de convolucionar** (sobre el input `N×N`). El stride
  actúa después, sobre datos ya separados por región. Máscaras siguen `N×N`.
  Más limpio.
- **B — reconstruir la máscara a la resolución de salida.** Frágil (errores ±1px).

Si las dos ramas terminan con dimensiones espaciales distintas (strides
distintos), la **suma `c + p` deja de alinear**. Dos salidas:

- Forzar misma dimensión con `adaptive_avg_pool2d(feat, M)` antes de sumar, o
- Cambiar a **pooling independiente por rama → concatenación de vectores**, que
  tolera dimensiones distintas.

**Regla:** si vas a buscar strides por rama de forma independiente, la
**concatenación** da más libertad que la suma. Decidir esto ANTES de lanzar la
búsqueda porque cambia el `forward`.

---

## 8. Distinción crítica entre "solapamiento" en cada etapa

- **Muestreo foveado (armado de entrada):** EXCLUYENTE. Un píxel de la original
  va al centro (alta res) *o* al borde (baja res). Lienzo con ceros; sin conflicto.
- **Máscaras de los kernels (procesamiento):** CONTRIBUTIVO. En la penetración
  ambas ramas aportan y **se suman**. Máscaras valen 1 en la zona compartida.

```python
out[zona] = pieza                          # armado: reemplaza (un origen/píxel)
feat = c*center_mask + p*periph_mask       # procesamiento: suma (varios orígenes/píxel)
```

---

## 9. Rangos factibles (resumen cualitativo, ya como función de la región)

Para referencia rápida; los valores exactos salen de las funciones de §3.

**Kernel interno (centro):** rango `3 .. region//2` impares. Sweet spot 3×3.
Con centro grande (N alto) aparecen 5, 7, 9... automáticamente.

**Kernel externo (periferia):** limitado por `periph_band` (delgada). Típico
`[3]` o `[3,5]`. Kernel grande en anillo fino desborda hacia centro/padding.

**Stride interno:** típico `[1,2]`; 3 solo con centros grandes. **Stride
externo:** casi siempre `[1]` por lo delgada que es la banda.

**Borde difuso (`border_px`):** múltiplos de `border_reduce`, acotado por que
el recorte real quepa (`≤ 3·fovea_px` por defecto — y por la imagen, que sólo B
conoce). Define **cuánto contexto** ve la red, y ya no lo define nadie más.

**Reducción del borde (`border_reduce`):** los divisores de `border_px`. Define
**con cuánta resolución** se ve ese contexto, y por tanto `N` y el coste. Ojo:
antes esto se llamaba `downsample`/`d` y definía *las dos cosas a la vez*.

**Solapes (`overlap_fovea_px`, `overlap_border_px`):** `0 .. fovea/2-1` y
`0 .. border-reduce`, cada uno sobre su rejilla. Ambos con 0 legal: son el
control de la elección de §8.

Espacio típico (N=20): |k_center|·|k_periph|·|s_center|·|s_periph|·|d| ≈
3·2·2·1·varios. Manejable con grid exhaustivo; reservar Optuna para canales,
lr, dropout.

---

## 10. Estimación de coste de entrenamiento

- **CNN simétrica 20×20** (1→32, 32→64): ~37.1M ops/imagen.
- **Versión asimétrica** (centro 16 + margen reducido): ~29.8M ops/imagen
  teórico (~80% del volumen).
- Con overhead (dos caminos, máscaras): **~10–15% más rápido** en la práctica.
  Ej. simétrica ~100 s/época → asimétrica ~87–90 s/época. Orden de magnitud.

---

## 11. Decisiones tomadas y pendientes

**Tomadas:**
- Todo es parámetro; los rangos se **calculan** a partir de las longitudes.
- La geometría se declara en **px reales de la imagen** y `N` se **deriva**
  (2026-08-25, §2.1). El borde (`border_px`) y su reducción (`border_reduce`) son
  parámetros **separados**: cuánto contexto y con cuánta resolución.
- Solape **contributivo** (ambos kernels suman en la banda compartida), y con
  **dos lados independientes** (`overlap_fovea_px`, `overlap_border_px`), ambos
  con 0 legal.
- Enmascarar ANTES de convolucionar (opción A).
- La **fóvea** es fija por experimento: es la ventana etiquetada de B
  (contrato ①a). Lo que antes se decía de `N` se dice ahora de ella; `N` ya no es
  un parámetro, es una consecuencia.
- **El borde de la imagen se le dice a la cabeza, no se le enseña por el input**
  (2026-08-31, §6bis): `edge_inputs` ∈ {`off`, `pad`, `dist`}, cuatro escalares
  concatenados a las features justo antes de la `Linear`. Cierra la mitad de **F7**
  que quedaba abierta desde C11 — y la cierra por el lado barato: el canal de
  máscara `N×N` sigue sin construirse, y ahora hay que justificar por qué haría
  falta habiendo 48 pesos que hacen el trabajo.

**Pendientes / a experimentar:**
- **`edge_inputs` no se ha barrido.** Existe y entrena; que mueva el f1 es la
  hipótesis, no un resultado. El criterio está escrito antes de mirar en
  `docs/plan-edge-inputs-2026-08-31.md`.
- `avg_pool2d` vs `max_pool2d` para reducir el borde (trazos finos EMNIST).
- **Suma vs. concatenación** de ramas (decide si se pueden buscar strides por
  rama independientes). Recomendado concat si strides difieren.
- **Otro método de reducción del borde.** Hoy es un promedio (o máximo) de
  bloques de `border_reduce` px, y por eso el borde tiene que ser múltiplo suyo.
  La reparametrización deja esa pieza aislada: cambiarla no toca la definición
  del borde ni la de la fóvea. Si el método nuevo no cae en rejilla, la
  restricción `border_px % border_reduce == 0` se va con él.
- **Los cuatro ejes geométricos nuevos, sin medir**: `border_px`,
  `border_reduce`, `overlap_fovea_px`, `overlap_border_px`. Ninguno se ha barrido
  con la ortografía nueva. Ojo al leer resultados viejos del eje `d`: medían
  *área y compresión a la vez* (§2.1).
- ¿Kernels periféricos con forma distinta o sparsity, aprovechando que el borde
  ya condensa más contexto?
- Integración con el modelo RAM (glimpses secuenciales) si se retoma esa línea.
