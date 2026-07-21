# CLAUDE.md — Red neuronal con muestreo foveado y ramas por región (totalmente parametrizada)

Documento de contexto para que Claude retome este proyecto sin perder las
decisiones de diseño. Describe una red neuronal (no una CNN pura) que procesa
una entrada con **resolución variable por región** (centro en alta resolución,
periferia en baja resolución pero con mayor campo visual) y con **kernels
independientes por región** que se **suman** en las zonas de solape.

**Principio rector:** *todo dato es un parámetro*. Nada está hardcodeado a
20×20 ni a downsample=2. Las dimensiones concretas y los **rangos de búsqueda**
(kernels, strides, downsample) se **calculan** en función de un lado de entrada
`N` y unas pocas fracciones. Cambiar `N` recalcula todo automáticamente.

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

Estos son los únicos grados de libertad reales. Antes eran constantes; ahora
todo sale de aquí:

```python
N          # lado de la entrada compuesta que consume la NN (antes 20)
c_frac     # fracción del input que ocupa el centro (antes 16/20 = 0.8)
d          # factor de downsampling de la periferia (antes 2)
pen_frac   # fracción de penetración hacia el centro (antes 2/20 = 0.1)
n_layers   # nº de capas conv por rama (para acotar strides; antes 2)
```

Dimensiones derivadas (con paridad controlada):

```python
center_out    = round_to_even(N * c_frac)       # centro en el input
periph_out    = (N - center_out) // 2           # grosor del anillo en el input
penetration   = max(1, round(N * pen_frac))     # filas/cols compartidas
periph_band   = periph_out + penetration        # banda útil del kernel externo
periph_real   = periph_out * d                  # ← ANTES fijo en 4; ahora = periph_out·d
original_size = center_out + 2 * periph_real     # imagen original necesaria
```

**Punto clave:** `periph_real` ya **no** es 4. "Cuánto ve la periferia" es
`periph_out · d`, producto de dos parámetros buscables. Si `periph_out=2` y
querés que vean 6px reales, subís `d=3`. Para "ven entre 5 y 9 px": con
`periph_out=2` → `d∈[3,4]` (dan 6 y 8); con `periph_out=1` → `d∈[5,9]`.

Validaciones de consistencia obligatorias:

```python
assert center_out % 2 == 0                       # anillo reparte simétrico
assert 2 * periph_out + center_out == N
assert penetration < center_out // 2             # el núcleo no desaparece
```

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

# centro=16 → [3,5,7];  centro=32 (N=40) → [3,5,7,9,11,13,15]
k_center_options = kernel_range(center_out)
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

s_center_options = stride_range(center_out, n_layers)   # centro 16 → [1,2]
s_periph_options = stride_range(periph_band, n_layers)  # banda fina → [1]
```

La periferia, por delgada, casi siempre devuelve `[1]` — ahora emerge del
cálculo en vez de ser una regla escrita a mano.

### 3.3 Padding (derivado, NUNCA buscado)

```python
padding = k // 2      # por rama; conserva 20×20 espacial con stride=1
```
Fuerza `kernel_size` impar (padding entero). Un kernel par desalinea máscaras.

### 3.4 Downsample (su propio rango factible)

```python
def downsample_range(periph_out, N, max_original=None):
    """d tal que el anillo reduzca a >=1px y la original no explote."""
    d_min, d_max = 1, 8
    if max_original:
        while (periph_out * d_max * 2 + (N - 2*periph_out)) > max_original and d_max > 1:
            d_max -= 1
    return list(range(d_min, d_max + 1))
```

### 3.5 Ensamblado del espacio de búsqueda

```python
def build_search_space(N, c_frac=0.8, pen_frac=0.1, n_layers=2):
    center_out  = round_to_even(N * c_frac)
    periph_out  = (N - center_out) // 2
    penetration = max(1, round(N * pen_frac))
    periph_band = periph_out + penetration

    return {
        "k_center":   kernel_range(center_out),
        "k_periph":   kernel_range(periph_band),
        "s_center":   stride_range(center_out, n_layers),
        "s_periph":   stride_range(periph_band, n_layers),
        "downsample": downsample_range(periph_out, N, max_original=2*N),
        # derivados no-buscables:
        "_center_out": center_out,
        "_periph_out": periph_out,
        "_penetration": penetration,
        "_original_size": center_out + 2 * periph_out * 1,  # ×d al elegir downsample
    }
```

Un solo `N` (más las fracciones) define **todo**: dimensiones y los tres
rangos buscables. `N=28` (MNIST), `N=32`, `N=64` → los rangos se recalculan
solos.

---

## 4. Geometría del muestreo foveado (construcción de la entrada)

Descomposición de la imagen original de `original_size` (ejemplo con los
valores clásicos N=20, c_frac=0.8, d=2 → original 24×24):

```
Imagen original: original_size × original_size   (ej. 24×24)
├── Centro center_out×center_out → se toma TAL CUAL     → ocupa center_out px centrales
│   (px periph_real .. periph_real+center_out-1)
└── Anillo de periph_real px     → se reduce ÷d → periph_out → ocupa el anillo del input
    (borde de la original)

Resultado: entrada compuesta N×N  →  periph_out + center_out + periph_out = N
```

Correspondencia de coordenadas (ejemplo N=20, original 24):

| Zona           | Original (24px)   | Reducción | Input (20px)      |
|----------------|-------------------|-----------|-------------------|
| Anillo externo | px 0–3 / 20–23    | ÷2        | px 0–1 / 18–19    |
| Centro         | px 4–19           | ×1        | px 2–17           |

---

## 5. Construcción de la entrada (código de referencia)

El "**lienzo con relleno cero**" es un tensor `N×N` inicializado en ceros que
sirve de superficie donde se "pegan" las piezas. Aquí la asignación es
**excluyente** (cada píxel del input tiene un único origen: centro *o* borde),
lo cual es correcto para el muestreo foveado.

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

```
Zonas dentro de la entrada N×N (ejemplo margin=periph_out=2, penetration=2):
├── Anillo externo:  px 0-1 y 18-19   -> solo kernel periférico
├── Zona compartida: px 2-3 y 16-17   -> AMBOS kernels (se suman)  ← penetración
└── Núcleo central:  px 4-15          -> solo kernel central
```

```python
def build_masks(N, periph_out, center_out, penetration):
    center_mask = torch.zeros(1, 1, N, N)
    periph_mask = torch.zeros(1, 1, N, N)

    # Kernel central: toda la región interna center_out×center_out
    lo, hi = periph_out, N - periph_out
    center_mask[..., lo:hi, lo:hi] = 1

    # Kernel periférico: anillo externo + penetración hacia adentro.
    inner_lo = periph_out + penetration
    inner_hi = N - periph_out - penetration
    periph_mask[...] = 1
    periph_mask[..., inner_lo:inner_hi, inner_lo:inner_hi] = 0

    # En la banda de penetración AMBAS valen 1 -> se suman.
    return center_mask, periph_mask
```

```python
import torch.nn as nn

class FoveatedRegionalNN(nn.Module):
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

**Downsample:** `[1..8]` acotado por que la original no explote (`≤ 2N` por
defecto). Define cuánto contexto ve la periferia (`periph_out·d`).

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
- Todo es parámetro; los rangos se **calculan** a partir de `N` y fracciones.
- `periph_real = periph_out · d` (cuánto ve la periferia es buscable, no fijo en 4).
- Penetración **contributiva** (ambos kernels suman en la banda compartida).
- Enmascarar ANTES de convolucionar (opción A).
- `N` es **fijo por experimento** (no se mezclan escalas de imagen en una corrida).

**Pendientes / a experimentar:**
- `avg_pool2d` vs `max_pool2d` para reducir la periferia (trazos finos EMNIST).
- **Suma vs. concatenación** de ramas (decide si se pueden buscar strides por
  rama independientes). Recomendado concat si strides difieren.
- ¿`c_frac` y `pen_frac` fijas por aplicación o también buscables? (Si buscables,
  muchas combinaciones son geométricamente inválidas → grid aparte, con asserts.)
- Política de redondeo/paridad al derivar `center_out`, `periph_out` de `N`.
- ¿Kernels periféricos con forma distinta o sparsity, aprovechando que la
  periferia ya condensa más contexto?
- Integración con el modelo RAM (glimpses secuenciales) si se retoma esa línea.
