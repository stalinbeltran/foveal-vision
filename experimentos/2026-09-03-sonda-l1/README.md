# Sonda L1 — ¿pueden los kernels de la primera capa aprender filtros genéricos?

**Corrido el 2026-09-03 02:33:54 → 04:40:13 UTC.** 8 redes entrenadas, 2,1 h, **0 máquinas
alquiladas, 0 $** (el coste es reloj del droplet dev, 2 vCPU).
Código de `foveal-vision` en el commit **`c19cb745b`**; snapshot en [`codigo/`](codigo/).

---

## 1. La pregunta

En `fov16-optimo-mask` los 16 kernels de la primera capa **no aprendieron filtros genéricos**:
su energía en el subespacio clásico 6D es **0,688** contra **0,667** de un kernel aleatorio —
o sea **1,03×**, indistinguible del azar.

**Hipótesis: L1 no está bajo presión.** No hay reducción tras ella, y detrás hay una cabeza de
153.660 parámetros que puede extraer las esquinas de casi cualquier proyección.

Este experimento **quita esa red de seguridad** y pregunta: si la presión cae entera sobre L1,
¿aprende filtros genéricos?

## 2. La red

```
x  (1, 20, 20)                    ← la vista foveada, la MISMA que ve producción
   → Conv2d(1 → K, k×k, stride 1, padding k//2, replicate, con sesgo)
   → ReLU
   → z  (K, 20, 20)               ← el código: el entregable
   → ConvTranspose2d(K → 1, k×k, stride 1, padding k//2, SIN sesgo)
   → x̂ (1, 20, 20)
```

**Nada entre codificador y decodificador**: sin batchnorm, sin pooling, sin cabeza. *El modelo
son los kernels.* El decodificador es lineal y sin sesgo a propósito — si pudiera compensar un
código malo, la presión sobre los kernels desaparecería.

| k | 3 | 5 | 7 | 9 |
|---|---:|---:|---:|---:|
| parámetros (`K·k²·2 + K`, K=16) | **304** | **816** | **1.584** | **2.608** |

*(para comparar: `fov16-optimo-mask` tiene 168.652)*

**Entrenamiento**: auto-supervisado — reconstruir la entrada, sin etiquetas ni esquinas.

```
pérdida = mse(x̂, x) / var(x)  +  λ · mean(|z|)
```

- `var(x)` es una **constante fija del train** (0,3066), no la del lote: si cambiara por lote, λ
  significaría algo distinto en cada paso.
- La L1 va sobre las **activaciones**, no sobre los pesos: penalizar pesos hace la red pequeña,
  penalizar activaciones la hace selectiva.
- Tras **cada** paso del optimizador, los átomos del decodificador se renormalizan a **L2 = 1**.
  Sin eso el modelo escala el codificador por 0,01 y el decodificador por 100 — misma
  reconstrucción, penalización 100× menor — y aprendería a hacer `z` **pequeño** en vez de
  **disperso**.

Adam, `lr` 3e-3, lote 256, **30 épocas**. Dataset `dirty1000-80px-16px-r20260827`
(84.000 ventanas de train, 28.000 de validación), semilla **1**.

**Ocho runs**: `k` ∈ {3, 5, 7, 9} × λ ∈ {**0** (control), **calibrada**}. La λ calibrada se
bisecta por celda hasta ~10 % de activación; salió **28,28** en las cuatro.

## 3. El resultado: **NO**

Ninguna de las 8 pasa el criterio, y falla de forma **estructurada**:

| brazo | señal de forma | salud del código |
|---|---|---|
| **λ calibrada** | Gabor Δ/margen **0,197–0,360**, orientación ≤ 0,056 — todos bajo el umbral de 0,40 | **0 muertos, 0 saturados, 16 vivos**, activación 5,0–6,7 % |
| **λ = 0** | Gabor Δ/margen **0,615–0,836**, pasa de sobra | 1–7 muertos, 7–9 **saturados**; en k=5, **cero vivos** |

**El brazo λ=0 aprende kernels DELTA** — la solución identidad: kernel delta → `z` copia de `x`
→ decodificador delta, y `R²` de reconstrucción exactamente **1,000**. Medido por la anchura de
la envolvente del Gabor ajustado: **σ 0,49–0,57 px** contra **1,41–1,49 px** de un kernel
aleatorio.

**Bajo presión real —código disperso y sano— los kernels salen localizados y NO orientados**, no
Gabors. Sin presión, el modelo coge el atajo.

### ⚠ Lo que más vale de este experimento

**Sin las métricas sin plantilla, el estudio habría concluido lo contrario.** El ajuste a Gabor
—la métrica principal del encargo— supera su nulo al 5 % en **las 8** configuraciones, y con el
umbral absoluto propuesto (0,25) los cuatro runs de λ=0 lo habrían pasado holgadamente: *«éxito,
salen Gabors»* sobre **deltas**. Lo desmintieron `conc_orient` y `conc_banda`, porque un delta
tiene espectro **plano e isótropo**.

> **Una métrica con el nulo bien puesto sigue pudiendo ser engañada por una degeneración de la
> familia que ajusta.** Hay que mirar los **parámetros** del ajuste, no sólo su R².

La tabla completa, los matices y el criterio —escrito **antes** de mirar— están en
[`instrucciones/03-plan-y-criterio.md`](instrucciones/03-plan-y-criterio.md) §9.
Reporte público: [#22](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/estudios/2026/09-septiembre/2026-09-03-sonda-l1-tanteo-eje-k.md).

## 4. Qué hay aquí

| | |
|---|---|
| [`instrucciones/01-encargo-original.md`](instrucciones/01-encargo-original.md) | el encargo tal como llegó (commit `4cbfe0e84`) |
| [`instrucciones/02-respuestas-del-dueno.md`](instrucciones/02-respuestas-del-dueno.md) | su revisión: 4 respuestas + el quinto problema que encontró (`7959c558d`) |
| [`instrucciones/03-plan-y-criterio.md`](instrucciones/03-plan-y-criterio.md) | el criterio **congelado antes de mirar**, y el §9 con el veredicto |
| [`nn/modelo.py`](nn/modelo.py) | la red **autocontenida** — no importa nada del repo |
| `nn/pesos/*.pt` | los 8 checkpoints (80 KB en total) |
| `resultados/<run>/` | `config.json`, `metrics.jsonl` (una línea por época), `summary.json`, kernels `.npy`, hojas de contactos y mapas `z` |
| `resultados/tabla.md` · `tabla.csv` · `resumen.json` | la comparativa de los 8 |
| [`codigo/`](codigo/) | **snapshot congelado** del código. La copia viva está en `src/fv/probe/` |

## 5. Cómo se usa

```bash
cd experimentos/2026-09-03-sonda-l1

# comprobar que los 8 pesos cargan y casan con sus kernels guardados
../../.venv/bin/python nn/modelo.py

# cargar uno y mirarlo
../../.venv/bin/python -c "
import sys; sys.path.insert(0, 'nn')
from modelo import cargar
m = cargar('nn/pesos/k9-K16-lcal-s1.pt')
print(m.K, m.k, m.enc.weight.shape)
"
```

**Para volver a correrlo** (con la copia viva del código, no con el snapshot):

```bash
cd ~/src/foveal-vision
.venv/bin/python scripts/sonda_l1.py --tanteo-k --canales 16 --epocas 30   # ~2,1 h
```

Desde Telegram: `/use sonda-l1`.

## 6. Lo que quedó pendiente

1. **Una sola semilla.** El patrón es consistente en los cuatro `k`, pero **acota, no declara**.
   Repetirlo con 3 semillas cuesta ~1 h.
2. **`K` no se barrió**: todo es K=16. La sobrecompletitud (K=32 con k=9 sería 32×) es justo el
   eje que el encargo asociaba a que emergiera estructura.
3. **La rejilla completa (48 runs, 12–15 h) NO se lanzó**, y la recomendación es no lanzarla: el
   eje `k` está barrido y el patrón es idéntico en los cuatro valores.
4. **El umbral de magnitud (0,40) es del dueño y sigue negociable.** Con 0,30 pasarían dos runs
   del brazo calibrado; con 0,20, seis de los ocho. **No se cambia después de mirar.**
5. **La fase 2** —congelar el codificador ganador como L1 de la rama central y reentrenar—
   pedía que la fase 1 tuviera éxito, así que **no se ejecuta**.
6. **El control λ=0 merece un diseño mejor.** «Sin esparsidad» acabó siendo «solución
   identidad»; separar *«¿ayuda la esparsidad?»* de *«¿evita el atajo?»* pide un control que
   bloquee la identidad de otro modo (p. ej. atar decodificador y codificador).

---

*Los ajustes futuros de este experimento se guardan **aquí**.*
