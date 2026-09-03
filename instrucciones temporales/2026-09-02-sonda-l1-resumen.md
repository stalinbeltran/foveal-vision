# Resumen de lo dicho sobre el encargo de `instruccioneslargas.md` (sonda L1)

**Fecha:** 2026-09-02 · **Commit:** [`65dd66024`](https://github.com/stalinbeltran/foveal-vision) en `main`
de `foveal-vision`, empujado.

> Esto es **un resumen de lo que te conté**, no una fuente nueva. Lo que manda es:
> el encargo en [`instruccioneslargas.md`](../instruccioneslargas.md), el criterio en
> [`docs/plan-sonda-l1-2026-09-02.md`](../docs/plan-sonda-l1-2026-09-02.md), y el bloque
> `⏳ 2026-09-02 — SONDA L1` al principio de [`CLAUDE.md`](../CLAUDE.md).
> Si algo de aquí choca con alguno de esos tres, **gana el otro**.

---

## 1. En una línea

**El encargo está implementado entero y probado. La rejilla NO se ha lanzado**, y no debe
lanzarse hasta que decidas dos cosas (§4 de aquí), porque tal como está escrito el criterio de
éxito **no lo puede cumplir ninguna configuración**.

---

## 2. Lo que ya estaba y lo que faltaba

Cuando empecé había un commit de 3 h antes (`31ad236b`, 22:07 UTC) con parte del trabajo. El
fichero `instruccioneslargas.md` entró al repo **después**, a las 22:16 UTC. De las siete
secciones del encargo:

| § | Qué pedía | Estado al empezar |
|---|---|---|
| 1 | la estructura del autoencoder | ✅ estaba |
| 2 | normalización de contraste local | ✅ estaba |
| 3 | esparsidad + renormalización del decodificador | ✅ estaba |
| 4 | rejilla completa `k × K × λ` | 🟡 a medias (faltaba «repetir las 3 mejores») |
| 5 | las ocho métricas | ❌ faltaban **tres**, incluida la principal |
| 6 | módulo aislado + artefactos + figuras + tabla | ❌ faltaba casi todo |
| 7 | criterio escrito antes de mirar | ❌ no existía |
| 8 | fase 2 | — sólo si la fase 1 sale bien |

---

## 3. Lo que hice

### 3.1 §6 — el módulo aislado

La sonda vivía entera en `scripts/`, que es el antipatrón que este proyecto ya tiene anotado
(*«`scripts/` adelantando a `src/`»*). Ahora:

```
src/fv/probe/          model · data · gabor · metrics · run · figures · table
scripts/sonda_l1.py    SOLO la CLI
```

**El aislamiento que pide el encargo tiene test**: ningún fichero de `fv.probe` importa
`fv.models`, y de `fv.fovea` entran **exactamente** `build_view` y `dims_of` (el cargador de
ventanas, que es la excepción que el encargo permite). La geometría de producción viaja como
`dict` desde el script, así que `NETWORK_DEFAULTS` conserva **una sola definición** sin que el
módulo dependa de ella.

> Por qué se comprueba leyendo el código y no la documentación: un import de más **no rompe nada
> visible**. Ata el experimento a la red que estudia, y entonces «es un experimento aparte» deja
> de ser cierto sin que falle nada.

### 3.2 §5.4 — el ajuste a Gabor, la métrica **principal**, y no existía

Ajuste 2D por mínimos cuadrados no lineales, multiarranque (32 arranques fijos ⇒ **determinista**)
y batcheado en **torch** — no `scipy`, que no está instalado en ninguna máquina de la flota. La
amplitud se resuelve en forma cerrada, así que `R² = cos²(kernel, gabor)`: 7 parámetros libres en
vez de 8, y acotado en [0, 1] por construcción.

Con su **línea base aleatoria**, que es lo único que lo hace legible (§4.2 de aquí).

### 3.3 §5.3, §5.6 y §5.8 — tres métricas mal o ausentes

- **Kernels muertos**: el umbral era `1e-4`. El encargo dice *«activos en <0,1 % de las
  posiciones»*, o sea `1e-3`. Corregido, con test que fija el umbral como **dato del encargo**.
- **Dimensión efectiva**: faltaban las componentes de PCA al 95 % que pide el §5.6. Ahora se
  reportan **las dos** — la del encargo y el *participation ratio*, que no está topado por
  `min(K, k²)` y por eso es el comparable entre columnas.
- **Alineación codificador/decodificador (§5.8)**: no existía. Se compara **sin voltear**, porque
  `conv_transpose2d(w)` es el adjunto exacto de `conv2d(w)`. Hay un test que lo comprueba
  numéricamente: si dejara de ser cierto, la métrica daría ~0 con pesos atados, o sea **la
  conclusión contraria a la verdadera**.

### 3.4 §6 — artefactos, figuras y tabla

Por run: `config.json`, `metrics.jsonl` (**una línea por época**), `checkpoint.pt` y los kernels
de **los dos lados** en `.npy`. Más:

- **hoja de contactos** por run, con **escala de color común** a los K kernels — autoescalar cada
  uno haría que un kernel muerto pareciese tan estructurado como uno vivo;
- **mapas `z`** de las 3 mejores (la figura que contesta visualmente *«¿es la imagen resultante
  más genérica?»*);
- **tabla comparativa** de toda la rejilla con las ocho métricas, en markdown y CSV.

Todo con **Pillow**, no matplotlib: matplotlib no está instalado en ninguna máquina de la flota, y
una figura que necesita un `pip install` en una máquina que se rehace sin aviso es una figura que
nadie ve. `scripts/demo_contrafactico.py` ya había sentado ese precedente.

⚠ **`checkpoint.pt` NO entra en git**, y es deliberado: el `.gitignore` del repo de datos tira
todo `.pt` desde que fijaste (2026-08-31) que los pesos de un run no se guardan por defecto. Aquí
**no hace falta excepción**: los kernels son el entregable (§1 del encargo: *«el modelo son los
kernels»*) y viajan en `.npy`, que sí entra. El experimento sigue siendo reproducible desde git.

### 3.5 §4 — `--repetir-mejores`

Rankea agrupando **por combinación**, no por semilla suelta: si ya hay varias semillas, la mejor
es la de mejor **media**, no la de la semilla más afortunada.

### 3.6 §7 — el criterio, escrito antes de mirar

[`docs/plan-sonda-l1-2026-09-02.md`](../docs/plan-sonda-l1-2026-09-02.md).

---

## 4. ⚠ Las dos cosas que tienes que decidir antes de gastar las 12 h

Las dos salieron de **medir antes de lanzar nada**, y las dos afectan a umbrales que el propio
encargo propone. El plan las deja escritas como **enmiendas propuestas** y **no las aplica**.

### 4.1 ⚠⚠ La rejilla de λ del encargo no llega a la banda de activación que su criterio exige

El §3 del encargo fija el objetivo de activación en **5–15 %** y el §7 lo mete en el criterio de
éxito. *Medido el 2026-09-02* (k=7, K=16, 6 épocas, `--limite 8000`, semilla 1 — un **tanteo de
rango**, no un resultado):

| λ | activa % | R² rec int | Gabor Δ |
|---:|---:|---:|---:|
| **0,0** *(control)* | 45,6 | 0,975 | +0,070 |
| **0,03** | 44,9 | 0,975 | +0,072 |
| **0,1** | 43,4 | 0,974 | +0,072 |
| **0,3** *(tope de la rejilla)* | **39,9** | 0,974 | +0,074 |
| 1,0 | 31,7 | 0,973 | +0,065 |
| 3,0 | 22,3 | 0,969 | +0,078 |
| **6,0** | **15,9** | 0,961 | +0,069 |
| **10,0** | **12,9** | 0,949 | +0,056 |
| **20,0** | **10,4** | 0,919 | +0,004 |
| **40,0** | **8,0** | 0,845 | +0,019 |

1. **Los cuatro valores de la rejilla caen en la misma zona**: 45,6 % → 39,9 % es todo el
   recorrido que compran, y el tope queda **por encima del 30 %** que el propio encargo llama
   *«λ es baja»*. Con la rejilla tal cual, **la cláusula de activación no la cumple nadie**, y
   entonces el éxito es inalcanzable **por construcción**, no por el resultado.
2. **La banda 5–15 % vive en λ ≈ 6–40**: **20× a 130×** por encima del tope propuesto.
3. **Y ahí el Gabor Δ no sube: baja.** Si eso aguanta a 30 épocas, la respuesta a *«¿aporta la
   esparsidad?»* es **no** — y es un resultado válido. Pero conviene medirlo **en el rango donde
   la esparsidad existe de verdad**, no en uno donde ni siquiera se ha encendido.

⚠ **Lo que ese tanteo NO dice:** son 6 épocas, 8.000 ventanas, **un** punto de (k, K) y **una**
semilla. Las cifras se moverán con 30 épocas sobre 84.000. Lo estructural es el **orden de
magnitud** del desajuste, que no lo explica el presupuesto de épocas.

**Enmienda propuesta:** λ ∈ {0 · 0,3 · 3 · 10 · 30}. Cuesta 60 runs en vez de 48 → **~15,0 h**
en vez de 12,0.

### 4.2 ⚠ El nulo del Gabor en 3×3 hace **imposible** el umbral propuesto ahí

*Medido el 2026-09-02*, mediana del R² sobre **64 kernels aleatorios** del mismo tamaño:

| `k` | nulo (R² del ruido) | techo de `Gabor Δ` = 1 − nulo | ¿alcanza el umbral de **0,25**? |
|---:|---:|---:|---|
| **3** | **0,879** | **0,121** | ❌ **imposible por aritmética** |
| 5 | 0,515 | 0,485 | sí, gastando la mitad del margen |
| 7 | 0,337 | 0,663 | sí |
| 9 | 0,228 | 0,772 | sí |

Con 7 parámetros libres sobre 9 muestras, **un Gabor ajusta cualquier 3×3**. El encargo ya avisa
de que *«un Gabor ajusta ruido mejor de lo que uno espera»*; esto es cuánto.

**Enmienda propuesta:** el umbral de Gabor Δ se juzga sólo en k ∈ {5, 7, 9}; **k=3 se lee por el
enriquecimiento** (métrica 5), que es como sigue siendo comparable con el **0,688** de la premisa.
Si no, el ancla «fracasa» siempre por aritmética y ese falso fracaso contamina la lectura de las
demás columnas.

### 4.3 Y dos preguntas más, menores

- ¿Rejilla entera (84.000 ventanas, **~12-15 h**) o `--limite 20000` (**~4-5 h**)?
- ¿Los umbrales **0,25 / 0,10** se quedan como están? Con los nulos ya medidos, 0,25 en k=9 es
  exigir explicar el 45 % del margen disponible.

---

## 5. El coste, medido (no estimado)

*2026-09-02, este droplet, 2 vCPU, con `--cronometrar`:*

| Qué | Reloj |
|---|---:|
| combinación más cara (k=9, K=32) | **101,3 s/época** |
| rejilla de 48 runs × 30 épocas | **~12,0 h** *(extrapolado por `K·k²`)* |
| + las 3 mejores × 3 semillas | ~1,5 h |
| lo mismo con `--limite 20000` | **~4,0 h** *(34,0 s/época)* |

**No alquila nada**: corre en esta máquina y satura sus 2 vCPU. Lo que se pierde al apagar es
**trabajo, no dinero** — pero son 12 h de él, así que el freno cuenta igual.

---

## 6. Lo verificado ejecutando (no razonando)

| Qué | Resultado |
|---|---|
| suite de `foveal-vision` | **557 pasan**, 3 skip (eran 533) |
| tests de la sonda | **50** en `tests/test_sonda_l1.py` (eran 26) |
| run real de punta a punta | artefactos + figuras generados y revisados a ojo |
| el freno nombra la sonda | `🔴 NO CERRAR — 1 trabajo(s) vivo(s): sonda_l1.py`, con la sonda corriendo |
| la unidad de systemd | `active`, cgroup **propio** `/system.slice/sonda-l1.service` |
| el ejecutor de Telegram | `/use sonda-l1` → `estado`, `--tabla` y el lanzamiento desacoplado, los tres por el arnés real |

Además pasé el trabajo por el agente `verificador`, que reprodujo de forma independiente los
nulos del Gabor, la tabla de λ (números **idénticos**: el barrido es determinista) y el
aislamiento del módulo.

### 6.1 ⚠ Un fallo que encontró esa verificación, ya arreglado

`desacoplar-persistente.sh` registra la unidad con `Restart=on-failure`, y el aviso iba al final
de la tubería: **el código de salida lo decidía `notify.mjs`**.

- **Medido:** la sonda **terminó bien**, el aviso falló (el arnés no pasa `BOT_TOKEN`), y la
  unidad se quedó en `Result=exit-code` **reiniciándose cada 30 s**. O sea: **12 h de rejilla
  relanzadas por un aviso.**
- **Y al revés también:** un trabajo que reventara salía como `success` y **no** se reintentaba —
  que es justo cuando el reintento sirve, porque `--rejilla` se reanuda saltando los runs hechos.

Ahora manda el código del **trabajo** y el aviso no puede cambiarlo. Vive en
`scripts/sonda_l1_desacoplada.sh` — **un fichero, no una línea escapada dentro de un JSON**, que
es lo que lo hace probable — con **un test por cada dirección del fallo**.

Y el ejecutor ahora **sólo desacopla lo que entrena** (`--rejilla`, `--repetir-mejores`,
`--solo`): un flag mal escrito ya no puede levantar una unidad de 12 h.

---

## 7. Dos cosas del proceso que te dije, y conviene que queden

1. **El §7 del encargo dice «párate a que el dueño confirme los umbrales antes de escribir el
   entrenamiento», y esa parada no se pudo respetar**: el entrenamiento ya existía cuando el
   encargo llegó al repo (`31ad236b` a las 22:07 UTC; `instruccioneslargas.md` a las 22:16). La
   respeté donde quedaba algo que parar: **la rejilla no se lanza**.
2. **La fase 2 (§8) no está implementada, a propósito.** Depende de qué `k` y `K` ganen, y
   escribir hoy el cableado de una geometría que aún no se conoce es escribir la mitad que habrá
   que rehacer.

---

## 8. Qué hay que hacer ahora

1. **Contestar el §6 del plan** ([`docs/plan-sonda-l1-2026-09-02.md`](../docs/plan-sonda-l1-2026-09-02.md)):
   las cuatro preguntas del §4 de aquí.
2. **Lanzarla**, desde Telegram:
   ```
   /use sonda-l1
   --rejilla                 # las 48/60, ~12-15 h
   --rejilla --limite 20000  # ~4-5 h
   estado                    # cómo va
   parar                     # detenerla; lo escrito se conserva y se reanuda
   ```
3. **Cuando termine**, su reporte va al repo central `estudios-redes-neuronales`, en
   `reportes/estudios/2026/09-septiembre/`, con su fila en la tabla de `reportes/README.md`.
   Instancias: **0** (no alquila). Coste: **0 $** — aquí el coste es **reloj**.

---

## 9. Ficheros tocados

| Fichero | |
|---|---|
| `src/fv/probe/{__init__,model,data,gabor,metrics,run,figures,table}.py` | nuevo |
| `scripts/sonda_l1.py` | reescrito: ahora sólo la CLI |
| `scripts/sonda_l1_desacoplada.sh` | nuevo |
| `tests/test_sonda_l1.py` | 26 → 50 tests |
| `telegram/executors/sonda-l1.json` | rutado y mensajes |
| `docs/plan-sonda-l1-2026-09-02.md` | nuevo — **el criterio** |
| `CLAUDE.md` | bloque `⏳ 2026-09-02 — SONDA L1` |

---

<sub>**Fuera de este encargo, en la misma sesión:** pediste el merge de `tema-2` a `main`. No
había nada que fusionar —en los cinco repos `tema-2` es ancestro de `origin/main`—, pero al
comprobarlo salió un falso positivo del freno: `git-pendiente.mjs` se comparaba contra
`origin/<rama>` en vez de contra todos los remotos, y una rama remota vieja hacía que `main`
entero se leyera como «se perdería». Arreglado y empujado en `telegram-coordinator`
(`efbb033`), con tres tests, dos de los cuales fallan con el código anterior.</sub>
