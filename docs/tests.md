# Tests

Qué se testea, y por qué esa lista y no otra. Método heredado del hermano, donde está probado:

> **Los contratos de [organizacion.md](organizacion.md) §2 SON el plan de pruebas.** Un contrato
> sin test es un comentario, y los comentarios se pudren.

---

## 1. El mecanismo: `xfail(strict=True)` es la lista de tareas, y es ejecutable

Los tests de contrato se escriben **todos en la fase 0.5, antes que `src/`**, marcados:

```python
@pytest.mark.xfail(strict=True, reason="contrato ①: sin implementar, plan.md fase 3")
def test_contract_01_rejects_window_size_mismatch(): ...
```

- Mientras no exista la implementación → xfail esperado → suite en verde (nadie convive con una
  suite roja).
- El día que su fase lo implemente → **XPASS estricto → la suite FALLA** y obliga a quitar el
  marcador. Una fase que deja sus xfails puestos **no está terminada**.
- El `reason` cita documento y sección: la lista no puede desviarse de la realidad.

## 2. Regla central: testea la costura, no la función

La lección medida del hermano (su contrato ⑤): de una función compartida se testeaba la mitad
que no podía romperse (la función), y no la mitad que sí (las dos copias de la fórmula que
tenían que coincidir). Aquí:

- El test del ⑤ no pregunta «¿es correcta `build_foveated_input`?» sino **«¿construyen el
  dataloader y la inferencia la misma vista?»** (identidad del módulo + vista bit-idéntica sobre
  la misma ventana).
- El de las métricas no pregunta si la fórmula es correcta sino **si la tabla por ventana mide
  lo mismo que el run reportó** (un solo sitio: `fv.metrics`).

## 3. Un test por contrato

`tests/test_contracts.py`, nombrados por número (`test_contract_01_...`). La lista, con la fase
de [plan.md](plan.md) que quita cada xfail:

| | Contrato | El test afirma | Lo quita |
|---|---|---|---|
| ① | ventana etiquetada / vista computable | `check_run` con mismatch → `window_size_mismatch` / `view_needs_images` / `original_size_exceeds_image`, **antes de reservar el nombre**; y por HTTP → 400 sin job ni run creados | 3 (validador), 4 (HTTP) |
| ② | geometría consistente | `derive_dims` con `c_frac`/`pen_frac` inválidos → error con razón (`center_not_even`, `penetration_too_large`, `kernel_must_be_odd`, `merge_sum_needs_equal_strides`); **control**: una config válida pasa | 2 |
| ②b | rangos calculados | `kernel_range`/`stride_range`/`build_search_space` reproducen los ejemplos numéricos de instructionsNewNN.md §3 (N=20 → k_center [3,5,7], s_center [1,2], s_periph [1]) | 2 |
| ③ | procedencia | el run registra red/receta **por nombre y valor** + huella de B; `DELETE` de un B en uso → 409 con la lista (**control**: uno libre → 204) | 4 (y 2 el DELETE) |
| ④ | checkpoint autodescriptivo | `load_model(ckpt)` reconstruye la red foveada sin YAML, geometría incluida | 4 |
| ⑤ | vista única | dataloader e inferencia usan **el mismo** `fv.fovea` y producen vistas bit-idénticas sobre la misma ventana; las máscaras de rama, ídem | 3 (dataloader) / 6 (inferencia) |
| ⑦ | dirección de imports | `fovea`, `metrics`, `validation`, `matrixview` no importan nada de `fv`; `models` solo de `fovea`; leído de los imports (§4) | 2 |
| ⑧ | comparabilidad | reconstruir B con otro contenido cambia la huella (y con el mismo, no); el seed de B solo decide el split; el split es **por imagen** | 2 |
| ⑨ | objetivo del recorrido | spec con `objective` dependiente de un peso barrido → `objective_varies_with_space`; con objetivo de tarea → pasa; validación **sin cargar optuna** | 7 |
| ⑩ | X fuera de D | dos runs que solo difieren en `device` tienen la misma identidad de receta | 3 |
| ⑪ | reproducibilidad | misma semilla + misma config ⇒ mismos pesos (init **y** entrenamiento, con **control** de otra semilla que difiere) | 4 |
| ⑫ | estudio planifica, recorrido ejecuta | `next-sweep` deriva la base (`center_out==window_size`, ①a) y la valida con `check_run` **sin reservar**; un recorrido con **base inline** (sin `base_network`, solo `base_network_value`) se prepara/corre/rankea igual que uno con red nombrada (mismo gate); el generador del siguiente eje fija el ganador y registra `field_origin` | H (barrido) |

Además, del muestreo foveado (nacen con la fase 2, sin xfail largo):

- `build_foveated_input` reproduce la correspondencia de coordenadas de instructionsNewNN.md §4
  (N=20, original 24: anillo px 0–3 ÷2 → px 0–1; centro 4–19 → 2–17), y el centro es
  **bit-idéntico** al recorte directo (el muestreo no toca el centro).
- Las máscaras: excluyente en el armado (cada píxel un origen), contributivo en el
  procesamiento (ambas valen 1 exactamente en la banda de penetración; suman 1 fuera).
- El camino vectorizado de la vista == el camino de referencia (la trampa medida: 48× de
  diferencia; el lento solo como oráculo del test).

### La velocidad delata dónde está la validación

Si un test de contrato necesita entrenar para detectar la violación, la validación está en la
capa equivocada. `test_contracts.py` por debajo de ~10 s.

## 4. Tests de arquitectura (⑦)

Se testea leyendo imports. Capas objetivo:

| Módulo | Puede importar de |
|---|---|
| `fv.fovea` (G) | — (arrays y aritmética) |
| `fv.metrics` | — |
| `fv.validation` | — (dos dicts) |
| `fv.matrixview` | — (es lo que lo mantiene extraíble) |
| `fv.imageops` | — |
| `fv.datasets` (A) | `imageops`, `validation` (solo el resize) |
| `fv.models` (C) | **solo `fovea`** |
| `fv.windows` (B) | `datasets`, `fovea` |
| `fv.training` (D) | `windows`, `models`, `validation`, `metrics`, `fovea` |
| `fv.inference` (F) | `models`, `fovea`, `matrixview` — **no `windows`** |
| `fv.diagnostics` (E×B) | `windows`, `training`, `inference`, `metrics`, `fovea` |
| `fv.sweeps` (H) | `training`, `validation`, `fovea` (los rangos) |
| `fv.api` | todos |

## 5. Lo que NO se testea

- **Resultados de investigación.** `f1 > 0.75` no es un test: rompe por razones legítimas y
  entrena a la gente a arreglar el umbral. Van a [protocolo.md](protocolo.md) §5.
- Valores exactos de la pérdida (sí invariantes: «un paso de entrenamiento reduce la loss»).
- Que torch funcione.
- El render de la UI píxel a píxel — salvo la paleta, que tiene validador ejecutable y se corre.

## 6. Convenciones

- Datasets sintéticos diminutos construidos en el test; nunca se toca `data/` ni `runs/` reales.
- Un test de contrato se llama como su contrato; el `reason` de un xfail cita documento y
  sección.
- Antes de cada commit que toque código: `.\.venv\Scripts\python -m pytest -q`.
