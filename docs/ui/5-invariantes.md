# UI · Tipo 5 — Especificación de invariantes de dominio (las puertas)

> **Qué decide**: qué tiene que bloquear, avisar o derivar un formulario, y cómo se enseña una
> negativa.
> **Qué NO decide**: la restricción en sí. Los contratos ①–⑬ viven en
> [organizacion.md](../organizacion.md) §2 y se hacen cumplir en `fv.validation` — **la UI no los
> reimplementa, los refleja**.
> **Cómo se hace cumplir**: **ejecutable** (la mitad de servidor): `check_run`/`check_network`,
> `tests/test_contracts.py`, y los 400/409 probados por HTTP. La mitad de cliente (que el control
> exista y se vea) se verifica con Playwright.

---

## Las reglas

**U5.1 — El formulario refleja el validador; nunca lo reimplementa.** Si la UI decide por su cuenta
qué combinación es válida, hay dos definiciones de la misma regla y **la más laxa es por la que
entra un recorrido automático**. El caso medido (2026-07-26): `validate_plan` era más laxa que
`check_sweep` para el contrato ⑨, y el `<select>` de Estudios lo tapaba **no ofreciendo** la opción
mala — el hueco existía igual y se cerró en la puerta, no en el `<select>`.

**U5.2 — Toda negativa se enseña con razón y arreglo.** `code` + `message` + `hint`, visibles.
Un `400` al entrar vale mil veces más que un stack trace dentro del hilo del job media hora
después ([api.md](../api.md) R4).

**U5.3 — Ausente ≠ cero, también en pantalla.** Un campo que falta se dibuja como falta (`—`, «sin
dato», «no medido»), jamás como 0 ni como vacío ambiguo. Casos vivos: `mean_iou: null` cuando no
hubo emparejamientos (**no** 0), `value: null` en el ranking cuando el monitor nunca midió (**no**
la última época de consuelo), `spearman` `None` cuando una serie es constante.

**U5.4 — Los derivados se piden, no se escriben.** Redes muestra `center_out`, `periph_out`,
`penetration`, `original_size` y los rangos calculados **en vivo desde el servidor**
(`POST /networks/validate`, contrato ②) antes de guardar. Un derivado calculado en el front es una
copia que diverge (U4.2); un derivado escrito a mano en un YAML, también.

**U5.5 — Los bloqueos concretos que la UI debe tener puestos.** Cada uno cita su contrato; el
detalle está allí, no aquí:

| En pantalla | Se bloquea / se avisa | Contrato |
|---|---|---|
| Entrenar, Recorridos, Estudios | B y C incompatibles (`window_size_mismatch`, `view_needs_images`, `original_size_exceeds_image`) **antes de reservar nombre** | ① |
| Redes | asserts de geometría con su razón (`penetration_too_large`, `kernel_must_be_odd`, `merge_sum_needs_equal_strides`) | ② |
| Recorridos, Estudios | objetivo que depende de lo que se barre (`objective_varies_with_space`, `objective_depends_on_geometry`) | ⑨ |
| Recorridos, Estudios | `N` y `c_frac` **no son ejes** (`axis_breaks_window_size`) — se rechaza con el arreglo: barre `d`, o usa otro B | ①a |
| Recetas | `device`/`num_workers` **no están** en la receta | ⑩ |
| Estudios | el ganador se **sugiere**; lo confirma el usuario | ⑫ / D-W1 |
| Métrica de tarea | `task_needs_source`, `holdout_shares_source`, `window_dataset_changed` | ⑬ / ⑧ |

**U5.6 — Borrar dice a quién arrastra, y se niega si hay algo vivo.** Cascada confirmada antes de
tocar nada (recorrido → sus runs; estudio → sus recorridos → sus runs); 409 con la razón si algo
está `queued`/`running`; y en el borrado por-nombre la negativa **nombra al referente** (qué
recorrido o estudio fija ese B o esa receta). Un run hijo no se borra solo: sus puntos se comparan
juntos.

**U5.7 — Se gatea por pertenencia a la lista, no por verdad recordada.** Un nombre de run guardado
en el navegador puede haber sido borrado o renombrado: la pantalla lo usa **solo si sigue en la
lista** que acaba de traer, y la petición condenada ni revienta ni pisa la carga válida.

**U5.8 — Un run no se sobrescribe jamás**, así que la UI no ofrece «reemplazar»: ofrece otro
nombre. El 409 se enseña con la razón.

**U5.9 — La validación ocurre antes de reservar el nombre.** Consecuencia para la UI: el mensaje
correcto de un rechazo es *«no se creó nada»*, y debe poder afirmarse. Validar después de reservar
deja un `runs/<name>/` muerto y el reintento contesta «ya existe» — un callejón sin salida con
forma de error normal.
