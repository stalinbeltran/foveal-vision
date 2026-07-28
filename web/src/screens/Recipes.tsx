import React, { useEffect, useState } from "react";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field } from "../components/ui";

// D — every field carries its definition inline; a hyperparameter without a
// definition does not enter the form. device/num_workers are NOT here (X).
const HELP: Record<string, string> = {
  lr: "tamaño del paso; barrer en escala log",
  optimizer: "adam | adamw | sgd (momentum explícito)",
  momentum: "inercia de sgd; el default 0 sesga cualquier comparación",
  weight_decay: "L2; en adamw desacoplada",
  batch_size: "es D, no X: cambiarlo cambia el resultado (acoplado a lr)",
  epochs: "pasadas sobre el train",
  scheduler: "none | cosine — sin él, barrer lr optimiza otro régimen",
  patience: "parada temprana per-run; 0 = off (distinta de la poda del recorrido)",
  lambda_pos: "peso de posición vs existencia; ojo al contrato 9 al rankear",
  pos_weight: "peso de la clase positiva en la BCE (parte del desbalance de B)",
  smooth_l1_beta: "umbral cuadrático→lineal; 1.0 anula el Huber con coords [0,1]",
  monitor: "qué elige best.pt; nombra la métrica de val (val_f1), no el objetivo (f1)",
  seed: "eje de réplica, no un hiperparámetro a optimizar",
};

// Enumeraciones que hoy solo viven aquí. Una definición, sí — pero en el
// dominio equivocado: son vocabulario de D y debería servirlas el API, como ya
// hace con los objetivos. Pregunta abierta: decisiones.md F16.
const OPTIMIZERS = ["adam", "adamw", "sgd"];
const SCHEDULERS = ["none", "cosine"];

export default function Recipes() {
  const [list, setList] = useState<any[]>([]);
  const [defaults, setDefaults] = useState<any>(null);
  const [form, setForm] = usePersistedState<any>("recipes.form", { name: "" });
  const [error, setError] = useState<unknown>(null);
  // el vocabulario cerrado de D lo sirve el API desde la MISMA constante contra
  // la que valida la puerta (U4.2): una copia aquí es la que ofrecía objetivos
  // donde van monitores
  const [vocabulary, setVocabulary] = useState<any>(null);
  const [usedBy, setUsedBy] = useState<Record<string, string[]>>({});
  const [confirming, setConfirming] = useState(false);

  const refresh = () => api.get("/recipes").then((d) => {
    setList(d.recipes);
    setDefaults(d.defaults);
    setVocabulary(d.vocabulary ?? null);
    setUsedBy(d.used_by ?? {});
    setForm((f: any) => ({ ...d.defaults, ...f }));
  }).catch(setError);
  useEffect(() => { refresh(); }, []);

  // Una receta es fuente (C/D se editan; un run no, U5.8). Sobrescribir es una
  // acción distinta de crear, así que se pide aparte: el 409 «ya existe → elige
  // otro nombre» era la única salida que ofrecía la pantalla, y editar la receta
  // que acabas de abrir no es elegir otro nombre.
  const exists = list.some((r) => r.name === form.name);

  const save = async (overwrite = false) => {
    setError(null);
    try {
      // se envía lo que D define (los defaults que sirve el API) más el nombre:
      // un formulario recordado puede traer restos —el sobre del fichero, un
      // campo retirado— que la puerta rechazaría con razón
      const body: any = { name: form.name };
      for (const k of Object.keys(defaults)) body[k] = form[k];
      if (overwrite) body.overwrite = true;
      await api.post("/recipes", body);
      setConfirming(false);
      await refresh();
    } catch (e) { setError(e); }
  };

  if (!defaults) return <h2 data-domain="D">Recetas (D)</h2>;
  const fields = Object.keys(defaults);

  // Un <select> cuyo value no está entre sus opciones enseña la PRIMERA y calla:
  // el control decía 'f1' mientras la receta guardaba 'val_loss'. Lo guardado se
  // dibuja siempre, marcado si no pertenece al vocabulario — enseñar el valor
  // real y su rareza, nunca sustituirlo por uno plausible (U5.3).
  const optionsFor = (k: string): { value: string; label: string }[] => {
    const known = k === "optimizer" ? OPTIMIZERS
      : k === "scheduler" ? SCHEDULERS
      : ((vocabulary?.monitor ?? []) as string[]);
    const opts = known.map((o) => ({ value: o, label: o }));
    const v = form[k];
    if (v != null && v !== "" && !known.includes(v))
      opts.unshift({ value: v, label: `${v} (no reconocido)` });
    return opts;
  };
  const options = (k: string) => optionsFor(k).map((o) => (
    <option key={o.value} value={o.value}>{o.label}</option>));
  const onSelect = (k: string) => (e: React.ChangeEvent<HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });
  return (
    <div>
      <h2 data-domain="D">Recetas (D)</h2>
      <p className="sub">Hiperparámetros que definen el resultado. device y num_workers NO van aquí
        (contrato ⑩): son ejecución, viven en Entrenar.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card" style={{ width: 340 }}>
          <Field label="nombre"><input value={form.name ?? ""}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          {fields.map((k) => (
            <Field key={k} label={k} help={HELP[k]}>
              {/* el data-testid va LITERAL en cada uno: uno calculado no existe
                  para el verificador que los declara contrato (U7.11) */}
              {k === "optimizer" ? (
                <select data-testid="optimizer-select" value={form[k] ?? ""}
                  onChange={onSelect(k)}>{options(k)}</select>
              ) : k === "scheduler" ? (
                <select data-testid="scheduler-select" value={form[k] ?? ""}
                  onChange={onSelect(k)}>{options(k)}</select>
              ) : k === "monitor" ? (
                <select data-testid="monitor-select" value={form[k] ?? ""}
                  onChange={onSelect(k)}>{options(k)}</select>
              ) : (
                <input type="number" step="any" value={form[k]}
                  onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
              )}
            </Field>
          ))}
          {exists ? (
            <button onClick={() => setConfirming(true)} disabled={!form.name}
              data-testid="update-btn">Actualizar «{form.name}»</button>
          ) : (
            <button onClick={() => save()} disabled={!form.name}>Guardar</button>
          )}
          {confirming ? (
            <div className="card" style={{ marginTop: 8 }} data-testid="overwrite-confirm">
              <strong>Se reemplaza la definición de «{form.name}»</strong>
              <p className="sub" style={{ marginTop: 4 }}>
                Los runs y recorridos ya hechos <b>no cambian</b>: copiaron los valores al
                entrenar.{" "}
                {(usedBy[form.name] ?? []).length
                  ? <>Pero la fijan por nombre {(usedBy[form.name] ?? []).length === 1
                      ? "el estudio" : "los estudios"}{" "}
                    <b>{(usedBy[form.name] ?? []).join(", ")}</b>: sus próximos pasos usarán los
                    valores nuevos.</>
                  : <>Ningún estudio la fija por nombre.</>}
              </p>
              <button onClick={() => save(true)}>Sí, reemplazar</button>{" "}
              <button className="secondary" onClick={() => setConfirming(false)}>Cancelar</button>
            </div>
          ) : null}
        </div>
        <div className="card grow">
          <h3 style={{ marginTop: 0 }}>Guardadas</h3>
          <table className="data" data-testid="recipes-table">
            <thead><tr><th>nombre</th><th>lr</th><th>opt</th><th>batch</th>
              <th>épocas</th><th>λ_pos</th><th>seed</th><th></th></tr></thead>
            <tbody>
              {list.map((r) => (
                <tr key={r.name} onClick={() => setForm({ ...defaults, ...r })}>
                  <td>{r.name}</td><td>{r.lr}</td><td>{r.optimizer ?? "adam"}</td>
                  <td>{r.batch_size}</td><td>{r.epochs}</td>
                  <td>{r.lambda_pos}</td><td>{r.seed}</td>
                  <td><button className="secondary" onClick={(ev) => {
                    ev.stopPropagation();
                    api.del(`/recipes/${r.name}`).then(refresh).catch(setError);
                  }}>borrar</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
