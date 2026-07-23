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
  monitor: "qué elige best.pt (val_loss | val_f1)",
  seed: "eje de réplica, no un hiperparámetro a optimizar",
};

export default function Recipes() {
  const [list, setList] = useState<any[]>([]);
  const [defaults, setDefaults] = useState<any>(null);
  const [form, setForm] = usePersistedState<any>("recipes.form", { name: "" });
  const [error, setError] = useState<unknown>(null);

  const refresh = () => api.get("/recipes").then((d) => {
    setList(d.recipes);
    setDefaults(d.defaults);
    setForm((f: any) => ({ ...d.defaults, ...f }));
  }).catch(setError);
  useEffect(() => { refresh(); }, []);

  const save = async () => {
    setError(null);
    try {
      const body = { ...form };
      await api.post("/recipes", body);
      await refresh();
    } catch (e) { setError(e); }
  };

  if (!defaults) return <h2>Recetas (D)</h2>;
  const fields = Object.keys(defaults);
  return (
    <div>
      <h2>Recetas (D)</h2>
      <p className="sub">Hiperparámetros que definen el resultado. device y num_workers NO van aquí
        (contrato ⑩): son ejecución, viven en Entrenar.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card" style={{ width: 340 }}>
          <Field label="nombre"><input value={form.name ?? ""}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          {fields.map((k) => (
            <Field key={k} label={k} help={HELP[k]}>
              {k === "optimizer" || k === "scheduler" || k === "monitor" ? (
                <select value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })}>
                  {(k === "optimizer" ? ["adam", "adamw", "sgd"]
                    : k === "scheduler" ? ["none", "cosine"]
                    : ["val_loss", "val_f1"]).map((o) => <option key={o}>{o}</option>)}
                </select>
              ) : (
                <input type="number" step="any" value={form[k]}
                  onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
              )}
            </Field>
          ))}
          <button onClick={save} disabled={!form.name}>Guardar</button>
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
