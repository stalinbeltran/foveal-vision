import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Badge, ErrorBox, Field, Working } from "../components/ui";

// H — fix B, build a space over C and/or D. Geometry axes offer the
// CALCULATED ranges ("auto"); the (9) block is active in the form; the budget
// declares its unit. State lives on disk: stop/resume survive restarts.
const GEO_AXES = ["d", "k_center", "k_periph", "s_center", "s_periph"];

export default function Sweeps() {
  const [sweeps, setSweeps] = useState<any[] | null>(null);
  const [wds, setWds] = useState<any[]>([]);
  const [nets, setNets] = useState<any[]>([]);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [trials, setTrials] = useState<any>(null);
  const [form, setForm] = useState<any>({
    name: "", window_dataset: "", base_network: "", base_recipe: "",
    objective: "f1", strategy: "grid", points: 0, epochs: 2,
    axes: { d: true, k_center: false, k_periph: false, s_center: false, s_periph: false },
    lr_list: "",
    lambda_list: "",
  });

  const refresh = () => api.get("/sweeps").then((d) => setSweeps(d.sweeps)).catch(setError);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    api.get("/window-datasets").then((d) => {
      setWds(d.window_datasets);
      if (d.window_datasets[0]) setForm((f: any) => ({ ...f, window_dataset: d.window_datasets[0].name }));
    }).catch(setError);
    api.get("/networks").then((d) => {
      setNets(d.networks);
      if (d.networks[0]) setForm((f: any) => ({ ...f, base_network: d.networks[0].name }));
    }).catch(setError);
    api.get("/recipes").then((d) => {
      setRecipes(d.recipes);
      if (d.recipes[0]) setForm((f: any) => ({ ...f, base_recipe: d.recipes[0].name }));
    }).catch(setError);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!sel) return;
    const load = () => api.get(`/sweeps/${sel}/trials`).then(setTrials).catch(setError);
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [sel]);

  const space: any = {};
  GEO_AXES.forEach((a) => { if (form.axes[a]) space[a] = "auto"; });
  if (form.lr_list.trim())
    space.lr = form.lr_list.split(",").map((s: string) => +s.trim()).filter((v: number) => v > 0);
  if (form.lambda_list.trim())
    space.lambda_pos = form.lambda_list.split(",").map((s: string) => +s.trim());
  const nineViolated = form.objective === "loss" &&
    ["lambda_pos", "pos_weight", "smooth_l1_beta"].some((k) => k in space);

  const launch = async () => {
    setError(null);
    try {
      await api.post("/sweeps", {
        name: form.name, window_dataset: form.window_dataset,
        base_network: form.base_network, base_recipe: form.base_recipe,
        space, strategy: form.strategy, objective: form.objective,
        budget: { points: form.points, epochs: form.epochs },
      });
      await refresh();
      setSel(form.name);
    } catch (e) { setError(e); }
  };

  return (
    <div>
      <h2>Recorridos (H)</h2>
      <p className="sub">Un espacio sobre C y/o D con B fijo → muchos runs, sin intervención humana.
        Los ejes de geometría usan los rangos calculados; el estado vive en disco y se reanuda.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card" style={{ width: 330 }}>
          <h3 style={{ marginTop: 0 }}>Nueva receta de recorrido</h3>
          <Field label="nombre"><input value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="dataset (B, fijo — contrato ⑧)">
            <select value={form.window_dataset}
              onChange={(e) => setForm({ ...form, window_dataset: e.target.value })}>
              {wds.map((w) => <option key={w.name}>{w.name}</option>)}
            </select></Field>
          <Field label="red base (C)">
            <select value={form.base_network}
              onChange={(e) => setForm({ ...form, base_network: e.target.value })}>
              {nets.map((n) => <option key={n.name}>{n.name}</option>)}
            </select></Field>
          <Field label="receta base (D)">
            <select value={form.base_recipe}
              onChange={(e) => setForm({ ...form, base_recipe: e.target.value })}>
              {recipes.map((r) => <option key={r.name}>{r.name}</option>)}
            </select></Field>
          <Field label="ejes de geometría (rango calculado: 'auto')">
            <div>
              {GEO_AXES.map((a) => (
                <label key={a} style={{ marginRight: 10 }}>
                  <input type="checkbox" checked={form.axes[a]}
                    onChange={(e) => setForm({ ...form, axes: { ...form.axes, [a]: e.target.checked } })} />
                  {" "}{a}
                </label>
              ))}
            </div>
          </Field>
          <Field label="lr (lista, coma)" help="vacío = no se barre">
            <input value={form.lr_list} placeholder="0.001, 0.003"
              onChange={(e) => setForm({ ...form, lr_list: e.target.value })} /></Field>
          <Field label="lambda_pos (lista, coma)">
            <input value={form.lambda_list} placeholder="0.5, 1.0"
              onChange={(e) => setForm({ ...form, lambda_list: e.target.value })} /></Field>
          <Field label="objetivo">
            <select value={form.objective}
              onChange={(e) => setForm({ ...form, objective: e.target.value })}>
              <option>f1</option><option>pos_err_px</option><option>loss</option>
            </select></Field>
          {nineViolated ? (
            <div className="error-box" data-testid="nine-block">
              <span className="code">[objective_varies_with_space]</span> la loss no puede
              rankear un espacio que barre pesos de la pérdida: λ→0 gana por definición.
              <div className="hintline">→ usa f1 o pos_err_px</div>
            </div>
          ) : null}
          <div className="row">
            <div className="grow"><Field label="puntos (0 = todos)">
              <input type="number" value={form.points}
                onChange={(e) => setForm({ ...form, points: +e.target.value })} /></Field></div>
            <div className="grow"><Field label="épocas/punto">
              <input type="number" value={form.epochs}
                onChange={(e) => setForm({ ...form, epochs: +e.target.value })} /></Field></div>
          </div>
          <p className="sub">Workers: 1 en CPU (torch ya usa todos los núcleos).</p>
          <button onClick={launch} disabled={!form.name || nineViolated}>Lanzar</button>
        </div>
        <div className="card grow">
          <h3 style={{ marginTop: 0 }}>Recorridos</h3>
          <Working on={!sweeps} />
          {sweeps ? (
            <table className="data" data-testid="sweeps-table">
              <thead><tr><th>nombre</th><th>estado</th><th>progreso</th><th>objetivo</th>
                <th></th></tr></thead>
              <tbody>
                {sweeps.map((s) => (
                  <tr key={s.name} className={sel === s.name ? "sel" : ""}
                      onClick={() => setSel(s.name)}>
                    <td>{s.name}</td>
                    <td><Badge status={s.state.status} /></td>
                    <td>{s.state.done ?? 0}/{s.state.total ?? s.spec.points?.length ?? "?"}</td>
                    <td>{s.spec.objective}</td>
                    <td>
                      <button className="secondary" onClick={(ev) => {
                        ev.stopPropagation();
                        api.post(`/sweeps/${s.name}/stop`).then(refresh).catch(setError);
                      }}>parar</button>{" "}
                      <button className="secondary" onClick={(ev) => {
                        ev.stopPropagation();
                        api.post(`/sweeps/${s.name}/resume`).then(refresh).catch(setError);
                      }}>reanudar</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {sel && trials ? (
            <div style={{ marginTop: 14 }}>
              <h4>{sel} — ranking por {trials.objective} ({trials.direction})</h4>
              <table className="data" data-testid="trials-table">
                <thead><tr><th>#</th><th>run</th><th>punto</th><th>{trials.objective}</th>
                  <th>estado</th><th>s/época</th></tr></thead>
                <tbody>
                  {trials.trials.map((t: any, i: number) => (
                    <tr key={t.trial}>
                      <td>{i + 1}</td>
                      <td><Link to={`/runs/${t.run}`}>{t.run}</Link></td>
                      <td className="mono">{JSON.stringify(t.point)}</td>
                      <td>{t.value?.toFixed ? t.value.toFixed(4) : t.value ?? "—"}</td>
                      <td><Badge status={t.status} /></td>
                      <td>{t.seconds_per_epoch ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {trials.discarded?.length ? (
                <p className="sub">{trials.discarded.length} puntos descartados por geometría
                  (con su razón en el spec) — los asserts matan esas combinaciones.</p>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
