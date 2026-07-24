import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { Badge, ErrorBox, Field, Working } from "../components/ui";

// I — the OAT study: an ordered plan of axes over H with B fixed. It GUIDES,
// it does NOT execute (D-H1): per step it derives the base from the problem,
// carries the winners, and generates a sweep; the WINNER is the user's to
// confirm (cost/quality rule, D-W1). The chain is dynamic — a winning n_layers
// unlocks channels[i] sub-steps.

function parseRange(text: string): any {
  const t = text.trim();
  if (t === "auto") return "auto";
  return t.split(",").map((s) => {
    const n = Number(s.trim());
    return Number.isNaN(n) ? s.trim() : n;
  }).filter((v) => v !== "");
}

export default function Studies() {
  const [list, setList] = useState<any[] | null>(null);
  const [wds, setWds] = useState<any[]>([]);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [winner, setWinner] = useState<any>(null);
  const [delta, setDelta] = usePersistedState("studies.delta", 0);
  const [costMetric, setCostMetric] = usePersistedState("studies.cost", "seconds_per_epoch");
  const [form, setForm] = usePersistedState<any>("studies.form", {
    name: "", window_dataset: "", base_recipe: "", objective: "f1", seeds: 3, epochs: 2,
    axes: [{ axis: "n_layers", range: "1, 2, 3" }],
  });

  const refresh = () => api.get("/studies").then((d) => setList(d.studies)).catch(setError);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    api.get("/window-datasets").then((d) => {
      setWds(d.window_datasets);
      if (d.window_datasets[0])
        setForm((f: any) => f.window_dataset ? f : { ...f, window_dataset: d.window_datasets[0].name });
    }).catch(setError);
    api.get("/recipes").then((d) => {
      setRecipes(d.recipes);
      if (d.recipes[0])
        setForm((f: any) => f.base_recipe ? f : { ...f, base_recipe: d.recipes[0].name });
    }).catch(setError);
    return () => clearInterval(t);
  }, []);

  // poll the selected study; when a step awaits confirmation, fetch its winner
  useEffect(() => {
    if (!sel) { setDetail(null); return; }
    const load = () => api.get(`/studies/${sel}`).then(setDetail).catch(setError);
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [sel]);

  const awaiting = detail?.awaiting_confirmation ?? null;
  const awaitingSweep = awaiting?.sweep ?? null;
  useEffect(() => {
    setWinner(null);
    if (!awaitingSweep) return;
    api.get(`/sweeps/${awaitingSweep}/winner?delta=${delta}&cost_metric=${costMetric}`)
      .then(setWinner).catch(() => setWinner(null));
  }, [awaitingSweep, delta, costMetric]);

  const create = async () => {
    setError(null);
    try {
      await api.post("/studies", {
        name: form.name, window_dataset: form.window_dataset,
        base_recipe: form.base_recipe, objective: form.objective,
        seeds: form.seeds, budget: { epochs: form.epochs },
        axes: form.axes.filter((a: any) => a.axis.trim())
          .map((a: any) => ({ axis: a.axis.trim(), range: parseRange(a.range) })),
      });
      await refresh();
      setSel(form.name);
      setForm((f: any) => ({ ...f, name: "" }));
    } catch (e) { setError(e); }
  };

  const advance = () => {
    setError(null);
    api.post(`/studies/${sel}/advance`, {}).then(() => api.get(`/studies/${sel}`).then(setDetail))
      .catch(setError);
  };
  const confirm = (point: any) => {
    setError(null);
    api.post(`/studies/${sel}/confirm`, { point }).then(() => {
      setWinner(null);
      api.get(`/studies/${sel}`).then(setDetail);
    }).catch(setError);
  };
  const remove = (name: string) => {
    if (!window.confirm(`¿Borrar el estudio '${name}'? Los recorridos generados quedan.`)) return;
    api.del(`/studies/${name}`).then(() => { if (sel === name) setSel(null); refresh(); }).catch(setError);
  };

  const setAxis = (i: number, k: string, v: string) =>
    setForm((f: any) => ({ ...f, axes: f.axes.map((a: any, j: number) => j === i ? { ...a, [k]: v } : a) }));
  const addAxis = () => setForm((f: any) => ({ ...f, axes: [...f.axes, { axis: "", range: "auto" }] }));
  const rmAxis = (i: number) => setForm((f: any) => ({ ...f, axes: f.axes.filter((_: any, j: number) => j !== i) }));

  return (
    <div>
      <h2>Estudios OAT (I)</h2>
      <p className="sub">Un plan ordenado de ejes sobre recorridos, con B fijo. Deriva la base del
        problema, arrastra el ganador y expande sub-ejes (channels[i]) al fijar n_layers.
        Guía paso a paso; el ganador lo confirmas tú.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card" style={{ width: 340 }}>
          <h3 style={{ marginTop: 0 }}>Nuevo estudio</h3>
          <Field label="nombre"><input value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="dataset (B, fijo)">
            <select value={form.window_dataset}
              onChange={(e) => setForm({ ...form, window_dataset: e.target.value })}>
              {wds.map((w) => <option key={w.name}>{w.name}</option>)}
            </select></Field>
          <Field label="receta base (D)">
            <select value={form.base_recipe}
              onChange={(e) => setForm({ ...form, base_recipe: e.target.value })}>
              {recipes.map((r) => <option key={r.name}>{r.name}</option>)}
            </select></Field>
          <div className="row">
            <div className="grow"><Field label="objetivo">
              <select value={form.objective}
                onChange={(e) => setForm({ ...form, objective: e.target.value })}>
                <option>f1</option><option>pos_err_px</option>
              </select></Field></div>
            <div className="grow"><Field label="semillas (confirmación)">
              <input type="number" value={form.seeds}
                onChange={(e) => setForm({ ...form, seeds: +e.target.value })} /></Field></div>
          </div>
          <Field label="épocas/punto">
            <input type="number" value={form.epochs}
              onChange={(e) => setForm({ ...form, epochs: +e.target.value })} /></Field>
          <Field label="ejes (orden = orden de barrido)">
            <div>
              {form.axes.map((a: any, i: number) => (
                <div className="row" key={i} style={{ marginBottom: 4 }}>
                  <input style={{ width: 110 }} placeholder="eje (channels[i]…)" value={a.axis}
                    onChange={(e) => setAxis(i, "axis", e.target.value)} />
                  <input className="grow" placeholder="auto o 1, 2, 3" value={a.range}
                    onChange={(e) => setAxis(i, "range", e.target.value)} />
                  <button className="linkbtn danger" onClick={() => rmAxis(i)}>×</button>
                </div>
              ))}
              <button className="linkbtn" onClick={addAxis}>+ eje</button>
            </div>
          </Field>
          <button onClick={create} disabled={!form.name || !form.axes.some((a: any) => a.axis.trim())}>
            Crear estudio</button>
        </div>

        <div className="card grow">
          <h3 style={{ marginTop: 0 }}>Estudios</h3>
          <Working on={!list} />
          {list ? (
            <table className="data" data-testid="studies-table">
              <thead><tr><th>nombre</th><th>ejes</th><th>pasos</th><th>siguiente</th><th></th></tr></thead>
              <tbody>
                {list.map((s) => (
                  <tr key={s.name} className={sel === s.name ? "sel" : ""}
                      onClick={() => setSel(s.name)}>
                    <td>{s.name}</td>
                    <td>{s.plan.axes?.length ?? 0}</td>
                    <td>{s.progress.steps?.length ?? 0}</td>
                    <td className="mono">{s.plan.window_dataset}</td>
                    <td><button className="linkbtn danger" onClick={(ev) => {
                      ev.stopPropagation(); remove(s.name);
                    }}>borrar</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {list && !list.length ? <p className="sub">No hay estudios.</p> : null}

          {sel && detail ? (
            <div style={{ marginTop: 14 }} data-testid="study-detail">
              <h4>{sel}</h4>
              <div className="sub">
                ganadores arrastrados:{" "}
                <span className="mono">
                  {Object.keys(detail.winners || {}).length
                    ? Object.entries(detail.winners).map(([k, v]: any) => `${k}=${JSON.stringify(v.value)}`).join("  ")
                    : "—"}
                </span>
              </div>
              <table className="data" style={{ marginTop: 8 }}>
                <thead><tr><th>#</th><th>eje</th><th>recorrido</th><th>base</th>
                  <th>puntos</th><th>ganador</th></tr></thead>
                <tbody>
                  {(detail.steps || []).map((st: any) => (
                    <tr key={st.step}>
                      <td>{st.step}</td>
                      <td className="mono">{st.axis}</td>
                      <td><Link to="/sweeps">{st.sweep}</Link></td>
                      <td className="mono">{st.base_label}</td>
                      <td>{st.points}{st.discarded ? ` (−${st.discarded})` : ""}</td>
                      <td className="mono">{st.confirmed ? JSON.stringify(st.winner) : "…"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {awaiting ? (
                <div className="card" style={{ marginTop: 12 }} data-testid="confirm-box">
                  <strong>Confirmar ganador del paso {awaiting.step} ({awaiting.axis})</strong>
                  <div className="row" style={{ marginTop: 6 }}>
                    <div className="grow"><Field label="δ (margen calidad)">
                      <input type="number" step={0.01} value={delta}
                        onChange={(e) => setDelta(+e.target.value)} /></Field></div>
                    <div className="grow"><Field label="coste">
                      <select value={costMetric} onChange={(e) => setCostMetric(e.target.value)}>
                        <option value="seconds_per_epoch">s/época</option>
                        <option value="num_params">parámetros</option>
                      </select></Field></div>
                  </div>
                  {winner ? (
                    <div>
                      <div className="sub">mejor objetivo:{" "}
                        <span className="mono">{JSON.stringify(winner.best.point)}</span>
                        {" "}({winner.best.value?.toFixed ? winner.best.value.toFixed(4) : winner.best.value})</div>
                      <div className="sub">sugerido (el más barato dentro de δ):{" "}
                        <span className="mono">{JSON.stringify(winner.suggested.point)}</span></div>
                      <button style={{ marginTop: 8 }} onClick={() => confirm(winner.suggested.point)}>
                        Confirmar sugerido y arrastrar</button>
                    </div>
                  ) : (
                    <p className="sub">Esperando a que el recorrido tenga puntos con valor…</p>
                  )}
                </div>
              ) : detail.done ? (
                <p className="sub" style={{ marginTop: 12 }}>Estudio completo.</p>
              ) : (
                <button style={{ marginTop: 12 }} onClick={advance} data-testid="advance-btn">
                  Generar y lanzar siguiente paso{detail.next_axis ? `: ${detail.next_axis}` : ""}
                </button>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
