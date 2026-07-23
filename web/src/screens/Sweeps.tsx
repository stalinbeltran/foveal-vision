import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
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
  const [baseDims, setBaseDims] = useState<any>(null);
  const [sf, setSf] = usePersistedState("sweeps.filters", {
    window_dataset: "", base_network: "", base_recipe: "", objective: "", q: "",
  });
  const [foldDone, setFoldDone] = usePersistedState("sweeps.foldDone", false);
  const [form, setForm] = usePersistedState<any>("sweeps.form", {
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
    // fill a default only when nothing was remembered — a restored value wins
    api.get("/window-datasets").then((d) => {
      setWds(d.window_datasets);
      if (d.window_datasets[0])
        setForm((f: any) => f.window_dataset ? f : { ...f, window_dataset: d.window_datasets[0].name });
    }).catch(setError);
    api.get("/networks").then((d) => {
      setNets(d.networks);
      if (d.networks[0])
        setForm((f: any) => f.base_network ? f : { ...f, base_network: d.networks[0].name });
    }).catch(setError);
    api.get("/recipes").then((d) => {
      setRecipes(d.recipes);
      if (d.recipes[0])
        setForm((f: any) => f.base_recipe ? f : { ...f, base_recipe: d.recipes[0].name });
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

  // The NN of each point = base network + the "punto". The base is the part FIXED
  // across all points, so show it once (name + resolved config + derived dims via
  // the same validator as Redes); each row's "punto" column carries what varies.
  const selSweep = (sweeps ?? []).find((s) => s.name === sel);
  const baseNet = selSweep?.spec?.base_network_value ?? null;
  const baseNetName = selSweep?.spec?.base_network ?? null;
  const baseRecipeName = selSweep?.spec?.base_recipe ?? null;
  const spaceKeys = Object.keys(selSweep?.spec?.space ?? {});
  const baseKey = baseNet ? JSON.stringify(baseNet) : "";
  useEffect(() => {
    if (!baseNet) { setBaseDims(null); return; }
    api.post("/networks/validate", baseNet)
      .then((v) => setBaseDims(v?.trace?.dims ?? null))
      .catch(() => setBaseDims(null));
  }, [baseKey]);

  const space: any = {};
  GEO_AXES.forEach((a) => { if (form.axes[a]) space[a] = "auto"; });
  if (form.lr_list.trim())
    space.lr = form.lr_list.split(",").map((s: string) => +s.trim()).filter((v: number) => v > 0);
  if (form.lambda_list.trim())
    space.lambda_pos = form.lambda_list.split(",").map((s: string) => +s.trim());
  const nineViolated = form.objective === "loss" &&
    ["lambda_pos", "pos_weight", "smooth_l1_beta"].some((k) => k in space);

  const removeSweep = (s: any) => {
    const n = s.state?.done ?? 0;
    if (!window.confirm(
      `¿Borrar el recorrido '${s.name}' y sus runs (${n} completados)? ` +
      `Se borran en cascada — no se puede deshacer.`)) return;
    setError(null);
    api.del(`/sweeps/${s.name}`).then(() => {
      if (sel === s.name) { setSel(null); setTrials(null); }
      refresh();
    }).catch(setError);
  };

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
      setForm((f: any) => ({ ...f, name: "" }));  // sweep name is single-use
    } catch (e) { setError(e); }
  };

  // Facets over the sweep list: same idea as Runs (B/C/D + objetivo + buscar).
  // Estado is not a facet here — it drives the partition into Activos/Terminados.
  const allSweeps = sweeps ?? [];
  const sdistinct = (path: (s: any) => any) =>
    [...new Set(allSweeps.map(path).filter((v) => v != null))].sort() as string[];
  const sopts = {
    window_dataset: sdistinct((s) => s.spec.window_dataset),
    base_network: sdistinct((s) => s.spec.base_network),
    base_recipe: sdistinct((s) => s.spec.base_recipe),
    objective: sdistinct((s) => s.spec.objective),
  };
  const ACTIVE = ["running", "queued"];
  const sfiltered = allSweeps.filter((s) => {
    if (sf.window_dataset && s.spec.window_dataset !== sf.window_dataset) return false;
    if (sf.base_network && s.spec.base_network !== sf.base_network) return false;
    if (sf.base_recipe && s.spec.base_recipe !== sf.base_recipe) return false;
    if (sf.objective && s.spec.objective !== sf.objective) return false;
    if (sf.q && !s.name.toLowerCase().includes(sf.q.toLowerCase())) return false;
    return true;
  });
  const bucketOf = (s: any) => (ACTIVE.includes(s.state?.status) ? 0 : 1);
  const ssorted = [...sfiltered].sort((a, b) =>
    bucketOf(a) - bucketOf(b) || a.name.localeCompare(b.name));
  const activeCount = sfiltered.filter((s) => bucketOf(s) === 0).length;
  const doneCount = sfiltered.length - activeCount;
  const sAnyFilter = Object.values(sf).some((v) => v !== "");
  let lastBucket = -1;

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
          <div className="filters" style={{ marginBottom: 12 }}>
            <select value={sf.window_dataset}
              onChange={(e) => setSf({ ...sf, window_dataset: e.target.value })}>
              <option value="">B: todos</option>
              {sopts.window_dataset.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <select value={sf.base_network}
              onChange={(e) => setSf({ ...sf, base_network: e.target.value })}>
              <option value="">C: todas</option>
              {sopts.base_network.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <select value={sf.base_recipe}
              onChange={(e) => setSf({ ...sf, base_recipe: e.target.value })}>
              <option value="">D: todas</option>
              {sopts.base_recipe.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <select value={sf.objective}
              onChange={(e) => setSf({ ...sf, objective: e.target.value })}>
              <option value="">Objetivo: todos</option>
              {sopts.objective.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <input placeholder="buscar…" value={sf.q}
              onChange={(e) => setSf({ ...sf, q: e.target.value })} />
            {sAnyFilter ? <button className="secondary" onClick={() =>
              setSf({ window_dataset: "", base_network: "", base_recipe: "", objective: "", q: "" })
            }>limpiar</button> : null}
          </div>
          <Working on={!sweeps} />
          {sweeps ? (
            <table className="data" data-testid="sweeps-table">
              <thead><tr><th>nombre</th><th>estado</th><th>progreso</th><th>objetivo</th>
                <th></th></tr></thead>
              <tbody>
                {ssorted.flatMap((s) => {
                  const bucket = bucketOf(s);
                  const rows: React.ReactNode[] = [];
                  if (bucket !== lastBucket) {
                    lastBucket = bucket;
                    const isDone = bucket === 1;
                    rows.push(
                      <tr key={`hdr-${bucket}`} className="grouprow"
                          onClick={() => isDone && setFoldDone(!foldDone)}
                          style={isDone ? { cursor: "pointer" } : undefined}>
                        <td colSpan={5}>
                          <span className="glabel">{isDone ? "Terminados" : "Activos"}</span>
                          {isDone ? doneCount : activeCount}
                          {isDone ? <span className="sub"> · {foldDone ? "▸ mostrar" : "▾ plegar"}</span> : null}
                        </td>
                      </tr>
                    );
                  }
                  if (bucket === 1 && foldDone) return rows;
                  rows.push(
                    <tr key={s.name} className={sel === s.name ? "sel" : ""}
                        onClick={() => setSel(s.name)}>
                      <td>{s.name}</td>
                      <td><Badge status={s.state.status} /></td>
                      <td>{s.state.done ?? 0}/{s.state.total ?? s.spec.points?.length ?? "?"}</td>
                      <td>{s.spec.objective}</td>
                      <td className="rowactions">
                        <button className="linkbtn" onClick={(ev) => {
                          ev.stopPropagation();
                          api.post(`/sweeps/${s.name}/stop`).then(refresh).catch(setError);
                        }}>parar</button>
                        <button className="linkbtn" onClick={(ev) => {
                          ev.stopPropagation();
                          api.post(`/sweeps/${s.name}/resume`).then(refresh).catch(setError);
                        }}>reanudar</button>
                        <button className="linkbtn danger" onClick={(ev) => {
                          ev.stopPropagation(); removeSweep(s);
                        }}>borrar</button>
                      </td>
                    </tr>
                  );
                  return rows;
                })}
              </tbody>
            </table>
          ) : null}
          {sweeps && !sfiltered.length ? (
            <p className="sub">{allSweeps.length ? "Ningún recorrido pasa los filtros." : "No hay recorridos."}</p>
          ) : null}
          {sel && trials ? (
            <div style={{ marginTop: 14 }}>
              <h4>{sel} — ranking por {trials.objective} ({trials.direction})</h4>
              <div data-testid="base-nn" style={{
                margin: "0 0 12px", padding: "8px 12px",
                background: "var(--surface-2)", border: "1px solid var(--border)",
                borderRadius: 8,
              }}>
                <div>
                  <strong>red base (C): {baseNetName ?? "—"}</strong>
                  {baseRecipeName ? <span className="sub" style={{ margin: 0 }}>
                    {"  ·  receta base (D): "}{baseRecipeName}</span> : null}
                </div>
                {baseNet ? (
                  <div className="mono" style={{ marginTop: 5, color: "var(--text)" }}>
                    {Object.entries(baseNet)
                      .filter(([k]) => !spaceKeys.includes(k))
                      .map(([k, v]) => `${k}=${v}`).join("   ")}
                  </div>
                ) : null}
                {baseDims ? (
                  <div className="sub" style={{ margin: "5px 0 0" }}>
                    dims: fóvea {baseDims.center_out}px · anillo {baseDims.periph_out}px
                    {" "}(ve {baseDims.periph_real}px reales) · penetración {baseDims.penetration}px
                    {" "}· recorte {baseDims.original_size}px
                  </div>
                ) : null}
                <div className="sub" style={{ margin: "5px 0 0" }}>
                  ejes barridos (varían por fila, columna «punto»):{" "}
                  <span className="mono">{spaceKeys.length ? spaceKeys.join(", ") : "—"}</span>
                </div>
              </div>
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
