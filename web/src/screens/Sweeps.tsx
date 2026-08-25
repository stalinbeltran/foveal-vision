import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ACTIVE_STATES, api, isTerminal } from "../api";
import { usePersistedState } from "../uiState";
import { Badge, ErrorBox, Field, Working } from "../components/ui";
import { SweepCurves } from "../components/SweepCurves";
import { WinnerVerdict } from "../components/WinnerVerdict";
import { TaskScore } from "../components/TaskScore";

// H — fix B, build a space over C and/or D. Geometry axes offer the
// CALCULATED ranges ("auto"); the (9) block is active in the form; the budget
// declares its unit. State lives on disk: stop/resume survive restarts.
//
// The axis/objective vocabulary comes from /sweeps/axes, which serves the same
// constants the validators use. This screen used to keep its own GEO_AXES list
// and its own <option>s: a copy that only diverges the day someone adds an axis
// in Python — and then the form silently cannot reach it.

export default function Sweeps() {
  const [sweeps, setSweeps] = useState<any[] | null>(null);
  const [wds, setWds] = useState<any[]>([]);
  const [nets, setNets] = useState<any[]>([]);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [trials, setTrials] = useState<any>(null);
  const [winner, setWinner] = useState<any>(null);
  const [axesMeta, setAxesMeta] = useState<any>(null);
  const [curves, setCurves] = useState<Record<string, any[]>>({});
  const curvesRef = useRef<Record<string, any[]>>({});
  // runs whose curve is final: fetched WITH a terminal status, never re-fetched
  const settledRef = useRef<Set<string>>(new Set());
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
    api.get("/sweeps/axes").then(setAxesMeta).catch(setError);
    return () => clearInterval(t);
  }, []);

  // the geometry axes (those whose range is CALCULATED) come from the backend
  const GEO_AXES: string[] = axesMeta?.geometry_auto ?? [];

  // Poll the trials AND fan out per-run metrics on the same 3s cadence, so a
  // running sweep animates its curve overlay. Curves live in a ref so the tick
  // sees the previous fill without re-subscribing the effect.
  //
  // A run's curve is only frozen ("settled") after it has been fetched WITH its
  // status already terminal. Caching on "terminal + we have something" was the
  // measured bug: the copy we had was taken by the previous tick, while the run
  // was still training, so every epoch trained between that tick and the end was
  // dropped from the chart for good (up to a whole minute of them if the tab was
  // backgrounded, since browsers throttle setInterval there). The loop writes the
  // last metrics line BEFORE flipping status to done, so a fetch made once the
  // status reads terminal always sees the complete file — one extra request per
  // run, exactly once.
  useEffect(() => {
    if (!sel) return;
    let alive = true;
    curvesRef.current = {}; setCurves({}); settledRef.current = new Set();
    setWinner(null);
    const load = async () => {
      let t: any;
      try { t = await api.get(`/sweeps/${sel}/trials`); }
      catch (e) { if (alive) setError(e); return; }
      if (!alive) return;
      setTrials(t);
      const runs = (t.trials || []).filter((r: any) => r.status);
      await Promise.all(runs.map(async (r: any) => {
        if (settledRef.current.has(r.run)) return;
        try {
          const m = await api.get(`/runs/${r.run}/metrics?since=0`);
          curvesRef.current[r.run] = m.records;
          if (isTerminal(r.status)) settledRef.current.add(r.run);
        } catch { /* a run mid-write or just gone: keep what we had, retry next tick */ }
      }));
      if (alive) setCurves({ ...curvesRef.current });
      // the verdict rides the same tick: δ omitted so the backend measures it
      // from the seeds (1-SE). 404/no-scored-trials just means "not yet".
      try {
        const w = await api.get(`/sweeps/${sel}/winner?cost_metric=seconds_per_epoch`);
        if (alive) setWinner(w);
      } catch { if (alive) setWinner(null); }
    };
    load();
    const iv = setInterval(load, 3000);
    return () => { alive = false; clearInterval(iv); };
  }, [sel]);

  // The NN of each point = base network + the "punto". The base is the part FIXED
  // across all points, so show it once (name + resolved config + derived dims via
  // the same validator as Redes); each row's "punto" column carries what varies.
  const selSweep = (sweeps ?? []).find((s) => s.name === sel);
  const baseNet = selSweep?.spec?.base_network_value ?? null;
  const baseNetName = selSweep?.spec?.base_network ?? null;
  // inline base (D-H2): no name — the synthetic base_label is the grouping key
  const baseLabel = selSweep?.spec?.base_label ?? null;
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
  // contract (9) previewed with the SAME list the validator uses (/sweeps/axes),
  // not a copy of it: a weight added there must light this warning up by itself
  const nineViolated = form.objective === "loss" &&
    (axesMeta?.loss_weight_params ?? []).some((k: string) => k in space);

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
  // group by the base's identity: its name, or the synthetic label when inline
  const baseKeyOf = (s: any) => s.spec.base_network ?? s.spec.base_label ?? null;
  const sopts = {
    window_dataset: sdistinct((s) => s.spec.window_dataset),
    base_network: sdistinct(baseKeyOf),
    base_recipe: sdistinct((s) => s.spec.base_recipe),
    objective: sdistinct((s) => s.spec.objective),
  };
  const ACTIVE = ACTIVE_STATES as readonly string[];
  const sfiltered = allSweeps.filter((s) => {
    if (sf.window_dataset && s.spec.window_dataset !== sf.window_dataset) return false;
    if (sf.base_network && baseKeyOf(s) !== sf.base_network) return false;
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
      <h2 data-domain="H">Recorridos (H)</h2>
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
              {(axesMeta?.objectives ?? []).map((o: string) => <option key={o}>{o}</option>)}
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
                  <strong>red base (C): {baseNetName ?? (baseLabel ? `${baseLabel} (inline)` : "—")}</strong>
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
                    dims: fóvea {baseDims.fovea_px}px · borde {baseDims.border_px}px
                    {" "}en {baseDims.border_cells} celdas · solape {baseDims.overlap_fovea_px}px
                    {" "}dentro / {baseDims.overlap_border_px}px fuera · entrada {baseDims.N}×{baseDims.N}
                    {" "}· recorte {baseDims.original_size}px
                  </div>
                ) : null}
                <div className="sub" style={{ margin: "5px 0 0" }}>
                  ejes barridos (varían por fila, columna «punto»):{" "}
                  <span className="mono">{spaceKeys.length ? spaceKeys.join(", ") : "—"}</span>
                </div>
              </div>
              <p className="sub" style={{ margin: "0 0 8px" }}>
                El valor es el del <strong>checkpoint que sobrevive</strong> (`best.pt`,
                elegido por <span className="mono">{(trials.monitors || []).join(", ") || "—"}</span>),
                no el de la última época — es el fichero que cargan Diagnóstico y
                Predecir, y el que arrastra un estudio. La columna «época» dice de
                dónde sale cada número.
              </p>
              {trials.monitor_matches_objective === false ? (
                <div className="warn" data-testid="monitor-mismatch" style={{
                  margin: "0 0 10px", padding: "8px 12px", borderRadius: 8,
                  background: "var(--surface-2)", border: "1px solid var(--warn)",
                }}>
                  <strong>El monitor y el objetivo no coinciden.</strong>{" "}
                  <span className="mono">{(trials.monitors || []).join(", ")}</span> elige
                  qué época se guarda; el ranking la mide con{" "}
                  <span className="mono">{trials.objective}</span>. Es legal, pero el
                  ganador se elige entre checkpoints seleccionados por otro criterio:
                  si lo que te importa es <span className="mono">{trials.objective}</span>,
                  entrena con <span className="mono">monitor: val_{trials.objective}</span>.
                </div>
              ) : null}
              <table className="data" data-testid="trials-table">
                <thead><tr><th>#</th><th>run</th><th>punto</th><th>{trials.objective}</th>
                  <th>época</th><th>última</th><th>estado</th><th>s/época</th></tr></thead>
                <tbody>
                  {trials.trials.map((t: any, i: number) => (
                    <tr key={t.trial}>
                      <td>{i + 1}</td>
                      <td><Link to={`/runs/${t.run}`}>{t.run}</Link></td>
                      <td className="mono">{JSON.stringify(t.point)}</td>
                      <td title={t.value_reason?.message ?? ""}>
                        {t.value?.toFixed ? t.value.toFixed(4) : t.value ?? "—"}</td>
                      <td className="sub" style={{ margin: 0 }}>
                        {t.epoch != null ? `${t.epoch}/${t.epochs}` : "—"}</td>
                      <td className="sub" style={{ margin: 0 }}>
                        {t.value_last?.toFixed ? t.value_last.toFixed(4) : "—"}</td>
                      <td><Badge status={t.status} /></td>
                      <td>{t.seconds_per_epoch ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {winner ? (
                <div className="card" style={{ marginTop: 12 }} data-testid="sweep-winner">
                  <h4 style={{ margin: "0 0 8px" }}>Veredicto — agregado por valor de eje
                    (media de las semillas)</h4>
                  <WinnerVerdict winner={winner} />
                  {/* the task metric ONLY for the suggested point (and the best
                      one when it is another point): measuring the 35 points of a
                      sweep is the difference between 0,6 s and half a minute,
                      and the ranking does not use this number anyway */}
                  <div style={{ marginTop: 12, borderTop: "1px solid var(--border)",
                                paddingTop: 10 }}>
                    <TaskScore runs={winner.suggested?.runs ?? []}
                               title="Medir la tarea del ganador sugerido" />
                    {JSON.stringify(winner.best?.point) !==
                      JSON.stringify(winner.suggested?.point) ? (
                      <div style={{ marginTop: 10 }}>
                        <TaskScore runs={winner.best?.runs ?? []}
                                   title="…y la del mejor por objetivo" />
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
              {trials.discarded?.length ? (
                <p className="sub">{trials.discarded.length} puntos descartados por geometría
                  (con su razón en el spec) — los asserts matan esas combinaciones.</p>
              ) : null}
              <SweepCurves key={sel} trials={trials} curves={curves} />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
