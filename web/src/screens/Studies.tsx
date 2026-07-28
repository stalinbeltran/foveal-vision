import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { Badge, ErrorBox, Field, Working } from "../components/ui";
import { WinnerVerdict } from "../components/WinnerVerdict";

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

// U1.6 — the definition an object was created with is re-readable in its
// detail: listing is not verifying. Three things this block is careful about:
// it reads the SAVED plan (GET /studies/{name}), never the remembered creation
// form (U7.3); it stays apart from progress (plan.json is what was asked,
// progress.json is what happened); and it prints the range of an axis as its
// LIST, not as its length.

const CHANNELS_AT = /^channels\[\d+\]$/;

// a plan axis vs a concrete one: `channels[i]` is a placeholder that the
// winning n_layers expands into channels[0..L-1] (driver §6.1), so the plan
// row must recognise its own sub-steps.
const axisMatches = (planAxis: string, concrete: string) =>
  planAxis === concrete || (planAxis === "channels[i]" && CHANNELS_AT.test(concrete));

const rangeText = (range: any) =>
  range === "auto" || range == null
    ? { text: "auto", note: "rango calculado al generar el paso" }
    : Array.isArray(range)
      ? { text: range.join(", "), note: `${range.length} ${range.length === 1 ? "punto" : "puntos"}` }
      : { text: JSON.stringify(range), note: "" };

// absent is drawn as absent, never as 0 nor as blank (U5.3)
const orDash = (v: any) => (v === undefined || v === null || v === "" ? "—" : String(v));

// the scalar plan fields this block names one by one. Anything else the plan
// carries is printed too (see «otros campos»): a field added in Python must
// not become invisible here — that is exactly how the two copies diverge.
const NAMED_PLAN_FIELDS = ["window_dataset", "base_recipe", "objective", "seeds",
                           "budget", "axes", "format_version"];

function StudyPlan({ plan, progress }: { plan: any; progress: any }) {
  const steps: any[] = progress?.steps ?? [];
  const queue: any[] = progress?.queue ?? [];
  const axes: any[] = plan?.axes ?? [];
  const epochs = plan?.budget?.epochs;
  const extra = Object.entries(plan ?? {}).filter(([k]) => !NAMED_PLAN_FIELDS.includes(k));

  // the ladder's state per DECLARED axis, read off the live progress: what is
  // still in the queue is pending, what produced steps is running or done.
  const stateOf = (axis: string) => {
    const mine = steps.filter((s) => axisMatches(axis, s.axis));
    const pending = queue.some((q: any) => axisMatches(axis, q.axis));
    if (!mine.length) return pending ? "pendiente" : "—";
    if (pending || mine.some((s) => !s.confirmed)) return "en curso";
    return "hecho";
  };

  const cell = { padding: "2px 14px 2px 0", verticalAlign: "top" as const };
  return (
    <div data-testid="study-plan" style={{
      margin: "0 0 12px", padding: "8px 12px",
      background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 8,
    }}>
      <strong>El plan (lo que pediste)</strong>
      <table style={{ marginTop: 6, borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={cell} className="sub">dataset (B, fijo)</td>
            <td style={cell} className="mono">{orDash(plan?.window_dataset)}</td>
            <td style={cell} className="sub">objetivo</td>
            <td style={cell} className="mono">{orDash(plan?.objective)}</td>
          </tr>
          <tr>
            <td style={cell} className="sub">receta base (D)</td>
            <td style={cell} className="mono">{orDash(plan?.base_recipe)}</td>
            <td style={cell} className="sub">semillas (confirmación)</td>
            <td style={cell} className="mono">{orDash(plan?.seeds)}</td>
          </tr>
          <tr>
            <td style={cell} className="sub">presupuesto</td>
            {/* con su unidad: un número pelado no dice de qué es presupuesto */}
            <td style={cell} className="mono" colSpan={3}>
              {epochs == null ? "—" : `${epochs} épocas/punto`}</td>
          </tr>
          {extra.map(([k, v]) => (
            <tr key={k}>
              <td style={cell} className="sub">{k}</td>
              <td style={cell} className="mono" colSpan={3}>{JSON.stringify(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="sub" style={{ margin: "8px 0 4px" }}>
        ejes (orden = orden de barrido)
      </div>
      <table className="data" data-testid="study-axes">
        <thead><tr><th>#</th><th>eje</th><th>rango</th><th>estado</th></tr></thead>
        <tbody>
          {axes.map((a: any, i: number) => {
            const r = rangeText(a.range);
            return (
              <tr key={i}>
                <td>{i + 1}</td>
                <td className="mono">{a.axis}
                  {a.depends_on ? <span className="sub"> (tras {a.depends_on})</span> : null}</td>
                <td className="mono">{r.text}
                  {r.note ? <span className="sub" style={{ margin: 0 }}>{"  "}({r.note})</span> : null}</td>
                <td>{stateOf(a.axis)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!axes.length ? <p className="sub">El plan no declara ejes.</p> : null}
    </div>
  );
}

export default function Studies() {
  const [list, setList] = useState<any[] | null>(null);
  const [wds, setWds] = useState<any[]>([]);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [axesMeta, setAxesMeta] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [winner, setWinner] = useState<any>(null);
  // "" = auto: the backend derives δ from the seed dispersion it measured
  // (1-SE, protocolo §1.5). A number overrides it.
  const [delta, setDelta] = usePersistedState<string>("studies.delta", "");
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
    api.get("/sweeps/axes").then(setAxesMeta).catch(setError);
    return () => clearInterval(t);
  }, []);

  // the axis vocabulary comes from the backend (single source of truth): C
  // structure fields + D training fields + the special channels[i] sub-axis.
  // channels[i] supersedes raw `channels` as the OAT way to sweep width.
  const geoAuto = new Set<string>(axesMeta?.geometry_auto ?? []);
  const cAxes = ((axesMeta?.network ?? []) as string[]).filter((a) => a !== "channels");
  const dAxes = (axesMeta?.recipe ?? []) as string[];
  const defaultRange = (axis: string) =>
    geoAuto.has(axis) ? "auto" : axis === "channels[i]" ? "8, 16, 32" : "";

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
    // δ omitted on purpose when the box is empty: the backend then measures it
    // from the seeds instead of trusting a constant nobody remembers to set
    const d = delta.trim() === "" ? "" : `delta=${Number(delta)}&`;
    api.get(`/sweeps/${awaitingSweep}/winner?${d}cost_metric=${costMetric}`)
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
  // picking the axis also seeds a sensible range: 'auto' for calculated-range
  // geometry, a width list for channels[i], empty (a list is required) otherwise
  const pickAxis = (i: number, axis: string) =>
    setForm((f: any) => ({ ...f, axes: f.axes.map((a: any, j: number) =>
      j === i ? { axis, range: defaultRange(axis) } : a) }));
  const addAxis = () => setForm((f: any) => ({ ...f, axes: [...f.axes, { axis: "", range: "" }] }));
  const rmAxis = (i: number) => setForm((f: any) => ({ ...f, axes: f.axes.filter((_: any, j: number) => j !== i) }));

  return (
    <div>
      <h2 data-domain="I">Estudios OAT (I)</h2>
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
                {/* mismo vocabulario que Recorridos y que el validador: esta
                    pantalla ofrecía dos objetivos y la otra tres, y nadie podía
                    saber si la diferencia era deliberada. Ahora 'loss' se ofrece
                    y validate_plan rechaza la combinación que rompe el ⑨. */}
                {(axesMeta?.objectives ?? []).map((o: string) => <option key={o}>{o}</option>)}
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
                  <select style={{ width: 130 }} value={a.axis} data-testid="axis-select"
                    onChange={(e) => pickAxis(i, e.target.value)}>
                    <option value="">— eje —</option>
                    <optgroup label="Estructura (C)">
                      {cAxes.map((x) => <option key={x} value={x}>{x}</option>)}
                      <option value="channels[i]">channels[i]</option>
                    </optgroup>
                    <optgroup label="Entrenamiento (D)">
                      {dAxes.map((x) => <option key={x} value={x}>{x}</option>)}
                    </optgroup>
                  </select>
                  <input className="grow"
                    placeholder={geoAuto.has(a.axis) ? "auto o 1, 2, 3" : "1, 2, 3"}
                    value={a.range} onChange={(e) => setAxis(i, "range", e.target.value)} />
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
              <thead><tr><th>nombre</th><th>dataset (B)</th><th>ejes</th><th>pasos</th>
                <th>siguiente</th><th></th></tr></thead>
              <tbody>
                {list.map((s) => (
                  <tr key={s.name} className={sel === s.name ? "sel" : ""}
                      onClick={() => setSel(s.name)}>
                    <td>{s.name}</td>
                    <td className="mono">{s.plan.window_dataset}</td>
                    <td>{s.plan.axes?.length ?? 0}</td>
                    <td>{s.progress.steps?.length ?? 0}</td>
                    {/* «siguiente» = lo que el estudio espera, del MISMO campo que
                        usa el detalle (summarize): eje pendiente, confirmación, o
                        completo. Antes esta celda pintaba el dataset. */}
                    <td className="mono">
                      {s.awaiting_confirmation ? "confirmar ganador"
                        : s.next_axis ? s.next_axis
                        : s.done ? "completo" : "—"}</td>
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
              {/* U1.6: primero la definición (plan.json, comiteable), después
                  el progreso (progress.json, estado vivo). Mezclados no se
                  puede distinguir lo pedido de lo ocurrido. */}
              <StudyPlan plan={detail.plan} progress={detail.progress} />
              <strong>El progreso (lo que ha pasado)</strong>
              <div className="sub" style={{ marginTop: 4 }}>
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
                    <div className="grow"><Field label="δ (margen calidad)"
                      help="vacío = medido de las semillas (1-SE); un número lo fija">
                      <input type="number" step={0.01} value={delta} placeholder="auto"
                        onChange={(e) => setDelta(e.target.value)} /></Field></div>
                    <div className="grow"><Field label="coste">
                      <select value={costMetric} onChange={(e) => setCostMetric(e.target.value)}>
                        <option value="seconds_per_epoch">s/época</option>
                        <option value="num_params">parámetros</option>
                      </select></Field></div>
                  </div>
                  {winner ? (
                    <div>
                      <WinnerVerdict winner={winner} />
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
