import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { ErrorBox, Field, Working } from "../components/ui";

// C — the most important screen of this project: editing N and the fractions
// shows the derived dims, the CALCULATED ranges and the zone diagram LIVE
// (FG1), via POST /networks/validate. Broken asserts show with their reason.

const DEFAULTS = { name: "", N: 20, c_frac: 0.8, d: 2, pen_frac: 0.1,
  k_center: 3, k_periph: 3, s_center: 1, s_periph: 1, ch1: 16, ch2: 32,
  merge: "concat", pool_mode: "avg", pad_mode: "edge" };

function ZoneDiagram({ dims }: { dims: any }) {
  const N = dims.N, s = Math.min(12, Math.floor(240 / N));
  const cells = [];
  const po = dims.periph_out, pen = dims.penetration;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const ring = x < po || y < po || x >= N - po || y >= N - po;
      const core = x >= po + pen && y >= po + pen && x < N - po - pen && y < N - po - pen;
      const color = ring ? "var(--corner-bl)" : core ? "var(--corner-tr)" : "var(--warn)";
      cells.push(<rect key={`${x}-${y}`} x={x * s} y={y * s} width={s - 1} height={s - 1}
        fill={color} opacity={ring ? 0.5 : core ? 0.35 : 0.7} />);
    }
  return (
    <div className="zonebox">
      <svg width={N * s} height={N * s} data-testid="zone-diagram">{cells}</svg>
      <div className="cap" style={{ fontSize: 12, color: "var(--text-dim)" }}>
        anillo (solo periferia) · banda de penetración (ambas ramas SUMAN) · núcleo (solo centro)
      </div>
    </div>
  );
}

export default function Networks() {
  const [list, setList] = useState<any[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [form, setForm] = useState<any>(DEFAULTS);
  const [validation, setValidation] = useState<any>(null);

  const refresh = () => api.get("/networks").then((d) => setList(d.networks)).catch(setError);
  useEffect(() => { refresh(); }, []);

  // live validation: the user sees what N and the fractions imply BEFORE saving
  useEffect(() => {
    const t = setTimeout(() => {
      api.post("/networks/validate", form).then(setValidation).catch(setError);
    }, 250);
    return () => clearTimeout(t);
  }, [form]);

  const save = async () => {
    setError(null);
    try { await api.post("/networks", form); await refresh(); }
    catch (e) { setError(e); }
  };

  const num = (k: string, step = 1, help?: string) => (
    <Field label={k} help={help}>
      <input type="number" step={step} value={form[k]}
        onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
    </Field>
  );

  return (
    <div>
      <h2>Redes foveadas (C)</h2>
      <p className="sub">Todo se deriva de N y las fracciones; los rangos de búsqueda se calculan, nunca se escriben.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card" style={{ width: 320 }}>
          <Field label="nombre"><input value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <div className="row">
            <div className="grow">{num("N", 2, "lado de la entrada compuesta (par)")}</div>
            <div className="grow">{num("c_frac", 0.01, "fracción de la fóvea")}</div>
          </div>
          <div className="row">
            <div className="grow">{num("d", 1, "downsample de la periferia")}</div>
            <div className="grow">{num("pen_frac", 0.01, "penetración")}</div>
          </div>
          <div className="row">
            <div className="grow">{num("k_center", 2, "kernel impar")}</div>
            <div className="grow">{num("k_periph", 2)}</div>
          </div>
          <div className="row">
            <div className="grow">{num("s_center")}</div>
            <div className="grow">{num("s_periph")}</div>
          </div>
          <div className="row">
            <div className="grow">{num("ch1")}</div>
            <div className="grow">{num("ch2")}</div>
          </div>
          <Field label="merge" help="concat tolera strides distintos; sum exige iguales">
            <select value={form.merge} onChange={(e) => setForm({ ...form, merge: e.target.value })}>
              <option>concat</option><option>sum</option>
            </select></Field>
          <Field label="pool_mode" help="cómo se reduce la periferia (eje a barrer)">
            <select value={form.pool_mode} onChange={(e) => setForm({ ...form, pool_mode: e.target.value })}>
              <option>avg</option><option>max</option>
            </select></Field>
          <button onClick={save} disabled={!form.name || !validation?.valid}>Guardar</button>
        </div>
        <div className="card grow" data-testid="validate-panel">
          <h3 style={{ marginTop: 0 }}>Lo que implica (en vivo)</h3>
          {!validation ? <Working on /> : validation.valid ? (
            <div className="row">
              <div>
                <dl className="kv">
                  <dt>fóvea (ventana etiquetada)</dt><dd>{validation.trace.dims.center_out}px</dd>
                  <dt>anillo</dt><dd>{validation.trace.dims.periph_out}px (ve {validation.trace.dims.periph_real}px reales)</dd>
                  <dt>penetración</dt><dd>{validation.trace.dims.penetration}px</dd>
                  <dt>recorte original</dt><dd>{validation.trace.dims.original_size}px</dd>
                  <dt>salida ramas</dt><dd>c {validation.trace.branch_out.center.join("×")} · p {validation.trace.branch_out.periph.join("×")}</dd>
                  <dt>parámetros</dt><dd>{validation.trace.num_params.toLocaleString()}</dd>
                </dl>
                <h4>Rangos calculados (los que usará un recorrido con "auto")</h4>
                <dl className="kv">
                  {Object.entries(validation.ranges).map(([k, v]: any) => (
                    <React.Fragment key={k}><dt>{k}</dt><dd>[{v.join(", ")}]</dd></React.Fragment>
                  ))}
                </dl>
              </div>
              <ZoneDiagram dims={validation.trace.dims} />
            </div>
          ) : (
            <div>
              {validation.problems.map((p: any, i: number) => (
                <div className="error-box" key={i}>
                  <span className="code">[{p.code}]</span> {p.message}
                  <div className="hintline">→ {p.hint}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Guardadas</h3>
        <table className="data" data-testid="networks-table">
          <thead><tr><th>nombre</th><th>N</th><th>c_frac</th><th>d</th><th>kernels</th>
            <th>strides</th><th>merge</th><th></th></tr></thead>
          <tbody>
            {list.map((n) => (
              <tr key={n.name} onClick={() => setForm({ ...DEFAULTS, ...n })}>
                <td>{n.name}</td><td>{n.N}</td><td>{n.c_frac}</td><td>{n.d}</td>
                <td>{n.k_center}/{n.k_periph}</td><td>{n.s_center}/{n.s_periph}</td>
                <td>{n.merge}</td>
                <td><button className="secondary" onClick={(ev) => {
                  ev.stopPropagation();
                  api.del(`/networks/${n.name}`).then(refresh).catch(setError);
                }}>borrar</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
