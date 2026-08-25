import React, { useEffect, useState } from "react";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field, Working } from "../components/ui";

// C — the most important screen of this project: editing the geometry (real px:
// fovea, border, how the border is reduced, and how much of each region the two
// branches share) shows the derived dims, the CALCULATED ranges and the zone
// diagram LIVE (FG1), via POST /networks/validate. `N` is DERIVED and shown as
// such: it is not a field anybody types (reparameterisation 2026-08-25).

// The C defaults are NOT written here: /networks serves them, resolved by the
// same full_config the builder uses. The copy that used to live in this file had
// already drifted (channels [16,16] vs the derived [16]*n_layers). Until they
// arrive the form is empty-but-typed, so nothing renders a made-up number.
const EMPTY = { name: "" } as any;

function ZoneDiagram({ dims }: { dims: any }) {
  const N = dims.N, s = Math.min(12, Math.floor(240 / N));
  const cells = [];
  // the two overlaps are independent: the border branch reaches `pen` cells INTO
  // the fovea, the fovea branch reaches `ob` cells OUT over the border
  const po = dims.border_cells, pen = dims.overlap_fovea_px;
  const ob = dims.overlap_border_cells ?? 0;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++) {
      const ring = (x < po - ob || y < po - ob || x >= N - po + ob || y >= N - po + ob);
      const core = x >= po + pen && y >= po + pen && x < N - po - pen && y < N - po - pen;
      const color = ring ? "var(--corner-bl)" : core ? "var(--corner-tr)" : "var(--warn)";
      cells.push(<rect key={`${x}-${y}`} x={x * s} y={y * s} width={s - 1} height={s - 1}
        fill={color} opacity={ring ? 0.5 : core ? 0.35 : 0.7} />);
    }
  return (
    <div className="zonebox">
      <svg width={N * s} height={N * s} data-testid="zone-diagram"
        data-view="FG1" data-fixes="C" data-varies="-" data-measures="la geometria derivada">{cells}</svg>
      <div className="cap" style={{ fontSize: 12, color: "var(--text-dim)" }}>
        anillo (solo periferia) · banda de penetración (ambas ramas SUMAN) · núcleo (solo centro)
      </div>
    </div>
  );
}

export default function Networks() {
  const [list, setList] = useState<any[]>([]);
  const [defaults, setDefaults] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [form, setForm] = usePersistedState<any>("networks.form", EMPTY);
  const [validation, setValidation] = useState<any>(null);
  const [confirming, setConfirming] = useState(false);
  const ready = form.fovea_px != null;  // the served defaults (or a remembered form) landed

  const refresh = () => api.get("/networks").then((d) => {
    setList(d.networks);
    setDefaults(d.defaults);
    // fill only what the form does not have yet: a remembered edit wins over the
    // defaults, and the defaults win over nothing — never over the user
    setForm((f: any) => ({ ...d.defaults, ...f }));
  }).catch(setError);
  useEffect(() => { refresh(); }, []);

  // live validation: the user sees what N and the fractions imply BEFORE saving
  useEffect(() => {
    if (!ready) return;           // no half-form probing: it would 400 on nothing
    const t = setTimeout(() => {
      api.post("/networks/validate", form).then(setValidation).catch(setError);
    }, 250);
    return () => clearTimeout(t);
  }, [form, ready]);

  // C, como D, es fuente y se edita (un run no, U5.8). Sobrescribir se pide
  // aparte: sin esto el único camino era el 409 «elige otro nombre», que no es
  // lo que hace quien abre una red guardada para cambiarle un número.
  const exists = list.some((n) => n.name === form.name);

  const save = async (overwrite = false) => {
    setError(null);
    try {
      await api.post("/networks", overwrite ? { ...form, overwrite: true } : form);
      setConfirming(false);
      await refresh();
    } catch (e) { setError(e); }
  };

  const num = (k: string, step = 1, help?: string) => (
    <Field label={k} help={help}>
      <input type="number" step={step} value={form[k]}
        onChange={(e) => setForm({ ...form, [k]: +e.target.value })} />
    </Field>
  );

  // n_layers drives the channel vector's length (D-C3): grow -> pad with 16
  // (the default channel, D-C2), shrink -> truncate, so it always fits.
  const setLayers = (L: number) => {
    L = Math.max(1, L | 0);
    const ch = (form.channels ?? []).slice(0, L);
    while (ch.length < L) ch.push(16);
    setForm({ ...form, n_layers: L, channels: ch });
  };
  const setChannels = (text: string) => {
    const ch = text.split(",").map((s) => parseInt(s.trim(), 10)).filter((v) => !isNaN(v));
    setForm({ ...form, channels: ch, n_layers: ch.length || form.n_layers });
  };

  return (
    <div>
      <h2 data-domain="C">Redes foveadas (C)</h2>
      <p className="sub">La geometría se declara en px reales: la fóvea es la ventana etiquetada y el borde es independiente de cómo se reduce. N se deriva; los rangos de búsqueda se calculan, nunca se escriben.</p>
      <ErrorBox error={error} />
      <Working on={!ready} />
      {!ready ? null : (
      <div className="row">
        <div className="card" style={{ width: 320 }}>
          <Field label="nombre"><input value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <div className="row">
            <div className="grow">{num("fovea_px", 2, "fóvea en px = ventana etiquetada de B")}</div>
            <div className="grow">{num("border_px", 1, "borde difuso en px reales, por lado")}</div>
          </div>
          <div className="row">
            <div className="grow">{num("border_reduce", 1, "px reales por celda de borde")}</div>
            <div className="grow">{num("overlap_fovea_px", 1, "px de fóvea que ve también la rama del borde")}</div>
          </div>
          <div className="row">
            <div className="grow">{num("overlap_border_px", 1, "px de borde que ve también la rama de la fóvea")}</div>
            <div className="grow" />
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
            <div className="grow">
              <Field label="n_layers" help="capas conv por rama (D-S2: simétrico)">
                <input type="number" step={1} min={1} value={form.n_layers}
                  onChange={(e) => setLayers(+e.target.value)} />
              </Field>
            </div>
            <div className="grow">
              <Field label="channels (por capa)" help="lista de longitud n_layers (D-C3)">
                <input value={(form.channels ?? []).join(", ")} placeholder="16, 32"
                  onChange={(e) => setChannels(e.target.value)} />
              </Field>
            </div>
          </div>
          <Field label="merge" help="concat tolera strides distintos; sum exige iguales">
            <select value={form.merge} onChange={(e) => setForm({ ...form, merge: e.target.value })}>
              <option>concat</option><option>sum</option>
            </select></Field>
          <Field label="pool_mode" help="cómo se reduce la periferia (eje a barrer)">
            <select value={form.pool_mode} onChange={(e) => setForm({ ...form, pool_mode: e.target.value })}>
              <option>avg</option><option>max</option>
            </select></Field>
          {exists ? (
            <button onClick={() => setConfirming(true)} data-testid="update-btn"
              disabled={!form.name || !validation?.valid}>Actualizar «{form.name}»</button>
          ) : (
            <button onClick={() => save()}
              disabled={!form.name || !validation?.valid}>Guardar</button>
          )}
          {confirming ? (
            <div className="card" style={{ marginTop: 8 }} data-testid="overwrite-confirm">
              <strong>Se reemplaza la definición de «{form.name}»</strong>
              <p className="sub" style={{ marginTop: 4 }}>
                Los runs y recorridos ya hechos <b>no cambian</b>: copiaron los valores de C al
                crearse (`base_network_value`). Lo que cambia es lo que se entrene a partir de ahora
                con este nombre.
              </p>
              <button onClick={() => save(true)}>Sí, reemplazar</button>{" "}
              <button className="secondary" onClick={() => setConfirming(false)}>Cancelar</button>
            </div>
          ) : null}
        </div>
        <div className="card grow" data-testid="validate-panel">
          <h3 style={{ marginTop: 0 }}>Lo que implica (en vivo)</h3>
          {!validation ? <Working on /> : validation.valid ? (
            <div className="row">
              <div>
                <dl className="kv">
                  <dt>fóvea (ventana etiquetada)</dt><dd>{validation.trace.dims.fovea_px}px</dd>
                  <dt>borde difuso</dt><dd>{validation.trace.dims.border_px}px reales en {validation.trace.dims.border_cells} celdas ({validation.trace.dims.border_reduce}px/celda)</dd>
                  <dt>solape sobre la fóvea</dt><dd>{validation.trace.dims.overlap_fovea_px}px</dd>
                  <dt>solape sobre el borde</dt><dd>{validation.trace.dims.overlap_border_px}px</dd>
                  <dt>entrada compuesta (N, derivada)</dt><dd>{validation.trace.dims.N}×{validation.trace.dims.N}</dd>
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
      )}
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Guardadas</h3>
        <table className="data" data-testid="networks-table">
          <thead><tr><th>nombre</th><th>fóvea</th><th>borde</th><th>px/celda</th><th>kernels</th>
            <th>strides</th><th>merge</th><th></th></tr></thead>
          <tbody>
            {list.map((n) => (
              <tr key={n.name} onClick={() => setForm({ ...defaults, name: "", ...n })}>
                <td>{n.name}</td><td>{n.fovea_px}px</td><td>{n.border_px}px</td><td>{n.border_reduce}</td>
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
