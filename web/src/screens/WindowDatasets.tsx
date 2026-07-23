import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, waitJob } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field, Working } from "../components/ui";

// B — where the labelled window (the fovea) is decided; contract (1) is born here.
export default function WindowDatasets() {
  const [data, setData] = useState<any>(null);
  const [sources, setSources] = useState<any[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [form, setForm] = usePersistedState("wds.form", { name: "", source: "",
    window_size: 16, stride: 8, val_frac: 0.15, test_frac: 0.15, seed: 1 });

  const refresh = () => api.get("/window-datasets").then(setData).catch(setError);
  useEffect(() => {
    refresh();
    api.get("/sources").then((d) => {
      setSources(d.sources);
      if (d.sources[0]) setForm((f) => f.source ? f : { ...f, source: d.sources[0].id });
    }).catch(setError);
  }, []);

  const build = async () => {
    setError(null); setBusy(true);
    try {
      const r = await api.post("/window-datasets", form);
      const j = await waitJob(r.job.id);
      if (j.status === "error") setError(j.error);
      await refresh();
    } catch (e) { setError(e); }
    setBusy(false);
  };

  return (
    <div>
      <h2>Datasets de ventanas (B)</h2>
      <p className="sub">La ventana etiquetada es la fóvea (F1b). La vista foveada NO se hornea aquí:
        se construye en el dataloader, así que toda la geometría es barrible sin re-extraer.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card grow">
          <Working on={!data} />
          {data ? (
            <table className="data" data-testid="wds-table">
              <thead><tr><th>nombre</th><th>ventana</th><th>ventanas</th>
                <th>splits (t/v/t)</th><th>positivos TL</th><th>huella</th></tr></thead>
              <tbody>
                {data.window_datasets.map((m: any) => (
                  <tr key={m.name}>
                    <td><Link to={`/window-datasets/${m.name}`}>{m.name}</Link></td>
                    <td>{m.config.window_size}px · stride {m.config.stride}</td>
                    <td>{m.num_windows}</td>
                    <td>{m.windows_per_split.train}/{m.windows_per_split.val}/{m.windows_per_split.test}</td>
                    <td>{m.positives_per_corner?.TL}</td>
                    <td className="mono">{m.fingerprint?.slice(7, 17)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
        <div className="card" style={{ width: 300 }}>
          <h3 style={{ marginTop: 0 }}>Construir</h3>
          <Field label="nombre"><input value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
          <Field label="fuente">
            <select value={form.source}
              onChange={(e) => setForm({ ...form, source: e.target.value })}>
              {sources.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
            </select></Field>
          <Field label="window_size" help="lado de la ventana etiquetada = la fóvea de la red">
            <input type="number" value={form.window_size}
              onChange={(e) => setForm({ ...form, window_size: +e.target.value })} /></Field>
          <Field label="stride (de extracción)"><input type="number" value={form.stride}
            onChange={(e) => setForm({ ...form, stride: +e.target.value })} /></Field>
          <Field label="val_frac" help="sin val, un dataset no sirve para medir">
            <input type="number" step="0.05" value={form.val_frac}
              onChange={(e) => setForm({ ...form, val_frac: +e.target.value })} /></Field>
          <Field label="test_frac"><input type="number" step="0.05" value={form.test_frac}
            onChange={(e) => setForm({ ...form, test_frac: +e.target.value })} /></Field>
          <Field label="seed (del split, por imagen)"><input type="number" value={form.seed}
            onChange={(e) => setForm({ ...form, seed: +e.target.value })} /></Field>
          <button onClick={build} disabled={busy || !form.name}>Extraer</button>
          <Working on={busy} label="extrayendo…" />
        </div>
      </div>
    </div>
  );
}
