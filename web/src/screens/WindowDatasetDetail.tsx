import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Working } from "../components/ui";
import { WindowCanvas } from "../components/WindowCanvas";

// Look at the RAW data — pixels and label, no model needed: inspecting the
// dataset must never require a trained run.
export default function WindowDatasetDetail() {
  const { name } = useParams();
  const nav = useNavigate();
  const [manifest, setManifest] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [split, setSplit] = useState("train");
  const [positivesOnly, setPositivesOnly] = useState(true);
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<any>(null);
  const [windows, setWindows] = useState<any[]>([]);

  useEffect(() => {
    api.get(`/window-datasets/${name}`).then(setManifest).catch(setError);
  }, [name]);

  useEffect(() => {
    setWindows([]);
    api.get(`/window-datasets/${name}/windows?split=${split}` +
            `&positives_only=${positivesOnly}&offset=${offset}&limit=24`)
      .then(async (p) => {
        setPage(p);
        const ws = await Promise.all(p.indexes.map((i: number) =>
          api.get(`/window-datasets/${name}/windows/${i}`)));
        setWindows(ws);
      }).catch(setError);
  }, [name, split, positivesOnly, offset]);

  const del = async () => {
    setError(null);
    try { await api.del(`/window-datasets/${name}`); nav("/window-datasets"); }
    catch (e) { setError(e); }
  };

  return (
    <div>
      <h2>{name}</h2>
      <p className="sub">El dato crudo que se etiquetó — anillo = esquina verdadera.</p>
      <ErrorBox error={error} />
      {manifest ? (
        <div className="card">
          <dl className="kv">
            <dt>fuente</dt><dd>{manifest.source_id}</dd>
            <dt>ventana</dt><dd>{manifest.config.window_size}px, stride {manifest.config.stride}</dd>
            <dt>ventanas</dt><dd>{manifest.num_windows} de {manifest.num_samples} imágenes</dd>
            <dt>positivos</dt><dd>{JSON.stringify(manifest.positives_per_corner)}</dd>
            <dt>huella</dt><dd className="mono">{manifest.fingerprint}</dd>
            <dt>usado por</dt><dd>{manifest.used_by?.join(", ") || "—"}</dd>
          </dl>
          <button className="secondary" onClick={del} style={{ marginTop: 8 }}>
            Borrar (409 si algún run lo usa)</button>
        </div>
      ) : <Working on />}
      <div className="card">
        <div className="row" style={{ alignItems: "center" }}>
          <select value={split} onChange={(e) => { setSplit(e.target.value); setOffset(0); }}
            style={{ width: 120 }}>
            <option>train</option><option>val</option><option>test</option>
          </select>
          <label><input type="checkbox" checked={positivesOnly}
            onChange={(e) => { setPositivesOnly(e.target.checked); setOffset(0); }} />
            {" "}solo con esquina</label>
          <button className="secondary" disabled={offset <= 0}
            onClick={() => setOffset(Math.max(0, offset - 24))}>←</button>
          <span>{offset}–{offset + 24} de {page?.total ?? "…"}</span>
          <button className="secondary" disabled={!page || offset + 24 >= page.total}
            onClick={() => setOffset(offset + 24)}>→</button>
        </div>
        <div className="thumbgrid" style={{ marginTop: 10 }} data-testid="window-grid">
          {windows.map((w) => (
            <div className="thumb" key={w.index}>
              <WindowCanvas pixels={w.pixels} y={w.y} scale={6} />
              <div className="cap">#{w.index} · img {w.sample_idx} · ({w.window_xy.join(",")})</div>
            </div>
          ))}
        </div>
        <Working on={!windows.length && !!page?.total} label="cargando ventanas…" />
      </div>
    </div>
  );
}
