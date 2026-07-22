import React, { useEffect, useState } from "react";
import { api, CORNERS, CORNER_CSS } from "../api";
import { ErrorBox, Field, Working } from "../components/ui";
import { WindowCanvas } from "../components/WindowCanvas";
import { MatrixCanvas } from "../components/MatrixCanvas";

// E x B: one pass over the split (a cache), many views. Moving the threshold
// re-reads stored scores — it never re-runs the model. The gallery goes
// worst-first; a click opens the probes (input view F0, kernels, feature maps).
export default function Diagnostics() {
  const [runs, setRuns] = useState<any[]>([]);
  const [run, setRun] = useState("");
  const [split, setSplit] = useState("val");
  const [threshold, setThreshold] = useState(0.5);
  const [summary, setSummary] = useState<any>(null);
  const [evidence, setEvidence] = useState<any>(null);
  const [gallery, setGallery] = useState<any>(null);
  const [windowsPix, setWindowsPix] = useState<Record<number, any>>({});
  const [error, setError] = useState<unknown>(null);
  const [probe, setProbe] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.get("/runs").then((d) => {
      const done = d.runs.filter((r: any) => ["done", "cancelled"].includes(r.status));
      setRuns(done);
      if (done[0]) setRun(done[0].name);
    }).catch(setError);
  }, []);

  useEffect(() => {
    if (!run) return;
    setError(null); setSummary(null); setGallery(null); setProbe(null); setBusy(true);
    Promise.all([
      api.get(`/runs/${run}/diagnostics/summary?split=${split}&threshold=${threshold}`),
      api.get(`/runs/${run}/diagnostics/evidence?split=${split}`),
      api.get(`/runs/${run}/diagnostics/windows?split=${split}&threshold=${threshold}&limit=12`),
    ]).then(async ([s, e, g]) => {
      setSummary(s); setEvidence(e); setGallery(g);
      const r = runs.find((x) => x.name === run);
      if (r?.window_dataset) {
        const pix: Record<number, any> = {};
        await Promise.all(g.items.map(async (it: any) => {
          pix[it.window_idx] = await api.get(
            `/window-datasets/${r.window_dataset}/windows/${it.window_idx}`);
        }));
        setWindowsPix(pix);
      }
    }).catch(setError).finally(() => setBusy(false));
  }, [run, split, threshold]);

  const openProbes = async (it: any) => {
    const r = runs.find((x) => x.name === run);
    if (!r) return;
    setProbe({ loading: true, item: it });
    try {
      const [iv, fm, k] = await Promise.all([
        api.post(`/runs/${run}/input-view`,
          { window_dataset: r.window_dataset, index: it.window_idx }),
        api.post(`/runs/${run}/feature-maps`,
          { window_dataset: r.window_dataset, index: it.window_idx }),
        api.get(`/runs/${run}/kernels`),
      ]);
      setProbe({ item: it, inputView: iv, featureMaps: fm, kernels: k });
    } catch (e) { setError(e); setProbe(null); }
  };

  return (
    <div>
      <h2>Diagnóstico (E×B)</h2>
      <p className="sub">Una pasada sobre el split, muchas vistas. Mover el umbral relee scores
        guardados: no vuelve a correr el modelo.</p>
      <ErrorBox error={error} />
      <div className="card row" style={{ alignItems: "flex-end" }}>
        <div style={{ width: 220 }}><Field label="run">
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => <option key={r.name}>{r.name}</option>)}
          </select></Field></div>
        <div style={{ width: 120 }}><Field label="split">
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option>train</option><option>val</option><option>test</option>
          </select></Field></div>
        <div style={{ width: 240 }}><Field label={`threshold: ${threshold.toFixed(2)} (gratis)`}>
          <input type="range" min={0.05} max={0.95} step={0.05} value={threshold}
            onChange={(e) => setThreshold(+e.target.value)} /></Field></div>
        <Working on={busy} label="calculando la tabla (primera vez) o leyendo el caché…" />
      </div>
      {summary ? (
        <div className="row">
          <div className="card grow" data-testid="diag-summary">
            <h3 style={{ marginTop: 0 }}>Resumen — fija (run, split), varía el umbral, mide detección y posición</h3>
            <dl className="kv">
              <dt>ventanas / positivos</dt><dd>{summary.windows} / {summary.positives}</dd>
              <dt>precision · recall · f1</dt>
              <dd>{summary.detection.precision.toFixed(3)} · {summary.detection.recall.toFixed(3)} ·{" "}
                <b>{summary.detection.f1.toFixed(3)}</b></dd>
              <dt>pos_err_px (ventana)</dt><dd>{summary.pos_err_px?.toFixed(2) ?? "—"}</dd>
              <dt>· banda ciega (ev &lt; 0.05)</dt><dd>{summary.pos_err_px_blind?.toFixed(2) ?? "—"}</dd>
              <dt>· banda visible</dt><dd>{summary.pos_err_px_visible?.toFixed(2) ?? "—"}</dd>
              <dt>fracción ciega</dt><dd>{(summary.blind_share * 100).toFixed(1)}%</dd>
            </dl>
            <p className="sub">Detección y posición van POR SEPARADO: un global promedia lo posible
              con lo imposible. La banda ciega es la población que la periferia existe para arreglar.</p>
          </div>
          {evidence ? (
            <div className="card grow">
              <h3 style={{ marginTop: 0 }}>Evidencia (V18) — cuánto del párrafo cabe en la fóvea</h3>
              <table className="data">
                <thead><tr><th>banda</th><th>esquinas</th><th>score medio</th><th>err px</th></tr></thead>
                <tbody>
                  {evidence.bands.map((b: any) => (
                    <tr key={b.band}><td className="mono">{b.band}</td><td>{b.count}</td>
                      <td>{b.mean_score?.toFixed(3) ?? "—"}</td>
                      <td>{b.mean_err_px?.toFixed(2) ?? "—"}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
      {gallery ? (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Galería peor-primero ({gallery.total} ventanas) — clic: sondas</h3>
          <div className="thumbgrid" data-testid="gallery">
            {gallery.items.map((it: any) => {
              const pix = windowsPix[it.window_idx];
              return (
                <div className="thumb" key={it.row} onClick={() => openProbes(it)}
                  style={{ cursor: "pointer" }}>
                  {pix ? <WindowCanvas pixels={pix.pixels} y={it.y_true}
                    pred={it.xy_pred} scale={6} /> : <Working on />}
                  <div className="cap">#{it.window_idx} · err {it.err_px.filter((e: any) => e != null)
                    .map((e: number) => e.toFixed(1)).join("/") || "—"}</div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
      {probe && !probe.loading ? (
        <div className="card" data-testid="probes">
          <h3 style={{ marginTop: 0 }}>Sondas de la ventana #{probe.item.window_idx}</h3>
          <div className="row" style={{ marginBottom: 6 }}>
            {CORNERS.map((c, i) => {
              const s = probe.item.scores[i];
              return (
                <div key={c} className="meter" style={{ width: 200 }}>
                  <span className={`corner-${c}`} style={{ width: 26 }}>{c}</span>
                  <div className="track">
                    <div className="fill" style={{ width: `${s * 100}%`,
                      background: CORNER_CSS[c] }} />
                    <div className="thr" style={{ left: `${threshold * 100}%` }} />
                  </div>
                  <span className="mono">{s.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
          <h4>F0 — la entrada canal a canal (cobertura mín {probe.inputView.coverage_min.toFixed(2)})</h4>
          <div className="row">
            {probe.inputView.channels.map((ch: any, i: number) => (
              <MatrixCanvas key={i} payload={ch} scale={7} />
            ))}
          </div>
          <h4>V1 — kernels de capa 1, por rama (divergente ±0: el signo es lo que un kernel es)</h4>
          <div className="row">
            {(["center", "periph"] as const).map((b) => (
              <div key={b}>
                <div className="cap">{b}</div>
                <div className="row">
                  {probe.kernels.branches[b].maps.slice(0, 8).map((m: any, i: number) => (
                    <MatrixCanvas key={i} payload={m} scale={10} />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <h4>V2 — feature maps (capa 1, primeros 6 por rama)</h4>
          <div className="row">
            {(["center", "periph"] as const).map((b) => (
              <div key={b}>
                <div className="cap">{b}</div>
                <div className="row">
                  {probe.featureMaps.branches[b][0].maps.slice(0, 6).map((m: any, i: number) => (
                    <MatrixCanvas key={i} payload={m} scale={5} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : probe?.loading ? <Working on label="cargando sondas…" /> : null}
    </div>
  );
}
