import React, { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { Badge, ErrorBox } from "../components/ui";
import { LineChart } from "../components/LineChart";

// E detail: full provenance, execution (X) apart, live curves as SMALL
// MULTIPLES (loss / f1 / px are three scales: never one chart, never a double
// axis) with the epoch axis aligned. Metrics arrive incrementally (?since=).
export default function RunDetail() {
  const { name } = useParams();
  const nav = useNavigate();
  const [detail, setDetail] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [records, setRecords] = useState<any[]>([]);
  const next = useRef(0);

  useEffect(() => {
    setRecords([]); next.current = 0;
    const load = () => {
      api.get(`/runs/${name}`).then(setDetail).catch(setError);
      api.get(`/runs/${name}/metrics?since=${next.current}`).then((m) => {
        if (m.records.length) {
          // StrictMode double-fires the first load: dedupe by epoch
          setRecords((r) => {
            const seen = new Set(r.map((x) => x.epoch));
            const add = m.records.filter((x: any) => !seen.has(x.epoch));
            return add.length ? [...r, ...add] : r;
          });
          next.current = Math.max(next.current, m.next);
        }
      }).catch(() => {});
    };
    load();
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [name]);

  const stop = () => api.post(`/runs/${name}/stop`).catch(setError);
  const del = async () => {
    try { await api.del(`/runs/${name}`); nav("/runs"); } catch (e) { setError(e); }
  };

  const prov = detail?.config?.provenance;
  const pts = (f: (r: any) => number | null) =>
    records.map((r) => [r.epoch, f(r)] as [number, number]).filter((p) => p[1] != null);

  return (
    <div>
      <h2>{name} <Badge status={detail?.status?.status} /></h2>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card grow">
          <h3 style={{ marginTop: 0 }}>Procedencia</h3>
          {prov ? (
            <dl className="kv">
              <dt>dataset (B)</dt><dd>{prov.window_dataset.name}
                <span className="mono"> · {prov.window_dataset.fingerprint.slice(7, 17)}</span></dd>
              <dt>red (C)</dt><dd>{prov.network.name} (N={prov.network.value.N},
                c_frac={prov.network.value.c_frac}, d={prov.network.value.d},
                k={prov.network.value.k_center}/{prov.network.value.k_periph})</dd>
              <dt>receta (D)</dt><dd>{prov.recipe.name} (lr={prov.recipe.value.lr},
                λ_pos={prov.recipe.value.lambda_pos}, seed={prov.recipe.value.seed})</dd>
              <dt>recorrido</dt><dd>{prov.sweep ?? "—"}</dd>
              <dt>commit</dt><dd className="mono">{prov.git_commit?.slice(0, 12)}</dd>
              <dt>entorno (X)</dt><dd>{prov.environment.python} · torch {prov.environment.torch} ·
                {" "}{prov.environment.device}</dd>
            </dl>
          ) : null}
          {detail?.summary?.monitor ? (
            <p>monitor <b>{detail.summary.monitor}</b> → best{" "}
              <b>{detail.summary.best?.toFixed?.(4) ?? String(detail.summary.best)}</b>{" "}
              (época {detail.summary.best_epoch}) · {detail.summary.epochs_run}/
              {detail.summary.epochs_requested} épocas
              {detail.summary.cancelled ? " · parado a mano" : ""}</p>
          ) : null}
          <div className="row">
            <button className="secondary" onClick={stop}>Parar (fin de época)</button>
            <button className="secondary" onClick={del}>Borrar</button>
            <Link to="/diagnostics" style={{ alignSelf: "center" }}>→ Diagnóstico</Link>
          </div>
        </div>
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Curvas (V14 — small multiples, eje de épocas alineado)</h3>
          <LineChart title="loss" series={[
            { label: "train", points: pts((r) => r.train_loss), color: "var(--div-neg)" },
            { label: "val", points: pts((r) => r.val?.loss), color: "var(--div-pos)" }]} />
          <LineChart title="val f1" series={[
            { label: "val", points: pts((r) => r.val?.f1) }]} />
          <LineChart title="val pos_err_px (px de la ventana)" series={[
            { label: "val", points: pts((r) => r.val?.pos_err_px) }]} />
        </div>
      </div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Épocas</h3>
        <table className="data">
          <thead><tr><th>época</th><th>train loss</th><th>val loss</th><th>f1</th>
            <th>pos_err_px</th><th>lr</th><th>s</th></tr></thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.epoch}>
                <td>{r.epoch}</td><td>{r.train_loss.toFixed(4)}</td>
                <td>{r.val.loss?.toFixed(4)}</td><td>{r.val.f1?.toFixed(3)}</td>
                <td>{r.val.pos_err_px?.toFixed?.(2) ?? "—"}</td>
                <td>{r.lr}</td><td>{r.seconds}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
