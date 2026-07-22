import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Badge, ErrorBox, Working } from "../components/ui";

// E — the LIST answers "what runs are there": one row per run, provenance by
// name, best monitor, s/epoch. The detail lives at /runs/:name.
export default function Runs() {
  const [runs, setRuns] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);

  const refresh = () => api.get("/runs").then((d) => setRuns(d.runs)).catch(setError);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      <h2>Runs (E)</h2>
      <p className="sub">Modelos entrenados con su procedencia por nombre. Un run del recorrido enlaza a su padre.</p>
      <ErrorBox error={error} />
      <div className="card">
        <Working on={!runs} />
        {runs ? (
          <table className="data" data-testid="runs-table">
            <thead><tr><th>run</th><th>estado</th><th>B</th><th>C</th><th>D</th>
              <th>recorrido</th><th>monitor</th><th>best</th><th>s/época</th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.name}>
                  <td><Link to={`/runs/${r.name}`}>{r.name}</Link></td>
                  <td><Badge status={r.status} /></td>
                  <td>{r.window_dataset}</td><td>{r.network}</td><td>{r.recipe}</td>
                  <td>{r.sweep ?? "—"}</td>
                  <td>{r.monitor}</td>
                  <td>{r.best?.toFixed ? r.best.toFixed(4) : r.best ?? "—"}</td>
                  <td>{r.seconds_per_epoch ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        {runs && !runs.length ? <p>No hay runs. Lanza uno desde Entrenar.</p> : null}
      </div>
    </div>
  );
}
