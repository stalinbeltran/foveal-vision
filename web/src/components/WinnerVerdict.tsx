import React from "react";

// The verdict of a sweep, as the protocol states it (protocolo.md §1.5): a point
// is a DISTRIBUTION over its seeds, and a difference that does not clear that
// band is a TIE. The old view showed a single number per point and crowned the
// top row — which reads as a result even when the gap is a fifth of the noise.
//
// So: mean ± the observed min–max band, n seeds, δ (and where δ came from), and
// a verdict line that says out loud whether the sweep distinguishes anything.
// The suggestion stays a suggestion; the user confirms.

const f4 = (v: any) => (typeof v === "number" ? v.toFixed(4) : v ?? "—");

export function WinnerVerdict(props: { winner: any; compact?: boolean }) {
  const w = props.winner;
  if (!w?.best) return null;
  const tie = !!w.tie;
  const frontier = new Set((w.frontier ?? []).map((t: any) => JSON.stringify(t.point)));
  const band = (t: any) =>
    t.n_seeds > 1 ? `${f4(t.value_min)} – ${f4(t.value_max)}` : "—";

  return (
    <div data-testid="winner-verdict">
      <div style={{
        padding: "8px 12px", borderRadius: 8, marginBottom: 8,
        background: "var(--surface-2)",
        border: `1px solid ${tie ? "var(--warn)" : "var(--border)"}`,
      }}>
        <div><strong>{tie ? "Empate técnico" : "Ganador distinguible"}</strong>{" "}
          <span className="sub" style={{ margin: 0 }}>
            δ = {f4(w.delta)} · {w.delta_source}
          </span></div>
        <div className="sub" style={{ margin: "4px 0 0" }}>{w.tie_reason}</div>
        {w.monitor_matches_objective === false ? (
          <div className="sub" style={{ margin: "4px 0 0" }}>
            ⚠ el valor es el del checkpoint guardado (elegido por el monitor), medido
            con <span className="mono">{w.objective}</span>: monitor y objetivo no coinciden.
          </div>
        ) : null}
      </div>
      <div className="sub">mejor media:{" "}
        <span className="mono">{JSON.stringify(w.best.point)}</span>{" "}
        {f4(w.best.value)} (banda {band(w.best)}, n={w.best.n_seeds})</div>
      <div className="sub">sugerido (el más barato dentro de δ):{" "}
        <span className="mono">{JSON.stringify(w.suggested.point)}</span>{" "}
        {f4(w.suggested.value)}</div>
      {props.compact ? null : (
        <table className="data" data-testid="winner-table" style={{ marginTop: 8 }}>
          <thead><tr><th>punto</th><th>media</th><th>banda (min–max)</th><th>n</th>
            <th>{w.cost_metric === "num_params" ? "params" : "s/época"}</th>
            <th>frontera</th></tr></thead>
          <tbody>
            {(w.trials ?? []).map((t: any, i: number) => {
              const inFrontier = frontier.has(JSON.stringify(t.point));
              return (
                <tr key={i}>
                  <td className="mono">{JSON.stringify(t.point)}</td>
                  <td>{f4(t.value)}</td>
                  <td className="sub" style={{ margin: 0 }}>{band(t)}</td>
                  <td>{t.n_seeds}</td>
                  <td>{w.cost_metric === "num_params" ? t.num_params ?? "—"
                    : t.seconds_per_epoch != null ? Number(t.seconds_per_epoch).toFixed(1) : "—"}</td>
                  <td>{inFrontier ? (tie ? "dentro de δ (empatado)" : "mejor") : ""}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
