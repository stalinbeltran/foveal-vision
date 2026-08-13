import React, { useMemo, useState } from "react";
import { LineChart, Series } from "./LineChart";
import { usePersistedState } from "../uiState";

// H — the whole sweep as ONE picture: every run's val curves overlaid on the
// three small multiples (loss / f1 / px), so you watch a space train and spot
// the diverging point at a glance. Two modes, per the user's call:
//   · "lines" — one coloured line per run; toggle each, hide/show all (identity
//     rides the legend + hover emphasis, never colour alone).
//   · "band"  — group runs that share a config EXCEPT the seed, and draw the
//     mean ± spread (min–max) per group. With one seed/config the band is
//     degenerate (a flat line) — honest, not hidden.
// Colour follows the entity (run's trial index / group order), never the rank,
// so hiding or sorting never repaints the survivors.

type Trial = { trial: number; run: string; point: Record<string, any>; status: string | null };
type Records = any[];

const PALETTE = 8;
const colorOf = (i: number) => `var(--series-${(i % PALETTE) + 1})`;

const METRICS = [
  { key: "loss", title: "val loss", get: (r: any) => r.val?.loss },
  { key: "f1", title: "val f1", get: (r: any) => r.val?.f1 },
  { key: "px", title: "val pos_err_px (px de la ventana)", get: (r: any) => r.val?.pos_err_px },
] as const;

const fmtPoint = (p: Record<string, any>) => {
  const keys = Object.keys(p || {});
  if (!keys.length) return "base";
  return keys.map((k) => `${k}=${JSON.stringify(p[k])}`).join(", ");
};
const stripSeed = (p: Record<string, any>) => {
  const { seed, ...rest } = p || {};
  return rest;
};
const ptsOf = (recs: Records, get: (r: any) => any): [number, number][] =>
  (recs || []).map((r) => [r.epoch, get(r)] as [number, number]).filter((p) => p[1] != null);

type Entity = {
  id: string; label: string; color: string; n: number;
  // one of the two shapes, by mode:
  records?: Records;                                    // lines
  metrics?: Record<string, { mean: [number, number][]; band: [number, number, number][] }>;  // band
  cutAt?: number;   // last epoch where ALL replicas of the group have data
};

export function SweepCurves(props: { trials: { trials: Trial[] }; curves: Record<string, Records> }) {
  const { trials, curves } = props;
  const [mode, setMode] = usePersistedState<"lines" | "band">("sweeps.curveMode", "lines");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [emph, setEmph] = useState<string | null>(null);

  // only runs that already produced at least one metric line can be drawn
  const rows = useMemo(
    () => trials.trials.filter((t) => (curves[t.run]?.length ?? 0) > 0),
    [trials, curves]);

  const lineEntities = useMemo<Entity[]>(() =>
    rows.map((t) => ({
      id: t.run, label: fmtPoint(t.point), color: colorOf(t.trial), n: 1,
      records: curves[t.run],
    })), [rows, curves]);

  const bandEntities = useMemo<Entity[]>(() => {
    const groups = new Map<string, { key: string; label: string; members: Trial[] }>();
    rows.forEach((t) => {
      const key = JSON.stringify(stripSeed(t.point));
      if (!groups.has(key)) groups.set(key, { key, label: fmtPoint(stripSeed(t.point)), members: [] });
      groups.get(key)!.members.push(t);
    });
    const list = [...groups.values()].sort((a, b) => a.key.localeCompare(b.key));
    return list.map((g, gi) => {
      // The band is only honest where EVERY replica of the group has reached the
      // epoch. Averaging whatever happens to be present makes the band narrow by
      // itself at the right edge (fewer seeds = less spread) and the mean jump —
      // it reads as "the seeds converge at the end" when it only means "some runs
      // are not there yet". So cut at the last epoch all of them reached, and say
      // where the cut is instead of drawing a shrinking tail.
      const lastOf = (t: Trial) => {
        const recs = curves[t.run] || [];
        return recs.length ? recs[recs.length - 1].epoch : 0;
      };
      const cutAt = Math.min(...g.members.map(lastOf));
      const maxSeen = Math.max(...g.members.map(lastOf));
      const metrics: Entity["metrics"] = {};
      METRICS.forEach((m) => {
        const byEpoch = new Map<number, number[]>();
        g.members.forEach((t) => (curves[t.run] || []).forEach((r) => {
          const v = m.get(r);
          if (v == null || r.epoch > cutAt) return;
          if (!byEpoch.has(r.epoch)) byEpoch.set(r.epoch, []);
          byEpoch.get(r.epoch)!.push(v);
        }));
        const epochs = [...byEpoch.keys()].sort((a, b) => a - b);
        const mean: [number, number][] = [];
        const band: [number, number, number][] = [];
        epochs.forEach((e) => {
          const vs = byEpoch.get(e)!;
          // a metric missing in SOME replica of an otherwise complete epoch
          // (pos_err_px can be null) is the same problem one level down
          if (vs.length < g.members.length) return;
          mean.push([e, vs.reduce((a, b) => a + b, 0) / vs.length]);
          band.push([e, Math.min(...vs), Math.max(...vs)]);
        });
        metrics![m.key] = { mean, band };
      });
      return { id: g.key, label: g.label, color: colorOf(gi), n: g.members.length,
               metrics, cutAt: maxSeen > cutAt ? cutAt : undefined };
    });
  }, [rows, curves]);

  const entities = mode === "lines" ? lineEntities : bandEntities;
  const visible = entities.filter((e) => !hidden.has(e.id));
  const cutGroups = visible.filter((e) => e.cutAt != null);

  const toggle = (id: string) => setHidden((h) => {
    const n = new Set(h);
    n.has(id) ? n.delete(id) : n.add(id);
    return n;
  });
  const hideAll = () => setHidden(new Set(entities.map((e) => e.id)));
  const showAll = () => setHidden(new Set());

  if (!rows.length)
    return (
      <div className="card" data-testid="sweep-curves">
        <h4 style={{ margin: "0 0 6px" }}>Curvas del recorrido</h4>
        <p className="working">Aún no hay curvas: ningún run del recorrido tiene métricas todavía.</p>
      </div>
    );

  // emphasis only bites when the hovered entity is actually on screen: hovering
  // a hidden legend item must not recess everything (there is nothing to lift)
  const emphActive = emph != null && !hidden.has(emph);
  // draw the emphasised entity last so it sits on top of the recessed ones
  const ordered = [...visible].sort((a, b) =>
    (a.id === emph ? 1 : 0) - (b.id === emph ? 1 : 0));

  const seriesFor = (metricKey: string, get: (r: any) => any): Series[] =>
    ordered.map((e) => ({
      label: e.label, color: e.color, faded: emphActive && emph !== e.id,
      points: mode === "lines" ? ptsOf(e.records!, get) : e.metrics![metricKey].mean,
      band: mode === "band" ? e.metrics![metricKey].band : undefined,
    }));

  return (
    <div className="card" data-testid="sweep-curves">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <h4 data-view="V14" data-fixes="B y el recorrido" data-varies="la epoca x el punto"
          data-measures="loss y metricas de val"
          style={{ margin: "0 0 6px" }}>Curvas del recorrido — val por época
          ({visible.length}/{entities.length} visibles)</h4>
        <div className="curvemode" role="group" aria-label="modo de trazado">
          <button className={mode === "lines" ? "" : "secondary"}
            onClick={() => setMode("lines")}>líneas por run</button>
          <button className={mode === "band" ? "" : "secondary"}
            onClick={() => setMode("band")}>media ± banda</button>
        </div>
      </div>
      <p className="sub" style={{ margin: "0 0 10px" }}>
        {mode === "lines"
          ? "Una línea por run; pasa el ratón por la leyenda para resaltar, o oculta lo que estorbe."
          : "Agrupado por config (misma red/receta, distinta semilla): línea media, sombra = min–max. " +
            "La banda se corta donde deja de haber TODAS las réplicas — si no, se estrecharía sola."}
        {mode === "band" && cutGroups.length ? (
          <span data-testid="band-cut">{" "}
            <strong>{cutGroups.length}</strong>{" "}
            {cutGroups.length === 1 ? "grupo cortado" : "grupos cortados"} por réplicas
            aún en marcha:{" "}
            <span className="mono">
              {cutGroups.map((e) => `${e.label} (hasta ep ${e.cutAt})`).join(" · ")}
            </span>.
          </span>
        ) : null}
      </p>
      <div className="row" style={{ gap: 18 }}>
        {METRICS.map((m) => (
          <LineChart key={m.key} title={m.title} showLegend={false}
            series={seriesFor(m.key, m.get)} />
        ))}
      </div>
      <div className="row" style={{ gap: 6, margin: "8px 0 4px" }}>
        <button className="secondary" onClick={showAll} disabled={!hidden.size}>mostrar todo</button>
        <button className="secondary" onClick={hideAll}
          disabled={hidden.size === entities.length}>ocultar todo</button>
      </div>
      <div className="curvelegend" data-testid="sweep-legend">
        {entities.map((e) => {
          const off = hidden.has(e.id);
          return (
            <label key={e.id} className={"legenditem" + (off ? " off" : "")}
              onMouseEnter={() => setEmph(e.id)} onMouseLeave={() => setEmph(null)}>
              <input type="checkbox" checked={!off} onChange={() => toggle(e.id)} />
              <span className="swatch" style={{ background: off ? "var(--border)" : e.color }} />
              <span className="mono">{e.label}</span>
              {mode === "band" && e.n > 1 ? <span className="sub"> ×{e.n}</span> : null}
            </label>
          );
        })}
      </div>
    </div>
  );
}
