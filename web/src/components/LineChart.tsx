import React from "react";

// Small-multiples friendly SVG line chart: ONE metric per panel, never a
// double axis (loss, f1 and px are three scales — ui.md R4).
//
// A series may carry a `band` ([x, lo, hi] per point): a filled area drawn
// behind the mean line, for the seed-aggregated (mean ± spread) view in
// Recorridos. `faded` dims a series (emphasis: hovering one legend entry
// recesses the rest). `showLegend` keeps the in-SVG labels for RunDetail; the
// multi-run overlay turns them off and drives identity from an external HTML
// legend with checkboxes.

export type Series = {
  label: string;
  points: [number, number][];
  band?: [number, number, number][];
  color?: string;
  faded?: boolean;
};

export function LineChart(props: {
  series: Series[];
  title: string; width?: number; height?: number; showLegend?: boolean;
}) {
  const W = props.width ?? 340, H = props.height ?? 120;
  const showLegend = props.showLegend ?? true;
  const pad = 30;
  // the y/x extent must cover the band edges too, not just the mean points
  const extent = props.series.flatMap((s) => [
    ...s.points,
    ...(s.band ? s.band.flatMap(([x, lo, hi]) => [[x, lo], [x, hi]] as [number, number][]) : []),
  ]);
  if (!extent.length) return <div className="working">{props.title}: sin datos</div>;
  const xs = extent.map((p) => p[0]), ys = extent.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - pad - 8);
  const sy = (y: number) => H - 18 - ((y - y0) / (y1 - y0 || 1)) * (H - 30);
  return (
    <svg width={W} height={H} role="img" aria-label={props.title}>
      <text x={pad} y={12} fontSize={11} fill="var(--text-dim)">{props.title}</text>
      <text x={4} y={sy(y1) + 4} fontSize={9} fill="var(--text-dim)">{y1.toPrecision(3)}</text>
      <text x={4} y={sy(y0) + 4} fontSize={9} fill="var(--text-dim)">{y0.toPrecision(3)}</text>
      {props.series.map((s, i) => {
        const color = s.color ?? "var(--accent)";
        const op = s.faded ? 0.18 : 1;
        // band polygon: upper edge left→right, then lower edge right→left
        const bandPath = s.band && s.band.length
          ? s.band.map(([x, , hi]) => `${sx(x)},${sy(hi)}`).join(" ") + " " +
            [...s.band].reverse().map(([x, lo]) => `${sx(x)},${sy(lo)}`).join(" ")
          : null;
        return (
          <g key={i}>
            {bandPath ? (
              <polygon points={bandPath} fill={color} opacity={s.faded ? 0.06 : 0.15} />
            ) : null}
            <polyline fill="none" stroke={color} strokeWidth={s.faded ? 1 : 1.5}
              opacity={op}
              points={s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />
            {showLegend ? (
              <text x={W - 8} y={14 + i * 12} fontSize={10} textAnchor="end"
                fill={color} opacity={op}>{s.label}</text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}
