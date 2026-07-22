import React from "react";

// Small-multiples friendly SVG line chart: ONE metric per panel, never a
// double axis (loss, f1 and px are three scales — ui.md R4).

export function LineChart(props: {
  series: { label: string; points: [number, number][]; color?: string }[];
  title: string; width?: number; height?: number;
}) {
  const W = props.width ?? 340, H = props.height ?? 120;
  const pad = 30;
  const all = props.series.flatMap((s) => s.points);
  if (!all.length) return <div className="working">{props.title}: sin datos</div>;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - pad - 8);
  const sy = (y: number) => H - 18 - ((y - y0) / (y1 - y0 || 1)) * (H - 30);
  return (
    <svg width={W} height={H} role="img" aria-label={props.title}>
      <text x={pad} y={12} fontSize={11} fill="var(--text-dim)">{props.title}</text>
      <text x={4} y={sy(y1) + 4} fontSize={9} fill="var(--text-dim)">{y1.toPrecision(3)}</text>
      <text x={4} y={sy(y0) + 4} fontSize={9} fill="var(--text-dim)">{y0.toPrecision(3)}</text>
      {props.series.map((s, i) => (
        <g key={i}>
          <polyline fill="none" stroke={s.color ?? "var(--accent)"} strokeWidth={1.5}
            points={s.points.map((p) => `${sx(p[0])},${sy(p[1])}`).join(" ")} />
          <text x={W - 8} y={14 + i * 12} fontSize={10} textAnchor="end"
            fill={s.color ?? "var(--accent)"}>{s.label}</text>
        </g>
      ))}
    </svg>
  );
}
