import React, { useEffect, useRef, useState } from "react";

// Paints a matrixview payload. The colour work is DECLARED by the payload
// (sequential | diverging): the painter never guesses whether it looks at a
// signed weight or a magnitude. Normalisation is per map. Click toggles the
// number table (the accessible twin of every heatmap).

type MapPayload = {
  label?: string | null;
  matrix: number[][];
  min: number; max: number; mean: number;
  color: "sequential" | "diverging";
};

function css(varName: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function hex2rgb(h: string): [number, number, number] {
  const v = h.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16),
          parseInt(v.slice(4, 6), 16)];
}

function mix(a: [number, number, number], b: [number, number, number], t: number) {
  return [Math.round(a[0] + (b[0] - a[0]) * t),
          Math.round(a[1] + (b[1] - a[1]) * t),
          Math.round(a[2] + (b[2] - a[2]) * t)] as [number, number, number];
}

export function MatrixCanvas({ payload, scale = 8 }: { payload: MapPayload; scale?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [showTable, setShowTable] = useState(false);
  const m = payload.matrix;
  const rows = m.length, cols = m[0]?.length ?? 0;

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !rows) return;
    cv.width = cols;
    cv.height = rows;
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(cols, rows);
    const neg = hex2rgb(css("--div-neg") || "#2166ac");
    const pos = hex2rgb(css("--div-pos") || "#b2182b");
    const neutral: [number, number, number] = [245, 244, 240];
    const dark: [number, number, number] = [22, 30, 40];
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const v = m[y][x];
        let rgb: [number, number, number];
        if (payload.color === "diverging") {
          const a = Math.max(Math.abs(payload.min), Math.abs(payload.max)) || 1;
          const t = Math.max(-1, Math.min(1, v / a));
          rgb = t >= 0 ? mix(neutral, pos, t) : mix(neutral, neg, -t);
        } else {
          const span = payload.max - payload.min || 1;
          const t = (v - payload.min) / span;
          rgb = mix(neutral, dark, t);
        }
        const i = (y * cols + x) * 4;
        img.data[i] = rgb[0]; img.data[i + 1] = rgb[1];
        img.data[i + 2] = rgb[2]; img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [payload]);

  return (
    <div className="thumb" style={{ width: cols * scale }}>
      <canvas ref={ref} style={{ width: cols * scale, height: rows * scale }}
        title={`min ${payload.min.toFixed(3)} · max ${payload.max.toFixed(3)} — clic: tabla`}
        onClick={() => setShowTable((s) => !s)} />
      {payload.label ? <div className="cap">{payload.label}</div> : null}
      {showTable ? (
        <div style={{ overflowX: "auto", maxWidth: 420 }}>
          <table className="data mono" style={{ fontSize: 10 }}>
            <tbody>
              {m.map((row, i) => (
                <tr key={i}>{row.map((v, j) => <td key={j}>{v.toFixed(2)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
