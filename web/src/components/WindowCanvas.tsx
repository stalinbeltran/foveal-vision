import React, { useEffect, useRef } from "react";
import { CORNER_CSS, Corner } from "../api";

// A raw labelled window (uint8 pixels) with true corners as rings and, when
// given, predictions as dots — the error is the line between them.
//
// `cornerOrder` says what row i of `y` MEANS; it comes with the payload that
// carries `y` (the dataset's manifest order). Hardcoding it here would silently
// mis-colour every corner the day a dataset is built in another order.

export function WindowCanvas(props: {
  pixels: number[][];
  y?: number[][];              // (N,3) truth [exists, x, y] normalised
  pred?: number[][];           // (N,2) predicted xy normalised
  cornerOrder?: Corner[];
  scale?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const n = props.pixels.length;
  const scale = props.scale ?? 8;

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !n) return;
    const s = scale;
    cv.width = n * s;
    cv.height = n * s;
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    for (let yy = 0; yy < n; yy++)
      for (let xx = 0; xx < n; xx++) {
        const v = props.pixels[yy][xx];
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(xx * s, yy * s, s, s);
      }
    const order = props.cornerOrder ?? [];
    const colorOf = (i: number) => {
      const css = CORNER_CSS[order[i]];
      if (!css) return "#f00";
      return getComputedStyle(document.documentElement)
        .getPropertyValue(css.slice(4, -1)).trim() || "#f00";
    };
    for (let c = 0; c < order.length; c++) {
      const t = props.y?.[c];
      const p = props.pred?.[c];
      const col = colorOf(c);
      if (t && t[0] >= 0.5) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(t[1] * n * s, t[2] * n * s, s * 0.9, 0, Math.PI * 2);
        ctx.stroke();
        if (p) {
          ctx.beginPath();
          ctx.moveTo(t[1] * n * s, t[2] * n * s);
          ctx.lineTo(p[0] * n * s, p[1] * n * s);
          ctx.stroke();
        }
      }
      if (p && t && t[0] >= 0.5) {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(p[0] * n * s, p[1] * n * s, s * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [props.pixels, props.y, props.pred, scale]);

  return <canvas ref={ref} style={{ imageRendering: "pixelated" }} />;
}
