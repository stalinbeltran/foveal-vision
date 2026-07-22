import React, { useEffect, useRef } from "react";
import { CORNERS, CORNER_CSS } from "../api";

// A raw labelled window (uint8 pixels) with true corners as rings and, when
// given, predictions as dots — the error is the line between them.

export function WindowCanvas(props: {
  pixels: number[][];
  y?: number[][];              // (4,3) truth [exists, x, y] normalised
  pred?: number[][];           // (4,2) predicted xy normalised
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
    const colorOf = (i: number) =>
      getComputedStyle(document.documentElement)
        .getPropertyValue(CORNER_CSS[CORNERS[i]].slice(4, -1)).trim() || "#f00";
    for (let c = 0; c < 4; c++) {
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
