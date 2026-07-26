import React, { useState } from "react";
import { api } from "../api";
import { ErrorBox, Working } from "./ui";

// The metric that MATTERS (protocolo.md §2): paragraph per IMAGE, scored against
// the source (A) — not the per-window proxy the ranking uses. ONE renderer, two
// screens (RunDetail and the sweep verdict): the second copy is where the two
// would start disagreeing about what a band means.
//
// It is fired by a BUTTON, never by the 2–3 s poll: it re-infers whole images
// (0.6 s for a 20-image val, longer for a real one). Cheap-looking UI that
// silently costs seconds per tick is how a screen becomes unusable.

const f4 = (v: any) => (typeof v === "number" ? v.toFixed(4) : "—");

function One({ p }: { p: any }) {
  const sem = p.macro?.sem;
  return (
    <div style={{ marginTop: 8 }} data-testid="task-row">
      <div>
        <span className="mono">{p.run}</span>{" "}
        <strong>{f4(p.macro?.f1)}</strong>
        {sem != null ? <span> ± {f4(sem)}</span> : null}{" "}
        <span className="sub" style={{ margin: 0 }}>
          macro (media por imagen, n = {p.images}) · micro {f4(p.micro?.f1)} ·
          {" "}IoU medio {p.mean_iou == null ? "sin emparejar" : p.mean_iou.toFixed(3)}
          {p.cached ? " · de caché" : ""}
        </span>
      </div>
      <div className="sub" style={{ margin: "2px 0 0" }}>
        P/R macro {f4(p.macro?.precision)} / {f4(p.macro?.recall)} ·
        {" "}micro tp {p.micro?.tp} fp {p.micro?.fp} fn {p.micro?.fn} ·
        {" "}split {p.split} · {p.checkpoint} · fuente{" "}
        <span className="mono">{p.source}</span>
      </div>
      <div className="sub" style={{ margin: "2px 0 0" }}>
        knobs (F): {Object.entries(p.knobs ?? {})
          .map(([k, v]) => `${k}=${v}`).join(" · ")}
      </div>
    </div>
  );
}

export function TaskScore(props: { runs: string[]; split?: string; title?: string }) {
  const [rows, setRows] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const split = props.split ?? "val";
  const runs = props.runs ?? [];

  const measure = async () => {
    setError(null); setBusy(true); setRows(null);
    try {
      const out: any[] = [];
      for (const r of runs)   // sequential: each one is full-image inference
        out.push(await api.get(`/runs/${r}/task-score?split=${split}`));
      setRows(out);
    } catch (e) { setError(e); } finally { setBusy(false); }
  };

  const small = rows?.length ? rows[0].images < 100 : false;
  const sem = rows?.length ? rows[0].macro?.sem : null;
  const mean = rows?.length
    ? rows.reduce((a, r) => a + r.macro.f1, 0) / rows.length : null;

  return (
    <div data-testid="task-score">
      <div className="row" style={{ alignItems: "center", gap: 8 }}>
        <button className="secondary" onClick={measure} disabled={busy || !runs.length}>
          {props.title ?? "Medir la métrica de tarea"}
        </button>
        <span className="sub" style={{ margin: 0 }}>
          párrafo por imagen contra la fuente (A) · {runs.length} run
          {runs.length === 1 ? "" : "s"} · no es el objetivo del ranking
        </span>
      </div>
      <Working on={busy} label="infiriendo imágenes completas…" />
      <ErrorBox error={error} />
      {rows?.length ? (
        <>
          {rows.map((p, i) => <One key={i} p={p} />)}
          {rows.length > 1 ? (
            <div className="sub" style={{ marginTop: 6 }}>
              media de las {rows.length} semillas: <strong>{f4(mean)}</strong>
            </div>
          ) : null}
          {small ? (
            <div className="warn" data-testid="task-small-sample" style={{
              marginTop: 8, padding: "8px 12px", borderRadius: 8,
              background: "var(--surface-2)", border: "1px solid var(--warn)",
            }}>
              <strong>Con {rows[0].images} imágenes este número distingue
                poco.</strong>{" "}
              El error estándar es ±{sem != null ? sem.toFixed(3) : "?"}, y las
              diferencias entre puntos vecinos de un recorrido son de 0,01 a 0,05:
              úsalo como informe del ganador, no para decidir entre puntos
              (metrica-de-tarea.md §4).
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
