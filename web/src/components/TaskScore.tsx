import React, { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorBox, Working } from "./ui";

const SPLITS = ["val", "test", "train"];

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
      {p.holdout_touches ? (
        // F14: the ledger only works if it is SEEN. "Once" is a rule nobody can
        // check while the count lives in a file nobody opens.
        <div className="sub" data-testid="task-holdout-touches"
             style={{ margin: "2px 0 0", color: "var(--warn)" }}>
          <strong>El holdout de este run se ha mirado {p.holdout_touches}{" "}
          {p.holdout_touches === 1 ? "vez" : "veces"}</strong>{" "}
          (queda anotado en <span className="mono">runs/{p.run}/holdout.jsonl</span>,
          también cuando el número sale de caché). El protocolo dice una sola vez,
          al final y solo con el ganador.
        </div>
      ) : null}
    </div>
  );
}

export function TaskScore(props: {
  runs: string[]; split?: string; title?: string;
  // `chooser` opens the holdout path (metrica-de-tarea.md §6.4.1): scoring
  // against ANOTHER B, which is what a holdout is. Off where it makes no sense
  // — the sweep verdict measures its winner's own val, and offering to point it
  // at a holdout there would invite touching the holdout while still choosing.
  chooser?: boolean;
}) {
  const [rows, setRows] = useState<any[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [wds, setWds] = useState<any[]>([]);
  const [dataset, setDataset] = useState("");     // "" = el del propio run
  const [split, setSplit] = useState(props.split ?? "val");
  const runs = props.runs ?? [];

  useEffect(() => {
    if (!props.chooser) return;
    api.get("/window-datasets").then((d) => setWds(d.window_datasets ?? []))
      .catch(() => { /* the block still works against the run's own B */ });
  }, [props.chooser]);

  const measure = async () => {
    setError(null); setBusy(true); setRows(null);
    try {
      const q = `split=${split}` + (dataset ? `&window_dataset=${dataset}` : "");
      const out: any[] = [];
      for (const r of runs)   // sequential: each one is full-image inference
        out.push(await api.get(`/runs/${r}/task-score?${q}`));
      setRows(out);
    } catch (e) { setError(e); } finally { setBusy(false); }
  };

  const small = rows?.length ? !!rows[0].small_sample : false;
  const sem = rows?.length ? rows[0].macro?.sem : null;
  const mean = rows?.length
    ? rows.reduce((a, r) => a + r.macro.f1, 0) / rows.length : null;

  return (
    <div data-testid="task-score">
      {props.chooser ? (
        <div className="row" style={{ alignItems: "center", gap: 8, marginBottom: 6 }}>
          <label className="sub" style={{ margin: 0 }}>dataset</label>
          <select data-testid="task-dataset" value={dataset}
                  onChange={(e) => setDataset(e.target.value)}>
            <option value="">el del propio run</option>
            {wds.map((w: any) => (
              <option key={w.name} value={w.name}>{w.name}</option>
            ))}
          </select>
          <label className="sub" style={{ margin: 0 }}>split</label>
          <select data-testid="task-split" value={split}
                  onChange={(e) => setSplit(e.target.value)}>
            {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      ) : null}
      {props.chooser && dataset ? (
        <div className="warn" data-testid="task-holdout-warn" style={{
          marginBottom: 8, padding: "8px 12px", borderRadius: 8,
          background: "var(--surface-2)", border: "1px solid var(--warn)",
        }}>
          <strong>Vas a puntuar contra otro dataset.</strong> Si es un holdout,
          <strong> se toca una sola vez, al final y solo con el ganador</strong>
          {" "}(protocolo.md §3). El val ya hace dos trabajos —elegir{" "}
          <span className="mono">best.pt</span> y rankear—, así que su número está
          sesgado al alza; el del holdout no, y por eso solo vale mirándolo una vez.
          Un dataset que salga de <em>la misma fuente</em> que el de entrenamiento
          se rechaza: no sería un holdout.
        </div>
      ) : null}
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
