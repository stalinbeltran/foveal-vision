import React, { useEffect, useRef, useState } from "react";
import { api, CORNER_CSS, Corner, isTerminal } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field, Working } from "../components/ui";

// F — run + full image -> ALL the stages (raw / corners / paragraphs),
// switchable overlays; knobs are live sliders in WINDOW units, never retrain.
// The previous frame stays (dimmed) while a new one computes — with a spoken
// acknowledgement, or a slow response reads as a lost click.
export default function Predict() {
  const [runs, setRuns] = useState<any[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [run, setRun] = usePersistedState("predict.run", "");
  const [source, setSource] = usePersistedState("predict.source", "");
  const [index, setIndex] = useState(0);
  const [count, setCount] = useState(1);
  const [knobs, setKnobs] = usePersistedState("predict.knobs", { threshold: 0.5, stride: 0, nms_radius: 0 });
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [show, setShow] = usePersistedState("predict.show", { raw: false, corners: true, paragraphs: true, truth: true });
  const seq = useRef(0);

  // Lists are LIVE: a run that finishes (or a new source) while this screen is
  // open must appear without a reload — poll like Runs/Sweeps/Studies (3s). Each
  // pass reconciles the selection so an object deleted elsewhere is dropped from
  // the picker instead of 404-ing on the next predict.
  const refreshLists = () => {
    api.get("/runs").then((d) => {
      const done = d.runs.filter((r: any) => isTerminal(r.status));
      setRuns(done);
      setRun((cur) => (cur && done.some((r: any) => r.name === cur)) ? cur : (done[0]?.name ?? ""));
    }).catch(setError);
    api.get("/sources").then((d) => {
      setSources(d.sources);
      setSource((cur) => (cur && d.sources.some((s: any) => s.id === cur)) ? cur : (d.sources[0]?.id ?? ""));
    }).catch(setError);
  };
  useEffect(() => {
    refreshLists();
    const t = setInterval(refreshLists, 3000);
    return () => clearInterval(t);
  }, []);

  // Gate on membership, not truthiness: a remembered run/source (localStorage
  // predict.run / predict.source) may name an object that was deleted or
  // renamed; firing against it races the list fetches and surfaces run_not_found.
  // These booleans (NOT `runs`/`sources`) are the effect deps, so the 3s list
  // refresh above doesn't re-run the model on every poll — only a real
  // membership change does.
  const runReady = !!run && runs.some((r) => r.name === run);
  const sourceReady = !!source && sources.some((s) => s.id === source);

  useEffect(() => {
    if (!sourceReady) return;
    api.get(`/sources/${source}`).then((m) => setCount(m.count)).catch(setError);
  }, [source, sourceReady]);

  useEffect(() => {
    if (!runReady || !sourceReady) return;
    const mySeq = ++seq.current;
    setBusy(true);
    const body: any = { source, index, threshold: knobs.threshold };
    if (knobs.stride > 0) body.stride = knobs.stride;
    if (knobs.nms_radius > 0) body.nms_radius = knobs.nms_radius;
    const t = setTimeout(() => {
      api.post(`/runs/${run}/predict`, body).then((r) => {
        if (seq.current === mySeq) setResult(r);  // answers arrive out of order
      }).catch(setError).finally(() => {
        if (seq.current === mySeq) setBusy(false);
      });
    }, 250);
    return () => clearTimeout(t);
  }, [run, source, index, knobs, runReady, sourceReady]);

  const W = result?.image_size?.[0] ?? 96, H = result?.image_size?.[1] ?? 72;
  const cs = (c: string) => CORNER_CSS[c as keyof typeof CORNER_CSS];

  return (
    <div>
      <h2 data-domain="F" data-view="V11" data-fixes="E, imagen"
        data-varies="la etapa" data-measures="que se pierde y donde">Predecir (F)</h2>
      <p className="sub">Las tres etapas — sin la cruda, «el párrafo salió mal» no es diagnosticable.
        Los knobs van en unidades de la ventana y no reentrenan nada.</p>
      <ErrorBox error={error} />
      <div className="card row" style={{ alignItems: "flex-end" }}>
        <div style={{ width: 200 }}><Field label="run">
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => <option key={r.name}>{r.name}</option>)}
          </select></Field></div>
        <div style={{ width: 220 }}><Field label="fuente">
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            {sources.map((s) => <option key={s.id} value={s.id}>{s.id}</option>)}
          </select></Field></div>
        <div style={{ width: 140 }}><Field label={`imagen ${index}`}>
          <input type="range" min={0} max={Math.max(0, count - 1)} value={index}
            onChange={(e) => setIndex(+e.target.value)} /></Field></div>
        <div style={{ width: 180 }}><Field label={`threshold ${knobs.threshold.toFixed(2)}`}>
          <input type="range" min={0.05} max={0.95} step={0.05} value={knobs.threshold}
            onChange={(e) => setKnobs({ ...knobs, threshold: +e.target.value })} /></Field></div>
        <div style={{ width: 180 }}>
          <Field label={`stride ${knobs.stride || "auto (n/2)"}`}
            help="en px de la ventana">
          <input type="range" min={0} max={result?.knobs?.window_size ?? 16} step={1}
            value={knobs.stride}
            onChange={(e) => setKnobs({ ...knobs, stride: +e.target.value })} /></Field></div>
        <Working on={busy} label="prediciendo…" />
      </div>
      <div className="row">
        <div className="card">
          {result ? (
            <div style={{ position: "relative", display: "inline-block",
                          opacity: busy ? 0.55 : 1 }} data-testid="predict-stage">
              <img src={`/api/sources/${source}/samples/${index}/image`} alt="imagen"
                style={{ imageRendering: "pixelated", width: W * 5, height: H * 5,
                         display: "block" }} />
              <svg viewBox={`0 0 ${W} ${H}`}
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                {show.truth && result.truth?.map((t: any, i: number) => (
                  <polygon key={`t${i}`}
                    points={t.quad.map((p: number[]) => p.join(",")).join(" ")}
                    fill="none" stroke="var(--text-dim)" strokeDasharray="2 1.5"
                    strokeWidth={0.5} />
                ))}
                {show.raw && result.raw.map((d: any, i: number) => (
                  <circle key={`r${i}`} cx={d.x} cy={d.y} r={1} fill="none"
                    stroke={cs(d.corner)} strokeWidth={0.3} opacity={0.5} />
                ))}
                {show.corners && result.corners.map((d: any, i: number) => (
                  <g key={`c${i}`}>
                    <circle cx={d.x} cy={d.y} r={1.8} fill="none"
                      stroke={cs(d.corner)} strokeWidth={0.6} />
                    <text x={d.x + 2.4} y={d.y + 1} fontSize={4}
                      fill={cs(d.corner)}>{d.corner}</text>
                  </g>
                ))}
                {show.paragraphs && result.paragraphs.map((b: any, i: number) => (
                  <rect key={`p${i}`} x={b.x0} y={b.y0} width={b.x1 - b.x0}
                    height={b.y1 - b.y0} fill="none" stroke="var(--accent)"
                    strokeWidth={0.8} />
                ))}
              </svg>
            </div>
          ) : <Working on />}
          <div style={{ marginTop: 8 }}>
            {(["raw", "corners", "paragraphs", "truth"] as const).map((k) => (
              <label key={k} style={{ marginRight: 12 }}>
                <input type="checkbox" checked={show[k]}
                  onChange={(e) => setShow({ ...show, [k]: e.target.checked })} />
                {" "}{k === "truth" ? "verdad (quads)" : k}
              </label>
            ))}
          </div>
        </div>
        {result ? (
          <div className="card grow" data-testid="predict-numbers">
            <dl className="kv">
              <dt>ventanas crudas ≥ umbral</dt><dd>{result.raw.length}</dd>
              <dt>esquinas tras NMS</dt><dd>{result.corners.length}</dd>
              <dt>párrafos (TL→BR)</dt><dd>{result.paragraphs.length}</dd>
              <dt>knobs usados</dt><dd className="mono">{JSON.stringify(result.knobs)}</dd>
            </dl>
            <p className="sub">El payload devuelve los knobs con que se calculó: los sliders son
              en vivo y las respuestas llegan desordenadas.</p>
            {/* el vocabulario de esquinas viene en la respuesta (corner_order),
                no de una constante del front */}
            {(result.corner_order ?? []).map((c: Corner) => {
              const n = result.corners.filter((d: any) => d.corner === c).length;
              return <div key={c}><span className={`corner-${c}`}>{c}</span>: {n} esquinas</div>;
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}
