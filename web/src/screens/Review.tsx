import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, isTerminal } from "../api";
import { usePersistedState } from "../uiState";
import { BoxedImage } from "../components/BoxedImage";
import { ErrorBox, Field, Working } from "../components/ui";

// F x B -- mirar a ojo lo que la red detecta sobre un SPLIT, con las cajas
// encima. La metrica de tarea dice CUANTO acierta; esto dice QUE falla.
//
// Pensada para el movil, que es donde se revisa: la barra de control es
// pegajosa, la rejilla baja a 2 columnas, y la miniatura la redimensiona el
// backend. Ver el bloque `@media` de tokens.css.
//
// El rango mirado se registra SOLO en el servidor, al inferir. Aqui no hay
// boton de "guardar": una anotacion que depende de que el usuario pulse algo
// despues es justo la que no se escribe.

const DIA = 86400000;

function cuandoRelativo(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "sin fecha";
  const hoy = new Date(); hoy.setHours(0, 0, 0, 0);
  const d = Math.floor((hoy.getTime() - new Date(t).setHours(0, 0, 0, 0)) / DIA);
  if (d <= 0) return "hoy";
  if (d === 1) return "ayer";
  if (d < 7) return "esta semana";
  if (d < 14) return "la semana pasada";
  if (d < 31) return "este mes";
  return "hace más de un mes";
}

// El orden de los grupos es FIJO y no alfabetico: es una linea de tiempo.
const ORDEN = ["hoy", "ayer", "esta semana", "la semana pasada", "este mes",
  "hace más de un mes", "sin fecha"];

export default function Review() {
  const [runs, setRuns] = useState<any[]>([]);
  const [run, setRun] = usePersistedState("review.run", "");
  const [split, setSplit] = usePersistedState("review.split", "val");
  const [n, setN] = usePersistedState("review.n", 10);
  const [offset, setOffset] = usePersistedState("review.offset", 0);
  const [showTruth, setShowTruth] = usePersistedState("review.truth", true);
  const [ctx, setCtx] = useState<any>(null);
  const [batch, setBatch] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [marks, setMarks] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const seq = useRef(0);

  // Solo runs terminados y CON checkpoint: sin best.pt no hay nada que inferir,
  // y ofrecerlo en el select es prometer una pantalla que va a fallar.
  useEffect(() => {
    api.get("/runs").then((d) => {
      const ok = d.runs.filter((r: any) => isTerminal(r.status));
      setRuns(ok);
      setRun((cur) => (cur && ok.some((r: any) => r.name === cur))
        ? cur : (ok[0]?.name ?? ""));
    }).catch(setError);
  }, []);

  const runReady = !!run && runs.some((r) => r.name === run);

  const recargarHistorial = () => {
    api.get("/review/sessions?days=45").then((d) => setSessions(d.sessions))
      .catch(() => { /* el historial es contexto: su fallo no rompe la revision */ });
    api.get("/review/marks").then((d) => setMarks(d.marks)).catch(() => {});
  };

  useEffect(() => {
    if (!runReady) return;
    api.get(`/review/context?run=${encodeURIComponent(run)}&split=${split}&count=${n}`)
      .then(setCtx).catch(setError);
    recargarHistorial();
  }, [run, split, n, runReady]);

  // La inferencia sale SOLA al cambiar de split (o de run, o de N): eso es lo
  // que se pidio. El debounce es por el input de N, que dispara por tecla.
  useEffect(() => {
    if (!runReady) return;
    const mio = ++seq.current;
    setBusy(true);
    const t = setTimeout(() => {
      api.post("/review/batch", { run, split, offset, count: n })
        .then((r) => { if (seq.current === mio) { setBatch(r); setError(null); } })
        .catch((e) => { if (seq.current === mio) { setError(e); setBatch(null); } })
        .finally(() => { if (seq.current === mio) { setBusy(false); recargarHistorial(); } });
    }, 250);
    return () => clearTimeout(t);
  }, [run, split, offset, n, runReady]);

  const marcar = async (img: any, marked: boolean) => {
    if (!batch) return;
    // pintado optimista: en el movil esperar al servidor para que se rellene
    // una estrella se lee como que el toque no entro
    setBatch({
      ...batch,
      images: batch.images.map((x: any) =>
        x.index === img.index ? { ...x, marked } : x),
    });
    try {
      await api.post("/review/marks", {
        window_dataset: batch.window_dataset, split: batch.split,
        index: img.index, marked, source: batch.source, run: batch.run,
      });
      recargarHistorial();
    } catch (e) { setError(e); }
  };

  const porFecha = useMemo(() => {
    const g: Record<string, any[]> = {};
    for (const s of sessions) (g[cuandoRelativo(s.when)] ??= []).push(s);
    return ORDEN.filter((k) => g[k]?.length).map((k) => [k, g[k]] as const);
  }, [sessions]);

  // Los contadores salen del LOTE cuando lo hay, y solo si no, del contexto.
  // Mezclarlos daba una linea que se contradecia sola ("0 de 10 ya revisadas ·
  // 0 pendientes"): `ctx` se pide antes de inferir, asi que su `reviewed` es de
  // antes de que esta misma tanda quedara registrada. Un numero de dos fuentes
  // con dos edades es peor que un numero viejo.
  const vista = batch ?? ctx;
  const total = vista?.total ?? 0;
  const hasta = Math.min(offset + n, total);
  const pendientes = vista?.pending ?? 0;
  const revisadas = vista?.reviewed ?? 0;

  return (
    <div className="review">
      <h2 data-domain="F×B" data-view="FR1" data-fixes="run + split"
        data-varies="el rango" data-measures="que detecta y que se le escapa">
        Revisar detecciones</h2>
      <p className="sub">Las cajas de la red sobre las imágenes del split. Lo que se
        mire queda registrado, para poder elegir otras la próxima vez.</p>

      <div className="card revbar">
        <div className="revbar-line">
          <Field label="split">
            <select value={split} onChange={(e) => { setSplit(e.target.value); setOffset(0); }}>
              {(ctx?.splits ?? ["train", "val", "test"]).map((s: string) =>
                <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
          <Field label="cuántas">
            <input type="number" min={1} max={60} value={n}
              onChange={(e) => setN(Math.max(1, Math.min(60, +e.target.value || 1)))} />
          </Field>
        </div>
        <div className="revbar-line">
          <Field label="run">
            <select value={run} onChange={(e) => setRun(e.target.value)}>
              {runs.map((r) => <option key={r.name}>{r.name}</option>)}
            </select>
          </Field>
        </div>
        <div className="revbar-nav">
          <button className="secondary" disabled={offset <= 0 || busy}
            onClick={() => setOffset(Math.max(0, offset - n))}>◀</button>
          <span className="rangelabel">
            {total ? `${offset + 1}–${hasta}` : "—"} <span className="sub2">de {total}</span>
          </span>
          <button className="secondary" disabled={hasta >= total || busy}
            onClick={() => setOffset(offset + n)}>▶</button>
          <button className="secondary" disabled={busy || !pendientes}
            onClick={() => setOffset(batch?.next_offset ?? ctx?.next_offset ?? 0)}>
            sin revisar
          </button>
        </div>
        <label className="inline">
          <input type="checkbox" checked={showTruth}
            onChange={(e) => setShowTruth(e.target.checked)} /> ver la verdad
        </label>
      </div>

      <ErrorBox error={error} />
      <Working on={busy} label="infiriendo…" />

      {vista ? (
        <p className="sub revstate">
          <b>{revisadas}</b> de {total} ya revisadas · <b>{pendientes}</b> pendientes
          {vista.storage?.in_data_repo === false ? (
            <span className="warnline"> · ⚠ fuera del repo de datos: esto NO se
              commitea en ningún sitio ({vista.storage?.path})</span>
          ) : null}
        </p>
      ) : null}

      {batch ? (
        <div className="thumbgrid rev" data-testid="review-grid">
          {batch.images.map((img: any) => (
            <div className={`revthumb${img.marked ? " marked" : ""}`} key={img.index}>
              <Link to={`/review/${encodeURIComponent(batch.window_dataset)}/${batch.split}/${img.index}?run=${encodeURIComponent(batch.run)}`}>
                <BoxedImage source={batch.source} index={img.index}
                  width={img.width} height={img.height}
                  paragraphs={img.paragraphs} truth={img.truth}
                  showTruth={showTruth} fetchWidth={320} />
              </Link>
              <div className="revcap">
                <span className="mono">#{img.index}</span>
                <span className={img.f1 >= 0.99 ? "ok" : img.f1 >= 0.5 ? "" : "bad"}>
                  f1 {img.f1.toFixed(2)}
                </span>
                <button className={`markbtn${img.marked ? " on" : ""}`}
                  aria-pressed={img.marked}
                  title={img.marked ? "quitar la marca" : "marcar para volver"}
                  onClick={() => marcar(img, !img.marked)}>
                  {img.marked ? "★" : "☆"}
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {marks.length ? (
        <div className="card">
          <h3>Marcadas para volver ({marks.length})</h3>
          <div className="marklist">
            {marks.map((m) => (
              <Link key={`${m.window_dataset}|${m.split}|${m.index}`}
                className="markchip"
                to={`/review/${encodeURIComponent(m.window_dataset)}/${m.split}/${m.index}`}>
                <b>#{m.index}</b> <span className="sub2">{m.split}</span>
              </Link>
            ))}
          </div>
        </div>
      ) : null}

      {porFecha.length ? (
        <div className="card">
          <h3>Ya revisadas</h3>
          {porFecha.map(([etiqueta, ss]) => (
            <details key={etiqueta} open={etiqueta === "hoy"}>
              <summary>{etiqueta} <span className="sub2">({ss.length})</span></summary>
              <table className="data">
                <tbody>
                  {ss.map((s: any, i: number) => (
                    <tr key={i}>
                      <td className="mono">{new Date(s.when).toLocaleTimeString()}</td>
                      <td>{s.split}</td>
                      <td className="mono">{s.offset + 1}–{s.offset + s.count}</td>
                      <td className="sub2">{s.window_dataset}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          ))}
        </div>
      ) : null}
    </div>
  );
}
