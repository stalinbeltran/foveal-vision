import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { BoxedImage } from "../components/BoxedImage";
import { ErrorBox, Field, Working } from "../components/ui";

// F x B -- mirar a ojo un SPLIT del dataset, con las cajas encima si hay modelo.
//
// ⚠ La pregunta de esta pantalla es "QUE DATASET del repo de datos quiero
// mirar", no "que run". Empezo al reves --el select principal era el run-- y en
// el server real eso son 859 opciones, con lo que la pantalla quedaba
// inservible. Ahora el dataset manda, la lista sale ya filtrada a los que
// TIENEN `windows.npz` (2 de 18 hoy) y los runs los devuelve el servidor ya
// reducidos a los de ese dataset.
//
// Y el run es OPCIONAL: sin modelo se ven las imagenes sin cajas, que es lo unico
// que se puede hacer en una maquina que solo tiene el repo de datos (los `*.pt`
// no viajan por git). Mirar sin modelo sigue siendo mirar, y queda registrado.

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
  const [run, setRun] = usePersistedState("review.run", "");
  const [ds, setDs] = usePersistedState("review.dataset", "");
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

  const runs: any[] = ctx?.runs ?? [];
  const datasets: any[] = ctx?.datasets ?? [];

  const recargarHistorial = () => {
    api.get("/review/sessions?days=45").then((d) => setSessions(d.sessions))
      .catch(() => { /* el historial es contexto: su fallo no rompe la revision */ });
    api.get("/review/marks").then((d) => setMarks(d.marks)).catch(() => {});
  };

  useEffect(() => {
    const q = new URLSearchParams({ split, count: String(n) });
    if (ds) q.set("window_dataset", ds);
    api.get(`/review/context?${q}`).then((c) => {
      setCtx(c); setError(null);
      // el servidor decide cual es el dataset valido (y cae al primero si el
      // recordado ya no existe): el front no puede tener su propia regla
      if (c.window_dataset !== ds) setDs(c.window_dataset);
      // un run recordado que no es de este dataset no sirve aqui
      if (run && !c.runs.some((r: any) => r.name === run)) setRun("");
    }).catch(setError);
    recargarHistorial();
  }, [ds, split, n]);

  useEffect(() => {
    if (!ctx?.window_dataset) return;
    const mio = ++seq.current;
    setBusy(true);
    const t = setTimeout(() => {
      api.post("/review/batch", {
        window_dataset: ctx.window_dataset, split, offset, count: n,
        run: run || undefined,
      })
        .then((r) => { if (seq.current === mio) { setBatch(r); setError(null); } })
        .catch((e) => { if (seq.current === mio) { setError(e); setBatch(null); } })
        .finally(() => { if (seq.current === mio) { setBusy(false); recargarHistorial(); } });
    }, 250);
    return () => clearTimeout(t);
  }, [ctx?.window_dataset, run, split, offset, n]);

  const marcar = async (img: any, marked: boolean) => {
    if (!batch) return;
    setBatch({
      ...batch,
      images: batch.images.map((x: any) =>
        x.index === img.index ? { ...x, marked } : x),
    });
    try {
      await api.post("/review/marks", {
        window_dataset: batch.window_dataset, split: batch.split,
        index: img.index, marked, source: batch.source, run: batch.run ?? "",
      });
      recargarHistorial();
    } catch (e) { setError(e); }
  };

  const porFecha = useMemo(() => {
    const g: Record<string, any[]> = {};
    for (const s of sessions) (g[cuandoRelativo(s.when)] ??= []).push(s);
    return ORDEN.filter((k) => g[k]?.length).map((k) => [k, g[k]] as const);
  }, [sessions]);

  // ⚠ Nada se pinta si no habla del dataset que ESTA ELEGIDO. Medido el
  // 2026-08-29 contra el server real: al cambiar de dataset la pagina paso ~3 s
  // diciendo "de 150" con el de 200 seleccionado, y ensenando sus miniaturas.
  //
  // Comparar el lote contra el CONTEXTO no arreglaba nada, y el intento quedo
  // escrito porque es la trampa: al cambiar el select, ctx y batch estan los DOS
  // atrasados, coinciden entre si, y el par obsoleto se valida solo. La
  // referencia tiene que ser `ds`, que es lo que el usuario acaba de elegir.
  const alDia = (x: any) => x && x.window_dataset === ds && x.split === split;
  const loteVigente = alDia(batch) ? batch : null;
  const vista = loteVigente ?? (alDia(ctx) ? ctx : null);
  const total = vista?.total ?? 0;
  const hasta = Math.min(offset + n, total);
  const pendientes = vista?.pending ?? 0;
  const revisadas = vista?.reviewed ?? 0;
  const conModelo = loteVigente?.inferred ?? false;
  const hayVerdad = vista?.truth_available ?? false;
  const base = vista?.image_base ?? "";

  return (
    <div className="review">
      <h2 data-domain="F×B" data-view="FR1" data-fixes="dataset + split"
        data-varies="el rango" data-measures="que detecta y que se le escapa">
        Revisar detecciones</h2>
      <p className="sub">Las imágenes del split, con las cajas de la red encima si
        eliges un run. Lo que se mire queda registrado, para poder elegir otras la
        próxima vez.</p>

      <div className="card revbar">
        <div className="revbar-line">
          <Field label="dataset">
            <select value={ds} onChange={(e) => { setDs(e.target.value); setOffset(0); }}>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.images})
                </option>
              ))}
            </select>
          </Field>
        </div>
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
          <Field label={`run (${runs.length})`}>
            {/* ⛔ se MARCA, no se esconde: un run sin best.pt escondido no se
                distingue de un run que no existe, y entonces no sabes si
                entrenar aqui o buscar en otro sitio */}
            <select value={run} onChange={(e) => setRun(e.target.value)}>
              <option value="">— sin modelo (sólo imágenes) —</option>
              {runs.map((r) => (
                <option key={r.name} value={r.name} disabled={!r.has_checkpoint}>
                  {r.has_checkpoint ? "" : "⛔ "}{r.name}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="revbar-nav">
          <button className="secondary" disabled={offset <= 0 || busy}
            onClick={() => setOffset(Math.max(0, offset - n))}>◀</button>
          <span className="rangelabel">
            {vista
              ? <>{total ? `${offset + 1}–${hasta}` : "—"} <span className="sub2">de {total}</span></>
              : <span className="sub2">cargando…</span>}
          </span>
          <button className="secondary" disabled={hasta >= total || busy}
            onClick={() => setOffset(offset + n)}>▶</button>
          <button className="secondary" disabled={busy || !pendientes}
            onClick={() => setOffset(vista?.next_offset ?? 0)}>sin revisar</button>
        </div>
        {hayVerdad ? (
          <label className="inline">
            <input type="checkbox" checked={showTruth}
              onChange={(e) => setShowTruth(e.target.checked)} /> ver la verdad
          </label>
        ) : null}
      </div>

      <ErrorBox error={error} />
      <Working on={busy || !vista} label="cargando…" />

      {vista ? (
        <p className="sub revstate">
          <b>{revisadas}</b> de {total} ya revisadas · <b>{pendientes}</b> pendientes
          {vista.storage?.in_data_repo === false ? (
            <span className="warnline"> · ⚠ fuera del repo de datos: esto NO se
              commitea en ningún sitio ({vista.storage?.path})</span>
          ) : null}
        </p>
      ) : null}

      {/* Lo que FALTA se dice, en vez de dibujar una rejilla sin cajas que se lea
          como "la red no detecta nada". Son dos ausencias distintas y se separan:
          el modelo (pesos) y la verdad (la fuente). */}
      {ctx && !conModelo ? (
        <div className="card avisobox" data-testid="review-missing">
          <b>Sin modelo:</b> se ven las imágenes del dataset, sin cajas.{" "}
          {runs.length === 0
            ? <>Este dataset no tiene ningún run en esta máquina.</>
            : runs.some((r) => r.has_checkpoint)
              ? <>Elige un run arriba.</>
              : <>Ninguno de sus {runs.length} runs tiene <code>best.pt</code> aquí:
                 los pesos no viajan por git (<code>*.pt</code> está en el
                 <code>.gitignore</code> del repo de datos).</>}
        </div>
      ) : null}
      {ctx && !hayVerdad ? (
        <div className="card avisobox">
          <b>Sin la verdad:</b> las imágenes salen del <code>windows.npz</code>, que
          sí viaja por git; los párrafos reales viven en la fuente
          (<code>{ctx.source}</code>), que no está en esta máquina.{" "}
          <span className="sub2">Publícala con <code>fv-publish-source</code>.</span>
        </div>
      ) : null}

      {loteVigente ? (
        <div className="thumbgrid rev" data-testid="review-grid">
          {loteVigente.images.map((img: any) => (
            <div className={`revthumb${img.marked ? " marked" : ""}`} key={img.index}>
              <Link to={`/review/${encodeURIComponent(loteVigente.window_dataset)}/${loteVigente.split}/${img.index}${run ? `?run=${encodeURIComponent(run)}` : ""}`}>
                <BoxedImage base={base} index={img.index}
                  width={img.width} height={img.height}
                  paragraphs={img.paragraphs} truth={img.truth}
                  showTruth={showTruth} fetchWidth={320} />
              </Link>
              <div className="revcap">
                <span className="mono">#{img.index}</span>
                {typeof img.f1 === "number" ? (
                  <span className={img.f1 >= 0.99 ? "ok" : img.f1 >= 0.5 ? "" : "bad"}>
                    f1 {img.f1.toFixed(2)}
                  </span>
                ) : conModelo ? (
                  <span className="sub2">{img.paragraphs.length} cajas</span>
                ) : null}
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
