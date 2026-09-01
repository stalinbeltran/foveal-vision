import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import {
  CLAVE_KNOBS, CLAVE_RUN, KNOBS_DEFECTO, SIN_ELEGIR, cuerpoKnobs,
} from "../reviewPrefs";
import { BoxedImage } from "../components/BoxedImage";
import { ErrorBox, Field, Working } from "../components/ui";

// F x B -- mirar a ojo un SPLIT del dataset, con las ESQUINAS encima si hay
// modelo (y los recuadros si se piden: ver `verEsquinas`/`verCajas`).
//
// ⚠ La pregunta de esta pantalla es "QUE DATASET del repo de datos quiero
// mirar", no "que run". Empezo al reves --el select principal era el run-- y en
// el server real eso son 859 opciones, con lo que la pantalla quedaba
// inservible. Ahora el dataset manda, la lista sale ya filtrada a los que
// TIENEN `windows.npz` (2 de 18 hoy) y los runs los devuelve el servidor ya
// reducidos a los de ese dataset.
//
// Y el run es OPCIONAL: sin modelo se ven las imagenes sin cajas, que es lo unico
// que se puede hacer en una maquina que solo tiene el repo de datos y ningun
// modelo (los `*.pt` de un run no viajan por git; desde el 2026-08-30 SI viaja
// uno de demostracion, `demo-*`). Mirar sin modelo sigue siendo mirar, y queda
// registrado.

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
  // ⚠ La clave cambio de `review.run` a `review.run2` a proposito: el defecto
  // anterior era "" y se escribia en localStorage con solo abrir la pantalla, o
  // sea que TODO navegador que ya la hubiera visto quedaria marcado como "elegi
  // mirar sin modelo" y no adoptaria nunca la sugerencia. Con una clave nueva la
  // distincion empieza limpia; la vieja queda de basura inofensiva.
  const [run, setRun] = usePersistedState(CLAVE_RUN, SIN_ELEGIR);
  const [knobs, setKnobs] = usePersistedState(CLAVE_KNOBS, KNOBS_DEFECTO);
  const [ds, setDs] = usePersistedState("review.dataset", "");
  const [split, setSplit] = usePersistedState("review.split", "val");
  const [n, setN] = usePersistedState("review.n", 10);
  const [offset, setOffset] = usePersistedState("review.offset", 0);
  const [showTruth, setShowTruth] = usePersistedState("review.truth", true);
  // Lo que se dibuja sobre la miniatura. Por decisión del dueño (2026-09-01) el
  // defecto es **las esquinas**, no los recuadros: en una miniatura de 320 px un
  // párrafo es un rectángulo grande que tapa lo que se está mirando, y lo que se
  // triA es si la red puso las esquinas donde tocaba. Los recuadros siguen a un
  // clic. ⚠ Claves propias y NO las del detalle: allí se mira UNA imagen grande
  // y lo que conviene ver es distinto — compartirlas haría que elegir en una
  // pantalla cambiara la otra sin pedirlo.
  const [verEsquinas, setVerEsquinas] = usePersistedState("review.grid.corners", true);
  const [verCajas, setVerCajas] = usePersistedState("review.grid.boxes", false);
  const [abierto, setAbierto] = usePersistedState("review.form", false);
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
      // Un run recordado que no es de este dataset no sirve aqui; y si no se ha
      // elegido nunca, se adopta el que sugiere el servidor. Los dos casos se
      // resuelven juntos a proposito: si se limpiara a "" por separado, quedaria
      // indistinguible de "el usuario quiere mirar sin modelo".
      const valido = run !== SIN_ELEGIR && run !== ""
        && c.runs.some((r: any) => r.name === run);
      if (!valido && run !== "") setRun(c.run_sugerido ?? "");
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
        run: run && run !== SIN_ELEGIR ? run : undefined,
        // las ESQUINAS sí, la nube cruda NO: medido el 2026-09-01, un lote de 10
        // pasa de 2 KB a 35 con las dos, y ~3/4 de eso son crudas que una
        // miniatura no puede ni dibujar. El detalle sí las pide.
        with_detections: true,
        ...cuerpoKnobs(knobs),
      })
        .then((r) => { if (seq.current === mio) { setBatch(r); setError(null); } })
        .catch((e) => { if (seq.current === mio) { setError(e); setBatch(null); } })
        .finally(() => { if (seq.current === mio) { setBusy(false); recargarHistorial(); } });
    }, 250);
    return () => clearTimeout(t);
    // los knobs entran en las dependencias enteros: mover cualquiera vuelve a
    // inferir el mismo rango, que es la unica forma de VER lo que hace. El
    // debounce de 250 ms de arriba es lo que evita una peticion por paso del
    // deslizador.
  }, [ctx?.window_dataset, run, split, offset, n, knobs.threshold, knobs.stride,
      knobs.nms_radius, knobs.min_size]);

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
  // Los knobs que el servidor devolvio del ultimo lote: con ellos se ve en que
  // se convirtio cada "auto". Sin lote todavia no hay nada que ensenar --
  // inventarse los numeros aqui seria la segunda copia de la regla.
  const efectivos = loteVigente?.knobs ?? null;
  // el tope de los deslizadores sale de la ventana de la RED, no de un 16 fijo:
  // una red con otra fovea necesita otro rango
  const ventana = efectivos?.window_size ?? 16;
  // Lo que se ve con el panel cerrado. Sale del CONTEXTO y de los knobs pedidos,
  // no del último lote: tiene que decir lo que está seleccionado AHORA, también
  // mientras se vuelve a inferir.
  const elegido = run === SIN_ELEGIR ? "" : run;
  const resumen = [
    ds || "—", split, elegido || "sin modelo",
    `umbral ${knobs.threshold.toFixed(2)}`,
  ].join(" · ");

  return (
    <div className="review">
      <h2 data-domain="F×B" data-view="FR1" data-fixes="dataset + split"
        data-varies="el rango" data-measures="que detecta y que se le escapa">
        Revisar detecciones</h2>
      <p className="sub">Las imágenes del split, con las <b>esquinas</b> de la red
        encima si eliges un run —y los recuadros si los pides—. Lo que se mire queda
        registrado, para poder elegir otras la próxima vez.</p>

      <div className="card revbar">
        {/* El formulario va PLEGADO por defecto y la navegación no.

            En un móvil los selectores ocupaban la pantalla entera y las
            miniaturas --que son lo que se viene a mirar-- quedaban debajo del
            pliegue. Pero lo que se pliega es sólo la ELECCIÓN, que se toca una
            vez por sesión; ◀ ▶ y "sin revisar" se usan en cada imagen y se
            quedan fuera. Plegar algo que se usa a cada paso sería cambiar un
            estorbo por dos toques.

            Y plegado NO puede ser mudo: el resumen dice qué hay elegido, porque
            un panel cerrado que esconde el run y el umbral hace que una rejilla
            rara sea indistinguible de un ajuste olvidado. */}
        <button className="revbar-toggle" aria-expanded={abierto}
          data-testid="review-form-toggle" onClick={() => setAbierto(!abierto)}>
          {abierto ? "▾ ajustes" : <>▸ ajustes <span className="sub2">{resumen}</span></>}
        </button>
        {abierto ? (
          <>
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
            <select value={run === SIN_ELEGIR ? "" : run}
              onChange={(e) => setRun(e.target.value)}>
              <option value="">— sin modelo (sólo imágenes) —</option>
              {runs.map((r) => (
                <option key={r.name} value={r.name} disabled={!r.has_checkpoint}>
                  {r.has_checkpoint ? "" : "⛔ "}{r.name}
                </option>
              ))}
            </select>
          </Field>
        </div>
        {/* Los mandos de INFERENCIA. Van aqui, en la rejilla, y no solo en el
            detalle: el efecto de un umbral se juzga sobre VARIAS imagenes a la
            vez -- en una sola no se distingue "el umbral esta alto" de "esta
            imagen es dificil". */}
        <div className="revbar-line">
          <Field label={`umbral ${knobs.threshold.toFixed(2)}`}
            help="score minimo de una esquina">
            <input type="range" min={0.05} max={0.95} step={0.05}
              value={knobs.threshold}
              onChange={(e) => setKnobs({ ...knobs, threshold: +e.target.value })} />
          </Field>
          <Field label={`nms ${knobs.nms_radius || "auto"}`}
            help="px entre dos esquinas iguales">
            <input type="range" min={0} max={2 * ventana} step={1}
              value={knobs.nms_radius}
              onChange={(e) => setKnobs({ ...knobs, nms_radius: +e.target.value })} />
          </Field>
        </div>
        <div className="revbar-line">
          <Field label={`paso ${knobs.stride || "auto"}`} help="px entre ventanas">
            <input type="range" min={0} max={ventana} step={1} value={knobs.stride}
              onChange={(e) => setKnobs({ ...knobs, stride: +e.target.value })} />
          </Field>
          <Field label={`caja min ${knobs.min_size || "auto"}`} help="px de lado">
            <input type="range" min={0} max={2 * ventana} step={1}
              value={knobs.min_size}
              onChange={(e) => setKnobs({ ...knobs, min_size: +e.target.value })} />
          </Field>
        </div>
        {/* Lo que se ensena es lo que el servidor USO, no lo que se pidio: es la
            unica forma de saber en que numero se convirtio cada "auto", y
            `window_size` sale de la RED (la fovea con la que se entreno), asi
            que no es ajustable por mucho que aparezca en la misma linea. */}
        {efectivos ? (
          <div className="sub2 mono" data-testid="review-knobs">
            en uso: umbral {efectivos.threshold} · paso {efectivos.stride} · nms{" "}
            {efectivos.nms_radius} · caja min {efectivos.min_size} · ventana{" "}
            {efectivos.window_size} (de la red, fija){" "}
            <button className="secondary" style={{ padding: "2px 8px" }}
              onClick={() => setKnobs(KNOBS_DEFECTO)}>volver a auto</button>
          </div>
        ) : null}
          </>
        ) : null}
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
        <label className="inline">
          <input type="checkbox" checked={verEsquinas}
            onChange={(e) => setVerEsquinas(e.target.checked)} /> esquinas
        </label>
        <label className="inline">
          <input type="checkbox" checked={verCajas}
            onChange={(e) => setVerCajas(e.target.checked)} /> recuadros
        </label>
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
          como "la red no detecta nada".

          ⚠ Eran DOS avisos, uno por ausencia --el modelo (pesos) y la verdad (la
          fuente)--, y el de la verdad se quito el 2026-08-30 a peticion del
          usuario. El motivo es el patron B del proyecto: en esta maquina no hay
          NINGUNA fuente publicada, asi que ese aviso salia en todas las pantallas
          y en todos los datasets, siempre igual, y un aviso que sale siempre se
          deja de leer -- y de paso tapaba al de "sin modelo", que si cambia.
          La ausencia NO queda muda: cuando no hay verdad no aparece la casilla
          "ver la verdad" ni los numeros de f1/tp/fp por imagen, que es la senal
          en el sitio donde se echa de menos. La explicacion larga vive donde se
          arregla (`fv-publish-source`), no en una tira permanente. */}
      {ctx && !conModelo ? (
        <div className="card avisobox" data-testid="review-missing">
          <b>Sin modelo:</b> se ven las imágenes del dataset, sin cajas.{" "}
          {runs.length === 0
            ? <>Este dataset no tiene ningún run en esta máquina.</>
            : runs.some((r) => r.has_checkpoint)
              ? <>Elige un run arriba.</>
              : <>Ninguno de sus {runs.length} runs tiene <code>best.pt</code> aquí:
                 los pesos de un run no viajan por git (<code>*.pt</code> está en el
                 <code>.gitignore</code> del repo de datos). Los <code>demo-*</code> sí
                 viajan: si tampoco hay ninguno, este dataset no tiene modelo de
                 demostración publicado.</>}
        </div>
      ) : null}

      {loteVigente ? (
        <div className="thumbgrid rev" data-testid="review-grid">
          {loteVigente.images.map((img: any) => (
            <div className={`revthumb${img.marked ? " marked" : ""}`} key={img.index}>
              <Link to={`/review/${encodeURIComponent(loteVigente.window_dataset)}/${loteVigente.split}/${img.index}${run ? `?run=${encodeURIComponent(run)}` : ""}`}>
                {/* ⚠ `cornerLabels={false}`: «TL» sobre una miniatura de 320 px
                    tapa la esquina que viene a enseñar. El color ya la
                    identifica y el nombre está en el detalle. */}
                <BoxedImage base={base} index={img.index}
                  width={img.width} height={img.height}
                  paragraphs={img.paragraphs} truth={img.truth}
                  showTruth={showTruth} showPred={verCajas}
                  corners={img.corners} showCorners={verEsquinas}
                  windowSize={loteVigente.knobs?.window_size ?? 16}
                  cornerLabels={false}
                  fetchWidth={320} />
              </Link>
              <div className="revcap">
                <span className="mono">#{img.index}</span>
                {typeof img.f1 === "number" ? (
                  <span className={img.f1 >= 0.99 ? "ok" : img.f1 >= 0.5 ? "" : "bad"}>
                    f1 {img.f1.toFixed(2)}
                  </span>
                ) : conModelo ? (
                  // ⚠ Se cuenta lo que se está VIENDO. Decía «2 cajas» con los
                  // recuadros apagados, o sea un número de algo que no está en
                  // la imagen — y el pie de una miniatura es justo lo que se lee
                  // para decidir si abrirla.
                  <span className="sub2">
                    {verEsquinas ? `${(img.corners ?? []).length} esq` : ""}
                    {verEsquinas && verCajas ? " · " : ""}
                    {verCajas ? `${img.paragraphs.length} cajas` : ""}
                    {!verEsquinas && !verCajas ? "sin overlay" : ""}
                  </span>
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
