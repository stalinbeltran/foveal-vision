import React, { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import {
  CLAVE_KNOBS, CLAVE_RUN, KNOBS_DEFECTO, SIN_ELEGIR, cuerpoKnobs,
} from "../reviewPrefs";
import { BoxedImage } from "../components/BoxedImage";
import { CornerFilter, CornerFilterAviso } from "../components/CornerDots";
import { ErrorBox, Field, Working } from "../components/ui";
import { Corner } from "../api";

// Una imagen sola, grande, con sus numeros. Es la pagina a la que se llega
// tocando una miniatura, y existe porque en un movil la rejilla sirve para
// TRIAR y no para diagnosticar: ahi no se distingue una caja corrida de una
// caja partida en dos.
//
// La INFERENCIA VA A BOTON, y esa es la diferencia con la rejilla:
//  - la imagen se pinta enseguida (el contexto dice de donde sacarla, y eso es
//    una consulta barata), asi que la pagina nunca espera al modelo;
//  - y aqui si tiene sentido repetirla, porque se pueden cambiar el run y el
//    umbral. Un boton sin nada que cambiar seria un boton de recargar.
// Se lanza sola UNA vez al entrar para no llegar a una pagina vacia; a partir de
// ahi manda el boton.
//
// Pide la inferencia por el MISMO endpoint que la rejilla (`indices: [i]`), asi
// que las cajas de aqui son literalmente las de alli.
export default function ReviewDetail() {
  const { dataset = "", split = "val", index = "0" } = useParams();
  const [qs] = useSearchParams();
  const idx = parseInt(index, 10);
  // ⚠ La MISMA clave que la rejilla (`reviewPrefs`), no una propia: cuando eran
  // dos, elegir el run en la rejilla y abrir una imagen te devolvia al recordado
  // de esta pantalla -- que podia ser otro, o ninguno.
  const [run, setRun] = usePersistedState(CLAVE_RUN, SIN_ELEGIR);
  const [knobs, setKnobs] = usePersistedState(CLAVE_KNOBS, KNOBS_DEFECTO);
  const [showTruth, setShowTruth] = usePersistedState("review.truth", true);
  const [showPred, setShowPred] = useState(true);
  // Las esquinas de la inferencia, que es lo que la caja NO enseña: una caja
  // corrida y una esquina que se fue son fallos distintos con la misma pinta.
  const [showCorners, setShowCorners] = usePersistedState("review.corners", true);
  const [showRaw, setShowRaw] = usePersistedState("review.raw", false);
  // Se recuerdan las ranuras OCULTAS, no las visibles: si el vocabulario del
  // payload creciera, una lista de «visibles» guardada dejaría la ranura nueva
  // invisible y en silencio. Lo que no se conoce, se DIBUJA.
  const [ocultas, setOcultas] = usePersistedState<Corner[]>("review.slotsOff", []);
  const [ctx, setCtx] = useState<any>(null);
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [sucia, setSucia] = useState(false);   // ¿cambiaste algo desde la ultima?

  const runUrl = qs.get("run") || "";
  // el orden manda: lo que dice la URL (se llego pinchando una miniatura), luego
  // lo elegido, y si no se ha elegido nunca, lo que sugiere el servidor
  const elRun = runUrl || (run === SIN_ELEGIR ? (ctx?.run_sugerido ?? "") : run);

  // UNA consulta barata: de donde sale el PNG, si hay verdad, y que runs tiene
  // este dataset (ya filtrados por el servidor). Sin esto la imagen tendria que
  // esperar a la inferencia, que es justo lo que el boton evita.
  useEffect(() => {
    if (!dataset) return;
    api.get(`/review/context?window_dataset=${encodeURIComponent(dataset)}&split=${split}`)
      .then((c) => { setCtx(c); setError(null); })
      .catch(setError);
  }, [dataset, split]);

  const inferir = () => {
    if (Number.isNaN(idx)) return;
    setBusy(true);
    api.post("/review/batch", {
      run: elRun || undefined, window_dataset: dataset, split,
      // aqui SI se piden las detecciones: es una imagen sola y grande, que es
      // justo donde mirar lo que la red vio antes de la caja tiene sentido. La
      // rejilla no las pide (ver `with_detections` en el API)
      indices: [idx], with_detections: true, ...cuerpoKnobs(knobs),
    }).then((r) => { setData(r); setError(null); setSucia(false); })
      .catch((e) => { setError(e); })
      .finally(() => setBusy(false));
  };

  // una sola vez al entrar, para no aterrizar en una pagina vacia
  const [arrancado, setArrancado] = useState(false);
  useEffect(() => {
    if (arrancado || !ctx || Number.isNaN(idx)) return;
    setArrancado(true);
    inferir();
  }, [ctx, idx, arrancado]);

  const img = data?.images?.[0];
  // el tope de los deslizadores sale de la ventana de la RED (la fovea con la
  // que se entreno), no de un 16 cableado
  const ventana = data?.knobs?.window_size ?? 16;
  const base = data?.image_base ?? ctx?.image_base ?? "";
  const hayVerdad = data?.truth_available ?? ctx?.truth_available ?? false;
  const runs: any[] = ctx?.runs ?? [];
  const dims: [number, number] | null = ctx?.image_size ?? null;

  // el vocabulario lo sirve el payload (U4.2); sin modelo llega `null` y
  // entonces no hay ranuras de que hablar ni filtro que enseñar
  const orden: Corner[] = data?.corner_order ?? [];
  const alternar = (c: Corner) =>
    setOcultas(ocultas.includes(c) ? ocultas.filter((x) => x !== c) : [...ocultas, c]);
  const cuenta: Record<string, number> = Object.fromEntries(orden.map((c) => [c,
    (img?.corners ?? []).filter((d: any) => d.corner === c).length]));
  // cuántas detecciones se traga el filtro, contando SOLO las capas encendidas:
  // un número que no cuadra con lo que se ve se deja de creer
  const tapadas = (showCorners
      ? (img?.corners ?? []).filter((d: any) => ocultas.includes(d.corner)).length : 0)
    + (showRaw ? (img?.raw ?? []).filter((d: any) => ocultas.includes(d.corner)).length : 0);

  const marcar = async () => {
    if (!img) return;
    const marked = !img.marked;
    setData({ ...data, images: [{ ...img, marked }] });
    try {
      await api.post("/review/marks", {
        window_dataset: dataset, split, index: idx, marked,
        source: data.source, run: elRun || "",
      });
    } catch (e) { setError(e); }
  };

  return (
    <div className="review">
      <p className="backlink"><Link to="/review">‹ volver a la revisión</Link></p>
      <h2>#{idx} <span className="sub2">{split} · {dataset}</span></h2>
      <ErrorBox error={error} />

      {base ? (
        <div className="card detailwrap" data-testid="review-detail">
          <BoxedImage base={base} index={idx}
            width={img?.width ?? dims?.[0] ?? 80}
            height={img?.height ?? dims?.[1] ?? 60}
            paragraphs={img?.paragraphs} truth={img?.truth}
            showTruth={showTruth} showPred={showPred}
            corners={img?.corners} raw={showRaw ? img?.raw : []}
            showCorners={showCorners} hiddenCorners={ocultas}
            windowSize={ventana} />
        </div>
      ) : <Working on label="cargando la imagen…" />}

      <div className="card">
        <div className="revbar-line">
          <Field label={`run (${runs.length})`}>
            <select value={elRun} onChange={(e) => { setRun(e.target.value); setSucia(true); }}>
              <option value="">— sin modelo (sólo la imagen) —</option>
              {runs.map((r) => (
                <option key={r.name} value={r.name} disabled={!r.has_checkpoint}>
                  {r.has_checkpoint ? "" : "⛔ "}{r.name}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="revbar-line">
          <Field label={`umbral ${knobs.threshold.toFixed(2)}`}
            help="score minimo de una esquina">
            <input type="range" min={0.05} max={0.95} step={0.05}
              value={knobs.threshold}
              onChange={(e) => { setKnobs({ ...knobs, threshold: +e.target.value }); setSucia(true); }} />
          </Field>
          <Field label={`nms ${knobs.nms_radius || "auto"}`}
            help="px entre dos esquinas iguales">
            <input type="range" min={0} max={2 * ventana} step={1}
              value={knobs.nms_radius}
              onChange={(e) => { setKnobs({ ...knobs, nms_radius: +e.target.value }); setSucia(true); }} />
          </Field>
        </div>
        <div className="revbar-line">
          <Field label={`paso ${knobs.stride || "auto"}`} help="px entre ventanas">
            <input type="range" min={0} max={ventana} step={1} value={knobs.stride}
              onChange={(e) => { setKnobs({ ...knobs, stride: +e.target.value }); setSucia(true); }} />
          </Field>
          <Field label={`caja min ${knobs.min_size || "auto"}`} help="px de lado">
            <input type="range" min={0} max={2 * ventana} step={1}
              value={knobs.min_size}
              onChange={(e) => { setKnobs({ ...knobs, min_size: +e.target.value }); setSucia(true); }} />
          </Field>
        </div>
        {data?.knobs ? (
          <div className="sub2 mono" data-testid="review-knobs">
            en uso: umbral {data.knobs.threshold} · paso {data.knobs.stride} · nms{" "}
            {data.knobs.nms_radius} · caja min {data.knobs.min_size} · ventana{" "}
            {data.knobs.window_size} (de la red, fija){" "}
            <button className="secondary" style={{ padding: "2px 8px" }}
              onClick={() => { setKnobs(KNOBS_DEFECTO); setSucia(true); }}>
              volver a auto</button>
          </div>
        ) : null}
        <button className="inferbtn" onClick={inferir} disabled={busy}
          data-testid="infer-now">
          {busy ? "infiriendo…" : data ? "Inferir de nuevo" : "Inferir ahora"}
        </button>
        {sucia && !busy ? (
          <span className="sub2 dirtyhint"> · cambiaste algo: lo de abajo es de la
            inferencia anterior</span>
        ) : null}
        <Working on={busy} label="infiriendo…" />
      </div>

      {img ? (
        <div className="card">
          <div className="toggles">
            <label className="inline">
              <input type="checkbox" checked={showPred}
                onChange={(e) => setShowPred(e.target.checked)} /> párrafos detectados
            </label>
            <label className="inline">
              <input type="checkbox" checked={showCorners}
                onChange={(e) => setShowCorners(e.target.checked)} /> esquinas
            </label>
            <label className="inline">
              <input type="checkbox" checked={showRaw}
                onChange={(e) => setShowRaw(e.target.checked)} /> crudas (pre-NMS)
            </label>
            {hayVerdad ? (
              <label className="inline">
                <input type="checkbox" checked={showTruth}
                  onChange={(e) => setShowTruth(e.target.checked)} /> la verdad
              </label>
            ) : null}
            <button className={`markbtn big${img.marked ? " on" : ""}`}
              aria-pressed={img.marked} onClick={marcar}>
              {img.marked ? "★ marcada" : "☆ marcar para volver"}
            </button>
          </div>
          {showCorners && orden.length ? (
            <div data-testid="review-corners">
              <CornerFilter order={orden} hidden={ocultas}
                onToggle={alternar} onAll={() => setOcultas([])} counts={cuenta} />
              <CornerFilterAviso hidden={ocultas} tapadas={tapadas} />
            </div>
          ) : null}
          <dl className="kv">
            {typeof img.f1 === "number" ? (
              <>
                <dt>f1</dt><dd>{img.f1.toFixed(3)}</dd>
                <dt>aciertos / falsos / perdidos</dt>
                <dd>{img.tp} / {img.fp} / {img.fn}</dd>
              </>
            ) : null}
            <dt>párrafos detectados</dt>
            <dd>{data.inferred ? img.paragraphs.length : "— (sin modelo)"}</dd>
            {hayVerdad ? (
              <><dt>párrafos reales</dt><dd>{img.truth.length}</dd></>
            ) : null}
            {orden.map((c) => {
              const ss = (img.corners ?? []).filter((d: any) => d.corner === c)
                .map((d: any) => d.score);
              return (
                <React.Fragment key={c}>
                  <dt className={`corner-${c}`}>
                    {c}{ocultas.includes(c) ? " (oculta)" : ""}</dt>
                  {/* ausente ≠ cero (U5.3): sin esquinas no hay rango, y un
                      «0,000–0,000» se leería como una medida */}
                  <dd>{ss.length} · score {ss.length
                    ? `${Math.min(...ss).toFixed(3)}–${Math.max(...ss).toFixed(3)}`
                    : "—"}</dd>
                </React.Fragment>
              );
            })}
            <dt>run</dt><dd className="mono">{data.run ?? "— ninguno —"}</dd>
            <dt>knobs</dt>
            <dd className="mono">{data.knobs ? JSON.stringify(data.knobs) : "—"}</dd>
          </dl>
        </div>
      ) : null}
    </div>
  );
}
