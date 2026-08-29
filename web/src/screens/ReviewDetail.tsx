import React, { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { BoxedImage } from "../components/BoxedImage";
import { ErrorBox, Field, Working } from "../components/ui";

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
  const [run, setRun] = usePersistedState("review.run", "");
  const [threshold, setThreshold] = usePersistedState("review.threshold", 0.5);
  const [showTruth, setShowTruth] = usePersistedState("review.truth", true);
  const [showPred, setShowPred] = useState(true);
  const [ctx, setCtx] = useState<any>(null);
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [sucia, setSucia] = useState(false);   // ¿cambiaste algo desde la ultima?

  const runUrl = qs.get("run") || "";
  const elRun = runUrl || run;

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
      indices: [idx], threshold,
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
  const base = data?.image_base ?? ctx?.image_base ?? "";
  const hayVerdad = data?.truth_available ?? ctx?.truth_available ?? false;
  const runs: any[] = ctx?.runs ?? [];
  const dims: [number, number] | null = ctx?.image_size ?? null;

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
            showTruth={showTruth} showPred={showPred} />
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
        <Field label={`umbral ${threshold.toFixed(2)}`}>
          <input type="range" min={0.05} max={0.95} step={0.05} value={threshold}
            onChange={(e) => { setThreshold(+e.target.value); setSucia(true); }} />
        </Field>
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
            <dt>run</dt><dd className="mono">{data.run ?? "— ninguno —"}</dd>
            <dt>knobs</dt>
            <dd className="mono">{data.knobs ? JSON.stringify(data.knobs) : "—"}</dd>
          </dl>
        </div>
      ) : null}
    </div>
  );
}
