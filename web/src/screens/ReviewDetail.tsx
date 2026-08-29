import React, { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, isTerminal } from "../api";
import { usePersistedState } from "../uiState";
import { BoxedImage } from "../components/BoxedImage";
import { ErrorBox, Working } from "../components/ui";

// Una imagen sola, grande, con sus numeros. Es la pagina a la que se llega
// tocando una miniatura, y existe porque en un movil la rejilla sirve para
// TRIAR y no para diagnosticar: ahi no se distingue una caja corrida de una
// caja partida en dos.
//
// Pide la inferencia por el MISMO endpoint que la rejilla (`indices: [i]`), asi
// que las cajas de aqui son literalmente las de alli.
export default function ReviewDetail() {
  const { dataset = "", split = "val", index = "0" } = useParams();
  const [qs] = useSearchParams();
  const idx = parseInt(index, 10);
  const [run, setRun] = usePersistedState("review.run", "");
  const [showTruth, setShowTruth] = usePersistedState("review.truth", true);
  const [showPred, setShowPred] = useState(true);
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  // El run puede venir por la URL (desde la rejilla) o del recordado. Si no hay
  // ninguno, se coge el primero terminado en vez de dejar la pagina en blanco.
  const runUrl = qs.get("run") || "";
  useEffect(() => {
    if (runUrl) { setRun(runUrl); return; }
    if (run) return;
    api.get("/runs").then((d) => {
      const ok = d.runs.filter((r: any) => isTerminal(r.status));
      if (ok[0]) setRun(ok[0].name);
    }).catch(setError);
  }, [runUrl]);

  const elRun = runUrl || run;

  useEffect(() => {
    if (!elRun || Number.isNaN(idx)) return;
    setBusy(true);
    api.post("/review/batch", {
      run: elRun, window_dataset: dataset, split, indices: [idx],
    }).then((r) => { setData(r); setError(null); })
      .catch((e) => { setError(e); setData(null); })
      .finally(() => setBusy(false));
  }, [elRun, dataset, split, idx]);

  const img = data?.images?.[0];

  const marcar = async () => {
    if (!img) return;
    const marked = !img.marked;
    setData({ ...data, images: [{ ...img, marked }] });
    try {
      await api.post("/review/marks", {
        window_dataset: dataset, split, index: idx, marked,
        source: data.source, run: elRun,
      });
    } catch (e) { setError(e); }
  };

  return (
    <div className="review">
      <p className="backlink"><Link to="/review">‹ volver a la revisión</Link></p>
      <h2>#{idx} <span className="sub2">{split} · {dataset}</span></h2>
      <ErrorBox error={error} />
      <Working on={busy} label="infiriendo…" />
      {img ? (
        <>
          <div className="card detailwrap" data-testid="review-detail">
            <BoxedImage source={data.source} index={idx}
              width={img.width} height={img.height}
              paragraphs={img.paragraphs} truth={img.truth}
              showTruth={showTruth} showPred={showPred} />
          </div>
          <div className="card">
            <div className="toggles">
              <label className="inline">
                <input type="checkbox" checked={showPred}
                  onChange={(e) => setShowPred(e.target.checked)} /> párrafos detectados
              </label>
              <label className="inline">
                <input type="checkbox" checked={showTruth}
                  onChange={(e) => setShowTruth(e.target.checked)} /> la verdad
              </label>
              <button className={`markbtn big${img.marked ? " on" : ""}`}
                aria-pressed={img.marked} onClick={marcar}>
                {img.marked ? "★ marcada" : "☆ marcar para volver"}
              </button>
            </div>
            <dl className="kv">
              <dt>f1</dt><dd>{img.f1.toFixed(3)}</dd>
              <dt>aciertos / falsos / perdidos</dt>
              <dd>{img.tp} / {img.fp} / {img.fn}</dd>
              <dt>párrafos detectados</dt><dd>{img.paragraphs.length}</dd>
              <dt>párrafos reales</dt><dd>{img.truth.length}</dd>
              <dt>run</dt><dd className="mono">{data.run}</dd>
              <dt>knobs</dt><dd className="mono">{JSON.stringify(data.knobs)}</dd>
            </dl>
          </div>
        </>
      ) : null}
    </div>
  );
}
