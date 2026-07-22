import React, { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorBox, Working } from "../components/ui";

// A — read-only. Shows the ground truth: image + quads overlaid (the overlay
// and the img frame with the same rule: object-fit none, exact pixel size).
export default function Sources() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [count, setCount] = useState(0);
  const [index, setIndex] = useState(0);
  const [sample, setSample] = useState<any>(null);

  useEffect(() => {
    api.get("/sources").then(setData).catch(setError);
  }, []);

  useEffect(() => {
    if (!sel) return;
    setSample(null);
    api.get(`/sources/${sel}`).then((m) => setCount(m.count)).catch(setError);
    setIndex(0);
  }, [sel]);

  useEffect(() => {
    if (!sel) return;
    api.get(`/sources/${sel}/samples/${index}`).then(setSample).catch(setError);
  }, [sel, index]);

  return (
    <div>
      <h2>Fuentes (A)</h2>
      <p className="sub">Imágenes + geometría de párrafos. Solo lectura: aquí no se elige nada de ventanas.</p>
      <ErrorBox error={error} />
      <div className="row">
        <div className="card grow">
          <Working on={!data} />
          {data ? (
            <table className="data" data-testid="sources-table">
              <thead><tr><th>id</th><th>imágenes</th><th>derivada</th></tr></thead>
              <tbody>
                {data.sources.map((s: any) => (
                  <tr key={s.id} className={sel === s.id ? "sel" : ""}
                      onClick={() => setSel(s.id)}>
                    <td>{s.id}</td>
                    <td>{s.count ?? "?"}</td>
                    <td>{s.derived ? `← ${s.derived.from} ×${s.derived.scale?.[0]}` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {data && data.sources.length === 0 ? (
            <p>No hay fuentes. Genera una sintética con{" "}
              <code>python scripts/make_synth_source.py --name synth-01 --count 60</code>{" "}
              o apunta <code>FV_DATASETS_ROOT</code> al generador.</p>
          ) : null}
        </div>
        {sel ? (
          <div className="card grow">
            <h3 style={{ marginTop: 0 }}>{sel}</h3>
            <div className="row" style={{ alignItems: "center" }}>
              <button className="secondary" disabled={index <= 0}
                onClick={() => setIndex(index - 1)}>←</button>
              <span>imagen {index} / {count - 1}</span>
              <button className="secondary" disabled={index >= count - 1}
                onClick={() => setIndex(index + 1)}>→</button>
            </div>
            {sample ? (
              <div style={{ position: "relative", display: "inline-block", marginTop: 10 }}>
                <img src={`/api/sources/${sel}/samples/${index}/image`}
                  alt={`imagen ${index}`}
                  style={{ imageRendering: "pixelated", width: sample.width * 4,
                           height: sample.height * 4, display: "block" }} />
                <svg viewBox={`0 0 ${sample.width} ${sample.height}`}
                  style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                  {sample.blocks.map((b: any, i: number) => (
                    <polygon key={i}
                      points={b.quad.map((p: number[]) => p.join(",")).join(" ")}
                      fill="none" stroke="var(--accent)" strokeWidth={0.6} />
                  ))}
                </svg>
              </div>
            ) : <Working on={true} />}
            <p className="sub">La verdad de campo: los quads de párrafo dibujados sobre los píxeles.</p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
