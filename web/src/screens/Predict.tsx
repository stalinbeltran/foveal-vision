import React, { useEffect, useRef, useState } from "react";
import { api, CORNER_CSS, Corner, isTerminal } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field, Working } from "../components/ui";

// El TAMAÑO de un punto dice su score, y la escala es ABSOLUTA (0→1), no
// relativa a lo que haya en pantalla. Normalizar al min/max observado es la
// versión que se ve mejor y miente: mover el slider de `threshold` repintaría
// todos los puntos sin que el modelo hubiera cambiado de opinión — la imagen
// cambiaría de significado sin cambiar el dato. Es la misma razón por la que el
// color sigue a la entidad y nunca al rank (U3.7).
//
// ⚠ MEDIDO el 2026-08-31 con `demo-fov16-optimo` sobre 3 imágenes de
// `dirty1000-80px-16px-r20260827` (`predict_image`, stride por defecto):
// con `threshold` 0,5 los scores post-NMS van de 0,762 a 1,000 con **mediana
// 0,998**, así que en una red entrenada casi todos los círculos salen del mismo
// tamaño — y eso ES la lectura correcta: ninguna detección es dudosa. El tamaño
// informa cuando se baja el umbral (con 0,05 el primer cuartil cae a 0,561) o
// cuando la red es mala. Por eso el número exacto está SIEMPRE a mano: en el
// tooltip de cada punto y como rango por ranura en el panel de números — un
// círculo que no se distingue del de al lado no puede ser la única vía al dato.
//
// El radio va en unidades de la VENTANA, como los knobs de esta pantalla: así el
// punto pesa lo mismo respecto de lo que la red mira, sea cual sea el tamaño de
// la imagen. El suelo existe para que un 0,05 siga siendo visible y apuntable
// (con n=16 son ~7 px de diámetro en pantalla, que se dibuja a ×5).
const radioDe = (n: number) => (score: number) =>
  n * 0.045 + n * 0.12 * Math.min(1, Math.max(0, score));

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
  // Se recuerdan las ranuras OCULTAS, no las visibles, y la asimetría es la que
  // decide qué pasa con un valor viejo: si el vocabulario del payload creciera
  // (o cambiara de nombres), una lista de «visibles» guardada en el navegador
  // dejaría la ranura nueva invisible y en silencio. Guardando las ocultas, lo
  // que no se conoce se DIBUJA. Un dato escondido por un recuerdo es el fallo
  // caro de los dos.
  const [ocultas, setOcultas] = usePersistedState<Corner[]>("predict.slotsOff", []);
  const seq = useRef(0);

  // Lists are LIVE: a run that finishes (or a new source) while this screen is
  // open must appear without a reload — poll like Runs/Sweeps/Studies (3s). Each
  // pass reconciles the selection so an object deleted elsewhere is dropped from
  // the picker instead of 404-ing on the next predict.
  const refreshLists = () => {
    api.get("/runs").then((d) => {
      // Sólo los que la app PUEDE usar para inferir: `inference` los ordena
      // primero y dice de dónde saldrían los pesos («antesala» = entrenando
      // ahora, «catalogo» = aprobada y guardada).
      //
      // ⚠ Antes esto ofrecía TODOS los terminales — 861 el 2026-08-31, de los
      // que 860 fallaban al pulsar, y el preseleccionado era uno de ésos. Un
      // selector cuyo primer elemento da error enseña un fallo antes que una
      // imagen. Se marcan y se desactivan, no se esconden: un run escondido no
      // se distingue de uno que no existe, y entonces no sabes si aprobarlo o
      // reentrenarlo (es lo que /review ya hacía).
      const done = d.runs.filter((r: any) => isTerminal(r.status));
      done.sort((a: any, b: any) => Number(!a.inference) - Number(!b.inference));
      setRuns(done);
      const usable = (n: string) => done.some((r: any) => r.name === n && r.inference);
      setRun((cur) => (cur && usable(cur))
        ? cur
        : (done.find((r: any) => r.inference)?.name ?? ""));
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
  // el vocabulario de esquinas viene en la respuesta (corner_order), no de una
  // constante del front: el filtro ofrece lo que el payload declara (U4.2), así
  // que una ranura de más o de menos se refleja sola en vez de mentir.
  const ranuras: Corner[] = result?.corner_order ?? [];
  const visible = (c: string) => !ocultas.includes(c as Corner);
  const radio = radioDe(result?.knobs?.window_size ?? 16);
  const alternar = (c: Corner) =>
    setOcultas(ocultas.includes(c) ? ocultas.filter((x) => x !== c) : [...ocultas, c]);
  const rawVisto = (result?.raw ?? []).filter((d: any) => visible(d.corner));
  const corVisto = (result?.corners ?? []).filter((d: any) => visible(d.corner));
  // cuántas detecciones se está tragando el filtro, contando SOLO las etapas
  // encendidas: decir «12 ocultas» con la etapa cruda apagada sería un número
  // que no cuadra con lo que se ve, y un número que no cuadra se deja de creer
  const tapadas = (show.raw ? (result?.raw.length ?? 0) - rawVisto.length : 0)
    + (show.corners ? (result?.corners.length ?? 0) - corVisto.length : 0);

  return (
    <div>
      <h2 data-domain="F" data-view="V11" data-fixes="E, imagen"
        data-varies="la etapa" data-measures="que se pierde y donde">Predecir (F)</h2>
      <p className="sub">Las tres etapas — sin la cruda, «el párrafo salió mal» no es diagnosticable.
        Los knobs van en unidades de la ventana y no reentrenan nada.</p>
      <ErrorBox error={error} />
      {runs.length > 0 && !runs.some((r) => r.inference) ? (
        // El aviso que separa las dos causas: no es que falle la predicción, es
        // que ninguna red está aprobada aquí. Decirlo evita el diagnóstico que
        // ya costó una vez — «la red detecta mal» cuando lo que pasa es que no
        // hay red.
        <div className="card" style={{ borderColor: "var(--warn)" }}>
          <strong>Ninguno de los {runs.length} runs puede inferir aquí.</strong>
          <p className="sub" style={{ marginTop: 4 }}>
            Los pesos de un run <b>no se guardan por defecto</b>: sólo los de las
            redes aprobadas para inferencia (<code>inferencia.json</code> del repo
            de datos). Un run que nunca se aprobó no tiene pesos en ninguna parte
            y hay que <b>reentrenarlo</b>; uno que sí los tiene se aprueba con{" "}
            <code>POST /api/inference/staging/&lt;run&gt;/promote</code>.
          </p>
        </div>
      ) : null}
      <div className="card row" style={{ alignItems: "flex-end" }}>
        <div style={{ width: 200 }}><Field label="run">
          <select value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => (
              <option key={r.name} value={r.name} disabled={!r.inference}>
                {r.inference ? (r.inference === "antesala" ? "🟡 " : "") : "⛔ "}{r.name}
              </option>
            ))}
          </select></Field>
          <p className="sub" style={{ marginTop: 2 }}>
            ⛔ sin pesos aprobados · 🟡 entrenando (antesala)
          </p></div>
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
                {show.raw && rawVisto.map((d: any, i: number) => (
                  // la cruda es una NUBE: mismo tamaño por score que la de
                  // abajo (una sola escala, no dos que discutan) pero al 55 % y
                  // desvaída, para que las dos etapas encendidas a la vez se
                  // sigan distinguiendo por peso y no sólo por posición
                  <circle key={`r${i}`} cx={d.x} cy={d.y} r={radio(d.score) * 0.55}
                    fill="none" stroke={cs(d.corner)} strokeWidth={0.3} opacity={0.5}
                    pointerEvents="all">
                    <title>{`cruda ${d.corner} · score ${d.score.toFixed(3)} · (${d.x}, ${d.y})`}</title>
                  </circle>
                ))}
                {show.corners && corVisto.map((d: any, i: number) => {
                  const r = radio(d.score);
                  // El <svg> RECORTA, así que una etiqueta puesta a ciegas al
                  // lado del círculo se pierde en cuanto la esquina está pegada
                  // al borde — y las esquinas pegadas al borde son justo las que
                  // este proyecto mira (`edge_inputs`). Se mide antes de
                  // colocar: si no cabe a la derecha, va a la izquierda; y la
                  // altura se acota a la caja. El CÍRCULO no se mueve: eso sería
                  // mover el dato.
                  const ancho = 5;              // "TL" a fontSize 4, con holgura
                  const izq = d.x + r + 0.8 + ancho > W;
                  const lx = izq ? d.x - r - 0.8 : d.x + r + 0.8;
                  const ly = Math.min(H - 0.6, Math.max(3.4, d.y + 1));
                  return (
                    <g key={`c${i}`} pointerEvents="all">
                      <title>{`${d.corner} · score ${d.score.toFixed(3)} · (${d.x}, ${d.y})`}</title>
                      {/* anillo en color de superficie: dos esquinas que caen
                          casi encima se siguen leyendo como dos */}
                      <circle cx={d.x} cy={d.y} r={r} fill="none"
                        stroke="var(--surface)" strokeWidth={1.1} />
                      <circle cx={d.x} cy={d.y} r={r} fill="none"
                        stroke={cs(d.corner)} strokeWidth={0.6} />
                      {/* la etiqueta va en tinta de TEXTO, nunca en el color del
                          dato (U3.9); el color lo lleva el círculo de al lado.
                          El halo de superficie la hace legible sobre la imagen */}
                      <text x={lx} y={ly} fontSize={4}
                        textAnchor={izq ? "end" : "start"}
                        fill="var(--text)" stroke="var(--surface)" strokeWidth={0.7}
                        paintOrder="stroke">{d.corner}</text>
                    </g>
                  );
                })}
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
          {/* El filtro por ranura. Es leyenda Y filtro a la vez a propósito: la
              identidad de una esquina no puede ser sólo el color (U3.8), y la
              casilla que la enciende es el sitio donde ya está su nombre. */}
          <div className="curvelegend" data-testid="predict-corners"
            style={{ marginTop: 8, alignItems: "center" }}>
            <span className="sub">esquinas:</span>
            {ranuras.map((c) => {
              const off = !visible(c);
              const n = (result?.corners ?? []).filter((d: any) => d.corner === c).length;
              return (
                <label key={c} className={"legenditem" + (off ? " off" : "")}>
                  <input type="checkbox" checked={!off} onChange={() => alternar(c)} />
                  <span className="swatch"
                    style={{ background: off ? "var(--border)" : cs(c) }} />
                  <span className="mono">{c}</span>
                  <span className="sub">({n})</span>
                </label>
              );
            })}
            {ocultas.length ? (
              <button onClick={() => setOcultas([])}>todas</button>
            ) : null}
          </div>
          {ocultas.length ? (
            // Una imagen vacía POR EL FILTRO se lee exactamente igual que una
            // red que no detecta nada, y esa confusión ya costó un diagnóstico
            // en esta pantalla. Mientras el filtro esconda algo, lo dice.
            <p className="sub" style={{ marginTop: 4 }}>
              filtro activo: {ocultas.join(", ")} sin dibujar
              {tapadas > 0 ? ` · ${tapadas} detección(es) ocultas de las etapas encendidas` : ""}
            </p>
          ) : null}
          <p className="sub" style={{ marginTop: 4 }}>
            El tamaño del círculo es el <b>score</b> en escala absoluta 0→1 (no relativa
            a esta imagen): mover el umbral no cambia el tamaño de nada. El número exacto,
            en el tooltip de cada punto y por ranura en el panel de la derecha.
          </p>
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
                no de una constante del front.
                ⚠ Estos números NO los filtra el filtro: son la tabla gemela de
                lo que hay, y una ranura apagada se MARCA, no se descuenta. Y el
                rango de score es lo que hace legible que todos los círculos
                salgan iguales — con una red entrenada la mediana roza el 1,000
                y el tamaño deja de distinguir; el número sigue estando. */}
            {ranuras.map((c: Corner) => {
              const ss = result.corners.filter((d: any) => d.corner === c)
                .map((d: any) => d.score);
              const off = !visible(c);
              return (
                <div key={c} style={{ opacity: off ? 0.55 : 1 }}>
                  <span className={`corner-${c}`}>{c}</span>: {ss.length} esquinas
                  {" · score "}
                  {/* ausente ≠ cero (U5.3): sin esquinas no hay rango, y un
                      «0,000–0,000» se leería como una medida */}
                  {ss.length
                    ? `${Math.min(...ss).toFixed(3)}–${Math.max(...ss).toFixed(3)}`
                    : "—"}
                  {off ? " · oculta" : ""}
                </div>
              );
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}
