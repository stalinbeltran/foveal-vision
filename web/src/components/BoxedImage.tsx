import React from "react";

// COMO se dibuja una deteccion, definido UNA vez: la miniatura de la rejilla y
// la pagina de detalle son la misma imagen a dos tamanos, no dos dibujos. Dos
// copias de esto divergen en el color o en el grosor y entonces "la miniatura
// se ve distinta que el detalle" es un bug que nadie sabe donde mirar.
//
// La imagen la sirve y la REDIMENSIONA el backend (`?w=`), que en un movil es
// la diferencia entre 240 px y la imagen entera por cada miniatura. El overlay
// va en un <svg> con viewBox en coordenadas de la IMAGEN, asi que las cajas
// escalan solas con el contenedor y no hay que convertir nada a pixeles de
// pantalla.

export type Box = { x0: number; y0: number; x1: number; y1: number };

export function BoxedImage(props: {
  source: string;
  index: number;
  width: number;
  height: number;
  paragraphs?: Box[];
  truth?: number[][][];
  showTruth?: boolean;
  showPred?: boolean;
  /** ancho que se le pide al backend; sin el, la imagen entera */
  fetchWidth?: number;
  alt?: string;
}) {
  const { source, index, width: W, height: H } = props;
  const q = props.fetchWidth ? `?w=${props.fetchWidth}` : "";
  const showPred = props.showPred ?? true;
  const showTruth = props.showTruth ?? true;
  // el trazo se define en unidades de la IMAGEN (viewBox), asi que a 80 px de
  // ancho un 0.8 se ve igual de grueso en la miniatura y en el detalle
  const sw = Math.max(0.6, W / 110);
  return (
    <div className="boxed">
      <img
        src={`/api/sources/${source}/samples/${index}/image${q}`}
        alt={props.alt ?? `imagen ${index}`}
        loading="lazy"
        width={W}
        height={H}
      />
      <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
        {showTruth && (props.truth ?? []).map((quad, i) => (
          <polygon
            key={`t${i}`}
            points={quad.map((p) => p.join(",")).join(" ")}
            fill="none"
            stroke="var(--text-dim)"
            strokeDasharray={`${sw * 2.5} ${sw * 2}`}
            strokeWidth={sw * 0.8}
          />
        ))}
        {showPred && (props.paragraphs ?? []).map((b, i) => (
          <rect
            key={`p${i}`}
            x={b.x0}
            y={b.y0}
            width={b.x1 - b.x0}
            height={b.y1 - b.y0}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={sw}
          />
        ))}
      </svg>
    </div>
  );
}
