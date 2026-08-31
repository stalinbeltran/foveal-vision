import React from "react";
import { CORNER_CSS, Corner } from "../api";

// CÓMO se dibuja un punto de inferencia y cómo se filtra por ranura, definido
// UNA vez — igual que `BoxedImage` hace con las cajas. Lo usan Predecir (F) y el
// detalle de Revisar, que son la misma pregunta sobre la misma imagen; dos
// copias divergirían en la escala o en el color y entonces «aquí se ve distinto
// que allí» sería un fallo que nadie sabe dónde mirar.

export type Deteccion = { corner: string; score: number; x: number; y: number };

// El TAMAÑO de un punto dice su score, y la escala es ABSOLUTA (0→1), no
// relativa a lo que haya en pantalla. Normalizar al min/max observado es la
// versión que se ve mejor y miente: mover el slider del umbral repintaría todos
// los puntos sin que el modelo hubiera cambiado de opinión — la imagen
// cambiaría de significado sin cambiar el dato. Es la misma razón por la que el
// color sigue a la entidad y nunca al rank (U3.7).
//
// ⚠ MEDIDO el 2026-08-31 con `demo-fov16-optimo` sobre 3 imágenes de
// `dirty1000-80px-16px-r20260827`: con umbral 0,5 los scores post-NMS van de
// 0,762 a 1,000 con **mediana 0,998**, así que en una red entrenada casi todos
// los círculos salen del mismo tamaño — y eso ES la lectura correcta: ninguna
// detección es dudosa. El tamaño informa al bajar el umbral (con 0,05 el primer
// cuartil cae a 0,561), en la etapa cruda, o cuando la red es mala. Por eso el
// número exacto está SIEMPRE a mano (tooltip por punto, rango por ranura en la
// tabla): un círculo que no se distingue del de al lado no puede ser la única
// vía al dato.
//
// El radio va en unidades de la VENTANA, como los knobs de inferencia: así el
// punto pesa lo mismo respecto de lo que la red mira, sea cual sea el tamaño de
// la imagen. El suelo existe para que un 0,05 siga siendo visible y apuntable.
export const radioDe = (n: number) => (score: number) =>
  n * 0.045 + n * 0.12 * Math.min(1, Math.max(0, score));

export const colorDe = (c: string) => CORNER_CSS[c as keyof typeof CORNER_CSS];

/** Los puntos, para meter DENTRO de un <svg> cuyo viewBox son las coordenadas
 *  de la imagen (el mismo trato que las cajas de `BoxedImage`). */
export function CornerDots(props: {
  corners?: Deteccion[];
  /** la nube pre-NMS: misma escala al 55 % y desvaída, para que las dos etapas
      encendidas a la vez se distingan por peso y no sólo por posición */
  raw?: Deteccion[];
  /** el ancho de la ventana etiquetada, que es la unidad del radio */
  windowSize: number;
  /** ranuras que NO se dibujan */
  hidden?: string[];
  /** el tamaño de la imagen, para que una etiqueta no se salga del viewBox */
  width: number;
  height: number;
  /** con `false` no se dibujan las etiquetas de texto (miniaturas) */
  labels?: boolean;
}) {
  const { windowSize: n, width: W, height: H } = props;
  const oculta = (c: string) => (props.hidden ?? []).includes(c);
  const radio = radioDe(n || 16);
  const conLetra = props.labels ?? true;
  return (
    <>
      {(props.raw ?? []).filter((d) => !oculta(d.corner)).map((d, i) => (
        <circle key={`r${i}`} cx={d.x} cy={d.y} r={radio(d.score) * 0.55}
          fill="none" stroke={colorDe(d.corner)} strokeWidth={0.3} opacity={0.5}
          pointerEvents="all">
          <title>{`cruda ${d.corner} · score ${d.score.toFixed(3)} · (${d.x}, ${d.y})`}</title>
        </circle>
      ))}
      {(props.corners ?? []).filter((d) => !oculta(d.corner)).map((d, i) => {
        const r = radio(d.score);
        // El <svg> RECORTA, así que una etiqueta puesta a ciegas al lado del
        // círculo se pierde en cuanto la esquina está pegada al borde — y las
        // esquinas pegadas al borde son justo las que este proyecto mira
        // (`edge_inputs`). Se mide antes de colocar: si no cabe a la derecha,
        // va a la izquierda; y la altura se acota a la caja. El CÍRCULO no se
        // mueve: eso sería mover el dato.
        const izq = d.x + r + 5.8 > W;
        const lx = izq ? d.x - r - 0.8 : d.x + r + 0.8;
        const ly = Math.min(H - 0.6, Math.max(3.4, d.y + 1));
        return (
          <g key={`c${i}`} pointerEvents="all">
            <title>{`${d.corner} · score ${d.score.toFixed(3)} · (${d.x}, ${d.y})`}</title>
            {/* anillo en color de superficie: dos esquinas que caen casi encima
                se siguen leyendo como dos */}
            <circle cx={d.x} cy={d.y} r={r} fill="none" stroke="var(--surface)"
              strokeWidth={1.1} />
            <circle cx={d.x} cy={d.y} r={r} fill="none"
              stroke={colorDe(d.corner)} strokeWidth={0.6} />
            {conLetra ? (
              // la etiqueta va en tinta de TEXTO, nunca en el color del dato
              // (U3.9); el color lo lleva el círculo de al lado. El halo de
              // superficie la hace legible sobre la imagen
              <text x={lx} y={ly} fontSize={4} textAnchor={izq ? "end" : "start"}
                fill="var(--text)" stroke="var(--surface)" strokeWidth={0.7}
                paintOrder="stroke">{d.corner}</text>
            ) : null}
          </g>
        );
      })}
    </>
  );
}

/** Leyenda Y filtro a la vez, a propósito: la identidad de una esquina no puede
 *  ser sólo el color (U3.8), y la casilla que la enciende es el sitio donde ya
 *  está su nombre. El vocabulario lo sirve el payload (U4.2), nunca una
 *  constante del front. */
export function CornerFilter(props: {
  order: Corner[];
  hidden: string[];
  onToggle: (c: Corner) => void;
  onAll: () => void;
  /** cuántas detecciones tiene cada ranura, para que apagar una diga qué cuesta */
  counts?: Record<string, number>;
}) {
  // ⚠ Sin `data-testid` a proposito: es contrato con el verificador (U7.11) y su
  // extractor casa un LITERAL, asi que pasarlo por prop lo esconde. Ademas marca
  // un sitio de la PANTALLA, no de este componente: lo pone quien lo usa.
  if (!props.order.length) return null;
  return (
    <div className="curvelegend" style={{ alignItems: "center" }}>
      <span className="sub">esquinas:</span>
      {props.order.map((c) => {
        const off = props.hidden.includes(c);
        return (
          <label key={c} className={"legenditem" + (off ? " off" : "")}>
            <input type="checkbox" checked={!off} onChange={() => props.onToggle(c)} />
            <span className="swatch"
              style={{ background: off ? "var(--border)" : colorDe(c) }} />
            <span className="mono">{c}</span>
            {props.counts ? <span className="sub">({props.counts[c] ?? 0})</span> : null}
          </label>
        );
      })}
      {props.hidden.length ? (
        <button onClick={props.onAll}>todas</button>
      ) : null}
    </div>
  );
}

/** El aviso que impide el diagnóstico equivocado: una imagen vacía POR EL FILTRO
 *  se lee exactamente igual que una red que no detecta nada. */
export function CornerFilterAviso(props: { hidden: string[]; tapadas: number }) {
  if (!props.hidden.length) return null;
  return (
    <p className="sub" style={{ marginTop: 4 }}>
      filtro activo: {props.hidden.join(", ")} sin dibujar
      {props.tapadas > 0 ? ` · ${props.tapadas} detección(es) ocultas` : ""}
    </p>
  );
}
