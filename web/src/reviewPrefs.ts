// Lo que la rejilla de revisión y el detalle COMPARTEN, en un solo sitio.
//
// Están separados en dos pantallas pero son el mismo ajuste: bajar el umbral en
// la rejilla y encontrárselo otra vez en 0,50 al abrir una imagen se lee como
// «no se guardó». Y dos copias de estos valores por defecto derivarían en
// silencio en cuanto una de las dos pantallas cambiase.

// «Todavía no he elegido run» tiene que poder distinguirse de «he elegido MIRAR
// SIN MODELO», que es un modo legítimo de esta pantalla (revisar a ojo sin que
// las cajas te condicionen). Con "" no habría forma de saber cuál de los dos es,
// y auto-elegir un run pisaría la decisión del usuario en cada cambio de split.
// El centinela no puede ser el nombre de un run.
export const SIN_ELEGIR = "\u0000";
export const CLAVE_RUN = "review.run2";

// Los mandos de INFERENCIA (F): post-hoc, en unidades de la ventana etiquetada.
// No cambian la red — cambian cómo se leen sus scores.
export type Knobs = {
  threshold: number;
  stride: number;
  nms_radius: number;
  min_size: number;
};

// ⚠ `0` significa AUTO, no cero. Los tres últimos los deriva el servidor del
// tamaño de ventana de la red (`stride = n/2`, `nms_radius = n/2`,
// `min_size = 4`; `fv/inference/predict.py`). Escribir aquí esos números sería
// una segunda copia de la regla, y una red con otra fóvea heredaría los de la
// anterior sin que nadie lo notara. Por eso lo que se ENSEÑA es lo que el
// servidor devuelve en `knobs`, no lo que se pidió.
export const KNOBS_DEFECTO: Knobs = {
  threshold: 0.5, stride: 0, nms_radius: 0, min_size: 0,
};
export const CLAVE_KNOBS = "review.knobs";

// El cuerpo que viaja a /review/batch. Un `auto` NO se manda: mandar 0 sería
// pedir stride 0 (que no avanza) o nms_radius 0 (sin supresión), que es otra
// cosa distinta de «decide tú».
export function cuerpoKnobs(k: Knobs): Record<string, number> {
  const body: Record<string, number> = { threshold: k.threshold };
  if (k.stride > 0) body.stride = k.stride;
  if (k.nms_radius > 0) body.nms_radius = k.nms_radius;
  if (k.min_size > 0) body.min_size = k.min_size;
  return body;
}
