import React, { useEffect, useState } from "react";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field, Working } from "../components/ui";

// X — el log de errores: lo que se rompió cuando nadie estaba mirando.
//
// DISEÑADA PARA QUE HAYA MUCHOS, que es el encargo literal. Con un log grande la
// pregunta no es «enséñamelos» —nadie lee 4.000 líneas— sino «¿de qué hay?», y
// por eso el filtro NO es una caja de búsqueda a secas:
//
//  · las FACETAS (cuántos por nivel, código, origen y versión) las cuenta el
//    servidor y se pintan como botones con su número. Filtrar deja de ser
//    adivinar un valor a ciegas y pasa a ser elegir de una lista que ya dice
//    cuánto hay detrás de cada opción.
//  · por defecto se ven sólo los `error`. Los `rechazo` son la puerta
//    funcionando —hay 109 códigos de ésos— y mezclarlos haría un log que nadie
//    lee. Están a un clic, con su cuenta a la vista para que no parezcan
//    escondidos.
//  · el filtrado y la paginación van en el SERVIDOR (U4.3): el navegador nunca
//    recibe el fichero entero.
//
// Y la fila colapsada enseña lo que sirve para decidir si abrir: cuándo, qué,
// dónde y con qué VERSIÓN del código — que es lo que sitúa un error de hace
// tres días (la avería del proceso más viejo que el artefacto, 2026-09-01).
export default function Errores() {
  const [f, setF] = usePersistedState("errores.filtros", {
    nivel: "error" as string, code: "", origen: "", q: "", desde: "", hasta: "",
  });
  const [pagina, setPagina] = useState(0);
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [abierta, setAbierta] = useState<string | null>(null);
  const LIMIT = 50;

  const cargar = () => {
    setBusy(true);
    const p = new URLSearchParams({ limit: String(LIMIT), offset: String(pagina * LIMIT) });
    (["nivel", "code", "origen", "q", "desde", "hasta"] as const)
      .forEach((k) => { if (f[k]) p.set(k, f[k]); });
    api.get(`/errores?${p}`).then((d) => { setData(d); setError(null); })
      .catch(setError).finally(() => setBusy(false));
  };
  // en vivo: un error que ocurre con la pantalla abierta tiene que aparecer
  useEffect(() => { cargar(); const t = setInterval(cargar, 5000); return () => clearInterval(t); },
    [f, pagina]);
  useEffect(() => { setPagina(0); }, [f]);

  const poner = (k: string, v: string) => setF({ ...f, [k]: f[k as keyof typeof f] === v ? "" : v });
  const facetas = data?.facetas ?? {};
  const paginas = Math.ceil((data?.total ?? 0) / LIMIT);

  return (
    <div>
      <h2 data-domain="X" data-view="FG4" data-fixes="la maquina"
        data-varies="el filtro" data-measures="que se rompio y cuando">Errores</h2>
      <p className="sub">Lo que falló aunque nadie estuviera mirando. Se guarda en el repo de
        datos, un fichero por mes, y sólo se añade.{" "}
        {data ? <span className="mono">{data.donde}</span> : null}</p>
      <ErrorBox error={error} />

      <div className="card">
        <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
          <div style={{ width: 260 }}>
            <Field label="buscar" help="en código, mensaje, ruta o versión">
              <input value={f.q} onChange={(e) => setF({ ...f, q: e.target.value })}
                placeholder="p. ej. checkpoint" /></Field></div>
          <div style={{ width: 150 }}><Field label="desde (UTC)">
            <input type="date" value={f.desde}
              onChange={(e) => setF({ ...f, desde: e.target.value })} /></Field></div>
          <div style={{ width: 150 }}><Field label="hasta (UTC)">
            <input type="date" value={f.hasta}
              onChange={(e) => setF({ ...f, hasta: e.target.value })} /></Field></div>
          <Working on={busy} label="buscando…" />
        </div>

        {/* Las facetas. Cada botón lleva SU CUENTA: es lo que convierte el filtro
            en una lista de lo que hay en vez de un campo que hay que acertar. */}
        {(["nivel", "origen", "code", "version"] as const).map((campo) => {
          const vals = Object.entries(facetas[campo] ?? {});
          if (!vals.length) return null;
          const activo = campo === "version" ? "" : (f as any)[campo];
          return (
            <div key={campo} className="curvelegend" style={{ marginBottom: 4 }}>
              <span className="sub" style={{ minWidth: 62 }}>{campo}:</span>
              {vals.slice(0, 12).map(([v, n]) => (
                <button key={v} className={activo === v ? "" : "secondary"}
                  style={{ padding: "2px 8px", fontSize: 12 }}
                  disabled={campo === "version"}
                  onClick={() => campo !== "version" && poner(campo, v)}>
                  {v} <span className="sub">({n as number})</span>
                </button>
              ))}
              {vals.length > 12 ? <span className="sub">+{vals.length - 12} más</span> : null}
            </div>
          );
        })}

        {data ? (
          <p className="sub" style={{ marginTop: 6 }} data-testid="errores-cuenta">
            {data.total} de {data.total_sin_filtro} · meses con log: {data.meses.join(", ") || "—"}
            {/* ⚠ Un filtro que esconde tiene que decir cuánto esconde: si no,
                "no hay errores" y "los filtré todos" se leen igual. */}
            {data.total < data.total_sin_filtro
              ? ` · el filtro oculta ${data.total_sin_filtro - data.total}`
              : ""}
            {f.nivel !== "rechazo" && (facetas.nivel?.rechazo ?? 0) === 0 && f.nivel === "error"
              ? " · (los rechazos —la puerta funcionando— se ven quitando el filtro de nivel)"
              : ""}
          </p>
        ) : null}
      </div>

      <div className="card" data-testid="errores-tabla">
        {!data ? <Working on /> : data.errores.length === 0 ? (
          <p className="sub" style={{ margin: 0 }}>
            {data.total_sin_filtro === 0
              ? "No hay ningún error registrado. (El log empieza el día que se instaló: no dice nada de antes.)"
              : "Ninguno casa con este filtro."}
          </p>
        ) : (
          <table className="data">
            <thead><tr>
              <th>cuándo (UTC)</th><th>nivel</th><th>código</th><th>dónde</th>
              <th>versión</th><th></th>
            </tr></thead>
            <tbody>
              {data.errores.map((e: any, i: number) => {
                const id = `${e.cuando}#${i}`;
                return (
                  <React.Fragment key={id}>
                    <tr onClick={() => setAbierta(abierta === id ? null : id)}
                      style={{ cursor: "pointer" }}>
                      <td className="mono">{e.cuando.replace("T", " ").replace("+00:00", "")}</td>
                      <td><span className={`badge ${e.nivel === "error" ? "error" : ""}`}>
                        {e.nivel}</span></td>
                      <td className="mono">{e.code}
                        {e.repeticiones ? <span className="sub"> ×{e.repeticiones + 1}</span> : null}</td>
                      <td className="mono sub">{e.origen} · {e.donde || "—"}</td>
                      <td className="mono sub">{e.version}</td>
                      <td className="sub">{abierta === id ? "▾" : "▸"}</td>
                    </tr>
                    {abierta === id ? (
                      <tr><td colSpan={6}>
                        <dl className="kv">
                          <dt>mensaje</dt><dd>{e.message}</dd>
                          {/* el hint es la mitad util de una negativa (R4): sin
                              el, el log dice que fallo y no que hacer */}
                          {e.hint ? <><dt>arreglo</dt><dd>{e.hint}</dd></> : null}
                          <dt>máquina · pid</dt><dd className="mono">{e.maquina} · {e.pid}</dd>
                        </dl>
                        {e.traza ? (
                          <pre className="traza">{e.traza}</pre>
                        ) : null}
                      </td></tr>
                    ) : null}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
        {paginas > 1 ? (
          <div className="row" style={{ marginTop: 8, alignItems: "center" }}>
            <button className="secondary" disabled={pagina === 0}
              onClick={() => setPagina(pagina - 1)}>‹ anteriores</button>
            <span className="sub">página {pagina + 1} de {paginas}</span>
            <button className="secondary" disabled={pagina + 1 >= paginas}
              onClick={() => setPagina(pagina + 1)}>siguientes ›</button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
