import React, { useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { loadSession, saveSession } from "./uiState";
import Sources from "./screens/Sources";
import WindowDatasets from "./screens/WindowDatasets";
import WindowDatasetDetail from "./screens/WindowDatasetDetail";
import Networks from "./screens/Networks";
import Recipes from "./screens/Recipes";
import Train from "./screens/Train";
import Sweeps from "./screens/Sweeps";
import Studies from "./screens/Studies";
import Runs from "./screens/Runs";
import RunDetail from "./screens/RunDetail";
import Diagnostics from "./screens/Diagnostics";
import Predict from "./screens/Predict";
import Review from "./screens/Review";
import ReviewDetail from "./screens/ReviewDetail";

// The app-wide session control: filters/forms live in localStorage per browser;
// "Guardar" snapshots them to a committable JSON, "Cargar" pulls it back (and
// reloads so every screen re-reads). This is what carries a working session to
// the GPU server.
function SessionBar() {
  const [msg, setMsg] = useState<string>("");
  const save = async () => {
    setMsg("guardando…");
    try { await saveSession(); setMsg("sesión guardada"); }
    catch { setMsg("error al guardar"); }
  };
  const load = async () => {
    setMsg("cargando…");
    try {
      const had = await loadSession();
      if (!had) { setMsg("no hay sesión guardada"); return; }
      setMsg("cargada — recargando…");
      setTimeout(() => window.location.reload(), 300);
    } catch { setMsg("error al cargar"); }
  };
  return (
    <div className="session" data-testid="session-bar">
      <div className="group">Sesión</div>
      <button className="linkbtn" onClick={save}>Guardar sesión</button>
      <button className="linkbtn" onClick={load}>Cargar sesión</button>
      {msg ? <div className="sessionmsg">{msg}</div> : null}
    </div>
  );
}

// A crash in one screen used to unmount the WHOLE app: a white page, no reason,
// no way out — the silent failure this project refuses everywhere else (api.md
// R4: razón + arreglo). The boundary keeps the nav alive and shows what broke,
// plus the one action that always works (clear the remembered state, which is
// where stale values hide). Keyed by route so navigating away recovers.
type EBProps = { children: React.ReactNode; routeKey: string };
class ErrorBoundary extends React.Component<EBProps, { error: Error | null }> {
  constructor(props: EBProps) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: EBProps) {
    if (prev.routeKey !== this.props.routeKey && this.state.error)
      this.setState({ error: null });
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="card" data-testid="screen-error">
        <h3 style={{ marginTop: 0 }}>Esta pantalla ha fallado</h3>
        <p className="mono">{String(this.state.error.message || this.state.error)}</p>
        <p className="sub">Las demás pantallas siguen funcionando: el fallo es del
          render de esta. Si acaba de aparecer tras una actualización, suele ser un
          valor recordado del navegador que ya no encaja.</p>
        <button className="secondary" onClick={() => {
          Object.keys(localStorage).filter((k) => k.startsWith("fv.ui."))
            .forEach((k) => localStorage.removeItem(k));
          window.location.reload();
        }}>Olvidar preferencias de esta sesión y recargar</button>
      </div>
    );
  }
}

function Boundary({ children }: { children: React.ReactNode }) {
  const loc = useLocation();
  return <ErrorBoundary routeKey={loc.pathname}>{children}</ErrorBoundary>;
}

// One screen, one domain. Groups follow domain dependency, not steps: in
// research you iterate on a point and come back — no numbered pipeline.

// QUÉ PANTALLAS SE OFRECEN. Es una lista DECLARADA y no un `if` repartido por el
// nav, porque encenderlas y apagarlas tiene que ser una línea y verse de un
// vistazo — el 2026-08-31 el dueño pidió dejar sólo Revisar («no se usarán por
// el momento»), y «por el momento» quiere decir que esto se revierte.
//
// ⚠ Las RUTAS siguen montadas: lo que se quita es lo que se OFRECE, no lo que
// existe. Un enlace guardado a /runs/<name> sigue funcionando, y las pantallas
// ocultas no dejan de servir al que sepa su URL. Esto no es una puerta —la
// puerta es el token de `fv.api.web`— es la lista de lo que se usa hoy.
const NAV: { grupo: string; items: [string, string][] }[] = [
  { grupo: "Datos", items: [["/sources", "Fuentes"], ["/window-datasets", "Ventanas"]] },
  { grupo: "Modelo", items: [["/networks", "Redes"], ["/recipes", "Recetas"]] },
  { grupo: "Entrenar", items: [["/train", "Entrenar"], ["/sweeps", "Recorridos"],
                               ["/studies", "Estudios"], ["/runs", "Runs"]] },
  { grupo: "Analizar", items: [["/diagnostics", "Diagnóstico"], ["/predict", "Predecir"],
                               ["/review", "Revisar"]] },
];

// Lo único que se ofrece hoy. Para volver a enseñar una pantalla, añade su ruta
// aquí; para enseñarlas todas, pon `null`.
const VISIBLES: Set<string> | null = new Set(["/review"]);

// A dónde va "/". Tiene que ser una pantalla VISIBLE, o la app abre en algo que
// su propio menú dice que no existe.
const INICIO = "/review";

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <h1>foveal-vision</h1>
        {NAV.map(({ grupo, items }) => {
          const vis = items.filter(([to]) => !VISIBLES || VISIBLES.has(to));
          // un grupo sin pantallas visibles no deja su título huérfano
          if (!vis.length) return null;
          return (
            <React.Fragment key={grupo}>
              <div className="group">{grupo}</div>
              {vis.map(([to, label]) => <NavLink key={to} to={to}>{label}</NavLink>)}
            </React.Fragment>
          );
        })}
        <SessionBar />
      </nav>
      <main className="main">
        <Boundary>
        <Routes>
          <Route path="/" element={<Navigate to={INICIO} replace />} />
          <Route path="/sources" element={<Sources />} />
          <Route path="/window-datasets" element={<WindowDatasets />} />
          <Route path="/window-datasets/:name" element={<WindowDatasetDetail />} />
          <Route path="/networks" element={<Networks />} />
          <Route path="/recipes" element={<Recipes />} />
          <Route path="/train" element={<Train />} />
          <Route path="/sweeps" element={<Sweeps />} />
          <Route path="/studies" element={<Studies />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:name" element={<RunDetail />} />
          <Route path="/diagnostics" element={<Diagnostics />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/review" element={<Review />} />
          <Route path="/review/:dataset/:split/:index" element={<ReviewDetail />} />
        </Routes>
        </Boundary>
      </main>
    </div>
  );
}
