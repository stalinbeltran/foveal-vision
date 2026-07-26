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
export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <h1>foveal-vision</h1>
        <div className="group">Datos</div>
        <NavLink to="/sources">Fuentes</NavLink>
        <NavLink to="/window-datasets">Ventanas</NavLink>
        <div className="group">Modelo</div>
        <NavLink to="/networks">Redes</NavLink>
        <NavLink to="/recipes">Recetas</NavLink>
        <div className="group">Entrenar</div>
        <NavLink to="/train">Entrenar</NavLink>
        <NavLink to="/sweeps">Recorridos</NavLink>
        <NavLink to="/studies">Estudios</NavLink>
        <NavLink to="/runs">Runs</NavLink>
        <div className="group">Analizar</div>
        <NavLink to="/diagnostics">Diagnóstico</NavLink>
        <NavLink to="/predict">Predecir</NavLink>
        <SessionBar />
      </nav>
      <main className="main">
        <Boundary>
        <Routes>
          <Route path="/" element={<Navigate to="/sources" replace />} />
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
        </Routes>
        </Boundary>
      </main>
    </div>
  );
}
