import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { usePersistedState } from "../uiState";
import { ErrorBox, Field } from "../components/ui";

// B x C x D + X -> E. Names only (R7); device aside (X). Contract (1) becomes
// visible here: on picking B and C they either match or they don't — and the
// backend is the authority (400 with reason), the screen just surfaces it.
export default function Train() {
  const [wds, setWds] = useState<any[]>([]);
  const [nets, setNets] = useState<any[]>([]);
  const [recipes, setRecipes] = useState<any[]>([]);
  const [runs, setRuns] = useState<any[]>([]);
  const [form, setForm] = usePersistedState("train.form", {
    name: "", window_dataset: "", network: "", recipe: "", device: "cpu" });
  const [error, setError] = useState<unknown>(null);
  const [launched, setLaunched] = useState<string | null>(null);
  const [compat, setCompat] = useState<string | null>(null);

  useEffect(() => {
    // a remembered choice wins over the "first item" default
    api.get("/window-datasets").then((d) => {
      setWds(d.window_datasets);
      if (d.window_datasets[0]) setForm((f) => f.window_dataset ? f : { ...f, window_dataset: d.window_datasets[0].name });
    }).catch(setError);
    api.get("/networks").then((d) => {
      setNets(d.networks);
      if (d.networks[0]) setForm((f) => f.network ? f : { ...f, network: d.networks[0].name });
    }).catch(setError);
    api.get("/recipes").then((d) => {
      setRecipes(d.recipes);
      if (d.recipes[0]) setForm((f) => f.recipe ? f : { ...f, recipe: d.recipes[0].name });
    }).catch(setError);
    api.get("/runs").then((d) => setRuns(d.runs)).catch(() => {});
  }, []);

  // surface (1) live: fovea of the chosen net vs window of the chosen B
  useEffect(() => {
    const net = nets.find((n) => n.name === form.network);
    const wd = wds.find((w) => w.name === form.window_dataset);
    if (!net || !wd) { setCompat(null); return; }
    api.post("/networks/validate", net).then((v) => {
      if (!v.valid) { setCompat("la red no es válida (mira Redes)"); return; }
      const fovea = v.trace.dims.center_out;
      const win = wd.config.window_size;
      setCompat(fovea === win
        ? `✓ casan: fóvea ${fovea}px == ventana ${win}px`
        : `✗ NO casan: fóvea ${fovea}px vs ventana ${win}px — el backend lo rechazará (contrato ①)`);
    }).catch(() => setCompat(null));
  }, [form.network, form.window_dataset, nets, wds]);

  // honest cost estimate: only from runs with the same B fingerprint AND net
  const estimate = (() => {
    const wd = wds.find((w) => w.name === form.window_dataset);
    if (!wd) return null;
    const comparable = runs.filter((r) =>
      r.window_dataset === form.window_dataset && r.network === form.network &&
      r.seconds_per_epoch);
    if (!comparable.length) return "sin runs comparables: no se inventa un número";
    const s = comparable.reduce((a, r) => a + r.seconds_per_epoch, 0) / comparable.length;
    return `~${s.toFixed(1)} s/época, medido sobre ${comparable.length} run(s) comparables`;
  })();

  const launch = async () => {
    setError(null); setLaunched(null);
    try {
      await api.post("/runs", form);
      setLaunched(form.name);
      setForm((f) => ({ ...f, name: "" }));  // a run name is single-use: don't remember it
    } catch (e) { setError(e); }
  };

  return (
    <div>
      <h2>Entrenar</h2>
      <p className="sub">Tres nombres (B, C, D) y el device aparte — la rigidez es a propósito:
        es lo que hace que la procedencia se sostenga sola.</p>
      <ErrorBox error={error} />
      <div className="card" style={{ maxWidth: 420 }}>
        <Field label="nombre del run" help="nuevo: un run no se sobrescribe jamás">
          <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></Field>
        <Field label="dataset de ventanas (B)">
          <select value={form.window_dataset}
            onChange={(e) => setForm({ ...form, window_dataset: e.target.value })}>
            {wds.map((w) => <option key={w.name}>{w.name}</option>)}
          </select></Field>
        <Field label="red (C)">
          <select value={form.network} onChange={(e) => setForm({ ...form, network: e.target.value })}>
            {nets.map((n) => <option key={n.name}>{n.name}</option>)}
          </select></Field>
        <Field label="receta (D)">
          <select value={form.recipe} onChange={(e) => setForm({ ...form, recipe: e.target.value })}>
            {recipes.map((r) => <option key={r.name}>{r.name}</option>)}
          </select></Field>
        <Field label="device (ejecución, X — fuera de la receta)">
          <select value={form.device} onChange={(e) => setForm({ ...form, device: e.target.value })}>
            <option>cpu</option><option>cuda</option>
          </select></Field>
        {compat ? <p data-testid="compat" style={{
          color: compat.startsWith("✓") ? "var(--ok)" : "var(--error)" }}>{compat}</p> : null}
        {estimate ? <p className="sub">Coste estimado: {estimate}</p> : null}
        <button onClick={launch} disabled={!form.name}>Lanzar</button>
        {launched ? <p>Lanzado. Míralo en <Link to={`/runs/${launched}`}>Runs → {launched}</Link>.</p> : null}
      </div>
    </div>
  );
}
