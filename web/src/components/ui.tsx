import React from "react";
import { ApiError } from "../api";

// An error shows the code, the reason and the fix — swallowing the hint throws
// away the useful half (api.md R4).
export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as ApiError;
  return (
    <div className="error-box">
      <span className="code">[{e.code ?? "error"}]</span> {e.message}
      {e.hint ? <div className="hintline">→ {e.hint}</div> : null}
    </div>
  );
}

// A slow response without an acknowledgement reads as a lost click.
export function Working({ on, label }: { on: boolean; label?: string }) {
  return on ? <div className="working">{label ?? "trabajando…"}</div> : null;
}

export function Badge({ status }: { status?: string | null }) {
  return <span className={`badge ${status ?? ""}`}>{status ?? "—"}</span>;
}

export function Field(props: {
  label: string; help?: string; children: React.ReactNode;
}) {
  return (
    <label className="field">
      {props.label}
      {props.help ? <span>{props.help}</span> : null}
      {props.children}
    </label>
  );
}
