// Thin API client. Every backend error carries {code, message, hint} (R4);
// ApiError keeps them so screens can show the reason AND the fix.

export class ApiError extends Error {
  code: string;
  hint: string;
  status: number;
  constructor(status: number, detail: any) {
    const d = detail?.detail ?? detail ?? {};
    super(d.message || `HTTP ${status}`);
    this.code = d.code || "error";
    this.hint = d.hint || "";
    this.status = status;
  }
}

async function req(path: string, init?: RequestInit): Promise<any> {
  const r = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(r.status, body);
  return body;
}

export const api = {
  get: (p: string) => req(p),
  post: (p: string, body?: any) =>
    req(p, { method: "POST", body: JSON.stringify(body ?? {}) }),
  patch: (p: string, body?: any) =>
    req(p, { method: "PATCH", body: JSON.stringify(body ?? {}) }),
  put: (p: string, body?: any) =>
    req(p, { method: "PUT", body: JSON.stringify(body ?? {}) }),
  del: (p: string) => req(p, { method: "DELETE" }),
};

export async function waitJob(jobId: string, onTick?: (j: any) => void): Promise<any> {
  for (;;) {
    const j = await api.get(`/jobs/${jobId}`);
    onTick?.(j);
    if (["done", "error", "cancelled"].includes(j.status)) return j;
    await new Promise((res) => setTimeout(res, 700));
  }
}

// The run/sweep state vocabulary, ONCE (ui/4-datos.md U4.2, ui/8-lexico.md U8.5).
// It was written out in four screens and the copies had already drifted: one of
// them waited for a state called "failed" — which this system does not have —
// and none of them knew about `interrupted`, so an interrupted run never settled
// and was re-fetched forever. The names are the backend's (status.json);
// ⚠ pendiente: que los sirva el API en vez de declararlos aquí (decisiones.md F16).
export const TERMINAL_STATES = ["done", "error", "cancelled", "interrupted"] as const;
export const ACTIVE_STATES = ["queued", "running"] as const;
export const isTerminal = (s?: string | null) =>
  TERMINAL_STATES.includes(s as (typeof TERMINAL_STATES)[number]);

// The corner ORDER is not a constant of the front: it belongs to the dataset
// (manifest.corner_order) and travels in every payload indexed by it — the
// diagnostics summary and the predict answer. What lives here is only the COLOUR
// each name gets, which is a UI decision. Keeping a local order was a second
// definition of fv.metrics.CORNER_NAMES waiting to disagree with it.
export type Corner = "TL" | "TR" | "BR" | "BL";
export const CORNER_CSS: Record<Corner, string> = {
  TL: "var(--corner-tl)", TR: "var(--corner-tr)",
  BR: "var(--corner-br)", BL: "var(--corner-bl)",
};
