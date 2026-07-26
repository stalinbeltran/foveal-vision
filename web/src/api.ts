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
