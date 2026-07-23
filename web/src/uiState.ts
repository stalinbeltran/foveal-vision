// Remembered UI defaults. Two layers, by the user's choice:
//  - localStorage: instant per-browser recall on every screen (usePersistedState).
//  - a committable JSON on the backend (snapshot/restore) so a working session
//    travels to the GPU server, written only when the user asks.
// This is NOT a source of truth for any domain object — just conveniences that
// pre-fill filters and forms. Screens still fetch the real B/C/D/E/H from the API.

import { useEffect, useState } from "react";
import { api } from "./api";

const NS = "fv.ui.";

function load<T>(key: string, initial: T): T {
  try {
    const v = localStorage.getItem(NS + key);
    return v == null ? initial : (JSON.parse(v) as T);
  } catch {
    return initial;
  }
}

// useState that mirrors to localStorage under a namespaced key. The key is the
// slice name (e.g. "runs.filters"); the initial value is the default the screen
// wants when nothing was ever remembered.
export function usePersistedState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => load(key, initial));
  useEffect(() => {
    try {
      localStorage.setItem(NS + key, JSON.stringify(value));
    } catch {
      /* storage full or disabled: recall is a convenience, never fatal */
    }
  }, [key, value]);
  return [value, setValue] as const;
}

// Every remembered slice, as one object — what "Guardar sesión" persists.
export function snapshot(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (k && k.startsWith(NS)) {
      try {
        out[k.slice(NS.length)] = JSON.parse(localStorage.getItem(k)!);
      } catch {
        /* skip a corrupt slice */
      }
    }
  }
  return out;
}

// Replace all remembered slices with a loaded snapshot. The caller reloads the
// page so every hook re-initialises from the new localStorage.
export function restore(obj: Record<string, unknown>): void {
  const toDelete: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (k && k.startsWith(NS)) toDelete.push(k);
  }
  toDelete.forEach((k) => localStorage.removeItem(k));
  Object.entries(obj).forEach(([k, v]) =>
    localStorage.setItem(NS + k, JSON.stringify(v)));
}

export async function saveSession(): Promise<void> {
  await api.put("/ui-state", snapshot());
}

// Load the committable JSON into localStorage. Returns false if there was
// nothing saved yet (so the caller can tell the user, not reload into nothing).
export async function loadSession(): Promise<boolean> {
  const data = await api.get("/ui-state");
  if (!data || Object.keys(data).length === 0) return false;
  restore(data);
  return true;
}
