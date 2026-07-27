/**
 * Validate THIS project's palette: read tokens.css, feed the categorical ramp to
 * the computable checks, for both surfaces.
 *
 * One implementation, two entry points (ui/4-datos.md U4.2):
 *   - `npm run validate:palette`      -- for whoever is editing tokens.css
 *   - `scripts\verify_spec.py`        -- spawns this with --json (rules U3.8/U3.13)
 *
 * The project-specific part is only WHICH tokens play which role, and that is a
 * naming convention the spec already fixes: `--series-N` is the categorical ramp,
 * `--bg` is the chart surface, `--text` is the ink. The checks themselves and
 * their thresholds are the design-system-agnostic ones and are NOT restated here.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { validate, contrast } from "./validate_palette.js";

// The report carries booleans or band words; this is the same mapping the ported
// validator prints with, imported by value so there is no second table of glyphs.
const GLYPH = { true: "PASS", false: "FAIL", pass: "PASS", floor: "WARN", fail: "FAIL", relief: "WARN" };

const HERE = dirname(fileURLToPath(import.meta.url));
const TOKENS = resolve(HERE, "..", "src", "theme", "tokens.css");

/** Tokens of one theme block. `light` is `:root{...}` before the dark media query. */
function readThemes(css) {
  const darkAt = css.indexOf("prefers-color-scheme: dark");
  const cut = darkAt === -1 ? css.length : darkAt;
  const grab = (text) => {
    const out = {};
    for (const m of text.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/gi)) out[m[1]] = m[2].trim();
    return out;
  };
  const light = grab(css.slice(0, cut));
  const dark = { ...light, ...grab(css.slice(cut)) }; // dark overrides light
  return { light, dark };
}

function seriesOf(tokens) {
  return Object.keys(tokens)
    .filter((k) => /^--series-\d+$/.test(k))
    .sort((a, b) => Number(a.split("-").pop()) - Number(b.split("-").pop()))
    .map((k) => tokens[k]);
}

const css = readFileSync(TOKENS, "utf8");
const themes = readThemes(css);
const asJson = process.argv.includes("--json");
const out = { file: TOKENS, tokens: themes, modes: {}, ok: true };

for (const mode of ["light", "dark"]) {
  const tokens = themes[mode];
  const palette = seriesOf(tokens);
  const surface = tokens["--bg"];
  const result = validate(palette, { mode, surface });
  // Ink against surface is a separate question from mark-vs-surface, and it is
  // the one U3.13 cares about: text must stay readable in BOTH themes.
  const ink = {
    text: contrast(tokens["--text"], surface),
    "text-dim": contrast(tokens["--text-dim"], surface),
  };
  out.modes[mode] = {
    surface,
    slots: palette.length,
    ok: result.ok,
    report: result.report.map(([check, state, detail]) => ({ check, state: GLYPH[state] ?? String(state), detail })),
    ink,
  };
  out.ok = out.ok && result.ok;

  if (!asJson) {
    console.log(`\nPaleta (${mode}, superficie ${surface}): ${palette.length} series`);
    for (const { check, state, detail } of out.modes[mode].report) {
      console.log(`  [${state.padEnd(4)}] ${check.padEnd(22)} ${detail}`);
    }
    for (const [name, ratio] of Object.entries(ink)) {
      console.log(`  [${(ratio >= 4.5 ? "PASS" : "WARN").padEnd(4)}] ${("tinta " + name).padEnd(22)} ${ratio.toFixed(2)}:1 vs superficie`);
    }
  }
}

if (asJson) console.log(JSON.stringify(out, null, 2));
else console.log(`\n  -> ${out.ok ? "TODO PASA" : "FALLA -- arregla los checks marcados"}\n`);

process.exit(out.ok ? 0 : 1);
