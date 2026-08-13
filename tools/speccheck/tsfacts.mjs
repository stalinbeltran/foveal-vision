/**
 * Emit structural facts about the front, as JSON, for the spec validator.
 *
 * Why a side-car and not a regex: the interesting rules are structural, and this
 * codebase talks about its own vocabularies in comments all the time ("el orden
 * de las esquinas viaja con los datos (corner_order...)"). A grep cannot tell a
 * COMMENT that mentions `corner_order` from a literal that REDEFINES it; the
 * parser can. Facts only -- every judgement is made in Python.
 *
 * Usage: node tools/speccheck/tsfacts.mjs <repo-root> "<glob-ish dir>" > facts.json
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { createRequire } from "node:module";

const root = resolve(process.argv[2] || ".");
const dir = resolve(root, process.argv[3] || "web/src");
const require_ = createRequire(join(root, "web", "package.json"));
const ts = require_("typescript");

function walk(d, out = []) {
  for (const name of readdirSync(d)) {
    const p = join(d, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if (/\.(ts|tsx)$/.test(name)) out.push(p);
  }
  return out;
}

/** Text of the condition of the nearest enclosing `if`, or "" (for call guards). */
function guardOf(node, src) {
  for (let p = node.parent; p; p = p.parent) {
    if (ts.isIfStatement(p)) return p.expression.getText(src);
  }
  return "";
}

/** True when some ancestor is a call to setInterval/setTimeout (a poll). */
function inTimer(node, src) {
  for (let p = node.parent; p; p = p.parent) {
    if (ts.isCallExpression(p)) {
      const callee = p.expression.getText(src);
      if (/^(setInterval|setTimeout|window\.setInterval|window\.setTimeout)$/.test(callee)) return true;
    }
  }
  return false;
}

const out = { root, dir: relative(root, dir).replace(/\\/g, "/"), files: {} };

for (const file of walk(dir)) {
  const rel = relative(root, file).replace(/\\/g, "/");
  const text = readFileSync(file, "utf8");
  const src = ts.createSourceFile(file, text, ts.ScriptTarget.ES2022, true,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);

  const f = {
    imports: [], identifiers: new Set(), jsxElements: new Set(), jsxAttrs: new Set(),
    strings: [], calls: [], numericComparisons: [],
  };

  const line = (n) => src.getLineAndCharacterOfPosition(n.getStart(src)).line + 1;

  const visit = (node) => {
    if (ts.isImportDeclaration(node)) f.imports.push(node.moduleSpecifier.getText(src).slice(1, -1));
    else if (ts.isIdentifier(node)) f.identifiers.add(node.text);
    else if (ts.isJsxOpeningLikeElement(node)) f.jsxElements.add(node.tagName.getText(src));
    else if (ts.isJsxAttribute(node)) f.jsxAttrs.add(node.name.getText(src));
    else if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      f.strings.push({ v: node.text, line: line(node), timer: inTimer(node, src) });
    } else if (ts.isTemplateExpression(node)) {
      // `/runs/${r}/task-score?...` -- the constant halves still name the route
      for (const span of [node.head, ...node.templateSpans.map((s) => s.literal)])
        if (span.text) f.strings.push({ v: span.text, line: line(node), timer: inTimer(node, src) });
    } else if (ts.isCallExpression(node)) {
      f.calls.push({ callee: node.expression.getText(src), line: line(node), guard: guardOf(node, src) });
    } else if (ts.isBinaryExpression(node) &&
               [ts.SyntaxKind.LessThanToken, ts.SyntaxKind.GreaterThanToken,
                ts.SyntaxKind.LessThanEqualsToken, ts.SyntaxKind.GreaterThanEqualsToken]
                 .includes(node.operatorToken.kind)) {
      // A literal number on either side of a comparison is a threshold written
      // in code -- the shape U6.2 cares about.
      const l = node.left.getText(src), r = node.right.getText(src);
      if (/^\d+(\.\d+)?$/.test(r) || /^\d+(\.\d+)?$/.test(l))
        f.numericComparisons.push({ text: node.getText(src).replace(/\s+/g, " "), line: line(node) });
    }
    ts.forEachChild(node, visit);
  };
  visit(src);

  out.files[rel] = {
    imports: f.imports,
    identifiers: [...f.identifiers],
    jsxElements: [...f.jsxElements],
    jsxAttrs: [...f.jsxAttrs],
    strings: f.strings,
    calls: f.calls,
    numericComparisons: f.numericComparisons,
  };
}

process.stdout.write(JSON.stringify(out));
