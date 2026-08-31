/**
 * DrugEx Hub — Syntax Highlighting & Chemical Formula Formatter
 */

import { escapeHtml } from "./ui.js";

const PYTHON_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
  "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
  "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
  "return", "try", "while", "with", "yield", "True", "False", "None"
]);

const DRUGEX_BUILTINS = new Set([
  "SequenceRNN", "SequenceTransformer", "GraphTransformer", "DrugExEnvironment",
  "ParetoCrowdingDistance", "SmoothClippedScore", "ClippedScore", "MinMaxScore",
  "Property", "RDKitROCSScorer", "CDPKitROCSScorer", "OpenEyeROCSScorer",
  "RDKitConformerGenerator", "OmegaConformerGenerator", "CDPKitConformerGenerator",
  "VocSmiles", "SequenceExplorer", "FragSequenceExplorer", "FragGraphExplorer",
  "Chem", "AllChem", "rdShapeAlign", "CDPLChem", "CDPLShape", "oechem", "oeshape"
]);

export function highlightPython(code) {
  if (!code) return "";
  const lines = code.split("\n");
  
  const highlightedLines = lines.map(line => {
    // Check for comment
    const commentIdx = line.indexOf("#");
    let codePart = commentIdx !== -1 ? line.slice(0, commentIdx) : line;
    const commentPart = commentIdx !== -1 ? line.slice(commentIdx) : "";

    // Highlight strings
    codePart = codePart.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, match => {
      return `<span class="syn-str">${escapeHtml(match)}</span>`;
    });

    // Highlight decorators
    codePart = codePart.replace(/@\w+/g, match => {
      return `<span class="syn-dec">${escapeHtml(match)}</span>`;
    });

    // Highlight numbers
    codePart = codePart.replace(/\b\d+(\.\d+)?\b/g, match => {
      return `<span class="syn-num">${match}</span>`;
    });

    // Highlight words (keywords, builtins, function calls)
    codePart = codePart.replace(/\b([a-zA-Z_]\w*)\b/g, (match) => {
      if (PYTHON_KEYWORDS.has(match)) {
        return `<span class="syn-kw">${match}</span>`;
      }
      if (DRUGEX_BUILTINS.has(match)) {
        return `<span class="syn-cls">${match}</span>`;
      }
      return match;
    });

    // Highlight function defs
    codePart = codePart.replace(/(def\s+)([a-zA-Z_]\w*)/g, (_, kw, fn) => {
      return `${kw}<span class="syn-fn">${fn}</span>`;
    });

    let res = codePart;
    if (commentPart) {
      res += `<span class="syn-com">${escapeHtml(commentPart)}</span>`;
    }
    return res;
  });

  return highlightedLines.join("\n");
}

export function highlightCode(code, lang = "python") {
  if (lang === "python" || lang === "py") {
    return highlightPython(code);
  }
  return escapeHtml(code);
}

export function formatFormula(formula) {
  if (!formula) return "";
  return formula.replace(/([A-Z][a-z]?)(\d+)/g, "$1<sub>$2</sub>");
}

export function formatMath(tex) {
  if (!tex) return "";
  // Clean, accessible KaTeX-style math display
  return `<span class="math-tex">${escapeHtml(tex)}</span>`;
}
