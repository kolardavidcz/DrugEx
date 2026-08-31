/**
 * DrugEx Hub — Precision Syntax Highlighting & Chemical Formula Formatter
 */

import { escapeHtml } from "./ui.js";

const PYTHON_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
  "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
  "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
  "return", "try", "while", "with", "yield", "True", "False", "None", "self"
]);

const DRUGEX_BUILTINS = new Set([
  "SequenceRNN", "SequenceTransformer", "GraphTransformer", "DrugExEnvironment",
  "ParetoCrowdingDistance", "SmoothClippedScore", "ClippedScore", "MinMaxScore",
  "MinMaxGaussian", "Gaussian", "Property", "RDKitROCSScorer", "CDPKitROCSScorer",
  "OpenEyeROCSScorer", "RDKitConformerGenerator", "OmegaConformerGenerator",
  "CDPKitConformerGenerator", "SchrodingerConformerGenerator", "VocSmiles",
  "SequenceExplorer", "FragSequenceExplorer", "FragGraphExplorer", "Chem",
  "AllChem", "rdShapeAlign", "CDPLChem", "CDPLShape", "oechem", "oeshape",
  "oeomega", "FileMonitor", "SequenceDataSet", "GraphFragDataSet", "DataLoader",
  "torch", "np", "pd", "plt", "os", "sys", "shutil", "tempfile", "Path"
]);

/**
 * Single-pass lexical tokenizer for Python syntax highlighting.
 * Ensures that newly injected HTML span tags are NEVER matched by subsequent rules.
 */
export function highlightPython(code) {
  if (!code) return "";

  // Token regex matching in strict order of priority
  // Group 1: Comments (#...)
  // Group 2: Strings (triple-quoted and single/double quoted with escapes)
  // Group 3: Decorators (@decorator)
  // Group 4: Numbers (integers, floats, scientific notation)
  // Group 5: Identifiers / Keywords / Function names
  const TOKEN_REGEX = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(@[a-zA-Z_]\w*)|\b(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)\b|\b([a-zA-Z_]\w*)\b/gm;

  let lastIndex = 0;
  let out = "";
  let match;
  let prevWord = "";

  while ((match = TOKEN_REGEX.exec(code)) !== null) {
    // Append any unformatted text (whitespace, symbols like () : , = etc.)
    const plain = code.slice(lastIndex, match.index);
    if (plain) {
      out += escapeHtml(plain);
    }
    lastIndex = TOKEN_REGEX.lastIndex;

    const [fullMatch, comment, str, decorator, number, word] = match;

    if (comment !== undefined) {
      out += `<span class="syn-com">${escapeHtml(comment)}</span>`;
      prevWord = "";
    } else if (str !== undefined) {
      out += `<span class="syn-str">${escapeHtml(str)}</span>`;
      prevWord = "";
    } else if (decorator !== undefined) {
      out += `<span class="syn-dec">${escapeHtml(decorator)}</span>`;
      prevWord = "";
    } else if (number !== undefined) {
      out += `<span class="syn-num">${escapeHtml(number)}</span>`;
      prevWord = "";
    } else if (word !== undefined) {
      if (prevWord === "def") {
        out += `<span class="syn-fn">${escapeHtml(word)}</span>`;
      } else if (PYTHON_KEYWORDS.has(word)) {
        out += `<span class="syn-kw">${escapeHtml(word)}</span>`;
      } else if (DRUGEX_BUILTINS.has(word)) {
        out += `<span class="syn-cls">${escapeHtml(word)}</span>`;
      } else {
        out += escapeHtml(word);
      }
      prevWord = word;
    }
  }

  // Append any trailing plain text
  if (lastIndex < code.length) {
    out += escapeHtml(code.slice(lastIndex));
  }

  return out;
}

export function highlightCode(code, lang = "python") {
  if (lang === "python" || lang === "py" || lang === "bash" || lang === "sh") {
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
  return `<span class="math-tex">${escapeHtml(tex)}</span>`;
}
