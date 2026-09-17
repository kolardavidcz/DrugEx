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

/**
 * Check if a terminal output block contains in-between runtime states or pedagogical insights.
 */
export function hasTerminalTrace(output) {
  if (!output) return false;
  return /(?:^[~💡]|\[(?:STAV|STOPA|STATE|TRACE|POZNATEK|VYSVĚTLENÍ|POZNÁMKA|INSIGHT|TELEMETRIE|TELEMETRY)\])/m.test(output);
}

/**
 * Format terminal output with rich in-between runtime traces and pedagogical insights.
 * - Lines starting with '~' or '[STAV]' / '[STOPA]' are styled as in-between execution states (electric violet).
 * - Lines starting with '💡' or '[POZNATEK]' / '[VYSVĚTLENÍ]' are styled as teacher insights (warm amber).
 * - Lines starting with '[Epoch ...]', '[Epocha ...]', '[Stage ...]' are highlighted as milestones (cyan).
 * - Lines starting with '✓' or progress bars '100%|' are highlighted as success (emerald).
 */
export function formatTerminalOutput(output) {
  if (!output) return "";
  const lines = output.split("\n");
  const out = lines.map(line => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("~")) {
      const content = trimmed.replace(/^~\s*/, "");
      const badge = trimmed.includes("[STOPA") || trimmed.includes("[TRACE") ? "STOPA" : "STAV";
      return `<span class="output-trace"><span class="output-trace-badge">${badge}</span><span class="output-trace-text">${escapeHtml(content)}</span></span>`;
    }
    if (trimmed.startsWith("💡")) {
      const content = trimmed.replace(/^💡\s*/, "");
      const badge = trimmed.includes("[VYSVĚTLENÍ") ? "VYSVĚTLENÍ" : (trimmed.includes("[POZNÁMKA") ? "POZNÁMKA" : "POZNATEK");
      return `<span class="output-insight"><span class="output-insight-badge">${badge}</span><span class="output-insight-text">${escapeHtml(content)}</span></span>`;
    }
    if (trimmed.startsWith("[STAV]") || trimmed.startsWith("[STOPA]") || trimmed.startsWith("[STATE]") || trimmed.startsWith("[TRACE]")) {
      const content = trimmed.replace(/^\[(?:STAV|STOPA|STATE|TRACE)\]\s*/, "");
      const badge = trimmed.startsWith("[STOPA]") || trimmed.startsWith("[TRACE]") ? "STOPA" : "STAV";
      return `<span class="output-trace"><span class="output-trace-badge">${badge}</span><span class="output-trace-text">${escapeHtml(content)}</span></span>`;
    }
    if (trimmed.startsWith("[POZNATEK]") || trimmed.startsWith("[VYSVĚTLENÍ]") || trimmed.startsWith("[POZNÁMKA]") || trimmed.startsWith("[INSIGHT]") || trimmed.startsWith("[NOTE]")) {
      const content = trimmed.replace(/^\[(?:POZNATEK|VYSVĚTLENÍ|POZNÁMKA|INSIGHT|NOTE)\]\s*/, "");
      const badge = trimmed.startsWith("[VYSVĚTLENÍ]") ? "VYSVĚTLENÍ" : (trimmed.startsWith("[POZNÁMKA]") || trimmed.startsWith("[NOTE]") ? "POZNÁMKA" : "POZNATEK");
      return `<span class="output-insight"><span class="output-insight-badge">${badge}</span><span class="output-insight-text">${escapeHtml(content)}</span></span>`;
    }
    if (/^\[(?:Epoch|Epocha|Stage|Fáze|Phase|Worker)\s+[^\]]+\]/.test(trimmed)) {
      return `<span class="output-milestone">${escapeHtml(line)}</span>`;
    }
    if (trimmed.startsWith("✓") || trimmed.includes("100%|")) {
      return `<span class="output-success">${escapeHtml(line)}</span>`;
    }
    return escapeHtml(line);
  });
  return out.join("\n");
}
