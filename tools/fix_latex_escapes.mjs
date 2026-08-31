/**
 * Deterministic LaTeX escape fixer for JavaScript template strings
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const lecturesDir = path.resolve(__dirname, "../app/js/lectures");

const files = fs.readdirSync(lecturesDir).filter(f => f.endsWith(".js"));

const LATEX_COMMANDS = [
  "forall", "exists", "frac", "text", "times", "tau", "to", "mathbf", "mathbb",
  "mathcal", "sum", "prod", "int", "in", "ge", "le", "dots", "quad", "land",
  "nabla", "theta", "epsilon", "sigma", "tanh", "odot", "mid", "succ", "rho",
  "alpha", "exp", "begin", "end", "cases", "left", "right", "sqrt", "aligned"
];

files.forEach(file => {
  const filePath = path.join(lecturesDir, file);
  let content = fs.readFileSync(filePath, "utf-8");

  // Replace each unescaped \cmd (or formfeed/tab corrupted) with \\cmd
  // In source code text:
  LATEX_COMMANDS.forEach(cmd => {
    // Regex matching single backslash followed by cmd not already preceded by backslash
    const regex = new RegExp(`(?<!\\\\)\\\\${cmd}\\b`, "g");
    content = content.replace(regex, `\\\\${cmd}`);
  });

  fs.writeFileSync(filePath, content, "utf-8");
  console.log(`Processed ${file}`);
});
