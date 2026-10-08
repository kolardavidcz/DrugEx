import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(projectRoot, '..', 'repo', 'drugex');
const docsPath = path.join(projectRoot, 'index.html');

const html = fs.readFileSync(docsPath, 'utf8');
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
const scriptCode = scriptMatch[1];
const dbStart = scriptCode.indexOf('const DATABASE = {');
const appStart = scriptCode.indexOf('function App()');
const dbSnippet = scriptCode.slice(dbStart, appStart).trim();
const context = {};
vm.createContext(context);
const evalCode = dbSnippet.replace(/^const DATABASE =/, 'globalThis.DATABASE =');
vm.runInContext(evalCode, context);
const db = context.DATABASE;

console.log(`Auditing ${Object.keys(db).length} symbols against repo/drugex...`);

const coreSymbols = Object.entries(db).filter(([k, v]) => v.category === 'core');
console.log(`Core symbols count: ${coreSymbols.length}`);

// We will check each core symbol:
// 1. Does the module file exist in repo/drugex?
// 2. Does the class or function exist in that file?
// 3. For classes: does __init__ exist and what are its parameters?
// 4. For functions: what are its parameters?

const mismatches = [];

for (const [key, item] of coreSymbols) {
  // Map module to file path
  // key format e.g. drugex::training::scorers::qsprpred::QSPRPredScorer or drugex::training::generators::SequenceRNN
  const parts = key.split('::');
  const modParts = item.module ? item.module.split('.') : parts.slice(0, -1);
  const symbolName = item.title;
  
  // modParts e.g. ['drugex', 'training', 'scorers', 'qsprpred']
  // file could be repo/drugex/training/scorers/qsprpred.py or repo/drugex/training/scorers/qsprpred/__init__.py
  let relPath = modParts.slice(1).join('/'); // drop 'drugex'
  let targetFile = path.join(repoRoot, relPath + '.py');
  if (!fs.existsSync(targetFile)) {
    targetFile = path.join(repoRoot, relPath, '__init__.py');
  }
  
  if (!fs.existsSync(targetFile)) {
    // Try finding by symbolName across repo
    mismatches.push({
      key,
      title: symbolName,
      module: item.module,
      issue: `File not found: ${targetFile}`
    });
    continue;
  }
  
  const pyCode = fs.readFileSync(targetFile, 'utf8');
  // Check if symbol is defined or imported
  const classDefRegex = new RegExp(`class\\s+${symbolName}\\b(?:\\(([^)]*)\\))?:`, 'm');
  const funcDefRegex = new RegExp(`def\\s+${symbolName}\\b\\s*\\(([^)]*)\\)`, 'm');
  const importRegex = new RegExp(`(?:from\\s+[\\w\\.]+\\s+import\\s+.*\\b${symbolName}\\b|import\\s+.*\\b${symbolName}\\b)`, 'm');
  
  const isClass = classDefRegex.test(pyCode);
  const isFunc = funcDefRegex.test(pyCode);
  const isImport = importRegex.test(pyCode);
  
  if (!isClass && !isFunc && !isImport) {
    mismatches.push({
      key,
      title: symbolName,
      module: item.module,
      targetFile,
      issue: `Symbol ${symbolName} not found as class, func, or import in ${path.relative(repoRoot, targetFile)}`
    });
  }
}

console.log(`\nFound ${mismatches.length} file/definition mismatches:`);
for (const m of mismatches) {
  console.log(`- [${m.key}] (${m.module}): ${m.issue}`);
}
