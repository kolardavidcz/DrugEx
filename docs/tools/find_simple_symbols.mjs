import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
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

const lines = html.split('\n');

const simpleSymbols = [];
for (const [key, item] of Object.entries(db)) {
  let lineNum = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(`"${key}": {`) || lines[i].includes(`'${key}': {`)) {
      lineNum = i + 1;
      break;
    }
  }

  const code = item.example?.code || '';
  const output = item.example?.output || '';
  const codeLines = code.trim().split('\n').filter(l => l.trim().length > 0 && !l.trim().startsWith('#'));
  const outLines = output.trim().split('\n').filter(l => l.trim().length > 0);

  // Criteria for simple / minimal:
  // 1. Very few code lines (<= 6)
  // 2. Trivial print statement (e.g. just "Initialized ...", "... ready", "Parameters: ...")
  // 3. Output is 1-2 trivial lines
  const isTrivial = 
    codeLines.length <= 6 ||
    outLines.length <= 1 ||
    code.includes('initialized.') ||
    code.includes('initialized with') ||
    code.includes('instantiated successfully') ||
    code.includes('configured for') ||
    code.includes('ready for') ||
    code.includes('ready.') ||
    code.includes('finished.') ||
    code.includes('Parameters:') ||
    code.includes('Transformer parameters:') ||
    code.includes('Graph transformer initialized') ||
    code.includes('Calculated smooth score:') ||
    code.includes('Clipped score for') ||
    code.includes('Absolute diff score:');

  if (isTrivial) {
    simpleSymbols.push({
      key,
      title: item.title,
      lineNum,
      codeLines: codeLines.length,
      outLines: outLines.length,
      code,
      output
    });
  }
}

simpleSymbols.sort((a, b) => a.lineNum - b.lineNum);
console.log(`Found ${simpleSymbols.length} simple / minimal symbols across DATABASE:`);
for (const s of simpleSymbols) {
  console.log(`Line ${s.lineNum} [${s.codeLines} code lines, ${s.outLines} out lines] ${s.key}`);
}
