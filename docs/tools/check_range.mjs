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

for (const [key, item] of Object.entries(db)) {
  let lineNum = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(`"${key}": {`) || lines[i].includes(`'${key}': {`)) {
      lineNum = i + 1;
      break;
    }
  }
  if (lineNum >= 12300 && lineNum <= 13410) {
    const code = item.example?.code || '';
    const output = item.example?.output || '';
    const codeLines = code.trim().split('\n').filter(l => l.trim().length > 0 && !l.trim().startsWith('#'));
    const outLines = output.trim().split('\n').filter(l => l.trim().length > 0);
    console.log(`Line ${lineNum}: [${codeLines.length} code, ${outLines.length} out] ${key} (${item.title})`);
  }
}
