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

const list = [];
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

  if (codeLines.length < 10 || outLines.length < 3) {
    list.push({ key, title: item.title, lineNum, codeLines: codeLines.length, outLines: outLines.length });
  }
}

console.log("Total short/simple examples:", list.length);
list.sort((a, b) => a.lineNum - b.lineNum);
for (const item of list) {
  console.log(`Line ${item.lineNum.toString().padStart(5)} [${item.codeLines.toString().padStart(2)} code, ${item.outLines.toString().padStart(2)} out]: ${item.key}`);
}
