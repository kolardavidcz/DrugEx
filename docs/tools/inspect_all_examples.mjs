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
const results = [];

for (const [key, item] of Object.entries(db)) {
  const code = item.example?.code || '';
  const output = item.example?.output || '';
  const codeLines = code.trim().split('\n').filter(l => l.trim().length > 0 && !l.trim().startsWith('#'));
  const totalCodeLines = code.split('\n').length;
  const outLines = output.trim().split('\n').length;

  // find line number in docs/index.html
  let lineNum = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(`"${key}": {`) || lines[i].includes(`'${key}': {`)) {
      lineNum = i + 1;
      break;
    }
  }

  results.push({
    key,
    title: item.title,
    category: item.category,
    lineNum,
    codeLines: codeLines.length,
    totalCodeLines,
    outLines,
    codeSnippet: code.slice(0, 100).replace(/\n/g, ' '),
    outputSnippet: output.slice(0, 80).replace(/\n/g, ' ')
  });
}

console.log('\n=== ALL SYMBOLS WITH lineNum >= 11500 ===');
const lateSymbols = results.filter(r => r.lineNum >= 11500);
for (const r of lateSymbols) {
  console.log(`\nLine ${r.lineNum}: ${r.key} (${r.title})`);
  console.log(`Code lines: ${r.codeLines}, total: ${r.totalCodeLines}`);
  console.log('--- CODE ---');
  console.log(db[r.key].example?.code);
  console.log('--- OUTPUT ---');
  console.log(db[r.key].example?.output);
}

