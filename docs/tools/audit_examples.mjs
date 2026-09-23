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

// Compute needsUpgrade
const needsUpgrade = [];
for (const [key, item] of Object.entries(db)) {
  const code = item.example?.code || '';
  const out = item.example?.output || '';
  const lines = code.trim().split('\n').filter(l => l.trim().length > 0 && !l.trim().startsWith('#'));
  
  const isShort = lines.length < 5;
  const isTrivialCall = lines.length <= 6 && (
    code.includes('instantiated successfully') ||
    code.includes('engine initialized') ||
    code.includes('ready for') ||
    code.includes('batches molecular graph tensors') ||
    code.includes('Abstract Molecule base class') ||
    code.includes('Abstract supplier interface') ||
    code.includes('database loaded successfully') ||
    code.includes('configured with multivariate')
  );
  
  if (isShort || isTrivialCall) {
    needsUpgrade.push({
      key,
      title: item.title,
      lines: lines.length,
      code: item.example.code,
      output: item.example.output
    });
  }
}

if (process.argv.includes('--dump-short')) {
  for (const [k, v] of Object.entries(db)) {
    const lines = v.example?.code?.trim().split('\n') || [];
    if (lines.length <= 5) {
      console.log(`=== [${lines.length} lines] ${k} ===`);
      console.log(v.example.code);
      console.log('--- OUTPUT ---');
      console.log(v.example.output);
      console.log('==============\n');
    }
  }
  process.exit(0);
}

if (process.argv.includes('--check-rest')) {
  const rest = [];
  const needsKeys = new Set(needsUpgrade.map(x => x.key));
  for (const [k, v] of Object.entries(db)) {
    if (!needsKeys.has(k)) {
      const code = v.example?.code || '';
      const out = v.example?.output || '';
      if (code.includes('TODO') || code.includes('placeholder') || out.includes('TODO')) {
        rest.push({ k, reason: 'has TODO' });
      } else if (code.split('\n').length < 6) {
        rest.push({ k, reason: 'lines < 6' });
      }
    }
  }
  console.log(`Remaining items needing attention: ${rest.length}`);
  for (const r of rest) console.log(`- ${r.k} (${r.reason})`);
  process.exit(0);
}

if (process.argv.includes('--lines')) {
  const lines = html.split('\n');
  for (const item of needsUpgrade) {
    const keyStr = `"${item.key}": {`;
    const altKeyStr = `'${item.key}': {`;
    let foundLine = -1;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes(keyStr) || lines[i].includes(altKeyStr)) {
        foundLine = i + 1;
        break;
      }
    }
    console.log(`${foundLine}\t${item.key}`);
  }
  process.exit(0);
}

console.log(`Audited ${Object.keys(db).length} symbols.`);
console.log(`Found ${needsUpgrade.length} examples requiring realistic multi-step upgrade:\n`);

const byCat = {};
for (const item of needsUpgrade) {
  const cat = db[item.key].category || 'other';
  byCat[cat] = byCat[cat] || [];
  byCat[cat].push(item);
}

for (const [cat, items] of Object.entries(byCat)) {
  console.log(`\n=================== CATEGORY: ${cat} (${items.length} items) ===================`);
  for (const item of items) {
    console.log(`- [${item.lines} lines] ${item.key} (${item.title})`);
  }
}
