import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const repoRoot = path.join(projectRoot, 'repo');
const docsPath = path.join(projectRoot, 'docs', 'cppreference', 'index.html');

console.log('🔍 Auditing gh-pages and codebase symbols vs docs/cppreference/index.html...');

// 1. Get current DATABASE from index.html
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
const currentDb = context.DATABASE;
const currentKeys = new Set(Object.keys(currentDb));
console.log(`Current DATABASE symbols count: ${currentKeys.size}`);

// 2. Extract Sphinx genindex.html from origin/gh-pages
const genindexHtml = execSync('git -C ' + repoRoot + ' show origin/gh-pages:docs/genindex.html', {
  maxBuffer: 20 * 1024 * 1024,
  encoding: 'utf8'
});

const regex = /<li><a href="api\/([^"]+)#(drugex\.[^"]+)">([^<]+)<\/a>/g;
let match;
const sphinxClasses = new Map();
const sphinxFunctions = new Map();
const sphinxMethods = new Map();

while ((match = regex.exec(genindexHtml)) !== null) {
  const [_, page, fullPath, text] = match;
  if (text.includes('(class in ')) {
    const clsName = text.replace(/ \(class in .*\)/, '').trim();
    sphinxClasses.set(fullPath, { name: clsName, path: fullPath, page, raw: text });
  } else if (text.includes('(in module ')) {
    const fnName = text.replace(/ \(in module .*\)/, '').replace(/\(\)/, '').trim();
    sphinxFunctions.set(fullPath, { name: fnName, path: fullPath, page, raw: text });
  } else if (text.includes(' method)')) {
    sphinxMethods.set(fullPath, { raw: text, page });
  }
}

console.log(`\nSphinx gh-pages:`);
console.log(`  Documented Classes: ${sphinxClasses.size}`);
console.log(`  Documented Module Functions: ${sphinxFunctions.size}`);
console.log(`  Documented Methods: ${sphinxMethods.size}`);

// Let's filter out test classes/functions from Sphinx
const productionClasses = new Map();
for (const [p, item] of sphinxClasses) {
  if (!p.includes('.tests.') && !p.endsWith('.tests') && !item.name.startsWith('Test')) {
    productionClasses.set(p, item);
  }
}
const productionFunctions = new Map();
for (const [p, item] of sphinxFunctions) {
  if (!p.includes('.tests.') && !item.name.startsWith('test_')) {
    productionFunctions.set(p, item);
  }
}

console.log(`\nProduction Classes in Sphinx: ${productionClasses.size}`);
console.log(`Production Functions in Sphinx: ${productionFunctions.size}`);

// Check which production classes / functions are mapped or unmapped in current DATABASE
const unmappedClasses = [];
for (const [fullPath, item] of productionClasses) {
  // Check if symbol name is anywhere in currentKeys
  const found = Array.from(currentKeys).some(k => 
    k.endsWith('::' + item.name) || 
    k.includes('::' + item.name) ||
    currentDb[k].title === item.name ||
    (item.name.includes('.') && k.endsWith('::' + item.name.split('.').pop())) ||
    (item.name.includes('Collector') && currentDb[k].title === 'StandardizationCollector') ||
    (fullPath.includes('sequence_transformer.Block') && currentDb[k].title === 'SequenceTransformerBlock') ||
    (fullPath.includes('graph_transformer.Block') && currentDb[k].title === 'GraphTransformerBlock')
  );
  if (!found) {
    unmappedClasses.push({ fullPath, name: item.name });
  }
}

const unmappedFunctions = [];
for (const [fullPath, item] of productionFunctions) {
  const found = Array.from(currentKeys).some(k => k.endsWith('::' + item.name) || currentDb[k].title === item.name);
  if (!found) {
    unmappedFunctions.push({ fullPath, name: item.name });
  }
}

console.log(`\nUnmapped Sphinx Classes (${unmappedClasses.length}):`);
for (const u of unmappedClasses) {
  console.log(`  - ${u.fullPath} (${u.name})`);
}

console.log(`\nUnmapped Sphinx Functions (${unmappedFunctions.length}):`);
for (const u of unmappedFunctions) {
  console.log(`  - ${u.fullPath} (${u.name})`);
}

// 3. Scan repo/drugex Python files using Node.js
function walkDir(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const full = path.join(dir, file);
    const stat = fs.statSync(full);
    if (stat && stat.isDirectory()) {
      if (file !== 'tests' && file !== '__pycache__') {
        results = results.concat(walkDir(full));
      }
    } else if (file.endsWith('.py') && !file.startsWith('test')) {
      results.push(full);
    }
  }
  return results;
}

const pyFiles = walkDir(path.join(repoRoot, 'drugex'));
const codebaseClasses = [];
const codebaseFuncs = [];

for (const f of pyFiles) {
  const content = fs.readFileSync(f, 'utf8');
  const rel = path.relative(path.join(repoRoot, 'drugex'), f).replace(/\\/g, '/');
  let mod = 'drugex.' + rel.replace(/\.py$/, '').replace(/\//g, '.');
  if (mod.endsWith('.__init__')) mod = mod.slice(0, -9);

  // Top-level classes: class Name( or class Name:
  const classMatches = content.matchAll(/^class\s+([A-Za-z0-9_]+)(?:\s*\((.*?)\))?\s*:/gm);
  for (const m of classMatches) {
    const name = m[1];
    if (!name.startsWith('_')) {
      codebaseClasses.push({ mod, name, full: `${mod}.${name}`, superclass: m[2] || '' });
    }
  }

  // Top-level functions: def name(
  const funcMatches = content.matchAll(/^def\s+([A-Za-z0-9_]+)\s*\(/gm);
  for (const m of funcMatches) {
    const name = m[1];
    if (!name.startsWith('_')) {
      codebaseFuncs.push({ mod, name, full: `${mod}.${name}` });
    }
  }
}

console.log(`\nTotal Codebase Non-Test Classes: ${codebaseClasses.length}`);
console.log(`Total Codebase Non-Test Functions: ${codebaseFuncs.length}`);

const unmappedCodebaseClasses = [];
for (const c of codebaseClasses) {
  const found = Array.from(currentKeys).some(k => k.endsWith('::' + c.name) || currentDb[k].title === c.name);
  if (!found) {
    unmappedCodebaseClasses.push(c);
  }
}

const unmappedCodebaseFuncs = [];
for (const f of codebaseFuncs) {
  const found = Array.from(currentKeys).some(k => k.endsWith('::' + f.name) || currentDb[k].title === f.name);
  if (!found) {
    unmappedCodebaseFuncs.push(f);
  }
}

console.log(`\nUnmapped Codebase Classes (${unmappedCodebaseClasses.length}):`);
for (const u of unmappedCodebaseClasses) {
  console.log(`  - ${u.full} (superclass: ${u.superclass})`);
}

console.log(`\nUnmapped Codebase Functions (${unmappedCodebaseFuncs.length}):`);
for (const u of unmappedCodebaseFuncs) {
  console.log(`  - ${u.full}`);
}

// Group unmapped by module
const unmappedByModule = {};
for (const c of unmappedCodebaseClasses) {
  unmappedByModule[c.mod] = unmappedByModule[c.mod] || { classes: [], funcs: [] };
  unmappedByModule[c.mod].classes.push(c.name);
}
for (const f of unmappedCodebaseFuncs) {
  unmappedByModule[f.mod] = unmappedByModule[f.mod] || { classes: [], funcs: [] };
  unmappedByModule[f.mod].funcs.push(f.name);
}

console.log('\n--- Unmapped Symbols by Module ---');
for (const [mod, data] of Object.entries(unmappedByModule)) {
  console.log(`Module: ${mod}`);
  if (data.classes.length) console.log(`  Classes: ${data.classes.join(', ')}`);
  if (data.funcs.length) console.log(`  Funcs:   ${data.funcs.join(', ')}`);
}


const byCat = { core: [], ecosystem: [], experimental: [] };
for (const [k, v] of Object.entries(currentDb)) {
  if (byCat[v.category]) byCat[v.category].push({ key: k, title: v.title });
}
console.log(`\nCurrent Ecosystem Symbols (${byCat.ecosystem.length}):`);
for (const s of byCat.ecosystem) console.log(`  - ${s.key} (${s.title})`);

console.log(`\nCurrent Experimental Symbols (${byCat.experimental.length}):`);
for (const s of byCat.experimental) console.log(`  - ${s.key} (${s.title})`);



