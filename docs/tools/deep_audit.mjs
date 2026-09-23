import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import zlib from 'node:zlib';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const repoRoot = path.join(projectRoot, 'repo');
const docsPath = path.join(projectRoot, 'docs', 'cppreference', 'index.html');

console.log('[Audit] Running Deep Audit on Sphinx gh-pages and codebase...');

// 1. Parse objects.inv
const objectsInvBuf = execSync('git -C ' + repoRoot + ' show origin/gh-pages:docs/objects.inv');
let headerOffset = 0;
let newlineCount = 0;
while (newlineCount < 4 && headerOffset < objectsInvBuf.length) {
  if (objectsInvBuf[headerOffset] === 0x0a) {
    newlineCount++;
  }
  headerOffset++;
}
const compressed = objectsInvBuf.slice(headerOffset);
const decompressed = zlib.inflateSync(compressed).toString('utf8');
const invLines = decompressed.split('\n').filter(Boolean);

console.log(`Sphinx objects.inv contains ${invLines.length} entries.`);

const invByType = {};
for (const line of invLines) {
  // format: name domain:role priority uri dispname
  const parts = line.split(/\s+/);
  if (parts.length >= 4) {
    const [name, type, prio, uri, ...rest] = parts;
    invByType[type] = invByType[type] || [];
    invByType[type].push({ name, uri, dispname: rest.join(' ') });
  }
}

for (const [t, arr] of Object.entries(invByType)) {
  console.log(`  - ${t}: ${arr.length}`);
}

console.log('\nSphinx Exceptions:');
for (const e of (invByType['py:exception'] || [])) {
  console.log(`  - ${e.name} -> ${e.uri}`);
}

console.log('\nSphinx Documentation Pages (std:doc):');
for (const d of (invByType['std:doc'] || [])) {
  console.log(`  - ${d.name} -> ${d.uri}`);
}

console.log('\nSphinx Modules (py:module):');
for (const m of (invByType['py:module'] || [])) {
  console.log(`  - ${m.name}`);
}

// 2. Load DATABASE
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
const keys = Object.keys(db);

console.log(`\nCurrently registered symbols in docs: ${keys.length}`);

// 3. Inspect py:class and py:function from Sphinx
const sphinxClasses = (invByType['py:class'] || []).filter(c => !c.name.includes('.tests.') && !c.name.endsWith('.tests'));
const sphinxFuncs = (invByType['py:function'] || []).filter(f => !f.name.includes('.tests.'));
const sphinxData = (invByType['py:data'] || []).filter(d => !d.name.includes('.tests.'));
const sphinxAttributes = (invByType['py:attribute'] || []).filter(a => !a.name.includes('.tests.'));

console.log(`\nNon-test Sphinx classes: ${sphinxClasses.length}`);
console.log(`Non-test Sphinx functions: ${sphinxFuncs.length}`);
console.log(`Non-test Sphinx data/constants: ${sphinxData.length}`);
console.log(`Non-test Sphinx attributes: ${sphinxAttributes.length}`);

// Check mapping
const missingSphinxClasses = [];
for (const sc of sphinxClasses) {
  const shortName = sc.name.split('.').pop();
  const found = keys.some(k => k === sc.name || k.endsWith('::' + shortName) || db[k].title === shortName);
  if (!found) {
    missingSphinxClasses.push(sc);
  }
}

const missingSphinxFuncs = [];
for (const sf of sphinxFuncs) {
  const shortName = sf.name.split('.').pop();
  const found = keys.some(k => k === sf.name || k.endsWith('::' + shortName) || db[k].title === shortName);
  if (!found) {
    missingSphinxFuncs.push(sf);
  }
}

console.log(`\nMissing Sphinx Classes: ${missingSphinxClasses.length}`);
for (const m of missingSphinxClasses) console.log(`  [MISSING] ${m.name} -> ${m.uri}`);

console.log(`\nMissing Sphinx Functions: ${missingSphinxFuncs.length}`);
for (const m of missingSphinxFuncs) console.log(`  [MISSING] ${m.name} -> ${m.uri}`);

// 4. Codebase inspection for classes and functions
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
const nestedClasses = [];

for (const f of pyFiles) {
  const content = fs.readFileSync(f, 'utf8');
  const rel = path.relative(path.join(repoRoot, 'drugex'), f).replace(/\\/g, '/');
  let mod = 'drugex.' + rel.replace(/\.py$/, '').replace(/\//g, '.');
  if (mod.endsWith('.__init__')) mod = mod.slice(0, -9);

  const lines = content.split('\n');
  let currentTopClass = '';
  for (const line of lines) {
    const topCls = line.match(/^class\s+([A-Za-z0-9_]+)(?:\s*\((.*?)\))?\s*:/);
    if (topCls) {
      currentTopClass = topCls[1];
      if (!currentTopClass.startsWith('_')) {
        codebaseClasses.push({ mod, name: currentTopClass, full: `${mod}.${currentTopClass}`, superclass: topCls[2] || '' });
      }
    }
    const nestCls = line.match(/^\s{4}class\s+([A-Za-z0-9_]+)(?:\s*\((.*?)\))?\s*:/);
    if (nestCls) {
      const nName = nestCls[1];
      if (!nName.startsWith('_')) {
        nestedClasses.push({ mod, parent: currentTopClass, name: nName, full: `${mod}.${currentTopClass}.${nName}`, superclass: nestCls[2] || '' });
      }
    }
  }

  const funcMatches = content.matchAll(/^def\s+([A-Za-z0-9_]+)\s*\(/gm);
  for (const m of funcMatches) {
    const name = m[1];
    if (!name.startsWith('_')) {
      codebaseFuncs.push({ mod, name, full: `${mod}.${name}` });
    }
  }
}

console.log(`\nNested classes found in codebase: ${nestedClasses.length}`);
for (const nc of nestedClasses) {
  console.log(`  - ${nc.full} (parent: ${nc.parent}, super: ${nc.superclass})`);
}

console.log(`\nTotal Codebase Non-Test Classes: ${codebaseClasses.length}`);
console.log(`Total Codebase Non-Test Functions: ${codebaseFuncs.length}`);

console.log(`\n--- Codebase Classes not in Sphinx gh-pages ---`);
const sphinxClassNames = new Set(sphinxClasses.map(c => c.name.split('.').pop()));
const extraCodebaseClasses = codebaseClasses.filter(c => !sphinxClassNames.has(c.name));
console.log(`Extra Codebase Classes: ${extraCodebaseClasses.length}`);
for (const ec of extraCodebaseClasses) {
  console.log(`  - ${ec.full} (extends: ${ec.superclass})`);
}

const missingCodebaseFuncs = [];
for (const cf of codebaseFuncs) {
  const found = keys.some(k => k === cf.full || k.endsWith('::' + cf.name) || db[k].title === cf.name);
  if (!found) {
    missingCodebaseFuncs.push(cf);
  }
}

console.log(`\n--- Codebase Functions (${codebaseFuncs.length}) ---`);
for (const f of codebaseFuncs) {
  const found = keys.some(k => k === f.full || k.endsWith('::' + f.name) || db[k].title === f.name);
  console.log(`  ${found ? '[FOUND]' : '[MISSING]'} ${f.full}`);
}

const missingCodebaseClasses = [];
for (const cc of codebaseClasses) {
  const found = keys.some(k => k === cc.full || k.endsWith('::' + cc.name) || db[k].title === cc.name);
  if (!found) {
    missingCodebaseClasses.push(cc);
  }
}

console.log(`\nMissing Codebase Classes: ${missingCodebaseClasses.length}`);
for (const m of missingCodebaseClasses) console.log(`  [MISSING] ${m.full} (${m.superclass})`);

console.log(`\nMissing Codebase Functions: ${missingCodebaseFuncs.length}`);
for (const m of missingCodebaseFuncs) console.log(`  [MISSING] ${m.full}`);

// 5. Audit quality of existing entries
console.log(`\n--- Quality Audit of Existing ${keys.length} Symbols ---`);
let trivialExamples = 0;
let missingParams = 0;
let missingMembers = 0;
let missingNotes = 0;

for (const k of keys) {
  const item = db[k];
  // Check example code
  if (item.example.code.includes('print("') && item.example.code.split('\n').length <= 2) {
    console.log(`  [WARN] Trivial 1-2 line print example in ${k}`);
    trivialExamples++;
  }
  // Check if class has no members defined
  if (item.kind === 'class' && (!item.members || item.members.length === 0)) {
    console.log(`  [WARN] Class with 0 members: ${k}`);
    missingMembers++;
  }
  // Check if function has no parameters
  if (item.kind === 'function' && (!item.parameters || item.parameters.length === 0)) {
    // If synopsis has args but parameters is empty
    if (item.synopsis.includes('(') && !item.synopsis.includes('()')) {
      console.log(`  [WARN] Function with args in synopsis but 0 parameters: ${k}`);
      missingParams++;
    }
  }
}

console.log(`\nQuality Summary:`);
console.log(`  Trivial examples: ${trivialExamples}`);
console.log(`  Classes with 0 members: ${missingMembers}`);
console.log(`  Functions with missing parameters: ${missingParams}`);
