import fs from 'node:fs';
import { execSync } from 'node:child_process';
import vm from 'node:vm';

const diffOutput = execSync('git diff 16947f0~1 16947f0 -- docs/index.html', {
  cwd: '/home/kolar/build_projects/drugex_edu',
  encoding: 'utf8',
  maxBuffer: 50 * 1024 * 1024
});

const keyMatches = new Set();
const regex = /^\+\s*"([a-zA-Z0-9_:]+)":\s*\{/gm;
let m;
while ((m = regex.exec(diffOutput)) !== null) {
  keyMatches.add(m[1]);
}

console.log(`Keys with definition header touched in diff (+ "key": {): ${keyMatches.size}`);
console.log(Array.from(keyMatches));

// Also check which keys had ANY changes inside their block
// We can parse docs/index.html before and after commit 16947f0!
const htmlBefore = execSync('git show 16947f0~1:docs/index.html', {
  cwd: '/home/kolar/build_projects/drugex_edu',
  encoding: 'utf8',
  maxBuffer: 50 * 1024 * 1024
});
const htmlAfter = execSync('git show 16947f0:docs/index.html', {
  cwd: '/home/kolar/build_projects/drugex_edu',
  encoding: 'utf8',
  maxBuffer: 50 * 1024 * 1024
});

function parseDB(html) {
  const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
  const scriptCode = scriptMatch[1];
  const dbStart = scriptCode.indexOf('const DATABASE = {');
  const appStart = scriptCode.indexOf('function App()');
  const dbCode = scriptCode.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');
  const context = {};
  vm.createContext(context);
  vm.runInContext(dbCode, context);
  return context.DATABASE;
}

const dbBefore = parseDB(htmlBefore);
const dbAfter = parseDB(htmlAfter);

const keysBefore = Object.keys(dbBefore);
const keysAfter = Object.keys(dbAfter);

console.log(`Total symbols before: ${keysBefore.length}, after: ${keysAfter.length}`);

const changedKeys = [];
for (const k of keysAfter) {
  const b = dbBefore[k];
  const a = dbAfter[k];
  if (!b) {
    changedKeys.push({ key: k, type: 'added' });
  } else if (JSON.stringify(b) !== JSON.stringify(a)) {
    // Find what fields changed
    const diffFields = [];
    for (const field of Object.keys(a)) {
      if (JSON.stringify(a[field]) !== JSON.stringify(b[field])) {
        diffFields.push(field);
      }
    }
    changedKeys.push({ key: k, type: 'modified', fields: diffFields });
  }
}

console.log(`\nTotal changed symbols between 16947f0~1 and 16947f0: ${changedKeys.length}`);
changedKeys.forEach(c => {
  console.log(`- ${c.key} [${c.type}]: ${c.fields ? c.fields.join(', ') : ''}`);
});
