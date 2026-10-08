import fs from 'node:fs';
import { execSync } from 'node:child_process';

// Read index.html from commit 16947f0
const repoCwd = '/home/kolar/build_projects/drugex_edu';
const newHtml = execSync('git show 16947f0:docs/index.html', { cwd: repoCwd, maxBuffer: 30 * 1024 * 1024 }).toString('utf8');
const oldHtml = execSync('git show 16947f0~1:docs/index.html', { cwd: repoCwd, maxBuffer: 30 * 1024 * 1024 }).toString('utf8');

// Function to extract DATABASE entries and their full text
function extractDatabaseEntries(html) {
  const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
  if (!scriptMatch) throw new Error('No script match');
  const code = scriptMatch[1];
  const dbStart = code.indexOf('const DATABASE = {');
  const appStart = code.indexOf('function App()');
  const dbCode = code.slice(dbStart, appStart);
  
  // Find all keys and their positions
  const keyRegex = /^\s*"([^"]+)":\s*\{/gm;
  const entries = [];
  let match;
  while ((match = keyRegex.exec(dbCode)) !== null) {
    entries.push({ key: match[1], index: match.index });
  }

  const result = new Map();
  for (let i = 0; i < entries.length; i++) {
    const key = entries[i].key;
    const start = entries[i].index;
    const end = (i + 1 < entries.length) ? entries[i + 1].index : dbCode.lastIndexOf('};');
    const content = dbCode.slice(start, end).trim();
    result.set(key, content);
  }
  return result;
}

const newEntries = extractDatabaseEntries(newHtml);
const oldEntries = extractDatabaseEntries(oldHtml);

console.log(`Total symbols in new: ${newEntries.size}`);
console.log(`Total symbols in old: ${oldEntries.size}`);

const changedSymbols = [];
const unchangedSymbols = [];
const addedSymbols = [];
const removedSymbols = [];

for (const [key, content] of newEntries.entries()) {
  if (!oldEntries.has(key)) {
    addedSymbols.push(key);
  } else if (oldEntries.get(key) !== content) {
    changedSymbols.push(key);
  } else {
    unchangedSymbols.push(key);
  }
}

for (const key of oldEntries.keys()) {
  if (!newEntries.has(key)) {
    removedSymbols.push(key);
  }
}

console.log(`Changed symbols: ${changedSymbols.length}`);
console.log(`Unchanged symbols: ${unchangedSymbols.length}`);
console.log(`Added symbols: ${addedSymbols.length}`);
console.log(`Removed symbols: ${removedSymbols.length}`);

console.log('\n--- Changed symbols list ---');
changedSymbols.sort().forEach((k, idx) => {
  console.log(`${idx + 1}. ${k}`);
});
