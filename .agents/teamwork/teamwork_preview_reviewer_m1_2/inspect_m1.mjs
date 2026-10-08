import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const indexPath = '/home/kolar/build_projects/drugex_edu/docs/index.html';
const html = fs.readFileSync(indexPath, 'utf8');

const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  console.error('No script found');
  process.exit(1);
}

const scriptCode = scriptMatch[1];
const dbStart = scriptCode.indexOf('const DATABASE = {');
const secStart = scriptCode.indexOf('const LIBRARY_SECTIONS = [');
const secEnd = dbStart;
const appStart = scriptCode.indexOf('function App()');

const secCode = scriptCode.slice(secStart, secEnd).trim().replace(/^const LIBRARY_SECTIONS =/, 'globalThis.LIBRARY_SECTIONS =');
const dbCode = scriptCode.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');

const context = {};
vm.createContext(context);
vm.runInContext(secCode, context);
vm.runInContext(dbCode, context);

const sections = context.LIBRARY_SECTIONS;
const db = context.DATABASE;

console.log('Total sections:', sections.length);
sections.forEach((sec, idx) => {
  const symCount = sec.subsections.reduce((acc, sub) => acc + sub.keys.length, 0);
  console.log(`Section ${idx + 1}: [${sec.id}] ${sec.title} (${symCount} symbols)`);
});

const sec1 = sections[0]; // data
const sec6 = sections[5]; // parallel

const sec1Keys = sec1.subsections.flatMap(sub => sub.keys);
const sec6Keys = sec6.subsections.flatMap(sub => sub.keys);

console.log(`\nSection 1 (${sec1.id}) symbols (${sec1Keys.length}):`);
console.log(sec1Keys);

console.log(`\nSection 6 (${sec6.id}) symbols (${sec6Keys.length}):`);
console.log(sec6Keys);

const totalM1Keys = [...sec1Keys, ...sec6Keys];
console.log(`\nTotal M1 symbols count: ${totalM1Keys.length}`);
