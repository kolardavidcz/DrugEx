import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const docsPath = path.resolve(__dirname, '..', 'index.html');

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

const items = [];
for (const [key, item] of Object.entries(db)) {
  const desc = item.description || '';
  const notes = item.notes || '';
  const shortMembers = (item.members || []).filter(m => (m.desc || '').length < 50);
  const shortParams = (item.parameters || []).filter(p => (p.desc || '').length < 30);
  const shortVars = (item.variables || []).filter(v => (v.desc || '').length < 30);

  let score = 0;
  if (desc.length < 180) score += 2;
  if (notes.length < 150) score += 2;
  if (shortMembers.length > 0) score += shortMembers.length;
  if (shortParams.length > 0) score += shortParams.length;

  if (score > 2) {
    items.push({
      key,
      title: item.title,
      category: item.category,
      descLen: desc.length,
      notesLen: notes.length,
      shortMembers: shortMembers.map(m => m.name),
      shortParams: shortParams.map(p => p.name)
    });
  }
}

console.log(`Found ${items.length} symbols with potential for meaningful explanation/notes enhancement:`);
items.slice(0, 15).forEach(it => {
  console.log(`- [${it.key}] (${it.category}) desc=${it.descLen}c, notes=${it.notesLen}c, shortParams=${it.shortParams.join(', ') || 'none'}`);
});
