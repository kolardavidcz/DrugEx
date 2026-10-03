import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync('/home/kolar/build_projects/drugex_edu/docs/index.html', 'utf8');
const script = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/)[1];
const dbSnippet = script.slice(script.indexOf('const DATABASE = {'), script.indexOf('function App()')).trim();
const ctx = {};
vm.createContext(ctx);
vm.runInContext(dbSnippet.replace(/^const DATABASE =/, 'globalThis.DATABASE ='), ctx);
const db = ctx.DATABASE;
const keys = Object.keys(db);

const kinds = {};
const categories = {};
let withParams = 0;
let withReturns = 0;
let withMembers = 0;
let withVariables = 0;
let totalMembers = 0;
let totalParams = 0;
let totalSeeAlso = 0;
const badges = {};

for (const k of keys) {
  const item = db[k];
  kinds[item.kind] = (kinds[item.kind] || 0) + 1;
  categories[item.category] = (categories[item.category] || 0) + 1;
  if (item.parameters && item.parameters.length) { withParams++; totalParams += item.parameters.length; }
  if (item.returns) withReturns++;
  if (item.members && item.members.length) { withMembers++; totalMembers += item.members.length; }
  if (item.variables && item.variables.length) withVariables++;
  if (item.seeAlso) totalSeeAlso += item.seeAlso.length;
  for (const b of (item.badges || [])) badges[b] = (badges[b] || 0) + 1;
}

console.log(JSON.stringify({
  totalSymbols: keys.length,
  categories,
  kinds,
  withParams,
  totalParams,
  withReturns,
  withMembers,
  totalMembers,
  withVariables,
  totalSeeAlso,
  avgSeeAlso: (totalSeeAlso / keys.length).toFixed(2),
  topBadges: Object.entries(badges).sort((a,b) => b[1]-a[1]).slice(0, 15)
}, null, 2));
