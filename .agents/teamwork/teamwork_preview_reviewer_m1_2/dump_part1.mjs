import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync('/home/kolar/build_projects/drugex_edu/docs/index.html', 'utf8');
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
const scriptCode = scriptMatch[1];
const dbStart = scriptCode.indexOf('const DATABASE = {');
const appStart = scriptCode.indexOf('function App()');
const dbCode = scriptCode.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');

const context = {};
vm.createContext(context);
vm.runInContext(dbCode, context);
const db = context.DATABASE;

const sampleKeys = [
  'drugex::data::corpus::vocabulary::VocGraph',
  'drugex::data::corpus::vocabulary::VocSmiles',
  'drugex::data::datasets::SmilesDataSet',
  'drugex::data::datasets::SmilesFragDataSet',
  'drugex::data::datasets::GraphFragDataSet',
  'drugex::data::fragments::Fragmenter'
];

for (const k of sampleKeys) {
  console.log(`\n=================== ${k} ===================`);
  const item = db[k];
  console.log('SYNOPSIS:');
  console.log(item.synopsis);
  console.log('\nDESCRIPTION:');
  console.log(item.description);
  console.log('\nNOTES:');
  console.log(item.notes);
  console.log('\nPARAMETERS:');
  console.log(JSON.stringify(item.parameters, null, 2));
  console.log('\nMEMBERS:');
  console.log(JSON.stringify(item.members, null, 2));
  console.log('\nEXAMPLE:');
  console.log(item.example);
}
