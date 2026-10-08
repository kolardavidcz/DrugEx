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

const tensorSymbols = [
  'drugex::data::corpus::interfaces::Vocabulary',
  'drugex::data::corpus::interfaces::SequenceVocabulary',
  'drugex::data::corpus::vocabulary::VocSmiles',
  'drugex::data::corpus::vocabulary::VocNonGPT',
  'drugex::data::corpus::vocabulary::VocGraph',
  'drugex::data::interfaces::DataSet',
  'drugex::data::datasets::SmilesDataSet',
  'drugex::data::datasets::SmilesFragDataSet',
  'drugex::data::datasets::GraphFragDataSet',
  'drugex::data::interfaces::FragmentPairEncoder',
  'drugex::data::fragments::SequenceFragmentEncoder',
  'drugex::data::fragments::GraphFragmentEncoder',
  'drugex::data::datasets::TargetCreator',
  'drugex::data::datasets::TgtData'
];

console.log('--- CHECKING TENSOR-RELATED SYMBOLS ---');
for (const k of tensorSymbols) {
  const item = db[k];
  if (!item) {
    console.log(`MISSING: ${k}`);
    continue;
  }
  const text = `${item.description}\nNOTES: ${item.notes}\nMEMBERS:\n${(item.members || []).map(m => `  ${m.name}: ${m.desc}`).join('\n')}`;
  console.log(`\n=== ${k} ===`);
  console.log(text);
}
