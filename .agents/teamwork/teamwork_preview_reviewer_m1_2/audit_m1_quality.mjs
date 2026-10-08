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

const m1Symbols = [
  'drugex::data::corpus::interfaces::Corpus',
  'drugex::data::interfaces::DataSet',
  'drugex::data::interfaces::FragmentPairEncoder',
  'drugex::data::corpus::interfaces::Vocabulary',
  'drugex::data::corpus::interfaces::SequenceVocabulary',
  'drugex::data::corpus::vocabulary::VocSmiles',
  'drugex::data::corpus::vocabulary::VocNonGPT',
  'drugex::data::corpus::vocabulary::VocGraph',
  'drugex::data::corpus::SequenceCorpus',
  'drugex::data::datasets::SmilesDataSet',
  'drugex::data::datasets::SmilesFragDataSet',
  'drugex::data::datasets::GraphFragDataSet',
  'drugex::data::fragments::Fragmenter',
  'drugex::data::fragments::SequenceFragmentEncoder',
  'drugex::data::fragments::GraphFragmentEncoder',
  'drugex::data::fragments::FragmentCorpusEncoder',
  'drugex::data::fragments::FragmentPairsSupplier',
  'drugex::data::fragments::FragmentPairsEncodedSupplier',
  'drugex::data::fragments::FragmentPairsSplitter',
  'drugex::data::processing::Standardization',
  'drugex::data::processing::CorpusEncoder',
  'drugex::data::processing::RandomTrainTestSplitter',
  'drugex::data::interfaces::DataSplitter',
  'drugex::data::interfaces::DataToLoader',
  'drugex::molecules::interfaces::Molecule',
  'drugex::molecules::interfaces::MolSupplier',
  'drugex::molecules::interfaces::BaseMolSupplier',
  'drugex::molecules::converters::MolConverter',
  'drugex::molecules::converters::DefaultStandardizer',
  'drugex::molecules::converters::CleanSMILES',
  'drugex::molecules::converters::dummyMolsFromFragments',
  'drugex::molecules::converters::default::SmilesToDrEx',
  'drugex::molecules::converters::default::DrExToSMILES',
  'drugex::molecules::converters::default::RDKitToDrEx',
  'drugex::molecules::converters::default::Identity',
  'drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment',
  'drugex::molecules::suppliers::StandardizedSupplier',
  'drugex::molecules::suppliers::ListSupplier',
  'drugex::molecules::mol::DrExMol',
  'drugex::parallel::ParallelProcessor',
  'drugex::parallel::collectors::ListExtend',
  'drugex::parallel::ParallelSupplierEvaluator',
  'drugex::parallel::interfaces::ResultCollector',
  'drugex::parallel::interfaces::ListCollector',
  'drugex::parallel::collectors::ListAppend',
  'drugex::molecules::converters::ConversionException',
  'drugex::molecules::converters::StandardizationException',
  'drugex::molecules::interfaces::ItemException',
  'drugex::molecules::mol::InitializationException',
  'drugex::parallel::interfaces::ParallelException',
  'drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException',
  'drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException',
  'drugex::data::fragments::FragmentPairsCollector',
  'drugex::data::datasets::TargetCreator',
  'drugex::data::datasets::TgtData',
  'drugex::data::processing::StandardizationCollector',
  'drugex::data::utils::getDataPaths',
  'drugex::data::utils::getVocPaths',
  'drugex::data::processing::Standardization::Collector'
];

console.log('--- 1. PLACEHOLDER / DUMMY TEXT AUDIT ---');
const suspiciousPatterns = [/\btodo\b/i, /\btbd\b/i, /\blorem\b/i, /\bplaceholder\b/i, /\bfoo\b/i, /\bbar\b/i];
let foundSuspicious = 0;

for (const k of m1Symbols) {
  const item = db[k];
  const allText = JSON.stringify(item);
  for (const pat of suspiciousPatterns) {
    if (pat.test(allText)) {
      // Exclude legitimate words if any
      console.warn(`[SUSPICIOUS PATTERN ${pat}] in ${k}`);
      foundSuspicious++;
    }
  }
}
if (foundSuspicious === 0) {
  console.log('[PASS] Zero placeholder or dummy patterns found across all 59 symbols.');
}

console.log('\n--- 2. DUPLICATE DESCRIPTION AUDIT ---');
const descriptions = new Map();
let duplicateDesc = 0;
for (const k of m1Symbols) {
  const desc = db[k].description.trim();
  if (descriptions.has(desc)) {
    console.warn(`[DUPLICATE DESCRIPTION] between ${descriptions.get(desc)} and ${k}`);
    duplicateDesc++;
  } else {
    descriptions.set(desc, k);
  }
}
if (duplicateDesc === 0) {
  console.log('[PASS] Zero duplicate descriptions found (100% unique symbol descriptions).');
}

console.log('\n--- 3. DETAILED STATS PER SYMBOL ---');
for (const k of m1Symbols) {
  const item = db[k];
  const paramCount = (item.parameters || []).length;
  const memberCount = (item.members || []).length;
  const seeAlsoCount = (item.seeAlso || []).length;
  const descLen = item.description.length;
  const notesLen = item.notes.length;
  const exCodeLen = item.example?.code?.length || 0;
  const exOutLen = item.example?.output?.length || 0;
  
  if (descLen < 50 || notesLen < 20 || exCodeLen < 50 || exOutLen < 10) {
    console.log(`[LOW DEPTH] ${k}: desc=${descLen}, notes=${notesLen}, code=${exCodeLen}, out=${exOutLen}`);
  }
}
