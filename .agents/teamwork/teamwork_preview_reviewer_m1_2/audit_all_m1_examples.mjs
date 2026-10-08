import fs from 'node:fs';
import vm from 'node:vm';
import { execSync } from 'node:child_process';

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

console.log(`Verifying syntax of all ${m1Symbols.length} example codes...`);

const syntaxErrors = [];

for (const sym of m1Symbols) {
  const item = db[sym];
  if (!item || !item.example || !item.example.code) {
    syntaxErrors.push({ sym, err: 'Missing example code' });
    continue;
  }
  const code = item.example.code;
  // Write to a temp python file in our agent dir and check with python3 -m py_compile
  const tempFile = '/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/temp_check.py';
  fs.writeFileSync(tempFile, code, 'utf8');

  try {
    execSync(`python3 -m py_compile ${tempFile}`, { stdio: 'pipe' });
  } catch (e) {
    syntaxErrors.push({ sym, err: e.stderr ? e.stderr.toString() : e.message });
  }
}

if (fs.existsSync('/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/temp_check.py')) {
  fs.unlinkSync('/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/temp_check.py');
}

console.log(`Syntax errors found: ${syntaxErrors.length} / ${m1Symbols.length}`);
if (syntaxErrors.length > 0) {
  console.log(syntaxErrors);
} else {
  console.log('[SUCCESS] All 59 example codes are 100% syntactically valid Python code!');
}
