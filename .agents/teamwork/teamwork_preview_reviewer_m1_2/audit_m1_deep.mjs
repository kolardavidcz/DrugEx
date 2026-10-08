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

// The 59 symbols modified in M1
const m1Symbols = [
  // Group 1: Data and corpus interfaces, sequence/graph vocabularies
  'drugex::data::corpus::interfaces::Corpus',
  'drugex::data::interfaces::DataSet',
  'drugex::data::interfaces::FragmentPairEncoder',
  'drugex::data::corpus::interfaces::Vocabulary',
  'drugex::data::corpus::interfaces::SequenceVocabulary',
  'drugex::data::corpus::vocabulary::VocSmiles',
  'drugex::data::corpus::vocabulary::VocNonGPT',
  'drugex::data::corpus::vocabulary::VocGraph',
  'drugex::data::corpus::SequenceCorpus',
  // Group 2: Datasets, fragmenters, encoders, and suppliers
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
  // Group 3: Molecular representations, suppliers, and converters
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
  // Group 4: Parallel execution and collectors
  'drugex::parallel::ParallelProcessor',
  'drugex::parallel::collectors::ListExtend',
  'drugex::parallel::ParallelSupplierEvaluator',
  'drugex::parallel::interfaces::ResultCollector',
  'drugex::parallel::interfaces::ListCollector',
  'drugex::parallel::collectors::ListAppend',
  // Group 5: Core error hierarchy
  'drugex::molecules::converters::ConversionException',
  'drugex::molecules::converters::StandardizationException',
  'drugex::molecules::interfaces::ItemException',
  'drugex::molecules::mol::InitializationException',
  'drugex::parallel::interfaces::ParallelException',
  'drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException',
  'drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException',
  // Group 6: Collectors, target creators, and path helpers
  'drugex::data::fragments::FragmentPairsCollector',
  'drugex::data::datasets::TargetCreator',
  'drugex::data::datasets::TgtData',
  'drugex::data::processing::StandardizationCollector',
  'drugex::data::utils::getDataPaths',
  'drugex::data::utils::getVocPaths',
  'drugex::data::processing::Standardization::Collector'
];

console.log(`Auditing ${m1Symbols.length} M1 symbols...`);

const issues = [];
const stats = {
  tensorDocumented: 0,
  failureModesDocumented: 0,
  boundaryConditionsDocumented: 0,
  academicDepthWordCountTotal: 0,
  averageWordCount: 0,
  examplesChecked: 0
};

for (const sym of m1Symbols) {
  const item = db[sym];
  if (!item) {
    issues.push({ sym, error: 'Symbol missing in DATABASE' });
    continue;
  }

  const combinedText = `${item.description} ${item.notes} ${(item.parameters || []).map(p => p.description).join(' ')}`;
  const wordCount = combinedText.split(/\s+/).filter(Boolean).length;
  stats.academicDepthWordCountTotal += wordCount;

  // Check tensor / matrix dimension documentation where applicable
  const hasTensorMentions = /(tensor|matrix|shape|dimension|\([NnBMVL,\s\+\d]+\)|dim=|torch\.Tensor|np\.ndarray)/i.test(combinedText);
  if (hasTensorMentions) stats.tensorDocumented++;

  // Check failure modes / exception / invalid handling
  const hasFailureModes = /(exception|raise|invalid|error|fail|empty|fallback|skip|unparseable|mismatch)/i.test(combinedText);
  if (hasFailureModes) stats.failureModesDocumented++;

  // Check boundary conditions / ranges / constraints
  const hasBoundaries = /(interval|range|boundary|none|default|0|1|threshold|limit|valid|chunk|batch)/i.test(combinedText);
  if (hasBoundaries) stats.boundaryConditionsDocumented++;

  // Check example code
  if (item.example) {
    stats.examplesChecked++;
    if (!item.example.code || item.example.code.trim().length === 0) {
      issues.push({ sym, error: 'Example code empty' });
    }
    if (typeof item.example.output !== 'string') {
      issues.push({ sym, error: 'Example output missing or non-string' });
    }
    // Check if example imports from proper modules
    if (!item.example.code.includes('drugex') && !item.example.code.includes('import') && item.kind !== 'exception') {
      issues.push({ sym, error: 'Example code does not contain import or drugex reference' });
    }
  }

  // Check synopsis hygiene
  if (item.kind === 'class') {
    if (item.members && item.members.some(m => m.name.startsWith('__init__'))) {
      issues.push({ sym, error: '__init__ found in members table' });
    }
    // Check method ordering
    let seenMagic = false;
    for (const m of (item.members || [])) {
      if (m.name.startsWith('__')) {
        seenMagic = true;
      } else if (seenMagic) {
        issues.push({ sym, error: `Method ordering violation: regular method ${m.name} after magic method` });
        break;
      }
    }
  }
}

stats.averageWordCount = Math.round(stats.academicDepthWordCountTotal / m1Symbols.length);

console.log('\n--- AUDIT METRICS ---');
console.log(`Total symbols audited: ${m1Symbols.length}`);
console.log(`Average technical word count per symbol: ${stats.averageWordCount} words`);
console.log(`Symbols with tensor/array dimensions documented: ${stats.tensorDocumented} / ${m1Symbols.length}`);
console.log(`Symbols with failure modes & error handling: ${stats.failureModesDocumented} / ${m1Symbols.length}`);
console.log(`Symbols with boundary conditions & valid ranges: ${stats.boundaryConditionsDocumented} / ${m1Symbols.length}`);
console.log(`Examples verified: ${stats.examplesChecked} / ${m1Symbols.length}`);
console.log(`Issues found: ${issues.length}`);
if (issues.length > 0) {
  console.log(issues);
}
