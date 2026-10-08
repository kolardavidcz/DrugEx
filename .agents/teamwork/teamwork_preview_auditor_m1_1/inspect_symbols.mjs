import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync('/home/kolar/build_projects/drugex_edu/docs/index.html', 'utf8');
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
const code = scriptMatch[1];
const dbStart = code.indexOf('const DATABASE = {');
const appStart = code.indexOf('function App()');
const dbSnippet = code.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');

const context = {};
vm.createContext(context);
vm.runInContext(dbSnippet, context);
const db = context.DATABASE;

// The 59 M1 symbols
const m1Symbols = [
  "drugex::data::corpus::SequenceCorpus",
  "drugex::data::corpus::interfaces::Corpus",
  "drugex::data::corpus::interfaces::SequenceVocabulary",
  "drugex::data::corpus::interfaces::Vocabulary",
  "drugex::data::corpus::vocabulary::VocGraph",
  "drugex::data::corpus::vocabulary::VocNonGPT",
  "drugex::data::corpus::vocabulary::VocSmiles",
  "drugex::data::datasets::GraphFragDataSet",
  "drugex::data::datasets::SmilesDataSet",
  "drugex::data::datasets::SmilesFragDataSet",
  "drugex::data::datasets::TargetCreator",
  "drugex::data::datasets::TgtData",
  "drugex::data::fragments::FragmentCorpusEncoder",
  "drugex::data::fragments::FragmentPairsCollector",
  "drugex::data::fragments::FragmentPairsEncodedSupplier",
  "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException",
  "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException",
  "drugex::data::fragments::FragmentPairsSplitter",
  "drugex::data::fragments::FragmentPairsSupplier",
  "drugex::data::fragments::Fragmenter",
  "drugex::data::fragments::GraphFragmentEncoder",
  "drugex::data::fragments::SequenceFragmentEncoder",
  "drugex::data::interfaces::DataSet",
  "drugex::data::interfaces::DataSplitter",
  "drugex::data::interfaces::DataToLoader",
  "drugex::data::interfaces::FragmentPairEncoder",
  "drugex::data::processing::CorpusEncoder",
  "drugex::data::processing::RandomTrainTestSplitter",
  "drugex::data::processing::Standardization",
  "drugex::data::processing::Standardization::Collector",
  "drugex::data::processing::StandardizationCollector",
  "drugex::data::utils::getDataPaths",
  "drugex::data::utils::getVocPaths",
  "drugex::molecules::converters::CleanSMILES",
  "drugex::molecules::converters::ConversionException",
  "drugex::molecules::converters::DefaultStandardizer",
  "drugex::molecules::converters::MolConverter",
  "drugex::molecules::converters::StandardizationException",
  "drugex::molecules::converters::default::DrExToSMILES",
  "drugex::molecules::converters::default::Identity",
  "drugex::molecules::converters::default::RDKitToDrEx",
  "drugex::molecules::converters::default::SmilesToDrEx",
  "drugex::molecules::converters::dummyMolsFromFragments",
  "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment",
  "drugex::molecules::interfaces::BaseMolSupplier",
  "drugex::molecules::interfaces::ItemException",
  "drugex::molecules::interfaces::MolSupplier",
  "drugex::molecules::interfaces::Molecule",
  "drugex::molecules::mol::DrExMol",
  "drugex::molecules::mol::InitializationException",
  "drugex::molecules::suppliers::ListSupplier",
  "drugex::molecules::suppliers::StandardizedSupplier",
  "drugex::parallel::ParallelProcessor",
  "drugex::parallel::ParallelSupplierEvaluator",
  "drugex::parallel::collectors::ListAppend",
  "drugex::parallel::collectors::ListExtend",
  "drugex::parallel::interfaces::ListCollector",
  "drugex::parallel::interfaces::ParallelException",
  "drugex::parallel::interfaces::ResultCollector"
];

console.log(`Auditing ${m1Symbols.length} Milestone M1 symbols...`);

const suspiciousPatterns = [
  /TODO/i,
  /FIXME/i,
  /TBD/i,
  /placeholder/i,
  /lorem ipsum/i,
  /dummy text/i,
  /test test/i,
  /facade/i,
  /dummy implementation/i
];

let issues = 0;
const descriptions = new Map();
const notes = new Map();

for (const sym of m1Symbols) {
  const item = db[sym];
  if (!item) {
    console.error(`FATAL: Symbol ${sym} not found in DATABASE!`);
    issues++;
    continue;
  }

  // Check description
  if (!item.description || item.description.trim().length < 40) {
    console.error(`ISSUE [${sym}]: Description is too short or missing (${item.description?.length} chars)`);
    issues++;
  }
  for (const pat of suspiciousPatterns) {
    if (pat.test(item.description)) {
      console.error(`SUSPICIOUS [${sym}]: Description matches pattern ${pat}: "${item.description}"`);
      issues++;
    }
  }

  // Check notes
  if (!item.notes || item.notes.trim().length < 30) {
    console.error(`ISSUE [${sym}]: Notes is too short or missing (${item.notes?.length} chars)`);
    issues++;
  }
  for (const pat of suspiciousPatterns) {
    if (pat.test(item.notes)) {
      console.error(`SUSPICIOUS [${sym}]: Notes matches pattern ${pat}: "${item.notes}"`);
      issues++;
    }
  }

  // Check uniqueness to detect copy-paste facades
  if (descriptions.has(item.description)) {
    console.error(`DUPLICATE DESCRIPTION: [${sym}] shares exact description with [${descriptions.get(item.description)}]`);
    issues++;
  } else {
    descriptions.set(item.description, sym);
  }

  // Note: some aliases like StandardizationCollector and Standardization::Collector might share notes if intentional
  if (notes.has(item.notes)) {
    console.log(`INFO: [${sym}] has identical notes to [${notes.get(item.notes)}]`);
  } else {
    notes.set(item.notes, sym);
  }

  // Check synopsis
  if (!item.synopsis || item.synopsis.trim().length < 15) {
    console.error(`ISSUE [${sym}]: Synopsis is too short or missing`);
    issues++;
  }

  // Check example code & output
  if (!item.example || !item.example.code || item.example.code.trim().length < 40) {
    console.error(`ISSUE [${sym}]: Example code is too short or missing (${item.example?.code?.length} chars)`);
    issues++;
  }
  if (!item.example || typeof item.example.output !== 'string' || item.example.output.trim().length === 0) {
    console.error(`ISSUE [${sym}]: Example output is empty or missing`);
    issues++;
  }

  // Check members for classes
  if (item.kind === 'class') {
    if (!Array.isArray(item.members) || item.members.length === 0) {
      console.error(`ISSUE [${sym}]: Class has empty members array`);
      issues++;
    } else {
      for (const m of item.members) {
        if (!m.name || !m.desc || m.desc.trim().length < 10) {
          console.error(`ISSUE [${sym}]: Member "${m.name}" has inadequate description: "${m.desc}"`);
          issues++;
        }
      }
    }
  }

  // Check parameters if present
  if (Array.isArray(item.parameters)) {
    for (const p of item.parameters) {
      if (!p.name || !p.type || !p.desc || p.desc.trim().length < 10) {
        console.error(`ISSUE [${sym}]: Parameter "${p.name}" has inadequate type/desc`);
        issues++;
      }
    }
  }

  // Check variables if present
  if (Array.isArray(item.variables)) {
    for (const v of item.variables) {
      if (!v.name || !v.type || !v.desc || v.desc.trim().length < 10) {
        console.error(`ISSUE [${sym}]: Variable "${v.name}" has inadequate type/desc`);
        issues++;
      }
    }
  }
}

console.log(`\nScan finished. Total integrity issues detected: ${issues}`);
