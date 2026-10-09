/**
 * Validation script for DrugEx cppreference documentation platform.
 * Verifies that the modular documentation platform (index.html, app.js, style.css,
 * and 11 discrete data modules in data/) contains a comprehensive, structurally complete,
 * and error-free reference database with zero broken cross-links and zero emojis.
 */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const targetArg = process.argv[2];
const docsPath = targetArg 
  ? path.resolve(process.cwd(), targetArg) 
  : path.join(projectRoot, 'index.html');

console.log('[Audit] Validating DrugEx Reference Documentation Platform...');
console.log(`Target: ${docsPath}`);

if (!fs.existsSync(docsPath)) {
  console.error(`Fatal: Documentation file not found at ${docsPath}`);
  process.exit(1);
}

let errors = 0;

const html = fs.readFileSync(docsPath, 'utf8');

// 1. Verify basic HTML document structure and modular script imports
if (!html.includes('<!DOCTYPE html>') || !html.includes('<div id="root"></div>')) {
  console.error('Fatal: Malformed HTML template structure.');
  process.exit(1);
}

if (!html.includes('<link rel="stylesheet" href="style.css">')) {
  console.error('Fatal: style.css stylesheet link missing in index.html.');
  errors++;
}

if (!html.includes('<script type="text/babel" src="app.js"></script>')) {
  console.error('Fatal: app.js Babel script tag missing in index.html.');
  errors++;
}

// 2. Verify all 11 modular data script tags in index.html
const requiredDataScripts = [
  'data/sections.js',
  'data/01_morl.js',
  'data/02_neural.js',
  'data/03_explorers.js',
  'data/04_scoring.js',
  'data/05_modifiers.js',
  'data/06_data.js',
  'data/07_molecules.js',
  'data/08_workflows.js',
  'data/09_ecosystem.js',
  'data/10_experimental.js'
];

for (const scriptPath of requiredDataScripts) {
  if (!html.includes(`<script src="${scriptPath}"></script>`)) {
    console.error(`Fatal: Missing modular script tag in index.html: <script src="${scriptPath}"></script>`);
    errors++;
  }
}

// 3. Load modular data files into isolated VM context
const dataDir = path.join(projectRoot, 'data');
const context = {
  window: {},
  globalThis: {}
};
context.window = context;
context.globalThis = context;
vm.createContext(context);

const dataFiles = [
  'sections.js',
  '01_morl.js',
  '02_neural.js',
  '03_explorers.js',
  '04_scoring.js',
  '05_modifiers.js',
  '06_data.js',
  '07_molecules.js',
  '08_workflows.js',
  '09_ecosystem.js',
  '10_experimental.js'
];

for (const df of dataFiles) {
  const filePath = path.join(dataDir, df);
  if (!fs.existsSync(filePath)) {
    console.error(`Fatal: Required data file missing: ${filePath}`);
    process.exit(1);
  }
  const code = fs.readFileSync(filePath, 'utf8');
  try {
    vm.runInContext(code, context);
  } catch (err) {
    console.error(`Syntax or evaluation error in data file ${df}:`, err);
    process.exit(1);
  }
}

const db = context.DATABASE || context.window.DATABASE;
if (!db || typeof db !== 'object') {
  console.error('Fatal: DATABASE object was not populated by data modules.');
  process.exit(1);
}

const keys = Object.keys(db);
console.log(`[PASS] DATABASE evaluated successfully: ${keys.length} total symbols registered across 10 modular files.`);

if (keys.length < 234) {
  console.error(`Validation failed: Expected at least 234 symbols for complete coverage, got ${keys.length}.`);
  process.exit(1);
}

// 4. Validate schema of every single entry
const categories = { core: 0, ecosystem: 0, experimental: 0 };
const brokenLinks = [];

for (const key of keys) {
  const item = db[key];

  // Category tally
  if (categories[item.category] !== undefined) {
    categories[item.category]++;
  } else {
    console.error(`Warning: Key "${key}" has unexpected category "${item.category}"`);
    errors++;
  }

  // Required fields
  const required = ['category', 'module', 'title', 'kind', 'badges', 'synopsis', 'description', 'notes', 'example', 'seeAlso'];
  for (const field of required) {
    if (!item[field]) {
      console.error(`Key "${key}" is missing required field: "${field}"`);
      errors++;
    }
  }

  // Check class member completeness and constructor parameter hygiene
  if (item.kind === 'class') {
    if (!Array.isArray(item.members) || item.members.length === 0) {
      console.error(`Class "${key}" has empty members array.`);
      errors++;
    }
    const syn = item.synopsis || '';
    const hasZeroArgInit = /def\s+__init__\s*\(\s*self\s*\)\s*(->\s*None\s*)?:/.test(syn);
    if (hasZeroArgInit && Array.isArray(item.parameters) && item.parameters.length > 0) {
      console.error(`Class "${key}" defines zero-argument __init__(self) -> None in synopsis, but has non-empty parameters array.`);
      errors++;
    }
  }

  // Check example structure
  if (item.example) {
    if (typeof item.example.code !== 'string' || item.example.code.trim().length === 0) {
      console.error(`Key "${key}" has empty or non-string example code.`);
      errors++;
    }
    if (typeof item.example.output !== 'string') {
      console.error(`Key "${key}" has missing or non-string example output.`);
      errors++;
    }
  }

  // Check seeAlso cross links
  if (Array.isArray(item.seeAlso)) {
    for (const targetKey of item.seeAlso) {
      if (!db[targetKey]) {
        brokenLinks.push({ source: key, target: targetKey });
        errors++;
      }
    }
  } else {
    console.error(`Key "${key}" seeAlso is not an array.`);
    errors++;
  }
}

console.log('Category Breakdown:');
console.log(`   - DrugEx Core:       ${categories.core} symbols`);
console.log(`   - Chemo Ecosystem:   ${categories.ecosystem} symbols`);
console.log(`   - Experimental:      ${categories.experimental} symbols`);

if (brokenLinks.length > 0) {
  console.error(`Found ${brokenLinks.length} broken cross-links in seeAlso:`);
  for (const b of brokenLinks) {
    console.error(`   - [${b.source}] -> references missing [${b.target}]`);
  }
}

// 5. Validate LIBRARY_SECTIONS Concentrated Main Index (Matching cppreference.com)
console.log('\nValidating Concentrated Main Page Library Index (cppreference.com format)...');
const sections = context.LIBRARY_SECTIONS || context.window.LIBRARY_SECTIONS;

if (!Array.isArray(sections) || sections.length !== 10) {
  console.error(`Expected 10 library categories in LIBRARY_SECTIONS, got ${sections?.length}`);
  errors++;
} else {
  const indexedKeys = new Set();
  for (const sec of sections) {
    if (!sec.id || !sec.title || !Array.isArray(sec.subsections)) {
      console.error(`Invalid section structure for section ${sec.id || 'unknown'}`);
      errors++;
      continue;
    }
    for (const sub of sec.subsections) {
      if (!sub.subtitle || !Array.isArray(sub.keys)) {
        console.error(`Invalid subsection structure in section "${sec.title}"`);
        errors++;
        continue;
      }
      for (const k of sub.keys) {
        if (!db[k]) {
          console.error(`Section "${sec.title}" references non-existent key: "${k}"`);
          errors++;
        }
        if (indexedKeys.has(k)) {
          console.error(`Duplicate key in LIBRARY_SECTIONS: "${k}"`);
          errors++;
        }
        indexedKeys.add(k);
      }
    }
  }
  
  const missingKeys = keys.filter(k => !indexedKeys.has(k));
  if (missingKeys.length > 0) {
    console.error(`${missingKeys.length} symbols missing from LIBRARY_SECTIONS:`, missingKeys);
    errors += missingKeys.length;
  } else {
    console.log(`[PASS] All ${keys.length} symbols 100% indexed across ${sections.length} thematic libraries with 0 duplicates.`);
  }
}

// 6. Validate helper functions and logic in app.js
console.log('\nValidating helper functions in app.js...');
const appJsPath = path.join(projectRoot, 'app.js');
if (!fs.existsSync(appJsPath)) {
  console.error(`Fatal: app.js file not found at ${appJsPath}`);
  process.exit(1);
}

const appCode = fs.readFileSync(appJsPath, 'utf8');

context.React = {
  useState: () => [],
  useMemo: (fn) => fn(),
  useEffect: () => {},
  useRef: () => ({ current: null })
};

const appStart = appCode.indexOf('function App()');
const helpersCode = appStart !== -1 ? appCode.slice(0, appStart) : appCode;

try {
  vm.runInContext(helpersCode, context);
  console.log('[PASS] app.js helper functions evaluated cleanly in sandbox VM context.');
} catch (err) {
  console.error('Error evaluating helper functions from app.js in sandbox VM:', err);
  errors++;
}

const getSymbolTag = context.getSymbolTag;
if (typeof getSymbolTag !== 'function') {
  console.error('getSymbolTag is not a function in app.js.');
  errors++;
} else {
  const tests = [
    { key: 'experimental::usrcat::USRCAT_Screening', expected: '60D' },
    { key: 'experimental::electroshape::ElectroShape_4D', expected: '4D' },
    { key: 'rdkit::Chem::rdShapeAlign::AlignMol', expected: 'RDKit' },
    { key: 'drugex::training::generators::SequenceTransformer', expected: 'GPU' },
    { key: 'drugex::training::environment::DrugExEnvironment', expected: 'v3.4' }
  ];
  for (const t of tests) {
    const item = db[t.key];
    const actual = getSymbolTag(t.key, item);
    if (actual !== t.expected) {
      console.error(`getSymbolTag("${t.key}") expected "${t.expected}", got "${actual}"`);
      errors++;
    }
  }
  console.log('[PASS] getSymbolTag verified for core badges (60D, 4D, RDKit, GPU, v3.4).');
}

// Validate isAbstractSymbol helper
const isAbstractSymbol = context.isAbstractSymbol;
if (typeof isAbstractSymbol !== 'function') {
  console.error('isAbstractSymbol is not a function in app.js.');
  errors++;
} else {
  const absTrueTests = [
    'drugex::training::environment::Environment',
    'drugex::training::scorers::Scorer',
    'drugex::training::generators::Generator',
    'drugex::training::explorers::Explorer',
    'drugex::training::rewards::RewardScheme',
    'drugex::data::corpus::interfaces::Vocabulary'
  ];
  const absFalseTests = [
    'drugex::training::environment::DrugExEnvironment',
    'drugex::training::generators::SequenceRNN',
    'drugex::training::explorers::SequenceExplorer',
    'drugex::training::rewards::ParetoCrowdingDistance',
    'drugex::molecules::converters::ConversionException'
  ];
  for (const k of absTrueTests) {
    if (!isAbstractSymbol(k, db[k])) {
      console.error(`isAbstractSymbol("${k}") expected true, got false`);
      errors++;
    }
  }
  for (const k of absFalseTests) {
    if (isAbstractSymbol(k, db[k])) {
      console.error(`isAbstractSymbol("${k}") expected false, got true`);
      errors++;
    }
  }
  const totalAbstract = Object.keys(db).filter(k => isAbstractSymbol(k, db[k])).length;
  if (totalAbstract !== 30) {
    console.error(`Expected exactly 30 abstract classes in database, found ${totalAbstract}`);
    errors++;
  } else {
    console.log(`[PASS] isAbstractSymbol verified: exactly ${totalAbstract} abstract base classes identified.`);
  }
}

// Validate getEffectiveMembers helper
const getEffectiveMembers = context.getEffectiveMembers;
if (typeof getEffectiveMembers !== 'function') {
  console.error('getEffectiveMembers is not a function in app.js.');
  errors++;
} else {
  let classesWithInit = 0;
  let methodOrderErrors = 0;
  let checkedClasses = 0;
  for (const k of keys) {
    const item = db[k];
    if (item.kind === 'class') {
      checkedClasses++;
      const eff = getEffectiveMembers(item);
      if (eff.some(m => m.name.startsWith('__init__'))) {
        console.error(`Class "${k}" unexpectedly has __init__ in getEffectiveMembers (constructors belong in header synopsis).`);
        classesWithInit++;
        errors++;
      }
      let seenMagic = false;
      for (const m of eff) {
        if (m.name.startsWith('__')) {
          seenMagic = true;
        } else if (seenMagic) {
          console.error(`Class "${k}" has regular method "${m.name}" after magic dunder method.`);
          methodOrderErrors++;
          errors++;
          break;
        }
      }
    }
  }
  if (classesWithInit === 0 && methodOrderErrors === 0) {
    console.log(`[PASS] getEffectiveMembers verified: 0 classes have __init__ in member functions; 100% of classes (${checkedClasses}) strictly order regular methods before magic methods.`);
  }
}

// Validate getMagicMethodOperator helper
const getMagicMethodOperator = context.getMagicMethodOperator;
if (typeof getMagicMethodOperator !== 'function') {
  console.error('getMagicMethodOperator is not a function in app.js.');
  errors++;
} else {
  const opTests = [
    { name: '__init__(args)', expectedOp: '(constructor)', expectedCls: 'op-constructor' },
    { name: '__call__(smiles)', expectedOp: 'obj(...)', expectedCls: 'op-call' },
    { name: '__getitem__(key)', expectedOp: 'obj[...]', expectedCls: 'op-subscript' },
    { name: '__len__()', expectedOp: 'len(obj)', expectedCls: 'op-len' },
    { name: '__iter__()', expectedOp: 'iter(obj)', expectedCls: 'op-iter' },
    { name: '__contains__(token)', expectedOp: 'x in obj', expectedCls: 'op-subscript' },
    { name: 'fit(train_loader)', expectedOp: null },
    { name: 'generate(100)', expectedOp: null }
  ];
  for (const ot of opTests) {
    const res = getMagicMethodOperator(ot.name);
    if (ot.expectedOp === null) {
      if (res !== null) {
        console.error(`getMagicMethodOperator("${ot.name}") expected null, got:`, res);
        errors++;
      }
    } else {
      if (!res || res.op !== ot.expectedOp || res.cls !== ot.expectedCls) {
        console.error(`getMagicMethodOperator("${ot.name}") mismatch: expected op="${ot.expectedOp}" cls="${ot.expectedCls}", got:`, res);
        errors++;
      }
    }
  }
  console.log('[PASS] getMagicMethodOperator verified across special Python operator mappings.');
}

// Validate getMicroVizType helper
const getMicroVizType = context.getMicroVizType;
if (typeof getMicroVizType !== 'function') {
  console.error('getMicroVizType is not a function in app.js.');
  errors++;
} else {
  const microVizExpected = {
    pareto_crowding: [
      'drugex::training::rewards::ParetoCrowdingDistance',
      'drugex::utils::pareto::get_Pareto_fronts',
      'drugex::training::rewards::ParetoRewardScheme',
      'drugex::training::rewards::RewardScheme',
      'drugex::training::rewards::WeightedSum'
    ],
    pareto_tanimoto: [
      'drugex::training::rewards::ParetoTanimotoDistance'
    ],
    modifier_curve: [
      'drugex::training::scorers::ScoreModifier',
      'drugex::training::scorers::SmoothClippedScore',
      'drugex::training::scorers::ClippedScore',
      'drugex::training::scorers::MinMaxGaussian',
      'drugex::training::scorers::modifiers::ThresholdedLinear',
      'drugex::training::scorers::modifiers::SmoothHump',
      'drugex::training::scorers::modifiers::Squared',
      'drugex::training::scorers::modifiers::AbsoluteScore',
      'drugex::training::scorers::modifiers::Chained',
      'drugex::training::scorers::Gaussian',
      'drugex::training::scorers::Linear'
    ],
    rl_policy_cycle: [
      'drugex::training::environment::DrugExEnvironment',
      'drugex::training::environment::Environment',
      'drugex::training::explorers::SequenceExplorer',
      'drugex::training::explorers::FragSequenceExplorer',
      'drugex::training::explorers::GraphExplorer',
      'drugex::training::explorers::FragGraphExplorer',
      'drugex::training::explorers::Explorer',
      'drugex::training::explorers::FragExplorer',
      'drugex::training::scorers::SmilesChecker',
      'drugex::train::Reinforce'
    ]
  };

  let microVizTestedCount = 0;
  for (const [expectedType, symList] of Object.entries(microVizExpected)) {
    for (const sym of symList) {
      if (!db[sym]) {
        console.error(`getMicroVizType test symbol "${sym}" does not exist in DATABASE.`);
        errors++;
        continue;
      }
      const actualType = getMicroVizType(sym);
      if (actualType !== expectedType) {
        console.error(`getMicroVizType("${sym}") expected "${expectedType}", got "${actualType}"`);
        errors++;
      } else {
        microVizTestedCount++;
      }
    }
  }
  if (getMicroVizType('unknown::key') !== null) {
    console.error('getMicroVizType("unknown::key") expected null, got non-null.');
    errors++;
  }
  console.log(`[PASS] getMicroVizType verified across ${microVizTestedCount} target symbols with 4 visualization types.`);
}

// 7. Strict Emoji Prohibition Check
console.log('\nValidating Strict Zero-Emoji Rule...');
const emojiRegex = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F600}-\u{1F64F}\u{1F1E6}-\u{1F1FF}]/u;

function collectDocFiles(dir) {
  let docFiles = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== '.git' && entry.name !== '.venv') {
        docFiles = docFiles.concat(collectDocFiles(fullPath));
      }
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (['.html', '.md', '.mjs', '.js', '.css', '.json'].includes(ext)) {
        docFiles.push(fullPath);
      }
    }
  }
  return docFiles;
}

const docsDir = projectRoot;
const repoRootDir = fs.existsSync(path.join(projectRoot, 'repo'))
  ? projectRoot
  : path.resolve(projectRoot, '..');

const rootDocMarkdown = ['README.md', 'AGENT.md', 'TEST_INFRA.md', 'TEST_READY.md']
  .map(file => path.join(repoRootDir, file))
  .filter(file => fs.existsSync(file));

const filesToScan = Array.from(new Set([
  docsPath,
  ...collectDocFiles(docsDir),
  ...rootDocMarkdown
]));

let emojiErrors = 0;
for (const file of filesToScan) {
  if (fs.existsSync(file)) {
    const content = fs.readFileSync(file, 'utf8');
    const match = content.match(emojiRegex);
    if (match) {
      console.error(`Error: Emoji detected in ${path.relative(repoRootDir, file)}: "${match[0]}"`);
      emojiErrors++;
      errors++;
    }
  }
}
if (emojiErrors === 0) {
  console.log(`[PASS] Verified 0 emojis across documentation files (${filesToScan.length} scanned).`);
}

if (errors > 0) {
  console.error(`Validation failed with ${errors} error(s).`);
  process.exit(1);
}

console.log('[SUCCESS] All assertions passed: 100% schema compliance, 0 broken cross-links, modular cppreference platform verified!\n');
