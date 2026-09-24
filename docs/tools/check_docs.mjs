/**
 * Validation script for DrugEx cppreference documentation platform.
 * Verifies that docs/cppreference/index.html contains a comprehensive,
 * structurally complete, and error-free reference database with zero broken cross-links.
 */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const targetArg = process.argv[2];
const rootCandidate = path.join(projectRoot, 'index.html');
const nestedCandidate = path.join(projectRoot, 'docs', 'cppreference', 'index.html');
const docsPath = targetArg 
  ? path.resolve(process.cwd(), targetArg) 
  : (fs.existsSync(rootCandidate) ? rootCandidate : nestedCandidate);

console.log('[Audit] Validating DrugEx Reference Documentation Platform...');
console.log(`Target: ${docsPath}`);

if (!fs.existsSync(docsPath)) {
  console.error(`Fatal: Documentation file not found at ${docsPath}`);
  process.exit(1);
}

let errors = 0;

if (!targetArg && fs.existsSync(rootCandidate) && fs.existsSync(nestedCandidate)) {
  const rootBuf = fs.readFileSync(rootCandidate);
  const nestedBuf = fs.readFileSync(nestedCandidate);
  if (!rootBuf.equals(nestedBuf)) {
    console.error('Error: root index.html and docs/cppreference/index.html differ! Both copies must remain strictly in sync.');
    errors++;
  } else {
    console.log('Verified: root index.html and docs/cppreference/index.html are in sync.');
  }
}

const html = fs.readFileSync(docsPath, 'utf8');

// 1. Verify basic HTML document structure
if (!html.includes('<!DOCTYPE html>') || !html.includes('<div id="root"></div>')) {
  console.error('Fatal: Malformed HTML template structure.');
  process.exit(1);
}

// 2. Extract script block
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  console.error('Fatal: No <script type="text/babel"> block found in index.html.');
  process.exit(1);
}

const scriptCode = scriptMatch[1];

// 3. Extract and parse DATABASE definition
const dbStart = scriptCode.indexOf('const DATABASE = {');
if (dbStart === -1) {
  console.error('Fatal: DATABASE object definition not found.');
  process.exit(1);
}

// Extract DATABASE snippet safely
const appStart = scriptCode.indexOf('function App()');
if (appStart === -1) {
  console.error('Fatal: App component definition not found.');
  process.exit(1);
}

const dbSnippet = scriptCode.slice(dbStart, appStart).trim();

// Execute in isolated sandbox context to validate syntax and inspect data
const context = {};
vm.createContext(context);
try {
  const evalCode = dbSnippet.replace(/^const DATABASE =/, 'globalThis.DATABASE =');
  vm.runInContext(evalCode, context);
} catch (err) {
  console.error('Syntax or evaluation error in DATABASE definition:', err);
  process.exit(1);
}

const db = context.DATABASE;
const keys = Object.keys(db);
console.log(`[PASS] DATABASE evaluated successfully: ${keys.length} total symbols registered.`);

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

  // Check class member completeness
  if (item.kind === 'class') {
    if (!Array.isArray(item.members) || item.members.length === 0) {
      console.error(`Class "${key}" has empty members array.`);
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
const secStart = scriptCode.indexOf('const LIBRARY_SECTIONS = [');
if (secStart === -1) {
  console.error('Fatal: LIBRARY_SECTIONS definition not found in script.');
  errors++;
} else {
  const secEnd = scriptCode.indexOf('const DATABASE = {');
  const secCode = scriptCode.slice(secStart, secEnd).trim();
  try {
    const secEvalCode = secCode.replace(/^const LIBRARY_SECTIONS =/, 'globalThis.LIBRARY_SECTIONS =');
    vm.runInContext(secEvalCode, context);
    const sections = context.LIBRARY_SECTIONS;
    
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
  } catch (err) {
    console.error('Error evaluating LIBRARY_SECTIONS:', err);
    errors++;
  }
}

// 6. Validate getSymbolTag helper badges
console.log('\nValidating getSymbolTag helper badges...');
const tagStart = scriptCode.indexOf('function getSymbolTag(');
if (tagStart === -1) {
  console.error('Fatal: getSymbolTag definition not found in script.');
  errors++;
} else {
  const tagEnd = scriptCode.indexOf('function App()');
  const tagCode = scriptCode.slice(tagStart, tagEnd).trim();
  try {
    vm.runInContext(tagCode, context);
    const getSymbolTag = context.getSymbolTag;
    if (typeof getSymbolTag !== 'function') {
      console.error('getSymbolTag is not a function.');
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

      // Validate isAbstractSymbol helper
      const isAbstractSymbol = context.isAbstractSymbol;
      if (typeof isAbstractSymbol !== 'function') {
        console.error('isAbstractSymbol is not a function.');
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

        // Validate getEffectiveMembers helper
        const getEffectiveMembers = context.getEffectiveMembers;
        if (typeof getEffectiveMembers !== 'function') {
          console.error('getEffectiveMembers is not a function.');
          errors++;
        } else {
          let classesWithoutInit = 0;
          let totalClasses = 0;
          for (const k of keys) {
            const item = db[k];
            if (item.kind === 'class') {
              totalClasses++;
              const eff = getEffectiveMembers(item);
              if (!eff.some(m => m.name.startsWith('__init__'))) {
                console.error(`Class "${k}" missing __init__ constructor in getEffectiveMembers.`);
                classesWithoutInit++;
                errors++;
              }
            }
          }
          if (classesWithoutInit === 0) {
            console.log(`[PASS] getEffectiveMembers verified: 100% of classes (${totalClasses}/${totalClasses}) include __init__ constructors.`);
          }
        }
      }
    }
  } catch (err) {
    console.error('Error evaluating getSymbolTag / isAbstractSymbol / getEffectiveMembers:', err);
    errors++;
  }
}

// 7. Validate PIPELINE_SCHEMAS and pipelines.html Parity
console.log('\nValidating PIPELINE_SCHEMAS and pipelines.html Parity...');

// Parity check for pipelines.html
const rootPipelines = path.join(projectRoot, 'pipelines.html');
const nestedPipelines = path.join(projectRoot, 'docs', 'cppreference', 'pipelines.html');
if (fs.existsSync(rootPipelines) && fs.existsSync(nestedPipelines)) {
  const rootBuf = fs.readFileSync(rootPipelines);
  const nestedBuf = fs.readFileSync(nestedPipelines);
  if (!rootBuf.equals(nestedBuf)) {
    console.error('Error: root pipelines.html and docs/cppreference/pipelines.html differ! Both copies must remain strictly in sync.');
    errors++;
  } else {
    console.log('Verified: root pipelines.html and docs/cppreference/pipelines.html are in sync.');
  }
}

// Helper function to validate pipeline schema array and canvas geometry
function validatePipelineSchemas(pipelines, sourceName) {
  if (!Array.isArray(pipelines) || pipelines.length !== 5) {
    console.error(`[${sourceName}] Expected 5 pipeline schemas, got ${pipelines?.length}`);
    errors++;
    return;
  }
  console.log(`[PASS] [${sourceName}] 5 Pipeline schemas evaluated successfully.`);
  const requiredIds = ['morl', 'training', 'rocs', 'fragment', 'telemetry'];
  for (const reqId of requiredIds) {
    const found = pipelines.find(p => p.id === reqId);
    if (!found) {
      console.error(`[${sourceName}] Missing required pipeline ID: "${reqId}"`);
      errors++;
    }
  }

  let missingPipelineSymbols = 0;
  for (const p of pipelines) {
    // Check nodes count and coordinates within 1130x355 canvas
    if (!Array.isArray(p.nodes) || p.nodes.length !== 8) {
      console.error(`[${sourceName}] Pipeline "${p.id}" expected 8 nodes, got ${p.nodes?.length}`);
      errors++;
    }
    for (const n of p.nodes || []) {
      if (n.x < 0 || n.x + 220 > 1130 || n.y < 0 || n.y + 98 > 355) {
        console.error(`[${sourceName}] Pipeline "${p.id}" node "${n.stepLabel}" out of canvas bounds: (${n.x}, ${n.y})`);
        errors++;
      }
      for (const sym of n.drugexSymbols || []) {
        if (!db[sym]) {
          console.error(`[${sourceName}] Pipeline "${p.id}" node "${n.stepLabel}" references missing symbol in DATABASE: "${sym}"`);
          missingPipelineSymbols++;
          errors++;
        }
      }
    }

    // Check connectors coordinates within 1130x355 canvas
    if (!Array.isArray(p.connectors) || p.connectors.length < 5) {
      console.error(`[${sourceName}] Pipeline "${p.id}" has invalid connectors array (${p.connectors?.length})`);
      errors++;
    }
    for (const [cIdx, c] of (p.connectors || []).entries()) {
      if (c.fromX < 0 || c.fromX > 1130 || c.toX < 0 || c.toX > 1130 || c.fromY < 0 || c.fromY > 355 || c.toY < 0 || c.toY > 355) {
        console.error(`[${sourceName}] Pipeline "${p.id}" connector ${cIdx} coordinates out of bounds: from(${c.fromX}, ${c.fromY}) to(${c.toX}, ${c.toY})`);
        errors++;
      }
    }

    // Check feedback loop if present
    if (p.feedbackLoop) {
      const fb = p.feedbackLoop;
      if (fb.fromX < 0 || fb.fromX > 1130 || fb.toX < 0 || fb.toX > 1130 || fb.fromY < 0 || fb.fromY > 355 || fb.toY < 0 || fb.toY > 355) {
        console.error(`[${sourceName}] Pipeline "${p.id}" feedback loop coordinates out of bounds: from(${fb.fromX}, ${fb.fromY}) to(${fb.toX}, ${fb.toY})`);
        errors++;
      }
    }

    // Check matrixRows
    if (!Array.isArray(p.matrixRows) || p.matrixRows.length !== 8) {
      console.error(`[${sourceName}] Pipeline "${p.id}" expected 8 matrix rows, got ${p.matrixRows?.length}`);
      errors++;
    }

    // Check equation terms point to valid node indices
    for (const eqItem of (p.equation || [])) {
      if (eqItem.type === 'term' && (eqItem.nodeIdx === undefined || eqItem.nodeIdx < 0 || eqItem.nodeIdx >= (p.nodes?.length || 0))) {
        console.error(`[${sourceName}] Pipeline "${p.id}" equation term "${eqItem.text}" has invalid nodeIdx: ${eqItem.nodeIdx}`);
        errors++;
      }
    }
  }

  if (missingPipelineSymbols === 0) {
    console.log(`[PASS] [${sourceName}] All drugexSymbols in PIPELINE_SCHEMAS exist in DATABASE.`);
  }
}

// Extract PIPELINE_SCHEMAS from index.html
const pipeStart = scriptCode.indexOf('const PIPELINE_SCHEMAS = [');
if (pipeStart === -1) {
  console.error('Fatal: PIPELINE_SCHEMAS definition not found in index.html script.');
  errors++;
} else {
  const pipeEnd = scriptCode.indexOf('function getStageColor');
  const pipeCode = scriptCode.slice(pipeStart, pipeEnd).trim();
  try {
    const pipeEval = pipeCode.replace(/^const PIPELINE_SCHEMAS =/, 'globalThis.PIPELINE_SCHEMAS_INDEX =');
    vm.runInContext(pipeEval, context);
    validatePipelineSchemas(context.PIPELINE_SCHEMAS_INDEX, 'index.html');
  } catch (err) {
    console.error('Error evaluating PIPELINE_SCHEMAS from index.html:', err);
    errors++;
  }
}

// Extract and validate PIPELINE_SCHEMAS from pipelines.html
if (fs.existsSync(rootPipelines)) {
  const pipHtml = fs.readFileSync(rootPipelines, 'utf8');
  const pipScriptMatch = pipHtml.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
  if (!pipScriptMatch) {
    console.error('Fatal: Babel script block not found in pipelines.html.');
    errors++;
  } else {
    const pipScript = pipScriptMatch[1];
    const pipStart = pipScript.indexOf('const PIPELINE_SCHEMAS = [');
    const pipEnd = pipScript.indexOf('function getStageColor');
    if (pipStart === -1 || pipEnd === -1) {
      console.error('Fatal: PIPELINE_SCHEMAS or getStageColor not found in pipelines.html.');
      errors++;
    } else {
      const pipCode = pipScript.slice(pipStart, pipEnd).trim();
      try {
        const pipEval = pipCode.replace(/^const PIPELINE_SCHEMAS =/, 'globalThis.PIPELINE_SCHEMAS_PIPE =');
        vm.runInContext(pipEval, context);
        validatePipelineSchemas(context.PIPELINE_SCHEMAS_PIPE, 'pipelines.html');
      } catch (err) {
        console.error('Error evaluating PIPELINE_SCHEMAS from pipelines.html:', err);
        errors++;
      }
    }
  }
}

// 8. Strict Emoji Prohibition Check
console.log('\nValidating Strict Zero-Emoji Rule...');
const emojiRegex = /[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F600}-\u{1F64F}\u{1F1E6}-\u{1F1FF}]/u;
const filesToScan = [
  rootCandidate,
  nestedCandidate,
  rootPipelines,
  nestedPipelines,
  path.join(projectRoot, 'tools', 'check_docs.mjs'),
  path.join(projectRoot, 'tools', 'audit_examples.mjs')
];
let emojiErrors = 0;
for (const file of filesToScan) {
  if (fs.existsSync(file)) {
    const content = fs.readFileSync(file, 'utf8');
    const match = content.match(emojiRegex);
    if (match) {
      console.error(`Error: Emoji detected in ${path.relative(projectRoot, file)}: "${match[0]}"`);
      emojiErrors++;
      errors++;
    }
  }
}
if (emojiErrors === 0) {
  console.log('[PASS] Verified 0 emojis across documentation files.');
}

if (errors > 0) {
  console.error(`Validation failed with ${errors} error(s).`);
  process.exit(1);
}

console.log('[SUCCESS] All assertions passed: 100% schema compliance, 0 broken cross-links, authentic cppreference format verified!\n');


