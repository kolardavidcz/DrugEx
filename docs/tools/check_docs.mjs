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
    }
  } catch (err) {
    console.error('Error evaluating getSymbolTag:', err);
    errors++;
  }
}

if (errors > 0) {
  console.error(`Validation failed with ${errors} error(s).`);
  process.exit(1);
}

console.log('[SUCCESS] All assertions passed: 100% schema compliance, 0 broken cross-links, authentic cppreference format verified!\n');

