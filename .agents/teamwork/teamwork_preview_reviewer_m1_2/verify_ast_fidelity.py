import ast
import json
import os
import re

# Load DATABASE from docs/index.html
with open('/home/kolar/build_projects/drugex_edu/docs/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

m = re.search(r'const DATABASE = (\{[\s\S]*?\n\s*\});\n\s*const', html)
if not m:
    # Try another pattern
    db_start = html.find('const DATABASE = {')
    app_start = html.find('function App()')
    db_code = html[db_start + len('const DATABASE = '):app_start].strip()
    if db_code.endswith(';'):
        db_code = db_code[:-1].strip()
else:
    db_code = m.group(1)

# Alternatively, parse via node output to JSON
import subprocess
json_str = subprocess.check_output([
    'node', '-e',
    """
    const fs = require('fs');
    const vm = require('vm');
    const html = fs.readFileSync('/home/kolar/build_projects/drugex_edu/docs/index.html', 'utf8');
    const scriptMatch = html.match(/<script type="text\\/babel">([\\s\\S]*?)<\\/script>/);
    const scriptCode = scriptMatch[1];
    const dbStart = scriptCode.indexOf('const DATABASE = {');
    const appStart = scriptCode.indexOf('function App()');
    const dbCode = scriptCode.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');
    const context = {};
    vm.createContext(context);
    vm.runInContext(dbCode, context);
    console.log(JSON.stringify(context.DATABASE));
    """
], universal_newlines=True)

db = json.loads(json_str)

m1_symbols = [
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
]

repo_root = '/home/kolar/build_projects/drugex_edu/repo'

print(f"Auditing {len(m1_symbols)} symbols against Python AST...")
discrepancies = []

for sym in m1_symbols:
    entry = db[sym]
    mod_path = entry.get('module', '')
    title = entry.get('title', '')
    kind = entry.get('kind', '')
    
    # Map module to file path
    # e.g., drugex.data.corpus.vocabulary -> repo/drugex/data/corpus/vocabulary.py
    rel_path = mod_path.replace('.', '/')
    py_path = os.path.join(repo_root, rel_path + '.py')
    if not os.path.exists(py_path):
        # try __init__.py in package
        py_path = os.path.join(repo_root, rel_path, '__init__.py')
        
    if not os.path.exists(py_path):
        discrepancies.append((sym, f"Module file not found: {py_path}"))
        continue
        
    with open(py_path, 'r', encoding='utf-8') as f:
        src = f.read()
    
    try:
        tree = ast.parse(src)
    except Exception as e:
        discrepancies.append((sym, f"Could not parse Python AST in {py_path}: {e}"))
        continue

    # Look for the class or function
    class_name = title.split('.')[-1]
    found_node = None
    
    if '::' in title:
        # nested class, e.g. Standardization::Collector
        parts = title.split('::')
        parent_class = parts[0].split('.')[-1]
        inner_class = parts[1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == parent_class:
                for inner in node.body:
                    if isinstance(inner, ast.ClassDef) and inner.name == inner_class:
                        found_node = inner
                        break
    elif '.' in title and not title.startswith('drugex'):
        # nested class, e.g. FragmentPairsEncodedSupplier.FragmentEncodingException
        parts = title.split('.')
        parent_class = parts[0]
        inner_class = parts[1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == parent_class:
                for inner in node.body:
                    if isinstance(inner, ast.ClassDef) and inner.name == inner_class:
                        found_node = inner
                        break
    else:
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == class_name:
                found_node = node
                break

    if found_node is None:
        # Check if imported from another module or defined dynamically
        # Look for import alias
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == class_name or alias.asname == class_name:
                        found_node = 'imported'
                        break
        if not found_node:
            discrepancies.append((sym, f"Symbol {class_name} not found in {py_path}"))
            continue

    # If class, check __init__ arguments
    if isinstance(found_node, ast.ClassDef):
        init_node = None
        for n in found_node.body:
            if isinstance(n, ast.FunctionDef) and n.name == '__init__':
                init_node = n
                break
        if init_node:
            ast_params = [a.arg for a in init_node.args.args if a.arg != 'self']
            doc_params = [p['name'] for p in entry.get('parameters', [])]
            # Check if all doc_params are in ast_params or *args/**kwargs
            has_varargs = init_node.args.vararg is not None or init_node.args.kwarg is not None
            for dp in doc_params:
                if dp not in ast_params and not has_varargs:
                    discrepancies.append((sym, f"Doc param '{dp}' not in AST params {ast_params}"))

print(f"\nDiscrepancies found: {len(discrepancies)}")
for d in discrepancies:
    print(f"  - [{d[0]}]: {d[1]}")
