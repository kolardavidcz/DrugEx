import re
import json
import subprocess
import sys
import ast

INDEX_HTML = "/home/kolar/build_projects/drugex_edu/docs/index.html"

# Groups from worker handoff:
GROUPS = {
    "Group 1": [
        "drugex::data::corpus::interfaces::Corpus",
        "drugex::data::interfaces::DataSet",
        "drugex::data::interfaces::FragmentPairEncoder",
        "drugex::data::corpus::interfaces::Vocabulary",
        "drugex::data::corpus::interfaces::SequenceVocabulary",
        "drugex::data::corpus::vocabulary::VocSmiles",
        "drugex::data::corpus::vocabulary::VocNonGPT",
        "drugex::data::corpus::vocabulary::VocGraph",
        "drugex::data::corpus::SequenceCorpus"
    ],
    "Group 2": [
        "drugex::data::datasets::SmilesDataSet",
        "drugex::data::datasets::SmilesFragDataSet",
        "drugex::data::datasets::GraphFragDataSet",
        "drugex::data::fragments::Fragmenter",
        "drugex::data::fragments::SequenceFragmentEncoder",
        "drugex::data::fragments::GraphFragmentEncoder",
        "drugex::data::fragments::FragmentCorpusEncoder",
        "drugex::data::fragments::FragmentPairsSupplier",
        "drugex::data::fragments::FragmentPairsEncodedSupplier",
        "drugex::data::fragments::FragmentPairsSplitter",
        "drugex::data::processing::Standardization",
        "drugex::data::processing::CorpusEncoder",
        "drugex::data::processing::RandomTrainTestSplitter",
        "drugex::data::interfaces::DataSplitter",
        "drugex::data::interfaces::DataToLoader"
    ],
    "Group 3": [
        "drugex::molecules::interfaces::Molecule",
        "drugex::molecules::interfaces::MolSupplier",
        "drugex::molecules::interfaces::BaseMolSupplier",
        "drugex::molecules::converters::MolConverter",
        "drugex::molecules::converters::DefaultStandardizer",
        "drugex::molecules::converters::CleanSMILES",
        "drugex::molecules::converters::dummyMolsFromFragments",
        "drugex::molecules::converters::default::SmilesToDrEx",
        "drugex::molecules::converters::default::DrExToSMILES",
        "drugex::molecules::converters::default::RDKitToDrEx",
        "drugex::molecules::converters::default::Identity",
        "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment",
        "drugex::molecules::suppliers::StandardizedSupplier",
        "drugex::molecules::suppliers::ListSupplier",
        "drugex::molecules::mol::DrExMol"
    ],
    "Group 4": [
        "drugex::parallel::ParallelProcessor",
        "drugex::parallel::collectors::ListExtend",
        "drugex::parallel::ParallelSupplierEvaluator",
        "drugex::parallel::interfaces::ResultCollector",
        "drugex::parallel::interfaces::ListCollector",
        "drugex::parallel::collectors::ListAppend"
    ],
    "Group 5": [
        "drugex::molecules::converters::ConversionException",
        "drugex::molecules::converters::StandardizationException",
        "drugex::molecules::interfaces::ItemException",
        "drugex::molecules::mol::InitializationException",
        "drugex::parallel::interfaces::ParallelException",
        "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException",
        "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException"
    ],
    "Group 6": [
        "drugex::data::fragments::FragmentPairsCollector",
        "drugex::data::datasets::TargetCreator",
        "drugex::data::datasets::TgtData",
        "drugex::data::processing::StandardizationCollector",
        "drugex::data::utils::getDataPaths",
        "drugex::data::utils::getVocPaths",
        "drugex::data::processing::Standardization::Collector"
    ]
}

def main():
    print("Verifying 59 M1 symbols...")
    all_symbols = []
    for grp, syms in GROUPS.items():
        all_symbols.extend(syms)
    print(f"Total symbols specified in M1 groups: {len(all_symbols)}")

    # Extract DATABASE from index.html using node vm or regex
    # We can invoke node to dump the JSON of DATABASE for these symbols
    cmd = [
        "node", "-e",
        """
        const fs = require('fs');
        const vm = require('vm');
        const html = fs.readFileSync('""" + INDEX_HTML + """', 'utf8');
        const scriptMatch = html.match(/<script type="text\\/babel">([\\s\\S]*?)<\\/script>/);
        const scriptCode = scriptMatch[1];
        const dbStart = scriptCode.indexOf('const DATABASE = {');
        const appStart = scriptCode.indexOf('function App()');
        const dbSnippet = scriptCode.slice(dbStart, appStart).trim();
        const context = {};
        vm.createContext(context);
        const evalCode = dbSnippet.replace(/^const DATABASE =/, 'globalThis.DATABASE =');
        vm.runInContext(evalCode, context);
        fs.writeFileSync('/tmp/m1_db.json', JSON.stringify(context.DATABASE));
        """
    ]
    subprocess.check_call(cmd)
    with open('/tmp/m1_db.json') as f:
        db = json.load(f)

    print(f"Total symbols in DATABASE: {len(db)}")
    
    missing = [s for s in all_symbols if s not in db]
    if missing:
        print(f"MISSING symbols in DATABASE: {missing}")
        sys.exit(1)
    else:
        print("All 59 symbols exist in DATABASE.")

    # Check each symbol for requirements
    errors = []
    magic_method_pattern = re.compile(r'^__(?:iter|getitem|setitem|len|call|contains|str|repr|eq|ne|enter|exit|next)__$')
    
    for s in all_symbols:
        entry = db[s]
        # 1. Synopsis check
        syn = entry.get("synopsis", "")
        if not syn:
            errors.append(f"{s}: synopsis missing")
        
        # 2. Description check
        desc = entry.get("description", "")
        if not desc or len(desc) < 40:
            errors.append(f"{s}: description too short or empty ({len(desc)} chars)")
        
        # 3. Parameters check
        params = entry.get("parameters", [])
        for p in params:
            if not p.get("name") or not p.get("type") or not p.get("desc"):
                errors.append(f"{s}: parameter {p.get('name')} missing fields")
        
        # 4. Members check:
        # - NO __init__
        # - Regular methods strictly precede magic methods
        members = entry.get("members", [])
        seen_magic = False
        for m in members:
            mname = m.get("name", "")
            if mname == "__init__":
                errors.append(f"{s}: __init__ found in members table!")
            is_magic = bool(magic_method_pattern.match(mname)) or (mname.startswith("__") and mname.endswith("__"))
            if is_magic:
                seen_magic = True
            elif seen_magic:
                errors.append(f"{s}: regular method {mname} appears after magic method!")
        
        # 5. Example check:
        ex = entry.get("example", {})
        code = ex.get("code", "")
        if not code:
            errors.append(f"{s}: example code is empty")
        else:
            # Check if example is valid Python syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"{s}: example code has syntax error: {e}")
        
        out = ex.get("output", "")
        if not out:
            errors.append(f"{s}: example output is empty")

    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:20]:
            print("  -", err)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more.")
    else:
        print("All 59 symbols passed schema, method-ordering, and example syntax checks!")

if __name__ == "__main__":
    main()
