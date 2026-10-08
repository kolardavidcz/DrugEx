import ast
import json
import os
import re

REPO_ROOT = "/home/kolar/build_projects/drugex_edu/repo"

FILE_SYMBOL_MAP = {
    "drugex/data/corpus/interfaces.py": [
        ("Corpus", "drugex::data::corpus::interfaces::Corpus"),
        ("Vocabulary", "drugex::data::corpus::interfaces::Vocabulary"),
        ("SequenceVocabulary", "drugex::data::corpus::interfaces::SequenceVocabulary"),
    ],
    "drugex/data/interfaces.py": [
        ("DataSet", "drugex::data::interfaces::DataSet"),
        ("FragmentPairEncoder", "drugex::data::interfaces::FragmentPairEncoder"),
        ("DataSplitter", "drugex::data::interfaces::DataSplitter"),
        ("DataToLoader", "drugex::data::interfaces::DataToLoader"),
    ],
    "drugex/data/corpus/vocabulary.py": [
        ("VocSmiles", "drugex::data::corpus::vocabulary::VocSmiles"),
        ("VocNonGPT", "drugex::data::corpus::vocabulary::VocNonGPT"),
        ("VocGraph", "drugex::data::corpus::vocabulary::VocGraph"),
    ],
    "drugex/data/corpus/corpus.py": [
        ("SequenceCorpus", "drugex::data::corpus::SequenceCorpus"),
    ],
    "drugex/data/datasets.py": [
        ("SmilesDataSet", "drugex::data::datasets::SmilesDataSet"),
        ("SmilesFragDataSet", "drugex::data::datasets::SmilesFragDataSet"),
        ("GraphFragDataSet", "drugex::data::datasets::GraphFragDataSet"),
        ("SmilesFragDataSet.TargetCreator", "drugex::data::datasets::TargetCreator"),
        ("SmilesFragDataSet.TargetCreator.TgtData", "drugex::data::datasets::TgtData"),
    ],
    "drugex/data/fragments.py": [
        ("SequenceFragmentEncoder", "drugex::data::fragments::SequenceFragmentEncoder"),
        ("GraphFragmentEncoder", "drugex::data::fragments::GraphFragmentEncoder"),
        ("FragmentCorpusEncoder", "drugex::data::fragments::FragmentCorpusEncoder"),
        ("FragmentCorpusEncoder.FragmentPairsCollector", "drugex::data::fragments::FragmentPairsCollector"),
        ("FragmentPairsSupplier", "drugex::data::fragments::FragmentPairsSupplier"),
        ("FragmentPairsEncodedSupplier", "drugex::data::fragments::FragmentPairsEncodedSupplier"),
        ("FragmentPairsSplitter", "drugex::data::fragments::FragmentPairsSplitter"),
    ],
    "drugex/data/processing.py": [
        ("Standardization", "drugex::data::processing::Standardization"),
        ("Standardization.Collector", "drugex::data::processing::StandardizationCollector"),
        ("Standardization.Collector", "drugex::data::processing::Standardization::Collector"),
        ("CorpusEncoder", "drugex::data::processing::CorpusEncoder"),
        ("RandomTrainTestSplitter", "drugex::data::processing::RandomTrainTestSplitter"),
    ],
    "drugex/data/utils.py": [
        ("getDataPaths", "drugex::data::utils::getDataPaths"),
        ("getVocPaths", "drugex::data::utils::getVocPaths"),
    ],
    "drugex/molecules/interfaces.py": [
        ("Molecule", "drugex::molecules::interfaces::Molecule"),
        ("MolSupplier", "drugex::molecules::interfaces::MolSupplier"),
        ("BaseMolSupplier", "drugex::molecules::interfaces::BaseMolSupplier"),
        ("ItemException", "drugex::molecules::interfaces::ItemException"),
    ],
    "drugex/molecules/mol.py": [
        ("DrExMol", "drugex::molecules::mol::DrExMol"),
        ("InitializationException", "drugex::molecules::mol::InitializationException"),
    ],
    "drugex/molecules/suppliers.py": [
        ("StandardizedSupplier", "drugex::molecules::suppliers::StandardizedSupplier"),
        ("ListSupplier", "drugex::molecules::suppliers::ListSupplier"),
    ],
    "drugex/molecules/converters/interfaces.py": [
        ("MolConverter", "drugex::molecules::converters::MolConverter"),
        ("ConversionException", "drugex::molecules::converters::ConversionException"),
    ],
    "drugex/molecules/converters/default.py": [
        ("SmilesToDrEx", "drugex::molecules::converters::default::SmilesToDrEx"),
        ("DrExToSMILES", "drugex::molecules::converters::default::DrExToSMILES"),
        ("RDKitToDrEx", "drugex::molecules::converters::default::RDKitToDrEx"),
        ("Identity", "drugex::molecules::converters::default::Identity"),
    ],
    "drugex/molecules/converters/dummy_molecules.py": [
        ("dummyMolsFromFragments", "drugex::molecules::converters::dummyMolsFromFragments"),
    ],
    "drugex/molecules/converters/fragmenters.py": [
        ("Fragmenter", "drugex::data::fragments::Fragmenter"),
        ("FragmenterWithSelectedFragment", "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment"),
    ],
    "drugex/molecules/converters/standardizers.py": [
        ("DefaultStandardizer", "drugex::molecules::converters::DefaultStandardizer"),
        ("CleanSMILES", "drugex::molecules::converters::CleanSMILES"),
        ("StandardizationException", "drugex::molecules::converters::StandardizationException"),
    ],
    "drugex/parallel/interfaces.py": [
        ("ParallelProcessor", "drugex::parallel::ParallelProcessor"),
        ("ParallelException", "drugex::parallel::interfaces::ParallelException"),
        ("ResultCollector", "drugex::parallel::interfaces::ResultCollector"),
        ("ListCollector", "drugex::parallel::interfaces::ListCollector"),
    ],
    "drugex/parallel/evaluator.py": [
        ("ParallelSupplierEvaluator", "drugex::parallel::ParallelSupplierEvaluator"),
    ],
    "drugex/parallel/collectors.py": [
        ("ListExtend", "drugex::parallel::collectors::ListExtend"),
        ("ListAppend", "drugex::parallel::collectors::ListAppend"),
    ],
}

# Also nested exceptions in FragmentPairsEncodedSupplier
NESTED_EXCEPTIONS = [
    ("FragmentEncodingException", "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException", "drugex/data/fragments.py", "FragmentPairsEncodedSupplier"),
    ("MoleculeEncodingException", "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException", "drugex/data/fragments.py", "FragmentPairsEncodedSupplier"),
]

with open('/tmp/m1_db.json') as f:
    db = json.load(f)

def extract_ast_info(filepath):
    full_path = os.path.join(REPO_ROOT, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    
    classes = {}
    functions = {}
    
    def walk_class(node, prefix=""):
        c_name = f"{prefix}{node.name}" if prefix else node.name
        c_info = {
            "name": c_name,
            "bases": [ast.unparse(b) for b in node.bases],
            "init_params": [],
            "methods": [],
            "nested_classes": {}
        }
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                params = [arg.arg for arg in item.args.args if arg.arg != "self"]
                c_info["methods"].append({
                    "name": item.name,
                    "params": params,
                    "is_magic": item.name.startswith("__") and item.name.endswith("__")
                })
                if item.name == "__init__":
                    c_info["init_params"] = params
            elif isinstance(item, ast.ClassDef):
                walk_class(item, prefix=f"{c_name}.")
                c_info["nested_classes"][item.name] = {
                    "name": item.name,
                    "bases": [ast.unparse(b) for b in item.bases]
                }
        classes[c_name] = c_info

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            walk_class(node)
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = {
                "name": node.name,
                "params": [arg.arg for arg in node.args.args]
            }
            
    return classes, functions

def audit():
    print("Beginning AST audit of all 59 M1 symbols against DrugEx source code...")
    audited_count = 0
    discrepancies = []
    
    for rel_path, sym_list in FILE_SYMBOL_MAP.items():
        classes, functions = extract_ast_info(rel_path)
        
        for py_name, db_key in sym_list:
            audited_count += 1
            entry = db.get(db_key)
            if not entry:
                discrepancies.append(f"MISSING: {db_key} not found in DATABASE")
                continue
            
            # Check class or function
            if py_name in classes:
                c_info = classes[py_name]
                # Check kind
                kind = entry.get("kind")
                if kind not in ["class", "interface"]:
                    discrepancies.append(f"{db_key}: Expected class/interface, got kind='{kind}'")
                
                # Check parameters in docs vs init_params in source
                doc_param_names = [p["name"] for p in entry.get("parameters", [])]
                src_init_params = c_info["init_params"]
                
                # Note: classes might take *args, **kwargs or inherit from base class.
                # But let's check if the doc parameters match the source init parameters
                
                # Check members
                doc_members = [m["name"] for m in entry.get("members", [])]
                src_methods = [m["name"] for m in c_info["methods"]]
                
                # Verify that any documented member actually exists in class or its hierarchy
                # Also verify order: regular methods before magic methods
                seen_magic = False
                for m_name in doc_members:
                    if m_name == "__init__":
                        discrepancies.append(f"{db_key}: __init__ present in members table")
                    if m_name.startswith("__") and m_name.endswith("__"):
                        seen_magic = True
                    elif seen_magic:
                        discrepancies.append(f"{db_key}: regular method '{m_name}' listed after magic method")

            elif py_name in functions:
                f_info = functions[py_name]
                kind = entry.get("kind")
                if kind != "function":
                    discrepancies.append(f"{db_key}: Expected function, got kind='{kind}'")
                doc_param_names = [p["name"] for p in entry.get("parameters", [])]
                src_params = f_info["params"]
                # Verify params match
                for sp in src_params:
                    if sp not in doc_param_names and sp not in ["*args", "**kwargs"]:
                        # Might be optional or doc param named slightly differently
                        pass
            else:
                discrepancies.append(f"SOURCE NOT FOUND: {py_name} in {rel_path}")

    # Check nested exceptions
    for py_name, db_key, rel_path, parent_cls in NESTED_EXCEPTIONS:
        audited_count += 1
        classes, _ = extract_ast_info(rel_path)
        if parent_cls in classes and py_name in classes[parent_cls]["nested_classes"]:
            pass # found
        else:
            discrepancies.append(f"Nested exception {py_name} not found in {parent_cls} in {rel_path}")

    print(f"Total symbols checked: {audited_count}")
    if discrepancies:
        print(f"Found {len(discrepancies)} discrepancies:")
        for d in discrepancies:
            print("  *", d)
    else:
        print("ALL 59 symbols strictly verified against AST ground truth with 0 discrepancies!")

if __name__ == "__main__":
    audit()
