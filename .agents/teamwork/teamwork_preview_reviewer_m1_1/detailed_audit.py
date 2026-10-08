import json
import inspect
import sys
import os

# Add repo to sys.path
sys.path.insert(0, "/home/kolar/build_projects/drugex_edu/repo")

import drugex
from drugex.data.corpus.interfaces import Corpus, Vocabulary, SequenceVocabulary
from drugex.data.interfaces import DataSet, FragmentPairEncoder, DataSplitter, DataToLoader
from drugex.data.corpus.vocabulary import VocSmiles, VocNonGPT, VocGraph
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, GraphFragDataSet, TargetCreator, TgtData
from drugex.data.fragments import (
    Fragmenter, SequenceFragmentEncoder, GraphFragmentEncoder, FragmentCorpusEncoder,
    FragmentPairsSupplier, FragmentPairsEncodedSupplier, FragmentPairsSplitter, FragmentPairsCollector
)
from drugex.data.processing import Standardization, CorpusEncoder, RandomTrainTestSplitter, StandardizationCollector
from drugex.data.utils import getDataPaths, getVocPaths

from drugex.molecules.interfaces import Molecule, MolSupplier, BaseMolSupplier, ItemException
from drugex.molecules.mol import DrExMol, InitializationException
from drugex.molecules.suppliers import StandardizedSupplier, ListSupplier
from drugex.molecules.converters.interfaces import MolConverter
from drugex.molecules.converters.default import SmilesToDrEx, DrExToSMILES, RDKitToDrEx, Identity
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.molecules.converters.fragmenters import FragmenterWithSelectedFragment
from drugex.molecules.converters.standardizers import DefaultStandardizer, CleanSMILES
from drugex.molecules.converters.interfaces import ConversionException, StandardizationException

from drugex.parallel.interfaces import ParallelException, ResultCollector, ListCollector
from drugex.parallel.evaluator import ParallelProcessor, ParallelSupplierEvaluator
from drugex.parallel.collectors import ListExtend, ListAppend

with open('/tmp/m1_db.json') as f:
    db = json.load(f)

# Symbol mapping to python object
SYMBOL_MAP = {
    # Group 1
    "drugex::data::corpus::interfaces::Corpus": Corpus,
    "drugex::data::interfaces::DataSet": DataSet,
    "drugex::data::interfaces::FragmentPairEncoder": FragmentPairEncoder,
    "drugex::data::corpus::interfaces::Vocabulary": Vocabulary,
    "drugex::data::corpus::interfaces::SequenceVocabulary": SequenceVocabulary,
    "drugex::data::corpus::vocabulary::VocSmiles": VocSmiles,
    "drugex::data::corpus::vocabulary::VocNonGPT": VocNonGPT,
    "drugex::data::corpus::vocabulary::VocGraph": VocGraph,
    "drugex::data::corpus::SequenceCorpus": SequenceCorpus,

    # Group 2
    "drugex::data::datasets::SmilesDataSet": SmilesDataSet,
    "drugex::data::datasets::SmilesFragDataSet": SmilesFragDataSet,
    "drugex::data::datasets::GraphFragDataSet": GraphFragDataSet,
    "drugex::data::fragments::Fragmenter": Fragmenter,
    "drugex::data::fragments::SequenceFragmentEncoder": SequenceFragmentEncoder,
    "drugex::data::fragments::GraphFragmentEncoder": GraphFragmentEncoder,
    "drugex::data::fragments::FragmentCorpusEncoder": FragmentCorpusEncoder,
    "drugex::data::fragments::FragmentPairsSupplier": FragmentPairsSupplier,
    "drugex::data::fragments::FragmentPairsEncodedSupplier": FragmentPairsEncodedSupplier,
    "drugex::data::fragments::FragmentPairsSplitter": FragmentPairsSplitter,
    "drugex::data::processing::Standardization": Standardization,
    "drugex::data::processing::CorpusEncoder": CorpusEncoder,
    "drugex::data::processing::RandomTrainTestSplitter": RandomTrainTestSplitter,
    "drugex::data::interfaces::DataSplitter": DataSplitter,
    "drugex::data::interfaces::DataToLoader": DataToLoader,

    # Group 3
    "drugex::molecules::interfaces::Molecule": Molecule,
    "drugex::molecules::interfaces::MolSupplier": MolSupplier,
    "drugex::molecules::interfaces::BaseMolSupplier": BaseMolSupplier,
    "drugex::molecules::converters::MolConverter": MolConverter,
    "drugex::molecules::converters::DefaultStandardizer": DefaultStandardizer,
    "drugex::molecules::converters::CleanSMILES": CleanSMILES,
    "drugex::molecules::converters::dummyMolsFromFragments": dummyMolsFromFragments,
    "drugex::molecules::converters::default::SmilesToDrEx": SmilesToDrEx,
    "drugex::molecules::converters::default::DrExToSMILES": DrExToSMILES,
    "drugex::molecules::converters::default::RDKitToDrEx": RDKitToDrEx,
    "drugex::molecules::converters::default::Identity": Identity,
    "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment": FragmenterWithSelectedFragment,
    "drugex::molecules::suppliers::StandardizedSupplier": StandardizedSupplier,
    "drugex::molecules::suppliers::ListSupplier": ListSupplier,
    "drugex::molecules::mol::DrExMol": DrExMol,

    # Group 4
    "drugex::parallel::ParallelProcessor": ParallelProcessor,
    "drugex::parallel::collectors::ListExtend": ListExtend,
    "drugex::parallel::ParallelSupplierEvaluator": ParallelSupplierEvaluator,
    "drugex::parallel::interfaces::ResultCollector": ResultCollector,
    "drugex::parallel::interfaces::ListCollector": ListCollector,
    "drugex::parallel::collectors::ListAppend": ListAppend,

    # Group 5
    "drugex::molecules::converters::ConversionException": ConversionException,
    "drugex::molecules::converters::StandardizationException": StandardizationException,
    "drugex::molecules::interfaces::ItemException": ItemException,
    "drugex::molecules::mol::InitializationException": InitializationException,
    "drugex::parallel::interfaces::ParallelException": ParallelException,
    "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException": FragmentPairsEncodedSupplier.FragmentEncodingException,
    "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException": FragmentPairsEncodedSupplier.MoleculeEncodingException,

    # Group 6
    "drugex::data::fragments::FragmentPairsCollector": FragmentPairsCollector,
    "drugex::data::datasets::TargetCreator": TargetCreator,
    "drugex::data::datasets::TgtData": TgtData,
    "drugex::data::processing::StandardizationCollector": StandardizationCollector,
    "drugex::data::utils::getDataPaths": getDataPaths,
    "drugex::data::utils::getVocPaths": getVocPaths,
    "drugex::data::processing::Standardization::Collector": StandardizationCollector,
}

print(f"Auditing {len(SYMBOL_MAP)} mapped symbols against Python AST/runtime...")

results = []
for key, obj in SYMBOL_MAP.items():
    entry = db[key]
    status = {"key": key, "issues": []}
    
    # 1. Check if class or function
    is_cls = inspect.isclass(obj)
    is_func = inspect.isfunction(obj)
    
    # Check kind in docs
    kind = entry.get("kind", "")
    if is_cls and kind not in ["class", "interface"]:
        status["issues"].append(f"Expected kind 'class'/'interface', found '{kind}'")
    elif is_func and kind != "function":
        status["issues"].append(f"Expected kind 'function', found '{kind}'")

    # 2. Check constructor / function signature parameters
    try:
        sig = inspect.signature(obj)
        py_params = list(sig.parameters.keys())
        if is_cls and "self" in py_params:
            py_params.remove("self")
        
        doc_params = [p["name"] for p in entry.get("parameters", [])]
        
        # Check parameter alignment
        # Some classes inherit __init__ without overriding or take *args, **kwargs
        status["py_params"] = py_params
        status["doc_params"] = doc_params
        
    except Exception as e:
        status["sig_error"] = str(e)

    # 3. For classes, check members
    if is_cls:
        doc_members = [m["name"] for m in entry.get("members", [])]
        py_methods = [name for name, val in inspect.getmembers(obj, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)) if not name.startswith("_") or name.startswith("__")]
        status["doc_members"] = doc_members
        
        # Check if all doc members actually exist on the class
        for dm in doc_members:
            if not hasattr(obj, dm):
                status["issues"].append(f"Member '{dm}' documented in entry does NOT exist on Python class {obj.__name__}")
                
    # 4. Check description & notes quality
    desc = entry.get("description", "")
    notes = entry.get("notes", [])
    if len(desc.split()) < 15:
        status["issues"].append(f"Description too short ({len(desc.split())} words)")
    if len(notes) < 2:
        status["issues"].append(f"Notes too sparse ({len(notes)} notes)")
        
    results.append(status)

print("\n--- AUDIT SUMMARY ---")
issue_count = sum(len(r["issues"]) for r in results)
print(f"Total issues found across 59 symbols: {issue_count}")
for r in results:
    if r["issues"]:
        print(f"\n[ISSUE] {r['key']}:")
        for iss in r["issues"]:
            print(f"  - {iss}")

with open('/tmp/m1_audit_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDetailed results saved to /tmp/m1_audit_results.json")
