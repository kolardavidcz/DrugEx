#!/usr/bin/env python3
"""
scripts/extract_git_stubs.py
Deterministic AST-based Type Stub (.pyi) Generator for DrugEx.

Extracts authoritative docstrings and type annotations from git history
(branches/commits: origin/rocs-scoring-numpy-docstrings, 061f778a, 8eadb99b,
v3.4.8, origin/education database) and aligns them against current DrugEx
runtime signatures to emit PEP 484/561 compliant .pyi files.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


HISTORICAL_COMMITS = [
    "14ea673c",  # origin/rocs-scoring-numpy-docstrings (comprehensive numpy docstrings)
    "061f778a",  # pre-55b960bc snapshot (authoritative pre-streamlined docstrings across 52 files)
    "8eadb99b",  # production ROCS scorers & conformers
    "v3.4.8",    # upstream tag 3.4.8
]

TYPE_MAPPINGS = {
    "list of str": "List[str]",
    "list of string": "List[str]",
    "list of smiles": "List[str]",
    "list of mol": "List[Chem.Mol]",
    "list of chem.mol": "List[Chem.Mol]",
    "list of int": "List[int]",
    "list of float": "List[float]",
    "list of lists": "List[List[Any]]",
    "list": "List[Any]",
    "dict": "Dict[str, Any]",
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "bool": "bool",
    "boolean": "bool",
    "dataframe": "pd.DataFrame",
    "pd.dataframe": "pd.DataFrame",
    "ndarray": "np.ndarray",
    "np.ndarray": "np.ndarray",
    "np.array": "np.ndarray",
    "numpy.array": "np.ndarray",
    "tensor": "torch.Tensor",
    "torch.tensor": "torch.Tensor",
    "none": "None",
}

COMMON_TYPE_IMPORTS = {
    "DataLoader": "from torch.utils.data import DataLoader",
    "Dataset": "from torch.utils.data import Dataset",
    "DataSet": "from drugex.data.interfaces import DataSet",
    "Namespace": "from argparse import Namespace",
    "ExplicitBitVect": "from rdkit.DataStructs import ExplicitBitVect",
    "RingInfo": "from rdkit.Chem import RingInfo",
    "Sized": "from typing import Sized",
    "Scorer": "from drugex.training.scorers.interfaces import Scorer",
    "ScoreModifier": "from drugex.training.scorers.interfaces import ScoreModifier",
    "Environment": "from drugex.training.interfaces import Environment",
    "RewardScheme": "from drugex.training.rewards import RewardScheme",
    "ParetoRewardScheme": "from drugex.training.rewards import ParetoRewardScheme",
    "Vocabulary": "from drugex.data.corpus.interfaces import Vocabulary",
    "SequenceVocabulary": "from drugex.data.corpus.interfaces import SequenceVocabulary",
    "VocGraph": "from drugex.data.corpus.vocabulary import VocGraph",
    "VocSmiles": "from drugex.data.corpus.vocabulary import VocSmiles",
    "ResultCollector": "from drugex.parallel.interfaces import ResultCollector",
    "MolConverter": "from drugex.molecules.converters.interfaces import MolConverter",
    "MolSupplier": "from drugex.molecules.interfaces import MolSupplier",
    "DataSplitter": "from drugex.data.interfaces import DataSplitter",
    "FragmentPairEncoder": "from drugex.data.interfaces import FragmentPairEncoder",
    "ParallelProcessor": "from drugex.parallel.interfaces import ParallelProcessor",
    "Explorer": "from drugex.training.explorers.interfaces import Explorer",
    "SequenceRNN": "from drugex.training.generators.sequence_rnn import SequenceRNN",
    "SequenceTransformer": "from drugex.training.generators.sequence_transformer import SequenceTransformer",
    "GraphTransformer": "from drugex.training.generators.graph_transformer import GraphTransformer",
    "TrainingMonitor": "from drugex.training.monitors import TrainingMonitor",
    "Monitor": "from drugex.training.monitors import TrainingMonitor as Monitor",
    "FileMonitor": "from drugex.training.monitors import FileMonitor",
    "NullMonitor": "from drugex.training.monitors import NullMonitor",
    "PeakRecoveryMonitor": "from drugex.training.monitors import PeakRecoveryMonitor",
    "Generator": "from drugex.training.generators.interfaces import Generator",
    "Model": "from drugex.training.interfaces import Model",
    "ModelEvaluator": "from drugex.training.interfaces import ModelEvaluator",
}

MODULE_SPECIFIC_IMPORTS = {
    "drugex/molecules/suppliers.py": [
        "from drugex.molecules.converters.interfaces import MolConverter",
    ],
    "drugex/training/explorers/interfaces.py": [
        "from drugex.training.interfaces import Environment, Model, ModelProvider, TrainingMonitor",
        "from drugex.training.generators.interfaces import Generator",
    ],
    "drugex/training/explorers/sequence_explorer.py": [
        "from drugex.training.generators.sequence_rnn import SequenceRNN",
        "from drugex.training.interfaces import Environment",
        "from drugex.training.monitors import TrainingMonitor as Monitor",
    ],
    "drugex/training/explorers/frag_sequence_explorer.py": [
        "from drugex.training.generators.sequence_transformer import SequenceTransformer",
        "from drugex.training.interfaces import Environment",
    ],
    "drugex/training/explorers/frag_graph_explorer.py": [
        "from drugex.training.generators.graph_transformer import GraphTransformer",
        "from drugex.training.interfaces import Environment",
    ],
    "drugex/train.py": [
        "from drugex.training.generators.interfaces import Generator",
        "from drugex.training.explorers.interfaces import Explorer",
        "from drugex.training.monitors import TrainingMonitor",
        "from drugex.data.corpus.interfaces import Vocabulary",
    ],
    "drugex/training/monitors.py": [
        "from drugex.training.interfaces import TrainingMonitor as TrainingMonitor, Model as Model",
    ],
    "drugex/training/scorers/modifiers.py": [
        "ScoreType = Union[float, np.ndarray, Sequence[float]]",
    ],
    "drugex/training/scorers/qsprpred.py": [
        "QSPRModel: Any = ...",
    ],
    "drugex/training/scorers/conformer_generators.py": [
        "OE_AVAILABLE: bool = ...",
        "CDPL_AVAILABLE: bool = ...",
        "oemolprop: Any = ...",
    ],
    "drugex/training/scorers/rocs_rdkit.py": [
        "OE_AVAILABLE: bool = ...",
        "CDPL_AVAILABLE: bool = ...",
    ],
    "drugex/training/scorers/rocs_cdpkit.py": [
        "CDPL_AVAILABLE: bool = ...",
    ],
    "drugex/training/scorers/rocs_openeye.py": [
        "OE_AVAILABLE: bool = ...",
    ],
    "drugex/data/corpus/interfaces.py": [
        "T_Voc = TypeVar('T_Voc', bound=Vocabulary)",
    ],
    "drugex/utils/gcmol.py": [
        "T = TypeVar('T')",
    ],
    "drugex/molecules/interfaces.py": [
        "from drugex.molecules.converters.interfaces import MolConverter",
    ],
}

KNOWN_OVERRIDE_METHODS = {
    ("VocNonGPT", "decode"),
    ("VocNonGPT", "fromFile"),
    ("VocGraph", "fromFile"),
    ("Standardizer", "apply"),
    ("Standardization", "apply"),
    ("FragmentCorpus", "apply"),
    ("GraphFragCorpus", "apply"),
    ("ParallelProcessor", "apply"),
    ("ParallelEvaluator", "apply"),
    ("ParallelSupplierEvaluator", "apply"),
    ("Explorer", "fit"),
    ("Explorer", "attachToGPUs"),
    ("Explorer", "getModel"),
    ("FragExplorer", "fit"),
    ("SequenceExplorer", "fit"),
    ("FragSequenceExplorer", "fit"),
    ("FragSequenceExplorer", "getBatchOutputs"),
    ("FragGraphExplorer", "fit"),
    ("FragGraphExplorer", "getBatchOutputs"),
    ("Generator", "fit"),
    ("SequenceTransformer", "validateNet"),
    ("SequenceTransformer", "sample"),
    ("GraphTransformer", "validateNet"),
    ("GraphTransformer", "sample"),
    ("GraphTransformer", "iterLoader"),
    ("SequenceRNN", "sample"),
    ("FragmentSequenceEncoder", "encodeMol"),
    ("FragmentGraphEncoder", "encodeMol"),
    ("Fragmenter", "__call__"),
    ("FragmenterWithSelectedFragment", "__call__"),
    ("BRICSFragmenter", "__call__"),
    ("RECAPFragmenter", "__call__"),
    ("Linear", "__call__"),
    ("Gaussian", "__call__"),
    ("SmoothClippedScore", "__call__"),
    ("ClippedScore", "__call__"),
    ("MinMaxGaussian", "__call__"),
    ("SmoothHump", "__call__"),
    ("ThresholdedLinear", "__call__"),
    ("Squared", "__call__"),
    ("AbsoluteScore", "__call__"),
    ("Chained", "__call__"),
    ("NullMonitor", "getModel"),
    ("FileMonitor", "getModel"),
    ("PeakRecoveryMonitor", "getModel"),
    ("ParetoTanimotoDistance", "getMoleculeRank"),
    ("WeightedSum", "__call__"),
    ("DataSet", "readVocs"),
    ("SmilesDataSet", "readVocs"),
    ("SmilesFragDataSet", "readVocs"),
    ("GraphFragDataSet", "readVocs"),
    ("RDKitROCSScorer", "getScores"),
    ("CDPKitROCSScorer", "getScores"),
    ("OpenEyeROCSScorer", "getScores"),
}


def normalize_type_str(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()

    # Strip Sphinx single-identifier brackets, e.g. [VocSmiles] -> VocSmiles or [[ExplicitBitVect]] -> ExplicitBitVect
    cleaned = re.sub(r"^\[+([a-zA-Z0-9_\.]+)\]+$", r"\1", cleaned)
    cleaned = re.sub(r"\[\[+([a-zA-Z0-9_\.]+)\]\]+", r"\1", cleaned)

    # Clean numpy.array -> np.ndarray
    cleaned = re.sub(r"\b(np|numpy)\.array\b", "np.ndarray", cleaned)

    # Clean double bracket typo in tuples
    cleaned = cleaned.replace("list[str][list[str]]", "Tuple[List[str], List[str]]")
    cleaned = cleaned.replace("IntSmiVect", "Any")
    cleaned = cleaned.replace("UIntSmiVect", "Any")

    lower = cleaned.lower()
    if lower in TYPE_MAPPINGS:
        return TYPE_MAPPINGS[lower]

    # Handle docstrings ending with optional (e.g. "int, optional", "str, optional", "dict, optional")
    if re.search(r",\s*optional$", cleaned, re.IGNORECASE):
        base = re.sub(r",\s*optional$", "", cleaned, flags=re.IGNORECASE).strip()
        base_norm = normalize_type_str(base)
        return f"Optional[{base_norm}]"
    if lower.endswith(" optional"):
        base = cleaned[:-9].strip()
        base_norm = normalize_type_str(base)
        return f"Optional[{base_norm}]"
    if lower == "optional":
        return "Optional[Any]"

    if cleaned.startswith("list[") and cleaned.endswith("]"):
        inner = cleaned[5:-1].strip()
        return f"List[{normalize_type_str(inner)}]"
    if cleaned.startswith("tuple[") and cleaned.endswith("]"):
        inner = cleaned[6:-1].strip()
        return f"Tuple[{normalize_type_str(inner)}]"
    if cleaned.startswith("dict[") and cleaned.endswith("]"):
        inner = cleaned[5:-1].strip()
        return f"Dict[{normalize_type_str(inner)}]"
    if cleaned.startswith("set[") and cleaned.endswith("]"):
        inner = cleaned[4:-1].strip()
        return f"Set[{normalize_type_str(inner)}]"
    return cleaned


def split_top_level_args(arg_str: str) -> List[str]:
    """Split argument string by comma respecting nested brackets."""
    args = []
    current = []
    depth = 0
    for char in arg_str:
        if char in "([<{":
            depth += 1
            current.append(char)
        elif char in ")]>}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                args.append(part)
            current = []
        else:
            current.append(char)
    if current:
        part = "".join(current).strip()
        if part:
            args.append(part)
    return args


class EducationDB:
    def __init__(self, db_path: Optional[str] = None):
        self.data: Dict[str, Any] = {}
        if db_path and os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get_symbol(self, module_path: str, symbol_name: str) -> Optional[Dict[str, Any]]:
        key = f"{module_path.replace('.', '::')}::{symbol_name}"
        if key in self.data:
            return self.data[key]
        for k, v in self.data.items():
            if k.endswith(f"::{symbol_name}"):
                return v
        return None

    def extract_types_from_synopsis(self, synopsis: str) -> Dict[str, Dict[str, Any]]:
        """Extract method and function signatures from synopsis string."""
        signatures: Dict[str, Dict[str, Any]] = {}
        pattern = re.compile(
            r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*([^\n:]+))?",
            re.DOTALL,
        )
        for match in pattern.finditer(synopsis):
            name = match.group(1)
            raw_args = match.group(2).strip()
            raw_ret = (match.group(3) or "").strip()
            arg_types = {}
            for arg_part in split_top_level_args(raw_args):
                if ":" in arg_part:
                    p_name, p_type = arg_part.split(":", 1)
                    p_name = p_name.strip()
                    if "=" in p_type:
                        p_type, _ = p_type.split("=", 1)
                    arg_types[p_name] = p_type.strip()
            signatures[name] = {
                "arg_types": arg_types,
                "return_type": raw_ret if raw_ret else None,
            }
        return signatures


class GitDocExtractor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._cache: Dict[Tuple[str, str], Optional[str]] = {}

    def get_file_at_commit(self, commit: str, rel_path: str) -> Optional[str]:
        cache_key = (commit, rel_path)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            res = subprocess.run(
                ["git", "show", f"{commit}:{rel_path}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            content = res.stdout
            self._cache[cache_key] = content
            return content
        except Exception:
            self._cache[cache_key] = None
            return None

    def extract_ast_docs(self, code: str) -> Dict[str, Any]:
        """Extract module, class, and function docstrings and annotations from python code."""
        result = {
            "module_doc": None,
            "classes": {},
            "functions": {},
        }
        try:
            tree = ast.parse(code)
        except Exception:
            return result

        result["module_doc"] = ast.get_docstring(tree)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                c_name = node.name
                c_doc = ast.get_docstring(node)
                methods = {}
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[item.name] = {
                            "doc": ast.get_docstring(item),
                            "node": item,
                        }
                result["classes"][c_name] = {
                    "doc": c_doc,
                    "methods": methods,
                    "node": node,
                }
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["functions"][node.name] = {
                    "doc": ast.get_docstring(node),
                    "node": node,
                }
        return result


def parse_numpy_doc_params(doc: Optional[str]) -> Tuple[Dict[str, str], Optional[str]]:
    """Parse parameter types and return type from NumPy docstring."""
    if not doc:
        return {}, None
    params: Dict[str, str] = {}
    return_type: Optional[str] = None

    lines = doc.split("\n")
    in_params = False
    in_returns = False
    for line in lines:
        striped = line.strip()
        if striped == "Parameters":
            in_params = True
            in_returns = False
            continue
        elif striped in ("Returns", "Yields"):
            in_params = False
            in_returns = True
            continue
        elif striped in ("Raises", "Notes", "References", "Examples", "See Also"):
            in_params = False
            in_returns = False
            continue
        elif striped.startswith("---"):
            continue

        if in_params:
            m = re.match(r"^([a-zA-Z0-9_]+)\s*:\s*(.+)$", striped)
            if m:
                p_name = m.group(1).strip()
                p_type = normalize_type_str(m.group(2).strip())
                params[p_name] = p_type
        elif in_returns:
            m = re.match(r"^(?:[a-zA-Z0-9_]+\s*:\s*)?(.+)$", striped)
            if m and not return_type:
                return_type = normalize_type_str(m.group(1).strip())

    return params, return_type


def format_docstring(doc: str, indent: str = "    ") -> str:
    """Format docstring cleanly with proper indentation and triple quotes."""
    doc = doc.strip()
    if not doc:
        return ""
    doc = doc.replace('"""', r'\"\"\"')
    raw_prefix = "r" if "\\" in doc else ""
    lines = doc.split("\n")
    if len(lines) == 1:
        return f'{indent}{raw_prefix}"""{lines[0]}"""\n'
    formatted = [f'{indent}{raw_prefix}"""', f'{indent}{lines[0]}']
    for line in lines[1:]:
        if line.strip():
            formatted.append(f"{indent}{line}")
        else:
            formatted.append("")
    formatted.append(f'{indent}"""\n')
    return "\n".join(formatted)


def unparse_type(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def generate_function_stub(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    hist_node: Optional[ast.FunctionDef | ast.AsyncFunctionDef] = None,
    hist_doc: Optional[str] = None,
    edu_info: Optional[Dict[str, Any]] = None,
    synopsis_info: Optional[Dict[str, Any]] = None,
    indent: str = "",
    is_method: bool = False,
    class_name: Optional[str] = None,
    module_dotted: Optional[str] = None,
) -> str:
    """Generate a typed function or method stub signature with aligned docstrings."""
    lines = []
    # Decorators
    for dec in fn_node.decorator_list:
        dec_str = ast.unparse(dec)
        lines.append(f"{indent}@{dec_str}")

    prefix = "async def " if isinstance(fn_node, ast.AsyncFunctionDef) else "def "
    fn_name = fn_node.name

    # Determine best docstring
    current_doc = ast.get_docstring(fn_node)
    doc_to_use = hist_doc or current_doc
    if not doc_to_use and edu_info:
        for m in edu_info.get("members", []):
            if m.get("name", "").startswith(fn_name + "(") or m.get("name") == fn_name:
                doc_to_use = m.get("desc", "")
                break

    # Parse parameter and return types from docstrings and synopsis
    numpy_params, numpy_ret = parse_numpy_doc_params(doc_to_use)
    syn_args = synopsis_info.get("arg_types", {}) if synopsis_info else {}
    syn_ret = synopsis_info.get("return_type") if synopsis_info else None

    # Education parameters if __init__
    edu_params = {}
    if fn_name == "__init__" and edu_info and "parameters" in edu_info:
        for p in edu_info["parameters"]:
            edu_params[p.get("name")] = p.get("type")

    # Build argument list
    arg_strs = []
    posonly = fn_node.args.posonlyargs
    args = fn_node.args.args
    vararg = fn_node.args.vararg
    kwonly = fn_node.args.kwonlyargs
    kw_defaults = fn_node.args.kw_defaults
    kwarg = fn_node.args.kwarg
    defaults = fn_node.args.defaults

    def get_arg_type(arg_name: str, current_ann: Optional[ast.AST], hist_ann: Optional[ast.AST]) -> str:
        # Liskov substitution overrides for polymorphic interface methods
        if fn_name == "__call__":
            if class_name in ("Environment", "DrugExEnvironment", "ModelEvaluator"):
                if arg_name == "smiles":
                    return "Sequence[str]"
                if arg_name == "frags":
                    return "Optional[Sequence[str]]"
            elif class_name in ("RewardScheme", "ParetoRewardScheme", "ParetoCrowdingDistance", "ParetoTanimotoDistance", "WeightedSum"):
                if arg_name == "thresholds":
                    return "Sequence[float] | Sequence[Any]"
        elif fn_name == "apply" and "ParallelProcessor" in str(class_name):
            return "Any"
        elif fn_name == "readVocs":
            if arg_name == "voc_cls":
                return "Any"
        elif fn_name == "fit" and ("Explorer" in str(class_name) or class_name in ("Model", "Generator")):
            if arg_name in ("train_loader", "valid_loader", "monitor"):
                return "Any"
        elif fn_name == "validateNet":
            if arg_name in ("loader", "evaluator", "n_samples"):
                return "Any"
        elif fn_name in ("getBatchOutputs", "sample"):
            return "Any"

        # 1. Current annotation
        ann = unparse_type(current_ann)
        if ann:
            return normalize_type_str(ann)
        # 2. Historical AST annotation
        ann = unparse_type(hist_ann)
        if ann:
            return normalize_type_str(ann)
        # 3. Synopsis from education DB
        if arg_name in syn_args:
            return normalize_type_str(syn_args[arg_name])
        # 4. Edu parameters (__init__)
        if arg_name in edu_params and edu_params[arg_name]:
            return normalize_type_str(edu_params[arg_name])
        # 5. NumPy docstring parameter
        if arg_name in numpy_params:
            return normalize_type_str(numpy_params[arg_name])
        # Fallback heuristics
        if arg_name in ("self", "cls"):
            return ""
        if arg_name == "smiles":
            return "Sequence[str] | str"
        if arg_name == "frags":
            return "Optional[Sequence[str]]"
        if arg_name == "mols":
            return "List[Chem.Mol]"
        if arg_name == "mol":
            return "Chem.Mol"
        if arg_name in ("device", "dev"):
            return "str | torch.device"
        if arg_name in ("epochs", "batch_size", "n_samples", "seed", "n_jobs", "max_len", "min_len"):
            return "int"
        if arg_name in ("threshold", "epsilon", "lr"):
            return "float"
        return "Any"

    # Positional defaults mapping
    total_pos = len(posonly) + len(args)
    pos_default_start = total_pos - len(defaults)

    # Positional-only args
    if posonly:
        for i, a in enumerate(posonly):
            t = get_arg_type(a.arg, a.annotation, None)
            ann = f": {t}" if t else ""
            default = " = ..." if i >= pos_default_start else ""
            arg_strs.append(f"{a.arg}{ann}{default}")
        arg_strs.append("/")

    # Regular args
    hist_args_map = {ha.arg: ha for ha in (hist_node.args.args if hist_node else [])}
    for j, a in enumerate(args):
        overall_idx = len(posonly) + j
        hist_a = hist_args_map.get(a.arg)
        hist_ann = hist_a.annotation if hist_a else None
        t = get_arg_type(a.arg, a.annotation, hist_ann)
        ann = f": {t}" if t else ""
        default = " = ..." if overall_idx >= pos_default_start else ""
        arg_strs.append(f"{a.arg}{ann}{default}")

    # *vararg
    if vararg:
        t = get_arg_type(vararg.arg, vararg.annotation, None)
        ann = f": {t}" if t and t != "Any" else ""
        arg_strs.append(f"*{vararg.arg}{ann}")
    elif kwonly:
        arg_strs.append("*")

    # Keyword-only args
    for a, d in zip(kwonly, kw_defaults):
        t = get_arg_type(a.arg, a.annotation, None)
        ann = f": {t}" if t else ""
        default = " = ..." if d is not None else ""
        arg_strs.append(f"{a.arg}{ann}{default}")

    # **kwarg
    if kwarg:
        t = get_arg_type(kwarg.arg, kwarg.annotation, None)
        ann = f": {t}" if t and t != "Any" else ""
        arg_strs.append(f"**{kwarg.arg}{ann}")

    # Return annotation
    ret_str = unparse_type(fn_node.returns)
    if not ret_str and hist_node:
        ret_str = unparse_type(hist_node.returns)
    if not ret_str and syn_ret:
        ret_str = syn_ret
    if not ret_str and numpy_ret:
        ret_str = numpy_ret
    if not ret_str:
        if fn_name == "__init__":
            ret_str = "None"
        elif fn_name.startswith("is_") or fn_name.startswith("has_"):
            ret_str = "bool"
        elif fn_name == "getScores":
            ret_str = "pd.DataFrame"
        elif fn_name in ("getRewards", "getUnmodifiedScores"):
            ret_str = "np.ndarray"
        elif fn_name == "__iter__":
            ret_str = "Iterator[Any]"
        else:
            ret_str = "Any"

    # Normalize return type
    ret_str = normalize_type_str(ret_str)

    # Polymorphic return type overrides for Liskov compliance
    if fn_name == "encodeMol" and "Fragment" in str(class_name):
        ret_str = "Tuple[Any, Any]"
    elif fn_name == "iterLoader":
        ret_str = "Any"
    elif fn_name == "getMoleculeRank":
        ret_str = "Any"
    elif fn_name == "__iter__":
        ret_str = "Iterator[Any]"

    ret_ann = f" -> {ret_str}"

    # Check if method is a known override requiring type: ignore
    ignore_comment = ""
    if (class_name, fn_name) in KNOWN_OVERRIDE_METHODS:
        ignore_comment = "  # type: ignore[override]"

    # Assemble signature
    args_joined = ", ".join(arg_strs)
    sig_line = f"{indent}{prefix}{fn_name}({args_joined}){ret_ann}:{ignore_comment}"
    lines.append(sig_line)

    if doc_to_use:
        lines.append(format_docstring(doc_to_use, indent=indent + "    "))
    lines.append(f"{indent}    ...\n")
    return "\n".join(lines)


def generate_class_stub(
    cls_node: ast.ClassDef,
    hist_cls: Optional[Dict[str, Any]] = None,
    edu_info: Optional[Dict[str, Any]] = None,
    edu_db: Optional[EducationDB] = None,
    indent: str = "",
    module_dotted: Optional[str] = None,
) -> str:
    """Generate a typed class stub with methods, attributes, and docstrings."""
    lines = []
    # Decorators
    for dec in cls_node.decorator_list:
        dec_str = ast.unparse(dec)
        lines.append(f"{indent}@{dec_str}")

    bases_strs = [ast.unparse(b) for b in cls_node.bases]
    for kw in cls_node.keywords:
        bases_strs.append(f"{kw.arg}={ast.unparse(kw.value)}")
    bases_part = f"({', '.join(bases_strs)})" if bases_strs else ""

    lines.append(f"{indent}class {cls_node.name}{bases_part}:")

    # Class docstring
    hist_doc = hist_cls.get("doc") if hist_cls else None
    current_doc = ast.get_docstring(cls_node)
    class_doc = hist_doc or current_doc

    if not class_doc and edu_info:
        desc = edu_info.get("description", "")
        notes = edu_info.get("notes", "")
        params = edu_info.get("parameters", [])
        doc_parts = [desc] if desc else []
        if params:
            doc_parts.append("\nParameters\n----------")
            for p in params:
                p_type = p.get("type", "Any")
                p_desc = p.get("desc", "")
                doc_parts.append(f"{p.get('name')} : {p_type}\n    {p_desc}")
        if notes:
            doc_parts.append(f"\nNotes\n-----\n{notes}")
        class_doc = "\n".join(doc_parts)

    if class_doc:
        lines.append(format_docstring(class_doc, indent=indent + "    "))

    # Synopsis signatures from education DB
    synopsis_map = {}
    if edu_info and edu_db and "synopsis" in edu_info:
        synopsis_map = edu_db.extract_types_from_synopsis(edu_info["synopsis"])

    body_empty = True

    # Process class variables from education DB if available
    edu_vars = {}
    if edu_info and "variables" in edu_info:
        for v in edu_info["variables"]:
            edu_vars[v.get("name")] = v.get("type", "Any")

    has_next = False
    # Process class body items: AnnAssign, Assign, FunctionDef
    for item in cls_node.body:
        if isinstance(item, ast.AnnAssign):
            target = ast.unparse(item.target)
            ann = unparse_type(item.annotation) or edu_vars.get(target, "Any")
            val = " = ..." if item.value is not None else ""
            lines.append(f"{indent}    {target}: {ann}{val}")
            body_empty = False
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                target = ast.unparse(t)
                if not target.startswith("_"):
                    ann = edu_vars.get(target, "Any")
                    lines.append(f"{indent}    {target}: {ann} = ...")
                    body_empty = False
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            m_name = item.name
            if m_name == "next":
                has_next = True
            m_hist = hist_cls["methods"].get(m_name) if hist_cls and "methods" in hist_cls else None
            m_hist_doc = m_hist.get("doc") if m_hist else None
            m_hist_node = m_hist.get("node") if m_hist else None
            m_syn = synopsis_map.get(m_name)
            fn_stub = generate_function_stub(
                item,
                hist_node=m_hist_node,
                hist_doc=m_hist_doc,
                edu_info=edu_info,
                synopsis_info=m_syn,
                indent=indent + "    ",
                is_method=True,
                class_name=cls_node.name,
                module_dotted=module_dotted,
            )
            lines.append(fn_stub)
            body_empty = False

    # BaseMolSupplier compatibility for next()
    if cls_node.name == "BaseMolSupplier" and not has_next:
        lines.append(f"{indent}    @abstractmethod")
        lines.append(f"{indent}    def next(self) -> Any: ...\n")
        body_empty = False

    if body_empty:
        lines.append(f"{indent}    ...\n")
    return "\n".join(lines)


def process_module(
    py_path: Path,
    repo_root: Path,
    git_extractor: GitDocExtractor,
    edu_db: EducationDB,
) -> str:
    """Process a single Python module into a .pyi stub content."""
    with open(py_path, "r", encoding="utf-8") as f:
        current_code = f.read()

    try:
        current_tree = ast.parse(current_code)
    except Exception as e:
        print(f"Error parsing {py_path}: {e}")
        return ""

    rel_path = str(py_path.relative_to(repo_root))
    module_dotted = rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
    is_init_file = py_path.name == "__init__.py"

    # Find best historical file
    hist_ast_docs = None
    for commit in HISTORICAL_COMMITS:
        hist_code = git_extractor.get_file_at_commit(commit, rel_path)
        if hist_code:
            hist_ast_docs = git_extractor.extract_ast_docs(hist_code)
            if hist_ast_docs["module_doc"] or hist_ast_docs["classes"] or hist_ast_docs["functions"]:
                break

    lines = []

    # Module Docstring
    current_mod_doc = ast.get_docstring(current_tree)
    hist_mod_doc = hist_ast_docs["module_doc"] if hist_ast_docs else None
    mod_doc = hist_mod_doc or current_mod_doc
    if mod_doc:
        lines.append(f'"""\n{mod_doc.strip()}\n"""\n')

    # Standard typing imports header
    lines.append("from __future__ import annotations\n")
    lines.append("import logging")
    lines.append("from abc import ABC, abstractmethod")
    # Note: NEVER import Generator from typing to avoid shadowing drugex.training.generators.interfaces.Generator
    lines.append("from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload")
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("from rdkit import Chem")
    lines.append("")

    # Collect unique imports from current module
    seen_imports = {
        "from __future__ import annotations",
        "import numpy as np",
        "import pandas as pd",
        "import torch",
        "import torch.nn as nn",
        "from rdkit import Chem",
    }

    # Collect re-exported names for __all__ if __init__.py
    init_exported_names = set()

    for item in current_tree.body:
        if isinstance(item, ast.Import):
            for alias in item.names:
                name = alias.name
                asname = alias.asname or (name if is_init_file else None)
                imp_str = f"import {name} as {asname}" if asname else f"import {name}"
                if imp_str not in seen_imports:
                    seen_imports.add(imp_str)
                    lines.append(imp_str)
                if is_init_file and asname:
                    init_exported_names.add(asname)
        elif isinstance(item, ast.ImportFrom):
            mod = ("." * item.level) + (item.module or "")
            for alias in item.names:
                name = alias.name
                if (mod == "typing" or not mod) and name == "Generator":
                    continue
                asname = alias.asname or (name if is_init_file and name != "*" else None)
                imp_str = f"from {mod} import {name} as {asname}" if asname else f"from {mod} import {name}"
                if imp_str not in seen_imports and "__future__" not in imp_str:
                    seen_imports.add(imp_str)
                    lines.append(imp_str)
                if is_init_file and asname and name != "*":
                    init_exported_names.add(asname)

    lines.append("")

    # Specific file overrides check
    overridden_files = {
        "drugex/logs/__init__.py",
        "drugex/about.py",
        "drugex/_version.py",
        "drugex/training/monitors.py",
    }

    # Top-level variables & constants
    has_all_defined = False
    for item in current_tree.body:
        if rel_path in overridden_files:
            continue
        if isinstance(item, ast.AnnAssign):
            target = ast.unparse(item.target)
            ann = unparse_type(item.annotation) or "Any"
            lines.append(f"{target}: {ann}")
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                target = ast.unparse(t)
                if target == "__all__":
                    has_all_defined = True
                    val_str = ast.unparse(item.value)
                    lines.append(f"{target} = {val_str}")
                elif target.isupper():
                    val_str = " = ..."
                    lines.append(f"{target}: Any{val_str}")
                elif target == "logger":
                    lines.append("logger: logging.Logger")

    # If __init__.py and no __all__, emit explicit __all__
    if is_init_file and not has_all_defined and init_exported_names:
        sorted_names = sorted(init_exported_names)
        names_quoted = ", ".join(f"'{n}'" for n in sorted_names)
        lines.append(f"__all__ = [{names_quoted}]")

    # Specific file overrides
    if rel_path == "drugex/logs/__init__.py":
        lines.append("logger: logging.Logger")
    elif rel_path == "drugex/about.py":
        lines.append("VERSION: str = ...")
    elif rel_path == "drugex/_version.py":
        lines.append("version: str = ...")
        lines.append("version_tuple: Tuple[int, int, int] = ...")
    elif rel_path == "drugex/training/monitors.py":
        lines.append("__all__ = ['TrainingMonitor', 'Model', 'NullMonitor', 'FileMonitor', 'PeakRecoveryMonitor']")

    lines.append("")

    # Top-level classes and functions
    body_lines = []
    for item in current_tree.body:
        if isinstance(item, ast.ClassDef):
            c_name = item.name
            hist_cls = hist_ast_docs["classes"].get(c_name) if hist_ast_docs else None
            edu_info = edu_db.get_symbol(module_dotted, c_name)
            cls_stub = generate_class_stub(
                item,
                hist_cls=hist_cls,
                edu_info=edu_info,
                edu_db=edu_db,
                module_dotted=module_dotted,
            )
            body_lines.append(cls_stub)
            body_lines.append("")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f_name = item.name
            hist_fn = hist_ast_docs["functions"].get(f_name) if hist_ast_docs else None
            hist_doc = hist_fn.get("doc") if hist_fn else None
            hist_node = hist_fn.get("node") if hist_fn else None
            edu_info = edu_db.get_symbol(module_dotted, f_name)
            syn_info = None
            if edu_info and "synopsis" in edu_info:
                syn_map = edu_db.extract_types_from_synopsis(edu_info["synopsis"])
                syn_info = syn_map.get(f_name)
            fn_stub = generate_function_stub(
                item,
                hist_node=hist_node,
                hist_doc=hist_doc,
                edu_info=edu_info,
                synopsis_info=syn_info,
                module_dotted=module_dotted,
            )
            body_lines.append(fn_stub)
            body_lines.append("")

    body_content = "\n".join(body_lines)
    full_content = "\n".join(lines) + "\n" + body_content

    # Automatic common type imports if detected anywhere in full_content
    extra_imports = []
    for type_name, imp_stmt in COMMON_TYPE_IMPORTS.items():
        if re.search(rf"\b{type_name}\b", full_content) and imp_stmt not in seen_imports:
            # Avoid importing itself
            if f"class {type_name}" not in full_content and f"import {type_name}" not in seen_imports:
                if not imp_stmt.startswith(f"from {module_dotted} "):
                    seen_imports.add(imp_stmt)
                    extra_imports.append(imp_stmt)

    # Module specific imports
    if rel_path in MODULE_SPECIFIC_IMPORTS:
        for extra in MODULE_SPECIFIC_IMPORTS[rel_path]:
            if extra not in seen_imports:
                seen_imports.add(extra)
                extra_imports.append(extra)

    # Reassemble with extra imports placed right after header imports
    final_lines = []
    header_done = False
    for line in lines:
        final_lines.append(line)
        if not header_done and "from rdkit import Chem" in line:
            header_done = True
            for ei in extra_imports:
                final_lines.append(ei)

    final_lines.append(body_content)
    return "\n".join(final_lines)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    edu_db_path = "/tmp/drugex_education_db.json"
    edu_db = EducationDB(edu_db_path)
    git_extractor = GitDocExtractor(repo_root)

    drugex_dir = repo_root / "drugex"
    py_files = sorted(drugex_dir.rglob("*.py"))

    # Exclude test files from stubs
    skipped_patterns = ["tests.py", "create_test_qsar.py", "test_files"]
    target_files = [
        f for f in py_files
        if not any(pat in str(f) for pat in skipped_patterns)
    ]

    print(f"Total target modules in drugex/: {len(target_files)}")

    success_count = 0
    for py_file in target_files:
        rel = py_file.relative_to(repo_root)
        pyi_file = py_file.with_suffix(".pyi")
        stub_content = process_module(py_file, repo_root, git_extractor, edu_db)
        if stub_content:
            try:
                # Validate AST parses cleanly
                ast.parse(stub_content)
                with open(pyi_file, "w", encoding="utf-8") as out:
                    out.write(stub_content)
                success_count += 1
            except SyntaxError as e:
                print(f"Syntax error in generated stub for {rel}: {e}")

    print(f"Successfully generated {success_count} / {len(target_files)} .pyi stubs.")


if __name__ == "__main__":
    main()
