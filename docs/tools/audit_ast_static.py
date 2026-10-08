import ast
import json
import re
from pathlib import Path

repo_root = Path("repo/drugex").resolve()

all_classes = {}
all_functions = {}

for py_path in repo_root.rglob("*.py"):
    if "tests" in py_path.parts:
        continue
    rel = py_path.relative_to(repo_root.parent)
    mod_name = ".".join(rel.with_suffix("").parts)
    if mod_name.endswith(".__init__"):
        mod_name = mod_name[:-9]
    
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as e:
        continue
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = {}
            init_args = []
            init_defaults = 0
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods[item.name] = [a.arg for a in item.args.args if a.arg != "self"]
                    if item.name == "__init__":
                        init_args = [a.arg for a in item.args.args if a.arg != "self"]
                        init_defaults = len(item.args.defaults)
            if node.name not in all_classes:
                all_classes[node.name] = []
            all_classes[node.name].append({
                "module": mod_name,
                "bases": [ast.unparse(b) for b in node.bases],
                "methods": methods,
                "init_args": init_args,
                "init_defaults": init_defaults,
                "file": str(py_path.relative_to(repo_root))
            })

dump_path = Path("docs/tools/db_dump.json")
with open(dump_path, "r", encoding="utf-8") as f:
    db = json.load(f)

call_issues = []

for key, item in db.items():
    code = item.get("example", {}).get("code", "")
    try:
        tree = ast.parse(code)
    except Exception:
        continue
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                
            if func_name and func_name in all_classes:
                cls_info = all_classes[func_name][0]
                args = cls_info["init_args"]
                num_defaults = cls_info["init_defaults"]
                num_required = len(args) - num_defaults
                required_args = args[:num_required] if num_required > 0 else []
                
                # Check positional and keyword args in call
                call_pos_count = len(node.args)
                call_kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                
                # If there are **kwargs or *args in call, skip
                if any(kw.arg is None for kw in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
                    continue
                
                # Check if any required arg is missing
                for idx, req in enumerate(required_args):
                    if idx >= call_pos_count and req not in call_kw_names:
                        call_issues.append((key, func_name, f"Call is missing required positional arg '{req}' (required: {required_args})"))
                        break

print(f"\nCall site signature issues in examples: {len(call_issues)}")
for k, fn, issue in call_issues:
    print(f"  [{k}]: Calling {fn} -> {issue}")
