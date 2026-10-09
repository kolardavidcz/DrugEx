import json
import re
import ast
import sys
from pathlib import Path

# Add repo to sys.path so we can verify if imports actually resolve!
repo_dir = Path("repo").resolve()
sys.path.insert(0, str(repo_dir))

data_dir = Path("docs/data").resolve()
matches = []
pattern = r'["\']([^"\']+)["\']:\s*\{\s*category:\s*["\']([^"\']+)["\'][\s\S]*?example:\s*\{\s*code:\s*`([^`]+)`,\s*output:\s*`([^`]*)`'

for f in sorted(data_dir.glob("*.js")):
    if f.name == "sections.js":
        continue
    content = f.read_text(encoding="utf-8")
    matches.extend(list(re.finditer(pattern, content)))

print(f"Found {len(matches)} examples across modular data files")

import_errors = []

for match in matches:
    key = match.group(1)
    cat = match.group(2)
    code = match.group(3)
    out = match.group(4)
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("drugex"):
                for alias in node.names:
                    name = alias.name
                    # Try to import or check
                    try:
                        imported_mod = __import__(mod, fromlist=[name])
                        if not hasattr(imported_mod, name):
                            import_errors.append((key, f"Module '{mod}' has no attribute '{name}'"))
                    except Exception as ex:
                        # Some modules may fail because of missing 3rd party deps (like openeye or cdpl)
                        msg = str(ex)
                        if "OpenEye" in msg or "CDPL" in msg or "openeye" in msg or "RAscore" in msg or "qsprpred" in msg:
                            pass # optional 3rd party
                        else:
                            import_errors.append((key, f"Failed to import '{name}' from '{mod}': {ex}"))

print(f"\nImport resolution issues across all 234 examples: {len(import_errors)}")
for k, err in import_errors:
    print(f"  [{k}]: {err}")
