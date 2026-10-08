import json
import re

with open('/tmp/m1_db.json') as f:
    db = json.load(f)

from check_all_59 import M1_SYMBOLS

placeholders = ["todo", "tbd", "lorem", "ipsum", "placeholder", "dummy implementation", "asdf", "foo bar"]

issues = []

for s in M1_SYMBOLS:
    item = db[s]
    code = item.get("example", {}).get("code", "")
    output = item.get("example", {}).get("output", "")
    desc = item.get("description", "")
    notes = str(item.get("notes", ""))
    
    # Check for placeholder strings
    for p in placeholders:
        if p in desc.lower():
            issues.append(f"{s}: placeholder '{p}' found in description")
        if p in notes.lower():
            issues.append(f"{s}: placeholder '{p}' found in notes")
        if p in code.lower() and p != "dummy": # 'dummyMolsFromFragments' is a legitimate drugex function name
            issues.append(f"{s}: placeholder '{p}' found in code")
        if p in output.lower():
            issues.append(f"{s}: placeholder '{p}' found in output")

    # Check realistic imports in code
    if "drugex" not in code and "rdkit" not in code and "torch" not in code:
        issues.append(f"{s}: example code does not import drugex, rdkit, or torch")

print(f"Total issues found during placeholder / example scan: {len(issues)}")
for iss in issues:
    print(" -", iss)
