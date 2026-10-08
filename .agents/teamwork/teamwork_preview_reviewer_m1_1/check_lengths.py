import json

with open('/tmp/m1_db.json') as f:
    db = json.load(f)

from check_all_59 import M1_SYMBOLS

short_notes = []
short_descs = []
for s in M1_SYMBOLS:
    item = db[s]
    n = item.get("notes", "")
    d = item.get("description", "")
    if len(str(n)) < 50:
        short_notes.append((s, len(str(n)), n))
    if len(str(d)) < 100:
        short_descs.append((s, len(str(d)), d))

print(f"Symbols with notes < 50 chars: {len(short_notes)}")
for s, l, n in short_notes:
    print(f"  {s}: ({l} chars) '{n}'")

print(f"\nSymbols with description < 100 chars: {len(short_descs)}")
for s, l, d in short_descs:
    print(f"  {s}: ({l} chars) '{d}'")
