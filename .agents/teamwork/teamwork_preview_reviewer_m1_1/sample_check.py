import subprocess
import json

symbols = [
    "drugex::data::corpus::interfaces::Corpus",
    "drugex::data::corpus::vocabulary::VocSmiles",
    "drugex::data::datasets::SmilesDataSet",
    "drugex::data::fragments::Fragmenter",
    "drugex::molecules::mol::DrExMol",
    "drugex::molecules::converters::MolConverter",
    "drugex::parallel::ParallelProcessor",
    "drugex::molecules::converters::ConversionException",
    "drugex::data::datasets::TargetCreator"
]

with open('/tmp/m1_db.json') as f:
    db = json.load(f)

for s in symbols:
    print("=" * 60)
    print(f"SYMBOL: {s}")
    entry = db[s]
    print(f"Synopsis: {entry.get('synopsis')}")
    print(f"Description:\n{entry.get('description')}\n")
    print("Parameters:")
    for p in entry.get("parameters", []):
        print(f"  - {p.get('name')}: {p.get('type')} = {p.get('default')} -> {p.get('desc')[:80]}...")
    print("Members:")
    for m in entry.get("members", [])[:5]:
        print(f"  - {m.get('name')}: {m.get('synopsis')} -> {m.get('desc')[:60]}...")
    print("Notes:")
    for n in entry.get("notes", []):
        print(f"  - {n[:100]}...")
    print("Example code:")
    print(entry.get("example", {}).get("code", "")[:200])
    print("Example output:")
    print(entry.get("example", {}).get("output", "")[:100])
