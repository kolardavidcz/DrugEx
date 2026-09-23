#import "../nature_theme.typ": *

= Příloha F: Referenční Příručka & Tahák (DrugEx, Slurm & RDKit)

Tato příloha slouží jako rychlá referenční karta (_Cheatsheet_) pro každodenní práci u terminálu. Obsahuje nejdůležitější příkazy rozhraní příkazové řádky DrugEx CLI, plánovače úloh Slurm a nejčastější one-linery chemoinformatické knihovny RDKit.

== F.1 DrugEx CLI: Referenční přehled příkazů

=== 1. Předzpracování a kódování dat (`drugex.dataset`)

```bash
# Příprava korpusu pro SequenceRNN (SMILES, 16 CPU procesů)
python -m drugex.dataset \
    -b workdir \
    -i raw_data/chembl_compounds.tsv \
    -mc SMILES \
    -o dataset_smiles \
    -mt smiles \
    -nof \
    -np 16

# Příprava fragmentů pro scaffold-based GraphTransformer (-s = scaffolds)
python -m drugex.dataset \
    -b workdir \
    -i raw_data/target_scaffolds.tsv \
    -mc SMILES \
    -o dataset_graph \
    -mt graph \
    -s \
    -np 8
```

Klíčové přepínače:
- `-b, --base_dir`: Kořenový pracovní adresář projektu.
- `-i, --input_file`: Vstupní soubor TSV/CSV se sloučeninami.
- `-mc, --mol_col`: Název sloupce obsahujícího SMILES (výchozí: `SMILES`).
- `-mt, --mol_type`: Formát molekuly: `smiles` (pro RNN/Transformers) nebo `graph` (pro grafové modely).
- `-nof, --no_filter`: Vypnutí filtrace neobvyklých prvků (ponechá všechny standardní atomy).
- `-s, --scaffolds`: Režim fixních scaffoldů pro fragmentové generátory.
- `-np, --n_proc`: Počet paralelních procesů CPU pro tokenizaci a standardizaci.

=== 2. Trénování generátoru: Fine-Tuning & Reinforcement Learning (`drugex.train`)

```bash
# A. Cílově zaměřený Fine-Tuning (FT) na GPU 0
python -m drugex.train \
    -b workdir \
    -i ccr2_corpus \
    -o ccr2_finetuned \
    -tm FT \
    -ag models/Papyrus05.5_smiles_rnn_PT.pkg \
    -mt smiles \
    -a rnn \
    -e 100 \
    -bs 64 \
    -pa 25 \
    -gpu 0

# B. Multi-Objective Reinforcement Learning (RL) s Paretovským schématem
python -m drugex.train \
    -b workdir \
    -i ccr2_corpus \
    -o ccr2_rl_model \
    -tm RL \
    -ag models/Papyrus05.5_smiles_rnn_PT.pkg \
    -pr models/ccr2_finetuned_smiles_rnn_FT.pkg \
    -mt smiles \
    -a rnn \
    -e 50 \
    -bs 128 \
    -ns 1024 \
    -eps 0.20 \
    -s PRCD \
    -sas \
    -qed \
    -gpu 0
```

Klíčové přepínače:
- `-tm, --train_mode`: Režim tréninku: `FT` (Supervised Fine-Tuning) nebo `RL` (Reinforcement Learning).
- `-ag, --agent_path`: Cesta k výchozímu balíčku modelu (`.pkg`), ze kterého se vychází.
- `-pr, --prior_path`: Cesta k fixní mutační síti (Prior) pro stabilizaci chemické gramatiky v RL.
- `-a, --algorithm`: Typ neuronové architektury: `rnn` (SequenceRNN) nebo `trans` (Transformer).
- `-e, --epochs`: Počet tréninkových epoch (typicky 100 pro FT, 50 pro RL).
- `-bs, --batch_size`: Velikost dávky pro aktualizaci vah (typicky 64 až 128).
- `-ns, --n_samples`: Počet molekul vzorkovaných v každé epoše Reinforcement Learning (RL).
- `-eps, --epsilon`: Poměr explorace ze zmrazeného Prioru (doporučeno $0.15$ až $0.25$).
- `-s, --scheme`: Vícekriteriální schéma: `PRCD` (Pareto Crowding Distance) nebo `WS` (Weighted Sum).
- `-sas, --sa_score`: Zapojení filtru syntetické dostupnosti SAScore.
- `-qed, --qed_score`: Zapojení skórovače léčivupodobnosti QED.

=== 3. Vzorkování a generování kandidátních molekul (`drugex.generate`)

```bash
# Vygenerování 10000 nových struktur s dávkou 1024 na GPU 0
python -m drugex.generate \
    -b workdir \
    -g ccr2_rl_model_smiles_rnn_RL \
    -o generated_leads \
    -n 10000 \
    -bs 1024 \
    -gpu 0
```

== F.2 Slurm HPC Tahák: Správa úloh na superpočítači

#table(
  columns: (2.5fr, 3.5fr),
  align: (left, left),
  [Příkaz Slurmu], [Význam a doporučené použití],
  [`sbatch job_script.sh`], [Odeslání dávkového skriptu do výpočetní fronty.],
  [`squeue -u $USER`], [Výpis všech aktivních a čekajících úloh přihlášeného uživatele.],
  [`scancel <JOB_ID>`], [Okamžité zrušení běžící nebo čekající úlohy.],
  [`sinfo -p gpu`], [Zobrazení stavu a dostupnosti GPU uzlů a oddílů.],
  [`scontrol show job <JOB_ID>`], [Detailní výpis alokace zdrojů, uzlu a limitů běžící úlohy.],
  [`sacct -j <JOB_ID> --format=...`], [Zpětná diagnostika ukončené úlohy (využitá paměť `MaxRSS`, čas `Elapsed`, návratový kód `ExitCode`).]
)

=== Klíčové proměnné prostředí Slurmu

```bash
# Automaticky nastaveno plánovačem uvnitř běžícího skriptu:
$SLURM_JOB_ID          # Unikátní číselný identifikátor úlohy
$SLURM_SUBMIT_DIR      # Adresář, ze kterého byl příkaz sbatch spuštěn
$SLURM_CPUS_PER_TASK   # Počet CPU jader přidělených úloze
$CUDA_VISIBLE_DEVICES  # Indexy grafických karet přidělených úloze (např. "0")
$SCRATCHDIR            # Cesta k ultrarychlému lokálnímu NVMe disku uzlu
```

== F.3 RDKit: Záchranné chemoinformatické one-linery

```python
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdShapeAlign
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# 1. Konverze SMILES na Mol s validací
mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

# 2. Rychlé generování 3D konformace (ETKDGv3)
mol_3d = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol_3d)

# 3. Morganův Fingerprint (ECFP4) a Tanimoto podobnost
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
sim = DataStructs.TanimotoSimilarity(fp1, fp2)

# 4. Výpočet Shape Tanimoto z 3D souřadnic (RDKit Gaussovský překryv)
shape_dist = rdShapeAlign.ShapeTanimotoDist(mol_ref, mol_fit)
shape_tanimoto = 1.0 - shape_dist

# 5. Detekce strukturních alertů PAINS
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)
has_pains = catalog.HasMatch(mol)

# 6. Kontrola počtu rotovatelných vazeb a velikosti kruhů
n_rot = Descriptors.NumRotatableBonds(mol)
ssr = Chem.GetSymmSSSR(mol)
max_ring_size = max([len(r) for r in ssr]) if ssr else 0
```
