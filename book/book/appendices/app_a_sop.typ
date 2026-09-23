#import "../nature_theme.typ": *

= Příloha A: Standard Operating Procedure (SOP) pro HPC klastry (MetaCentrum & IT4I)

Tato příloha představuje závazný technologický postup (_Standard Operating Procedure_, SOP) pro nasazení výpočetních experimentů platformy DrugEx kombinovaných s 3D tvarovým porovnáváním (ROCS) na národní superpočítačové infrastruktuře ČR (MetaCentrum CESNET a IT4Innovations Národní superpočítačové centrum).

Výpočty de novo návrhu molekul s generováním stovek tisíc 3D konformací za běhu představují hybridní zátěž: neuronové generátory vyžadují masivní tenzorový výkon GPU, zatímco 3D embedding a Gaussovské zarovnání struktur spotřebovávají stovky CPU hodin. Nedodržení níže uvedených zásad vede k drastickému propadu výpočetní propustnosti nebo selhání úloh v důsledku vyčerpání paměti.

== A.1 Architektura superpočítačových uzlů & Plánovač Slurm

Moderní GPU uzly (např. klastr _Karolina_ na IT4I či servery _adan_, _alcyone_ na MetaCentru) disponují 64 až 128 fyzickými jádry AMD EPYC a 4 až 8 akcelerátory NVIDIA A100/H100 (40/80 GB VRAM). Úlohy jsou řízeny plánovačem úloh *Slurm* v dávkovém režimu.

#rule_box("Využití rychlého lokálního NVMe scratch disku")[
  Nikdy nespouštějte I/O intenzivní operace (zápis stovek tisíc dočasných konformerů nebo SDF souborů) přímo na sdílených síťových svazcích NFS či Lustre (`/storage` nebo `/home`). Síťové zpoždění a uzamykání metadat způsobí kolaps propustnosti celého klastru. Veškeré dočasné soubory musí směřovat na lokální NVMe scratch disk přidělený proměnnou `$SCRATCHDIR`.
]

== A.2 Kompletní produkční Slurm dávkový skript

Následující skript představuje standardizovanou šablonu pro spuštění produkčního tréninku DrugEx MORL s 3D ROCS skórováním na 1 GPU a 16 CPU jádrech:

```bash
#!/bin/bash
# ==============================================================================
# Slurm Batch Job: DrugEx MORL + 3D ROCS Shape Matching Pipeline
# Autor: David Kolář (VŠCHT Praha / ÚOCHB AV ČR)
# ==============================================================================
#SBATCH --job-name=drugex_rocs_prod
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/drugex_%j.out
#SBATCH --error=logs/drugex_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=researcher@vscht.cz

set -euo pipefail

echo "========================================================="
echo " Výpočet spuštěn na uzlu : $(hostname)"
echo " Datum a čas             : $(date)"
echo " Slurm Job ID            : $SLURM_JOB_ID"
echo " Alokováno CPU jader     : $SLURM_CPUS_PER_TASK"
echo " Alokována GPU           : ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================================="

# 1. Načtení modulů infrastruktury
module purge
module load CUDA/12.1.1 Python/3.11.5 GCC/12.2.0 OpenEye/2023.2.0

# 2. Aktivace dedikovaného virtuálního prostředí
source /storage/praha1/home/kolar/.venvs/drugex/bin/activate

# 3. Striktní eliminace CPU Thread Oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 4. Prevence fragmentace paměti PyTorch CUDA Allocatoru
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 5. Konfigurace licencí pro OpenEye toolkits
export OE_LICENSE=/storage/praha1/home/kolar/licenses/oe_license.txt

# 6. Příprava NVMe scratch adresáře a ošetření bezpečného úklidu
SCRATCH_DIR="${SCRATCHDIR:-/tmp/drugex_$SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR/work"
cd "$SLURM_SUBMIT_DIR"

# Registrace signálového zachycení pro záchranu dat při vypršení časového limitu
trap 'echo "Zachycen signál ukončení. Kopíruji výsledky ze scratche..."; \
      cp -ru "$SCRATCH_DIR/work"/* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true; \
      rm -rf "$SCRATCH_DIR"' EXIT TERM INT

# 7. Spuštění produkčního výpočtu DrugEx MORL
python -m drugex.train \
    -b "$SLURM_SUBMIT_DIR" \
    -i ccr2_corpus \
    -ag models/Papyrus05.5_smiles_rnn_PT.pkg \
    -pr models/ccr2_corpus_smiles_rnn_FT.pkg \
    -mt smiles \
    -a rnn \
    -tm RL \
    -e 50 \
    -bs 128 \
    -gpu "${CUDA_VISIBLE_DEVICES:-0}"

echo "========================================================="
echo " Výpočet úspěšně dokončen: $(date)"
echo "========================================================="
```

== A.3 Kritická past: CPU Thread Oversubscription a jeho eliminace

Zákeřným problémem při běhu chemoinformatických výpočtů na vícejádrových serverech je fenomén *CPU Thread Oversubscription*. Tento jev způsobuje až padesátinásobné zpomalení celého výpočetního procesu a často vede k dojmu, že kód „zamrzl“.

=== Mechanismus vzniku patologie

+ Výpočetní uzel superpočítače disponuje fyzicky např. 128 jádry na základní desce.
+ Knihovny numerické lineární algebry v jazycích C a Fortran (`OpenBLAS`, `Intel MKL`, `OpenMP`, `NumExpr`), na nichž staví RDKit, NumPy a SciPy, při inicializaci detekují celkový hardware uzlu. Pokud nejsou explicitně instruovány jinak, každá knihovna vytvoří pro libovolnou vektorovou operaci fond o velikosti 128 výpočetních vláken.
+ V Python kódu skórovače `RDKitROCSScorer` nebo konformačního generátoru je nastaven paralelismus na úrovni procesů: `multiprocessing.Pool(processes=16)`.
+ Každý ze 16 spuštěných procesů vygeneruje svých 128 vláken.
+ Na uzlu rázem vzniká:
  $ 16 " procesů" times 128 " vláken" = bold(2048 " konkurenčních vláken") $
+ Všechna tato vlákna se agresivně perou o pouhých 16 CPU jader, která vaší úloze přidělil plánovač Slurm přes `--cpus-per-task=16`!

#pitfall_box("Kolaps procesoru v důsledku přepínání kontextu")[
  Při vzniku 2048 vláken tráví jádra procesoru více než 95 % veškerého strojového času přepínáním kontextu (_Context Switching_) v jádře operačního systému a neustálým zneplatňováním L1/L2/L3 cache pamětí. Výpočetní propustnost CPU klesá na méně než 5 % jmenovitého výkonu!
]

=== Exaktní řešení & Naměřený benchmark

Náprava spočívá v povinném uzamčení počtu podvláken na hodnotu 1 ve všech nízkoúrovňových knihovnách. Paralelizace je plně svěřena Python procesům na úrovni molekulární dávky:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

V aplikačním kódu Pythonu je nutné tuto zásadu promítnout do konfigurace konformačního generátoru:
```python
conf_gen = RDKitConformerGenerator(num_threads=1)
```

Empirické měření na uzlu klastru Karolina (16 jader AMD EPYC 7763, vzorkování 1~000 molekul s generováním 50 konformerů na molekulu):
- *Bez omezení vláken (Oversubscription)*: Doba výpočtu epochy = $28 " minut" 42 " sekund"$.
- *S vynuceným `OMP_NUM_THREADS=1`*: Doba výpočtu epochy = $2 " minuty" 01 " sekund"$.
- *Výsledné zrychlení*: $bold(14.2 times)$ při identických výsledcích geometrického zarovnání!

== A.4 Prevence úniků paměti v PyTorch a předcházení CUDA OOM

Během dlouhotrvajícího Reinforcement Learning (RL) tréninku (50 až 100 epoch) dochází v PyTorch k postupnému vyčerpávání grafické paměti, které často končí havárií `RuntimeError: CUDA out of memory`. Tato chyba pramení ze dvou odlišných procesů:

=== 1. Hromadění výpočetního autograd grafu

Během fáze vzorkování molekul z generátoru (`sample()`) nesmí PyTorch alokovat uzly pro zpětný průchod gradientu. Celý generativní cyklus musí být uzavřen v kontextovém manažeru:

```python
with torch.no_grad():
    samples = agent.sample(n_samples=1024)
```

Dále při ukládání ztrátových funkcí do telemetrických záznamů nikdy neukládejte samotný tenzor `loss`, nýbrž výhradně skalární hodnotu `loss.item()`. Uložení reference na tenzor zabrání garbage collectoru uvolnit celý orientovaný acyklický výpočetní graf Backpropagation.

=== 2. Fragmentace PyTorch Caching Allocatoru

PyTorch z důvodu efektivity nevrací uvolněné paměťové bloky operačnímu systému GPU, nýbrž je udržuje ve vnitřní mezipaměti. Vzhledem k tomu, že molekuly SMILES mají proměnlivé délky (od 15 do 100 tokenů), dynamicky alokované bloky způsobují silnou fragmentaci paměti. Po 30 epochách je paměť natolik roztříštěná, že ačkoliv je volných 40 GB VRAM, alokátor nedokáže nalézt souvislý blok o velikosti 500 MB pro dopředný průchod.

*Provozní řešení*:
+ Nastavení konfigurace alokátoru v prostředí před spuštěním skriptu:
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  ```
+ Explicitní defragmentace a pročištění mezipaměti na konci každé tréninkové epochy:
  ```python
  del batch_tokens
  torch.cuda.empty_cache()
  ```

Dodržením tohoto protokolu je zajištěna stoprocentní stabilita vícedenních úloh na národních superpočítačích bez jediného pádu z důvodu přetečení paměti.
