/**
 * DrugEx Hub — Module 6: Scaffolds, Fragmenters, CLI & HPC Automation
 * Comprehensive Handbook-Grade Textbook Materials for Bachelor Thesis at VŠCHT Praha / ÚOCHB AV ČR
 */

export const M6_LECTURES = {
  // =========================================================================
  // LECTURE 6.1
  // =========================================================================
  "l6_1": {
    id: "l6_1",
    tag: "Core",
    relevance: 10,
    title: "6.1 Fragmentový de novo design & chemické scaffoldy (BRICS, Murcko, Transformers)",
    summary: "Překonání limitů atom-po-atomu generování, 16 retrosyntetických pravidel BRICS L1-L16, FragSequenceExplorer vs FragGraphExplorer a scaffold-based RL.",
    slides: [
      {
        title: "1. Proč fragmentový design? Překonání limitů atom-po-atomu generování",
        content: `Standardní generativní modely navrhující molekuly atom po atomu (např. čisté SMILES RNN) trpí zásadním limitem: prohledávají teoretický chemický prostor všech možných uspořádání atomů ($> 10^{60}$ struktur).
        <br><br>
        V tomto obrovském prostoru model často generuje:
        <ul>
          <li><strong>Synteticky nepřístupné motivy</strong>: Labilní acetaly, peroxidové vazby, nestabilní 4-členné nebo 8-členné cykly s vysokým pnutím.</li>
          <li><strong>Nestabilní tautomery a reaktivní funkční skupiny</strong> (např. acyl chloridy, Michaelovy akceptory), které způsobují nespecifickou toxicitu (PAINS).</li>
          <li><strong>Dlouhé syntetické cesty</strong>: Molekuly vyžadující 15 a více reakčních kroků s minimálním celkovým výtěžkem.</li>
        </ul>
        <br>
        <strong>Fragment-Based Drug Design (FBDD)</strong> tento problém elegantně řeší: molekuly nejsou skládány z jednotlivých atomů, nýbrž z <strong>předpřipravených synteticky validních stavebních bloků (synthonů)</strong> odvozených z komerčně dostupných chemikálií.
        <br><br>
        Druhým klíčovým konceptem je <strong>Scaffold Hopping (přeskakování mezi scaffoldy)</strong>: schopnost zachovat klíčové farmakoforové skupiny na periferii molekuly a radikálně obměnit centrální nosné jádro (scaffold) pro zlepšení rozpustnosti, permeability nebo obejití patentové ochrany konkurence.`
      },
      {
        title: "2. 16 BRICS retrosyntetických štěpných pravidel (L1 - L16)",
        content: `Pro dekompozici velkých knihoven molekul na synteticky relevantní synthony využívá DrugEx algoritmus <strong>BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures)</strong> (Degen et al., ChemMedChem 2008).
        <br><br>
        BRICS definuje 16 typů syntetických vazeb L1 až L16 odpovídajících robustním a spolehlivým chemickým reakcím:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">BRICS Pravidla L1 – L8 (Acylace, Sulfonamidy, C-C vazby)</div>
            <ul>
              <li><strong>L1</strong>: Primární/sekundární aminový dusík (např. pro tvorbu amidů a močovin).</li>
              <li><strong>L2</strong>: Karbonylový uhlík (karboxylová kyselina, acyl chlorid, ester).</li>
              <li><strong>L3</strong>: Sulfonylová skupina (sulfonyl chlorid, sulfonamid).</li>
              <li><strong>L4</strong>: Sulfonamidový dusíkatý atom.</li>
              <li><strong>L5</strong>: Aromatický uhlík (Suzuki-Miyaura, Stille, Buchwald-Hartwig coupling).</li>
              <li><strong>L6</strong>: Heteroaromatický uhlík (kondenzované heterocykly).</li>
              <li><strong>L7</strong>: Alifatický etherový kyslík (Williamsonova syntéza).</li>
              <li><strong>L8</strong>: Alifatický thioetherový sírový atom.</li>
            </ul>
          </div>
          <div class="compare-col right">
            <div class="compare-heading">BRICS Pravidla L9 – L16 (Aminace, Močoviny, Alifatické řetězce)</div>
            <ul>
              <li><strong>L9</strong>: Sekundární/terciární alifatický amin (reduktivní aminace).</li>
              <li><strong>L10</strong>: Alifatický halogen / odstupující skupina (nukleofilní substituce).</li>
              <li><strong>L11</strong>: Močovinový dusíkatý atom.</li>
              <li><strong>L12</strong>: Karbamátový / karbonátový kyslík.</li>
              <li><strong>L13</strong>: Guanidinový uhlík.</li>
              <li><strong>L14</strong>: Guanidinový dusík.</li>
              <li><strong>L15</strong>: Alifatický $sp^3$ uhlík (alkylační reakce, Grignardova činidla).</li>
              <li><strong>L16</strong>: Alifatický $sp^2/sp$ uhlík (Heckova reakce, Sonogashira coupling).</li>
            </ul>
          </div>
        </div>
        <br>
        Kromě BRICS podporuje DrugEx také algoritmus <strong>RECAP</strong> (11 pravidel) a dekompozici na <strong>Bemis-Murcko Frameworks</strong> (nosná jádra tvořená pouze kruhy a linkery bez postranních řetězců).`
      },
      {
        title: "3. Architektury fragmentových modelů: FragSequence vs FragGraph",
        content: `DrugEx integruje dvě pokročilé transformerové architektury pro generování řízené fragmenty:
        <br><br>
        <h4>1. FragSequenceExplorer (Sekvenční Transformer / GPT-2 styl)</h4>
        Pracuje se sekvenční reprezentací ve formátu <code>Fragment.Molecule</code>:
        <ul>
          <li>Vstupní fragment (např. <code>c1cnccn1</code> pro pyrazinové jádro) je zakódován pomocí <code>SequenceFragmentEncoder</code> jako fixní prompt (prefix).</li>
          <li>Autoregresivní dekodér s maskovanou samo-pozorností (Self-Attention) doplňuje zbytek řetězce atom po atomu.</li>
          <li>Garantuje zachování vloženého fragmentu v každé vygenerované struktuře.</li>
        </ul>
        <br>
        <h4>2. FragGraphExplorer (Grafový Transformer / Node-Edge Attention)</h4>
        Pracuje s tenzory molekulárních grafů o rozměru $(N, 80, 5)$:
        <ul>
          <li>Každý uzel kóduje typ atomu, index v grafu, vazebného souseda, formální náboj a binární příznak růstu (growth mask).</li>
          <li>Uzly fixního fragmentu jsou zmrazeny a model predikuje připojování nových atomů a vazeb z definovaných vazebných míst (attachment points).</li>
        </ul>`,
        code: `from drugex.data.fragments import FragmentCorpusEncoder, SequenceFragmentEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.data.corpus.vocabulary import VocSmiles

# 1. Nastavení fragmentéru BRICS
fragmenter = Fragmenter(
    n_frags=4,          # Maximálně 4 fragmenty na molekulu
    n_combs=2,          # Kombinace až 2 fragmentů
    method='brics',     # Použití BRICS štěpných pravidel
    max_bonds=75
)

# 2. Inicializace sekvenčního fragmentového kodéru
voc = VocSmiles.fromFile("models/Papyrus05.5.vocab", encode_frags=True)
encoder = FragmentCorpusEncoder(
    fragmenter=fragmenter,
    encoder=SequenceFragmentEncoder(voc, update_voc=False, throw=True),
    n_proc=16
)`
      },
      {
        title: "4. Scaffold-based RL: Fixace jádra a růst variabilních substituentů",
        content: `V řadě reálných projektů medicinální chemie má výzkumný tým definovaný slibný centrální skelet (např. pyrazinové, chinazolinové nebo benzimidazolové jádro) a cílem je optimalizovat postranní řetězce (R-groups) tak, aby molekula dosáhla maximální 3D tvarové a farmakoforové shody s vazebnou kavitou:
        <br><br>
        Tento přístup je implementován v <code>tutorial/advanced/scaffold_based.ipynb</code>:
        <ol>
          <li>Definice fixního fragmentu nebo kombinace fragmentů (např. <code>frags = ['c1cnccn1', 'c1cnccn1.c1ccsc1']</code>).</li>
          <li>Využití konvertoru <code>dummyMolsFromFragments()</code>, který vytvoří fiktivní molekulární páry pro inicializaci dataloaderu.</li>
          <li>Spuštění <code>FragSequenceExplorer</code> nebo <code>FragGraphExplorer</code> v prostředí <code>DrugExEnvironment</code> s 3D ROCS skórovačem.</li>
        </ol>
        <br>
        <h4>Aplikace pro bifunkční modulátory (PROTACs & IDP Binders):</h4>
        Tato metodika je ideální pro návrh bifunkčních molekul:
        <ul>
          <li>Na vstupu zadáme dva oddělené fragmenty: vazač proteinu A a vazač proteinu B spojené tečkou (<code>fragA.fragB</code>).</li>
          <li>DrugEx automaticky vygeneruje chemický linker (Linker Design) s optimální prostorovou délkou a konformační flexibilitou pro překlenutí vzdálenosti mezi oběma vazebnými místy.</li>
        </ul>`
      },
      {
        title: "5. Kompletní kód Scaffold-Based RL v Pythonu",
        content: `Následující kód demonstruje kompletní inicializaci fragmentového modelu pro fixní lešení:`,
        code: `import os
from drugex.data.datasets import GraphFragDataSet
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators import GraphTransformer
from drugex.training.explorers import FragGraphExplorer
from config import create_rdkit_environment

# 1. Definice požadovaných scaffoldů (např. pyrazin + thiofen)
scaffolds = ['c1cnccn1', 'c1cnccn1.c1ccsc1']

# 2. Příprava datové sady ze scaffoldů
encoder = FragmentCorpusEncoder(
    fragmenter=dummyMolsFromFragments(),
    encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
    pairs_splitter=None,
    n_proc=1
)

dataset = GraphFragDataSet("datasets/scaffolds.tsv", rewrite=True)
encoder.apply(scaffolds, encodingCollectors=[dataset])

# 3. Načtení natrénovaného grafového transforméru
voc = VocGraph.fromFile("models/Graph_FT.vocab")
agent = GraphTransformer(voc_trg=voc, use_gpus=(0,))
agent.loadStatesFromFile("models/Graph_FT.pkg")
prior = GraphTransformer(voc_trg=voc, use_gpus=(0,))
prior.loadStatesFromFile("models/Graph_FT.pkg")

# 4. Inicializace prostředí a explorátoru
env = create_rdkit_environment()
explorer = FragGraphExplorer(
    agent=agent,
    mutate=prior,
    env=env,
    epsilon=0.15,
    n_samples=500,
    use_gpus=(0,)
)

# 5. Spuštění fragmentového RL tréninku
# explorer.fit(epochs=50)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 6.2
  // =========================================================================
  "l6_2": {
    id: "l6_2",
    tag: "Core",
    relevance: 10,
    title: "6.2 Příkazová řádka DrugEx: drugex dataset, train & generate (CLI Mastery)",
    summary: "Kompletní matice parametrů CLI pro preprocessing, trénování a generování molekul a produkční automatizační Bash skripty.",
    slides: [
      {
        title: "1. Architektura a filozofie CLI rozhraní DrugEx",
        content: `Pro produkční nasazení v superpočítačových centrech (HPC) a integraci do automatizovaných bioinformatických pipeline je DrugEx vybaven robustním rozhraním příkazové řádky (CLI - Command Line Interface).
        <br><br>
        CLI řeší klíčové nevýhody interaktivních Jupyter notebooků:
        <ul>
          <li><strong>100% Reprodukovatelnost</strong>: Každé spuštění automaticky ukládá kompletní JSON konfiguraci parametrů (např. <code>dataset.json</code>, <code>train.json</code>) a podrobný logovací soubor.</li>
          <li><strong>Nulová režie grafického serveru</strong>: Běží v čistě bezhlavém (headless) režimu bez nutnosti GUI nebo webových serverů.</li>
          <li><strong>Paralelizace a dávkové zpracování</strong>: Umožňuje snadné spouštění desítek úloh paralelně přes plánovače úloh (Slurm, PBS Pro).</li>
        </ul>
        <br>
        CLI se skládá ze 3 hlavních modulů:
        <ol>
          <li><code>python -m drugex.dataset</code>: Standardizace, filtrace, fragmentace a tokenizace molekul.</li>
          <li><code>python -m drugex.train</code>: Předtrénování (PT), fine-tuning (FT) a reinforcement learning (RL).</li>
          <li><code>python -m drugex.generate</code>: Vzorkování z finálních modelů a skórování.</li>
        </ol>`
      },
      {
        title: "2. Kompletní matice parametrů: python -m drugex.dataset",
        content: `Následující tabulka obsahuje vyčerpávající přehled parametrů modulu pro přípravu dat:
        <br><br>
        <table class="data-table">
          <thead>
            <tr>
              <th>Přepínač (krátký / dlouhý)</th>
              <th>Typ</th>
              <th>Výchozí hodnota</th>
              <th>Popis a medicinální význam</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>-b, --base_dir</code></td>
              <td><code>str</code></td>
              <td><code>.</code></td>
              <td>Kořenový adresář projektu obsahující složky <code>data/</code> a <code>generators/</code>.</td>
            </tr>
            <tr>
              <td><code>-i, --input_file</code></td>
              <td><code>str</code></td>
              <td><em>povinný</em></td>
              <td>Cesta ke vstupnímu souboru TSV/CSV/SDF s chemickými strukturami.</td>
            </tr>
            <tr>
              <td><code>-mc, --mol_col</code></td>
              <td><code>str</code></td>
              <td><code>SMILES</code></td>
              <td>Název sloupce obsahujícího SMILES řetězce ve vstupním souboru.</td>
            </tr>
            <tr>
              <td><code>-o, --out</code></td>
              <td><code>str</code></td>
              <td><em>povinný</em></td>
              <td>Prefix názvu výstupních souborů datasetu a slovníku.</td>
            </tr>
            <tr>
              <td><code>-mt, --mol_type</code></td>
              <td><code>str</code></td>
              <td><code>smiles</code></td>
              <td>Typ molekulární reprezentace: <code>smiles</code> (sekvence) nebo <code>graph</code> (grafy).</td>
            </tr>
            <tr>
              <td><code>-nof, --no_filter</code></td>
              <td><code>flag</code></td>
              <td><code>False</code></td>
              <td>Vypne fyzikální filtry (MW, rotovatelné vazby). <strong>Doporučeno zapnout pro cílové ligandy</strong>.</td>
            </tr>
            <tr>
              <td><code>-fm, --frag_method</code></td>
              <td><code>str</code></td>
              <td><code>recap</code></td>
              <td>Metoda fragmentace: <code>brics</code> (16 pravidel) nebo <code>recap</code> (11 pravidel).</td>
            </tr>
            <tr>
              <td><code>-nf, --n_frags</code></td>
              <td><code>int</code></td>
              <td><code>4</code></td>
              <td>Maximální počet fragmentů vygenerovaných na jednu molekulu.</td>
            </tr>
            <tr>
              <td><code>-sfl, --selected_fragment</code></td>
              <td><code>str</code></td>
              <td><code>None</code></td>
              <td>SMILES fixního jádra/scaffoldu pro cílenou fragmentaci.</td>
            </tr>
            <tr>
              <td><code>-np, --n_proc</code></td>
              <td><code>int</code></td>
              <td><code>1</code></td>
              <td>Počet paralelních procesů pro standardizaci a kódování datasetu.</td>
            </tr>
          </tbody>
        </table>`
      },
      {
        title: "3. Kompletní matice parametrů: python -m drugex.train",
        content: `Následující tabulka shrnuje klíčové parametry trénovacího modulu <code>drugex.train</code>:
        <br><br>
        <table class="data-table">
          <thead>
            <tr>
              <th>Parametr</th>
              <th>Hodnoty</th>
              <th>Režim</th>
              <th>Popis funkce</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>-tm, --train_mode</code></td>
              <td><code>PT</code>, <code>FT</code>, <code>RL</code></td>
              <td>Všechny</td>
              <td>Režim tréninku: <code>PT</code> (Pre-training), <code>FT</code> (Fine-tuning), <code>RL</code> (Reinforcement Learning).</td>
            </tr>
            <tr>
              <td><code>-ag, --agent_model</code></td>
              <td><code>cesta k .pkg</code></td>
              <td>FT, RL</td>
              <td>Výchozí model agenta (např. předtrénovaný Papyrus model).</td>
            </tr>
            <tr>
              <td><code>-pr, --prior_model</code></td>
              <td><code>cesta k .pkg</code></td>
              <td>RL</td>
              <td>Mutační síť (Prior kotva) pro stabilizaci politiky.</td>
            </tr>
            <tr>
              <td><code>-e, --epochs</code></td>
              <td><code>int (50-200)</code></td>
              <td>Všechny</td>
              <td>Počet trénovacích epoch (pro FT typicky 100, pro RL 50).</td>
            </tr>
            <tr>
              <td><code>-bs, --batch_size</code></td>
              <td><code>int (32-256)</code></td>
              <td>Všechny</td>
              <td>Velikost trénovací dávky (batch size).</td>
            </tr>
            <tr>
              <td><code>-lr, --learning_rate</code></td>
              <td><code>float</code></td>
              <td>PT, FT</td>
              <td>Rychlost učení (pro PT $10^{-3}$, pro FT $10^{-4}$).</td>
            </tr>
            <tr>
              <td><code>-eps, --epsilon</code></td>
              <td><code>float (0.1-0.3)</code></td>
              <td>RL</td>
              <td>Míra explorace z mutační sítě (doporučeno $0.2$).</td>
            </tr>
            <tr>
              <td><code>-s, --scheme</code></td>
              <td><code>PRCD</code>, <code>PRTD</code>, <code>WS</code></td>
              <td>RL</td>
              <td>Optimalizační schéma: <code>PRCD</code> (Pareto Crowding Distance), <code>WS</code> (Weighted Sum).</td>
            </tr>
            <tr>
              <td><code>--sa_score</code></td>
              <td><code>flag</code></td>
              <td>RL</td>
              <td>Aktivuje penalizaci za obtížnou syntetickou dostupnost (SAScore).</td>
            </tr>
            <tr>
              <td><code>-gpu, --use_gpus</code></td>
              <td><code>'0'</code> nebo <code>'0,1'</code></td>
              <td>Všechny</td>
              <td>Seznam ID grafických karet pro akceleraci výpočtu.</td>
            </tr>
          </tbody>
        </table>`
      },
      {
        title: "4. Kompletní matice parametrů: python -m drugex.generate",
        content: `Parametry pro masivní generování a skórování molekul:
        <br><br>
        <ul>
          <li><code>-g, --generator</code>: Název finálního modelu bez přípony <code>.pkg</code> (např. <code>CCR2_rdkit_reinforced</code>).</li>
          <li><code>-n, --num</code>: Celkový počet molekul k vygenerování (např. <code>10000</code>).</li>
          <li><code>-bs, --batch_size</code>: Velikost batche pro paralelní inferenci na GPU (např. <code>1024</code>).</li>
          <li><code>--keep_invalid</code>: Ponechá ve výstupu i syntakticky nevalidní SMILES (standardně vypnuto).</li>
          <li><code>--keep_duplicates</code>: Ponechá duplicitní molekuly v rámci vygenerované sady.</li>
          <li><code>--keep_undesired</code>: Ponechá molekuly, které nesplnily prahy prostředí (užitečné pro statistickou analýzu distribucí).</li>
        </ul>`,
        code: `# Příklad vygenerování 5000 validních a unikátních molekul
python -m drugex.generate \\
    -b /home/kolar/learn_projects/drugex/workdir \\
    -g CCR2_rdkit_reinforced \\
    -n 5000 \\
    -bs 1024 \\
    -gpu 0`
      },
      {
        title: "5. Produkční robustní Bash batch pipeline skript",
        content: `Následující skript demonstruje kompletní, robustní pipeline řetězec v Bash s ošetřením chyb:`,
        code: `#!/usr/bin/env bash
# ==============================================================================
# DrugEx Production Pipeline: From Raw Data to Reinforced Leads
# ==============================================================================
set -euo pipefail

WORKDIR="/home/kolar/learn_projects/drugex/workdir"
DATA_IN="../data/benchmarks/CCR_HUMAN_AL.tsv"
PRETRAINED="../models/Papyrus05.5_smiles_rnn_PT.pkg"
GPUS="0"

mkdir -p "$WORKDIR/data" "$WORKDIR/generators" "$WORKDIR/new_molecules"

echo "========================================================="
echo " [1/4] Preprocessing & Tokenizace datasetu CCR2"
echo "========================================================="
python -m drugex.dataset \\
    -b "$WORKDIR" \\
    -i "$DATA_IN" \\
    -mc SMILES \\
    -o ccr2_corpus \\
    -mt smiles \\
    -nof \\
    -np 16

echo "========================================================="
echo " [2/4] Fine-Tuning modelu (100 epoch)"
echo "========================================================="
python -m drugex.train \\
    -tm FT \\
    -b "$WORKDIR" \\
    -i ccr2_corpus \\
    -ag "$PRETRAINED" \\
    -e 100 \\
    -bs 64 \\
    -lr 0.0001 \\
    -p 30 \\
    -gpu "$GPUS"

echo "========================================================="
echo " [3/4] Generování 10 000 kandidátních molekul"
echo "========================================================="
python -m drugex.generate \\
    -b "$WORKDIR" \\
    -g ccr2_corpus_FT \\
    -n 10000 \\
    -bs 1024 \\
    -gpu "$GPUS"

echo "========================================================="
echo " Pipeline úspěšně dokončena: $(date)"
echo "========================================================="`
      }
    ]
  },

  // =========================================================================
  // LECTURE 6.3
  // =========================================================================
  "l6_3": {
    id: "l6_3",
    tag: "WOW",
    relevance: 10,
    title: "6.3 Škálování na GPU klastru: Slurm skripty & správa paměti (MetaCentrum / IT4Innovations)",
    summary: "Produkční Slurm batch šablony, správa licencí OE_LICENSE, eliminace CPU thread oversubscription a prevence GPU VRAM memory leaků.",
    slides: [
      {
        title: "1. Architektura superpočítačové infrastruktury (MetaCentrum & IT4Innovations)",
        content: `Výpočetní experimenty kombinující de novo generování molekul s rozsáhlým 3D konformačním vzorkováním (stovky tisíc konformerů za běh) přesahují možnosti běžných pracovních stanic a vyžadují národní superpočítačovou infrastrukturu:
        <br><br>
        <ul>
          <li><strong>MetaCentrum (CESNET / e-INFRA CZ)</strong>:
            Národní gridová a klastrová infrastruktura. Poskytuje dedikované GPU uzly s kartami NVIDIA A100 (40/80 GB) a H100. Klíčovou vlastností je <strong>rychlý lokální NVMe scratch disk (proměnná <code>$SCRATCHDIR</code>)</strong>, který eliminuje I/O zpoždění při práci s tisíci dočasnými soubory.
          </li>
          <li><strong>IT4Innovations (Superpočítačové centrum Ostrava)</strong>:
            Superpočítače <em>Karolina</em> a <em>Barbora</em> řízené plánovačem úloh <strong>Slurm</strong>. Využívají paralelní sdílený souborový systém Lustre.
          </li>
        </ul>
        <br>
        Běh úloh na superpočítači probíhá v <strong>dávkovém (batch) režimu</strong>: úloha je definována shellovým skriptem s direktivami <code>#SBATCH</code> a odeslána do fronty příkazem <code>sbatch script.sh</code>.`
      },
      {
        title: "2. Produkční Slurm dávkový skript (Kompletní šablona)",
        content: `Následující skript představuje kompletní, v praxi prověřenou šablonu dávkové úlohy pro klastr Karolina / MetaCentrum:`,
        code: `#!/bin/bash
# ==============================================================================
# Slurm Batch Job: DrugEx MORL + 3D ROCS Shape Matching Pipeline
# Bachelor Thesis: Flexible Targets & IDP Drug Design (VŠCHT / ÚOCHB)
# ==============================================================================
#SBATCH --job-name=drugex_rocs_ccr2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/drugex_%j.out
#SBATCH --error=logs/drugex_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=student@vscht.cz

set -euo pipefail

echo "========================================================="
echo " Výpočet spuštěn na uzlu : $(hostname)"
echo " Datum a čas             : $(date)"
echo " Slurm Job ID            : $SLURM_JOB_ID"
echo " Alokováno CPU jader     : $SLURM_CPUS_PER_TASK"
echo " Alokována GPU           : $CUDA_VISIBLE_DEVICES"
echo "========================================================="

# 1. Načtení softwarových modulů
module purge
module load CUDA/12.1.1 Python/3.11.5 GCC/12.2.0 OpenEye/2023.2.0

# 2. Aktivace virtuálního prostředí
source /storage/praha1/home/kolar/.venvs/drugex/bin/activate

# 3. Nastavení kritických proměnných prostředí
export OE_LICENSE=/storage/praha1/home/kolar/licenses/oe_license.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 4. Příprava lokálního scratch disku a ošetření bezpečného úklidu
SCRATCH_DIR="\${SCRATCHDIR:-/tmp/drugex_\$SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
cd "$SLURM_SUBMIT_DIR"

trap 'echo "Zachycen signál ukončení. Kopíruji data zpět..."; cp -r "$SCRATCH_DIR"/* "$SLURM_SUBMIT_DIR/results/" || true; rm -rf "$SCRATCH_DIR"' EXIT TERM

# 5. Spuštění samotného výpočtu
python tutorial/advanced/rocs/prepare_models.py --epochs 100 --batch-size 64
python tutorial/advanced/rocs/generate_molecules.py --num-samples 10000

echo "========================================================="
echo " Výpočet úspěšně dokončen: $(date)"
echo "========================================================="`
      },
      {
        title: "3. Kritická past 1: CPU Thread Oversubscription a jeho eliminace",
        content: `Při nasazení na klastru se často vyskytuje zákeřná chyba vedoucí k <strong>desetinásobnému až padesátinásobnému zpomalení výpočtu</strong>, známá jako <em>CPU Thread Oversubscription</em>:
        <br><br>
        <h4>Jak k problému dochází?</h4>
        <ol>
          <li>Výpočetní uzly na superpočítačích mají typicky 64 až 128 fyzických CPU jader na socket.</li>
          <li>Knihovny lineární algebry (OpenBLAS, Intel MKL, OpenMP, NumExpr) ve výchozím nastavení detekují celkový počet jader na serveru a pro každou matematickou operaci vytvoří 128 vláken.</li>
          <li>Když Python skript spustí <code>multiprocessing.Pool(n_jobs=16)</code> nebo <code>RDKitROCSScorer(n_jobs=16)</code>, každý ze 16 procesů vytvoří 128 vláken.</li>
          <li>Vznikne $16 \\times 128 = \\mathbf{2048\\text{ konkurenčních vláken}}$, která bojují o 16 CPU jader přidělených vaší úloze!</li>
        </ol>
        <br>
        <h4>Důsledky:</h4>
        Procesor tráví $95\\%$ času přepínáním kontextu (Context Switching) a dochází k zahlcení L1/L2/L3 cache pamětí.
        <br><br>
        <h4>Řešení (Povinné pro všechny dávkové skripty):</h4>
        Před spuštěním Pythonu exportujte proměnné omezující počet podvláken na 1:`,
        code: `# Vypnutí vnitřního multitaskingu v C/Fortran knihovnách
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# V Python kódu:
# conf_gen = RDKitConformerGenerator(num_threads=1)`
      },
      {
        title: "4. Kritická past 2: GPU VRAM paměťové úniky v PyTorch a OOM prevence",
        content: `Během dlouhotrvajícího RL tréninku (50+ epoch) dochází v PyTorch k postupnému zaplňování grafické paměti (GPU VRAM), které často končí chybou <code>RuntimeError: CUDA out of memory</code>:
        <br><br>
        <h4>Hlavní příčiny a jejich řešení:</h4>
        <br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">1. Hromadění výpočetního grafu (Autograd)</div>
            Při vzorkování molekul z modelu nesmí PyTorch počítat gradienty. Vzorkovací smyčka musí být striktně uzavřena v kontextu:
            <br>
            <code>with torch.no_grad():</code>
            <br><br>
            Při ukládání ztráty do logu používejte <code>loss.item()</code> namísto pouhého <code>loss</code> (jinak se v paměti drží celý tenzorový graf).
          </div>
          <div class="compare-col right">
            <div class="compare-heading">2. Fragmentace PyTorch Caching Allocatoru</div>
            PyTorch neuvolňuje alokované bloky paměti zpět operačnímu systému, nýbrž je drží v mezipaměti. Při proměnných délkách SMILES sekvencí dochází k fragmentaci.
            <br><br>
            <strong>Řešení:</strong>
            <br>
            1. <code>export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128</code>
            <br>
            2. Volání <code>torch.cuda.empty_cache()</code> na konci každé epochy.
          </div>
        </div>`,
        alert: {
          type: "warning",
          title: "Správa dočasných souborů v RAM disku",
          text: "Při použití OpenEye ROCS skórovače se generují soubory <code>.oeb.gz</code>. Ujistěte se, že skórovač běží v paměťovém disku <code>/dev/shm</code> a okamžitě maže dočasné soubory po přečtení skóre, aby nedošlo k zaplnění diskového prostoru uzlu."
        }
      },
      {
        title: "5. Škálování pomocí Slurm Array Jobs a statistické replikáty",
        content: `Ve vědecké diplomové práci nesmí být výsledky založeny na jediném náhodném běhu. Pro publikovatelné výsledky je nutné provést <strong>statistické vyhodnocení (např. 10 až 20 nezávislých RL replikátů)</strong> s různými náhodnými semínky (seeds):
        <br><br>
        K tomuto účelu slouží <strong>Slurm Array Jobs</strong>:`,
        code: `#!/bin/bash
#SBATCH --job-name=drugex_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=1-10%5      # 10 replikátů, maximálně 5 běží současně
#SBATCH --output=logs/array_%A_%a.out

# Číslo úlohy v poli ($SLURM_ARRAY_TASK_ID) použijeme jako random seed
SEED=\$SLURM_ARRAY_TASK_ID
OUTPUT_DIR="experiments/run_seed_\${SEED}"
mkdir -p "\$OUTPUT_DIR"

echo "Spouštím replikát č. \$SEED s random seedem \$SEED..."

python run_experiment.py \\
    --seed "\$SEED" \\
    --output-dir "\$OUTPUT_DIR" \\
    --epochs 50 \\
    --epsilon 0.2`
      }
    ]
  }
};
