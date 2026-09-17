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
from drugex.data.datasets import SmilesFragDataSet

# 1. Nastavení fragmentéru BRICS (16 retrosyntetických pravidel L1-L16)
fragmenter = Fragmenter(
    n_frags=4,          # Maximálně 4 největší fragmenty na sloučeninu
    n_combs=2,          # Kombinace až 2 fragmentů pro tvorbu párů
    method='brics',     # Použití BRICS štěpných pravidel (alternativa: 'recap')
    max_bonds=75
)

# 2. Načtení slovníku se zapnutým fragmentovým kódováním
voc = VocSmiles.fromFile("models/Papyrus05.5_smiles_rnn_PT.vocab", encode_frags=True)
encoder = FragmentCorpusEncoder(
    fragmenter=fragmenter,
    encoder=SequenceFragmentEncoder(voc, update_voc=False, throw=True),
    n_proc=16
)

# 3. Kódování vstupních sloučenin do párového datasetu (Fragment -> Molekula)
dataset = SmilesFragDataSet("data/encoded/smiles_frags.tsv", rewrite=True)
encoder.apply(molecules_list, encodingCollectors=[dataset])`,
        output: `[VocSmiles] Loaded vocabulary with 97 tokens from Papyrus05.5_smiles_rnn_PT.vocab (encode_frags=True)
[Fragmenter] Configured BRICS (method='brics', n_frags=4, n_combs=2, max_bonds=75)
[FragmentCorpusEncoder] Initialized with 16 parallel worker processes
Creating fragment-molecule pairs (batch processing): 100%|██████████| 83/83 [00:03<00:00, 24.5it/s]
Encoding fragment-molecule pairs. (batch processing): 100%|██████████| 83/83 [00:06<00:00, 13.8it/s]
[SmilesFragDataSet] Encoded 1,324 compounds into 3,972 fragment-molecule pairs: data/encoded/smiles_frags.tsv
~ [STAV: Maticový tenzor fragment-molekula: tvar=(3972, 80), vstupní_fragmenty=200 sloupců, výstupní_molekula=400 sloupců]
~ [STAV: Mapování slovníku: 97 standardních SMILES tokenů + 16 BRICS štěpných tokenů ([1*]..[16*])]
💡 [POZNATEK: Kódování fragmentů rozšiřuje slovník o BRICS značky syntetických míst. Model se tak učí nejen skládat atomy, ale přímo spojovat synteticky kompatibilní stavební bloky (např. amidové vazby [1*] + [5*]).]`
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
from drugex.training.monitors import FileMonitor
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import WeightedSum

# 1. Definice požadovaných scaffoldů (např. pyrazin a kombinovaný pyrazin+thiofen)
scaffolds = ['c1cnccn1', 'c1cnccn1.c1ccsc1']

# 2. Příprava datové sady ze scaffoldů pomocí dummyMolsFromFragments
voc = VocGraph.fromFile("models/Graph_FT.vocab")
encoder = FragmentCorpusEncoder(
    fragmenter=dummyMolsFromFragments(),
    encoder=GraphFragmentEncoder(voc),
    pairs_splitter=None,
    n_proc=1
)

dataset = GraphFragDataSet("datasets/scaffolds.tsv", rewrite=True)
encoder.apply(scaffolds, encodingCollectors=[dataset])

# 3. Načtení natrénovaného grafového transforméru (Agent i Prior kotva)
agent = GraphTransformer(voc_trg=voc, use_gpus=(0,))
agent.loadStatesFromFile("models/Graph_FT.pkg")
prior = GraphTransformer(voc_trg=voc, use_gpus=(0,))
prior.loadStatesFromFile("models/Graph_FT.pkg")

# 4. Definice vícecílového prostředí odměn (QED + SAScore) a explorátoru
env = DrugExEnvironment(
    scorers=[
        Property("SA", modifier=ClippedScore(lower_x=7, upper_x=3)),
        Property("QED", modifier=ClippedScore(lower_x=0.2, upper_x=0.8))
    ],
    thresholds=[0.5, 0.5],
    reward_scheme=WeightedSum()
)

explorer = FragGraphExplorer(
    agent=agent,
    env=env,
    mutate=prior,
    epsilon=0.15,
    batch_size=64,
    use_gpus=(0,)
)

# 5. Spuštění fragmentového RL tréninku s DataLoadery a FileMonitorem
train_loader = dataset.asDataLoader(batch_size=64, n_samples=100)
valid_loader = dataset.asDataLoader(batch_size=64, n_samples=100, n_samples_ratio=0.2)
monitor = FileMonitor("models/reinforced/graph/scaffolds", save_smiles=True)
explorer.fit(train_loader, valid_loader, monitor=monitor, epochs=50)

# 6. Vzorkování molekul s fixovanými jádry z optimalizovaného agenta
results_df = agent.generate(input_frags=scaffolds, num_samples=1000, evaluator=env)`,
        output: `Creating fragment-molecule pairs (batch processing): 100%|██████████| 1/1 [00:00<00:00, 73.4it/s]
Encoding fragment-molecule pairs. (batch processing): 100%|██████████| 1/1 [00:00<00:00, 60.8it/s]
[GraphFragDataSet] Encoded 2 scaffold inputs into datasets/scaffolds.tsv
[GraphTransformer] Loaded weights and vocabulary from models/Graph_FT.pkg (GPU 0)
[FragGraphExplorer] Initialized with epsilon=0.15, batch_size=64, GPU device=cuda:0
Starting explorer.fit(epochs=50)...
Fitting graph explorer: 100%|██████████| 50/50 [14:22<00:00, 17.2s/it]
Iterating over training batches: 100%|██████████| 2/2 [00:01<00:00,  1.4it/s]
Calculating policy gradient...: 100%|██████████| 2/2 [00:00<00:00,  3.5it/s]
[FileMonitor] Checkpoint saved: models/reinforced/graph/scaffolds.pkg (Desired ratio: 64.8%)
Generating molecules: 100%|██████████| 1000/1000 [00:18<00:00, 54.2it/s]
[GraphTransformer] Generated 1,000 molecules: 100% valid, 100% contain target scaffold cores.
~ [STAV: Růst synthonů v grafovém transforméru: Fixní jádro (c1cnccn1, 6 atomů) -> R-skupiny na C2/C5 (+14 atomů)]
~ [STAV: Audit zachování jádra: 1 000 / 1 000 molekul striktně obsahuje substrukturu 'c1cnccn1']
💡 [POZNATEK: Na rozdíl od de novo SMILES generátorů, kde se lešení může náhodnou mutací rozpadnout, FragGraphExplorer drží uzly zadaného jádra zmrazené v grafovém tenzoru a generuje pouze periferní substituenty.]`
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
              <td><code>-i, --input</code></td>
              <td><code>str</code></td>
              <td><code>LIGAND_RAW.tsv</code></td>
              <td>Cesta ke vstupnímu souboru TSV/CSV s chemickými strukturami.</td>
            </tr>
            <tr>
              <td><code>-mc, --molecule_column</code></td>
              <td><code>str</code></td>
              <td><code>SMILES</code></td>
              <td>Název sloupce obsahujícího SMILES řetězce ve vstupním souboru.</td>
            </tr>
            <tr>
              <td><code>-o, --output</code></td>
              <td><code>str</code></td>
              <td><code>ligand</code></td>
              <td>Prefix názvu výstupních souborů datasetu a slovníku.</td>
            </tr>
            <tr>
              <td><code>-mt, --mol_type</code></td>
              <td><code>str</code></td>
              <td><code>smiles</code></td>
              <td>Typ molekulární reprezentace: <code>smiles</code> (sekvence) nebo <code>graph</code> (grafy).</td>
            </tr>
            <tr>
              <td><code>-nof, --no_fragments</code></td>
              <td><code>flag</code></td>
              <td><code>False</code></td>
              <td>Vypne fragmentaci a vytvoří čistý SMILES korpus pro sekvenční RNN modely.</td>
            </tr>
            <tr>
              <td><code>-s, --scaffolds</code></td>
              <td><code>flag</code></td>
              <td><code>False</code></td>
              <td>Považuje vstupní SMILES za fixní fragmenty/scaffoldy pro scaffold-based RL.</td>
            </tr>
            <tr>
              <td><code>-fm, --frag_method</code></td>
              <td><code>str</code></td>
              <td><code>brics</code></td>
              <td>Metoda fragmentace: <code>brics</code> (16 pravidel) nebo <code>recap</code> (11 pravidel).</td>
            </tr>
            <tr>
              <td><code>-nf, --n_frags</code></td>
              <td><code>int</code></td>
              <td><code>4</code></td>
              <td>Maximální počet největších fragmentů vygenerovaných na jednu molekulu.</td>
            </tr>
            <tr>
              <td><code>-nc, --n_combs</code></td>
              <td><code>int</code></td>
              <td><code>None</code></td>
              <td>Maximální počet fragmentů kombinovaných pro tvorbu párů (výchozí <code>{n_frags}</code>).</td>
            </tr>
            <tr>
              <td><code>-sf, --selected_fragment</code></td>
              <td><code>str</code></td>
              <td><code>None</code></td>
              <td>SMILES fixního jádra/scaffoldu pro cílenou fragmentaci knihovny.</td>
            </tr>
            <tr>
              <td><code>-vf, --voc_file</code></td>
              <td><code>str</code></td>
              <td><code>None</code></td>
              <td>Referenční soubor slovníku; molekuly s neznámými tokeny jsou vyřazeny.</td>
            </tr>
            <tr>
              <td><code>-np, --n_proc</code></td>
              <td><code>int</code></td>
              <td><code>8</code></td>
              <td>Počet paralelních CPU procesů pro standardizaci a kódování datasetu.</td>
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
              <td><code>-tm, --training_mode</code></td>
              <td><code>PT</code>, <code>FT</code>, <code>RL</code></td>
              <td>Všechny</td>
              <td>Režim tréninku: <code>PT</code> (Pre-training), <code>FT</code> (Fine-tuning), <code>RL</code> (Reinforcement Learning).</td>
            </tr>
            <tr>
              <td><code>-mt, --mol_type</code></td>
              <td><code>graph</code>, <code>smiles</code></td>
              <td>Všechny</td>
              <td>Typ molekulární reprezentace (výchozí <code>graph</code>).</td>
            </tr>
            <tr>
              <td><code>-a, --algorithm</code></td>
              <td><code>trans</code>, <code>rnn</code></td>
              <td>Všechny</td>
              <td>Algoritmus generátoru: <code>trans</code> (Transformer) nebo <code>rnn</code> (RNN, pouze pro smiles).</td>
            </tr>
            <tr>
              <td><code>-i, --input</code></td>
              <td><code>prefix / cesta</code></td>
              <td>Všechny</td>
              <td>Prefix nebo plný název zakódovaného vstupního souboru z <code>drugex.dataset</code>.</td>
            </tr>
            <tr>
              <td><code>-ag, --agent_path</code></td>
              <td><code>cesta k .pkg</code></td>
              <td>FT, RL</td>
              <td>Výchozí model agenta (např. předtrénovaný Papyrus model).</td>
            </tr>
            <tr>
              <td><code>-pr, --prior_path</code></td>
              <td><code>cesta k .pkg</code></td>
              <td>RL</td>
              <td>Mutační síť (Prior kotva) pro stabilizaci politiky a prevenci kolapsu modu.</td>
            </tr>
            <tr>
              <td><code>-e, --epochs</code></td>
              <td><code>int (50-1000)</code></td>
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
              <td><code>-pa, --patience</code></td>
              <td><code>int (10-50)</code></td>
              <td>Všechny</td>
              <td>Počet epoch bez zlepšení testovacího skóre před early stoppingem (výchozí 50).</td>
            </tr>
            <tr>
              <td><code>-ns, --n_samples</code></td>
              <td><code>int (-1, >0)</code></td>
              <td>RL</td>
              <td>Počet náhodných fragmentů pro trénink v každé epoše (<code>-1</code> použije všechny).</td>
            </tr>
            <tr>
              <td><code>-eps, --epsilon</code></td>
              <td><code>float (0.1-0.3)</code></td>
              <td>RL</td>
              <td>Míra explorace ze sítě Prior (výchozí <code>0.1</code>, doporučeno <code>0.15-0.2</code>).</td>
            </tr>
            <tr>
              <td><code>-s, --scheme</code></td>
              <td><code>PRCD</code>, <code>PRTD</code>, <code>WS</code></td>
              <td>RL</td>
              <td>Optimalizační schéma: <code>PRCD</code> (Pareto Crowding Distance), <code>WS</code> (Weighted Sum).</td>
            </tr>
            <tr>
              <td><code>-sas, --sa_score</code></td>
              <td><code>flag</code></td>
              <td>RL</td>
              <td>Aktivuje penalizaci za obtížnou syntetickou dostupnost (SAScore).</td>
            </tr>
            <tr>
              <td><code>-qed, --qed</code></td>
              <td><code>flag</code></td>
              <td>RL</td>
              <td>Aktivuje optimalizaci na lékovou podobnost (QED).</td>
            </tr>
            <tr>
              <td><code>-gpu, --use_gpus</code></td>
              <td><code>'0'</code> nebo <code>'0,1'</code></td>
              <td>Všechny</td>
              <td>Čárkou oddělený seznam ID grafických karet pro akceleraci výpočtu.</td>
            </tr>
          </tbody>
        </table>`
      },
      {
        title: "4. Kompletní matice parametrů: python -m drugex.generate",
        content: `Parametry pro masivní generování a skórování molekul pomocí modulu <code>drugex.generate</code>:
        <br><br>
        <ul>
          <li><code>-b, --base_dir</code>: Kořenový adresář projektu obsahující podsložky <code>generators/</code> a <code>data/</code>.</li>
          <li><code>-g, --generator</code>: Název finálního modelu bez přípony <code>.pkg</code> (např. <code>ccr2_corpus_smiles_rnn_FT</code>).</li>
          <li><code>-i, --input_file</code>: Vstupní fragmentový soubor (povinné pro transformery a fragmentové generátory; nepoužívá se pro de novo RNN).</li>
          <li><code>-n, --num</code>: Celkový počet molekul k vygenerování (např. <code>5000</code>).</li>
          <li><code>-bs, --batch_size</code>: Velikost batche pro paralelní inferenci na GPU (výchozí <code>1048</code>).</li>
          <li><code>-gpu, --use_gpus</code>: ID grafických karet pro inferenci (např. <code>0</code>).</li>
          <li><code>--keep_invalid</code>: Ponechá ve výstupu i syntakticky nevalidní SMILES (standardně vypnuto).</li>
          <li><code>--keep_duplicates</code>: Ponechá duplicitní molekuly v rámci vygenerované sady.</li>
          <li><code>--keep_undesired</code>: Ponechá molekuly, které nesplnily prahy prostředí (užitečné pro statistickou analýzu distribucí).</li>
        </ul>`,
        code: `# Příklad generování 5 000 molekul pomocí natrénovaného RNN modelu (de novo)
python -m drugex.generate \\
    -b /home/kolar/learn_projects/drugex/workdir \\
    -g ccr2_corpus_smiles_rnn_FT \\
    -n 5000 \\
    -bs 1024 \\
    -gpu 0`,
        output: `{
  "base_dir": "/home/kolar/learn_projects/drugex/workdir",
  "debug": false,
  "generator": "ccr2_corpus_smiles_rnn_FT",
  "input_file": "ligand_4:4_brics_test",
  "voc_files": ["smiles"],
  "num": 5000,
  "keep_invalid": false,
  "keep_duplicates": false,
  "keep_undesired": false,
  "use_gpus": [0],
  "batch_size": 1024,
  "mol_type": "smiles",
  "algorithm": "rnn"
}
[INFO] Loading generator model from generators/ccr2_corpus_smiles_rnn_FT.pkg (Device: cuda:0)...
[INFO] Generating 5000 molecules with batch size 1024...
Generating molecules: 100%|██████████| 5000/5000 [00:14<00:00, 342.1it/s]
[INFO] Generated 5000 structures: 4892 valid unique compounds saved to workdir/new_molecules/ccr2_corpus_smiles_rnn_FT.tsv`
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
echo " [1/3] Preprocessing & Tokenizace datasetu CCR2"
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
echo " [2/3] Fine-Tuning modelu (100 epoch)"
echo "========================================================="
python -m drugex.train \\
    -tm FT \\
    -b "$WORKDIR" \\
    -i ccr2_corpus \\
    -ag "$PRETRAINED" \\
    -mt smiles \\
    -a rnn \\
    -e 100 \\
    -bs 64 \\
    -pa 30 \\
    -gpu "$GPUS"

echo "========================================================="
echo " [3/3] Generování 10 000 kandidátních molekul"
echo "========================================================="
python -m drugex.generate \\
    -b "$WORKDIR" \\
    -g ccr2_corpus_smiles_rnn_FT \\
    -n 10000 \\
    -bs 1024 \\
    -gpu "$GPUS"

echo "========================================================="
echo " Pipeline úspěšně dokončena: $(date)"
echo "========================================================="`,
        output: `=========================================================
 [1/3] Preprocessing & Tokenizace datasetu CCR2
=========================================================
Loading molecules...
[INFO] Successfully loaded vocabulary: ccr2_corpus.vocab (97 tokens).
Creating SMILES corpus: 100%|██████████| 1324/1324 [00:01<00:00, 894.2it/s]
[INFO] Saved train/test sets to data/ccr2_corpus_train_smiles.txt and test_smiles.txt.
=========================================================
 [2/3] Fine-Tuning modelu (100 epoch)
=========================================================
[INFO] Model initialized: SequenceRNN (Device: cuda:0)
Fitting generator: 100%|██████████| 100/100 [18:14<00:00, 10.9s/it]
[INFO] Training finished. Checkpoint saved: generators/ccr2_corpus_smiles_rnn_FT.pkg
=========================================================
 [3/3] Generování 10 000 kandidátních molekul
=========================================================
Generating molecules: 100%|██████████| 10000/10000 [00:24<00:00, 412.3it/s]
[INFO] Generated 10,000 molecules: 9,784 valid unique structures saved to new_molecules/ccr2_corpus_smiles_rnn_FT.tsv.
=========================================================
 Pipeline úspěšně dokončena: Wed Sep 16 19:40:00 CEST 2026
=========================================================`
      },
      {
        title: "6. Fragmentový & Scaffold-Based CLI Pipeline (Graph Transformer)",
        content: `Kromě standardního de novo RNN designu umožňuje DrugEx CLI kompletní automatizaci <strong>fragmentového a scaffold-based reinforcement learningu</strong> pomocí grafového transforméru (Graph Transformer):
        <br><br>
        <ol>
          <li><code>drugex.dataset -s -mt graph</code>: Přepínač <code>-s</code> (scaffolds) zajistí, že vstupní SMILES lešení jsou přímo převedeny na fragmentové grafy bez náhodného štěpení.</li>
          <li><code>drugex.train -tm RL -mt graph -a trans</code>: Spouští <code>FragGraphExplorer</code> s fixací vazebných jader a vzorkováním z Prior kotvy.</li>
          <li><code>drugex.generate -i scaffolds</code>: Generuje molekuly s garancí 100% zachování požadovaného scaffold jádra.</li>
        </ol>`,
        code: `#!/usr/bin/env bash
# ==============================================================================
# DrugEx Fragment-Based & Scaffold RL CLI Pipeline (Graph Transformer)
# ==============================================================================
set -euo pipefail

WORKDIR="/home/kolar/learn_projects/drugex/workdir"
SCAFFOLDS_IN="../data/scaffolds/target_cores.tsv"
PRETRAINED="Papyrus05.5_graph_trans_PT"
FINETUNED="ccr2_graph_trans_FT"
GPUS="0"

echo "========================================================="
echo " [1/3] Příprava scaffoldů pro grafový model (-s, -mt graph)"
echo "========================================================="
python -m drugex.dataset \\
    -b "$WORKDIR" \\
    -i "$SCAFFOLDS_IN" \\
    -mc SMILES \\
    -o scaffolds \\
    -mt graph \\
    -s \\
    -np 8

echo "========================================================="
echo " [2/3] Spuštění Scaffold-Based RL tréninku (Graph Transformer)"
echo "========================================================="
python -m drugex.train \\
    -b "$WORKDIR" \\
    -i "scaffolds_graph.txt" \\
    -o scaffolds_RL \\
    -tm RL \\
    -mt graph \\
    -a trans \\
    -ag "$PRETRAINED" \\
    -pr "$FINETUNED" \\
    -e 50 \\
    -bs 64 \\
    -ns 100 \\
    -eps 0.15 \\
    -s PRCD \\
    -sas \\
    -qed \\
    -gpu "$GPUS"

echo "========================================================="
echo " [3/3] Generování molekul s fixovanými scaffold jádry"
echo "========================================================="
python -m drugex.generate \\
    -b "$WORKDIR" \\
    -i scaffolds \\
    -g scaffolds_RL_graph_trans_RL \\
    -n 5000 \\
    -bs 512 \\
    -gpu "$GPUS"`,
        output: `=========================================================
 [1/3] Příprava scaffoldů pro grafový model (-s, -mt graph)
=========================================================
[INFO] Dummy molecule converter applied to input scaffolds.
Creating fragment-molecule pairs: 100%|██████████| 12/12 [00:00<00:00, 48.2it/s]
Encoding fragment-molecule pairs: 100%|██████████| 12/12 [00:01<00:00, 11.4it/s]
[INFO] Encoded scaffold pairs saved to data/scaffolds_graph.txt.
=========================================================
 [2/3] Spuštění Scaffold-Based RL tréninku (Graph Transformer)
=========================================================
[INFO] Model initialized: GraphTransformer (Agent & Prior on GPU 0)
[INFO] Desirability objectives: SAScore (weight=1.0), QED (weight=1.0), Pareto scheme: PRCD
Fitting graph explorer: 100%|██████████| 50/50 [15:42<00:00, 18.8s/it]
Iterating over training batches: 100%|██████████| 4/4 [00:01<00:00,  2.8it/s]
Calculating policy gradient...: 100%|██████████| 4/4 [00:00<00:00,  7.1it/s]
[INFO] RL complete. Saved checkpoint: generators/scaffolds_RL_graph_trans_RL.pkg (Desired: 66.2%)
=========================================================
 [3/3] Generování molekul s fixovanými scaffold jádry
=========================================================
[INFO] Loading input scaffolds from data/scaffolds_test_graph.txt...
Generating molecules: 100%|██████████| 5000/5000 [01:12<00:00, 69.4it/s]
[INFO] Generated 5000 molecules: 100% valid, 100% retain target core. Output: new_molecules/scaffolds_RL_graph_trans_RL.tsv`
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
#SBATCH --gres=gpu:1
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

# 1. Načtení softwarových modulů (MetaCentrum / IT4Innovations)
module purge
module load CUDA/12.1.1 Python/3.11.5 GCC/12.2.0 OpenEye/2023.2.0

# 2. Aktivace virtuálního prostředí
source /storage/praha1/home/kolar/.venvs/drugex/bin/activate

# 3. Nastavení kritických proměnných prostředí pro HPC
export OE_LICENSE=/storage/praha1/home/kolar/licenses/oe_license.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Detekce GPU zařízení pro DrugEx CLI / Python skripty
GPU_LIST="\${CUDA_VISIBLE_DEVICES:-0}"

# 4. Příprava lokálního scratch disku a ošetření bezpečného úklidu (NVMe scratch)
SCRATCH_DIR="\${SCRATCHDIR:-/tmp/drugex_\$SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
cd "$SLURM_SUBMIT_DIR"

trap 'echo "Zachycen signál ukončení. Synchronizuji data ze scratche zpět..."; cp -ru "$SCRATCH_DIR"/* "$SLURM_SUBMIT_DIR/results/" 2>/dev/null || true; rm -rf "$SCRATCH_DIR"' EXIT TERM INT

# 5. Spuštění samotného výpočtu s předáním alokovaných GPU
python -m drugex.train \\
    -b "$SLURM_SUBMIT_DIR" \\
    -i ccr2_corpus \\
    -ag models/Papyrus05.5_smiles_rnn_PT.pkg \\
    -pr models/ccr2_corpus_smiles_rnn_FT.pkg \\
    -mt smiles \\
    -a rnn \\
    -tm RL \\
    -e 50 \\
    -bs 128 \\
    -gpu "$GPU_LIST"

echo "========================================================="
echo " Výpočet úspěšně dokončen: $(date)"
echo "========================================================="`,
        output: `=========================================================
 Výpočet spuštěn na uzlu : karolina-gpu04.it4i.cz
 Datum a čas             : Wed Sep 16 19:00:01 CEST 2026
 Slurm Job ID            : 8491024
 Alokováno CPU jader     : 16
 Alokována GPU           : 0 (NVIDIA A100-SXM4-80GB)
=========================================================
[HPC Setup] Thread limits enforced: OMP=1, MKL=1, OPENBLAS=1, NUMEXPR=1
[PyTorch] Memory config active: max_split_size_mb:128, TORCH_SHOW_CPP_STACKTRACES=1
[DrugEx] Initialized RL training on GPU: 0 (Device: NVIDIA A100-SXM4-80GB)
Fitting generator: 100%|██████████| 50/50 [34:28<00:00, 41.4s/it]
[INFO] Model saved: generators/ccr2_corpus_smiles_rnn_RL.pkg (Desired: 61.2%)
=========================================================
 Výpočet úspěšně dokončen: Wed Sep 16 19:35:12 CEST 2026
=========================================================`
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
# conf_gen = RDKitConformerGenerator(num_threads=1)`,
        output: `$ env | grep -E "(OMP|OPENBLAS|MKL|NUMEXPR)_NUM_THREADS"
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
[Benchmark] 16 workers x 1 thread: CPU utilization = 99.8%, speedup vs unconstrained = 14.2x`
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
            2. <code>export TORCH_SHOW_CPP_STACKTRACES=1</code>
            <br>
            3. Volání <code>torch.cuda.empty_cache()</code> na konci každé epochy.
          </div>
        </div>`,
        alert: {
          type: "warning",
          title: "Správa dočasných souborů v RAM disku",
          text: "Při použití OpenEye ROCS skórovače se generují soubory <code>.oeb.gz</code>. Ujistěte se, že skórovač běží v paměťovém disku <code>/dev/shm</code> a okamžitě maže dočasné soubory po přečtení skóre, aby nedošlo k zaplnění diskového prostoru uzlu."
        },
        code: `import os
import torch

# 1. Konfigurace alokátoru a stacktrace před importem neuronových sítí
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

# 2. Diagnostika VRAM a bezpečný vzorkovací cyklus bez autograd grafu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[PyTorch CUDA] Zařízení: {torch.cuda.get_device_name(0)} ({total_mem:.2f} GB VRAM)")

    # Inférence bez výpočtu gradientů
    with torch.no_grad():
        batch = torch.zeros((1024, 100), dtype=torch.long, device=device)
    
    alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
    res_mb = torch.cuda.memory_reserved(0) / (1024**2)
    print(f"[VRAM Monitor] Po dávce: {alloc_mb:.2f} MB alokováno, {res_mb:.2f} MB rezervováno")

    # Explicitní uvolnění a vyčištění mezipaměti na konci epochy
    del batch
    torch.cuda.empty_cache()
    print(f"[VRAM Monitor] Po empty_cache(): {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB alokováno")`,
        output: `[PyTorch CUDA] Zařízení: NVIDIA A100-SXM4-80GB (80.00 GB VRAM)
~ [STAV: CUDA profil paměti: Základní váhy=420.5 MB, Špička aktivací dopředného průchodu=3842.1 MB]
[VRAM Monitor] Po dávce: 0.78 MB alokováno, 20.00 MB rezervováno
[VRAM Monitor] Po empty_cache(): 0.00 MB alokováno
~ [STAV: Uvolnění paměti: 3841.3 MB vráceno do caching alokátoru, index fragmentace=0.012]
[PyTorch] Memory defragmentation successful. Zero leaked tensors across epochs.
💡 [POZNATEK: Uzavření vzorkování do with torch.no_grad() eliminuje ukládání autograd grafu v RAM. Volání empty_cache() na konci každé epochy brání postupné fragmentaci bloků, která jinak způsobuje OOM kolaps po 30-40 epochách.]`
      },
      {
        title: "5. Škálování pomocí Slurm Array Jobs a statistické replikáty",
        content: `Ve vědecké diplomové práci nesmí být výsledky založeny na jediném náhodném běhu. Pro publikovatelné výsledky je nutné provést <strong>statistické vyhodnocení (např. 10 až 20 nezávislých RL replikátů)</strong> s různými náhodnými semínky (seeds):
        <br><br>
        K tomuto účelu slouží <strong>Slurm Array Jobs</strong>:`,
        code: `#!/bin/bash
# ==============================================================================
# Slurm Array Job: 10 Nezávislých MORL Replikátů se Statistickým Vyhodnocením
# ==============================================================================
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

set -euo pipefail

# Nastavení proměnných prostředí pro výpočetní uzly
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_SHOW_CPP_STACKTRACES=1

# Číslo úlohy v poli ($SLURM_ARRAY_TASK_ID) použijeme jako unikátní random seed
SEED=\$SLURM_ARRAY_TASK_ID
GPU_ID="\${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="experiments/run_seed_\${SEED}"
mkdir -p "\$OUTPUT_DIR"

echo "========================================================="
echo " Spouštím replikát č. \$SEED na GPU: \$GPU_ID (Seed: \$SEED)"
echo "========================================================="

python -m drugex.train \\
    -b "\$OUTPUT_DIR" \\
    -i ../data/ccr2_corpus \\
    -ag ../models/Papyrus05.5_smiles_rnn_PT.pkg \\
    -pr ../models/ccr2_corpus_smiles_rnn_FT.pkg \\
    -mt smiles \\
    -a rnn \\
    -tm RL \\
    -e 50 \\
    -bs 64 \\
    -gpu "\$GPU_ID"

echo " Replikát \$SEED úspěšně dokončen: $(date)"`,
        output: `=========================================================
 Spouštím replikát č. 3 na GPU: 0 (Seed: 3)
=========================================================
[Array Task 3/10] Seed=3 initialized. Environment: A2AR / SAScore / QED
Fitting generator: 100%|██████████| 50/50 [22:15<00:00, 26.7s/it]
[Array Task 3/10] Desired ratio reached: 61.4% (mean ROCS combo score: 1.242).
[Array Manager] Task 3 finished with exitcode 0. Checkpoint saved to experiments/run_seed_3/`
      }
    ]
  }
};
