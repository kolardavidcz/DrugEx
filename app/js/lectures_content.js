/**
 * DrugEx Hub — Comprehensive Multi-Module Lecture Slide Database (18 Lectures)
 */

export const LECTURE_DATA = {
  // ==========================================
  // MODULE 1: DE NOVO GENERATION & REPRESENTATIONS
  // ==========================================
  "l1_1": {
    id: "l1_1",
    tag: "Core",
    relevance: 10,
    title: "1.1 Molekulární reprezentace, tokenizace & slovník (VocSmiles)",
    summary: "1D SMILES vs 2D grafy vs fragmentové vaky, tokenizační smlouva VocSmiles a SmilesStandardizer.",
    slides: [
      {
        title: "1. Úvod do molekulárních reprezentací pro generativní AI",
        content: `Při aplikaci hlubokého učení v medicinální chemii je zásadní volba reprezentace chemické struktury. V praxi se setkáváme se třemi hlavními modalitami:
        <ul>
          <li><strong>1D Textové sekvence (SMILES, SELFIES, DeepSMILES)</strong>: Úsporné lineární kódování molekulárního grafu.</li>
          <li><strong>2D Molekulární grafy</strong>: Přímá topologická reprezentace atomů (uzly) a kovalentních vazeb (hrany).</li>
          <li><strong>Fragmentové slovníky (BRICS, Murcko)</strong>: Reprezentace chemických stavebních bloků pro synteticky realizovatelný design.</li>
        </ul>`,
        compare: {
          leftTitle: "1D SMILES (Lineární řetězec)",
          leftContent: "Výhody: Kompaktní paměť, snadné použití s jazykovými modely (RNN, GPT).<br>Nevýhody: Nebezpečí nevalidní syntaxe (neuzavřené cykly, nespárované závorky).",
          rightTitle: "2D Graf (Matice sousednosti)",
          rightContent: "Výhody: 100% syntaktická platnost, přirozené scaffold-constrained generování.<br>Nevýhody: Vyšší výpočetní náročnost, složitější difúzní nebo autoregresní vzorkování."
        }
      },
      {
        title: "2. Tokenizační kontrakt DrugEx: VocSmiles",
        content: `Třída <code>VocSmiles</code> převádí SMILES řetězec na posloupnost diskrétních celočíselných tokenů. Zpracovává nejen jednopísmenné atomy (<code>C</code>, <code>N</code>, <code>O</code>), ale i vícerozměrné chemické entity, chirální flagy a aromatické heteroatomy:`,
        code: `from drugex.data.corpus.vocabulary import VocSmiles

# Inicializace slovníku ze SMILES souboru
voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)
print(f"Velikost slovníku: {voc.size} unikátních tokenů")

# Kódování a dekódování molekuly
smiles = "Cc1ccccc1NC(=O)C"
encoded_tensor = voc.encode([smiles])  # [START, 'C', 'c', '1', ..., END]
decoded_smiles = voc.decode(encoded_tensor)
print(f"Dekódováno: {decoded_smiles}")`,
        alert: {
          type: "note",
          title: "Řídicí tokeny",
          text: "VocSmiles automaticky rezervuje speciální indexy pro tokeny <code>START</code> (GO), <code>END</code> (STOP) a <code>PAD</code> (zarovnání délky v batchi)."
        }
      },
      {
        title: "3. Sanitační pipeline: SmilesStandardizer",
        content: `Trénovací data z databází jako Papyrus nebo ChEMBL obsahují šum: různé soli, nekonzistentní tautomery, náboje a směsi stereoizomerů. <code>SmilesStandardizer</code> sjednocuje data do kanonické formy:`,
        code: `from drugex.molecules.converters.standardizers import SmilesStandardizer

standardizer = SmilesStandardizer(
    drop_unspecified=True,  # Odstranit nedefinované stereochemii
    drop_isotopes=True      # Nahradit izotopy standardními atomy
)

raw_smi = "CC(=O)[O-].[Na+]"  # Octan sodný se solí
clean_smi = standardizer(raw_smi)
print(f"Kanonický ligand: {clean_smi}")  # CC(=O)O (odsoleno a neutralizováno)`
      }
    ]
  },

  "l1_2": {
    id: "l1_2",
    tag: "Core",
    relevance: 10,
    title: "1.2 Generativní modely: Sequence RNN, Sequence & Graph Transformer",
    summary: "Architektury generátorů v DrugEx: Autoregresní GRU/LSTM sytém, causally masked Transformer a grafový maticový generátor.",
    slides: [
      {
        title: "1. Architektura Sequence RNN (GRU / LSTM)",
        content: `<code>SequenceRNN</code> představuje základní a vysoce stabilní generátor DrugEx. Skládá se z embeddingové vrstvy, několika vrstev GRU/LSTM buněk a lineární projekční hlavy do dimenze slovníku:`,
        code: `import torch
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

# Vytvoření SequenceRNN s buňkami LSTM
agent = SequenceRNN(voc, is_lstm=True)
agent.loadStatesFromFile("Papyrus05.5_smiles_rnn_PT.pkg")
agent.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Autoregresní vzorkování 5 nových molekul
samples = agent.sample(n_samples=5)
print("Vygenerované SMILES:", samples)`
      },
      {
        title: "2. Sequence Transformer & Causal Masking",
        content: `SMILES Transformer využívá self-attention mechanismus s kauzální maskou, která brání modelu v náhledu na budoucí tokeny při generování:
        <div class="math-card">
          $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V$$
        </div>
        kde $M$ je dolní trojúhelníková maska ($M_{ij} = -\\infty$ pro $j > i$).`
      },
      {
        title: "3. Graph Transformer & Konstrukce molekulárního grafu",
        content: `<code>GraphTransformer</code> nepracuje s textem, ale generuje současně matici atomových typů a matici vazeb. Tím garantuje 100% valenční validitu chemických struktur bez chyb typu 'neuzavřený kruh'.`
      }
    ]
  },

  "l1_3": {
    id: "l1_3",
    tag: "Core",
    relevance: 9,
    title: "1.3 Transfer Learning: Pre-training na Papyrus & Target Fine-Tuning",
    summary: "Dvoustupňový tréninkový proces: zkoumání obecného chemického prostoru vs adaptace na cílové bioaktivní ligandy.",
    slides: [
      {
        title: "1. Princip dvoustupňového transferového učení",
        content: `Trénování de novo generátoru od nuly vyžaduje obrovské množství dat. DrugEx uplatňuje transferové učení:
        <ol>
          <li><strong>Pre-training (Obecný chemický prostor)</strong>: Trénink na databázi Papyrus (~1.5M molekul) pro zvládnutí chemické syntaktiky a základních farmakoforů.</li>
          <li><strong>Fine-tuning (Cílový bioaktivní prostor)</strong>: Doučení na 50–500 známých aktivních ligandech specifického cíle (např. CCR2).</li>
        </ol>`,
        alert: {
          type: "tip",
          title: "Využití pro bakalářskou práci",
          text: "Fine-tuned model se v DrugEx stává tzv. <em>Mutate Network</em> (prior politikou) pro následné zpětnovazební učení (RL)."
        }
      },
      {
        title: "2. Příprava fine-tuning skriptu (prepare_models.py)",
        content: `Spuštění jemného doladění modelu na ligandech CCR2:`,
        code: `python -m drugex.dataset -b tutorial/CLI/examples -i CCR_HUMAN_AL.tsv -mc SMILES -o ccr2_ft -nof
python -m drugex.train -tm FT -b tutorial/CLI/examples -i ccr2_ft -ag models/pretrained/Papyrus05.5_smiles_rnn_PT.pkg -e 100 -bs 64 -gpu 0`
      }
    ]
  },

  // ==========================================
  // MODULE 2: MULTI-OBJECTIVE RL & PARETO
  // ==========================================
  "l2_1": {
    id: "l2_1",
    tag: "Core",
    relevance: 10,
    title: "2.1 Formulace MORL: Agent vs Mutate (Prior) & Policy Gradient",
    summary: "Proč jednokriteriální RL vede k degradaci molekul, dual-network setup a epsilon-greedy směrování akcí.",
    slides: [
      {
        title: "1. Selhání jednokriteriálního RL v chemii",
        content: `Pokud optimalizujeme pouze jedno kritérium (např. QSAR predikci afinity), generátor brzy nalezne 'patologické zkratky':
        <ul>
          <li>Monstrózní molekulové hmotnosti (MW > 800 Da).</li>
          <li>Nekonečné hydrofobní alifatické řetězce (LogP > 8).</li>
          <li>Zcela nesyntetizovatelné cyklické spiro-systémy.</li>
        </ul>
        Řešením je <strong>Multi-Objective Reinforcement Learning (MORL)</strong>.`
      },
      {
        title: "2. Duální politika: Agent vs Mutate Network",
        content: `Aby nedošlo ke kolapsu modelu do jediné struktury, DrugEx kombinuje dvě sítě:
        <div class="math-card">
          $$P(a_t | s_t) = (1 - \\epsilon) \\pi_\\theta(a_t | s_t) + \\epsilon \\pi_0(a_t | s_t)$$
        </div>
        kde $\\pi_\\theta$ je trénovaný Agent a $\\pi_0$ je fixní Mutate network (fine-tuned model).`
      }
    ]
  },

  "l2_2": {
    id: "l2_2",
    tag: "WOW",
    relevance: 10,
    title: "2.2 Paretova optimalita & Pareto Crowding Distance",
    summary: "Nedominované třídění řešení v multidimenzionálním prostoru cílů a algoritmus výpočtu Crowding Distance pro zachování diverzity.",
    slides: [
      {
        title: "1. Paretovo nedominované třídění (Non-dominated Sorting)",
        content: `Molekula $A$ dominuje molekulu $B$, pokud je ve všech cílech alespoň stejně dobrá a v alespoň jednom striktně lepší. Molekuly, které nejsou nikým dominovány, tvoří <strong>První Paretovu frontu (Rank 1)</strong>.`
      },
      {
        title: "2. Algoritmus Pareto Crowding Distance",
        content: `Pro zabránění shlukování na jednom místě fronty se počítá Crowding Distance $d_i$:
        <div class="math-card">
          $$d_i = \\sum_{m=1}^{M} \\frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$$
        </div>
        Molekuly na okrajích a v málo zaplněných regionech dostávají vyšší odměnu.`,
        code: `from drugex.training.rewards import ParetoCrowdingDistance

# Vytvoření Paretova schématu odměn
reward_scheme = ParetoCrowdingDistance()`
      }
    ]
  },

  "l2_3": {
    id: "l2_3",
    tag: "Core",
    relevance: 9,
    title: "2.3 Skórovací funkce & Desirability Modifiers (SmoothClippedScore)",
    summary: "Transformace fyzikálních parametrů na normalizované odměny [0, 1], syntetická dostupnost (SAScore, RAScore) a QSAR prediktory.",
    slides: [
      {
        title: "1. Normalizace skóre: Desirability Modifiers",
        content: `Fyzikální a biologické vlastnosti mají různé škály (SAScore 1-10, ROCS 0-2, LogP -2 až 6). Modifikátory převádějí surové hodnoty na hladké odměny $r \\in [0, 1]$:`
      },
      {
        title: "2. SmoothClippedScore v praxi",
        content: `Klesající sigmoidální křivka pro penalizaci vysokého SAScore (obtížná syntéza):`,
        code: `from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property

# SAScore: hodnoty pod 3.0 jsou ideální (r=1.0), nad 5.0 penalizovány (r=0.0)
sa_scorer = Property('SA')
sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))`
      }
    ]
  },

  // ==========================================
  // MODULE 3: 3D SHAPE MATCHING & IDP
  // ==========================================
  "l3_1": {
    id: "l3_1",
    tag: "Core",
    relevance: 10,
    title: "3.1 Fyzikální podstata ROCS: Shape Tanimoto & Color TanimotoCombo",
    summary: "Gaussovská reprezentace atomů, objemové překryvové integrály a farmakoforové barevné shody (ImplicitMillsDean).",
    slides: [
      {
        title: "1. Gaussovský objemový model atomů",
        content: `V metodě ROCS je hustota atomu reprezentována sférickou Gaussovskou funkcí:
        <div class="math-card">
          $$\\rho_i(\\mathbf{r}) = p_i \\exp\\left(-\\alpha_i |\\mathbf{r} - \\mathbf{r}_i|^2\\right)$$
        </div>
        Objemový překryv $V(A, B)$ mezi molekulami $A$ a $B$ je analyticky integrovatelný.`
      },
      {
        title: "2. Tvarová a barevná podobnost (TanimotoCombo)",
        content: `Celková podobnost kombinuje prostorový tvar a chemické barvy (farmakofory):
        <div class="math-card">
          $$T_{combo} = T_{shape} + T_{color} = \\frac{V_{shape}(A, B)}{V_A + V_B - V_{shape}(A,B)} + \\frac{V_{color}(A, B)}{C_A + C_B - V_{color}(A,B)}$$
        </div>
        Hodnota $T_{combo} \\in [0, 2]$.`
      }
    ]
  },

  "l3_2": {
    id: "l3_2",
    tag: "Legendary",
    relevance: 10,
    title: "3.2 Výzva flexibilních cílů & Intrinsically Disordered Proteins (IDP)",
    summary: "Když krystalové struktury chybí: Ligand-based 3D strategie, ansámbly vazebných konformací a konsensuální supermolekuly.",
    slides: [
      {
        title: "1. Proč tradiční dokování u IDP selhává",
        content: `Intrinsicky neuspořádané proteiny (IDP) a flexibilní membránové receptory nemají rigidní krystalovou kapsu. Dokování do statické struktury produkuje náhodné artefakty.<br><br>
        <strong>Řešení:</strong> Použití 3D ligand-based tvarového porovnávání. Pokud známe ligandy vážící se na IDP, jejich tvar definuje požadovanou 3D obálku pro de novo generátor.`
      }
    ]
  },

  "l3_3": {
    id: "l3_3",
    tag: "Tricky",
    relevance: 10,
    title: "3.3 Hloubková analýza conformer_generators.py (RDKit, OMEGA, CDPKit)",
    summary: "Architektura generátorů konformací, stereoisomerní enumerace, rotační vazby, heavy atom limity a vláknová bezpečnost.",
    slides: [
      {
        title: "1. Porovnání 4 konformačních generátorů v DrugEx",
        content: `Modul <code>drugex.training.scorers.conformer_generators</code> obsahuje:
        <ul>
          <li><strong>RDKitConformerGenerator</strong>: Využívá ETKDGv3, thread-safe, rychlé.</li>
          <li><strong>OmegaConformerGenerator</strong>: Komerční OpenEye OMEGA s GPU akcelerací.</li>
          <li><strong>CDPKitConformerGenerator</strong>: Čistě open-source ConfGen s energetickým oknem.</li>
          <li><strong>SchrodingerConformerGenerator</strong>: ConfGenX s opravou z-offsetu.</li>
        </ul>`
      },
      {
        title: "2. Inicializace RDKitConformerGenerator",
        code: `from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

conf_gen = RDKitConformerGenerator(
    max_conformers=50,      # Max počet konformací na izomer
    max_isomers=4,          # Max počet stereoizomerů
    max_heavy_atoms=45,     # Filtrace příliš velkých molekul
    max_rotatable_bonds=15, # Filtrace extrémně flexibilních molekul
    num_threads=1           # 1 thread na worker při paralelismu
)`
      }
    ]
  },

  // ==========================================
  // MODULE 4: ROCS SCORER BACKENDS
  // ==========================================
  "l4_1": {
    id: "l4_1",
    tag: "Core",
    relevance: 10,
    title: "4.1 RDKit ROCS Scorer: rdShapeAlign, Worker Init & Pose Invariance",
    summary: "Implementace rocs_rdkit.py: paralelní multiprocessing bez memory leaků, deduplikace SMILES a skupinové skórování.",
    slides: [
      {
        title: "1. Architektura RDKitROCSScorer",
        content: `<code>RDKitROCSScorer</code> provádí zarovnání generovaných konformerů vůči referenčním molekulám pomocí <code>rdShapeAlign.AlignMol</code>:`,
        code: `from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

scorer = RDKitROCSScorer(
    conformer_generator=conf_gen,
    references="CCR2_reference_ligands.sdf",
    score_type="TanimotoCombo",
    use_colors=True,
    n_jobs=-1  # Využít všechna dostupná CPU jádra
)`
      }
    ]
  },

  "l4_2": {
    id: "l4_2",
    tag: "WOW",
    relevance: 10,
    title: "4.2 CDPKit ROCS Scorer: CDPL.Shape, WorkerContext & Open-Source Pipeline",
    summary: "Implementace rocs_cdpkit.py: čistě open-source Gaussovské zarovnání přes PrincipalAxesAlignment a bezlicenční workflow.",
    slides: [
      {
        title: "1. Bezlicenční 3D tvarové skórování s CDPKit",
        content: `<code>CDPKitROCSScorer</code> využívá C++ knihovnu CDPL pro Gaussovské zarovnání hlavních os (Principal Axes Alignment) bez nutnosti OpenEye licence:`,
        code: `from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer

cdp_scorer = CDPKitROCSScorer(
    conformer_generator=conf_gen,
    references="CCR2_reference_ligands.sdf",
    n_jobs=-1
)`
      }
    ]
  },

  "l4_3": {
    id: "l4_3",
    tag: "Core",
    relevance: 9,
    title: "4.3 OpenEye ROCS Scorer: CLI Subprocess, .sq Queries & GPU Akcelerace",
    summary: "Implementace rocs_openeye.py: řízení binárky rocs, načítání VROCS Shape Queries (.sq) a licence OpenEye.",
    slides: [
      {
        title: "1. Průmyslový standard OpenEye ROCS",
        content: `<code>OpenEyeROCSScorer</code> spouští optimalizovanou binárku <code>rocs</code> a umožňuje použití předpočítaných dotazů <code>.sq</code>:`,
        code: `from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer

oe_scorer = OpenEyeROCSScorer(
    conformer_generator=omega_gen,
    references={"binding_site_1": "model1.sq"},
    score_type="TanimotoCombo"
)`
      }
    ]
  },

  // ==========================================
  // MODULE 5: EXPERIMENTAL PIPELINE & BENCHMARK
  // ==========================================
  "l5_1": {
    id: "l5_1",
    tag: "Core",
    relevance: 10,
    title: "5.1 Stanovení ROCS prahu & ROC analýza (threshold_analysis.py)",
    summary: "Metodika oddělení aktivních látek od decoyů (CCR2 aktivní vs DUD-E), Youdenův index J = TPR - FPR a self-similarity.",
    slides: [
      {
        title: "1. Vědecká metodika stanovení prahu",
        content: `Před spuštěním RL je nutné stanovit dělící práh pro odměnu:
        <ol>
          <li>Oskórování 75 aktivních ligandů CCR2 a 500 decoyů.</li>
          <li>Sestrojení ROC křivky (True Positive Rate vs False Positive Rate).</li>
          <li>Nalezení maxima Youdenova indexu $J = \\text{TPR} - \\text{FPR}$.</li>
        </ol>`,
        code: `from tutorial.advanced.rocs.threshold_analysis import run_threshold_analysis

results = run_threshold_analysis(
    actives_csv="actives_ccr2_N75.csv",
    decoys_csv="decoys_ccr2_N500.csv",
    references_sdf="CCR2_reference_ligands.sdf"
)
print("Optimální práh:", results['optimal_threshold']) # ~0.871`
      }
    ]
  },

  "l5_2": {
    id: "l5_2",
    tag: "Core",
    relevance: 10,
    title: "5.2 End-to-End RL tréninkový cyklus (config.py, rocs_rl_tutorial.ipynb)",
    summary: "Příprava modelů, spuštění RL explorátoru, monitorování poměru validních žádaných molekul (desired_ratio) a konvergence.",
    slides: [
      {
        title: "1. Spuštění SequenceExplorer tréninku",
        code: `from drugex.training.explorers import SequenceExplorer
from tutorial.advanced.rocs.config import setup_rl_rdkit

agent, mutate, env, out_dir = setup_rl_rdkit()

explorer = SequenceExplorer(
    agent=agent,
    mutate=mutate,
    crover=None,
    env=env,
    epsilon=0.2,
    n_samples=1000
)

# Trénování po dobu 50 epoch
explorer.fit(epochs=50, output_dir=str(out_dir))`
      }
    ]
  },

  "l5_3": {
    id: "l5_3",
    tag: "Core",
    relevance: 10,
    title: "5.3 Generování kandidátů, filtrace novosti & chemické ověření",
    summary: "Vzorkování molekul (generate_molecules.py), výpočet interní diverzity, novelty vůči ChEMBL a příprava pro bio-testy.",
    slides: [
      {
        title: "1. Vzorkování a filtrace vygenerovaných molekul",
        code: `from drugex.generate import generate_molecules

# Vygenerování 10 000 nových struktur z natrénovaného reinforced modelu
df_mols = generate_molecules(
    model_pkg="CCR2_rdkit_reinforced.pkg",
    num_samples=10000,
    keep_undesired=False
)
print(f"Vygenerováno {len(df_mols)} žádoucích molekul.")`
      }
    ]
  },

  // ==========================================
  // MODULE 6: SCAFFOLDS, CLI & HPC
  // ==========================================
  "l6_1": {
    id: "l6_1",
    tag: "Core",
    relevance: 9,
    title: "6.1 Fragmentový návrh, BRICS štěpení & FragSequenceExplorer",
    summary: "Cílený růst z fragmentů, design linkerů pro vícedoménové proteiny a fragmentační konvertory.",
    slides: [
      {
        title: "1. BRICS fragmentace v DrugEx",
        content: `BRICS štěpí molekuly na 16 typech synteticky proveditelných vazeb (amidy, estery, ethery). Modely <code>FragSequenceExplorer</code> umožňují připojovat k pevnému scaffoldům nové substituenty.`
      }
    ]
  },

  "l6_2": {
    id: "l6_2",
    tag: "Core",
    relevance: 10,
    title: "6.2 Příkazová řádka DrugEx: drugex dataset, train & generate",
    summary: "Kompletní matice parametrů CLI pro automatizované skriptování a batch zpracování velkých sad dat.",
    slides: [
      {
        title: "1. CLI Matrix příkazů",
        code: `# 1. Preprocessing dat
python -m drugex.dataset -b data_dir -i ligands.tsv -mc SMILES -o target_ds -mt smiles

# 2. RL Trénink
python -m drugex.train -tm RL -b data_dir -i target_ds -ag agent.pkg -pr mutate.pkg -sas -e 50 -gpu 0

# 3. Generování sloučenin
python -m drugex.generate -b data_dir -g reinforced_model.pkg -n 5000 -gpu 0`
      }
    ]
  },

  "l6_3": {
    id: "l6_3",
    tag: "WOW",
    relevance: 9,
    title: "6.3 Škálování na GPU klastru: Slurm skripty & správa paměti",
    summary: "Šablony Slurm úloh s alokací GPU, proměnnými prostředí (OE_LICENSE) a ochranou proti přetečení RAM/VRAM.",
    slides: [
      {
        title: "1. Produkční Slurm dávkový skript (drugex_job.sh)",
        code: `#!/bin/bash
#SBATCH --job-name=drugex_rocs_rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=drugex_%j.log

module load cuda/12.1 python/3.11
source ~/.venv/bin/activate
export OE_LICENSE=/storage/licenses/oe_license.txt
export OMP_NUM_THREADS=1

python tutorial/advanced/rocs/rocs_rl_tutorial.ipynb`
      }
    ]
  }
};
