/**
 * DrugEx Hub — Master Deep Lecture Database (18 Exhaustive Lectures)
 * Tailored for Bachelor's Thesis on De Novo Drug Design & ROCS Shape Matching for Flexible Targets / IDPs
 */

export const LECTURE_DATA = {
  // =========================================================================
  // MODULE 1: DE NOVO GENERATION & MOLECULAR REPRESENTATIONS
  // =========================================================================

  "l1_1": {
    id: "l1_1",
    tag: "Core",
    relevance: 10,
    title: "1.1 Molekulární reprezentace, tokenizace & slovník (VocSmiles)",
    summary: "1D SMILES vs 2D grafy vs fragmentové vaky, tokenizační smlouva VocSmiles a SmilesStandardizer.",
    slides: [
      {
        title: "1. Paradigma molekulárních reprezentací pro generativní modely",
        content: `Při algoritmickém návrhu léčiv (De Novo Drug Design) je volba reprezentace molekuly klíčovým faktorem určujícím jak vyjadřovací schopnost modelu, tak složitost optimalizace.
        <br><br>
        V chemoinformatice a hlubokém učení pracujeme se třemi hlavními reprezentacemi:
        <ul>
          <li><strong>1D Textové sekvence (SMILES, SELFIES, DeepSMILES)</strong>: Linearizovaný zápis molekulárního grafu pomocí regulární gramatiky. Umožňuje přímé nasazení jazykových modelů (RNN, GPT Transformer), ale nese riziko syntakticky nevalidních sekvencí.</li>
          <li><strong>2D Molekulární grafy (Attributed Molecular Graphs)</strong>: Přirozená reprezentace, kde uzly představují atomy (s příznaky typu atomu, hybridizace, formálního náboje) a hrany reprezentují kovalentní vazby. Eliminuje syntaktické chyby linearizace.</li>
          <li><strong>Fragmentové vaky & Synthony (BRICS, Murcko)</strong>: Reprezentace molekuly jako kompozice větších synteticky proveditelných fragmentů, což garantuje vysokou syntetickou dostupnost (synthesizability).</li>
        </ul>`,
        compare: {
          leftTitle: "1D SMILES (Linearizace grafu)",
          leftContent: "<strong>Výhody:</strong> Extrémně nízká paměťová náročnost, rychlé dávkování (batching), možnost využití standardních NLP architektur (LSTM, GPT).<br><strong>Nevýhody:</strong> Gramatická křehkost (neuzavřené kruhy např. <code>c1ccccc</code> bez koncové <code>1</code>, nespárované závorky).",
          rightTitle: "2D Grafy (Matice sousednosti)",
          rightContent: "<strong>Výhody:</strong> 100% chemická syntaktická validita, invariantnost vůči permutaci atomů, přirozené fixování scaffoldů.<br><strong>Nevýhody:</strong> Vyšší výpočetní a paměťová náročnost ($O(N^2)$ pro matice vazeb), složitější autoregresní vzorkování."
        }
      },
      {
        title: "2. Tokenizační kontrakt DrugEx: VocSmiles & Regulární výrazy",
        content: `Standardní znaková tokenizace (po jednotlivých písmenech) je v chemoinformatice nevhodná, protože nerozezná dvoupísmenné atomy (např. <code>Cl</code> vs <code>C</code> + <code>l</code>) ani specifické chemické modifikátory (např. <code>[nH]</code>, <code>[O-]</code>, <code>@@</code>).
        <br><br>
        Třída <code>VocSmiles</code> (v <code>drugex/data/corpus/vocabulary.py</code>) využívá specializovaný chemoinformatický regulární výraz:
        <div class="math-card">
          $$\\text{Regex} = \\texttt{\\[[^\\]]+\\]|Br|Cl|Si|Na|Ca|Fe|@@|@|\\/|\\\\|\\%\\d{2}|\\d|=|#|\\$|:|~|\\.|[a-zA-Z]}$$
        </div>
        Tento regulární výraz přesně rozkládá SMILES na chemicky smysluplné atomy, řády vazeb, čísla cyklů a stereochemické deskriptory.`,
        code: `from drugex.data.corpus.vocabulary import VocSmiles

# Načtení slovníku z předtrénovaného korpusu Papyrus
voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)
print(f"Celková velikost slovníku: {voc.size} unikátních tokenů")

# Ukázka tokenizace molekuly s chirálním centrem a heterocyklem
smiles = "Cc1ccc(NC(=O)[C@@H](N)C)cc1"
encoded = voc.encode([smiles]) # Vrací PyTorch Tensor indexů
print("Kódovaný tenzor:", encoded)

# Zpětné dekódování na kanonický SMILES
decoded = voc.decode(encoded)
print("Dekódovaný SMILES:", decoded[0])`,
        alert: {
          type: "important",
          title: "Speciální řídicí tokeny ve VocSmiles",
          text: "VocSmiles rezervuje pevné indexy pro řídicí tokeny: <code>&lt;START&gt;</code> (index 1, inicializuje generování), <code>&lt;END&gt;</code> (index 2, ukončuje řetězec) a <code>&lt;PAD&gt;</code> (index 0, zarovnává sekvence v batchi na stejnou délku)."
        }
      },
      {
        title: "3. Sanitační pipeline: SmilesStandardizer",
        content: `Před trénováním generátoru nebo skórováním je nezbytné data standardizovat. Databáze jako Papyrus nebo ChEMBL obsahují nekonzistence (směsi solí, různé tautomerní formy, izotopy).
        <br><br>
        <code>SmilesStandardizer</code> (v <code>drugex/molecules/converters/standardizers.py</code>) provádí:
        <ol>
          <li><strong>Odsolení (Desalting)</strong>: Odstranění protiiontů a rozpouštědel (např. <code>[Na+]</code>, <code>[Cl-]</code>, voda) a zachování pouze největšího kovalentního fragmentu.</li>
          <li><strong>Neutralizaci nábojů</strong>: Kde je to chemicky vhodné (např. protonace karboxylátů <code>[O-]</code> na <code>OH</code>).</li>
          <li><strong>Tautomerní kanonizaci</strong>: Převedení na nejstabilnější tautomerní formu.</li>
          <li><strong>Odstranění izotopů</strong>: Nahrazení izotopů ($^{13}C, ^{2}H$) standardními prvky.</li>
        </ol>`,
        code: `from drugex.molecules.converters.standardizers import SmilesStandardizer

standardizer = SmilesStandardizer(
    drop_unspecified=True,  # Odstraní molekuly s nejasnou stereochemií
    drop_isotopes=True      # Nahradí izotopy standardními atomy
)

# Příklad surového vstupu se solí
raw_mol = "CC(=O)[O-].[Na+].O"
clean_smi = standardizer(raw_mol)
print(f"Vyčištěný SMILES: {clean_smi}") # Výstup: CC(=O)O`
      }
    ]
  },

  "l1_2": {
    id: "l1_2",
    tag: "Core",
    relevance: 10,
    title: "1.2 Generativní modely: Sequence RNN, Sequence & Graph Transformer",
    summary: "Architektury generátorů v DrugEx: Autoregresní GRU/LSTM systém, causally masked Transformer a grafový maticový generátor.",
    slides: [
      {
        title: "1. Sequence RNN (Recurrent Neural Network)",
        content: `<code>SequenceRNN</code> (v <code>drugex/training/generators/sequence_rnn.py</code>) je základním autoregresním modelem v DrugEx. Využívá buňky GRU nebo LSTM s více skrytými vrstvami:
        <br><br>
        Model počítá podmíněnou pravděpodobnost generování dalšího tokenu $x_t$ na základě všech předchozích tokenů $x_{<t}$ a skrytého stavu $h_{t-1}$:
        <div class="math-card">
          $$P(X) = \\prod_{t=1}^{T} P(x_t | x_1, x_2, \\dots, x_{t-1}) = \\prod_{t=1}^{T} \\text{softmax}\\left(W_{out} h_t + b\\right)$$
        </div>`,
        code: `import torch
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)

# Inicializace SequenceRNN s 3 vrstvami LSTM a embeddingem dimenze 128
agent = SequenceRNN(voc, is_lstm=True)
agent.loadStatesFromFile("Papyrus05.5_smiles_rnn_PT.pkg")
agent.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Vzorkování 5 molekul s teplotou T = 1.0
samples = agent.sample(n_samples=5)
print("Vygenerované molekuly:", samples)`
      },
      {
        title: "2. Sequence Transformer & Causal Self-Attention",
        content: `<code>SequenceTransformer</code> nahrazuje rekurentní buňky vícehlavou samo-pozorností (Multi-Head Self-Attention). Aby model při generování neviděl budoucí tokeny, aplikuje se kauzální maska $M$:
        <div class="math-card">
          $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V, \\quad M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}$$
        </div>
        Výhodou je masivní paralelizace při tréninku a schopnost zachytit dlouhé chemické závislosti (např. uzavírání velkých makrocyklů).`
      },
      {
        title: "3. Graph Transformer: Přímé generování molekulárních grafů",
        content: `<code>GraphTransformer</code> (v <code>drugex/training/generators/graph_transformer.py</code>) operuje přímo s maticemi uzlů $V \\in \\mathbb{R}^{N \\times d_n}$ (typy atomů, formální náboje) a hran $E \\in \\mathbb{R}^{N \\times N \\times d_e}$ (řády vazeb).
        <br><br>
        Tím zcela eliminuje syntaktické chyby linearizace a umožňuje přesné scaffold-constrained generování, kde je fixní část grafu (např. vazebné jádro) pevně ukotvena a model generuje pouze variabilní sousedství.`,
        compare: {
          leftTitle: "Sequence RNN (Text)",
          leftContent: "Rychlý trénink, stabilní konvergence při RL, ideální výchozí bod pro ligand-based shape matching.",
          rightTitle: "Graph Transformer (Topologie)",
          rightContent: "Komplexnější architektura, 100% syntaktická validita, přímá kontrola nad atomy a kruhy."
        }
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
        title: "1. Teoretický základ dvoustupňového transferového učení",
        content: `Trénování generátoru přímo na malé sadě ligandů (desítky až stovky struktur) vede k rychlému přeučení a nízké chemické diverzitě. DrugEx proto využívá dvoustupňový přenos znalostí:
        <ol>
          <li><strong>Pre-training (Obecný chemický prostor)</strong>: Model se učí na rozsáhlé databázi Papyrus v05.5 (~1,5 milionu bioaktivních molekul). Zde si osvojí pravidla valence, stability kruhů a chemickou gramatiku.</li>
          <li><strong>Fine-Tuning (Cílově zaměřený prostor)</strong>: Váhy předtrénovaného generátoru jsou doučeny na sadě známých ligandů cílového proteinu (např. CCR2 nebo studovaný IDP cíl) po dobu 50–100 epoch s nízkým learning rate ($10^{-4}$).</li>
        </ol>`,
        alert: {
          type: "tip",
          title: "Význam pro Reinforcement Learning",
          text: "Fine-tuned model se v DrugEx stává tzv. <em>Mutate Network</em> (fixní referenční sítí), která v RL explorátoru udržuje generátor v chemickém prostoru relevantním pro daný cíl."
        }
      },
      {
        title: "2. Praktický workflow přípravy modelů v CLI",
        content: `Příprava datasetu a spuštění fine-tuningu pomocí oficiálních DrugEx skriptů:`,
        code: `# 1. Krok: Preprocessing cílových ligandů CCR2
python -m drugex.dataset \\
    -b tutorial/CLI/examples \\
    -i CCR_HUMAN_AL.tsv \\
    -mc SMILES \\
    -o ccr2_finetune \\
    -mt smiles \\
    -nof

# 2. Krok: Spuštění fine-tuningu na předtrénovaném modelu Papyrus
python -m drugex.train \\
    -tm FT \\
    -b tutorial/CLI/examples \\
    -i ccr2_finetune \\
    -ag models/pretrained/Papyrus05.5_smiles_rnn_PT.pkg \\
    -e 100 \\
    -bs 64 \\
    -lr 0.0001 \\
    -gpu 0`
      }
    ]
  },

  // =========================================================================
  // MODULE 2: MULTI-OBJECTIVE RL & PARETO OPTIMALITY
  // =========================================================================

  "l2_1": {
    id: "l2_1",
    tag: "Core",
    relevance: 10,
    title: "2.1 Formulace MORL: Agent vs Mutate (Prior) & Policy Gradient",
    summary: "Proč jednokriteriální RL vede k degradaci molekul, dual-network setup a epsilon-greedy směrování akcí.",
    slides: [
      {
        title: "1. Matematická formulace Policy Gradient (REINFORCE)",
        content: `Ve zpětnovazebním učení pro generování sekvencí je generátor chápán jako parametrizovaná politika $\\pi_\\theta$, která generuje akci (token $a_t$) na základě dosavadního stavu sekvence $s_t$.
        <br><br>
        Ztrátová funkce s využitím celkové odměny $R(X)$ a baseline $b$:
        <div class="math-card">
          $$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{X \\sim \\pi_\\theta} \\left[ \\sum_{t=1}^{T} \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\cdot \\left( R(X) - b \\right) \\right]$$
        </div>
        Kde $R(X) \\in [0, 1]$ je vícekriteriální odměna z prostředí (např. kombinace ROCS tvarové shody a SAScore).`
      },
      {
        title: "2. Duální síť: Agent $\\pi_\\theta$ vs Mutate Network $\\pi_0$",
        content: `Při standardním RL hrozí tzv. <em>mode collapse</em> (kolaps politiky do jediné triviální molekuly s vysokým skóre). DrugEx tomu předchází duálním vzorkováním s parametrem $\\epsilon \\in [0, 1]$ (obvykle $\\epsilon = 0.2$):
        <div class="math-card">
          $$P(a_t | s_t) = (1 - \\epsilon) \\pi_\\theta(a_t | s_t) + \\epsilon \\pi_0(a_t | s_t)$$
        </div>
        <ul>
          <li><strong>Agent ($\\pi_\\theta$)</strong>: Učící se model, který je optimalizován gradientem odměny (exploatace).</li>
          <li><strong>Mutate / Prior ($\\pi_0$)</strong>: Fixní model (předtrénovaný fine-tuned generátor), který udržuje vysokou strukturní diverzitu a zabraňuje zapomenutí chemické gramatiky (explorace).</li>
        </ul>`
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
        title: "1. Princip Paretovy dominance v chemickém prostoru",
        content: `V reálném objevování léčiv musíme současně optimalizovat $M$ komplementárních nebo protichůdných cílů (např. 3D ROCS tvarová shoda, bioaktivita na cíli, rozpustnost, syntetická dostupnost).
        <br><br>
        Vektor řešení $\\mathbf{f}(A) = (f_1(A), f_2(A), \\dots, f_M(A))$ <strong>dominuje</strong> $\\mathbf{f}(B)$, pokud:
        <div class="math-card">
          $$\\forall i \\in \\{1, \\dots, M\\}: f_i(A) \\ge f_i(B) \\quad \\land \\quad \\exists j \\in \\{1, \\dots, M\\}: f_j(A) > f_j(B)$$
        </div>
        Molekuly, které nejsou dominovány žádnou jinou molekulou v populaci, tvoří <strong>První Paretovu frontu (Rank 1)</strong>.`
      },
      {
        title: "2. Výpočet Pareto Crowding Distance",
        content: `Aby model nepreferoval pouze jeden bod na Paretově frontě, DrugEx (Liu et al., 2021) implementuje výpočet <em>Crowding Distance</em> podél fronty:
        <div class="math-card">
          $$d_i = \\sum_{m=1}^{M} \\frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$$
        </div>
        Krajní body fronty dostávají nekonečnou vzdálenost ($d = \\infty$). Molekuly v méně hustých regionech Paretovy fronty tak získávají vyšší odměnu, což nutí RL generátor prozkoumávat celou šíři kompromisu mezi vlastnostmi.`,
        code: `from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.environment import DrugExEnvironment

# Inicializace Paretova schématu odměn
reward_scheme = ParetoCrowdingDistance()

# Vytvoření prostředí DrugEx
env = DrugExEnvironment(
    scorers=[rocs_scorer, sa_scorer],
    thresholds=[0.871, 0.1],
    reward_scheme=reward_scheme
)`
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
        title: "1. Modifikátory žádoucnosti (Desirability Modifiers)",
        content: `Surové hodnoty vlastností mají velmi odlišné jednotky a rozsahy:
        <ul>
          <li>ROCS TanimotoCombo: interval $[0, 2]$.</li>
          <li>SAScore (Syntetická dostupnost): interval $[1, 10]$ (nižší je lepší).</li>
          <li>Molekulová hmotnost (MW): stovky Daltonů (ideál 300–500 Da).</li>
        </ul>
        Modifikátory (v <code>drugex/training/scorers/modifiers.py</code>) transformují libovolnou surovou metriku na normalizovanou odměnu $r \\in [0, 1]$.`
      },
      {
        title: "2. Matematika SmoothClippedScore",
        content: `<code>SmoothClippedScore</code> vytváří hladkou kosínovou/sigmoidální křivku se spojitými derivacemi:
        <div class="math-card">
          $$S(x) = \\begin{cases} 
          1.0 & x \\le \\text{upper\\_x} \\\\ 
          0.5 \\left( 1 + \\cos\\left( \\pi \\frac{x - \\text{upper\\_x}}{\\text{lower\\_x} - \\text{upper\\_x}} \\right) \\right) & \\text{upper\\_x} < x < \\text{lower\\_x} \\\\ 
          0.0 & x \\ge \\text{lower\\_x} 
          \\end{cases}$$
        </div>
        Pokud je $\\text{lower\\_x} > \\text{upper\\_x}$, jde o penalizační funkci (např. pro SAScore, kde hodnoty $\\le 3.0$ mají odměnu 1.0 a hodnoty $\\ge 5.0$ mají odměnu 0.0).`,
        code: `from drugex.training.scorers.modifiers import SmoothClippedScore, ClippedScore
from drugex.training.scorers.properties import Property

# SAScore s hladkou penalizací
sa_scorer = Property('SA')
sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))

# Molekulová hmotnost (MW) s horním limitem 500 Da
mw_scorer = Property('MW')
mw_scorer.setModifier(ClippedScore(lower_x=300, upper_x=500))`
      }
    ]
  },

  // =========================================================================
  // MODULE 3: 3D SHAPE MATCHING, IDP & CONFORMER ENGINES
  // =========================================================================

  "l3_1": {
    id: "l3_1",
    tag: "Core",
    relevance: 10,
    title: "3.1 Fyzikální podstata ROCS: Shape Tanimoto & Color TanimotoCombo",
    summary: "Gaussovská reprezentace atomů, objemové překryvové integrály a farmakoforové barevné shody (ImplicitMillsDean).",
    slides: [
      {
        title: "1. Gaussovská prostorová reprezentace molekulárního objemu",
        content: `Tradiční reprezentace molekuly jako pevných koulí (hard spheres s Van der Waalsovými poloměry) vede k nespojitým skokům při prostorových překryvech. Metoda ROCS (Grant, Pickup, Nicholls) nahrazuje každý atom $i$ sférickou Gaussovskou funkcí:
        <div class="math-card">
          $$\\rho_i(\\mathbf{r}) = p_i \\exp\\left(-\\alpha_i |\\mathbf{r} - \\mathbf{r}_i|^2\\right)$$
        </div>
        kde $p_i$ je centrální hustota a $\\alpha_i$ určuje poloměr atomu.
        <br><br>
        Celkový objem molekuly $A$ je součtem Gaussovských hustot všech jejích atomů: $\\rho_A(\\mathbf{r}) = \\sum_{i \\in A} \\rho_i(\\mathbf{r})$. Objemový integrál překryvu $V(A, B)$ je díky Gaussovskému tvaru analyticky integrovatelný v uzavřené formě:
        <div class="math-card">
          $$V(A, B) = \\int \\rho_A(\\mathbf{r}) \\rho_B(\\mathbf{r}) d\\mathbf{r} = \\sum_{i \\in A} \\sum_{j \\in B} p_i p_j \\left( \\frac{\\pi}{\\alpha_i + \\alpha_j} \\right)^{3/2} \\exp\\left( -\\frac{\\alpha_i \\alpha_j}{\\alpha_i + \\alpha_j} |\\mathbf{r}_i - \\mathbf{r}_j|^2 \\right)$$
        </div>`
      },
      {
        title: "2. Shape Tanimoto, Color Tanimoto & TanimotoCombo",
        content: `Na základě analytického objemu definujeme tři klíčové metriky podobnosti:
        <ol>
          <li><strong>Shape Tanimoto ($T_{shape} \\in [0, 1]$)</strong>: Čistě prostorová tvarová shoda objemů:
            <div class="math-card">
              $$T_{shape}(A, B) = \\frac{V(A, B)}{V(A, A) + V(B, B) - V(A, B)}$$
            </div>
          </li>
          <li><strong>Color Tanimoto ($T_{color} \\in [0, 1]$)</strong>: Shoda farmakoforových barevných center (donory a akceptory H-vazeb, anionty, kationty, aromatická a hydrofobní centra dle silového pole <code>ImplicitMillsDean</code>).</li>
          <li><strong>TanimotoCombo ($T_{combo} \\in [0, 2]$)</strong>: Celkové kompozitní skóre:
            <div class="math-card">
              $$T_{combo} = T_{shape} + T_{color}$$
            </div>
          </li>
        </ol>`
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
        title: "1. Limity strukturního dokování u IDP a flexibilních proteinů",
        content: `Strukturový návrh léčiv (SBDD) předpokládá rigidní vazebnou kapsu ("zámek a klíč"). U vysoce flexibilních proteinů, alosterických přechodů a zejména <strong>intrinsicky neuspořádaných proteinů (IDP)</strong> však rigidní kapsa neexistuje:
        <ul>
          <li>IDP existují jako dynamický konformační ansámbl bez definované sekundární či terciární struktury.</li>
          <li>Krystalografie ani Cryo-EM neposkytují atomární rozlišení vazebného místa.</li>
          <li>Klasické dokování do rigidní mřížky selhává a generuje falešně pozitivní artefakty.</li>
        </ul>`,
        alert: {
          type: "important",
          title: "Východisko pro bakalářskou práci",
          text: "Pokud máme experimentálně ověřené ligandy schopné vazby na flexibilní protein/IDP, jejich 3D konformační obálka reprezentuje negativní otisk vazebného rozhraní. De novo generování řízené 3D ROCS tvarovou shodou dokáže vytvořit nové, tvarově komplementární sloučeniny bez nutnosti znalosti struktury proteinu."
        }
      },
      {
        title: "2. Strategie Supermolekul a referenčních skupin (Reference Groups)",
        content: `Pro zachycení plasticity vazby se v DrugEx využívají dvě strategie:
        <ul>
          <li><strong>Supermolekula (Supermolecule, <code>supermol_123.sdf</code>)</strong>: Sloučení několika překrytých aktivních konformací do jedné konsensuální referenční šablony.</li>
          <li><strong>Multi-Reference Grouping</strong>: Definice skupiny referenčních ligandů v různých vazebných módech. Skórovač vrací maximální shodu:
            <div class="math-card">
              $$\\text{Score}_{group}(X) = \\max_{R \\in \\text{Group}} T_{combo}(X, R)$$
            </div>
          </li>
        </ul>`
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
        title: "1. Srovnávací přehled 4 konformačních generátorů v DrugEx",
        content: `Soubor <code>drugex/training/scorers/conformer_generators.py</code> implementuje abstraktní rozhraní <code>ConformerGenerator</code> se 4 konkrétními backendy:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">1. RDKitConformerGenerator</div>
            Využívá <code>AllChem.ETKDGv3()</code>. Open-source, rychlé generování, podpora thread-level paralelizace (<code>num_threads</code>), stereoisomerní enumerace přes <code>EnumerateStereoisomers</code>.
          </div>
          <div class="compare-col right">
            <div class="compare-heading">2. CDPKitConformerGenerator</div>
            Využívá <code>CDPL.ConfGen</code>. Čistě open-source C++ engine, generuje ensemble konformací s RMSD pruningem a nastavitelným energetickým oknem (<code>energy_window=20.0 kcal/mol</code>).
          </div>
        </div>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">3. OmegaConformerGenerator</div>
            Využívá OpenEye OMEGA (<code>oeomega</code>). Komerční zlatý standard s podporou GPU akcelerace a integrací <code>OEFilter</code>.
          </div>
          <div class="compare-col right">
            <div class="compare-heading">4. SchrodingerConformerGenerator</div>
            Spouští Schrodinger <code>confgenx</code> přes subprocess s automatickou korekcí z-offsetu (+0.01 Å) pro eliminaci chyb u planárních makrocyklů.
          </div>
        </div>`
      },
      {
        title: "2. Klíčové parametry a vláknová bezpečnost (Thread Safety)",
        content: `Při nasazení v RL smyčce je kritické nastavení parametrů generátoru konformací:`,
        code: `from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

conf_gen = RDKitConformerGenerator(
    max_conformers=50,       # Počet konformací na izomer (dostatečné pro vzorkování)
    max_isomers=4,           # Enumerace až 4 stereoizomerů pro nejasná chirální centra
    max_heavy_atoms=45,      # Odfiltrování monstrózních molekul s > 45 těžkými atomy
    max_rotatable_bonds=15,  # Odfiltrování molekul s > 15 rotovatelnými vazbami
    num_threads=1,           # DŮLEŽITÉ: 1 thread na proces při n_jobs > 1 (zabraňuje CPU oversubscription)
    show_progress=False
)`
      }
    ]
  },

  // =========================================================================
  // MODULE 4: ROCS SCORER BACKENDS (feature/rocs-scoring)
  // =========================================================================

  "l4_1": {
    id: "l4_1",
    tag: "Core",
    relevance: 10,
    title: "4.1 RDKit ROCS Scorer: rdShapeAlign, Worker Init & Pose Invariance",
    summary: "Implementace rocs_rdkit.py: paralelní multiprocessing bez memory leaků, deduplikace SMILES a skupinové skórování.",
    slides: [
      {
        title: "1. Vnitřní architektura RDKitROCSScorer (rocs_rdkit.py)",
        content: `<code>RDKitROCSScorer</code> provádí 3D tvarové porovnávání pomocí nativní C++ implementace v RDKit (<code>rdShapeAlign.AlignMol</code>):
        <br><br>
        Klíčové optimalizace v kódu:
        <ol>
          <li><strong>Deduplikace SMILES (<code>_deduplicate_smiles</code>)</strong>: Během RL generování generátor často vyprodukuje identické struktury. Deduplikace zajistí, že generování konformací a 3D zarovnání proběhne pro každou unikátní strukturu pouze jednou a výsledek se rozkopíruje.</li>
          <li><strong>Multiprocessing Worker Initializer (<code>_rdkit_worker_init</code>)</strong>: Referenční molekuly a nastavení jsou odeslány do procesů v <code>Pool</code> pouze jednou při startu, což eliminuje gigabajty serializačního overheadu při každé epoše.</li>
          <li><strong>Pose Invariance & Auto-Embedding (<code>_ensure_reference_conformers</code>)</strong>: Pokud referenční molekula nemá 3D souřadnice, je automaticky vložena přes ETKDGv3.</li>
        </ol>`,
        code: `from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

conf_gen = RDKitConformerGenerator(max_conformers=50, max_isomers=4, num_threads=1)

# Inicializace RDKit ROCS skórovače s multi-processingem
rocs_scorer = RDKitROCSScorer(
    conformer_generator=conf_gen,
    references="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
    score_type="TanimotoCombo", # 'TanimotoCombo' | 'shape' | 'color'
    use_colors=True,
    show_progress=True,
    n_jobs=-1                   # -1 využije všechna dostupná CPU jádra
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
        title: "1. Plně otevřené Gaussovské zarovnání s CDPKit (rocs_cdpkit.py)",
        content: `<code>CDPKitROCSScorer</code> využívá moderní chemoinformatickou knihovnu CDPKit (Chemical Data Processing Toolkit). Využívá třídy:
        <ul>
          <li><code>CDPL.Shape.GaussianShapeGenerator</code>: Generování Gaussovských tvarových reprezentací.</li>
          <li><code>CDPL.Shape.PrincipalAxesAlignmentStartGenerator</code>: Hledání výchozích orientací podél hlavních os setrvačnosti.</li>
          <li><code>CDPL.Shape.GaussianShapeAlignment</code>: Gradientní optimalizace překryvu s omezením na maximální počet iterací (<code>MAX_OPTIMIZATION_ITERATIONS = 20</code>).</li>
          <li><code>CDPL.Shape.calcTanimotoComboScore</code>: Analytický výpočet TanimotoCombo skóre.</li>
        </ul>`,
        code: `from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer

# Inicializace čistě open-source CDPKit ROCS workflow
cdp_conf_gen = CDPKitConformerGenerator(max_conformers=50, max_isomers=4)

cdp_scorer = CDPKitROCSScorer(
    conformer_generator=cdp_conf_gen,
    references="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
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
        title: "1. Integrace komerčního standardu OpenEye ROCS (rocs_openeye.py)",
        content: `<code>OpenEyeROCSScorer</code> kombinuje Python API pro manipulaci s molekulami s přímým voláním optimalizované binárky <code>rocs</code> přes <code>subprocess.run</code>:
        <br><br>
        Klíčové přepínače CLI příkazu generovaného v <code>_build_rocs_command</code>:
        <ul>
          <li><code>-query &lt;file.sq / file.sdf&gt;</code>: Referenční soubor nebo Shape Query.</li>
          <li><code>-dbase &lt;conformers.oeb.gz&gt;</code>: Databáze generovaných konformerů.</li>
          <li><code>-report one -stats best</code>: Vrátí pouze nejlepší překryv pro každou molekulu.</li>
          <li><code>-nostructs</code>: Neukládá 3D výstupní struktury (masivní zrychlení I/O operací).</li>
          <li><code>-chemff ImplicitMillsDean</code>: Farmakoforové barevné silové pole.</li>
        </ul>`,
        code: `import os
from drugex.training.scorers.conformer_generators import OmegaConformerGenerator
from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer

# Ověření platnosti OpenEye licence
os.environ["OE_LICENSE"] = "/path/to/oe_license.txt"

omega_gen = OmegaConformerGenerator(max_conformers=50, use_gpu=True)

oe_scorer = OpenEyeROCSScorer(
    conformer_generator=omega_gen,
    references={"ccr2_pocket": "rocs_rl_ccr/rdkit_cdpkit/model1.sq"},
    score_type="TanimotoCombo",
    color_force_field="ImplicitMillsDean"
)`
      }
    ]
  },

  // =========================================================================
  // MODULE 5: EXPERIMENTAL PIPELINE & BENCHMARK (CCR2)
  // =========================================================================

  "l5_1": {
    id: "l5_1",
    tag: "Core",
    relevance: 10,
    title: "5.1 Stanovení ROCS prahu & ROC analýza (threshold_analysis.py)",
    summary: "Metodika oddělení aktivních látek od decoyů (CCR2 aktivní vs DUD-E), Youdenův index J = TPR - FPR a self-similarity.",
    slides: [
      {
        title: "1. Vědecká nutnost rigorózní kalibrace prahu odměny",
        content: `V Reinforcement Learningu určuje dělící práh (<code>ROCS_THRESHOLD</code>), které molekuly jsou klasifikovány jako <em>Desired</em> a získávají odměnu.
        <br><br>
        <strong>Rizika špatné volby prahu:</strong>
        <ul>
          <li><strong>Práh příliš vysoký (např. 1.5)</strong>: Generátor náhodným vzorkováním nenalezne žádnou molekulu splňující práh $\\rightarrow$ nulový gradient odměny $\\rightarrow$ trénink selže.</li>
          <li><strong>Práh příliš nízký (např. 0.4)</strong>: Generátor dostává odměnu i za náhodné inaktivní decoye $\\rightarrow$ model generuje netečné struktury.</li>
        </ul>`
      },
      {
        title: "2. Metodika ROC křivky & Youdenova indexu J",
        content: `Skript <code>tutorial/advanced/rocs/threshold_analysis.py</code> provádí:
        <ol>
          <li>Oskórování 75 experimentálně ověřených aktivních ligandů CCR2 a 500 DUD-E decoyů.</li>
          <li>Výpočet ROC křivky (True Positive Rate vs False Positive Rate).</li>
          <li>Nalezení maxima <strong>Youdenova indexu ($J$)</strong>:
            <div class="math-card">
              $$J = \\text{Sensitivity} + \\text{Specificity} - 1 = \\text{TPR} - \\text{FPR}$$
            </div>
          </li>
        </ol>
        Pro CCR2 benchmark byl stanoven optimální práh <strong>0.871</strong> při ploše pod křivkou <strong>AUC = 0.941</strong>.`,
        code: `from tutorial.advanced.rocs.threshold_analysis import run_threshold_analysis

# Spuštění kompletní ROC analýzy a generování reportu
results = run_threshold_analysis(
    actives_csv="rocs_rl_ccr/rdkit_cdpkit/actives_decoys/actives_ccr2_N75.csv",
    decoys_csv="rocs_rl_ccr/rdkit_cdpkit/actives_decoys/decoys_ccr2_N500.csv",
    references_sdf="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
    output_dir="threshold_results"
)

print(f"Optimální práh: {results['optimal_threshold']:.3f}")
print(f"ROC AUC: {results['roc_auc']:.4f}")`
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
        title: "1. Kompletní inicializace a tréninková smyčka",
        content: `Tréninkový cyklus integruje generátor <code>SequenceRNN</code>, mutační síť, prostředí <code>DrugExEnvironment</code> a explorátor <code>SequenceExplorer</code>:`,
        code: `from drugex.training.explorers import SequenceExplorer
from tutorial.advanced.rocs.config import setup_rl_rdkit

# 1. Načtení modelů a vytvoření prostředí s RDKit ROCS + SAScore
agent, mutate, env, output_dir = setup_rl_rdkit()

# 2. Konfigurace RL explorátoru
explorer = SequenceExplorer(
    agent=agent,
    mutate=mutate,
    crover=None,
    env=env,
    epsilon=0.2,       # 20% explorace z mutační sítě
    n_samples=1000     # 1000 generovaných molekul na epochu
)

# 3. Spuštění tréninku na 50 epoch
explorer.fit(
    epochs=50,
    output_dir=str(output_dir)
)`
      },
      {
        title: "2. Monitorování konvergence: desired_ratio & průměrné skóre",
        content: `Během tréninku sledujeme soubor <code>CCR2_rdkit_reinforced_fit.tsv</code>:
        <ul>
          <li><code>desired_ratio</code>: Podíl molekul splňujících $\\text{ROCS} \\ge 0.871$ a $\\text{SAScore} \\ge 0.1$. V epoše 1 bývá $\\approx 0.05$, v epoše 50 dosahuje $> 0.55$.</li>
          <li><code>mean_score</code>: Průměrná celková odměna roste z $\\approx 0.20$ k $> 0.70$.</li>
          <li><code>sa_score</code>: Zůstává stabilní kolem $0.85$, což potvrzuje, že molekuly neztrácejí syntetickou dostupnost.</li>
        </ul>`
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
        title: "1. Vzorkování a chemická filtrace z reinforced modelu",
        content: `Po dokončení RL tréninku vygenerujeme 10 000 sloučenin pomocí <code>generate_molecules.py</code> a podrobíme je validačnímu filtru:
        <ol>
          <li><strong>Chemická validita</strong>: Procento syntakticky správných struktur (obvykle $>98\\%$).</li>
          <li><strong>Unikátnost (Uniqueness)</strong>: Podíl vzájemně neidentických struktur ($>90\\%$).</li>
          <li><strong>Novost (Novelty)</strong>: Podíl molekul, které se nenacházejí v trénovací sadě Papyrus ani v databázi ChEMBL ($>85\\%$).</li>
          <li><strong>Interní diverzita (Internal Diversity)</strong>: Průměrná Tanimoto distance mezi vygenerovanými molekulami:
            <div class="math-card">
              $$I(S) = 1 - \\frac{2}{|S|(|S|-1)} \\sum_{i < j} T(s_i, s_j)$$
            </div>
          </li>
        </ol>`,
        code: `from drugex.generate import generate_molecules

df_mols = generate_molecules(
    model_pkg="CCR2_rdkit_reinforced.pkg",
    num_samples=10000,
    keep_undesired=False
)
print(f"Úspěšně vygenerováno {len(df_mols)} vysoce afinitních a synteticky dostupných kandidátů.")`
      }
    ]
  },

  // =========================================================================
  // MODULE 6: SCAFFOLDS, CLI & HPC AUTOMATION
  // =========================================================================

  "l6_1": {
    id: "l6_1",
    tag: "Core",
    relevance: 9,
    title: "6.1 Fragmentový návrh, BRICS štěpení & FragSequenceExplorer",
    summary: "Cílený růst z fragmentů, design linkerů pro vícedoménové proteiny a fragmentační konvertory.",
    slides: [
      {
        title: "1. BRICS štěpení a fragmentový design v DrugEx",
        content: `Algoritmus <strong>BRICS (Breaks on Retrosynthetically Interesting Chemical Substructures)</strong> (v <code>drugex/molecules/converters/fragmenters.py</code>) štěpí molekuly na 16 typech synteticky proveditelných vazeb:
        <ul>
          <li>L1–L2: Amidické vazby (např. reakce aminu s acylchloridem).</li>
          <li>L3–L4: Estery a ethery.</li>
          <li>L5–L6: C-C vazby mezi aromatickými kruhy (Suzuki-Miyaura coupling).</li>
        </ul>
        Fragmentové modely <code>FragSequenceExplorer</code> a <code>FragGraphExplorer</code> umožňují fixovat fragment vázající se do specifické sub-kapsy a nechat generátor navrhnout variabilní zbytek molekuly.`
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
        title: "1. Přehled CLI rozhraní a matice parametrů",
        content: `DrugEx poskytuje 3 hlavní CLI moduly:
        <ul>
          <li><code>python -m drugex.dataset</code>: Příprava, standardizace, fragmentace a kódování datasetů.</li>
          <li><code>python -m drugex.train</code>: Trénování generátorů (PT = Pre-training, FT = Fine-tuning, RL = Reinforcement Learning).</li>
          <li><code>python -m drugex.generate</code>: Dávkové generování a skórování molekul z natrénovaného modelu.</li>
        </ul>`,
        code: `# Kompletní 3-krokový produkční workflow v příkazové řádce:

# 1. Krok: Příprava dat pro grafový model s fixním scaffoldem
python -m drugex.dataset -b ./workdir -i A2AR_LIGANDS.tsv -mc SMILES -o arl -mt graph -sf "c1[nH]c2c(n1)nc(nc2O)O"

# 2. Krok: Spuštění RL tréninku s GPU akcelerací
python -m drugex.train -tm RL -b ./workdir -i arl -o arl_rl -ag arl_ft.pkg -pr Papyrus_PT.pkg -sas -e 50 -bs 64 -gpu 0

# 3. Krok: Vygenerování 5000 kandidátních molekul
python -m drugex.generate -b ./workdir -i arl_test_graph.txt -g arl_rl.pkg -n 5000 -gpu 0`
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
        title: "1. Produkční Slurm dávkový skript pro akademický superpočítač",
        content: `Kompletní šablona dávkového skriptu pro spuštění výpočtů na univerzitním výpočetním klastru (např. VŠCHT / MetaCentrum / IT4Innovations):`,
        code: `#!/bin/bash
#SBATCH --job-name=drugex_thesis_rocs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=drugex_run_%j.log
#SBATCH --error=drugex_run_%j.err

echo "Spousteni DrugEx na uzlu: $(hostname)"
echo "Alokovana GPU: $CUDA_VISIBLE_DEVICES"

# Nacteni modulu a virtualniho prostredi
module load cuda/12.1 python/3.11
source ~/.venv/bin/activate

# Nastaveni OpenEye licence a vlaken
export OE_LICENSE=/storage/projects/licenses/oe_license.txt
export OMP_NUM_THREADS=1

# Spusteni tréninkoveho tutorialu
python tutorial/advanced/rocs/rocs_rl_tutorial.ipynb

echo "Uloha dokoncena: $(date)"`
      }
    ]
  }
};
