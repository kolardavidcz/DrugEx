/**
 * DrugEx Hub — Module 1: De Novo Molecular Generation & Representations
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M1_LECTURES = {
  // =========================================================================
  // LECTURE 1.1
  // =========================================================================
  "l1_1": {
    id: "l1_1",
    tag: "Core",
    relevance: 10,
    title: "1.1 Molekulární reprezentace, tokenizace & slovník (VocSmiles)",
    summary: "1D SMILES vs 2D grafy vs fragmentové vaky, tokenizační smlouva VocSmiles, regulární výrazy a sanitace přes SmilesStandardizer.",
    slides: [
      {
        title: "1. Paradigma molekulárních reprezentací v chemoinformatice a AI",
        content: `Výpočetní reprezentace molekuly je základním kamenem všech generativních algoritmů. V tradiční chemoinformatice byly molekuly po desetiletí kódovány jako fixní binární vektory (např. Morganovy fingerprinty, MACCS klíče), které sice umožňují rychlé vyhledávání podobnosti, ale jsou <strong>nerozbalitelné</strong> (jednosměrné) — z otisku nelze zrekonstruovat původní molekulární strukturu.
        <br><br>
        Generativní de novo design vyžaduje reprezentace, které jsou <strong>invertibilní</strong> a umožňují generovat nové molekuly atom po atomu nebo fragment po fragmentu. V moderní literatuře rozlišujeme tři hlavní reprezentace:
        <br><br>
        <h4>1. 1D Lineární řetězce (SMILES, DeepSMILES, SELFIES)</h4>
        SMILES (Simplified Molecular Input Line Entry System) vyvinutý Weiningerem v roce 1988 je dosud nejrozšířenějším standardem. Jde o linearizaci molekulárního grafu pomocí průchodu do hloubky (DFS), kde:
        <ul>
          <li>Atomy jsou reprezentovány atomovými symboly (např. <code>C</code>, <code>N</code>, <code>O</code>, <code>S</code>, <code>P</code>, <code>F</code>, <code>Cl</code>, <code>Br</code>, <code>I</code>).</li>
          <li>Aromatické atomy v konjugovaných systémech jsou značeny malými písmeny (<code>c</code>, <code>n</code>, <code>o</code>, <code>s</code>).</li>
          <li>Větvení molekulárního řetězce je uzavřeno v kulatých závorkách <code>(...)</code>.</li>
          <li>Rozpojení a uzavření cyklických struktur je kódováno shodnými celočíselnými indexy bezprostředně za atomy (např. <code>c1ccccc1</code> pro benzen). Dvojciferné kruhy se uvozují procentem (např. <code>%12</code>).</li>
          <li>Řády vazeb: jednoduchá vazba (implicitní nebo <code>-</code>), dvojná vazba (<code>=</code>), trojná vazba (<code>#</code>), aromatická vazba (<code>:</code>).</li>
          <li>Stereochemie na tetraedrických centrech: <code>@</code> (proti směru hodinových ručiček) a <code>@@</code> (ve směru hodinových ručiček). Geometrická izomerie na dvojných vazbách: <code>/</code> a <code>\\</code>.</li>
        </ul>
        <br>
        <h4>2. 2D Molekulární grafy (Attributed Molecular Graphs)</h4>
        Molekula je formálně definována jako graf $G = (V, E)$, kde uzly $V$ reprezentují atomy a hrany $E$ reprezentují kovalentní vazby:
        <ul>
          <li>Každému uzlu $v_i \\in V$ náleží vektor příznaků $\\mathbf{x}_i \\in \\mathbb{R}^{d_v}$ (např. typ prvku, formální náboj, hybridizace $sp/sp^2/sp^3$, počet připojených vodíků, aromatický status).</li>
          <li>Každé hraně $e_{ij} \\in E$ náleží typ vazby (jednoduchá, dvojná, trojná, aromatická).</li>
        </ul>
        <br>
        <h4>3. Fragmentové reprezentace (Synthons & Bag of Fragments)</h4>
        Molekuly jsou dekomponovány na synteticky proveditelné stavební bloky pomocí standardizovaných štěpných pravidel (např. BRICS nebo RECAP). Tím se generativní prostor omezuje pouze na molekuly, které lze snadno syntetizovat z komerčně dostupných výchozích látek.`
      },
      {
        title: "2. Hloubkové srovnání reprezentací pro generativní AI",
        content: `Výběr reprezentace zásadně ovlivňuje stabilitu tréninku i chemickou kvalitu navržených kandidátů:`,
        compare: {
          leftTitle: "1D SMILES Sekvence (Jazykové modely)",
          leftContent: `<strong>Silné stránky:</strong>
          <ul>
            <li>Extrémně kompaktní reprezentace s minimální paměťovou režií.</li>
            <li>Přímá aplikovatelnost nejmodernějších NLP architektur (LSTM, GRU, GPT Transformer).</li>
            <li>Vysoká rychlost tréninku a paralelizovaného vzorkování na GPU (tisíce molekul za sekundu).</li>
          </ul>
          <strong>Slabé stránky:</strong>
          <ul>
            <li>Gramatická křehkost: Náhodná záměna jednoho tokenu může způsobit nevalidní syntaxi (např. neuzavřený kruh <code>c1ccccc</code>).</li>
            <li>Sousední atomy v prostoru mohou být v lineárním řetězci vzdáleny desítky tokenů.</li>
          </ul>`,
          rightTitle: "2D Molekulární grafy (Grafové modely)",
          rightContent: `<strong>Silné stránky:</strong>
          <ul>
            <li>100% syntaktická a valenční validita generovaných struktur.</li>
            <li>Invariantnost vůči libovolné permutaci pořadí atomů v molekule.</li>
            <li>Přirozené a rigorózní scaffold-constrained generování (pevné jádro s volnými substituenty).</li>
          </ul>
          <strong>Slabé stránky:</strong>
          <ul>
            <li>Výpočetně náročné maticové operace ($O(N^2)$ pro tenzory sousednosti).</li>
            <li>Složitější difúzní nebo autoregresní vzorkování ve srovnání s rychlým generováním textových tokenů.</li>
          </ul>`
        }
      },
      {
        title: "3. Tokenizační kontrakt DrugEx: VocSmiles & Regulární výrazy",
        content: `Standardní znaková tokenizace běžná v NLP je v chemickém modelování hrubou chybou. Pokud bychom řetězec <code>Clc1ccccc1</code> rozdělili po písmenech na <code>['C', 'l', 'c', '1', ...]</code>, model by interpretoval <code>'C'</code> jako alifatický uhlík a <code>'l'</code> jako neznámý token, namísto atomu chloru.
        <br><br>
        V DrugEx je tokenizace zapouzdřena ve třídě <code>VocSmiles</code> (v souboru <code>drugex/data/corpus/vocabulary.py</code>). Tokenizér využívá precizně zkonstruovaný chemoinformatický regulární výraz:
        <div class="math-card">
          $$\\text{Pattern} = \\texttt{\\[[^\\]]+\\]|Br|Cl|Si|Na|Ca|Fe|@@|@|\\/|\\\\|\\%\\d{2}|\\d|=|#|\\$|:|~|\\.|[a-zA-Z]}$$
        </div>
        <br>
        Tento regulární výraz provádí deterministický rozklad v následujícím pořadí priorit:
        <ol>
          <li><strong>Explicitní atomové bloky v závorkách</strong>: <code>\\[[^\\]]+\\]</code> (např. <code>[NH+]</code>, <code>[C@@H]</code>, <code>[O-]</code>, <code>[se]</code>).</li>
          <li><strong>Dvoupísmenné halogeny a kovy</strong>: <code>Br</code>, <code>Cl</code>, <code>Si</code>, <code>Na</code>, <code>Ca</code>, <code>Fe</code>.</li>
          <li><strong>Stereochemické deskriptory</strong>: <code>@@</code> (ve směru hodinových ručiček), <code>@</code> (proti směru), <code>/</code>, <code>\\</code>.</li>
          <li><strong>Dvouciferná čísla kruhů</strong>: <code>%\\d{2}</code> (např. <code>%10</code>, <code>%12</code>) následovaná jednocifernými čísly kruhů <code>\\d</code>.</li>
          <li><strong>Řády vazeb a operátory</strong>: <code>=</code> (dvojná), <code>#</code> (trojná), <code>$</code> (čtyřná), <code>:</code> (aromatická), <code>~</code> (nespecifikovaná), <code>.</code> (oddělovač fragmentů/solí).</li>
          <li><strong>Jednopísmenné alifatické a aromatické atomy</strong>: <code>[a-zA-Z]</code> (např. <code>C</code>, <code>c</code>, <code>N</code>, <code>n</code>, <code>O</code>, <code>o</code>, <code>S</code>, <code>s</code>, <code>P</code>, <code>p</code>, <code>F</code>, <code>I</code>).</li>
        </ol>`,
        code: `import re
from drugex.data.corpus.vocabulary import VocSmiles

# Ukázka interního fungování regulárního výrazu VocSmiles
SMILES_REGEX = r"\\[[^\\]]+\\]|Br|Cl|Si|Na|Ca|Fe|@@|@|\\/|\\\\|%\\d{2}|\\d|=|#|\\$|:|~|\\.|[a-zA-Z]"
pattern = re.compile(SMILES_REGEX)

sample_smiles = "Cc1ccc(Cl)c([C@@H](N)C(=O)[O-])c1"
tokens = pattern.findall(sample_smiles)
print("Rozklad na chemické tokeny:")
print(tokens)
# Výstup: ['C', 'c', '1', 'c', 'c', 'c', '(', 'Cl', ')', 'c', '1', '(', '[C@@H]', '(', 'N', ')', 'C', '(', '=', 'O', ')', '[O-]', ')', 'c', '1']

# Načtení kompletního slovníku VocSmiles
voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)
print(f"Velikost slovníku: {voc.size} tokenů")
encoded_tensor = voc.encode([sample_smiles])
print(f"Tvar kódovaného tenzoru: {encoded_tensor.shape}")`,
        alert: {
          type: "important",
          title: "Struktura indexů řídicích tokenů ve VocSmiles",
          text: `VocSmiles vyhrazuje specifické indexy pro řízení běhu:
          <ul>
            <li><code>Index 0 (&lt;PAD&gt;)</code>: Zarovnání délky v tenzorových batších (padding).</li>
            <li><code>Index 1 (&lt;START&gt; / &lt;GO&gt;)</code>: Inicializační signál pro autoregresní generování.</li>
            <li><code>Index 2 (&lt;END&gt; / &lt;STOP&gt;)</code>: Signál ukončení sekvence (model přestane generovat další tokeny).</li>
            <li><code>Index 3 (&lt;UNK&gt;)</code>: Zástupný symbol pro neznámé znaky mimo slovník.</li>
            <li><code>Indexy 4+</code>: Chemické tokeny (atomy, vazby, stereochemie).</li>
          </ul>`
        }
      },
      {
        title: "4. Sanitační pipeline: SmilesStandardizer",
        content: `Data pocházející z veřejných bioaktivních repozitářů (ChEMBL, PubChem, Papyrus) obsahují rozsáhlý šum. Přítomnost krystalizačních solí, nekonzistentních tautomerních forem a směsí stereoizomerů by vedla ke zkreslení generativního modelu.
        <br><br>
        Třída <code>SmilesStandardizer</code> (v <code>drugex/molecules/converters/standardizers.py</code>) implementuje přísný čtyřfázový čistící protokol:
        <ol>
          <li><strong>Odsolení (Desalting / Salt Stripping)</strong>: Rozdělení molekuly na kovalentně nespojené komponenty pomocí <code>.</code> a zachování pouze největšího fragmentu (tzv. active pharmaceutical ingredient - API). Menší protiionty (např. $Na^+$, $Cl^-$, acetát, sulfát) a molekuly krystalové vody jsou odstraněny.</li>
          <li><strong>Nábojová neutralizace (Charge Neutralization)</strong>: Kde je to chemicky možné bez porušení valence, ionizované funkční skupiny jsou převedeny na neutrální formu (např. karboxylátové anionty $[O^-]$ jsou protonovány na hydroxylovou skupinu $OH$, kvarterní aminy jsou deprotonovány).</li>
          <li><strong>Tautomerní kanonizace (Tautomer Enumeration & Canonicalization)</strong>: Aplikace tautomerních transformací podle RDKit MolVS protokolu k nalezení energeticky nejstabilnějšího reprezentanta (např. keto-enol rovnováha posunuta ke stabilnímu keto tautomeru).</li>
          <li><strong>Odstranění izotopů (Isotope Stripping)</strong>: Nahrazení izotopických atomů ($^{13}C$, $^{2}H$ deuterium, $^{15}N$) jejich přirozenými izotopickými ekvivalenty.</li>
          <li><strong>Validace chirálních center (Stereo Handling)</strong>: Při nastavení <code>drop_unspecified=True</code> jsou vyřazeny molekuly s nejasně definovanou stereochemií na asymetrických uhlících.</li>
        </ol>`,
        code: `from drugex.molecules.converters.standardizers import SmilesStandardizer

standardizer = SmilesStandardizer(
    drop_unspecified=True,  # Zahodit nedefinovanou stereochemii
    drop_isotopes=True      # Odstranit izotopické značení
)

# Test na surové sloučenině z bio-databáze
raw_smiles = "[Na+].CC(=O)[O-].[2H]c1ccccc1" # Směs octanu sodného s deuterovaným benzenem
clean_smiles = standardizer(raw_smiles)

print(f"Původní surový vstup : {raw_smiles}")
print(f"Kanonický výstup     : {clean_smiles}") # Výstup: c1ccccc1 (největší neutrální fragment bez deuteria)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 1.2
  // =========================================================================
  "l1_2": {
    id: "l1_2",
    tag: "Core",
    relevance: 10,
    title: "1.2 Generativní modely: Sequence RNN, Sequence & Graph Transformer",
    summary: "Architektury generátorů v DrugEx: Autoregresní GRU/LSTM systém, causally masked Transformer a grafový maticový generátor.",
    slides: [
      {
        title: "1. Architektura Sequence RNN (LSTM / GRU)",
        content: `Generování molekulárních sekvencí lze formulovat jako úlohu autoregresního modelování jazyka. Pravděpodobnost vzniku celé sekvence tokenů $X = (x_1, x_2, \\dots, x_T)$ je rozložena podle řetízkového pravidla pravděpodobnosti:
        <div class="math-card">
          $$P(X) = \prod_{t=1}^{T} P(x_t \\mid x_1, x_2, \\dots, x_{t-1}; \\theta)$$
        </div>
        <br>
        Třída <code>SequenceRNN</code> (v <code>drugex/training/generators/sequence_rnn.py</code>) implementuje tuto pravděpodobnost pomocí vícevrstvé rekurentní sítě LSTM nebo GRU:
        <ol>
          <li><strong>Embeddingová vrstva ($E \\in \\mathbb{R}^{|V| \\times d_{emb}}$)</strong>: Mapuje diskrétní token $x_{t-1}$ na spojitý vektor $\\mathbf{e}_{t-1}$.</li>
          <li><strong>LSTM rekurentní vrstvy ($L$ vrstev)</strong>: V každém časovém kroku $t$ aktualizují skrytý stav $\\mathbf{h}_t$ a stav buňky $\\mathbf{c}_t$ na základě vstupu $\\mathbf{e}_{t-1}$ a předchozího stavu $\\mathbf{h}_{t-1}$:
            <div class="math-card">
              $$\\begin{aligned}
              \\mathbf{f}_t &= \\sigma(W_f [\\mathbf{h}_{t-1}, \\mathbf{e}_{t-1}] + b_f) \\quad &\\text{(Zapomínací brána)} \\
              \\mathbf{i}_t &= \\sigma(W_i [\\mathbf{h}_{t-1}, \\mathbf{e}_{t-1}] + b_i) \\quad &\\text{(Vstupní brána)} \\
              \tilde{\\mathbf{c}}_t &= \\tanh(W_c [\\mathbf{h}_{t-1}, \\mathbf{e}_{t-1}] + b_c) \\quad &\\text{(Kandidátní stav)} \\
              \\mathbf{c}_t &= \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \tilde{\\mathbf{c}}_t \\quad &\\text{(Aktualizovaný stav buňky)} \\
              \\mathbf{o}_t &= \\sigma(W_o [\\mathbf{h}_{t-1}, \\mathbf{e}_{t-1}] + b_o) \\quad &\\text{(Výstupní brána)} \\
              \\mathbf{h}_t &= \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t) \\quad &\\text{(Skrytý stav)}
              \\end{aligned}$$
            </div>
          </li>
          <li><strong>Lineární projekční hlava</strong>: Projekce $\\mathbf{h}_t$ do dimenze slovníku $|V|$ pro získání neznormalizovaných logitů $\\mathbf{z}_t = W_{out} \\mathbf{h}_t + b_{out}$.</li>
          <li><strong>Teplotní vzorkování (Softmax with Temperature $T$)</strong>:
            <div class="math-card">
              $$P(x_t = k \\mid x_{<t}) = \\frac{\\exp(z_{t,k} / T)}{\sum_{j=1}^{|V|} \\exp(z_{t,j} / T)}$$
            </div>
            kde $T < 1.0$ vede ke konzervativnímu generování nejčastějších fragmentů, zatímco $T > 1.0$ podporuje vyšší exploraci a strukturní novost.
          </li>
        </ol>`,
        code: `import torch
import torch.nn as nn
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

# Inicializace slovníku a načtení natrénovaného modelu
voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)
model = SequenceRNN(voc, is_lstm=True)
model.loadStatesFromFile("Papyrus05.5_smiles_rnn_PT.pkg")
model.eval()

# Inspekce generativního vzorkování
with torch.no_grad():
    # Vzorkování 10 molekul s nastavením teploty
    molecules = model.sample(n_samples=10)
    print("Vygenerované molekuly (SMILES):")
    for i, m in enumerate(molecules, 1):
        print(f"[{i:02d}] {m}")`
      },
      {
        title: "2. Sequence Transformer & Kauzální Samo-Pozornost (Causal Self-Attention)",
        content: `Ačkoliv jsou RNN modely výpočetně efektivní, trpí postupným zapomínáním informací při dlouhých sekvencích (např. komplexní polycyklické sloučeniny s délkou SMILES $> 80$ tokenů).
        <br><br>
        <code>SequenceTransformer</code> (v <code>drugex/training/generators/sequence_transformer.py</code>) využívá architekturu Transformer Decoder. Místo rekurentních spojů používá mechanismus vícehlavé pozornosti (Multi-Head Attention) s přidaným pozičním kódováním (Positional Encoding):
        <br><br>
        Pro zachování autoregresní vlastnosti (zákaz nahlížení do budoucích tokenů) se aplikuje <strong>kauzální maska</strong> $M \\in \\mathbb{R}^{T \\times T}$:
        <div class="math-card">
          $$\\text{CausalAttention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M \\right) V$$
        </div>
        kde maskovací matice $M$ má hodnoty:
        <div class="math-card">
          $$M_{ij} = \\begin{cases} 0 & \\text{pro } j \\le i \\ -\infty & \\text{pro } j > i \\end{cases}$$
        </div>
        Díky tomu jsou při tréninku všechny pozice sekvence vyhodnocovány <strong>paralelně v jediném dopředném průchodu</strong>, což zkracuje čas tréninku na velkých korpusech na zlomek času potřebného pro RNN.`
      },
      {
        title: "3. Graph Transformer: Tenzorová reprezentace a přímé generování grafu",
        content: `<code>GraphTransformer</code> (v <code>drugex/training/generators/graph_transformer.py</code>) zcela opouští linearizaci do SMILES a modeluje přímo distribuci molekulárních grafů.
        <br><br>
        V každém kroku model operuje se dvěma provázanými tenzory:
        <ul>
          <li><strong>Uzlový tenzor (Node Tensor) $\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times d_v}$</strong>: Kóduje typy atomů pro maximálně $N$ atomů v molekule.</li>
          <li><strong>Hranový tenzor (Edge Adjacency Tensor) $\\mathbf{A} \\in \\mathbb{R}^{B \\times N \\times N \\times d_e}$</strong>: Kóduje typy vazeb mezi všemi dvojicemi atomů $(i, j)$.</li>
        </ul>
        <br>
        Generování probíhá iterativním přidáváním atomů a současnou predikcí vazeb k již existujícím atomům. Tím je dosaženo <strong>100% chemické validity</strong> bez rizika syntaktických chyb závorek nebo čísel cyklů.`
      }
    ]
  },

  // =========================================================================
  // LECTURE 1.3
  // =========================================================================
  "l1_3": {
    id: "l1_3",
    tag: "Core",
    relevance: 9,
    title: "1.3 Transfer Learning: Pre-training na Papyrus & Target Fine-Tuning",
    summary: "Dvoustupňový tréninkový proces: zkoumání obecného chemického prostoru vs adaptace na cílové bioaktivní ligandy.",
    slides: [
      {
        title: "1. Teoretické základy transferového učení v generativní chemii",
        content: `Při aplikaci generativní AI na konkrétní biologický cíl (např. receptor CCR2 nebo studovaný IDP cíl) narážíme na zásadní omezení: počet známých experimentálně ověřených ligandů pro daný cíl bývá malý (desítky až stovky sloučenin).
        <br><br>
        Pokud bychom hlubokou neuronovou síť se stovkami tisíc parametrů trénovali pouze na této malé sadě:
        <ul>
          <li>Model by se během několika epoch přeučil (overfitting).</li>
          <li>Generované molekuly by byly pouhými triviálními variacemi trénovacích struktur bez novosti.</li>
          <li>Model by neuměl správně uzavírat neobvyklé cykly nebo navrhovat netradiční bioisosterní náhrady.</li>
        </ul>
        <br>
        Řešením je <strong>dvoustupňový transfer znalostí (Two-Stage Transfer Learning)</strong>:
        <ol>
          <li><strong>Fáze 1: Pre-training na databázi Papyrus v05.5 (~1,5 milionu struktur)</strong>:
            Model se učí obecnou distribuci chemického prostoru "všech lékových molekul". Minimalizuje křížovou entropii (Cross-Entropy Loss):
            <div class="math-card">
              $$\\mathcal{L}_{PT}(\\theta) = -\\frac{1}{|D_{Papyrus}|} \sum_{X \\in D_{Papyrus}} \sum_{t=1}^{T} \log P(x_t \\mid x_{<t}; \\theta)$$
            </div>
          </li>
          <li><strong>Fáze 2: Fine-Tuning na cílových ligandech (CCR2 / IDP target)</strong>:
            Váhy předtrénovaného modelu jsou doučeny na sadě známých aktivních ligandů s velmi nízkým learning rate ($\eta = 10^{-4}$ nebo $5 \\times 10^{-5}$) po dobu 50–100 epoch. Distribuce generovaných molekul se posune do relevantní chemické oblasti specifické pro vazebné místo.
          </li>
        </ol>`,
        alert: {
          type: "tip",
          title: "Využití pro bakalářskou práci",
          text: "Fine-tuned model se v DrugEx stává tzv. <strong>Mutate Network</strong> (prior politikou) $\pi_0$, která v následném Reinforcement Learningu slouží jako stabilizační kotva chránící model před mode collapse."
        }
      },
      {
        title: "2. Kompletní příprava dat a spuštění fine-tuningu v CLI",
        content: `Následující příkazy demonstrují kompletní workflow přípravy a tréninku modelu pro receptor CCR2 (benchmark pro bakalářskou práci):`,
        code: `# Krok 1: Stažení předtrénovaných modelů DrugEx (pokud nejsou lokálně)
python -m drugex.download -m smiles_rnn

# Krok 2: Zpracování a standardizace datasetu známých ligandů CCR2
python -m drugex.dataset \\
    -b tutorial/CLI/examples \\
    -i CCR_HUMAN_AL.tsv \\
    -mc SMILES \\
    -o ccr2_target_corpus \\
    -mt smiles \\
    -nof

# Krok 3: Spuštění Fine-Tuningu (FT)
# -tm FT : Fine-Tuning režim
# -ag : Vstupní předtrénovaný model (Papyrus RNN)
# -e 100 : 100 trénovacích epoch
# -bs 64 : Velikost batche
# -lr 0.0001 : Nízký learning rate pro prevenci katastrofického zapomínání
python -m drugex.train \\
    -tm FT \\
    -b tutorial/CLI/examples \\
    -i ccr2_target_corpus \\
    -ag models/pretrained/Papyrus05.5_smiles_rnn_PT.pkg \\
    -e 100 \\
    -bs 64 \\
    -lr 0.0001 \\
    -gpu 0`
      }
    ]
  }
};
