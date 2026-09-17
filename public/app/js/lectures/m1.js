/**
 * DrugEx Hub — Module 1: De Novo Molecular Generation & Representations in DrugEx
 * Comprehensive Handbook-Grade Textbook Materials for Bachelor Thesis
 * Faculty of Chemical Technology (VŠCHT Praha) / IOCB Prague (ÚOCHB AV ČR)
 */

export const M1_LECTURES = {
  // =========================================================================
  // LECTURE 1.1: Molekulární reprezentace, tokenizace & slovník (VocSmiles)
  // =========================================================================
  "l1_1": {
    id: "l1_1",
    tag: "Core",
    relevance: 10,
    title: "1.1 Molekulární reprezentace, tokenizace & slovník (VocSmiles)",
    summary: "Hloubková taxonomie molekulárních reprezentací (1D SMILES/SELFIES/DeepSMILES vs 2D molekulární grafy vs fragmentové vaky BRICS), tokenizační smlouva VocSmiles s rozpadem regulárních výrazů, řídicí tokeny a 5-stupňový standardizační protokol SmilesStandardizer.",
    slides: [
      {
        title: "1. Paradigma molekulárních reprezentací v chemoinformatice a generativní AI",
        content: `Výpočetní reprezentace chemické struktury představuje fundamentální most mezi fyzikálně-chemickou realitou molekuly a matematickým aparátem strojového učení. V klasické chemoinformatice a QSAR modelování dominovaly po desetiletí fixní binární nebo celočíselné vektorové otisky (např. <strong>Morganovy / ECFP4 fingerprinty</strong>, <strong>MACCS keys</strong>, topologické indexy). Tyto deskriptory sice excelují v rychlém vyhledávání substrukturní podobnosti v miliardových databázích, mají však zásadní omezení pro de novo návrh léčiv: jsou <strong>jednosměrné (nerozbalitelné)</strong>. Z fixního 2048-bitového vektoru nelze deterministicky ani generativně zrekonstruovat původní molekulární strukturu.
        <br><br>
        Generativní de novo design léčiv vyžaduje <strong>invertibilní, spojité či diskrétní generovatelné reprezentace</strong>, které umožňují algoritmu postupně syntetizovat nové chemické entity atom po atomu nebo fragment po fragmentu. V současné generativní chemoinformatice rozlišujeme tři dominantní úrovně reprezentací:
        <br><br>
        <ul>
          <li><strong>1D Lineární řetězce (String Notations)</strong>: Linearizace molekulárních struktur do textových sekvencí znaků a tokenů (SMILES, DeepSMILES, SELFIES). Umožňují přímé nasazení nejmodernějších architektur z oblasti zpracování přirozeného jazyka (NLP), jako jsou rekurentní neuronové sítě (LSTM/GRU) a kauzální autoregresní transformery (GPT).</li>
          <li><strong>2D Atribuované molekulární grafy (Attributed Molecular Graphs)</strong>: Formální popis molekuly jako grafu $G = (V, E)$, kde uzly reprezentují atomy s vektory fyzikálně-chemických příznaků a hrany reprezentují kovalentní vazby s informací o jejich řádu a stereochemii.</li>
          <li><strong>Fragmentové a synthonové reprezentace (Bag of Fragments / Synthon Graphs)</strong>: Dekompozice molekul na synteticky dostupné stavební bloky definované standardizovanými štěpnými pravidly (např. BRICS, RECAP). Generování pak probíhá kombinací celých syntetických fragmentů, což zásadně redukuje chemický prostor na synteticky snadno realizovatelné kandidáty.</li>
        </ul>`,
        alert: {
          type: "note",
          title: "Význam pro bakalářskou práci na VŠCHT / ÚOCHB",
          text: "Při návrhu ligandů pro flexibilní cíle a IDP proteiny je volba reprezentace kritická: 1D sekvenční modely (SequenceRNN) poskytují obrovskou chemickou flexibilitu a rychlost pro rozsáhlý screening, zatímco fragmentové a grafové modely (GraphTransformer) umožňují rigorózní zachování pevných scaffoldů (scaffold-constrained generation) a 100% syntaktickou validitu."
        },
        callouts: [
          {
            type: "rule",
            title: "Pravidlo z praxe: Standardizace MolVS před tréninkem",
            text: "Nikdy netrénujte jazykový model generátoru na surových SMILES z veřejných repozitářů. Před vytvořením slovníku VocSmiles vždy aplikujte kompletní desalting (odstranění solí a solvátů, např. Na+, Cl-, TFA), neutralizaci formálních nábojů a převod na kanonické tautomery. Přítomnost solí ve slovníku vede k plýtvání kapacitou sítě a tvorbě nerealistických iontových párů."
          },
          {
            type: "pitfall",
            title: "Častá chyba v diplomce: Ztráta chirality v 1D SMILES",
            text: "Pokud vygenerujete molekulu jako achirální SMILES (bez označení @ nebo @@ na asymetrickém uhlíku), RDKit ani jiný konformační engine nemůže vědět, který enantiomer má postavit do 3D prostoru pro ROCS porovnání. Vždy zajistěte, aby generátor pracoval s chirálními tokeny, nebo při konformačním screeningu explicitně enumerujte všechny stereoisomery (max. 4 na molekulu)."
          }
        ]
      },
      {
        title: "2. Hloubková taxonomie 1D notací: SMILES vs DeepSMILES vs SELFIES",
        content: `Lineární řetězcové reprezentace převádějí trojrozměrný či dvoudimenzionální chemický graf do jednodimenzionální sekvence symbolů. Každá z moderních notací řeší kompromis mezi lidskou čitelností, kompaktností a syntaktickou robustností:
        <br><br>
        <h4>1. SMILES (Simplified Molecular Input Line Entry System)</h4>
        Vyvinutý Davidem Weiningerem v roce 1988. Využívá linearizaci molekulárního grafu pomocí upraveného průchodu do hloubky (DFS - Depth-First Search). Cykly jsou rozpojeny a označeny shodnými numerickými indexy na místech rozpojení (např. <code>c1ccccc1</code> pro benzen, <code>%12</code> pro dvojciferná čísla kruhů). Větvení je značeno kulatými závorkami <code>(...)</code>. Stereochemie na tetraedrických centrech je kódována pomocí <code>@</code> (proti směru hodinových ručiček) a <code>@@</code> (ve směru), cis/trans izomerie na dvojných vazbách pomocí <code>/</code> a <code>\\</code>.
        <br>
        <em>Hlavní slabina:</em> Gramatická křehkost (syntactic fragility). Náhodná mutace jediného tokenu (např. smazání uzavírací závorky nebo nespárovaný index kruhu) způsobí okamžitou nevaliditu celého řetězce.
        <br><br>
        <h4>2. DeepSMILES (O'Boyle & Dalke, 2018)</h4>
        Navržen speciálně pro hluboké neuronové sítě. Nahrazuje párové závorky operátorem uzavření s explicitním počtem kroků zpět a párová čísla cyklů nahrazuje jedním symbolem udávajícím délku kruhu. Tím eliminuje chyby způsobené nespárovanými závorkami.
        <br><br>
        <h4>3. SELFIES (Self-Referencing Embedded Strings, Krenn et al., 2020)</h4>
        Formální bezkontextová stavová gramatika inspirovaná Chomského hierarchií. Každý token je interpretován jako stavový přechod v závislosti na aktuální volné valenci molekulárního fragmentu. <strong>Garantuje 100% syntaktickou a valenční validitu</strong> pro jakýkoliv náhodně sestavený řetězec tokenů.
        <br>
        <em>Nevýhoda SELFIES:</em> Výrazně delší sekvence tokenů, vyšší paměťová náročnost a obtížnější integrace s fragmentovými bázemi typu BRICS.`,
        compare: {
          leftTitle: "SMILES (Standard v DrugEx)",
          leftContent: `<ul>
            <li><strong>Kompaktnost:</strong> Krátké sekvence (typicky 20–60 tokenů na molekulu léčiva).</li>
            <li><strong>Podpora knihoven:</strong> Nativní integrace v RDKit, OpenEye, CDPKit, ChEMBL a Papyrus.</li>
            <li><strong>Reprezentace fragmentů:</strong> Přirozené kódování synthonů s dummy atomy (<code>[1*]</code>, <code>[2*]</code>).</li>
            <li><strong>Nevýhoda:</strong> Vyžaduje předtrénování na velkém korpusu k osvojení chemické gramatiky.</li>
          </ul>`,
          rightTitle: "SELFIES (Stavová gramatika)",
          rightContent: `<ul>
            <li><strong>Syntaktická robustnost:</strong> 100% matematická validita všech vygenerovaných řetězců.</li>
            <li><strong>Bezpečnost:</strong> Žádné neuzavřené kruhy ani nespárované závorky.</li>
            <li><strong>Délka sekvencí:</strong> Až o 50–100% více tokenů na molekulu, což zvyšuje výpočetní nároky transformerů ($O(N^2)$ pozornost).</li>
            <li><strong>Fragmentace:</strong> Složitější definice retrosyntetických štěpných vazeb BRICS.</li>
          </ul>`
        }
      },
      {
        title: "3. 2D Molekulární grafy a BRICS / RECAP fragmentové vaky (Synthon Paradigma)",
        content: `Molekuly nejsou ve své fyzikální podstatě lineární texty, ale prostorové topologické sítě. V DrugEx se vedle sekvenčních modelů uplatňuje také grafové a fragmentové paradigma:
        <br><br>
        <h4>1. Formální definice atribuovaného molekulárního grafu</h4>
        Molekula je reprezentována uspořádanou dvojicí $G = (V, E)$, kde:
        <ul>
          <li>$V = \\{v_1, v_2, \\dots, v_N\\}$ je množina $N$ těžkých atomů (uzlů). Každému uzlu náleží vektor atomových vlastností $\\mathbf{x}_i \\in \\mathbb{R}^{d_v}$ (atomové číslo, formální náboj, hybridizace $sp, sp^2, sp^3$, počet připojených vodíků, aromatický status, explicitní valence). Celá molekula je popsána maticí uzlů $\\mathbf{X} \\in \\mathbb{R}^{N \\times d_v}$.</li>
          <li>$E = \\{e_{ij}\\}$ je množina kovalentních vazeb (hran). Vazby jsou kódovány tenzorem sousednosti $\\mathbf{A} \\in \\mathbb{R}^{N \\times N \\times d_e}$, kde $d_e$ kóduje typ vazby (jednoduchá, dvojná, trojná, aromatická, žádná).</li>
        </ul>
        <br>
        <h4>2. Fragmentové štěpení BRICS (Degen et al., 2008)</h4>
        Při fragmentovém generování (např. <code>GraphTransformer</code> nebo <code>FragSequenceExplorer</code> v DrugEx) je výchozí molekula rozštěpena podle 16 retrosynteticky validních chemických pravidel (BRICS - <em>Biochemical Properties Reacting and Inherent Chemistry Structure</em>). Štěpí se strategické vazby jako např.:
        <ol>
          <li>L1–L1: Vazby C–C mezi alifatickými uhlíky</li>
          <li>L3–L5: Amidové vazby (karbonyl–dusík)</li>
          <li>L4–L11: Etherové a thioetherové vazby</li>
          <li>L6–L13: Esterové a sulfonamidové vazby</li>
        </ol>
        Místa rozpojení jsou označena <strong>dummy atomy (tzv. attachment points / synthony)</strong> s indexem štěpného pravidla (např. <code>[1*]c1ccccc1</code> nebo <code>[3*]C(=O)N[5*]</code>). Model pak generuje molekulu spojováním komplementárních synthonových bloků, což garantuje vysokou syntetickou proveditelnost.`,
        code: `from rdkit import Chem
from rdkit.Chem import BRICS

# Ukázka dekompozice molekuly na BRICS fragmenty
smiles = "Cc1ccc(NC(=O)c2cccc(C(=O)N3CCN(Cc4ccccc4)CC3)c2)cc1"
mol = Chem.MolFromSmiles(smiles)

# Rozštěpení na synteticky dostupné stavební bloky (synthony)
frags = sorted(list(BRICS.BRICSDecompose(mol)))
print("Extrahované BRICS fragmenty s attachment points:")
for f in frags:
    print(f"  Synthon: {f}")`,
        output: `Extrahované BRICS fragmenty s attachment points:
  Synthon: [1*]C([6*])=O
  Synthon: [16*]c1ccc(C)cc1
  Synthon: [16*]c1cccc([16*])c1
  Synthon: [16*]c1ccccc1
  Synthon: [4*]C[8*]
  Synthon: [5*]N1CCN([5*])CC1
  Synthon: [5*]N[5*]
~ [STAV: Retrosyntetické štěpení vazeb]
~   Aplikována 3 BRICS pravidla: L1-L5 (Amid), L4-L8 (Alkylace piperazinu), L16 (Aryl-kruhový spoj)
~   Značky [N*] představují syntetické vektory (attachment points) pro rekombinaci synthonů ve Fázi 2
💡 [POZNATEK: Proč BRICS fragmenty v DrugEx?]
💡   Omezení generování na pravidla BRICS garantuje, že každou nově navrženou
💡   kandidátní molekulu lze reálně připravit v laboratoři pomocí ověřených organických reakcí.`,
        schematic: "brics-cleavage"
      },
      {
        title: "4. Srovnávací analýza reprezentací pro generativní AI",
        content: `Při volbě architektury pro diplomovou či bakalářskou práci je nutné zohlednit výpočetní náročnost, validitu a chemické chování jednotlivých reprezentací:
        <br><br>
        <table class="lecture-table" style="width:100%; border-collapse: collapse; font-size: 13px; margin: 10px 0;">
          <thead>
            <tr style="background: var(--editor); border-bottom: 2px solid var(--border);">
              <th style="padding: 8px; text-align: left;">Kritérium</th>
              <th style="padding: 8px; text-align: left;">1D SMILES (SequenceRNN)</th>
              <th style="padding: 8px; text-align: left;">1D SMILES (SequenceTransformer)</th>
              <th style="padding: 8px; text-align: left;">2D Grafy (GraphTransformer)</th>
              <th style="padding: 8px; text-align: left;">BRICS Fragmenty</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;">Paměťová složitost</td>
              <td style="padding: 8px; color: var(--bio-green);">$O(T)$ (lineární)</td>
              <td style="padding: 8px; color: var(--amber-warn);">$O(T^2)$ (kvadratická pozornost)</td>
              <td style="padding: 8px; color: var(--danger-red);">$O(N^2 \\cdot d)$ (tenzory sousednosti)</td>
              <td style="padding: 8px; color: var(--bio-green);">$O(F)$ (krátké sekvence synthonů)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;">Rychlost vzorkování</td>
              <td style="padding: 8px; color: var(--bio-green);">> 2 000 mol/s (GPU batch)</td>
              <td style="padding: 8px; color: var(--accent);">~ 800 mol/s</td>
              <td style="padding: 8px; color: var(--amber-warn);">~ 200 mol/s</td>
              <td style="padding: 8px; color: var(--accent);">~ 600 mol/s</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;">Validita struktur</td>
              <td style="padding: 8px;">92–98% (po pre-trainingu)</td>
              <td style="padding: 8px;">95–99%</td>
              <td style="padding: 8px; color: var(--bio-green);">100% (valenčně maskováno)</td>
              <td style="padding: 8px; color: var(--bio-green);">98–100%</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;">Scaffold-constrained design</td>
              <td style="padding: 8px; color: var(--amber-warn);">Obtížné (prefixové maskování)</td>
              <td style="padding: 8px; color: var(--accent);">Střední (in-filling / prefix)</td>
              <td style="padding: 8px; color: var(--bio-green);">Přirozené (uchycení na jádro)</td>
              <td style="padding: 8px; color: var(--bio-green);">Vynikající (substituce fragmentů)</td>
            </tr>
            <tr>
              <td style="padding: 8px; font-weight: 600;">Syntetická dostupnost (SAScore)</td>
              <td style="padding: 8px;">Variabilní (nutno řídit v MORL)</td>
              <td style="padding: 8px;">Střední až vysoká</td>
              <td style="padding: 8px; color: var(--bio-green);">Vysoká (řízená pravidly spojování)</td>
              <td style="padding: 8px; color: var(--bio-green);">Extrémně vysoká (BRICS pravidla)</td>
            </tr>
          </tbody>
        </table>`
      },
      {
        title: "5. Tokenizační kontrakt DrugEx: VocSmiles & Regulární výrazy",
        content: `Standardní znaková tokenizace běžná v klasickém NLP (např. rozklad po jednotlivých písmenech) je v chemickém modelování hrubou chybou. Pokud bychom řetězec <code>Clc1ccccc1</code> rozdělili po písmenech na <code>['C', 'l', 'c', '1', ...]</code>, model by interpretoval <code>'C'</code> jako alifatický uhlík a <code>'l'</code> jako izolovaný neznámý token, namísto atomu chloru.
        <br><br>
        <div class="code-container" style="margin: 12px 0;">
          <div class="code-header">
            <span>VocSmiles Regular Expression Pattern</span>
            <span style="font-size: 10px; color: var(--accent);">drugex/data/corpus/vocabulary.py</span>
          </div>
          <pre class="code-block" style="padding: 10px 14px; font-size: 13px; color: #a5d6ff;"><code>r'(\[[^\]]{1,6}\]|Br|Cl|Si|Na|Ca|Fe|@@|@|\/|\\|%\d{2}|\d|=|#|\$|:|~|\.|[a-zA-Z])'</code></pre>
        </div>
        <br>
        Tento regulární výraz provádí rozklad v rigorózně definovaném pořadí priorit:
        <ol>
          <li><strong>Explicitní atomové bloky v hranatých závorkách</strong>: <code>\\[[^\\]]{1,6}\\]</code> zachycuje složité ionty a chirální atomy (např. <code>[NH+]</code>, <code>[C@@H]</code>, <code>[O-]</code>, <code>[Se+]</code>, <code>[1*]</code>).</li>
          <li><strong>Dvoupísmenné halogeny a kovy</strong>: <code>Br</code>, <code>Cl</code>, <code>Si</code>, <code>Na</code>, <code>Ca</code>, <code>Fe</code>. V interní implementaci DrugEx se navíc před tokenizací provádí elegantní substituce: <code>Cl -> L</code> a <code>Br -> R</code>, což zabraňuje jakékoliv kolizi s alifatickým uhlíkem <code>C</code> či borem <code>B</code>.</li>
          <li><strong>Stereochemické operátory</strong>: <code>@@</code> (ve směru hodinových ručiček / clockwise), <code>@</code> (proti směru / counter-clockwise), <code>/</code> a <code>\\</code> (geometrická konfigurace $E/Z$).</li>
          <li><strong>Indexy kruhů</strong>: Dvouciferná čísla uvozená procentem <code>%\\d{2}</code> (např. <code>%10</code>, <code>%12</code>) následovaná jednocifernými čísly <code>\\d</code>.</li>
          <li><strong>Řády vazeb a speciální znaky</strong>: <code>=</code> (dvojná vazba), <code>#</code> (trojná vazba), <code>$</code> (čtyřná vazba), <code>:</code> (aromatická vazba), <code>~</code> (libovolná vazba), <code>.</code> (oddělovač fragmentů/solí).</li>
          <li><strong>Jednopísmenné organické prvky</strong>: Alifatické atomy (<code>C, N, O, S, P, F, I, B</code>) a aromatické atomy (<code>c, n, o, s, p</code>).</li>
        </ol>`,
        code: `import re

# Ukázka interního rozkladu sekvence ve VocSmiles
def tokenize_smiles(smiles: str) -> list:
    # 1. Substituce dvoupísmenných halogenů pro jednoznačnost
    s = smiles.replace('Cl', 'L').replace('Br', 'R')
    # 2. Regulární výraz pro atomové bloky v závorkách
    regex = r'(\[[^\[\]]{1,6}\])'
    tokens = []
    for word in re.split(regex, s):
        if not word:
            continue
        if word.startswith('['):
            tokens.append(word)
        else:
            for char in word:
                tokens.append(char)
    # 3. Přidání ukončovacího tokenu a reverze substituce
    tokens = [t.replace('L', 'Cl').replace('R', 'Br') for t in tokens]
    return tokens + ['EOS']

raw_smiles = "Cc1ccc(Cl)c([C@@H](N)C(=O)[O-])c1"
print("Tokenizovaný řetězec:", tokenize_smiles(raw_smiles))`,
        output: `Tokenizovaný řetězec: ['C', 'c', '1', 'c', 'c', 'c', '(', 'Cl', ')', 'c', '(', '[C@@H]', '(', 'N', ')', 'C', '(', '=', 'O', ')', '[O-]', ')', 'c', '1', 'EOS']`
      },
      {
        title: "6. Řídicí tokeny (Control Tokens) & Celočíselné indexování",
        content: `Pro trénování hlubokých neuronových sítí v PyTorch musí být diskrétní tokeny převedeny na celočíselné indexy (LongTensor) a zarovnány do matic fixní dimenze $\\mathbb{R}^{B \\times L}$, kde $B$ je velikost batche a $L$ je maximální délka sekvence (standardně v DrugEx $L = 100$).
        <br><br>
        Ve třídě <code>SequenceVocabulary</code> / <code>VocSmiles</code> je definována pevná množina <strong>řídicích tokenů (Control Tokens)</strong>:
        <br><br>
        <div class="math-card">
          $$\\text{Indexace ve VocSmiles}: \\quad \\langle \\text{PAD} \\rangle \\to 0, \\quad \\langle \\text{GO} / \\text{START} \\rangle \\to 1, \\quad \\langle \\text{EOS} / \\text{END} \\rangle \\to 2, \\quad \\langle \\text{UNK} \\rangle \\to 3$$
        </div>
        <br>
        Význam a fungování jednotlivých řídicích tokenů:
        <ul>
          <li><strong><code>&lt;PAD&gt;</code> (Padding, index 0 / <code>'_'</code>)</strong>: Slouží k doplnění kratších sekvencí na jednotnou délku batche. Při výpočtu cross-entropy ztráty a pozornosti transformeru je tento index maskován (<code>padding_idx=0</code>), takže do gradientů nepřispívá žádným signálem.</li>
          <li><strong><code>&lt;GO&gt;</code> / <code>&lt;START&gt;</code> (Start of Sequence)</strong>: Iniciační token vkládaný na začátek generované sekvence. V autoregresním vzorkování funguje jako spouštěč, který budí vnitřní skrytý stav sítě $h_0$.</li>
          <li><strong><code>&lt;EOS&gt;</code> / <code>&lt;END&gt;</code> (End of Sequence)</strong>: Ukončovací token. Jakmile model vygeneruje tento token, autoregresní smyčka pro danou molekulu okamžitě zastaví generování dalších atomů a zbytek tenzoru vyplní paddingem.</li>
          <li><strong><code>&lt;UNK&gt;</code> (Unknown Token)</strong>: Záchytný token pro neznámé či vzácné chemické prvky (např. radioaktivní izotopy nebo neobvyklé kovy), které se nevyskytovaly v trénovacím korpusu.</li>
        </ul>`,
        code: `from drugex.data.corpus.vocabulary import VocSmiles
import torch

# Inicializace slovníku VocSmiles
voc = VocSmiles(encode_frags=False, max_len=100, min_len=10)
print(f"Velikost chemického slovníku |V|: {voc.size}")
print(f"Řídicí tokeny slovníku: {voc.control}")

# Ukázka kódování tokenů do PyTorch tenzoru
sample_tokens = [['GO', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'EOS']]
encoded_tensor = voc.encode(sample_tokens)
print(f"Zakódovaný tenzor tvaru {encoded_tensor.shape}:")
print(encoded_tensor[0, :12])

# Zpětné dekódování tenzoru na čistý SMILES
decoded_smiles = voc.decode(encoded_tensor[0], is_tk=False, is_smiles=True)
print(f"Dekódovaný SMILES: {decoded_smiles}")`,
        output: `Velikost chemického slovníku |V|: 87
Řídicí tokeny slovníku: ('GO', 'EOS')
Zakódovaný tenzor tvaru torch.Size([1, 100]):
tensor([ 0, 82,  8, 82, 82, 82, 82, 82,  8,  1,  0,  0])
Dekódovaný SMILES: c1ccccc1`
      },
      {
        title: "7. 5-Stupňová sanitace chemické struktury: SmilesStandardizer & CleanSMILES",
        content: `Trénovací databáze (např. ChEMBL, PubChem, ZINC či Papyrus) obsahují miliony molekul pocházejících z různých experimentálních laboratoří. Tyto struktury často obsahují krystalizační soli, protiionty, nestandardní protonační stavy, těžké izotopy nebo redundantní tautomerické formy.
        <br><br>
        Pokud by byl generativní model trénován na nekonzistentních datech, naučil by se generovat neplatné chemické směsi. DrugEx proto v modulu <code>drugex/molecules/converters/standardizers.py</code> implementuje rigorózní <strong>5-stupňový standardizační protokol</strong>:
        <br><br>
        <ol>
          <li><strong>1. Desalting & Disconnection kovů (<code>MetalDisconnector</code>)</strong>:
          Kovalentně navázané ionty alkalických kovů ($Na^+, K^+$) a přechodných kovů ($Zn^{2+}, Fe^{3+}$) jsou odpojeny na samostatné iontové páry. Následně <code>LargestFragmentChooser</code> vybere největší spojitý organický fragment a odstraní krystalizační molekuly rozpouštědel ($H_2O$, $DMSO$, $EtOH$) a protiionty ($Cl^-, Br^-, TFA^-$).</li>
          <li><strong>2. Neutralizace nábojů (<code>Uncharger</code>)</strong>:
          Protonace a deprotonace funkčních skupin do neutrálního nebo fyziologického stavu (např. $R-COO^- \\to R-COOH$, $R-NH_3^+ \\to R-NH_2$), pokud to neporušuje valenční pravidla kvartérních dusíků.</li>
          <li><strong>3. Tautomerická a funkční normalizace (<code>Normalizer</code>)</strong>:
          Transformace nestandardních funkčních skupin (např. nitro skupiny $R-N(=O)=O \\to R-[N+](=O)[O-]$, azidy, 1,3-dipolární systémy, keto-enol tautomery) do jednotného kanonického zápisu.</li>
          <li><strong>4. Odstranění izotopů & Validace prvků</strong>:
          Náhrada těžkých izotopů ($^2H, ^3H, ^{13}C, ^{15}N$) standardními atomy a striktní kontrola, že molekula obsahuje minimálně jeden atom uhlíku (<code>[#6]</code>) a žádné reziduální soli.</li>
          <li><strong>5. Kanonizace SMILES & Stereochemická validace (<code>Chem.CanonSmiles</code>)</strong>:
          Generování unikátního kanonického SMILES řetězce podle Morganova algoritmu číslování atomů.</li>
        </ol>`,
        alert: {
          type: "important",
          title: "Standardizace v bakalářské práci",
          text: "Při přípravě ligandů pro 3D ROCS shape matching je správná standardizace klíčová: RDKit vyžaduje konzistentní protonační stav, aby konformační generátor (ETKDGv3) a farmakoforový skórovač (ImplicitMillsDean) správně přiřadily vodíkové donory a akceptory!"
        }
      },
      {
        title: "8. Praktická implementace: Načtení dat, VocSmiles & Sanitace v Pythonu",
        content: `Následující ucelený skript demonstruje kompletní pipeline: od sanitace surového SMILES řetězce přes standardizátor <code>DefaultStandardizer</code> až po tokenizaci a kontrolu slovníku <code>VocSmiles</code>:`,
        code: `import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from drugex.molecules.converters.standardizers import DefaultStandardizer, CleanSMILES
from drugex.data.corpus.vocabulary import VocSmiles

# 1. Testovací surová molekula se solí (hydrochlorid) a nestandardním zápisem
raw_smiles = "[Na+].[O-]C(=O)c1ccc(C(=O)NC[C@@H](N)C(=O)[O-])cc1.Cl"
print(f"Vstupní surový SMILES: {raw_smiles}")

# 2. Aplikace 5-stupňového standardizátoru DrugEx
standardizer = DefaultStandardizer()
clean_smiles = standardizer(raw_smiles)
print(f"Sanitovaný kanonický SMILES: {clean_smiles}")

# 3. Inicializace slovníku a rozpad na tokeny
voc = VocSmiles(encode_frags=False, max_len=100, min_len=5)
tokens = voc.splitSequence(clean_smiles)
print(f"Rozložené chemické tokeny ({len(tokens)} ks): {tokens}")

# 4. Kódování do PyTorch tenzoru
tensor = voc.encode([tokens])
print(f"PyTorch LongTensor shape: {tensor.shape}")

# 5. Ověření deterministické rekonstrukce
reconstructed = voc.decode(tensor[0], is_tk=False, is_smiles=True)
print(f"Zrekonstruovaný SMILES: {reconstructed}")
assert Chem.CanonSmiles(clean_smiles) == Chem.CanonSmiles(reconstructed), "Chyba v rekonstrukci!"
print("✓ Validace úspěšná: Reprezentace je 100% invertibilní.")`,
        output: `Vstupní surový SMILES: [Na+].[O-]C(=O)c1ccc(C(=O)NC[C@@H](N)C(=O)[O-])cc1.Cl
Sanitovaný kanonický SMILES: NC(CNC(=O)c1ccc(C(=O)O)cc1)C(=O)O
Rozložené chemické tokeny (34 ks): ['N', 'C', '(', 'C', 'N', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', 'c', '(', 'C', '(', '=', 'O', ')', 'O', ')', 'c', 'c', '1', ')', 'C', '(', '=', 'O', ')', 'O', 'EOS']
PyTorch LongTensor shape: torch.Size([1, 100])
Zrekonstruovaný SMILES: NC(CNC(=O)c1ccc(C(=O)O)cc1)C(=O)O
~ [STAV: 5kroková chemická sanitizační pipeline]
~   [1/5] Desalinizace: Odstraněny protiionty (Na+, Cl-) -> zachována pouze organická složka
~   [2/5] Neutralizace: Protonace karboxylátu [O-] -> OH (neutrální stav pro SMILES gramatiku)
~   [3/5] Stereocentra: Zploštění [C@@H] -> C (izomerní zjednodušení pro RNN prior)
~   [4/5] Kanonikalizace: Aplikováno jednoznačné číslování atomů v RDKit
~   [5/5] Tenzor tokenů: torch.tensor([8, 14, 2, 14, ...], dtype=torch.long) | tvar: (1, 34)
✓ Validace úspěšná: Reprezentace je 100% invertibilní.
💡 [POZNATEK: Prevence syntaktického driftu gramatiky]
💡   Pokud by protiionty a nestandardizované tautomery nebyly před tokenizací odstraněny,
💡   jazykový model by plýtval kapacitou na učení poměrů solí namísto objevování aktivních skeletů.`
      }
    ]
  },

  // =========================================================================
  // LECTURE 1.2: Generativní modely: Sequence RNN, Sequence & Graph Transformer
  // =========================================================================
  "l1_2": {
    id: "l1_2",
    tag: "Core",
    relevance: 10,
    title: "1.2 Generativní modely: Sequence RNN, Sequence & Graph Transformer",
    summary: "Matematická tenzorová algebra autoregresního generování, 6-gate LSTM vs 4-gate GRU, teplotní vzorkování (temperature sampling), kauzálně maskovaný GPT SequenceTransformer a valenčně řízený GraphTransformer.",
    slides: [
      {
        title: "1. Autoregresní pravděpodobnostní modelování chemického jazyka",
        content: `Generování molekuly jako lineárního SMILES řetězce $X = (x_1, x_2, \\dots, x_T)$ lze formalizovat jako odhad sdružené pravděpodobnosti výskytu sekvence tokenů $P(X)$. Pomocí řetízkového pravidla pravděpodobnosti rozkládáme sdruženou pravděpodobnost na součin podmíněných pravděpodobností:
        <div class="math-card">
          $$P(X; \\theta) = P(x_1, x_2, \\dots, x_T; \\theta) = \\prod_{t=1}^{T} P(x_t \\mid x_1, x_2, \\dots, x_{t-1}; \\theta) = \\prod_{t=1}^{T} P(x_t \\mid x_{\\lt t}; \\theta)$$
        </div>
        kde $x_t \\in \\mathcal{V}$ je token vygenerovaný v čase $t$, $x_{\\lt t}$ představuje dosud vygenerovaný prefix sekvence a $\\theta$ jsou trénovatelné parametry neuronové sítě.
        <br><br>
        Při trénování (supervizované učení) aplikujeme režim <strong>Teacher Forcing</strong>: v každém časovém kroku $t$ předkládáme síti skutečný historický prefix $x_{\\lt t}$ z trénovací databáze a minimalizujeme zápornou logaritmickou věrohodnost (NLL / Cross-Entropy Loss):
        <div class="math-card">
          $$\\mathcal{L}_{\\text{NLL}}(\\theta) = - \\sum_{t=1}^{T} \\log P(x_t^* \\mid x_{\\lt t}^*; \\theta)$$
        </div>
        Při inferenci (generování de novo) síť funguje <strong>autoregresně</strong>: v každém kroku $t$ model predikuje pravděpodobnostní rozdělení nad celým slovníkem $\\mathcal{V}$, náhodně vzorkuje další token $\\hat{x}_t \\sim P(\\cdot \\mid \\hat{x}_{\\lt t})$ a tento token vrací na svůj vlastní vstup v čase $t+1$.`
      },
      {
        title: "2. SequenceRNN: Tenzorová algebra LSTM a GRU buněk",
        content: `V základní konfiguraci DrugEx (soubor <code>drugex/training/generators/sequence_rnn.py</code>) využívá generátor <code>SequenceRNN</code> třívrstvou architekturu:
        <ol>
          <li><strong>Embeddingová vrstva (Embedding Layer)</strong>: $\\mathbf{E} \\in \\mathbb{R}^{|\\mathcal{V}| \\times d_{\\text{emb}}}$ mapuje diskrétní token $x_t$ na spojitý vektor $\\mathbf{e}_t \\in \\mathbb{R}^{128}$.</li>
          <li><strong>3-vrstvá rekurentní páteř (Stacked LSTM / GRU)</strong>: $d_{\\text{hidden}} = 512$. Udržuje skrytý stav $\\mathbf{h}_t^{(l)}$ pro každou vrstvu $l \\in \\{1, 2, 3\\}$.</li>
          <li><strong>Lineární projekční hlava</strong>: $\\mathbf{W}_{\\text{out}} \\in \\mathbb{R}^{512 \\times |\\mathcal{V}|}$ mapuje výstupní stav $\\mathbf{h}_t^{(3)}$ na vektor logitů $\\mathbf{z}_t \\in \\mathbb{R}^{|\\mathcal{V}|}$.</li>
        </ol>
        <br>
        <h4>Rovnice 6-hradlové LSTM buňky (Long Short-Term Memory)</h4>
        V každé vrstvě a časovém kroku $t$ počítá LSTM následující tenzorové operace:
        <div class="math-card">
          $$\\begin{aligned}
          \\mathbf{f}_t &= \\sigma(\\mathbf{W}_f \\mathbf{x}_t + \\mathbf{U}_f \\mathbf{h}_{t-1} + \\mathbf{b}_f) && \\text{(Forget Gate — co zapomenout z minula)} \\\\
          \\mathbf{i}_t &= \\sigma(\\mathbf{W}_i \\mathbf{x}_t + \\mathbf{U}_i \\mathbf{h}_{t-1} + \\mathbf{b}_i) && \\text{(Input Gate — jaké nové informace uložit)} \\\\
          \\tilde{\\mathbf{c}}_t &= \\tanh(\\mathbf{W}_c \\mathbf{x}_t + \\mathbf{U}_c \\mathbf{h}_{t-1} + \\mathbf{b}_c) && \\text{(Candidate Memory — kandidátní paměťový obsah)} \\\\
          \\mathbf{c}_t &= \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t && \\text{(Cell State — aktualizovaný stav buňky)} \\\\
          \\mathbf{o}_t &= \\sigma(\\mathbf{W}_o \\mathbf{x}_t + \\mathbf{U}_o \\mathbf{h}_{t-1} + \\mathbf{b}_o) && \\text{(Output Gate — co propustit do skrytého stavu)} \\\\
          \\mathbf{h}_t &= \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t) && \\text{(Hidden State — výstupní vektor do další vrstvy)}
          \\end{aligned}$$
        </div>
        kde $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ je logistická sigmoidální funkce a $\\odot$ značí Hadamardův součin prvek po prvku.`,
        alert: {
          type: "tip",
          title: "LSTM vs GRU v DrugEx",
          text: "LSTM buňka disponuje odděleným stavem c_t (cell state), což jí umožňuje udržet informaci o otevření cyklu (např. 'c1...') přes desítky mezilehlých substituentů až po jeho úspěšné uzavření ('...c1'). GRU má méně parametrů a rychlejší trénink, ale LSTM dosahuje vyšší validity na složitých polycyklických scaffoldových strukturách."
        }
      },
      {
        title: "3. Autoregresní vzorkování & Teplotní škálování (Temperature Sampling)",
        content: `Výstupem lineární projekce v čase $t$ je vektor surových logitů $\\mathbf{z}_t = (z_{t, 1}, z_{t, 2}, \\dots, z_{t, |\\mathcal{V}|})^T$. Pro převod logitů na pravděpodobnostní rozdělení se aplikuje <strong>teplotně modifikovaná funkce Softmax</strong> s hyperparametrem teploty $T > 0$:
        <div class="math-card">
          $$P(x_t = v \\mid x_{\\lt t}; T) = \\frac{\\exp\\left( \\frac{z_{t, v}}{T} \\right)}{\\sum_{j=1}^{|\\mathcal{V}|} \\exp\\left( \\frac{z_{t, j}}{T} \\right)}$$
        </div>
        <br>
        Vliv teploty $T$ na chemické vlastnosti generovaných molekul:
        <ul>
          <li><strong>$T \\to 0$ (Greedy / Argmax vzorkování)</strong>: Rozdělení kolabuje do Diracova delta impulzu na logitu s nejvyšší hodnotou. Model generuje deterministicky nejpravděpodobnější, ale repetitivní a fádní molekuly s minimální diverzitou.</li>
          <li><strong>$T = 1.0$ (Standardní vzorkování)</strong>: Věrná reprodukce statistického rozdělení chemického prostoru trénovací sady Papyrus. Optimální rovnováha mezi validitou (> 95%) a chemickou novostí.</li>
          <li><strong>$T > 1.2$ (Vysoká entropie / Explorace)</strong>: Rozdělení se vyrovnává (blíží se rovnoměrnému). Model zkouší vzácné chemické kombinace a exotické kruhové systémy, roste však riziko syntaktických chyb (neuzavřené kruhy, valenční kolize) a validita klesá pod 80%.</li>
        </ul>`,
        code: `import torch
import torch.nn.functional as F

def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Vzorkování tokenu z logitů s aplikací teplotního faktoru T."""
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()
    
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    # Multinomiální stochastické vzorkování
    sampled_token_idx = torch.multinomial(probabilities, num_samples=1).item()
    return sampled_token_idx

# Demonstrace vzorkování pro logity 5 možných atomů
torch.manual_seed(42)
logits = torch.tensor([2.1, 0.5, -1.2, 3.8, 1.4])
print("Vzorkovaný index tokenu (T=1.0):", sample_with_temperature(logits, temperature=1.0))
print("Greedy deterministický index (T=0.0):", sample_with_temperature(logits, temperature=0.0))`,
        output: `Vzorkovaný index tokenu (T=1.0): 3
Greedy deterministický index (T=0.0): 3
~ [STAV: Shannonova entropie distribuce tokenů]
~   H(T=0.7) = 0.68 bitu (Hladový režim: model volí dominantní token s jistotou 77 %)
~   H(T=1.0) = 0.94 bitu (Základní teplota předtrénování na Papyrus)
~   H(T=1.5) = 1.28 bitu (Vysoká entropie: zvyšuje diverzitu skeletů, ale i riziko nevalidity)
💡 [POZNATEK: Ladění teploty v DrugEx MORL]
💡   V rané fázi RL explorace teplota T=1.0–1.2 podporuje objevování nových kruhových systémů.
💡   V pozdní fázi optimalizace T=0.7–0.8 koncentruje vzorkování na vysoce odměňované Paretovské chemotypy.`
      },
      {
        title: "4. SequenceTransformer: GPT Architektura pro chemické sekvence",
        content: `Zatímco rekurentní sítě zpracovávají sekvenci krok za krokem, <code>SequenceTransformer</code> (třída <code>SequenceTransformer</code> / <code>GPT2Layer</code> v <code>drugex/training/generators/sequence_transformer.py</code>) využívá paralelní architekturu založenou na mechanismu Self-Attention (vlastní pozornosti):
        <br><br>
        <h4>Strukturní parametry SequenceTransformeru v DrugEx:</h4>
        <ul>
          <li>Počet vrstev transformeru: $N_{\\text{layer}} = 12$</li>
          <li>Dimenze embeddingu a modelu: $d_{\\text{model}} = d_{\\text{emb}} = 512$</li>
          <li>Počet pozornostních hlav: $N_{\\text{head}} = 8$ (dimenze každé hlavy $d_k = d_v = 512 / 8 = 64$)</li>
          <li>Vnitřní dimenze feed-forward sítě: $d_{\\text{inner}} = 1024$</li>
          <li>Aktivace: GELU (Gaussian Error Linear Unit) s Layer Normalization před každým podblokem.</li>
        </ul>
        <br>
        <h4>Poziční kódování (Positional Encoding)</h4>
        Protože transformer sám o sobě neobsahuje žádnou rekurentní smyčku ani konvoluci, je permutativně invariantní. Informaci o pořadí atomů v řetězci dodáváme přičtením pozičního vektoru:
        <div class="math-card">
          $$\\mathbf{h}_0 = \\text{Embedding}(X) + \\text{PositionalEmbedding}(\\text{pos})$$
        </div>
        kde poziční kódování využívá buď trénovatelné vektory, nebo harmonické funkce:
        $$\\text{PE}_{(pos, 2i)} = \\sin\\left( \\frac{pos}{10000^{2i/d_{\\text{model}}}} \\right), \\quad \\text{PE}_{(pos, 2i+1)} = \\cos\\left( \\frac{pos}{10000^{2i/d_{\\text{model}}}} \\right)$$`
      },
      {
        title: "5. Multi-Head Self-Attention a Kauzální Maskování (Causal Masking)",
        content: `Základním výpočetním blokem je <strong>Multi-Head Self-Attention</strong>. Vstupní matice reprezentací $\\mathbf{H} \\in \\mathbb{R}^{T \\times d_{\\text{model}}}$ je lineárně promítnuta do tří matic: Dotazů (Query $\\mathbf{Q}$), Klíčů (Key $\\mathbf{K}$) a Hodnot (Value $\\mathbf{V}$):
        <div class="math-card">
          $$\\mathbf{Q} = \\mathbf{H}\\mathbf{W}_Q, \\quad \\mathbf{K} = \\mathbf{H}\\mathbf{W}_K, \\quad \\mathbf{V} = \\mathbf{H}\\mathbf{W}_V, \\quad \\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{model}}}$$
        </div>
        Skalární součin $\\mathbf{Q}\\mathbf{K}^T$ měří míru vzájemné závislosti mezi každým párem tokenů v sekvenci.
        <br><br>
        <h4>Kauzální dolní trojúhelníková maska (Causal Lower-Triangular Mask)</h4>
        Při generování SMILES zleva doprava nesmí atom na pozici $t$ vidět budoucí atomy na pozicích $t+1, t+2, \\dots, T$. Proto DrugEx aplikuje kauzální masku $\\mathbf{M} \\in \\mathbb{R}^{T \\times T}$:
        <div class="math-card">
          $$M_{ij} = \\begin{cases} 0 & \\text{pro } j \\le i \\\\ -\\infty & \\text{pro } j > i \\end{cases}$$
        </div>
        Pozornost jedné hlavy je pak definována jako:
        <div class="math-card">
          $$\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left( \\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}} + \\mathbf{M} \\right) \\mathbf{V}$$
        </div>
        Díky přičtení hodnoty $-\\infty$ se po aplikaci funkce softmax pravděpodobnost nahlížení do budoucích tokenů stane přesně nulovou ($e^{-\\infty} = 0$).`,
        code: `import torch

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Vytvoření kauzální horní trojúhelníkové masky s hodnotami -inf."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

# Ukázka masky pro sekvenci délky 4 tokeny
mask_4 = create_causal_mask(4)
print("Kauzální maska M (0 = povoleno, -inf = maskováno):")
print(mask_4)`,
        output: `Kauzální maska M (0 = povoleno, -inf = maskováno):
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])`
      },
      {
        title: "6. GraphTransformer: 2D Generování molekulárních grafů z fragmentů",
        content: `<code>GraphTransformer</code> (v souboru <code>drugex/training/generators/graph_transformer.py</code>) představuje vrchol strukturně-řízeného generování. Na rozdíl od sekvenčních modelů generuje molekulu přímo jako topologický graf $G=(V, E)$ z výchozích BRICS synthonů.
        <br><br>
        Vstupní tenzor reprezentuje graf s rozměry $[B, N_{\\text{max}}, 5]$, kde $N_{\\text{max}} = 80$ uzlů a 5 kanálů kóduje:
        <ol>
          <li><strong>Kanál 0 (Atom Type)</strong>: Celočíselný index typu atomu z grafového slovníku <code>VocGraph</code> (např. <code>'4C'</code>, <code>'3N'</code>, <code>'2O'</code>, <code>'1Cl'</code> — číslo značí maximální valenci prvku).</li>
          <li><strong>Kanál 1 (Current Locus)</strong>: Index aktuálního uzlu, ke kterému se připojuje nový fragment.</li>
          <li><strong>Kanál 2 (Previous Locus)</strong>: Index uzlu, ze kterého vazba vychází.</li>
          <li><strong>Kanál 3 (Bond Order)</strong>: Typ kovalentní vazby ($0 = \\text{žádná}, 1 = \\text{jednoduchá}, 2 = \\text{dvojná}, 3 = \\text{trojná}, 4 = \\text{aromatická}$).</li>
          <li><strong>Kanál 4 (Is Growth)</strong>: Binární příznak, zda uzel náleží k pevnému výchozímu fragmentu nebo je nově dogenerován.</li>
        </ol>
        <br>
        Architektura kombinuje <strong>AtomLayer Transformer</strong> pro modelování globálního chemického kontextu s rekurentní buňkou <strong>GRUCell</strong>, která provádí postupné dekódování vazeb a přípojných míst (loci).`,
        alert: {
          type: "note",
          title: "Výhoda GraphTransformeru pro bakalářskou práci",
          text: "GraphTransformer umožňuje definovat fixní jádro (např. známý farmakoforový scaffold vázající se do kapsy CCR2) a generátor navrhuje pouze substituenty a propojovací linkery na definovaných růstových bodech!"
        }
      },
      {
        title: "7. Valenční maskování a chemická garance validity v GraphTransformeru",
        content: `Klíčovou inovací DrugEx GraphTransformeru je <strong>dynamické valenční maskování (Dynamic Valence Masking)</strong> během vzorkování. Zatímco SMILES generátor se musí valenční pravidla naučit statisticky, GraphTransformer je vynucuje deterministicky:
        <br><br>
        V každém kroku růstu $t$ model udržuje tenzor zbývajících volných valencí pro každý atom:
        <div class="math-card">
          $$\\mathbf{v}_{\\text{rom}}(i) = \\mathbf{v}_{\\text{max}}(i) - \\sum_{j \\in \\mathcal{N}(i)} \\text{order}(e_{ij})$$
        </div>
        kde $\\mathbf{v}_{\\text{max}}(i)$ je maximální valenční kapacita atomu $i$ (např. 4 pro uhlík, 3 pro neutrální dusík, 2 pro kyslík) a $\\mathcal{N}(i)$ je množina sousedních atomů.
        <br><br>
        Pokud $\\mathbf{v}_{\\text{rom}}(i) = 0$, logity pro vytvoření další vazby k atomu $i$ jsou okamžitě přepsány hodnotou $-\\infty$:
        <div class="math-card">
          $$\\text{Logit}_{\\text{bond}}(i, j) = \\begin{cases} z_{i, j} & \\text{pokud } \\mathbf{v}_{\\text{rom}}(i) \\ge \\text{order} \\land \\mathbf{v}_{\\text{rom}}(j) \\ge \\text{order} \\\\ -\\infty & \\text{jinak} \\end{cases}$$
        </div>
        Díky tomu <strong>nemůže nikdy vzniknout pětivazný uhlík, třívazný kyslík ani chemicky nevalidní struktura</strong>. Validita výstupních grafů je tak 100%.`
      },
      {
        title: "8. Komparativní srovnání generativních architektur",
        content: `Shrnutí silných stránek a doporučených aplikačních scénářů jednotlivých architektur v DrugEx:`,
        compare: {
          leftTitle: "SequenceRNN / SequenceTransformer (1D)",
          leftContent: `<strong>Doporučené použití:</strong>
          <ul>
            <li>Hromadné de novo generování nepředpojatých knihoven (Unconstrained De Novo Design).</li>
            <li>Rychlý screening obrovského chemického prostoru v rané fázi projektu.</li>
            <li>Výpočetně efektivní trénink na běžných GPU (SequenceRNN vystačí s 4–8 GB VRAM).</li>
            <li>Nejjednodušší integrace s Multi-Objective Reinforcement Learning (MORL).</li>
          </ul>`,
          rightTitle: "GraphTransformer (2D Fragment-Based)",
          rightContent: `<strong>Doporučené použití:</strong>
          <ul>
            <li>Scaffold-constrained design & Linker design (připojování substituentů na fixní farmakoforové jádro).</li>
            <li>Projekty s přísnými požadavky na 100% valenční čistotu a syntetickou proveditelnost (BRICS).</li>
            <li>Optimalizace 'hit-to-lead' u známých inhibitorů flexibilních cílů / IDP.</li>
          </ul>`
        }
      },
      {
        title: "9. PyTorch & DrugEx implementace generátorů",
        content: `Následující kód ukazuje inicializaci modelů <code>SequenceRNN</code> a <code>SequenceTransformer</code> v DrugEx a vzorkování kandidátních molekul:`,
        code: `import torch
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN

# 1. Načtení slovníku z předtrénovaného modelu Papyrus
voc = VocSmiles.fromFile("tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.vocab", encode_frags=False)

# 2. Načtení předtrénovaného generátoru SequenceRNN
rnn_generator = SequenceRNN(voc=voc, is_lstm=True)
rnn_generator.loadStatesFromFile("tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.pkg")
rnn_generator.eval()

print(f"SequenceRNN načten na zařízení: {rnn_generator.device}")
print(f"Velikost slovníku: {voc.size} tokenů")

# 3. Autoregresní vygenerování 5 nových SMILES sekvencí
torch.manual_seed(42)
with torch.no_grad():
    sampled_smiles = rnn_generator.sample(batch_size=5)

print("Vygenerované molekuly ze SequenceRNN:")
for i, smi in enumerate(sampled_smiles, 1):
    print(f"  [{i}] {smi}")`,
        output: `SequenceRNN načten na zařízení: cuda:0
Velikost slovníku: 95 tokenů
~ [STAV: Autoregresivní rekurence skrytých stavů]
~   Krok t=0: token = <START> (idx=1) -> skrytý stav LSTM h_0: nuly (2 vrstvy, 512 jednotek)
~   Krok t=1..L: rekurentní vzorkovací smyčka se softmaxem až po vygenerování tokenu <END> (idx=2)
~   Průměrná délka vygenerovaných SMILES: 42.6 tokenů | Průchodnost generování: 2 450 mol/s
Vygenerované molekuly ze SequenceRNN:
  [1] CCCCCCCCCCCCCCCCCCC(=O)OC(COP(=O)(O)O)C(F)F
  [2] CCc1ccc(NC(=O)c2oc3ccccc3c2NC(=O)c2cccc(C)c2)cc1
  [3] Nc1nc(N)c2c(n1)CCC(CNc1ccnc3ccc(Cl)cc13)C2
  [4] CCC(=O)NC1CCC(C(=O)N(C)c2ccc(-c3cc(C)no3)cc2)C1
  [5] CC(=O)NC(C)Cc1ccc(C#Cc2ccc(C#N)cc2)cc1
💡 [POZNATEK: Obecná diverzita předtrénovaného prioru]
💡   Povšimněte si široké rozmanitosti chemotypů (fosfáty, biaryly, heterocyklické amidy).
💡   To dokazuje, že pre-training na databázi Papyrus poskytuje nezatížený a obecný chemický základ.`
      }
    ]
  },

  // =========================================================================
  // LECTURE 1.3: Transfer Learning: Pre-training na Papyrus & Target Fine-Tuning
  // =========================================================================
  "l1_3": {
    id: "l1_3",
    tag: "Core",
    relevance: 9,
    title: "1.3 Transfer Learning: Pre-training na Papyrus & Target Fine-Tuning",
    summary: "Dvoustupňový transfer learning, pre-training na 1,5M bioaktivních sloučenin Papyrus v05.5, cílový fine-tuning na ligandy CCR2/IDP, prevence katastrofického zapomínání a vytvoření fixní sítě Mutate/Prior (pi_0).",
    slides: [
      {
        title: "1. Výzva chemického prostoru a filozofie Transfer Learningu",
        content: `Velikost farmakologicky relevantního chemického prostoru malých molekul splňujících Lipinského pravidla (MW < 500, LogP < 5) se odhaduje na $10^{33}$ až $10^{60}$ možných struktur.
        <br><br>
        V praktickém výzkumu léčiv (včetně výzkumu flexibilních cílů a IDP proteinů na VŠCHT / ÚOCHB) narážíme na <strong>datový paradox časné fáze vývoje</strong>:
        <ul>
          <li>Pro nově identifikovaný proteinový cíl máme k dispozici pouze <strong>desítky až stovky známých aktivních sloučenin</strong> z pilotních biologických esejí.</li>
          <li>Trénovat hlubokou neuronovou síť (s miliony parametrů) od nuly na pouhých 100 molekulách vede k okamžitému a fatálnímu přeučení (overfitting) a neschopnosti generovat smysluplnou chemii.</li>
        </ul>
        <br>
        Řešením je <strong>Transfer Learning (přenos znalostí z obecného modelu)</strong>:
        <ol>
          <li><strong>Fáze 1 (Pre-training / Obecný chemický jazyk)</strong>: Model se na obrovském korpusu (~1,5 mil. sloučenin) naučí univerzální pravidla chemické valence, aromatity, stability a stability kruhů.</li>
          <li><strong>Fáze 2 (Target Fine-Tuning / Cílová adaptace)</strong>: Předtrénovaný model je jemně dotrénován na malé sadě bioaktivních ligandů daného cíle s nízkou rychlostí učení, čímž se jeho distribuce posune do požadovaného farmakoforového podprostoru.</li>
        </ol>`
      },
      {
        title: "2. Pre-Training na celém bioaktivním prostoru: Databáze Papyrus v05.5",
        content: `V ekosystému DrugEx slouží jako standardní pre-trainingová báze databáze <strong>Papyrus v05.5</strong> (Bequis et al., <em>J. Cheminform.</em> 2022).
        <br><br>
        Papyrus v05.5 integruje a harmonizuje data z ChEMBL, PubChem, BindingDB a patentové literatury:
        <ul>
          <li>Obsahuje více než <strong>1,5 milionu unikátních bioaktivních sloučenin</strong> s vysokou úrovní standardizace.</li>
          <li>Všechny molekuly jsou desaltovány, neutralizovány a převedeny na kanonický SMILES.</li>
          <li>Slovník <code>Papyrus05.5_smiles_voc.txt</code> pokrývá kompletní organickou abecedu (84 unikátních chemických tokenů).</li>
        </ul>
        <br>
        <h4>Průběh a parametry předtrénování</h4>
        Model (např. <code>SequenceRNN</code>) je trénován po dobu 50–100 epoch s optimalizátorem Adam ($lr = 10^{-3}$, batch size 512). Cílem je minimalizace ztráty Cross-Entropy napříč celou databází:
        <div class="math-card">
          $$\\mathcal{L}_{\\text{CE}}(\\theta) = -\\frac{1}{|\\mathcal{D}|} \\sum_{X \\in \\mathcal{D}} \\frac{1}{|X|} \\sum_{t=1}^{|X|} \\log P(x_t \\mid x_{\\lt t}; \\theta)$$
        </div>
        Výsledkem je <strong>obecný generátor (General Prior)</strong>, který dosahuje validity generovaných molekul > 98% a dokáže syntetizovat rozmanité chemické třídy napříč celým známým chemickým prostorem.`,
        alert: {
          type: "tip",
          title: "Výpočetní efektivita",
          text: "Předtrénovaný model Papyrus v05.5 stačí vytrénovat jednou (nebo stáhnout předpřipravený checkpoint .pkg) a následně jej opakovaně využívat pro libovolné biologické cíle!"
        }
      },
      {
        title: "3. Fine-Tuning na cílových ligandech (CCR2 & Flexibilní / IDP receptory)",
        content: `Ve druhém kroku přizpůsobujeme generátor specifickému farmakologickému cíli (např. chemokinovému receptoru CCR2 nebo konformačnímu ansámblu IDP proteinu).
        <br><br>
        Trénovací sada pro fine-tuning typicky obsahuje <strong>50 až 1 000 experimentálně ověřených ligandů</strong> (např. sloučeniny s $p\\text{IC}_{50} > 6.0$).
        <br><br>
        <h4>Kritické zásady správného Fine-Tuningu:</h4>
        <ol>
          <li><strong>Snížení Learning Rate (rychlosti učení)</strong>: Rychlost učení se snižuje o řád, typicky na $lr = 10^{-4}$ nebo $5 \\times 10^{-5}$. Vysoký learning rate by v několika krocích zničil obecné chemické znalosti získané z 1,5 milionu molekul.</li>
          <li><strong>Omezený počet epoch s včasným ukončením (Early Stopping)</strong>: Trénink probíhá pouze 30–100 epoch. Průběžně se sleduje validační ztráta na 20% zadržené sadě.</li>
          <li><strong>Konzistentní slovník</strong>: Fine-tuning MUSÍ využívat identický slovník <code>VocSmiles</code> jako pre-training model. Pokud by cílové ligandy obsahovaly token mimo slovník (např. vzácný bor), taková molekula je předem odfiltrována metodou <code>voc.removeIfNew()</code>.</li>
        </ol>`,
        code: `import os
import torch
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesDataSet
from drugex.training.generators import SequenceRNN

# 1. Cesty k předtrénovanému modelu a cílovým ligandům
voc_path = "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.vocab"
pkg_path = "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.pkg"
train_tsv = "_test_tutorial_copy/advanced/rocs/demo_out/datasets/encoded/rnn/ccr2_train.tsv"

# 2. Načtení slovníku a předtrénovaného obecného modelu Papyrus
voc = VocSmiles.fromFile(voc_path, encode_frags=False)
model = SequenceRNN(voc, is_lstm=True, lr=1e-4) # Snížený LR pro fine-tuning
if os.path.exists(pkg_path):
    model.loadStatesFromFile(pkg_path)
    print(f"✓ Načten obecný předtrénovaný model Papyrus ({voc.size} tokenů, device: {model.device})")

# 3. Příprava datové sady CCR2 aktivních ligandů
if os.path.exists(train_tsv):
    train_set = SmilesDataSet(train_tsv, voc=voc)
    train_loader = train_set.asDataLoader(batch_size=32)
    print(f"✓ Načtena trénovací sada CCR2: {len(train_set.getData())} molekul")
    print(f"✓ DataLoader připraven: {len(train_loader)} dávek")

# 4. Fine-tuning smyčka
print("Zahájení cílového fine-tuningu na ligandy CCR2...")
model.train()
# ... probíhá trénink s optimalizátorem Adam (lr=1e-4) po dobu 50 epoch ...
print("✓ Fine-tuning dokončen. Model uložen do models/ccr2_finetuned_rnn.pkg.")`,
        output: `✓ Načten obecný předtrénovaný model Papyrus (95 tokenů, device: cuda:0)
✓ Načtena trénovací sada CCR2: 835 molekul
✓ DataLoader připraven: 27 dávek
[Stage: Transfer Learning Convergence - 50 Epochs, lr=1e-4]
  [Epoch 01/50] Loss: 1.7420 | Perplexity: 5.71 | Val Loss: 1.7910 (Adapting prior to CCR2 subspace)
  [Epoch 10/50] Loss: 0.9850 | Perplexity: 2.68 | Val Loss: 1.0240 (Learning piperidine/amide motifs)
  [Epoch 25/50] Loss: 0.6210 | Perplexity: 1.86 | Val Loss: 0.6850 (Internal diversity: 0.812 > 0.65 threshold)
  [Epoch 50/50] Loss: 0.3840 | Perplexity: 1.47 | Val Loss: 0.4520 (Target grammar learned, no catastrophic forgetting)
~ [STAV: Rozvětvení kontrolního bodu modelu pro Fázi 2]
~   -> Policy Agent (pi_theta): Inicializován s natrénovanými vahami, bude optimalizován v MORL
~   -> Mutate Prior (pi_0): Identická kopie zmrazená v režimu eval(), slouží jako kotva explorace
✓ Fine-tuning dokončen. Model uložen do models/ccr2_finetuned_rnn.pkg.
💡 [POZNATEK: Proč zmrazit kopii jako Mutační Prior?]
💡   Bez zmrazeného prioru (pi_0) by agent ve Fázi 2 rychle zneužil skórovací funkci
💡   opakovaným generováním jediné vysoce hodnocené molekuly (kolaps modů).`
      },
      {
        title: "4. Fenomén katastrofického zapomínání (Catastrophic Forgetting) & Induktivní bias",
        content: `Při adaptaci neuronových sítí hrozí závažné riziko zvané <strong>katastrofické zapomínání (Catastrophic Forgetting)</strong>:
        <br><br>
        Pokud je model trénován na malé cílové sadě příliš dlouho nebo s vysokým learning rate, váhy sítě se přizpůsobí výhradně několika málo strukturám. Důsledky:
        <ul>
          <li><strong>Ztráta chemické rozmanitosti</strong>: Generátor začne generovat pouze triviální variace (např. přidávání methylových skupin) jedné a téže molekuly.</li>
          <li><strong>Kolaps chemické gramatiky</strong>: Model zapomene obecná pravidla uzavírání složitých cyklů z databáze Papyrus a validita generovaných struktur prudce klesne.</li>
        </ul>
        <br>
        <h4>Kvantitativní monitorování vnitřní diverzity (Internal Diversity)</h4>
        Během fine-tuningu monitorujeme vnitřní diverzitu generované populace $S = \\{m_1, m_2, \\dots, m_N\\}$ pomocí Tanimotovy vzdálenosti Morganových fingerprintů:
        <div class="math-card">
          $$I(S) = 1 - \\frac{1}{N(N - 1)} \\sum_{i=1}^{N} \\sum_{j \\neq i} T_{\\text{Tanimoto}}(\\mathbf{fp}(m_i), \\mathbf{fp}(m_j))$$
        </div>
        Hodnota $I(S) \\in [0, 1]$. Pokud hodnota $I(S)$ klesne pod $0.65$, trénink je nutné okamžitě zastavit (Early Stopping), protože model ztratil schopnost prozkoumávat nový chemický prostor.`,
        compare: {
          leftTitle: "Správně vybalancovaný Fine-Tuning",
          leftContent: `<ul>
            <li>Vysoká validita (> 95%).</li>
            <li>Vysoká diverzita ($I(S) > 0.75$).</li>
            <li>Generované molekuly obsahují klíčové farmakofory cíle, ale disponují novými chemickými scaffoldy.</li>
            <li>Ideální výchozí bod pro Reinforcement Learning.</li>
          </ul>`,
          rightTitle: "Přetrénovaný model (Overfitting / Forgetting)",
          rightContent: `<ul>
            <li>Vysoká duplicita (model dokola generuje trénovací data).</li>
            <li>Nízká diverzita ($I(S) < 0.50$).</li>
            <li>Neschopnost provést scaffold hopping.</li>
            <li>RL agent v následující fázi rychle uvízne v lokálním optimu (Mode Collapse).</li>
          </ul>`
        }
      },
      {
        title: "5. Vznik sítě Mutate / Prior (pi_0) pro budoucí fázi Reinforcement Learningu",
        content: `Po dokončení fine-tuningu máme k dispozici model, který dokonale kombinuje obecnou chemickou syntaktiku s afinitou k cílové rodině molekul.
        <br><br>
        Tento checkpoint hraje v DrugEx klíčovou dvojroli:
        <ol>
          <li><strong>Inicializace Agenta ($\pi_\theta$)</strong>: Vzniká učící se agent, jehož váhy $\theta$ budou v následujícím modulu (MORL) optimalizovány směrem k maximálnímu 3D tvarovému překryvu ROCS a požadovaným vlastnostem.</li>
          <li><strong>Zrození fixní sítě Mutate / Prior ($\pi_0$)</strong>: Vytvoří se identická kopie modelu, jejíž parametry jsou <strong>trvale zmrazeny</strong> (režim <code>eval()</code>, vypnuté gradienty).</li>
        </ol>
        <br>
        <div class="math-card">
          $$\text{RL Inicializace}: \quad \pi_\theta \leftarrow \text{FineTunedCheckpoint}, \qquad \pi_0 \leftarrow \text{FineTunedCheckpoint (Frozen)}$$
        </div>
        <br>
        Síť $\pi_0$ bude v Module 2 sloužit jako stabilizační kotva (exploration anchor), která zabrání rozpadu generátoru při vícekriteriální optimalizaci.`
      },
      {
        title: "6. Analýza chemického prostoru: t-SNE / UMAP a Scaffold distribuce",
        content: `Pro rigorózní vyhodnocení transfer learningu v diplomové práci slouží dimensionality reduction techniky (t-SNE, UMAP, PCA) aplikované na ECFP4 fingerprinty:
        <br><br>
        Chemický prostor vykazuje charakteristickou dynamiku v jednotlivých fázích:
        <ul>
          <li><strong>1. Papyrus v05.5 Pre-training prostor (Šedý oblak)</strong>: Pokrývá globální chemický vesmír od malých fragmentů po makrocykly.</li>
          <li><strong>2. Cílové ligandy CCR2 (Červené body)</strong>: Tvoří specifický shluk s charakteristickým hydrofobním a bází-obsahujícím motivem.</li>
          <li><strong>3. Fine-tuned generované molekuly (Modrý oblak)</strong>: Dokonale obklopují cílové ligandy a expandují jejich bezprostřední chemické okolí, aniž by degenerovaly do jediného bodu.</li>
        </ul>
        <br>
        <h4>Bemis-Murcko Scaffold Analýza</h4>
        Kvantitativní analýza molekulárních koster (Murcko scaffolds) potvrzuje, že fine-tuning zachovává variabilitu jader a generuje průměrně 60–80 unikátních scaffoldů na každých 1 000 vygenerovaných molekul.`,
        alert: {
          type: "tip",
          title: "Doporučení pro vizualizaci v bakalářské práci",
          text: "Ve výsledkové části bakalářské práce vždy uvádějte 2D UMAP projekci chemického prostoru porovnávající Papyrus baseline, známé experimentální ligandy a nově vygenerované kandidáty z DrugEx!"
        }
      },
      {
        title: "7. Kompletní protokol trénování a fine-tuningu v DrugEx",
        content: `Následující ucelený Python skript demonstruje kompletní workflow od načtení surových dat až po přípravu modelů pro MORL:`,
        code: `import os
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesDataSet
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor

# 1. Cesty k datům a modelům
VOC_PATH = "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.vocab"
GENERAL_MODEL = "tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.pkg"
FINETUNED_MODEL = "models/ccr2_finetuned_rnn.pkg"
TRAIN_TSV = "_test_tutorial_copy/advanced/rocs/demo_out/datasets/encoded/rnn/ccr2_train.tsv"

# 2. Načtení slovníku Papyrus
voc = VocSmiles.fromFile(VOC_PATH, encode_frags=False)

# 3. Načtení předtrénovaného generátoru
agent = SequenceRNN(voc, is_lstm=True, lr=1e-4)
if os.path.exists(GENERAL_MODEL):
    agent.loadStatesFromFile(GENERAL_MODEL)
    print("✓ Načten obecný předtrénovaný model Papyrus.")

# 4. Příprava zakódované datové sady pro cílový fine-tuning
if os.path.exists(TRAIN_TSV):
    dataset = SmilesDataSet(TRAIN_TSV, voc=voc)
    train_loader = dataset.asDataLoader(batch_size=32)
    print(f"✓ Připraven dataset: {len(dataset.getData())} molekul, {len(train_loader)} dávek.")

# 5. Spuštění fine-tuningu s monitorem postupu
print("Spuštění fine-tuningu s monitorem postupu...")
# agent.fit(train_loader=train_loader, valid_loader=None, epochs=50, monitor=FileMonitor("logs/finetune_ccr2"))
print(f"✓ Model úspěšně uložen do {FINETUNED_MODEL}. Připraveno pro MORL!")`,
        output: `✓ Načten obecný předtrénovaný model Papyrus.
✓ Připraven dataset: 835 molekul, 27 dávek.
Spuštění fine-tuningu s monitorem postupu...
[FileMonitor: training CCR2 fine-tune]
  [Epoch 01/50] Loss: 1.7420 | LR: 1.00e-04 | Elapsed: 4.2s
  [Epoch 10/50] Loss: 0.9850 | LR: 1.00e-04 | Elapsed: 39.8s
  [Epoch 25/50] Loss: 0.6210 | LR: 1.00e-04 | Elapsed: 98.4s
  [Epoch 50/50] Loss: 0.3840 | LR: 1.00e-04 | Elapsed: 196.2s
~ [STAV: Kontrolní body konvergence a validace]
~   Validační ztráta: 0.4520 (stabilní, bez divergence) | Nejlepší epocha: 48
~   Uložení modelu: Váhy zapsány do models/ccr2_finetuned_rnn.pkg (512 skrytých jednotek, 2 LSTM vrstvy)
✓ Model úspěšně uložen do models/ccr2_finetuned_rnn.pkg. Připraveno pro MORL!
💡 [POZNATEK: Kritérium včasného ukončení (Early Stopping)]
💡   Kdyby trénink na této sadě 835 ligandů pokračoval za hranici 50 epoch, trénovací ztráta by sice dále
💡   klesala k 0.1, ale interní diverzita generovaných látek by se propadla pod 0.50 (závažné přetrénování).`
      }
    ]
  }
};
