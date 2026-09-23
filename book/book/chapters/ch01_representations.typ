#import "../nature_theme.typ": *

= 1. Molekulární Reprezentace, Jazykové Modely & Chemické GPT

== 1.1 Paradigma molekulárních reprezentací v de novo designu léčiv

Výpočetní reprezentace chemické struktury představuje fundamentální most mezi fyzikálně-chemickou realitou molekuly a matematickým aparátem strojového učení. V klasické chemoinformatice a QSAR modelování dominovaly po desetiletí fixní binární či celočíselné vektorové deskriptory – zejména cirkulární Morganovy otisky (Extended Connectivity Fingerprints, ECFP4/ECFP6), strukturní klíče MACCS či topologické indexy. Tyto deskriptory vynikají při bleskovém prohledávání substrukturní podobnosti v rozsáhlých komerčních databázích, mají však zásadní slabinu pro generativní navrhování léčiv: jsou *jednosměrné (ireverzibilní)*. Z fixního 2048-bitového otisku nelze deterministicky ani analyticky zrekonstruovat původní molekulární strukturu.

Generativní de novo design vyžaduje *invertibilní, spojité či diskrétní generovatelné reprezentace*, které umožňují optimalizačnímu algoritmu syntetizovat nové chemické entity atom po atomu nebo fragment po fragmentu. V moderní generativní chemoinformatice rozlišujeme tři dominantní úrovně reprezentací:

1. *1D Lineární řetězcové notace (String Notations)*: Linearizace chemických struktur do textových sekvencí diskrétních znaků a tokenů (SMILES, DeepSMILES, SELFIES). Umožňují přímé nasazení architektur zpracování přirozeného jazyka (NLP) – kauzálních autoregresních transformerů (Chemical GPT) a rekurentních neuronových sítí (LSTM, GRU).
2. *2D Atribuované molekulární grafy (Attributed Molecular Graphs)*: Exaktní popis molekuly jako matematického grafu $G = (V, E)$, kde uzly $V$ reprezentují těžké atomy s vektory fyzikálně-chemických příznaků (atomové číslo, hybridizace, formální náboj, aromatický status) a hrany $E$ představují kovalentní vazby s informací o jejich řádu a stereochemii.
3. *Fragmentové a synthonové reprezentace (Bag of Fragments / Synthon Graphs)*: Dekompozice molekul na synteticky dostupné stavební bloky podle retrosyntetických štěpných pravidel (BRICS, RECAP). Generování probíhá kombinací celých syntetických bloků zakončených dummy atomy, což zásadně omezuje chemický prostor na synteticky snadno realizovatelné kandidáty.

#table(
  columns: (1.5fr, 1.3fr, 1.3fr, 1.3fr, 1.3fr),
  align: (left, left, left, left, left),
  table.header(
    [*Kritérium*],
    [*1D SMILES (RNN)*],
    [*1D SMILES (GPT)*],
    [*2D Grafy (GraphTrans)*],
    [*BRICS Fragmenty*]
  ),
  [Paměťová složitost], [$O(T)$ (lineární)], [$O(T^2)$ (kvadratická)], [$O(N^2 dot d)$ (tenzor soused.)], [$O(F)$ (krátké sekvence)],
  [Propustnost vzorkování], [> 2~500 mol/s], [~ 850 mol/s], [~ 200 mol/s], [~ 600 mol/s],
  [Syntaktická validita], [94–98 % (po pre-train.)], [96–99 %], [100 % (valenční maska)], [98–100 %],
  [Scaffold-constrained], [Prefixové maskování], [In-filling / Prefix], [Přirozené na uzlech], [Vynikající (substituce)],
  [Syntetická proveditelnost], [Variabilní (nutno řídit)], [Střední až vysoká], [Vysoká (řízená pravidly)], [Extrémní (ověřené reakce)]
)

#lab_context_box("Volba reprezentace pro flexibilní cíle a IDP proteiny", [
  Při de novo designu ligandů pro vnitřně neuspořádané proteiny (IDP) a flexibilní receptorové kapsy (např. chemokinový receptor CCR2) volíme hybridní strategii: 1D sekvenční modely (`SequenceRNN`, `SequenceTransformer`) slouží pro masivní exploraci chemického prostoru a generování nepředpojatých knihoven, zatímco fragmentové modely (`GraphTransformer`) umožňují rigorózní zachování pevných farmakoforových scaffoldů s garantovanou syntetickou proveditelností.
])

== 1.2 Fragmentové štěpení BRICS a synthonové paradigma

Při fragmentovém generování v DrugEx je výchozí molekula rozštěpena podle 16 retrosynteticky validních chemických pravidel BRICS (*Biochemical Properties Reacting and Inherent Chemistry Structure*, Degen et al., 2008). Tento přístup simuluje prověřené organické reakce (tvorbu amidických vazeb, nukleofilní substituce, cross-coupling reakce a etherifikace).

Místa rozpojení kovalentních vazeb jsou označena speciálními *dummy atomy (synthony)* s číselným indexem příslušného štěpného pravidla (např. `[1*]c1ccccc1` nebo `[4*]C(=O)N[5*]`). Tyto indexy definují chemickou komplementaritu – fragment s koncovkou `[4*]` (karbonyl) se smí v generátoru sloučit výhradně s fragmentem nesoucím koncovku `[8*]` či `[5*]` (aminový dusík).

#figure(
  image("../figures/fig_brics_reconstruction.svg", width: 95%),
  caption: [Retrosyntetická dekompozice a synthonová rekonstrukce BRICS fragmentů v DrugEx. Vstupní molekula je rozštěpena v místech strategických vazeb (např. amidická vazba pravidlem L1, alkylace piperazinu pravidlem L4/L8). Výsledné stavební bloky nesou explicitní syntetické vektory, což garantuje laboratorní proveditelnost nově složených sloučenin.]
)

#code_card(
  title: "BRICS dekompozice a extrakce synthonů v RDKit",
  lang: "python",
  code: "from rdkit import Chem
from rdkit.Chem import BRICS

# Retrosyntetický rozpad na syntetické stavební bloky (synthony)
mol = Chem.MolFromSmiles('Cc1ccc(NC(=O)c2cccc(C(=O)N3CCN(Cc4ccccc4)CC3)c2)cc1')
synthons = sorted(list(BRICS.BRICSDecompose(mol)))
# -> Celkem 7 synthonů s explicitními syntetickými vektory:
#    '[1*]C([6*])=O'      (amidický karbonyl)
#    '[16*]c1ccc(C)cc1'   (aromatické jádro)
#    '[5*]N1CCN([5*])CC1' (piperazinový můstek)"
)

#rule_box("Integrita synthonových vektorů", [
  Při práci s fragmentovými bázemi nikdy neodstraňujte číselné indexy u dummy atomů (`[1*]`, `[5*]`). Tyto indexy představují reaktivní vektory určující geometrii a chemickou kompatibilitu připojení. Nahrazení obecnou hvězdičkou `*` vede ke ztrátě syntetického řádu a tvorbě nestabilních chemických vazeb.
])

== 1.3 Tokenizační architektura a kontrakt chemického slovníku VocSmiles

Standardní znaková tokenizace běžná v obecném NLP (rozklad textu po jednotlivých písmenech) představuje v chemoinformatice kritickou chybu. Rozklad řetězce `Clc1ccccc1` na prosté znaky `['C', 'l', 'c', '1', ...]` vede k tomu, že model interpretuje `'C'` jako alifatický uhlík a `'l'` jako neznámý izolovaný token, namísto atomu chloru.

DrugEx definuje striktní tokenizační kontrakt ve třídě `VocSmiles` (`drugex/data/corpus/vocabulary.py`). Chemický slovník sestává z přibližně 95 unikátních tokenů, které pokrývají kompletní organickou abecedu. Před samotným rozpadem regulárním výrazem provádí DrugEx jednoznačnou dvoupísmennou substituci halogenů:

$ "Cl" arrow.long "L", quad "Br" arrow.long "R" $

Tato substituce eliminuje jakoukoliv kolizi atomu chloru s alifatickým uhlíkem `C` či atomu bromu s borem `B`. Následně je aplikován regulární výraz zachycující explicitní atomové bloky v hranatých závorkách `(\[[^\[\]]{1,6}\])` a jednoznakové strukturní operátory.

#telemetry_box("Struktura a řídicí tokeny slovníku VocSmiles", [
  Chemický slovník `VocSmiles` disponuje pevnou sadou čtyř řídicích tokenů (Control Tokens) na vyhrazených indexech:
  - `<PAD>` (Index 0): Slouží k doplnění sekvencí na jednotnou délku batche ($L = 100$). V loss funkci a v mechanizmu pozornosti je tento index maskován (`padding_idx=0`).
  - `<GO>` / `<START>` (Index 1): Iniciační token vkládaný na začátek sekvence. Aktivuje skrytý stav autoregresního generátoru.
  - `<EOS>` / `<END>` (Index 2): Ukončovací token. Vygenerování tohoto tokenu okamžitě zastaví vzorkovací smyčku.
  - `<UNK>` (Index 3): Záchytný token pro neobvyklé chemické prvky nepřítomné v trénovacím korpusu.
  
  Celková velikost slovníku: $|V| = 95$ tokenů. Matice embeddingu má rozměr $bold(E) in RR^(95 times 512)$.
])

== 1.4 Pětistupňová sanitace chemické struktury pro polyfenoly

Trénovací databáze (Papyrus, ChEMBL, ZINC) shromažďují sloučeniny pocházející z tisíců různých experimentálních protokolů. Tyto molekuly obsahují krystalizační protiionty, nestejné protonační stavy, těžké izotopy a redundantní tautomerní formy. Bez přísné předzpracující pipeline by generativní model plýtval svou kapacitou na reprodukci solí a syntaktického šumu.

DrugEx v modulu `drugex/molecules/converters/standardizers.py` implementuje rigorózní *5stupňový sanitační protokol* kombinující nástroje MolVS a RDKit prostřednictvím tříd `DefaultStandardizer` a `CleanSMILES`:

1. *Desalting & Metal Disconnection (`MetalDisconnector`)*: Kovalentně či koordinačně vázané ionty alkalických kovů ($"Na"^+$, $"K"^+$) a přechodných kovů ($"Zn"^(2+)$, $"Fe"^(3+)$) jsou odpojeny na volné iontové páry. Následně `LargestFragmentChooser` vybere největší spojitý organický skelet a odstraní solvatační molekuly ($"H"_2"O"$, $"DMSO"$, $"EtOH"$) i běžné protiionty ($"Cl"^-$, $"Br"^-$, $"TFA"^-$).
2. *Tautomerická a funkční normalizace (`Normalizer`)*: Standardizace nestandardních funkčních skupin (např. nitro skupin $R-"N"(=O)=O arrow.long R-["N"+](=O)["O"-]$, azidů, 1,3-dipolárních systémů a keto-enol tautomerů) do jednotného kanonického zápisu.
3. *Neutralizace nábojů (`Uncharger`)*: Protonace a deprotonace funkčních skupin do neutrálního stavu ($R-"COO"^- arrow.long R-"COOH"$, $R-"NH"_3^+ arrow.long R-"NH"_2$).
4. *Izotopové vyčištění & Atomární validace*: Nahrazení těžkých izotopů ($attach("H", tl: 2)$, $attach("H", tl: 3)$, $attach("C", tl: 13)$) standardními nuklidy a striktní ověření, že molekula obsahuje alespoň jeden atom uhlíku (`[#6]`) a žádné reziduální soli `[Na, Zn]`.
5. *Stereochemické zploštění a kanonizace*: Klíčový krok `Chem.MolToSmiles(rd_mol, 0)` s parametrem `isomericSmiles=False`.

=== Případová studie: Sanitace polyfenolu (-)-Epigallocatechin (EGC)

Polyfenolické přírodní látky, jako je (-)-Epigallocatechin (EGC, SMILES: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O`), představují extrémní testovací případ chemické sanitace:

#code_card(
  title: "Sanitace polyfenolu EGC pomocí DefaultStandardizer a CleanSMILES",
  lang: "python",
  code: "from drugex.molecules.converters.standardizers import DefaultStandardizer, CleanSMILES

# Surový SMILES (-)-Epigallocatechinu se stereocentry a solným příměskem (sodný fenolát)
raw_egc = '[Na+].C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)[O-])O)O'

# 5-stupňový sanitační protokol: desalting, neutralizace a achirální zploštění
clean_egc = DefaultStandardizer()(raw_egc)
canonical_egc = CleanSMILES(is_deep=True)(clean_egc)
# -> 'Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)C(O)C2' (desolvováno, neutralizováno, achirální)"
)

#pitfall_box("Proč unconstrained prior vyžaduje zploštění stereocenter?", [
  Všimněte si, že sanitovaný řetězec EGC ztratil stereochemické deskriptory `[C@H]` a přešel na achirální formu `OC(c1...)C(O)C2`. Tento krok je záměrný: kdyby obecný jazykový model (`VocSmiles`) musel udržovat prostorové izomery pro každé chirální centrum, velikost slovníku a kombinační prostor by explodovaly. Model by se potýkal se syntaktickým driftem (např. generováním nekonzistentních konfigurací na sousedních uhlících). V DrugEx se proto sekvenční prior učí obecnou topologii a *stereocentra jsou explicitně enumerována a energeticky minimalizována až ve Fázi 3D konformačního modelování (ETKDGv3)* pro tvarové srovnání v ROCS.
])

== 1.5 Didaktická elevace chemického GPT: Principy a metafory

Architektura Transformer způsobila revoluci v modelování chemického jazyka. Abychom pochopili její fundamentální převahu nad sekvenčními modely, zavedeme tři názorné didaktické metafory:

=== 1. Chemik u kruhového stolu (Self-Attention vs. Sekvenční RNN)

Tradiční rekurentní sítě (LSTM, GRU) fungují jako *chemik procházející úzkou chodbou*: v každém kroku drží v ruce skrytý stav $bold(h)_t$ a buněčnou paměť $bold(c)_t$. Když v kroku $t=2$ otevře aromatický kruh symbolem `c1`, musí tuto informaci nést přes desítky mezilehlých substituentů, větví a funkčních skupin. V důsledku postupného násobení váhovými maticemi dochází k exponenciálnímu útlumu gradientu. Pokud je molekula dlouhá 40 tokenů, RNN často „zapomene“, že kruh otevřela, a sekvenci ukončí bez uzavíracího čísla `1`, což vede k nevalidnímu řetězci.

Naproti tomu Transformer funguje jako *chemik sedící u kulatého stolu*, na němž jsou rozprostřeny všechny dosud položené atomy současně. Mechanismus Self-Attention (vlastní pozornosti) umožňuje modelu rozsvítit reflektor pozornosti z aktuální pozice na libovolný předchozí atom v jediném výpočetním kroku, bez ohledu na to, zda jsou od sebe vzdáleny 2 nebo 60 tokenů. Uzavření kruhu je tak řízeno přímou vazbou mezi atomem $t$ a atomem rozpojení cyklu.

=== 2. Query, Key, Value v chemickém prostoru

Matematický aparát Self-Attention operuje s trojicí matic vzniklých projekcí vstupního stavu $bold(H) in RR^(T times d_("model"))$:

$ bold(Q) = bold(H) bold(W)_Q, quad bold(K) = bold(H) bold(W)_K, quad bold(V) = bold(H) bold(W)_V $

V chemickém prostoru mají tyto tenzory přesnou fyzikální interpretaci:

- *Dotaz (Query $bold(Q)$)*: Představuje *elektrofilní / kationtové centrum* aktuálně generovaného atomu. Vyjadřuje poptávku: _„Jsem karbonylový uhlík a hledám nukleofilní centrum s volným elektronovým párem pro vznik kovalentní vazby.“_
- *Klíč (Key $bold(K)$)*: Představuje *nukleofilní reaktivní afinitu a valenční dostupnost* všech existujících atomů v sekvenci. Odpovídá na dotaz: _„Jsem sekundární aminový dusík s volnou valencí a vysokou elektronovou hustotou.“_ Skalární součin $bold(Q) bold(K)^T$ kvantifikuje chemickou afinitu mezi těmito dvěma centry.
- *Hodnota (Value $bold(V)$)*: Představuje *elektronový a prostorový příspěvek vznikající vazby* – hybridizaci ($s p^2, s p^3$), parciální náboj, geometrické pnutí a sterické stínění, které se přenesou do aktualizované reprezentace molekuly.

Pozornostní rozdělení pak počítáme vzorcem:

$ "Attention"(bold(Q), bold(K), bold(V)) = "softmax"lr((frac(bold(Q) bold(K)^T, sqrt(d_k)) + bold(M))) bold(V) $

=== 3. Kauzální maskování jako opona času

Při generování molekuly zleva doprava nesmí generátor „vidět do budoucnosti“ – výběr atomu v čase $t$ nesmí být ovlivněn atomy, které teprve vzniknou v časech $t+1, t+2, dots, T$. Kauzální maska $bold(M) in RR^(T times T)$ funguje jako neprostupná *opona času*:

$ M_(i j) = cases(0 & "pro" j <= i, -infinity & "pro" j > i) $

Jelikož pro hodnotu $-infinity$ platí $exp(-infinity) = 0$, funkce Softmax přiřadí budoucím atomům přesně nulovou pravděpodobnost:

#telemetry_box("Kauzální maskování na sekvenci benzenového kruhu", [
  Uvažujme autoregresní syntézu aromatického benzenového kruhu `c 1 c c c c 1`:
  
  $
  bold(M)_("aromatic") = mat(
    0, -infinity, -infinity, -infinity, -infinity, -infinity, -infinity;
    0, 0, -infinity, -infinity, -infinity, -infinity, -infinity;
    0, 0, 0, -infinity, -infinity, -infinity, -infinity;
    0, 0, 0, 0, -infinity, -infinity, -infinity;
    0, 0, 0, 0, 0, -infinity, -infinity;
    0, 0, 0, 0, 0, 0, -infinity;
    0, 0, 0, 0, 0, 0, 0;
  ) quad attach(arrow.long, t: "Softmax") quad
  "Maska vah" = mat(
    1.0, 0, 0, 0, 0, 0, 0;
    p_(2,1), p_(2,2), 0, 0, 0, 0, 0;
    p_(3,1), p_(3,2), p_(3,3), 0, 0, 0, 0;
    dots, dots, dots, dots, 0, 0, 0;
    p_(7,1), p_(7,2), p_(7,3), p_(7,4), p_(7,5), p_(7,6), p_(7,7);
  )
  $
  
  V čase $t=7$ (kdy je generován uzavírací token `1`) má model plný přístup ke všem atomům cyklu a mechanismus pozornosti alokuje maximální váhu $p_(7, 2)$ právě na pozici $t=2$, kde kruh začal.
])

== 1.6 Tenzorová algebra a běhová telemetrie generátorů

V ekosystému DrugEx jsou implementovány dva komplementární sekvenční modely: `SequenceRNN` (založený na stacked LSTM buňkách) a `SequenceTransformer` (založený na GPT-2 blocích).

=== Matematika 6-hradlové LSTM buňky v SequenceRNN

V každém časovém kroku $t$ a pro každou vrstvu $l in {1, 2}$ počítá `SequenceRNN` následující soustavu tenzorových rovnic:

$
bold(f)_t &= sigma(bold(W)_f bold(x)_t + bold(U)_f bold(h)_(t-1) + bold(b)_f) && "(Forget Gate — řídí zapomnění staré paměti)" \
bold(i)_t &= sigma(bold(W)_i bold(x)_t + bold(U)_i bold(h)_(t-1) + bold(b)_i) && "(Input Gate — rozhoduje o zápisu nového atomu)" \
tilde(bold(c))_t &= tanh(bold(W)_c bold(x)_t + bold(U)_c bold(h)_(t-1) + bold(b)_c) && "(Candidate Memory — kandidátní chemický stav)" \
bold(c)_t &= bold(f)_t dot bold(c)_(t-1) + bold(i)_t dot tilde(bold(c))_t && "(Cell State — aktualizovaná paměť otevřených kruhů)" \
bold(o)_t &= sigma(bold(W)_o bold(x)_t + bold(U)_o bold(h)_(t-1) + bold(b)_o) && "(Output Gate — propustnost do skrytého stavu)" \
bold(h)_t &= bold(o)_t dot tanh(bold(c)_t) && "(Hidden State — výstupní vektor pro predikci)"
$

kde $sigma(z) = frac(1, 1 + e^(-z))$ je logistická sigmoida a $dot$ značí součin prvek po prvku (Hadamardův součin).

=== Termodynamika vzorkování a teplotní škálování

Výstupní vrstva generátoru produkuje vektor surových logitů $bold(z)_t in RR^(|V|)$. Převod logitů na pravděpodobnost výběru jednotlivých atomů je řízen Boltzmannovou distribucí s teplotním parametrem $T > 0$:

$ P(x_t = v mid x_(< t); T) = frac(exp(z_(t, v) / T), sum_(j=1)^(|V|) exp(z_(t, j) / T)) $

Vliv teploty na Shannonovu entropii $H(T) = - sum_v P(v) log_2 P(v)$ definuje rovnováhu mezi konzervativní validitou a explorativní novostí:
- *Nízká teplota ($T = 0.7$, $H approx 0.68$ bitu)*: Model preferuje vysoce pravděpodobné chemické vazby (benzenové kruhy, jednoduché alifatické řetězce). Validita generovaných struktur přesahuje 98 %, ale klesá strukturní diverzita.
- *Základní teplota ($T = 1.0$, $H approx 0.94$ bitu)*: Věrná statistická reprodukce bioaktivního prostoru databáze Papyrus. Vyvážený poměr validity (> 95 %) a novosti.
- *Vysoká teplota ($T = 1.3$, $H approx 1.28$ bitu)*: Vysoká entropie nutí model prozkoumávat exotické heterocykly a neobvyklé můstkové systémy. Roste však riziko nespárovaných závorek a validita klesá pod 85 %.

#code_card(
  title: "Tenzorová architektura SequenceRNN a skryté stavy",
  lang: "python",
  code: "from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN

# Inicializace slovníku a 2-vrstvého LSTM generátoru
voc = VocSmiles.fromFile('Papyrus05.5_smiles_rnn_PT.vocab')  # |V| = 95 unikátních tokenů
model = SequenceRNN(voc=voc, is_lstm=True)                   # 2x LSTM buňka, hidden_dim = 512

# Tenzorové dimenze skrytých stavů (Batch = 64, Skrytá dimenze = 512):
# h_t: torch.Size([2, 64, 512])   -> skrytý stav reprezentující kontext sekvence
# c_t: torch.Size([2, 64, 512])   -> buněčný stav dlouhodobé paměti LSTM
# W_out: torch.Size([95, 512])    -> lineární projekční hlava na chemické logity"
)

== 1.7 Dvoustupňový Transfer Learning na bázi Papyrus v05.5

Velikost farmakologicky relevantního chemického prostoru malých molekul se odhaduje na $10^(33)$ až $10^(60)$ struktur. Při návrhu ligandů pro specifické flexibilní receptory (např. CCR2) však experimentátor disponuje pouze stovkami známých aktivních sloučenin. Trénování hluboké neuronové sítě od nuly na takto malé sadě vede k okamžitému kolapsu chemické gramatiky.

DrugEx řeší tento problém dvoustupňovým *Transfer Learningem*:

1. *Fáze 1: Obecný pre-training (General Prior)*: Generátor je trénován na harmonizované databázi *Papyrus v05.5* obsahující více než 1,5 milionu bioaktivních sloučenin. Model se naučí univerzální zákonitosti chemické valence, stability cyklů a aromatity. Minimalizuje se ztráta Cross-Entropy:
  $ cal(L)_("CE")(theta) = - frac(1, |cal(D)|) sum_(X in cal(D)) frac(1, |X|) sum_(t=1)^(|X|) log P(x_t mid x_(< t); theta) $
2. *Fáze 2: Cílový Fine-Tuning (Target Adaptation)*: Předtrénovaný model je adaptován na specifickou sadu experimentálních ligandů (např. 835 známých antagonistů CCR2). Learning Rate (lr) se snižuje o řád ($"lr" = 10^(-4)$) a sleduje se vnitřní diverzita generované populace:
  $ I(S) = 1 - frac(1, N(N - 1)) sum_(i=1)^N sum_(j eq.not i) T_("Tanimoto")(bold("fp")(m_i), bold("fp")(m_j)) $
  Pokud $I(S)$ klesne pod 0.65, trénink je ukončen mechanismem Early Stopping, aby se předešlo katastrofickému zapomínání (*Catastrophic Forgetting*).

#insight_box("Zrození sítě Mutate/Prior pro Fázi 2 (MORL)", [
  Po dokončení fine-tuningu se checkpoint modelu rozdělí do dvou entit:
  1. *Policy Agent ($pi_theta$)*: Učící se model, jehož váhy budou v Module 2 optimalizovány algoritmem Policy Gradient směrem k maximálnímu 3D tvarovému skóre ROCS a bioaktivitě.
  2. *Mutate Prior ($pi_0$)*: Identická kopie modelu, jejíž parametry jsou *trvale zmrazeny* (`eval()`, vypnuté gradienty). Tato síť slouží jako kotva zabraňující kolapsu politiky.
])

#code_card(
  title: "Cílový fine-tuning SequenceRNN na ligandy CCR2 (Transfer Learning)",
  lang: "python",
  code: "from drugex.training.generators import SequenceRNN

# 1. Adaptace předtrénovaného generátoru se sníženým learning rate (1e-4)
agent = SequenceRNN(voc, is_lstm=True, lr=1e-4)
agent.loadStatesFromFile('Papyrus05.5_smiles_rnn_PT.pkg')

# 2. Trénink s validací a kontrolou vnitřní diverzity I(S)
# -> Epoch 01/50: Loss = 1.742, Val Loss = 1.791 (Perplexity: 5.71)
# -> Epoch 25/50: Loss = 0.621, I(S) = 0.812 > 0.65 (dostatečná diverzita)
# -> Epoch 50/50: Loss = 0.384, Val Loss = 0.452 -> checkpoint ccr2_finetuned.pkg

# 3. Zmrazený mutační Prior pi_0 pro následné MORL prostředí
mutate = SequenceRNN(voc, is_lstm=True)
mutate.loadStatesFromFile('ccr2_finetuned.pkg')
mutate.eval()  # fixace vah kotvy explorace (žádné gradienty)"
)
