#import "../nature_theme.typ": *

= Kvíz M1: De Novo Generování & Molekulární Reprezentace v DrugEx

Tento test ověřuje znalosti molekulárních reprezentací (SMILES, molekulární grafy, fragmenty), tokenizačních kontraktů `VocSmiles`, standardizace struktur a architektur generátorů (SequenceRNN, Transformers).

_Poznámka: Správné odpovědi a detailní vysvětlení naleznete v samostatné kapitole *Klíč k řešení kvízů* na konci monografie._

#v(1em)

#let quiz_question(num, q_text, options) = block(
  width: 100%,
  stroke: 1pt + rgb("#e2e8f0"),
  fill: rgb("#f8fafc"),
  inset: 12pt,
  radius: 4pt,
  spacing: 1.4em,
  breakable: false
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 10.5pt, fill: nature_navy)[Otázka #num: #q_text]
  #v(0.5em)
  #enum(
    numbering: "A)",
    ..options.map(opt => text(size: 9.5pt, fill: rgb("#334155"))[#opt])
  )
]

#quiz_question(
  1,
  [Jaký je hlavní rozdíl mezi reprezentací molekuly pomocí SMILES a molekulárního grafu při generování nových sloučenin?],
  (
    [SMILES je 1D textová sekvence náchylná k nevalidní syntaxi závorek a cyklů, zatímco graf přímo reprezentuje topologii atomů a vazeb.],
    [SMILES obsahuje 3D kartézské souřadnice atomů, zatímco graf popisuje pouze 2D planární projekci.],
    [Grafová reprezentace vyžaduje výrazně méně paměti než 1D SMILES řetězec.],
    [SMILES neumí reprezentovat aromatické kruhy a heteroatomy, zatímco graf ano.]
  )
)

#quiz_question(
  2,
  [K čemu slouží třída `VocSmiles` v balíčku DrugEx?],
  (
    [K ukládání 3D kartézských souřadnic a výpočtu dipólového momentu.],
    [Definuje slovník povolených chemických tokenů (např. 'C', 'N', '=', '[nH]'), jejich mapování na celočíselné indexy a obsluhuje speciální tokeny (START, END, PAD).],
    [Slouží výhradně k vizualizaci 2D chemických struktur ve formátu PNG.],
    [Provádí kvantově-chemické výpočty HOMO-LUMO orbitalů.]
  )
)

#quiz_question(
  3,
  [Proč DrugEx před trénováním generátorů standardizuje molekuly pomocí `SmilesStandardizer`?],
  (
    [Aby se odstranily protiionty a soli, neutralizovaly náboje (kde je to vhodné), kanonizovaly tautomery a sjednotil formát SMILES pro stabilní trénink.],
    [Aby se molekuly převedly do binárního formátu SDF pro dokování.],
    [Aby se všechny molekuly zvětšily na minimální molekulovou hmotnost 500 Da.],
    [Standardizace slouží pouze ke generování 3D konformací pomocí MMFF94.]
  )
)

#quiz_question(
  4,
  [Jak funguje generování sekvence v modelu `SequenceRNN` v DrugEx?],
  (
    [Model vygeneruje všechny tokeny naráz v jediném dopředném průchodu bez rekurence.],
    [Model využívá náhodné mutace genů bez jakékoliv neuronové sítě.],
    [Autoregresně: v každém kroku předpovídá pravděpodobnostní distribuci následujícího tokenu na základě předchozích tokenů a skrytého stavu GRU/LSTM buňky, dokud nevygeneruje token END.],
    [Model optimalizuje pouze energii molekuly v silovém poli.]
  )
)

#quiz_question(
  5,
  [Jaký je účel dvoufázového trénování (Pre-training $->$ Fine-tuning) v DrugEx?],
  (
    [Fine-tuning je určen pouze pro kvantování vah na 8-bit integer.],
    [Pre-training na velké obecné databázi (např. Papyrus) naučí model základní chemickou gramatiku a valenční pravidla; fine-tuning na menší sadě aktivních ligandů zaměří generátor na specifický chemotyp cíle.],
    [Pre-training slouží k výběru hyperparametrů a fine-tuning k vygenerování 3D souřadnic.],
    [Dvoufázový trénink se používá jen tehdy, pokud nemáme k dispozici GPU.]
  )
)

#quiz_question(
  6,
  [Který parametr v příkazu `python -m drugex.dataset` určuje, že se data připravují pro grafový transformer?],
  (
    [`-mt graph`],
    [`-model_type rnn_lstm`],
    [`-encoding_format xyz`],
    [`-target_pocket 3d`]
  )
)

#quiz_question(
  7,
  [Jaká je hlavní výhoda `GraphTransformer` oproti `SequenceRNN` při de novo návrhu?],
  (
    [Grafový transformer pracuje přímo s topologií molekuly, umožňuje scaffold-constrained generování a eliminuje syntaktické chyby způsobené linearizací do SMILES.],
    [Grafový transformer je 100x rychlejší na CPU než jednoduché RNN.],
    [Grafový transformer nevyžaduje žádná trénovací data.],
    [Grafový transformer automaticky počítá vazebnou volnou energii delta G v kcal/mol.]
  )
)

#quiz_question(
  8,
  [Co způsobí nastavení vysoké teploty (např. $T = 1.5$) při vzorkování ze softmax distribuce generátoru?],
  (
    [Generátor bude generovat pouze nejčastější molekulu z trénovací sady.],
    [Dojde k vyhlazení pravděpodobnostní distribuce tokenů, což zvýší diverzitu generovaných struktur, ale může mírně snížit podíl chemicky validních molekul.],
    [Generátor začne generovat molekuly s vysokou afinitou bez nutnosti trénování.],
    [Teplota nemá na vzorkování žádný vliv.]
  )
)

#quiz_question(
  9,
  [Který speciální token v `VocSmiles` signalizuje začátek generování nového řetězce?],
  (
    [Token END],
    [Token PAD],
    [Token START (nebo GO token)],
    [Token UNK]
  )
)

#quiz_question(
  10,
  [Jaký je formát souboru s příponou `.pkg` vytvářeného v DrugEx během tréninku?],
  (
    [Čistý textový soubor obsahující pouze SMILES řetězce.],
    [PyTorch serializovaný balíček (state_dict) obsahující váhy modelu, architekturu a metadata generátoru.],
    [Binární 3D trajektorie molekulární dynamiky.],
    [Kompilovaná C++ dynamická knihovna pro OpenEye.]
  )
)
