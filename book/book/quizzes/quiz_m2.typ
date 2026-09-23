#import "../nature_theme.typ": *

= Kvíz M2: Multi-Objective Reinforcement Learning & Paretova Optimalita

Tento test prověřuje porozumění mechanismům Reinforcement Learning (RL), Paretově dominanci, funkci Paretovské vzdálenosti nahloučení (Crowding Distance), normalizačním modifikátorům a QSAR prediktorům.

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
  [Proč je jednokriteriální optimalizace (např. pouhé maximalizování QSAR bioaktivity) při de novo návrhu léčiv nevhodná?],
  (
    [Protože model začne generovat vysoce lipofilní, velké nebo synteticky nerealizovatelné molekuly (tzv. exploatace odměny / reward hacking), které sice mají vysoké predikované skóre, ale selhávají jako skutečná léčiva.],
    [Protože jednokriteriální optimalizace vyžaduje příliš mnoho GPU jader.],
    [Protože PyTorch nepodporuje jednokriteriální gradientní sestup.],
    [Jednokriteriální optimalizace funguje bezchybně a vícekriteriální není potřeba.]
  )
)

#quiz_question(
  2,
  [Jakou roli hraje mutační síť (mutate / prior network) a parametr $epsilon$ v DrugEx RL explorátoru?],
  (
    [Mutační síť generuje 3D konformace a epsilon určuje počet atomů vodíku.],
    [Slouží jako pevná referenční politika: s pravděpodobností $1 - epsilon$ se vzorkuje z trénovaného agenta (exploatace) a s pravděpodobností $epsilon$ z mutační sítě (explorace), což zabraňuje katastrofickému zapomínání chemického prostoru.],
    [Mutační síť slouží výhradně k výpočtu ztrátové funkce MSE.],
    [Epsilon určuje celkový počet epoch v tréninku.]
  )
)

#quiz_question(
  3,
  [Co definuje Paretovu dominanci mezi dvěma molekulami $A$ a $B$ v prostoru $K$ optimalizovaných cílů?],
  (
    [Molekula A dominuje B, pokud má vyšší molekulovou hmotnost.],
    [Molekula A dominuje B, pokud je ve všech $K$ kritériích alespoň stejně dobrá jako B ($f_k(A) >= f_k(B)$ pro všechna $k$) a v alespoň jednom kritériu je striktně lepší ($f_j(A) > f_j(B)$).],
    [Molekula A dominuje B, pokud má jednodušší SMILES řetězec.],
    [Dominance nastává pouze tehdy, když obě molekuly mají zcela identické skóre.]
  )
)

#quiz_question(
  4,
  [K čemu slouží výpočet vzdálenosti nahloučení v Paretově schématu v DrugEx (`ParetoCrowdingDistance`)?],
  (
    [K odhadu hustoty okolních řešení na Paretově frontě, čímž jsou vyšší odměnou zvýhodňována izolovanější (rozmanitější) řešení, aby se zabránilo shlukování do jednoho úzkého subprostoru.],
    [K výpočtu sterického překryvu molekul s vazebným místem receptoru.],
    [K měření vzdálenosti mezi trénovacími a validačními daty.],
    [K výpočtu euklidovské vzdálenosti v latentním prostoru autoencoderu.]
  )
)

#quiz_question(
  5,
  [Jak funguje modifikátor `SmoothClippedScore(lower_x=5.0, upper_x=3.0)` aplikovaný na `Property('SA')` (SAScore)?],
  (
    [Převede jakoukoliv hodnotu na pevnou konstantu 1.0.],
    [Vzhledem k tomu, že nižší SAScore znamená snazší syntézu, transformuje surové skóre hladkou sigmoidální křivkou: molekuly s $"SA" <= 3.0$ mají odměnu $approx 1.0$, molekuly s $"SA" >= 5.0$ mají odměnu $approx 0.0$ a mezi 3 a 5 je plynulý přechod.],
    [Vynásobí hodnotu SAScore číslem -1 bez jakékoliv normalizace.],
    [Funguje jako binární skokový filtr (vše pod 3 je 0, vše nad 5 je 1).]
  )
)

#quiz_question(
  6,
  [Jaký je rozsah výstupních hodnot odměny z jakéhokoliv modifikátoru skóre v DrugEx (např. `ClippedScore`, `MinMaxScore`)?],
  (
    [Libovolná reálná čísla v intervalu $(-infinity, +infinity)$],
    [Normalizovaný interval $[0.0, 1.0]$, kde 0.0 představuje zcela nežádoucí a 1.0 ideální stav.],
    [Celočíselné diskrétní hodnoty ${-1, 0, 1}$],
    [Procentuální hodnoty od $0 %$ do $1000 %$]
  )
)

#quiz_question(
  7,
  [Co se stane, pokud generovaná molekula není chemicky validní (např. má neplatnou valenci atomu)?],
  (
    [Python vyvolá kritickou výjimku a celý trénink okamžitě havaruje.],
    [Prostředí `DrugExEnvironment` jí automaticky přiřadí nulové skóre (odměnu 0.0) ve všech cílech, čímž je v gradientním updatu penalizována.],
    [Validátor se pokusí nahradit všechny problémové atomy uhlíkem.],
    [Molekula je ignorována a v epoše se místo 1000 molekul použije jen 999 bez penalizace.]
  )
)

#quiz_question(
  8,
  [Který z následujících nástrojů integrovaných v DrugEx slouží k predikci biologické aktivity pomocí QSAR modelů?],
  (
    [`QSPRpred` (podporující Random Forest, SVM, XGBoost i hluboké neuronové sítě)],
    [OpenFOAM],
    [GROMACS MD engine],
    [Blender 3D]
  )
)
