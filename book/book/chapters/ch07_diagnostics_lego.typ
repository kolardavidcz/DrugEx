#import "../nature_theme.typ": *

= 7. Kuchařka Hyperparametrů, LEGO Skládání Prostředí & Diagnostický Strom Patologií

Optimalizace de novo generativního modelu v chemoinformatice nepředstavuje pouhé spuštění učení s výchozími parametry. Jde o delikátní inženýrskou disciplínu, kde se střetává stochastická dynamika Multi-Objective Reinforcement Learning (MORL) se striktními fyzikálně-chemickými zákony vazebných afinit, konformační flexibility a syntetické proveditelnosti. 

Tato kapitola poskytuje ucelený metodický aparát pro výzkumníka: modulární stavebnici skórovacích prostředí, receptář pro specifické vědecké scénáře, systematický diagnostický strom pro detekci a léčbu patologií generátoru a čtyřkvadrantovou matici telemetrie v reálném čase.

== 7.1 Receptář skládání prostředí: DrugExEnvironment jako modulární LEGO

V jádru platformy DrugEx leží třída `DrugExEnvironment`. Nejde o monolitický systém, nýbrž o funkční kompoziční framework, do něhož lze zapojit libovolné množství biologických, tvarových i fyzikálně-chemických cílů. Každá komponenta se skládá ze tří nezávislých vrstev:
+ *Skórovač (`Scorer`)*: Vypočítává surovou číselnou metriku (např. $T_("combo")$ z 3D překryvu, pIC50 z QSAR modelu, SAScore z fragmentové analýzy nebo LogP).
+ *Modifikátor (`Modifier`)*: Přenáší surovou hodnotu skrze nelineární normalizační funkci do unifikovaného intervalu žádoucnosti $[0.0, 1.0]$.
+ *Prahová hodnota (`Threshold`)*: Stanovuje exaktní kritérium, kdy je molekula v daném parametru považována za vyhovující kandidátní léčivo.

#rule_box("Separace hodnocení a normalizace")[
  Nikdy neposílejte do optimalizačního prostředí surové fyzikální veličiny bez modifikátoru. Předání surového SAScore ($1$ až $10$) či molekulové hmotnosti ($150$ až $800$ Da) způsobí okamžitý kolaps gradientů v Paretovském třídění, neboť jedna škála zcela převálcuje ostatní cíle.
]

=== Skládání prostředí pro $N = 1$ s neznámým či flexibilním cílem

Častým úkolem v praxi je návrh ligandů pro cíl, u něhož není k dispozici krystalická struktura v atomárním rozlišení (např. flexibilní receptory, membránové proteiny či _intrinsically disordered proteins_ -- IDP), avšak máme k dispozici alespoň jeden ověřený referenční ligand ($N = 1$).

V takovém scénáři se prostředí skládá jako třívrstvý filtr:
1. *3D Tvarová a farmakoforová komplementarita (`RDKitROCSScorer`)*: Hodnocení konformačního překryvu generovaných struktur s bioaktivní konformací referenčního ligandu pomocí metriky TanimotoCombo ($T_("combo") = T_("shape") + T_("color") in [0, 2]$).
2. *Syntetická proveditelnost (`Property('SA')`)*: Normalizovaný SAScore řízený klesající sigmoidou `SmoothClippedScore`, která nemilosrdně penalizuje syntetické anomálie.
3. *Léčivupodobnostní filtr (QED Gating)*: Asymetrická transformace fyzikálně-chemických deskriptorů zaručující orální biologickou dostupnost.

#code_card(
  title: "Modulární sestavení DrugExEnvironment pro N=1 flexibilní cíl",
  lang: "python",
  code: "from drugex.training.environment import DrugExEnvironment
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.molecules.converters.conformer import RDKitConformerGenerator

# 1. Konformační generátor s thread pinningem (1 thread/worker) a sterickými mantinely
conf_gen = RDKitConformerGenerator(max_conformers=30, max_rotatable_bonds=10, num_threads=1)

# 2. Modulární cíle s hladkými sigmoidálními modifikátory (SmoothClippedScore)
rocs_scorer = RDKitROCSScorer([\"ref_ligand.sdf\"], conformer_generator=conf_gen, metric=\"TanimotoCombo\", n_jobs=16)
mod_rocs = SmoothClippedScore(lower_x=0.60, upper_x=1.20) # Odměna nad prahem 0.85
mod_sa = SmoothClippedScore(lower_x=5.0, upper_x=3.0)      # Přísná penalizace syntetických monster

# 3. Sestavení unifikovaného prostředí DrugExEnvironment
env = DrugExEnvironment(
    scorers=[rocs_scorer, Property(\"SA\"), Property(\"QED\")],
    modifiers=[mod_rocs, mod_sa, None], # QED je již v intervalu [0, 1]
    thresholds=[0.85, 0.50, 0.60]
)
# -> [DrugExEnvironment] Inicializováno: 3 cíle | Všechny modifikátory vyhlazeny pro MORL"
)

#insight_box("Proč hladká sigmoida namísto ostrého prahu?")[
  Při použití skokových modifikátorů (`ClippedScore`) se gradient odměny v okolí prahu chová jako Diracova delta funkce. Pokud molekula dosáhne skóre $0.849$ při prahu $0.850$, dostane nulovou odměnu a generátor neobdrží žádnou informaci o tom, že je jen malý krok od cíle. `SmoothClippedScore` naproti tomu poskytuje spojitý gradient $nabla_theta J(theta)$, který jemně vede politiku sítě správným směrem.
]

=== Recepty pro specifické výzkumné scénáře

Podle specifikace výzkumného zadání lze z modulárních komponent poskládat pět klíčových produkčních konfigurací:

#table(
  columns: (1.5fr, 2fr, 2fr, 2.5fr),
  align: (left, left, left, left),
  [Recept], [Cílový scénář], [Použité skórovače], [Optimalizační mechanismus],
  [1. Plný MORL Stack], [Vyvážený návrh biologicky aktivních a syntetizovatelných látek], [ROCS ($T_("combo")$) + QSAR pIC50 + SAScore + MW/LogP], [Pareto Crowding Distance (PCD) s rovnoměrnou explorací fronty],
  [2. 3D Scaffold Hopping], [Únik z patentové ochrany při zachování 3D tvaru], [Multi-Ref ROCS + SimSmiles (Tanimoto penále vůči mateřskému ligandu)], [Crowding Distance odměňuje odlehlé chemické rodiny se zachovaným tvarem],
  [3. Selektivní Design], [Aktivace On-Target a vyřazení Off-Target / hERG], [On-Target pIC50 (rostoucí sigmoida) + Off-Target/hERG (klesající sigmoida)], [Pareto nedominované řazení s penalizací kardiotoxického okna],
  [4. Fragmentový Růst], [Růst z fixního farmakoforového jádra přes BRICS], [BRICS synthony + FragSequenceExplorer + 3D Shape Expansion], [100% fixace syntetického jádra, optimalizace substituentů v pod-kapsách],
  [5. Optuna Tuning], [Automatické nalezení hyperparametrů na superpočítači], [Maximalizace Desired Ratio a Unikátnosti], [Bayesovská optimalizace Tree-structured Parzen Estimator (TPE)]
)

== 7.2 Diagnostický strom patologií generátoru & Chemoinformatické antipatterny

Zatímco v případové studii EGC (Kapitola 6.5 a 6.6) jsme sledovali praktickou eliminaci driftu a kolapsu v konkrétním chemickém prostoru polyfenolických bioisosterů, tato podkapitola formuluje *univerzální inženýrský diagnostický strom* pro jakýkoliv de novo projekt v DrugEx.

Během Reinforcement Learning (RL) se neuronový generátor chová jako extrémně vynalézavý optimalizátor: namísto návrhu elegantních molekul aktivně hledá matematické slabiny v zadaném skórovacím prostředí. Pokud v odměňovací funkci existuje sebemenší skulinka, model ji nemilosrdně využije (tzv. _Reward Hacking_).

Následující diagnostický strom systematizuje čtyři nejzávažnější patologie, jejich chemické projevy a exaktní algoritmy nápravy.

#v(0.6em)
#align(center)[
  #block(
    fill: rgb("#0f172a"),
    inset: (x: 18pt, y: 8pt),
    radius: 6pt,
    stroke: 1pt + rgb("#38bdf8")
  )[
    #text(font: "IBM Plex Sans", weight: "bold", size: 10.5pt, fill: rgb("#38bdf8"))[
      DETEKCE ANOMÁLIE V REÁLNÉM ČASE TRÉNINKU
    ]
  ]
  #v(0.2em)
  #text(size: 14pt, fill: rgb("#64748b"))[↓]
  #v(0.2em)
]

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  // Card 1: Tuková hydra
  block(
    fill: rgb("#fef2f2"),
    stroke: 1pt + rgb("#fca5a5"),
    radius: 6pt,
    inset: 8pt,
    width: 100%
  )[
    #text(weight: "bold", size: 8.5pt, fill: rgb("#991b1b"))[1. TUKOVÁ HYDRA]\
    #text(size: 7.5pt, fill: rgb("#b91c1c"))[_Grease Explosion_]\
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#fca5a5"))
    #v(0.2em)
    #text(size: 7.5pt)[
      *Příznaky:*\
      • MW > 650, LogP > 6.0\
      • Alifatické řetězce\
      • RotBonds $gt.eq 15$
    ]\
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt + rgb("#fca5a5"))
    #v(0.2em)
    #text(size: 7.5pt, fill: rgb("#991b1b"))[
      *LÉČBA:*\
      1. SAScore filtr\
      2. RotBonds $lt.eq 8$\
      3. Max Heavy $lt.eq 42$
    ]
  ],
  // Card 2: Makrocyklický hacking
  block(
    fill: rgb("#fffbeb"),
    stroke: 1pt + rgb("#fcd34d"),
    radius: 6pt,
    inset: 8pt,
    width: 100%
  )[
    #text(weight: "bold", size: 8.5pt, fill: rgb("#92400e"))[2. MAKROCYKLY]\
    #text(size: 7.5pt, fill: rgb("#b45309"))[_Ring Hacking_]\
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#fcd34d"))
    #v(0.2em)
    #text(size: 7.5pt)[
      *Příznaky:*\
      • Velké kruhy $N > 8$\
      • Deformace v ETKDG\
      • Vysoké torzní pnutí
    ]\
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt + rgb("#fcd34d"))
    #v(0.2em)
    #text(size: 7.5pt, fill: rgb("#92400e"))[
      *LÉČBA:*\
      1. SSSR RingSize $lt.eq 7$\
      2. Strain energy penále\
      3. Konformační filtr
    ]
  ],
  // Card 3: Grammar Drift
  block(
    fill: rgb("#f5f3ff"),
    stroke: 1pt + rgb("#c4b5fd"),
    radius: 6pt,
    inset: 8pt,
    width: 100%
  )[
    #text(weight: "bold", size: 8.5pt, fill: rgb("#5b21b6"))[3. GRAMMAR DRIFT]\
    #text(size: 7.5pt, fill: rgb("#6d28d9"))[_Ztráta syntaxe_]\
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#c4b5fd"))
    #v(0.2em)
    #text(size: 7.5pt)[
      *Příznaky:*\
      • Validita < 70 %\
      • Neuzavřené kruhy\
      • Pětivazný uhlík
    ]\
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt + rgb("#c4b5fd"))
    #v(0.2em)
    #text(size: 7.5pt, fill: rgb("#5b21b6"))[
      *LÉČBA:*\
      1. Prior $D_("KL")$ regularizace\
      2. Epsilon $gt.eq 0.15$\
      3. Nižší learning rate
    ]
  ],
  // Card 4: Gradient Collapse
  block(
    fill: rgb("#ecfdf5"),
    stroke: 1pt + rgb("#6ee7b7"),
    radius: 6pt,
    inset: 8pt,
    width: 100%
  )[
    #text(weight: "bold", size: 8.5pt, fill: rgb("#065f46"))[4. GRADIENT COLLAPSE]\
    #text(size: 7.5pt, fill: rgb("#047857"))[_Kolaps do modu_]\
    #v(0.2em)
    #line(length: 100%, stroke: 0.5pt + rgb("#6ee7b7"))
    #v(0.2em)
    #text(size: 7.5pt)[
      *Příznaky:*\
      • Loss = NaN\
      • Unikátnost < 20 %\
      • Zacyklení do benzenu
    ]\
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt + rgb("#6ee7b7"))
    #v(0.2em)
    #text(size: 7.5pt, fill: rgb("#065f46"))[
      *LÉČBA:*\
      1. Grad Clip ($L_2 lt.eq 1.0$)\
      2. SmoothClippedScore\
      3. Crowding Distance
    ]
  ]
)
#v(0.8em)

=== 1. Tuková hydra (_Grease Explosion / Fatty Hydra_)

*Příčina & Projevy*: Pokud prostředí odměňuje 3D sterický překryv (Shape Tanimoto) bez striktního omezení flexibility a syntetické náročnosti, model brzy zjistí, že nejjednodušší způsob, jak vyplnit sterický objem vazebné kapsy, je navěšení dlouhých alifatických řetězců ($-("CH"_2)_n-"CH"_3$) nebo větvených terc-butylových shluků. 

Molekuly vykazují vysoké tvarové skóre, avšak jejich molekulová hmotnost překračuje $700$ Da, $log P > 6.5$ a počet rotovatelných vazeb je $gt.eq 15$. Výsledkem je mastná, biologicky nepoužitelná substance bez specifických vazebných interakcí.

*Algoritmické řešení*:
+ *Striktní SAScore gating*: Alifatické monstr-struktury obsahují vzácné větvené motivy penalizované ve fragmentové databázi PubChem. Nastavení `SmoothClippedScore(lower_x=4.0, upper_x=2.8)` pro `Property('SA')`.
+ *Omezení rotovatelných vazeb*: Zavedení skórovače `Property('RotBonds')` s klesající funkcí žádoucnosti nad hodnotou 7 vazeb.
+ *Konformační filtr*: Nastavení `max_rotatable_bonds=10` přímo v `RDKitConformerGenerator`. Molekuly překračující tento limit nedostanou vygenerované konformace a jejich ROCS skóre je automaticky $0.0$.

=== 2. Makrocyklický hacking (_Macrocyclic Ring Hacking_)

*Příčina & Projevy*: Model zjistí, že namísto větvených řetězců může objem kapsy pokrýt uzavřením skeletu do gigantického cyklu ($12$ až $24$ atomů v kruhu). Tento makrocyklus se v distanční geometrii ETKDG snadno deformuje a vyplní tvarovou šablonu referenčního ligandu. Syntéza takových makrocyklů je však v laboratorní praxi extrémně obtížná a konformační entropie v roztoku zabraňuje efektivní vazbě.

*Algoritmické řešení*:
+ *Filtr velikosti cyklů*: Zavedení strukturního filtru v předzpracování odměny, který identifikuje největší kruh v molekule (`rdkit.Chem.GetSymmSSSR`):
  $ N_("ring_max") <= 7 $
  Každá struktura obsahující cyklus s více než 7 atomy (vyjma explicitně povolených přírodních makrocyklů) obdrží nulovou odměnu.
+ *Penalizace konformačního pnutí (_Strain Energy_)*: Makrocykly generované distanční geometrií mívají vysokou torzní energii v silovém poli MMFF94s ($Delta E_("strain") > 25$ kcal/mol). Zavedení penalizace za strain energy tento hacking spolehlivě eliminuje.

=== 3. Rozpad chemické gramatiky (_Grammar Drift_)

*Příčina & Projevy*: Během agresivního Reinforcement Learning tréninku s vysokým learning rate ($> 5 times 10^(-4)$) se parametry sítě $theta$ rychle vzdalují od výchozího stavu $theta_0$. Generátor zapomíná základní syntaktická pravidla jazyka SMILES: přestává uzavírat kruhy (např. token `c1` bez odpovídajícího `1`), zanechává otevřené závorky nebo generuje pětivazný uhlík. Chemická validita generované knihovny prudce klesá z výchozích $98\%$ pod $50\%$.

#pitfall_box("Příliš rychlý trénink bez záchranné sítě")[
  Pokud se snažíte urychlit trénink drastickým zvýšením learning rate bez zapojení mutační sítě (Prior), model se dostane do oblasti ireverzibilního gradientního kolapsu. Ztrátová funkce exploduje a generátor produkuje pouze náhodný shluk znaků.
]

*Algoritmické řešení*:
+ *Penalizace Kullback-Leiblerovy divergence vůči Prioru*: Do celkové ztrátové funkce se zavádí regularizační člen:
  $ cal(L)(theta) = -bb(E)_(x tilde pi_theta) [R(x) log pi_theta(x)] + beta dot D_("KL")(pi_theta parallel pi_0) $
  kde $pi_0$ je zmrazená síť předtrénovaná na obecné chemické databázi Papyrus. Koeficient $beta in [0.01, 0.05]$ drží agenta v mantinelech reálné organické syntézy.
+ *Udržení exploračního epsilonu*: Trvalé vzorkování z mutační sítě s poměrem $epsilon in [0.15, 0.25]$.

=== 4. Exploze gradientů & Kolaps do lokálního minima (_Gradient Collapse_)

*Příčina & Projevy*: V REINFORCE algoritmu je gradient úměrný odměně: $nabla_theta J(theta) approx R(m) nabla_theta log pi_theta(m)$. Pokud molekula náhodně obdrží extrémně vysokou odměnu v prostředí s neomezenou škálou, gradientní krok posune váhy sítě natolik, že pravděpodobnost generování specifického fragmentu okamžitě dosáhne $1.0$. Nastává _Mode Collapse_ -- model začne cyklicky generovat tentýž benzenový či naftalenový derivát a unikátnost struktur spadne pod $10\%$.

*Algoritmické řešení*:
+ *Gradient Clipping*: Zastropování $L_2$ normy tenzoru gradientů na hodnotě $1.0$:
  ```python
  torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
  ```
+ *Normalizace odměn v dávce (Batch Reward Standardization)*: Odměna $R(m)$ je před výpočtem gradientu centrována a škálována:
  $ tilde(R)(m) = frac(R(m) - mu_R, sigma_R + 10^(-6)) $
+ *Paretovská Crowding Distance*: Algoritmus explicitně penalizuje jedince v přehuštěných oblastech objektivního prostoru a brání zacyklení v jediném chemickém lokálním optimu.

== 7.3 Čtyřkvadrantová telemetrie v reálném čase

Aby výzkumník nemusel čekat desítky hodin na dokončení běhu a teprve poté zjistil, že model podlehl reward hackingu, implementuje DrugEx monitorovací telemetrii pracující v reálném čase. Telemetrická matice sleduje čtyři vzájemně ortogonální dimenze kvality:

#table(
  columns: (1fr, 1.2fr, 1.2fr, 2fr),
  align: (center, center, center, left),
  [Kvadrant], [Metrika], [Zdravé rozmezí], [Diagnostická interpretace anomálie],
  [Q1], [Validita ($V$)], [$V gt.eq 95 %$], [Pokud $V < 85 %$, dochází ke Grammar Driftu; snižte learning rate.],
  [Q2], [Unikátnost ($U$)], [$U gt.eq 85 %$], [Pokud $U < 40 %$, model kolabuje do lokálního modu (Mode Collapse).],
  [Q3], [SAScore ($S$)], [$S_("avg") in [2.2, 3.2]$], [Pokud $S > 4.5$, generátor tvoří synteticky nepřístupná monstra.],
  [Q4], [Desired Ratio ($D_r$)], [Konvergence k $gt.eq 50 %$], [Pokud $D_r < 5 %$ po 20 epochách, prahy jsou nastaveny příliš přísně.]
)

#telemetry_box("Běhová telemetrie z 45. epochy RL tréninku (CCR2 Benchmark)")[
  `[EPOCH 045/050] Batch Size: 1024 | GPU VRAM: 3.42 GB / 80.0 GB (Allocated)`\
  `--------------------------------------------------------------------------------`\
  `▶ Q1 (Validita)   : 99.2%  [OK]  (Validních SMILES: 1016 / 1024)`\
  `▶ Q2 (Unikátnost) : 91.8%  [OK]  (Unikátních struktur: 933 / 1016, Entropy: 4.82)`\
  `▶ Q3 (Syntéza)    : 2.64   [OK]  (SAScore průměr: 2.64 ± 0.38, Max: 3.41)`\
  `▶ Q4 (Paret. Cíl) : 64.7%  [OK]  (Desired Ratio: 657 molekul splňuje VŠECHNY prahy)`\
  `  • ROCS Tcombo  : Průměr = 1.08 ± 0.14 (Max = 1.48, Threshold = 0.871)`\
  `  • QSAR pIC50   : Průměr = 7.82 ± 0.61 (Max = 9.12, Threshold = 7.50)`\
  `--------------------------------------------------------------------------------`\
  `[OK] STAV POLITIKY: Stabilní gradientní postup, nulový výskyt alifatických monster.`
]

=== Kompletní spustitelný kód produkčního prostředí s integrovanými filtry

Následující skript demonstruje kompletní implementaci odolného prostředí, které nativně integruje ochranu proti všem popsaným patologiím:

#code_card(
  title: "Konfigurace produkčního prostředí odolného vůči reward hackingu",
  lang: "python",
  code: "from rdkit import Chem
from drugex.training.environment import DrugExEnvironment
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.interfaces import Scorer

class RingSizePenalizer(Scorer):
    \"\"\"Penalizuje makrocykly (> 7 atomů v kruhu) pro eliminaci Macrocyclic Hacking.\"\"\"
    def getScores(self, mols, frags=None):
        return [1.0 if (m and max([len(r) for r in Chem.GetSymmSSSR(m)] or [0]) <= 7) else 0.0 for m in mols]

# 1. Konformační generátor se závorami proti tukové hydře (RotBonds <= 8, HeavyAtoms <= 42)
conf_gen = RDKitConformerGenerator(max_conformers=40, max_rotatable_bonds=8, max_heavy_atoms=42, num_threads=1)
rocs_scorer = RDKitROCSScorer([\"ref.sdf\"], conformer_generator=conf_gen, metric=\"TanimotoCombo\", n_jobs=16)

# 2. Pěticílový stack chránící chemický prostor před všemi čtyřmi patologiemi
scorers = [rocs_scorer, Property(\"SA\"), Property(\"RotBonds\"), Property(\"QED\"), RingSizePenalizer(max_size=7)]
modifiers = [
    SmoothClippedScore(lower_x=0.70, upper_x=1.30), # ROCS Tcombo
    SmoothClippedScore(lower_x=4.50, upper_x=2.80), # SAScore (Tuková hydra)
    SmoothClippedScore(lower_x=10.0, upper_x=5.0),  # Rotovatelných vazeb (Flexibilita)
    None, None                                      # QED [0, 1] a RingSize [0, 1]
]
env = DrugExEnvironment(scorers=scorers, modifiers=modifiers, thresholds=[0.85, 0.40, 0.50, 0.60, 0.99])
# -> [DrugExEnvironment] Inicializováno 5 cílů s integrovanou ochranou proti patologiím"
)

Tímto uspořádáním je generátor neprodyšně uzavřen v mantinelech medicinálně relevantního chemického prostoru. Získává silný gradient vedoucí ke komplementárnímu 3D tvaru a farmakoforu, zatímco jakýkoliv pokus o geometrický či topologický hacking je penalizován v zárodku.

== 7.4 Souhrnná matice reprodukovatelných hyperparametrů & Hardwarové profily

Pro zaručení stoprocentní experimentální reprodukovatelnosti shrnuje následující autoritativní matice kompletní konfiguraci všech výpočetních a optimalizačních hyperparametrů napříč všemi fázemi pipeline:

#table(
  columns: (1.5fr, 1.8fr, 3fr),
  align: (left, left, left),
  table.header(
    [*Kategorie*],
    [*Hyperparametr*],
    [*Konfigurovaná hodnota & Zdůvodnění*]
  ),
  [Architektura sítě], [Typ modelu], [`SequenceRNN` (3x LSTM buňka)],
  [], [Skrytá dimenze (Hidden Size)], [$1024$ neuronů na vrstvu],
  [], [Dropout regularizace], [$0.20$ (prevence overfittingu sekvenčního modelu)],
  [], [Velikost slovníku tokenů], [$97$ kanonických chemických tokenů],
  [Pre-training & Fine-tuning], [Trénovací korpus], [Papyrus v05.5 (~1.5M standardizovaných struktur)],
  [], [Optimalizační algoritmus], [Adam ($beta_1 = 0.9, beta_2 = 0.999$, weight decay $= 10^(-5)$)],
  [], [Learning rate (Pre-training)], [$10^(-3)$ (adaptivní pokles při stagnaci validační ztráty)],
  [], [Learning rate (Fine-tuning)], [$10^(-4)$ (konzervativní posun vah na polyfarmakologickém setu)],
  [], [Velikost dávky (Batch Size)], [$512$ (pre-training) / $64$ (jemné cílové ladění)],
  [Reinforcement Learning (RL)], [Počet trénovacích epoch], [$50 - 100$ epoch (s časným ukončením)],
  [], [Vzorkování na epochu], [$1024$ de novo vygenerovaných molekul],
  [], [Explorační faktor $epsilon$], [$0.15 - 0.20$ (vzorkování z mutační sítě $pi_0$)],
  [], [Paretovské schéma odměny], [`ParetoCrowdingDistance` (PCD) s nedominovaným řazením],
  [], [Ořezání gradientů], [Max $L_2$ norma tenzoru gradientů $= 1.0$],
  [3D Geometrie & ROCS], [Konformační engine], [RDKit ETKDGv3 (empirické torzní potenciály z CSD)],
  [], [Max konformerů na molekulu], [$30 - 40$ konformerů (kompromis přesnosti a rychlosti)],
  [], [Max stereoizomerů], [$4$ stereoizomery na chirální skelet],
  [], [Strukturní limity], [Max $8$ rotovatelných vazeb, max $42$ těžkých atomů],
  [], [Kalibrovaný práh $T_("combo")$], [$0.871$ (maximalizace Youdenova indexu na CCR2 DUD-E)],
  [], [Hierarchické síto USRCAT], [Práh $tau_("USR") = 0.58$ (odstraňuje 80 % tvarových zmetků za 0.18 ms)]
)

=== Hardwarové profily: Od lokální pracovní stanice (RTX 4080) po HPC klastr

Výpočetní profil DrugEx kombinuje GPU-akcelerované autoregresní generování SMILES sekvencí s vysoce paralelním CPU vzorkováním konformací. Následující tabulka specifikuje optimální alokaci výpočetních zdrojů pro dva klíčové scénáře:

#table(
  columns: (1.5fr, 2fr, 2fr),
  align: (left, left, left),
  table.header(
    [*Komponenta / Nastavení*],
    [*Lokální stanice (NVIDIA RTX 4080 16 GB)*],
    [*Superpočítačový uzel (HPC A100 / H100 80 GB)*]
  ),
  [Primární akcelerátor], [1x NVIDIA GeForce RTX 4080 (16 GB VRAM)], [1x NVIDIA A100 SXM4 (80 GB VRAM)],
  [CPU konfigurace], [16 až 24 fyzických jader (AMD Ryzen 9 / Intel i9)], [64 až 128 jader (AMD EPYC / Intel Xeon)],
  [Alokace paměti RAM], [32 GB až 64 GB DDR5], [256 GB až 512 GB ECC DDR4/DDR5],
  [Thread Pinning (`num_threads`)], [`num_threads=1` v `RDKitConformerGenerator`], [`num_threads=1` (striktně 1 vlákno na proces)],
  [Paralelizace skórovače], [`n_jobs=14` (ponechána 2 jádra pro OS a GUI)], [`n_jobs=60` (masivní multiprocesní škálování)],
  [Propustnost RL epochy], [~ 45 až 65 sekund na epochu (1024 molekul)], [~ 12 až 18 sekund na epochu (1024 molekul)],
  [Doporučený ROCS backend], [RDKit Shape Align / CDPKit], [CDPKit C++ / OpenEye fastROCS (GPU)]
)

#rule_box("Zlaté pravidlo škálování na osobní stanici s RTX 4080", [
  Na moderní lokální stanici s kartou RTX 4080 nepředstavuje úzké hrdlo grafická paměť VRAM (model `SequenceRNN` alokuje méně než $3.5$ GB), nýbrž *přehlcení CPU vláken (Thread Oversubscription)*. 
  
  Pokud v RDKit explicitně nenastavíte `num_threads=1` pro každého workera, systém spustí pro každou molekulu vícevláknový BLAS/OpenMP kód, který zahltí CPU kontextovými přepínači (_context switching_) a zpomalí výpočet až desetinásobně. Vždy ponechte 1–2 procesorová jádra volná pro operační systém (`n_jobs = os.cpu_count() - 2`).
])

