#import "../nature_theme.typ": *

#pagebreak()

= Klíč k Řešení Kvízů & Podrobná Odborná Odůvodnění

Tato kapitola je striktně oddělena od zadání testů, aby sloužila výzkumníkům a novým členům týmu k ověření nabytých znalostí po samostatném vypracování jednotlivých modulových kvízů. Pro každou otázku je uvedena správná volba doplněná hloubkovým chemoinformatickým a biofyzikálním zdůvodněním.

#let key_item(q_num, correct_opt, explanation) = block(
  width: 100%,
  stroke: (left: 3pt + nature_emerald),
  fill: rgb("#f0fdf4"),
  inset: (x: 12pt, y: 9pt),
  radius: (right: 4pt),
  spacing: 1.1em,
  breakable: false
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 10pt, fill: nature_emerald)[
    Otázka #q_num: Správná odpověď #correct_opt
  ]
  #v(0.2em)
  #text(size: 9.5pt, fill: rgb("#064e3b"))[#explanation]
]

== Modul 1: De Novo Generování & Molekulární Reprezentace

#key_item(
  1,
  "A",
  [SMILES je jedno-dimenzionální linearizace grafu. Při autoregresním generování token po tokenu může model vytvořit syntakticky neplatný řetězec (např. neuzavřený kruh či nespárovanou závorku). Grafové modely naproti tomu generují přímo uzly a hrany matice sousednosti, což eliminuje gramatické chyby formátu SMILES.]
)

#key_item(
  2,
  "B",
  [`VocSmiles` je tokenizační slovník DrugEx pro řetězce SMILES. Pomocí regulárního výrazu rozkládá molekulu na chemické tokeny, mapuje je na celočíselné tenzory pro neuronovou síť a spravuje řídicí tokeny `START` (začátek sekvence), `END` (konec sekvence) a `PAD` (zarovnání délky dávky).]
)

#key_item(
  3,
  "A",
  [Standardizace dat odstraňuje experimentální šum (soli, rozpouštědla, nestejné tautomery) a zaručuje, že model je trénován výhradně na kanonických strukturách. Tím se předchází situaci, kdy by se generátor učil chemicky nekonzistentní varianty téže sloučeniny.]
)

#key_item(
  4,
  "C",
  [`SequenceRNN` je autoregresní generátor. Pravděpodobnost generování tokenu v čase $t$ je podmíněna celou předchozí sekvencí a vnitřním skrytým stavem sítě: $P(x_t mid(|) x_(<t))$. Generování začíná tokenem `START` a pokračuje, dokud síť nevygeneruje token `END` nebo nedosáhne maximální povolené délky.]
)

#key_item(
  5,
  "B",
  [Pre-training na korpusu Papyrus (~1.5 milionu látek) naučí generátor obecnou distribuci organické chemie. Následný fine-tuning na desítkách až stovkách známých ligandů cíle (např. CCR2) posune pravděpodobnostní rozdělení generátoru k žádaným bioaktivním motivům.]
)

#key_item(
  6,
  "A",
  [Přepínač `-mt graph` (molecule type) v `drugex.dataset` aktivuje grafové kódování molekul, které vytvoří tenzory uzlů a hran pro model `GraphTransformer`.]
)

#key_item(
  7,
  "A",
  [`GraphTransformer` operuje přímo s grafovou strukturou, což zaručuje 100% syntaktickou validitu generovaných molekul a usnadňuje fixaci vybraných farmakoforových jader (scaffold-constrained generation).]
)

#key_item(
  8,
  "B",
  [Teplota $T$ škáluje logity před softmaxem: $P(x_i) = exp(z_i / T) / sum_j exp(z_j / T)$. Zvýšení teploty ($T > 1.0$) vyhlazuje pravděpodobnostní distribuci, což zvyšuje diverzitu návrhů, avšak při $T > 1.4$ roste riziko syntakticky neplatných sekvencí.]
)

#key_item(
  9,
  "C",
  [Generování nové sekvence začíná předáním řídicího tokenu `START` (případně `GO`) jako prvního vstupu do generátoru, z něhož model predikuje první chemický atom.]
)

#key_item(
  10,
  "B",
  [Soubory `.pkg` jsou serializované binární balíčky PyTorch obsahující tenzory vah (`state_dict`), hyperparametry architektury sítě a metadata slovníku nezbytná pro inferenci.]
)

== Modul 2: Multi-Objective Reinforcement Learning & Paretova Optimalita

#key_item(
  1,
  "A",
  [Optimalizace jediné metriky (např. pouhé maximalizace predikované afinity) vede k exploataci odměny (_Reward Hacking_): model nalezne chybu v QSAR aproximaci a začne generovat obří lipofilní monstr-struktury. Vícekriteriální optimalizace (MORL) vyvažuje bioaktivitu s tvarem a syntetickou přístupností.]
)

#key_item(
  2,
  "B",
  [Mutační síť ($pi_0$) představuje zmrazenou referenční politiku. Vzorkování s pravděpodobností $epsilon$ z $pi_0$ brání katastrofickému zapomínání obecného chemického prostoru a udržuje model v mantinelech reálných molekul.]
)

#key_item(
  3,
  "B",
  [Paretova dominance je exaktní matematický vztah: molekula $A$ dominuje molekulu $B$, pokud ve všech sledovaných kritériích dosahuje alespoň stejného skóre ($f_k(A) >= f_k(B)$) a v alespoň jednom kritériu je striktně lepší ($f_j(A) > f_j(B)$).]
)

#key_item(
  4,
  "A",
  [Metrika vzdálenosti nahloučení (_Crowding Distance_) měří hustotu sousedních řešení podél Paretovy fronty. Řešení v málo zaplněných oblastech dostávají vyšší odměnu, což motivuje model k objevování rozmanitých chemotypů namísto zacyklení v jediném shluku.]
)

#key_item(
  5,
  "B",
  [`SmoothClippedScore` s obrácenými prahy (`lower_x=5.0, upper_x=3.0`) představuje klesající sigmoidální funkci. Vzhledem k tomu, že nižší SAScore značí snazší syntézu, molekuly pod 3.0 obdrží plnou odměnu $1.0$, molekuly nad 5.0 odměnu $0.0$ a mezi těmito hodnotami je hladký gradient.]
)

#key_item(
  6,
  "B",
  [Všechny modifikátory skóre v DrugEx převádějí surové fyzikální či biologické veličiny do normalizovaného intervalu odměny $[0.0, 1.0]$, což umožňuje jejich korektní a vyvážené srovnání v Paretovském třídění.]
)

#key_item(
  7,
  "B",
  [Nevalidní molekuly obdrží v `DrugExEnvironment` automaticky nulovou odměnu ($0.0$) ve všech cílech, čímž je generátor v gradientním kroku penalizován a rychle se odnaučí generovat neplatné valenční struktury.]
)

#key_item(
  8,
  "A",
  [`QSPRpred` je specializovaná knihovna pro strojové učení v chemoinformatice, která je integrována do DrugEx přes `QSPRpredScorer` a umožňuje predikci biologických aktivit na základě Random Forest, SVM či neuronových sítí.]
)

== Modul 3: 3D Tvarové Porovnávání (ROCS), Flexibilní Cíle & IDP

#key_item(
  1,
  "A",
  [Metrika TanimotoCombo je součtem prostorové tvarové shody ($T_("shape") in [0, 1]$) a shody farmakoforových barevných center ($T_("color") in [0, 1]$). Maximální teoretická hodnota je $2.0$.]
)

#key_item(
  2,
  "A",
  [Intrinsicky neuspořádané proteiny (IDP) postrádají stabilní krystalickou vazebnou kapsu, na niž by bylo možné dokovat. ROCS představuje ligand-based přístup, který modeluje komplementární tvarovou a elektrostatickou obálku bioaktivních ligandů.]
)

#key_item(
  3,
  "A",
  [Algoritmus ETKDGv3 kombinuje distanční geometrii s empirickými preferencemi torzních úhlů extrahovanými z Cambridgeské strukturní databáze (CSD), což zajišťuje generování fyzikálně věrných 3D konformací.]
)

#key_item(
  4,
  "A",
  [Molekuly s velkým počtem rotovatelných vazeb ($> 10$) mají exponenciálně rozsáhlý konformační prostor. Jejich adekvátní vzorkování v RL smyčce je výpočetně neúnosné a v biologických systémech vykazují vysokou entropickou ztrátu při vazbě.]
)

#key_item(
  5,
  "B",
  [Konformační generátory v DrugEx využívají stereoisomerní enumeraci: pro každou sloučeninu s nedefinovanou stereochemií vygenerují až `max_isomers` enantiomerů a diastereoizomerů, pro které následně vzorkují 3D konformace.]
)

#key_item(
  6,
  "B",
  [Při paralelním běhu ($n_("jobs") > 1$) musí každý dílčí proces používat `num_threads = 1`, aby nedocházelo k CPU thread oversubscription a destrukci výpočetní propustnosti procesoru.]
)

#key_item(
  7,
  "A",
  [Supermolekula představuje prostorové sjednocení několika bioaktivních ligandů do jediného referenčního tělesa, což umožňuje postihnout více sub-kapes a vazebných orientací současně.]
)

#key_item(
  8,
  "A",
  [`ImplicitMillsDean` je standardní farmakoforové barevné silové pole v OpenEye ROCS definující donory, akceptory, hydrofobní a aromatická interakční centra.]
)

== Modul 4: Kompletní Architektura ROCS Backendů

#key_item(
  1,
  "A",
  [Při vzorkování z generátoru se často opakují identické molekuly. Deduplikace SMILES před generováním 3D konformací ušetří $40$ až $70 %$ výpočetního času v každé epoše.]
)

#key_item(
  2,
  "A",
  [Použití inicializátoru `multiprocessing.Pool(initializer=...)` zajistí, že objemné referenční 3D molekuly jsou do worker procesů odeslány pouze jednou při startu a neserializují se znovu s každou dílčí dávkou.]
)

#key_item(
  3,
  "A",
  [`CDPKit` je 100% open-source C++ framework s Python rozhraním. Poskytuje plnohodnotné 3D Gaussovské tvarové porovnávání bez nutnosti drahých komerčních licencí OpenEye.]
)

#key_item(
  4,
  "A",
  [Třídní kontext `@dataclass CDPKitWorkerContext` čistě zapouzdřuje stav worker procesu a eliminuje riziko souběhů (race conditions) spojené s modulovými globálními proměnnými.]
)

#key_item(
  5,
  "A",
  [`OpenEyeROCSScorer` spouští oficiální optimalizovanou binárku `rocs` přes `subprocess.run` a následně parsuje vygenerovaný tabulkový TSV report.]
)

#key_item(
  6,
  "A",
  [Soubory `.sq` (Shape Query) představují specializovaný formát OpenEye vROCS obsahující definice tvarových a barevných těles včetně poloměrů a vah.]
)

#key_item(
  7,
  "A",
  [`RDKitROCSScorer` disponuje sanitační metodou `_ensure_reference_conformers`: pokud referenční SDF postrádá 3D souřadnice, automaticky provede `AddHs` a ETKDGv3 embedding.]
)

#key_item(
  8,
  "A",
  [Skupinové skórování umožňuje vyhodnocovat kompatibilitu s více vazebnými módy nebo alosterickými místy a vracet samostatný sloupec skóre pro každou skupinu referencí.]
)

== Modul 5: Experimentální Pipeline (CCR2 & IDP)

#key_item(
  1,
  "A",
  [Analýza prahů (`threshold_analysis.py`) identifikuje optimální bod na ROC křivce (Youdenův index), který maximalizuje oddělení skutečně aktivních ligandů od decoyů.]
)

#key_item(
  2,
  "A",
  [Youdenův index je definován jako $J = "Senzitivita" + "Specificita" - 1 = "TPR" - "FPR"$. Bod maximalizující $J$ představuje optimální kompromis mezi záchytem aktivních látek a odmítnutím decoyů.]
)

#key_item(
  3,
  "A",
  [V benchmarku CCR2 stanovila Youdenova analýza optimální dělící hranici na hodnotu $0.871$ TanimotoCombo skóre.]
)

#key_item(
  4,
  "A",
  [Metrika `desired_ratio` vyjadřuje podíl vygenerovaných struktur v dané epoše, které současně splňují stanovené prahy ve všech optimalizovaných kritériích současně.]
)

#key_item(
  5,
  "A",
  [Základní prostředí v ROCS tutoriálu integruje `RDKitROCSScorer` pro 3D tvarovou shodu a `Property('SA')` se sigmoidálním modifikátorem pro syntetickou dostupnost.]
)

#key_item(
  6,
  "A",
  [Čtveřice metrik Validita ($V$), Unikátnost ($U$), Novost ($N$) a Interní diverzita ($"Div"$) představuje mezinárodní standard pro komplexní hodnocení de novo generativních knihoven.]
)

#key_item(
  7,
  "A",
  [Test vnitřní podobnosti (Self-Similarity) ověřuje integritu pipeline: referenční molekuly otestované proti sobě samým musí dosáhnout vysokého skóre (blízkého $2.0$), což prokazuje správnost geometrie.]
)

#key_item(
  8,
  "A",
  [Pro urychlení tréninku na CPU je nejúčinnější snížit počet konformerů (`MAX_CONFORMERS=20-30`), zmenšit vzorkovací dávku (`RL_N_SAMPLES=500`) a striktně vynutit `num_threads=1` pro každý worker.]
)

== Modul 6: Fragmentový Návrh, Scaffolding & Produkční CLI / Slurm

#key_item(
  1,
  "A",
  [Algoritmus BRICS štěpí molekuly na $16$ typech synteticky robustních vazeb (např. amidy, estery, ethery, C-C vazby) a nahrazuje je číslovanými exit-vektory [1\*] až [16\*].]
)

#key_item(
  2,
  "A",
  [`FragSequenceExplorer` a `FragGraphExplorer` umožňují scaffold-based de novo design: fixují zadané nosné jádro molekuly a generují pouze variabilní substituenty nebo linkery.]
)

#key_item(
  3,
  "A",
  [Přepínač `-tm RL` (training mode = Reinforcement Learning) aktivuje Reinforcement Learning (RL) řízený vícekriteriálním prostředím skórovačů.]
)

#key_item(
  4,
  "A",
  [Parametr `-pr` specifikuje cestu k fixní mutační síti (Prior), která stabilizuje politiku agenta a brání zhroucení chemické gramatiky.]
)

#key_item(
  5,
  "A",
  [Nástroje OpenEye vyžadují nastavení proměnné prostředí `export OE_LICENSE=/cesta/k/oe_license.txt` ukazující na platný licenční soubor.]
)

#key_item(
  6,
  "A",
  [Direktiva `#SBATCH --gres=gpu:1` sdělí plánovači úloh Slurm, aby pro danou dávkovou úlohu vyhradil 1 dedikovanou grafickou kartu.]
)

#key_item(
  7,
  "A",
  [Při opakovaném spuštění `drugex.dataset` přesune systém předchozí verze souborů do automaticky vytvořených záložních složek `backup_1`, `backup_2` atd.]
)

#key_item(
  8,
  "A",
  [Příkaz `python -m drugex.generate -b tutorial/CLI/examples -g arl_graph_trans_RL -n 1000 -gpu 0` provede vzorkování 1~000 molekul z modelu a uloží je do tabulky TSV včetně skóre.]
)
