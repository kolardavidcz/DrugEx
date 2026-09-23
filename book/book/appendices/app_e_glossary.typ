#import "../nature_theme.typ": *

= Příloha E: Autoritativní Slovník Chemoinformatických Pojmů & Metrik

Tento glosář definuje 26 klíčových pojmů z oblasti algoritmické chemoinformatiky, de novo návrhu léčiv, 3D tvarového porovnávání a strojového učení. V souladu s lingvistickými a odbornými standardy ÚOCHB AV ČR a VŠCHT Praha se striktně vyhýbá nevhodným kalkům a důsledně užívá odborné termíny *ensemble konformací* nebo *konformační pool*.

#v(1em)

#let badge_like(txt) = box(
  fill: rgb("#e2e8f0"),
  inset: (x: 6pt, y: 3pt),
  radius: 3pt
)[#text(size: 8pt, font: "IBM Plex Sans", weight: "bold", fill: nature_slate)[#txt]]

#let term_entry(name, en_name, category, formula: none, def, target: none) = block(
  width: 100%,
  stroke: (left: 3pt + nature_blue),
  fill: rgb("#f8fafc"),
  inset: (x: 12pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.2em,
  breakable: false
)[
  #grid(
    columns: (1fr, auto),
    [
      #text(font: "IBM Plex Sans", weight: "bold", size: 11pt, fill: nature_navy)[#name] \
      #text(size: 9pt, fill: nature_gray)[_#en_name _]
    ],
    [
      #badge_like[#category]
    ]
  )
  #if formula != none [
    #v(0.3em)
    #align(center)[#formula]
  ]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#1e293b"))[#def]
  #if target != none [
    #v(0.3em)
    #text(size: 9pt, weight: "medium", fill: nature_teal)[🎯 Cílové rozmezí v DrugEx: #target]
  ]
]

#term_entry(
  "1. TanimotoCombo",
  "Tanimoto Combo Score",
  "3D ROCS & Geometrie",
  formula: $T_("combo") = T_("shape") + T_("color") = frac(I_(A B), I_(A A) + I_(B B) - I_(A B)) + frac(C_(A B), C_(A A) + C_(B B) - C_(A B))$,
  "Souhrnná metrika 3D molekulární podobnosti v metodě ROCS. Kombinuje shodu Gaussovského sterického objemu (Shape, [0, 1]) a shodu farmakoforových barevných center (Color, [0, 1]). Dosahuje maximální teoretické hodnoty 2.0 pro identické molekuly v identické konformaci.",
  target: "> 0.85 pro smysluplný vazebný překryv; > 1.20 pro vysoce komplementární kandidáty."
)

#term_entry(
  "2. Intrinsicky neuspořádané proteiny (IDP)",
  "Intrinsically Disordered Proteins",
  "Biofyzika & Cíle",
  "Biologické makromolekuly postrádající stabilní terciární strukturu, které existují jako dynamický rovnovážný konformační pool (ensemble konformací). Tradiční rigidní dokování na IDP selhává, protože neexistuje stálá vazebná kapsa. Tvarové porovnávání (ROCS) zde slouží jako ligand-based alternativa modelující komplementární prostorovou obálku bioaktivních konformací.",
  target: "Metodologická doména pro flexibilní cíle a indukované vazebné přechody."
)

#term_entry(
  "3. Paretovská vzdálenost nahloučení (PCD)",
  "Pareto Crowding Distance",
  "Algoritmy & AI",
  formula: $d_i = sum_(m=1)^M frac(f_m (i+1) - f_m (i-1), f_m^("max") - f_m^("min"))$,
  "Metrika hustoty řešení v objektivním prostoru převzatá z evolučního algoritmu NSGA-II a adaptovaná v DrugExu pro REINFORCE gradient. Odměňuje molekuly nacházející se v méně zaplněných oblastech Paretovy fronty, čímž brání kolapsu chemické diverzity (Mode Collapse). Hraniční extrémy mají přiřazenu vzdálenost nekonečno.",
  target: "Normalizováno do vah odměn R(x) v intervalu [0, 1]."
)

#term_entry(
  "4. Reinforcement Learning gradientem politiky (REINFORCE)",
  "Policy Gradient Reinforcement Learning",
  "Algoritmy & AI",
  formula: $nabla_theta J(theta) = bb(E)_(pi_theta) [sum_(t=1)^T R(m) dot nabla_theta log pi_theta (a_t mid(|) s_t)]$,
  "Základní algoritmus Reinforcement Learning (RL) v DrugExu. Aktualizuje parametry theta neuronového generátoru ve směru, který zvyšuje pravděpodobnost generování tokenů vedoucích k molekulám s vysokou vícekriteriální odměnou R(m).",
  target: "Typický learning rate v DrugEx: 1e-4 až 2e-4."
)

#term_entry(
  "5. Duální architektura Agent & Prior",
  "Prior and Agent Dual-Network Architecture",
  "Algoritmy & AI",
  formula: $cal(L)(theta) = -bb(E) [R(x) log pi_theta (x)] + beta dot D_("KL")(pi_theta parallel pi_0)$,
  "Dvousíťové uspořádání v DrugExu. Prior síť (pi_0) je zmrazená síť předtrénovaná na obecné chemické databázi (Papyrus/ChEMBL), která uchovává znalost chemické gramatiky. Agent (pi_theta) je učená síť optimalizující odměny. Penalizace Kullback-Leiblerovy divergence drží Agenta v mantinelech synteticky realizovatelné chemie.",
  target: "Explorační poměr epsilon = 0.15 až 0.25; koeficient beta = 0.01 až 0.05."
)

#term_entry(
  "6. SAScore (Syntetická dostupnost)",
  "Synthetic Accessibility Score",
  "Chemoinformatika",
  formula: $"SAScore" = "FragmentScore" - "ComplexityPenalty"$,
  "Metrika dle Ertla a Schuffenhauera hodnotící snadnost chemické syntézy molekuly. Vychází z frekvenční analýzy fragmentů v databázi PubChem a penalizuje strukturální složitost (chiralitu, spiro atomy, můstky, velké cykly). Škála 1 (triviální syntéza) až 10 (téměř nesyntetizovatelné).",
  target: "Požadavek pro praktickou syntézu v medicinální chemii: SAScore < 3.5; limit < 4.5."
)

#term_entry(
  "7. QED (Kvantitativní odhad léčivupodobnosti)",
  "Quantitative Estimate of Drug-likeness",
  "Chemoinformatika",
  formula: $"QED" = exp (1/k sum_(i=1)^k w_i ln d_i)$,
  "Kvantitativní odhad léčivupodobnosti navržený Bickertonem et al. Integruje 8 molekulárních vlastností (MW, logP, HBD, HBA, PSA, počet rotovatelných vazeb, počet aromatických kruhů a strukturní alerty) pomocí asymetrických žádoucích funkcí do jediného skóre v intervalu [0, 1].",
  target: "Pro de novo kandidáty QED > 0.60; ideálně > 0.75."
)

#term_entry(
  "8. Algoritmus ETKDGv3",
  "Experimental Torsion Knowledge Distance Geometry v3",
  "3D ROCS & Geometrie",
  "Algoritmus generování 3D konformerů v RDKit vyvinutý Rinikerem a Landrumem. Kombinuje distanční geometrii s empirickými preferencemi torzních úhlů extrahovanými z Cambridgeské strukturní databáze (CSD). Verze v3 přináší vylepšené vzorkování pro malé kruhy a makrocykly.",
  target: "Doporučeno 30–50 konformerů na molekulu pro spolehlivý 3D ROCS scoring."
)

#term_entry(
  "9. SmoothClippedScore",
  "Smooth Clipped Score Modifier",
  "Algoritmy & AI",
  formula: $S(x) = frac(1, 1 + 10^(alpha (x - x_0)))$,
  "Nelineární transformace v DrugExu převádějící fyzikálně-chemickou vlastnost (např. SAScore, MW, LogP) do normalizované odměny [0, 1]. Zabraňuje tomu, aby byl model nekonečně odměňován za extrémní honbu za čísly (např. triviální molekuly s MW 80) a vyhlazuje gradienty kolem požadovaného pásma.",
  target: "Pro SAScore: lower_x = 4.5 (odměna 0), upper_x = 2.8 (odměna 1)."
)

#term_entry(
  "10. BRICS (Retrosyntetická fragmentace)",
  "Breaking of Retrosynthetically Interesting Chemical Substructures",
  "Chemoinformatika",
  "Algoritmus retrosyntetického štěpení molekul navržený Degenem et al. Identifikuje 16 strategických typů chemických vazeb (např. amidy, estery, ethery, C-C vazby) a nahrazuje je syntonovými exit-vektory [1*] až [16*]. Umožňuje fragment-based generování s garancí chemické valenční proveditelnosti.",
  target: "Základ fragmentových modelů v DrugEx (SmilesFragDataSet, VocFrag)."
)

#term_entry(
  "11. Youdenův index (J)",
  "Youden's J Statistic (Informedness)",
  "Metodologie & Validace",
  formula: $J = "Senzitivita" + "Specificita" - 1 = "TPR" - "FPR"$,
  "Statistická metrika využívaná v ROC analýze pro objektivní stanovení dělícího prahu (threshold) mezi aktivními látkami a decoyi. Bod na ROC křivce, kde J dosahuje maxima, představuje optimální kompromis minimalizující falešnou pozitivitu i falešnou negativitu.",
  target: "Umožňuje kalibrovat práh TanimotoCombo (např. 0.871 pro CCR2) ve skriptu threshold_analysis.py."
)

#term_entry(
  "12. VocSmiles (Slovník tokenů)",
  "SMILES Vocabulary & Tokenizer Contract",
  "Chemoinformatika",
  "Tokenizační kontrakt v DrugExu definovaný regulárním výrazem rozdělujícím 1D SMILES na atomové symboly (včetně dvoupísmenných Br, Cl v hranatých závorkách i bez), vazby, větvení a izotopové značky. Zajišťuje invertibilní mapování mezi řetězcem a tenzorem celočíselných tokenů.",
  target: "Typická velikost slovníku: 60 až 100 tokenů."
)

#term_entry(
  "13. MolVS Standardizace",
  "Molecule Validation and Standardization",
  "Chemoinformatika",
  "Standardizační knihovna integrovaná v přípravě dat DrugExu. Odstraňuje anorganické soli a solváty (desalting), neutralizuje formální náboje, převádí funkční skupiny na kanonické tautomery a validuje valenční pravidla před tréninkem neuronových sítí.",
  target: "Povinný krok datové hygieny (drugex.data.processing)."
)

#term_entry(
  "14. Papyrus Databáze",
  "Papyrus Bioactivity Dataset",
  "Datové zdroje",
  "Rozsáhlá kurátorská databáze bioaktivních sloučenin (verze 05.5 obsahuje ~1,5 milionu látek s více než 60 miliony bioaktivit) sloučená z ChEMBL a BindingDB s harmonizovanými měřeními a stereochemií. Slouží jako referenční datový korpus pro předtrénování Prior sítí v DrugExu.",
  target: "Výchozí stav pro transfer learning a fine-tuning na specifický cíl."
)

#term_entry(
  "15. Decoy Sloučeniny",
  "Decoy Molecules (DUD-E)",
  "Metodologie & Validace",
  "Molekuly navržené tak, aby měly podobné fyzikálně-chemické vlastnosti jako známé aktivní ligandy (molekulovou hmotnost, logP, počet rotovatelných vazeb), ale zcela odlišnou 2D topologii a farmakoforové uspořádání. Používají se pro nestrannou kalibraci ROC křivek a validaci rozlišovací schopnosti ROCS scoreru.",
  target: "Poměr v benchmarku typicky 1 aktivní ku 30–50 decoyům."
)

#term_entry(
  "16. Kolaps diverzity generátoru (Mode Collapse)",
  "Mode Collapse Pathology",
  "Algoritmy & AI",
  "Patologický stav při Reinforcement Learningu (RL), kdy generátor ztratí chemickou rozmanitost a začne cyklicky produkovat jedinou nebo malou sadu struktur s vysokým skóre. V DrugExu je indikován poklesem unikátnosti (uniqueness < 30 %). Protilékem je zvýšení exploračního parametru epsilon a použití Crowding Distance.",
  target: "Cílová unikátnost v generované knihovně > 85 %."
)

#term_entry(
  "17. Obcházení odměny (Reward Hacking)",
  "Reward Hacking / Metric Exploitation",
  "Algoritmy & AI",
  "Situace, kdy optimalizační algoritmus nalezne způsob, jak dosáhnout vysoké matematické odměny generováním chemických anomálií (např. extrémně dlouhých alifatických řetězců při nekontrolované odměně za LogP, nebo makrocyklů při čistém objemovém ROCS bez SA filtru). Řešením je vícekriteriální Paretovo vyvážení se striktním SAScore.",
  target: "Eliminováno správnou kompozicí DrugExEnvironment."
)

#term_entry(
  "18. PAINS (Pan-Assay Interference Compounds)",
  "Pan-Assay Interference Compounds",
  "Chemoinformatika",
  "Substruktury vykazující falešně pozitivní aktivitu v biochemických testech (např. skrze reaktivní kovalentní modifikace, agregaci v roztoku, fluorescenci nebo produkci peroxidu vodíku, např. chinony, rhodaniny). V DrugExu jsou filtrovány přes RDKit FilterCatalog.",
  target: "Všechny finální kandidátní molekuly musí projít PAINS a Brenk filtrem s 0 alerty."
)

#term_entry(
  "19. OpenEye ROCS",
  "OpenEye Rapid Overlay of Chemical Structures",
  "3D ROCS & Geometrie",
  "Průmyslový zlatý standard v 3D tvarovém porovnávání vyvinutý OpenEye Scientific Software. Využívá analytickou reprezentaci atomů pomocí hladkých Gaussovských funkcí a optimalizovaný Color Force Field. V DrugExu představuje vysoce výkonný backend, vyžadující však komerční licenci.",
  target: "Až 100x rychlejší než čisté Python implementace."
)

#term_entry(
  "20. CDPKit (CDPL)",
  "Chemical Data Processing Toolkit",
  "3D ROCS & Geometrie",
  "Výkonný open-source C++ framework s Python vazbami pro chemoinformatiku a 3D farmakoforové modelování. V DrugExu slouží jako rychlý bezlicenční ekvivalent ROCS s plnou podporou 3D Gaussovského tvaru i barevných farmakoforů (cdpl.shape a cdpl.pharm).",
  target: "Ideální volba pro akademické klastry bez OpenEye licencí."
)

#term_entry(
  "21. RDKit rdShapeAlign",
  "RDKit Shape Alignment & Distance",
  "3D ROCS & Geometrie",
  formula: $D = "ShapeTanimotoDist"(A, B) ==> T_("shape") = 1.0 - D$,
  "Nativní open-source modul v RDKit pro výpočet tvarové nepodobnosti pomocí Gaussovského překryvu. V DrugExu je obalen do RDKitROCSScoreru, který doplňuje farmakoforovou shodu přes FeatureMap.",
  target: "Plně přenositelné řešení bez externích knihoven."
)

#term_entry(
  "22. Algoritmus NSGA-II",
  "Non-dominated Sorting Genetic Algorithm II",
  "Algoritmy & AI",
  "Klasický evoluční algoritmus pro vícekriteriální optimalizaci navržený Debem et al. Zavedl rychlé nedominované řazení do vrstev F1, F2, F3 a crowding distance pro zachování diverzity. Principy tohoto algoritmu jsou jádrem vícekriteriálního učení v DrugExu.",
  target: "Zajišťuje rovnoměrné pokrytí Paretovy fronty."
)

#term_entry(
  "23. Reprezentace SELFIES",
  "Self-Referencing Embedded Strings",
  "Chemoinformatika",
  "Alternativní 1D řetězcová reprezentace chemických molekul vyvinutá Aspuru-Guzikem a kolegy. Na rozdíl od SMILES je 100 % všech vygenerovaných kombinací SELFIES tokenů syntakticky i valenčně chemicky platných, což eliminuje chybovost generátoru na syntaktické úrovni.",
  target: "Validita generování 100 % (vůči ~95 % u SMILES bez pokročilé filtrace)."
)

#term_entry(
  "24. Paretova Fronta (Pareto Front)",
  "Pareto Optimal Front",
  "Algoritmy & AI",
  formula: $x^* in cal(P) <==> not exists x : (forall i, f_i (x) >= f_i (x^*) and exists j, f_j (x) > f_j (x^*))$,
  "Množina všech nedominovaných řešení v mnohorozměrném prostoru kritérií. Žádnou vlastnost molekuly na Paretově frontě nelze zlepšit, aniž by došlo ke zhoršení alespoň jedné jiné vlastnosti. Představuje optimální kompromis mezi afinitou, tvarem a syntetizovatelností.",
  target: "Počet molekul ve Frontě 1 typicky 5–20 % z generované dávky."
)

#term_entry(
  "25. Interní Diverzita Knihovny (Div)",
  "Internal Tanimoto Diversity",
  "Chemoinformatika",
  formula: $"Div"(S) = 1 - frac(2, |S| (|S| - 1)) sum_(i < j) T(m_i, m_j)$,
  "Míra strukturní pestrosti vygenerované knihovny molekul S založená na průměrné Tanimoto vzdálenosti Morganových fingerprintů (ECFP4) mezi všemi dvojicemi. Vysoká hodnota indikuje široké prozkoumání chemického prostoru, zatímco hodnota blízká 0 značí kolaps diverzity.",
  target: "Pro smysluplnou screeningovou knihovnu Div > 0.70; ideálně > 0.80."
)

#term_entry(
  "26. DrugExEnvironment",
  "Multi-Objective DrugEx Environment",
  "Algoritmy & AI",
  "Centrální výpočetní komponenta platformy DrugEx, která sdružuje jednotlivé skórovače (scorers), jejich normalizační modifikátory (modifiers), prahové hodnoty (thresholds) a vyhodnocovací schéma (ParetoCrowdingDistance). V každé epoše přebírá vygenerované SMILES a vrací matici odměn R(x).",
  target: "Modulární stavebnice integrující ROCS, SAScore, QSAR i fyzikálně-chemické filtry."
)
