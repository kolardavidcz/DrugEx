#import "../nature_theme.typ": *

= Příloha C: Vzorový Text Metodiky pro Publikaci & Technickou Dokumentaci

Tato příloha obsahuje standardizovaný, recenzním řízením prověřený text kapitoly *Výpočetní metody* (_Computational Methods_), který slouží jako rigorózní metodologický vzor pro vědecké publikace v mezinárodních impaktovaných časopisech, technické zprávy a interní dokumentaci lékového výzkumu.

Text je stylizován v exaktním vědeckém jazyce a je doplněn souhrnnou tabulkou všech reprodukovatelných výpočetních hyperparametrů.

== C.1 Vzorový text metodiky (Šablona pro vědeckou publikaci)

#block(
  fill: rgb("#f8fafc"),
  inset: 14pt,
  stroke: 1pt + rgb("#cbd5e1"),
  radius: 4pt
)[
#text(size: 10pt)[
*Výpočetní metody (Computational Methods)*

*1. Datové zdroje a molekulární standardizace* \
Pro předtrénování obecného generátoru chemických struktur byla využita rozsáhlá databáze bioaktivních látek Papyrus v05.5 (Bequignon et al., 2023) zahrnující přibližně $1.5$ milionu standardizovaných organických molekul. Vstupní data byla podrobena striktnímu standardizačnímu protokolu integrovanému v modulu `drugex.data.processing`: anorganické soli a solváty byly odstraněny (desalting), formální náboje byly neutralizovány (vyjma esenciálních kvarterních solí), funkční skupiny byly převedeny na kanonické tautomery a molekuly byly validovány podle valenčních pravidel RDKit a knihovny MolVS. Řetězce SMILES byly linearizovány a tokenizovány pomocí regulárního výrazu třídy `VocSmiles` do podoby celočíselných indexů o velikosti slovníku $97$ unikátních chemických tokenů.

*2. Architektury generátorů a transferové učení (Transfer Learning)* \
Jako základní generativní model byla využita rekurentní neuronová síť `SequenceRNN` tvořená třemi vrstvami LSTM buněk s $1024$ skrytými jednotkami v každé vrstvě a dropout regularizací ($0.2$). Model byl nejprve předtrénován na databázi Papyrus po dobu $100$ epoch s optimalizátorem Adam ($lr = 10^(-3)$, velikost dávky $512$). Následně proběhl cílově orientovaný fine-tuning na sadě experimentálně ověřených aktivních ligandů zkoumaného receptoru ($N = 75$ sloučenin pro benchmark CCR2) po dobu $100$ epoch ($lr = 10^(-4)$, batch size $64$, patience $30$). Tím vznikla specializovaná mutační síť ($pi_0$), schopná generovat strukturní motivy relevantní pro daný biologický cíl.

*3. Skórovací prostředí (DrugExEnvironment) a 3D tvarové porovnávání* \
Optimalizační prostředí integrovalo dvě komplementární skórovací funkce:
+ *3D Tvarové a farmakoforové hodnocení (ROCS)*: Prostorová komplementarita generovaných molekul vůči krystalografickým referenčním ligandům byla vyhodnocována pomocí modulu `RDKitROCSScorer` s metrikou TanimotoCombo ($T_("combo") = T_("shape") + T_("color") in [0, 2]$). 3D konformace byly pro každou molekulu vzorkovány algoritmem ETKDGv3 (Riniker & Landrum, 2015) s limitem maximálně $40$ konformerů a $4$ stereoizomerů na sloučeninu. Pro eliminaci výpočetně neúnosných a "lepivých" struktur byly filtrovány molekuly s více než $10$ rotovatelnými vazbami a více než $45$ těžkými atomy. Optimální dělící práh žádoucích molekul ($T_("combo") >= 0.871$) byl stanoven na základě ROC analýzy a maximalizace Youdenova indexu ($J = "TPR" - "FPR"$) na validační sadě aktivních látek a 500 decoyů generovaných v souladu s metodologií DUD-E.
+ *Syntetická přístupnost (SAScore)*: Pro potlačení synteticky nerealizovatelných motivů byl zapojen skórovač syntetické dostupnosti dle Ertla a Schuffenhauera (2009). Surové hodnoty SAScore byly transformovány nelineární klesající sigmoidou `SmoothClippedScore(lower_x=4.5, upper_x=2.8)` do intervalu odměn $[0.0, 1.0]$.

*4. Multi-Objective Reinforcement Learning (MORL)* \
Kompromis mezi 3D tvarovou shodou a syntetickou dostupností byl řízen schématem Paretovské vzdálenosti nahloučení (_Pareto Crowding Distance_, PCD; Liu et al., 2021). Trénink probíhal v rámci algoritmu REINFORCE po dobu $50$ epoch s generováním $1024$ molekul v každé epoše. Aby se zabránilo kolapsu chemické rozmanitosti (Mode Collapse), byla uplatněna smíšená politika: s pravděpodobností $1 - epsilon = 0.8$ byly molekuly vzorkovány z učeného agenta ($pi_theta$) a s pravděpodobností $epsilon = 0.2$ z fixní mutační sítě ($pi_0$). Gradienty parametrů byly zastropovány normou $1.0$ (gradient clipping).

*5. Vzorkování a filtrace kandidátních sloučenin* \
Z optimalizovaného modelu bylo finálně vygenerováno 10~000 struktur. Výsledná knihovna byla podrobena filtračnímu protokolu:
- Odstranění nevalidních SMILES a duplicitních molekul pro stanovení Validity ($V$) a Unikátnosti ($U$).
- Výpočet Novosti ($N$) definované jako podíl sloučenin nepřítomných v trénovací sadě Papyrus ani v sadě aktivních ligandů cíle.
- Stanovení interní diverzity knihovny ($"Div"$) na základě průměrné Tanimoto vzdálenosti Morganových fingerprintů (ECFP4, rádius 2, 2048 bitů) mezi všemi dvojicemi generovaných struktur.
- Odstranění struktur obsahujících reaktivní a nespecifické motivy pomocí filtrů PAINS a Brenk implementovaných v RDKit `FilterCatalog`.
]
]

== C.2 Souhrnná tabulka reprodukovatelných hyperparametrů

Následující tabulka uvádí kompletní sadu číselných parametrů použitých v experimentální pipeline, která slouží jako jednoznačná reference pro replikaci výsledků:

#table(
  columns: (2fr, 2fr, 3fr),
  align: (left, left, left),
  [Kategorie], [Hyperparametr], [Konfigurovaná hodnota & Zdůvodnění],
  [Architektura sítě], [Typ modelu], [`SequenceRNN` (3x LSTM buňka)],
  [], [Skrytá dimenze (Hidden Size)], [$1024$ neuronů na vrstvu],
  [], [Dropout], [$0.2$ (prevence overfittingu na sekvenci)],
  [], [Velikost slovníku], [$97$ tokenů (kanonický SMILES)],
  [Pre-training & Fine-tuning], [Trénovací korpus], [Papyrus v05.5 (~1.5M látek)],
  [], [Optimalizátor], [Adam ($beta_1 = 0.9, beta_2 = 0.999$)],
  [], [Learning rate (Pre-training)], [$10^(-3)$ (decay po 10 epochách)],
  [], [Learning rate (Fine-tuning)], [$10^(-4)$ (konzervativní adaptace na cíl)],
  [], [Batch size (Pre / Fine)], [$512$ / $64$ vzorků],
  [Reinforcement Learning (RL)], [Počet epoch], [$50$ epoch],
  [], [Vzorkování na epochu], [$1024$ generovaných molekul],
  [], [Explorační faktor $epsilon$], [$0.20$ (vzorkování z mutační sítě $pi_0$)],
  [], [Paretovské schéma], [`ParetoCrowdingDistance` (PCD)],
  [], [Gradient clipping], [Max $L_2$ norma = $1.0$],
  [3D ROCS Geometrie], [Konformační engine], [RDKit ETKDGv3 (fyzikální torze z CSD)],
  [], [Max konformerů na molekulu], [$40$ konformerů],
  [], [Max stereoisomerů], [$4$ stereoizomery (chiralita)],
  [], [Limity molekuly], [Max $10$ rot. vazeb, max $42$ těžkých atomů],
  [], [Youdenův práh $T_("combo")$], [$0.871$ (kalibrováno na CCR2 DUD-E)],
  [], [Paralelizace CPU], [`n_jobs=16, num_threads=1` (Thread Pinning)]
)
