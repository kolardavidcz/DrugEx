#import "../nature_theme.typ": *

= Kvíz M5: Experimentální Pipeline pro Závěrečnou Práci (CCR2 & IDP)

Tento test prověřuje porozumění kompletní experimentální pipeline: stanovení prahů pomocí ROC analýzy a Youdenova indexu, sledování telemetrie tréninku (`desired_ratio`) a hodnocení chemických metrik vygenerovaných knihoven.

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
  [Jaký je hlavní cíl analýzy stanovení prahu (`threshold_analysis.py`) před spuštěním RL tréninku?],
  (
    [Nalézt optimální dělící práh TanimotoCombo skóre (pomocí ROC analýzy a Youdenova indexu), který maximalizuje oddělení skutečně aktivních ligandů od decoyů, aby RL model dostával fyzikálně relevantní odměnu.],
    [Zjistit, kolik paměti zabírají CSV soubory na disku.],
    [Změřit provozní teplotu grafické karty.],
    [Zkontrolovat pravopis v názvech chemických sloučenin.]
  )
)

#quiz_question(
  2,
  [Jak je definován Youdenův index $J$ v ROC analýze?],
  (
    [$J = "Senzitivita" ("TPR") + "Specificita" ("TNR") - 1 = "TPR" - "FPR"$],
    [$J = "Precision" times "Recall"$],
    [$J = "TP" / ("TP" + "FP")$],
    [$J = "Area Under Curve" ("AUC")$]
  )
)

#quiz_question(
  3,
  [Jaká hodnota `ROCS_THRESHOLD` byla odvozena jako optimální v benchmarku CCR2 (`tutorial/advanced/rocs/config.py`)?],
  (
    [`0.871`],
    [`0.100`],
    [`1.950`],
    [`0.000`]
  )
)

#quiz_question(
  4,
  [Co představuje metrika `desired_ratio` sledovaná v průběhu RL tréninku v DrugEx?],
  (
    [Podíl vygenerovaných molekul v dané epoše, které současně splňují stanovené prahové hodnoty ve všech optimalizovaných kritériích (např. $"ROCS" >= 0.871$ a $"SAScore" >= 0.1$).],
    [Procento molekul obsahujících alespoň jedno benzenové jádro.],
    [Počet tenzorových GPU výpočtů za sekundu.],
    [Poměr mezi velikostí trénovací a validační sady.]
  )
)

#quiz_question(
  5,
  [Které dva skórovače jsou konfigurovány v základním prostředí `create_rdkit_environment()` v ROCS tutoriálu?],
  (
    [`RDKitROCSScorer` (s referenčním souborem `CCR2_reference_ligands.sdf`) a `Property('SA')` (se `SmoothClippedScore` modifikátorem).],
    [Pouze barevný histogram a počet atomů uhlíku.],
    [Dokovací software AutoDock Vina a kalkulátor ceny syntézy.],
    [Kalkulátor rozpustnosti a prediktor toxicity LD50.]
  )
)

#quiz_question(
  6,
  [Jaké metriky chemické kvality se standardně vyhodnocují na sadě 10~000 molekul vygenerovaných z natrénovaného modelu (`generate_molecules.py`)?],
  (
    [Chemická validita (procento platných SMILES), unikátnost (procento unikátních struktur), novost (novelty vůči Papyrus/ChEMBL) a interní diverzita (průměrná Tanimoto distance).],
    [Pouze barva roztoku a teplota tání.],
    [Počet znaků v textovém souboru TSV.],
    [Pouze datum a čas vytvoření souboru.]
  )
)

#quiz_question(
  7,
  [Co znamená, když v analýze prahů dosáhnou referenční ligandy v testu vnitřní podobnosti (Self-Similarity) vysokého skóre ($> 1.5$)?],
  (
    [Znamená to, že skórovač i konformační generátor fungují správně a referenční ligandy mají samy se sebou vysokou tvarovou a farmakoforovou shodu, což validuje geometrické nastavení.],
    [Znamená to, že soubor SDF je poškozený.],
    [Znamená to, že model má příliš mnoho parametrů.],
    [Self-similarity skóre nemá v praxi žádný význam.]
  )
)

#quiz_question(
  8,
  [Jaký je doporučený postup, pokud je trénink RL příliš pomalý na běžné pracovní stanici s CPU?],
  (
    [Snížit `MAX_CONFORMERS` (např. z 50 na 20–30), snížit `RL_N_SAMPLES` (např. z 1000 na 500) a využít paralelismus `n_jobs=-1` s `num_threads=1`.],
    [Vypnout operační systém a spustit kód v BIOSu.],
    [Smazat referenční SDF soubor.],
    [Zvýšit počet konformerů na 500.]
  )
)
