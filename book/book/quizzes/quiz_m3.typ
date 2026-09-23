#import "../nature_theme.typ": *

= Kvíz M3: 3D Tvarové Porovnávání (ROCS), Flexibilní Cíle & IDP

Tento test hodnotí porozumění biofyzikálním principům flexibilních cílů a IDP, 3D Gaussovskému překryvu, metrikám TanimotoCombo, generování konformací pomocí distanční geometrie ETKDGv3 a stereoisomerní enumeraci.

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
  [Jak je matematicky definována metrika TanimotoCombo v metodě ROCS?],
  (
    [Jako prostý součet tvarové podobnosti ($T_("shape") in [0, 1]$) a farmakoforové barevné podobnosti ($T_("color") in [0, 1]$), celkově v rozsahu $[0, 2]$.],
    [Jako euklidovská vzdálenost těžišť obou molekul v angströmech.],
    [Jako podíl molekulových hmotností násobený hodnotou logP.],
    [Jako kosínová podobnost 2D Morganových fingerprintů.]
  )
)

#quiz_question(
  2,
  [Proč je metoda ROCS klíčová pro návrh ligandů u intrinsicky neuspořádaných proteinů (IDP) nebo vysoce flexibilních cílů?],
  (
    [Protože IDP nemají stabilní 3D krystalickou vazebnou kapsu pro molekulární dokování; ROCS umožňuje ligand-based přístup založený na tvaru známých aktivních sloučenin.],
    [Protože ROCS dokáže automaticky zkrystalizovat jakýkoliv IDP protein za pokojové teploty.],
    [Protože IDP vyžadují výhradně anorganické sloučeniny.],
    [ROCS se pro flexibilní cíle nehodí a je určen pouze pro rigidní krystaly.]
  )
)

#quiz_question(
  3,
  [K čemu slouží algoritmus ETKDGv3 v `RDKitConformerGenerator`?],
  (
    [Generuje energeticky a geometricky realistické 3D konformace molekul na základě experimentálních torzních úhlů z Cambridge Structural Database (CSD).],
    [Slouží k převodu řetězců SMILES do formátu FASTA.],
    [Provádí ab initio kvantově-chemické výpočty vazebných délek.],
    [Filtruje molekuly podle pravidla Lipinského.]
  )
)

#quiz_question(
  4,
  [Proč `RDKitConformerGenerator` a ostatní generátory v DrugEx filtrují molekuly s příliš velkým počtem rotovatelných vazeb (`max_rotatable_bonds`) a těžkých atomů (`max_heavy_atoms`)?],
  (
    [Protože molekuly s vysokou flexibilitou mají obrovský konformační prostor, jejich generování konformací je výpočetně neúnosně pomalé a mívají špatné farmakokinetické vlastnosti.],
    [Protože RDKit nepodporuje molekuly s více než 5 atomy.],
    [Protože rotovatelné vazby způsobují zhroucení tenzorů v PyTorch.],
    [Filtrování se provádí pouze kvůli formátu souboru SDF.]
  )
)

#quiz_question(
  5,
  [Jak řeší generátory konformací v DrugEx nedefinovanou stereochemii v molekulách?],
  (
    [Všechna chirální centra jsou smazána a nahrazena planárními vazbami.],
    [Pomocí stereoisomerní enumerace (`EnumerateStereoisomers` / `StereoisomerGenerator`) vygenerují až `max_isomers` stereoizomerů a pro každý vytvoří sadu 3D konformací.],
    [Stereochemie je v DrugEx ignorována.],
    [Při nalezení chirálního centra se molekula okamžitě zahodí.]
  )
)

#quiz_question(
  6,
  [Jaký parametr v `RDKitConformerGenerator(num_threads=...)` je doporučen při spuštění s paralelním skórováním (`n_jobs > 1`)?],
  (
    [`num_threads = 0`],
    [`num_threads = 1` (aby nedocházelo k CPU oversubscription a vzájemnému blokování procesů)],
    [`num_threads = 64`],
    [`num_threads = -1`]
  )
)

#quiz_question(
  7,
  [Co je to 'Supermolekula' (Supermolecule) v kontextu tvarového porovnávání?],
  (
    [Jediná kompozitní 3D struktura vzniklá sloučením klíčových farmakoforových rysů několika aktivních ligandů do jednoho referenčního tvaru.],
    [Polymer s molekulovou hmotností nad 100~000 Da.],
    [Komplex proteinu s DNA.],
    [Název pro model Sequence Transformer v DrugEx.]
  )
)

#quiz_question(
  8,
  [Které silové pole se standardně používá v OpenEye ROCS pro vyhodnocení shody farmakoforových barev?],
  (
    [`ImplicitMillsDean`],
    [`AMBER99`],
    [`CHARMM36`],
    [`OPLS-AA`]
  )
)
