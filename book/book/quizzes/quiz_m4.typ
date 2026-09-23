#import "../nature_theme.typ": *

= Kvíz M4: Kompletní Architektura ROCS Backendů (feature/rocs-scoring)

Tento test hodnotí detailní znalost softwarové architektury a implementace tří komplementárních backendů pro 3D tvarové porovnávání v DrugExu: RDKit, CDPKit a OpenEye.

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
  [Jak `RDKitROCSScorer` v modulu `drugex.training.scorers.rocs_rdkit` optimalizuje výkon při skórování dávky molekul z RL generátoru?],
  (
    [Provádí deduplikaci SMILES řetězců (`_deduplicate_smiles`), takže generování konformací a skórování proběhne pro identické generované molekuly pouze jednou a výsledek se namapuje zpět na všechny výskyty.],
    [Přeskakuje všechny molekuly obsahující dusík.],
    [Počítá skóre pouze z 2D strukturních vzorců bez 3D zarovnání.],
    [Používá náhodná čísla namísto prostorového zarovnání.]
  )
)

#quiz_question(
  2,
  [Jakým způsobem předává `RDKitROCSScorer` referenční molekuly do paralelních worker procesů?],
  (
    [Přes globální inicializátor `initializer=_rdkit_worker_init` v `multiprocessing.Pool`, což zajistí jednorázové předání neměnného stavu a eliminuje serializační overhead při každé dílčí úloze.],
    [Ukládáním do textového souboru na disk před každou jednotlivou molekulou.],
    [Přes externí HTTP REST API server.],
    [Referenční molekuly se v paralelním režimu vůbec nepoužívají.]
  )
)

#quiz_question(
  3,
  [Jaká je hlavní výhoda `CDPKitROCSScorer` (`rocs_cdpkit.py`) pro akademické a open-source projekty?],
  (
    [Je 100% open-source (bez nutnosti komerční licence OpenEye), využívá CDPL.Shape Gaussovské tvary a dosahuje vysoké rychlosti bez nutnosti spouštění externích CLI binárek.],
    [Nevyžaduje žádný procesor ani paměť RAM.],
    [Funguje výhradně v prostředí Google Colab.],
    [Automaticky syntetizuje sloučeniny v mokré laboratoři.]
  )
)

#quiz_question(
  4,
  [Jak řeší `CDPKitScoringWorker` sdílení kontextu mezi procesy bez použití nebezpečných modulových globálních proměnných?],
  (
    [Využívá třídní kontext zapouzdřený v `@dataclass CDPKitWorkerContext`, který se nastaví metodou `initialize()` při spuštění workeru v Poolu.],
    [Používá globální proměnnou sys.path.],
    [Předává data přes systémovou schránku operačního systému.],
    [Všechny proměnné se zapisují do registru Windows.]
  )
)

#quiz_question(
  5,
  [Jak komunikuje `OpenEyeROCSScorer` (`rocs_openeye.py`) s výpočetním jádrem ROCS?],
  (
    [Spouští oficiální OpenEye CLI binárku `rocs` přes `subprocess.run` s přepínači `-query`, `-dbase`, `-report one`, `-stats best` a `-nostructs` a následně parsuje vygenerovaný TSV report.],
    [Posílá dotazy přes WebSocket na cloudový server OpenEye.],
    [Používá emulátor DOSBox.],
    [Čte paměť grafické karty přímo přes rozhraní DirectX.]
  )
)

#quiz_question(
  6,
  [Jaký typ referenčních souborů podporuje výhradně OpenEye backend, ale nepodporují jej RDKit a CDPKit?],
  (
    [OpenEye VROCS Shape Query soubory s příponou `.sq` (obsahující předpočítané farmakoforové a tvarové rysy).],
    [Formát FASTA pro aminokyseliny.],
    [Formát PDB pro makromolekuly.],
    [Obyčejný tabulkový CSV soubor.]
  )
)

#quiz_question(
  7,
  [Co se stane, pokud do `RDKitROCSScorer` předáme referenční molekulu, která nemá předgenerované 3D konformace?],
  (
    [Metoda `_ensure_reference_conformers` automaticky vygeneruje 3D konformaci pomocí `AllChem.EmbedMolecule` s algoritmem ETKDGv3.],
    [Skórovač okamžitě havaruje s chybou Segmentation Fault.],
    [Referenční molekula je tiše ignorována a skóre bude vždy rovno 0.],
    [Molekula se automaticky převede na 2D vektorový obrázek SVG.]
  )
)

#quiz_question(
  8,
  [Jak funguje skupinové skórování (Group Definitions / Multi-reference), když do `RDKitROCSScorer` zadáme slovník referencí např. `{'orthosteric': ['ref1.sdf'], 'allosteric': ['ref2.sdf']}`?],
  (
    [Pro každou molekulu skórovač vrátí vektor skóre (pro každou skupinu maximální TanimotoCombo skóre ze všech referencí v dané skupině), což umožňuje multi-target / multi-site optimalizaci.],
    [Všechny reference se náhodně pomíchají a vrátí se jen jedno číslo.],
    [Skórovač akceptuje pouze první soubor a zbytek ignoruje.],
    [Slovníkový zápis referencí není v DrugEx povolen.]
  )
)
