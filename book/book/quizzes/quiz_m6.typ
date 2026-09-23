#import "../nature_theme.typ": *

= Kvíz M6: Fragmentový Návrh, Scaffolding & Produkční CLI / Slurm

Tento test ověřuje znalosti fragmentového a scaffold-based de novo designu (BRICS pravidla L1–L16, FragSequence/Graph explorátory), práce s rozhraním příkazové řádky DrugEx CLI a správy úloh v plánovači Slurm na superpočítačových klastrech.

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
  [Jak funguje fragmentační algoritmus BRICS (Breaks on Retrosynthetically Interesting Chemical Substructures) v DrugEx?],
  (
    [Štěpí molekulu na strategických chemických vazbách odpovídajících běžným organickým syntetickým reakcím (např. amidické vazby, estery, ethery) a označuje vzniklé konce čísly syntetických typů [1\*] až [16\*].],
    [Náhodně odstraní polovinu atomů v molekule bez ohledu na valenci.],
    [Převede molekulu na jednotlivé izolované atomy uhlíku.],
    [Štěpí výhradně vazby C-H.]
  )
)

#quiz_question(
  2,
  [K čemu slouží modely `FragSequenceExplorer` a `FragGraphExplorer`?],
  (
    [Umožňují scaffold-based generování: vezmou pevně zadaný fragment nebo scaffold (např. jádro vázané v kapse) a generují pouze variabilní substituenty nebo propojovací linkery se 100% zachováním nosného jádra.],
    [Slouží ke kompresi videa v reálném čase.],
    [Automaticky kreslí 3D diagramy v AutoCADu.],
    [Počítají molekulovou hmotnost z empirického vzorce.]
  )
)

#quiz_question(
  3,
  [Který parametr v `drugex.train` určuje, že chceme spustit režim Reinforcement Learningu namísto Fine-Tuningu?],
  (
    [`-tm RL`],
    [`-mode supervised_learning`],
    [`-train_type standard_sgd`],
    [`-engine docker`]
  )
)

#quiz_question(
  4,
  [Jaký význam má parametr `-pr` (prior) při spouštění tréninku `drugex.train -tm RL`?],
  (
    [Určuje cestu k souboru s fixním modelem (prior / mutate network), který slouží jako stabilní reference pro exploraci chemického prostoru.],
    [Nastavuje prioritu procesu v operačním systému Windows.],
    [Určuje port lokálního HTTP serveru.],
    [Aktivuje tisk barevných protokolů na tiskárně.]
  )
)

#quiz_question(
  5,
  [Která proměnná prostředí musí být nastavena na GPU klastru pro správnou funkci OpenEye ROCS skórovače?],
  (
    [`export OE_LICENSE=/cesta/k/oe_license.txt`],
    [`export CUDA_VISIBLE_DEVICES=none`],
    [`export PYTHON_DISABLE_MATH=1`],
    [`export DRUGEX_OFFLINE_MODE=true`]
  )
)

#quiz_question(
  6,
  [Jaká direktiva ve Slurm skriptu `#SBATCH` alokuje 1 grafickou kartu NVIDIA pro DrugEx trénink?],
  (
    [`#SBATCH --gres=gpu:1` (nebo `--gpus=1`)],
    [`#SBATCH --cpu-only`],
    [`#SBATCH --memory=1KB`],
    [`#SBATCH --network=dialup`]
  )
)

#quiz_question(
  7,
  [Jak DrugEx chrání data před nechtěným přepsáním při opakovaném spouštění preprocessingových skriptů?],
  (
    [Automaticky vytváří záložní složky ve formátu `backup_{cislo}` v datovém adresáři.],
    [Pošle varovný e-mail administrátorovi klastru.],
    [Zablokuje zápis na disk na 24 hodin.],
    [Zálohování se v DrugEx neprovádí.]
  )
)

#quiz_question(
  8,
  [Který příkaz DrugEx vygeneruje 1~000 nových molekul z natrénovaného modelu `arl_graph_trans_RL.pkg` a uloží je do TSV tabulky?],
  (
    [`python -m drugex.generate -b tutorial/CLI/examples -g arl_graph_trans_RL -n 1000 -gpu 0`],
    [`python -m drugex.delete --all`],
    [`python -m drugex.print_stats`],
    [`python -m drugex.benchmark --quick`]
  )
)
