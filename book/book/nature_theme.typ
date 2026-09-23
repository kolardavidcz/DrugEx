// ============================================================================
// NATURE THEME - Modern Scientific Publishing Template for Typst
// Style: Modern Nature (Informative, Clean, High Contrast, Content-First)
// ============================================================================

#let nature_navy = rgb("#0f172a")
#let nature_blue = rgb("#0284c7")
#let nature_emerald = rgb("#16a34a")
#let nature_rose = rgb("#e11d48")
#let nature_amber = rgb("#d97706")
#let nature_teal = rgb("#0891b2")
#let nature_purple = rgb("#7c3aed")
#let nature_slate = rgb("#334155")
#let nature_gray = rgb("#64748b")
#let nature_light = rgb("#f8fafc")

#let nature_book(
  title: "DrugEx: De Novo Design Léčiv & 3D Tvarové Modelování",
  subtitle: "Teoreticko-praktická monografie pro výzkum a vývoj malých molekul",
  author: "David Kolář",
  date: "2026",
  continuation: false,
  body
) = {
  // Page setup - 1.0 cm margins from edges as requested
  set page(
    paper: "a4",
    margin: (top: 1.4cm, bottom: 1.4cm, left: 1.0cm, right: 1.0cm),
    header: none,
    footer: none
  )

  // Typography settings: Modern, friendly sans-serif
  set text(
    font: ("IBM Plex Sans", "Ubuntu Sans", "DejaVu Sans"),
    size: 10pt,
    fill: rgb("#1e293b"),
    lang: "cs",
    hyphenate: auto
  )

  set par(
    justify: true,
    leading: 0.72em,
    first-line-indent: 0pt,
    spacing: 1.05em,
  )

  set list(spacing: 0.75em)
  set enum(spacing: 0.75em)

  // Headings - Every chapter starts on a fresh page
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(width: 100%, breakable: false)[
      #v(1.0em)
      #text(font: "IBM Plex Sans", size: 22pt, weight: "bold", fill: nature_navy)[
        #it.body
      ]
      #v(0.4em)
      #line(length: 100%, stroke: 1.5pt + nature_blue)
      #v(0.8em)
    ]
  }

  show heading.where(level: 2): it => block(width: 100%, breakable: false)[
    #v(1.3em)
    #text(font: "IBM Plex Sans", size: 14.5pt, weight: "bold", fill: nature_slate)[
      #it.body
    ]
    #v(0.35em)
  ]

  show heading.where(level: 3): it => block(width: 100%, breakable: false)[
    #v(1.0em)
    #text(font: "IBM Plex Sans", size: 12pt, weight: "bold", fill: nature_slate)[
      #it.body
    ]
    #v(0.25em)
  ]

  show raw: set text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 8.5pt)
  set raw(theme: "vscode_dark.tmTheme")

  // High-contrast inline code badges
  show raw.where(block: false): it => box(
    fill: rgb("#f1f5f9"),
    stroke: 0.5pt + rgb("#cbd5e1"),
    radius: 2.5pt,
    inset: (x: 3.5pt, y: 1.5pt),
    outset: (y: 1pt),
    text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 8.5pt, fill: rgb("#0f172a"), it)
  )

  // Cozy, friendly table styling with comfortable padding and modern header
  set table(
    inset: (x: 9.5pt, y: 7.5pt),
    stroke: (x, y) => if y == 0 { (bottom: 2pt + nature_blue) } else { (bottom: 0.5pt + rgb("#e2e8f0")) },
    fill: (col, row) => if row == 0 { rgb("#e0f2fe") } else if calc.even(row) { rgb("#f8fafc") } else { rgb("#ffffff") },
  )

  show table.cell: it => {
    if it.y == 0 {
      set text(font: "IBM Plex Sans", weight: "bold", size: 9pt, fill: nature_navy)
      set align(left + horizon)
      it
    } else {
      set text(size: 8.8pt, fill: rgb("#334155"))
      set align(left + horizon)
      it
    }
  }

  // Title Page
  page(header: none, footer: none)[
    #v(4cm)
    #align(center)[
      #text(font: "IBM Plex Sans", size: 28pt, weight: "bold", fill: nature_navy)[
        #title
      ]
      #v(1em)
      #text(font: "IBM Plex Sans", size: 15pt, fill: nature_blue, weight: "medium")[
        #subtitle
      ]
      #v(2.5cm)
      #line(length: 30%, stroke: 2pt + nature_blue)
      #v(2.5cm)
      #text(font: "IBM Plex Sans", size: 13pt, weight: "bold", fill: nature_navy)[
        #author
      ]
      #v(0.5em)
      #text(font: "IBM Plex Sans", size: 10pt, fill: nature_gray)[
        VŠCHT Praha / ÚOCHB AV ČR · #date
      ]
    ]
  ]

  // Table of Contents (2-sloupcový blokový přehled: Co se člověk naučí)
  page(header: none, footer: none)[
    #let toc_card(num, title, tag, color, takeaways) = block(
      width: 100%,
      stroke: 1pt + color.lighten(40%),
      fill: color.lighten(94%),
      inset: (x: 8.5pt, y: 7pt),
      radius: 5pt,
      spacing: 0.6em,
    )[
      #grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        [
          #text(font: "IBM Plex Sans", weight: "bold", size: 8.8pt, fill: color.darken(35%))[
            #num #title
          ]
        ],
        [
          #box(
            fill: color.lighten(80%),
            stroke: 0.5pt + color.lighten(30%),
            radius: 2.5pt,
            inset: (x: 4.5pt, y: 1.5pt)
          )[
            #text(font: "IBM Plex Mono", size: 6.5pt, weight: "bold", fill: color.darken(25%))[#tag]
          ]
        ]
      )
      #v(0.2em)
      #line(length: 100%, stroke: 0.5pt + color.lighten(50%))
      #set par(leading: 0.52em)
      #text(size: 7.2pt, fill: rgb("#334155"))[
        #for item in takeaways [
          • #item \
        ]
      ]
    ]

    #v(0.2em)
    #align(center)[
      #text(font: "IBM Plex Sans", size: 17pt, weight: "bold", fill: nature_navy)[
        Obsah Monografie: Co si z jednotlivých kapitol odnesete
      ]
      #v(0.15em)
      #text(font: "IBM Plex Sans", size: 8.5pt, fill: nature_gray)[
        Strukturovaný přehled klíčových kompetencí, algoritmických konceptů a praktických dovedností
      ]
    ]
    #v(0.5em)

    #if not continuation [
      #grid(
        columns: (1fr, 1fr),
        gutter: 8pt,
        [
          #toc_card("1.", "Reprezentace molekul pro generátory", "REPREZENTACE & BRICS", nature_blue, (
            "Syntaxe SMILES a tokenizace bez syntaktických chyb (VocSmiles)",
            "BRICS fragmentace vs. generování atom-po-atomu",
            "Attention Spotlight: Chemist's Q, K, V v roli nukleofilů/elektrofilů",
            "Transfer Learning z 1.5M látek Papyrus před spuštěním RL"
          ))
          #toc_card("2.", "Multi-Objective RL & QSPRPred", "MORL & REINFORCE", nature_purple, (
            "REINFORCE gradient se slepým sochařem a Advantage baseline",
            "Kolaps dominance při M ≥ 4 a kaskádový Gated MORL",
            "ECFP6 detektiv s kružítkem a hashovací skartovačkou",
            "Chemprop D-MPNN: Kancelářská šuškanda bez zpětné ozvěny"
          ))
          #toc_card("3.", "3D Tvarové Modelování (ROCS) & IDP", "3D SHAPE & STRAIN", nature_emerald, (
            "Grant-Pickettův Gaussovský integrál tvaru a barvy (Tcombo)",
            "Řešení planární singularity v tenzoru setrvačnosti (+0.01 Å)",
            "Filtrování konformačního pnutí (ΔE_strain ≤ 6 kcal/mol)",
            "Modelování konformačního poolu pro flexibilní cíle a IDP"
          ))
          #toc_card("3.5", "Chemoinformatický Můstek: Ztracené Klenoty", "RASCORE & MODIFIKÁTORY", nature_purple, (
            "SAScore vs. moderní AI retrosyntéza RAScore (AiZynthFinder)",
            "Matematika modifikátorů: Hladká S-křivka vs. Gaussovský zvon",
            "Distanční geometrie ETKDGv3 s RMSD prunováním (0.5 Å)",
            "Thread Pinning (num_threads=1): Prevence kolapsu CPU"
          ))
          #toc_card("3.6", "Teoretický Můstek: Koncepty z Fáze 1", "GATED MORL & GAUSS", nature_teal, (
            "Důkaz kolapsu dominance (M ≥ 4) a záchrana přes Gated MORL",
            "Chemprop D-MPNN směrované zprávy: Konec falešné ozvěny",
            "Gaussovský překryv ROCS, bioaktivní pnutí a planární oprava",
            "Dávková deduplikace SMILES (_deduplicate_smiles) pro záchranu CPU"
          ))
        ],
        [
          #toc_card("4.", "Srovnávací „Rosetta Stone“ pro 3D ROCS", "ROSETTA STONE", nature_amber, (
            "Analytické srovnání: RDKit vs. CDPKit vs. OpenEye ROCS",
            "Proč CDPKit plnohodnotně nahrazuje komerční licenci",
            "Thread Pinning (num_threads=1): Prevence CPU starvation",
            "Korekce kalibračních posunů skóre mezi backendy"
          ))
          #toc_card("5.", "Pokročilé Integrace: USRCAT & ProLIF", "FAST 60D & PROLIF", nature_teal, (
            "USRCAT 60D invarianty: 10~000× zrychlení vůči ROCS (0.18 ms)",
            "Hierarchická kaskáda: Nultý stupeň Pharm2D a USRCAT síto",
            "Úspora 74.5 % výpočetního času (prevence GPU starvation)",
            "ProLIF receptor-aware IFP: Vynucení vazby v kapse CCR2"
          ))
          #toc_card("6.", "3D Scaffold Hop z Epigallocatechinu", "PŘÍPADOVÁ STUDIE", nature_rose, (
            "Záchrana labilního polyfenolu EGC stabilním bioisosterem",
            "BRICS 3D Exit-Vektory a spojitá Gaussovská odměna Reward_EV",
            "Dvoufázový trénink se zmrazeným Priorem (10% epsilon)",
            "Sledování trajektorie v chemickém prostoru (Scaffviz / t-SNE)",
            "5 klíčových biofyzikálních dilemat pro obhajobu práce"
          ))
          #toc_card("7.", "LEGO Skládání & Diagnostika Patologií", "LEGO & TELEMETRIE", nature_navy, (
            "Modulární DrugExEnvironment: Scorer → Modifier → Threshold",
            "Čtyřkvadrantová telemetrie v reálném čase (Q1 až Q4)",
            "Léčba patologií: Tuková hydra, Makrocykly, Grammar Drift",
            "Zakázkový RingSizePenalizer a hardwarový profil pro RTX 4080"
          ))
          #toc_card("A–E", "Metodologické Přílohy & Glosář", "METODIKA & GLOSÁŘ", nature_slate, (
            "Příloha C: Kompletní reprodukovatelná tabulka hyperparametrů",
            "Příloha D: Eliminace CPU starvation bottlenecku na RTX 4080",
            "Příloha E: Autoritativní výkladový slovník 26 klíčových pojmů"
          ))
        ]
      )
    ] else [
      #grid(
        columns: (1fr, 1fr),
        gutter: 10pt,
        [
          #toc_card("3.5", "Chemoinformatický Můstek: Ztracené Klenoty", "RASCORE & MODIFIKÁTORY", nature_purple, (
            "SAScore vs. moderní AI retrosyntéza RAScore (AiZynthFinder)",
            "Matematika modifikátorů: Hladká S-křivka vs. Gaussovský zvon",
            "Distanční geometrie ETKDGv3 s RMSD prunováním (0.5 Å)",
            "Thread Pinning (num_threads=1): Prevence kolapsu CPU"
          ))
          #toc_card("3.6", "Teoretický Můstek: Koncepty z Fáze 1", "GATED MORL & GAUSS", nature_teal, (
            "Důkaz kolapsu dominance (M ≥ 4) a záchrana přes Gated MORL",
            "Chemprop D-MPNN směrované zprávy: Konec falešné ozvěny",
            "Gaussovský překryv ROCS, bioaktivní pnutí a planární oprava",
            "Dávková deduplikace SMILES (_deduplicate_smiles) pro záchranu CPU"
          ))
          #toc_card("4.", "Srovnávací „Rosetta Stone“ pro 3D ROCS", "ROSETTA STONE", nature_amber, (
            "Analytické srovnání: RDKit vs. CDPKit vs. OpenEye ROCS",
            "Proč CDPKit plnohodnotně nahrazuje komerční licenci OpenEye",
            "Thread Pinning (num_threads=1): Záchrana CPU na RTX 4080",
            "Korekce kalibračních posunů skóre mezi různými backendy"
          ))
          #toc_card("5.", "Pokročilé Integrace: USRCAT & ProLIF", "FAST 60D & PROLIF", nature_teal, (
            "USRCAT 60D invarianty: 10~000× zrychlení vůči ROCS (0.18 ms)",
            "Hierarchická kaskáda: Nultý stupeň Pharm2D a USRCAT síto",
            "Úspora 74.5 % výpočetního času (prevence GPU starvation)",
            "ProLIF receptor-aware IFP: Vynucení vazby v kapse CCR2"
          ))
        ],
        [
          #toc_card("6.", "3D Scaffold Hop z Epigallocatechinu", "PŘÍPADOVÁ STUDIE", nature_rose, (
            "Záchrana labilního polyfenolu EGC stabilním bioisosterem",
            "BRICS 3D Exit-Vektory a spojitá Gaussovská odměna Reward_EV",
            "Dvoufázový trénink se zmrazeným Priorem (10% epsilon)",
            "Sledování trajektorie v chemickém prostoru (Scaffviz / t-SNE)",
            "5 klíčových biofyzikálních dilemat pro obhajobu práce"
          ))
          #toc_card("7.", "LEGO Skládání & Diagnostika Patologií", "LEGO & TELEMETRIE", nature_navy, (
            "Modulární DrugExEnvironment: Scorer → Modifier → Threshold",
            "Čtyřkvadrantová telemetrie v reálném čase (Q1 až Q4)",
            "Léčba patologií: Tuková hydra, Makrocykly, Grammar Drift",
            "Zakázkový RingSizePenalizer a profil pro lokální RTX 4080"
          ))
          #toc_card("A–E", "Metodologické Přílohy & Glosář", "METODIKA & GLOSÁŘ", nature_slate, (
            "Příloha C: Kompletní reprodukovatelná tabulka hyperparametrů",
            "Příloha D: Eliminace CPU starvation bottlenecku na RTX 4080",
            "Příloha E: Autoritativní výkladový slovník 26 klíčových pojmů"
          ))
        ]
      )
    ]
  ]

  body
}

// ============================================================================
// CALLOUT BOXES (BREAKABLE ACROSS PAGES)
// ============================================================================

#let rule_box(title, body) = block(
  width: 100%,
  stroke: (left: 3.5pt + nature_emerald),
  fill: rgb("#f0fdf4"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.15em,
  breakable: true
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 9.5pt, fill: nature_emerald)[
    ◆ PRAVIDLO Z PRAXE: #title
  ]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#064e3b"))[#body]
]

#let pitfall_box(title, body) = block(
  width: 100%,
  stroke: (left: 3.5pt + nature_rose),
  fill: rgb("#fff1f2"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.15em,
  breakable: true
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 9.5pt, fill: nature_rose)[
    ▲ ÚSKALÍ & ČASTÁ CHYBA: #title
  ]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#881337"))[#body]
]

#let insight_box(title, body) = block(
  width: 100%,
  stroke: (left: 3.5pt + nature_amber),
  fill: rgb("#fffbeb"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.15em,
  breakable: true
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 9.5pt, fill: nature_amber)[
    ● HLOUBKOVÝ POZNATEK: #title
  ]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#78350f"))[#body]
]

#let lab_context_box(title, body) = block(
  width: 100%,
  stroke: (left: 3.5pt + nature_teal),
  fill: rgb("#f0fdfa"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.15em,
  breakable: true
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 9.5pt, fill: nature_teal)[
    ✦ VÝZKUMNÝ KONTEXT: #title
  ]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#134e4a"))[#body]
]

#let telemetry_box(title, body) = block(
  width: 100%,
  stroke: (left: 3.5pt + nature_purple),
  fill: rgb("#f5f3ff"),
  inset: (x: 14pt, y: 10pt),
  radius: (right: 4pt),
  spacing: 1.15em,
  breakable: true
)[
  #text(font: "IBM Plex Sans", weight: "bold", size: 9.5pt, fill: nature_purple)[
    ■ TENZOROVÁ TELEMETRIE: #title
  ]
  #v(0.3em)
  #text(size: 9.5pt, fill: rgb("#4c1d95"))[#body]
]

// ============================================================================
// CODE & EXECUTION CARD (AUTHENTIC VS CODE DARK PLUS, BREAKABLE ACROSS PAGES)
// ============================================================================

#let code_card(title: none, lang: "python", code: "", output: none) = block(
  width: 100%,
  stroke: 1pt + rgb("#333333"),
  radius: 4pt,
  spacing: 1.15em,
  breakable: true
)[
  #if title != none [
    #block(
      width: 100%,
      fill: rgb("#252526"),
      inset: (x: 12pt, y: 6.5pt),
      stroke: (bottom: 1pt + rgb("#333333"))
    )[
      #grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        [
          #text(font: "IBM Plex Sans", weight: "bold", size: 8.5pt, fill: rgb("#cccccc"))[
            #text(font: "IBM Plex Mono", weight: "bold", fill: rgb("#007acc"))[›\_] #title
          ]
        ],
        [
          #box(
            fill: rgb(0, 122, 204, 30),
            stroke: 0.5pt + rgb("#007acc"),
            radius: 2.5pt,
            inset: (x: 5pt, y: 2pt)
          )[
            #text(font: "IBM Plex Mono", size: 7.5pt, weight: "bold", fill: rgb("#38bdf8"))[
              PYTHON 3.12 · DRUGEX
            ]
          ]
        ]
      )
    ]
  ]
  #block(
    width: 100%,
    fill: rgb("#1e1e1e"),
    inset: (x: 12pt, y: 9pt),
    breakable: true
  )[
    #set text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 8.5pt, fill: rgb("#d4d4d4"))
    #raw(code, lang: lang, block: true)
  ]
  #if output != none [
    #block(
      width: 100%,
      fill: rgb("#181818"),
      inset: (x: 12pt, y: 5.5pt),
      stroke: (top: 1pt + rgb("#2d2d2d"), bottom: 1pt + rgb("#252526"))
    )[
      #grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        [
          #text(font: "IBM Plex Sans", size: 8pt, weight: "bold", fill: rgb("#38bdf8"))[
            ▶ VÝSTUP / POZNÁMKA
          ]
        ],
        [
          #box(
            fill: rgb(74, 222, 128, 25),
            stroke: 0.5pt + rgb("#4ade80"),
            radius: 2.5pt,
            inset: (x: 5pt, y: 2pt)
          )[
            #text(font: "IBM Plex Mono", size: 7.5pt, weight: "bold", fill: rgb("#4ade80"))[
              ✓ RUNTIME EXIT 0
            ]
          ]
        ]
      )
    ]
    #block(
      width: 100%,
      fill: rgb("#141414"),
      inset: (x: 12pt, y: 8pt),
      breakable: true
    )[
      #set text(font: ("IBM Plex Mono", "DejaVu Sans Mono"), size: 8pt, fill: rgb("#89d185"))
      #raw(output, block: true)
    ]
  ]
]

// ============================================================================
// FIGURE & SCREENSHOT CARD
// ============================================================================

#let figure_card(body, caption: none, width: 100%) = block(
  width: 100%,
  stroke: 1pt + rgb("#cbd5e1"),
  radius: 6pt,
  clip: true,
  spacing: 1.4em,
  breakable: false
)[
  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    inset: (x: 0pt, y: 0pt)
  )[
    #align(center)[
      #if type(body) == str {
        image(body, width: width)
      } else {
        body
      }
    ]
  ]
  #if caption != none [
    #block(
      width: 100%,
      fill: rgb("#f8fafc"),
      inset: (x: 14pt, y: 8pt),
      stroke: (top: 1pt + rgb("#e2e8f0"))
    )[
      #text(font: "IBM Plex Sans", size: 8.5pt, fill: rgb("#475569"))[
        #text(weight: "bold", fill: nature_navy)[Obrázek:] #caption
      ]
    ]
  ]
]

