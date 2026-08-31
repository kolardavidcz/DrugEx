/**
 * DrugEx Hub — Comprehensive Lecture & Thesis Guide View Renderer
 */

import { el, clear, copyText } from "./ui.js";
import { state } from "./state.js";
import { highlightPython, formatFormula } from "./format.js";
import { LECTURE_DATA } from "./lectures_content.js";

export function renderLecture(container, lectureId) {
  if (!container) return;
  clear(container);

  const lec = LECTURE_DATA[lectureId];
  if (!lec) {
    container.innerHTML = `<div class="alert-box alert-warning">Přednáška "${lectureId}" nebyla nalezena.</div>`;
    return;
  }

  const currentStatus = state.getItemStatus(lectureId);

  // Extract module number (e.g. l3_2 -> module 3)
  const modNumMatch = lectureId.match(/^l(\d+)_/);
  const modNum = modNumMatch ? modNumMatch[1] : "1";

  const allLectureIds = [
    "l1_1", "l1_2", "l1_3",
    "l2_1", "l2_2", "l2_3",
    "l3_1", "l3_2", "l3_3",
    "l4_1", "l4_2", "l4_3",
    "l5_1", "l5_2", "l5_3",
    "l6_1", "l6_2", "l6_3"
  ];
  const currentIdx = allLectureIds.indexOf(lectureId);
  const prevLecId = currentIdx > 0 ? allLectureIds[currentIdx - 1] : null;
  const nextLecId = currentIdx < allLectureIds.length - 1 ? allLectureIds[currentIdx + 1] : null;
  const prevLec = prevLecId ? LECTURE_DATA[prevLecId] : null;
  const nextLec = nextLecId ? LECTURE_DATA[nextLecId] : null;

  const view = el("div", { className: "lecture-container" }, [
    // Header
    el("header", { className: "lecture-header" }, [
      el("div", { className: "lecture-meta" }, [
        el("span", { className: `item-tag ${lec.tag.toLowerCase()}` }, lec.tag),
        el("span", {}, `Relevance: ${lec.relevance}/10`),
        el("span", {}, `• ${lec.slides.length} slidů`)
      ]),
      el("h1", { className: "lecture-title" }, lec.title),
      el("p", { className: "lecture-desc" }, lec.summary),
      
      // Interactive Action Strip
      el("div", { className: "lecture-action-strip" }, [
        el("a", {
          href: `#/dojo/dojo_${modNum}`,
          className: "action-pill dojo-pill",
          title: "Přejít do interaktivního Doja tohoto modulu"
        }, [
          el("span", {}, "⚡"),
          el("span", {}, `Otevřít Dojo ${modNum}`)
        ]),
        el("a", {
          href: `#/quiz/quiz_m${modNum}`,
          className: "action-pill quiz-pill",
          title: "Spustit test znalostí tohoto modulu"
        }, [
          el("span", {}, "📝"),
          el("span", {}, `Spustit Test M${modNum}`)
        ]),
        el("a", {
          href: "#/thesis-guide",
          className: "action-pill",
          title: "Zobrazit průvodce bakalářskou prací"
        }, [
          el("span", {}, "🎓"),
          el("span", {}, "Průvodce BP")
        ]),
        el("button", {
          className: `action-pill ${currentStatus === "studied" ? "active" : ""}`,
          style: currentStatus === "studied" ? { background: "var(--bio-green)", color: "#000", borderColor: "var(--bio-green)" } : {},
          onClick: () => state.markItemStatus(lectureId, "studied")
        }, "✓ Prostudováno"),
        el("button", {
          className: `action-pill ${currentStatus === "known" ? "active" : ""}`,
          style: currentStatus === "known" ? { background: "var(--amber-warn)", color: "#000", borderColor: "var(--amber-warn)" } : {},
          onClick: () => state.markItemStatus(lectureId, "known")
        }, "↷ Znáno"),
        el("button", {
          className: "action-pill",
          onClick: () => window.print()
        }, "🖨️ Tisk (Ctrl+P)")
      ])
    ])
  ]);

  // Render Slides
  lec.slides.forEach((slide, idx) => {
    const card = el("section", { className: "slide-card" }, [
      el("div", { className: "slide-title-bar" }, [
        el("div", { className: "slide-title" }, [
          el("span", { style: { color: "var(--accent)" } }, "◈"),
          el("span", {}, slide.title)
        ]),
        el("span", { className: "slide-number" }, `Slide ${idx + 1}/${lec.slides.length}`)
      ]),
      el("div", { className: "slide-content", innerHTML: slide.content })
    ]);

    if (slide.code) {
      const codeBox = el("div", { className: "code-container" }, [
        el("div", { className: "code-header" }, [
          el("span", {}, slide.codeLang || "python"),
          el("button", {
            className: "tb-btn",
            style: { padding: "1px 6px", fontSize: "10.5px" },
            onClick: () => copyText(slide.code)
          }, "Copy")
        ]),
        el("pre", { className: "code-block" }, [
          el("code", { innerHTML: highlightPython(slide.code) })
        ])
      ]);
      card.appendChild(codeBox);
    }

    if (slide.compare) {
      const cmpGrid = el("div", { className: "compare-grid" }, [
        el("div", { className: "compare-col left" }, [
          el("div", { className: "compare-heading" }, slide.compare.leftTitle),
          el("div", { innerHTML: slide.compare.leftContent })
        ]),
        el("div", { className: "compare-col right" }, [
          el("div", { className: "compare-heading" }, slide.compare.rightTitle),
          el("div", { innerHTML: slide.compare.rightContent })
        ])
      ]);
      card.appendChild(cmpGrid);
    }

    if (slide.alert) {
      const alertBox = el("div", { className: `alert-box alert-${slide.alert.type}` }, [
        el("div", { className: "alert-title" }, slide.alert.title || slide.alert.type),
        el("div", { innerHTML: slide.alert.text })
      ]);
      card.appendChild(alertBox);
    }

    view.appendChild(card);
  });

  // Bottom Interactive Navigation Pagination
  const navGrid = el("nav", { className: "lecture-pagination", ariaLabel: "Navigace mezi přednáškami" });

  if (prevLec && prevLecId) {
    const prevCard = el("a", {
      href: `#/lecture/${prevLecId}`,
      className: "nav-card prev"
    }, [
      el("span", { className: "nav-card-dir" }, "← Předchozí Lekce"),
      el("span", { className: "nav-card-title" }, prevLec.title)
    ]);
    navGrid.appendChild(prevCard);
  } else {
    navGrid.appendChild(el("div", {})); // empty spacer for grid alignment
  }

  if (nextLec && nextLecId) {
    const nextCard = el("a", {
      href: `#/lecture/${nextLecId}`,
      className: "nav-card next"
    }, [
      el("span", { className: "nav-card-dir" }, "Další Lekce →"),
      el("span", { className: "nav-card-title" }, nextLec.title)
    ]);
    navGrid.appendChild(nextCard);
  }

  view.appendChild(navGrid);

  container.appendChild(view);

  // Auto-render KaTeX LaTeX Math formulas
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(view, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false
      });
    } catch (e) {
      console.warn("KaTeX rendering note:", e);
    }
  }
}

export function renderThesisGuide(container, thesisData) {
  if (!container) return;
  clear(container);

  const guide = thesisData || (state.curriculum && state.curriculum.thesis);

  const view = el("div", { className: "lecture-container thesis-guide-container" }, [
    el("header", { className: "lecture-header" }, [
      el("div", { className: "lecture-meta" }, [
        el("span", { className: "item-tag legendary" }, "Bakalářská Práce"),
        el("span", {}, "VŠCHT Praha / ÚOCHB AV ČR")
      ]),
      el("h1", { className: "lecture-title" }, guide.title),
      el("p", { className: "lecture-desc" }, guide.description)
    ]),

    // Milestones
    el("section", { className: "slide-card" }, [
      el("div", { className: "slide-title-bar" }, [
        el("div", { className: "slide-title" }, "🚩 Harmonogram & Náplň Činnosti Studenta")
      ]),
      el("div", { style: { display: "flex", flexDirection: "column", gap: "12px" } }, 
        (guide.milestones || []).map(m => el("div", {
          style: { background: "var(--editor)", padding: "12px 16px", borderRadius: "var(--radius)", borderLeft: "3px solid var(--accent)" }
        }, [
          el("div", { style: { fontWeight: "600", color: "var(--text-bright)", marginBottom: "4px" } }, m.title),
          el("div", { style: { fontSize: "var(--fs-sm)", color: "var(--text)" } }, m.desc)
        ]))
      )
    ]),

    // Protocol Template
    el("section", { className: "slide-card" }, [
      el("div", { className: "slide-title-bar" }, [
        el("div", { className: "slide-title" }, "📄 Šablona Výpočetních Metod pro Text Práce (LaTeX / MD)"),
        el("button", {
          className: "tb-btn",
          onClick: () => copyText(document.getElementById("methodsText")?.innerText || "")
        }, "Kopírovat do práce")
      ]),
      el("div", {
        id: "methodsText",
        style: { background: "var(--editor)", padding: "16px", borderRadius: "var(--radius)", fontSize: "var(--fs-xs)", fontFamily: "var(--font-mono)", whiteSpace: "pre-wrap", lineHeight: "1.6", color: "#93c5fd" }
      }, `### Výpočetní metody (Computational Methods)\n\nDe novo generování molekul bylo realizováno pomocí platformy DrugEx v3.4 s využitím vícekriteriálního zpětnovazebního učení (MORL). Jako výchozí generátor byl použit SequenceRNN předtrénovaný na databázi Papyrus v05.5 (~1,5 mil. sloučenin) a jemně dotrénovaný (fine-tuning) na známých ligandech po dobu 100 epoch.\n\nOptimalizační prostředí (DrugExEnvironment) integrovalo:\n1. 3D Tvarové porovnávání (ROCS): RDKitROCSScorer s metrikou TanimotoCombo a konformačním generátorem ETKDGv3 (max. 50 konformerů, 4 stereoisomery). Dělící práh byl stanoven na základě ROC analýzy a Youdenova indexu.\n2. Syntetická dostupnost: SAScore s modifikátorem SmoothClippedScore(lower_x=5.0, upper_x=3.0).\n\nVícekriteriální rovnováha byla řízena pomocí Pareto Crowding Distance. Trénink probíhal 50 epoch s exploračním poměrem epsilon = 0.2.`)
    ])
  ]);

  container.appendChild(view);
}

export function renderBenchmarkData(container) {
  if (!container) return;
  clear(container);

  const view = el("div", { className: "lecture-container" }, [
    el("header", { className: "lecture-header" }, [
      el("div", { className: "lecture-meta" }, [
        el("span", { className: "item-tag wow" }, "CCR2 Benchmark"),
        el("span", {}, "Testovací data z fulopjoz/DrugEx")
      ]),
      el("h1", { className: "lecture-title" }, "CCR2 Benchmark Dataset & Referenční Struktury"),
      el("p", { className: "lecture-desc" }, "Reálná data použitá pro kalibraci prahů, trénování a validaci tvarového hodnocení.")
    ]),

    el("section", { className: "slide-card" }, [
      el("div", { className: "slide-title-bar" }, [
        el("div", { className: "slide-title" }, "📁 Soubory Datové Sady v data/benchmarks/")
      ]),
      el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "12px" } }, [
        el("div", { style: { background: "var(--editor)", padding: "12px", borderRadius: "var(--radius)" } }, [
          el("div", { style: { fontWeight: "600", color: "var(--bio-green)" } }, "CCR2_reference_ligands.sdf"),
          el("div", { style: { fontSize: "var(--fs-xs)", color: "var(--text-muted)" } }, "3D krystalové pózy referenčních ligandů CCR2 pro RDKit, CDPKit a OpenEye.")
        ]),
        el("div", { style: { background: "var(--editor)", padding: "12px", borderRadius: "var(--radius)" } }, [
          el("div", { style: { fontWeight: "600", color: "var(--accent)" } }, "actives_ccr2_N75.csv"),
          el("div", { style: { fontSize: "var(--fs-xs)", color: "var(--text-muted)" } }, "75 experimentálně ověřených aktivních ligandů CCR2 pro ROC analýzu.")
        ]),
        el("div", { style: { background: "var(--editor)", padding: "12px", borderRadius: "var(--radius)" } }, [
          el("div", { style: { fontWeight: "600", color: "var(--danger-red)" } }, "decoys_ccr2_N500.csv"),
          el("div", { style: { fontSize: "var(--fs-xs)", color: "var(--text-muted)" } }, "500 DUD-E decoy molekul se shodnými fyzikálními vlastnostmi, ale odlišným tvarem.")
        ]),
        el("div", { style: { background: "var(--editor)", padding: "12px", borderRadius: "var(--radius)" } }, [
          el("div", { style: { fontWeight: "600", color: "var(--rocs-purple)" } }, "model1.sq, model2.sq, supermol_123.sdf"),
          el("div", { style: { fontSize: "var(--fs-xs)", color: "var(--text-muted)" } }, "Předpočítané OpenEye VROCS shape queries a konsensuální supermolekula.")
        ])
      ])
    ])
  ]);

  container.appendChild(view);
}

export function renderHandbookPrintView(container) {
  if (!container) return;
  clear(container);

  const view = el("div", { className: "handbook-print-container" });

  // 1. Cover Page
  const cover = el("div", { className: "handbook-cover-page" }, [
    el("div", { style: { fontSize: "42px", marginBottom: "12px" } }, "🧬"),
    el("h1", { className: "handbook-cover-title" }, "DrugEx Hub · De Novo Drug Design & ROCS Shape-Matching"),
    el("p", { className: "handbook-cover-subtitle" }, 
      "Kompletní Výuková Příručka, Metodologický Manuál a Teoretické Základy pro Bakalářskou Práci (VŠCHT Praha / ÚOCHB AV ČR) — Kombinace de novo generování molekul a 3D tvarového porovnávání pro flexibilní cíle a IDP."
    ),
    el("div", { style: { marginTop: "30px", fontSize: "11pt", color: "#64748b", lineHeight: "1.8" } }, [
      el("div", { style: { fontWeight: "700", color: "#0f172a" } }, "Bakalářská Práce: David Kolář"),
      el("div", {}, "Vysoká škola chemicko-technologická v Praze (VŠCHT Praha)"),
      el("div", {}, "Ústav organické chemie a biochemie AV ČR (ÚOCHB AV ČR)"),
      el("div", {}, "Softwarová platforma: DrugEx v3.4 (feature/rocs-scoring) · OpenEye ROCS / CDPKit / RDKit")
    ])
  ]);
  view.appendChild(cover);

  // 2. All 6 Modules & 18 Lectures
  const moduleMap = [
    { num: 1, title: "Modul 1: De Novo Generování & Molekulární Reprezentace", lectures: ["l1_1", "l1_2", "l1_3"] },
    { num: 2, title: "Modul 2: Vícekriteriální Zpětnovazební Učení (MORL) & Paretova Optimalita", lectures: ["l2_1", "l2_2", "l2_3"] },
    { num: 3, title: "Modul 3: 3D Tvarové Porovnávání (ROCS), IDP & Konformační Enginy", lectures: ["l3_1", "l3_2", "l3_3"] },
    { num: 4, title: "Modul 4: Hloubková Architektura ROCS Scorerů (RDKit, CDPKit, OpenEye)", lectures: ["l4_1", "l4_2", "l4_3"] },
    { num: 5, title: "Modul 5: Experimentální Pipeline & Validace na CCR2 Benchmarku", lectures: ["l5_1", "l5_2", "l5_3"] },
    { num: 6, title: "Modul 6: Fragmentový Design (BRICS), CLI & Škálování na HPC Superpočítačích", lectures: ["l6_1", "l6_2", "l6_3"] }
  ];

  moduleMap.forEach(mod => {
    // Module Divider
    const modDiv = el("div", { className: "module-print-divider" }, [
      el("h2", {}, mod.title)
    ]);
    view.appendChild(modDiv);

    // Render each lecture
    mod.lectures.forEach(lecId => {
      const lec = LECTURE_DATA[lecId];
      if (!lec) return;

      const lecSec = el("div", { style: { marginBottom: "28pt" } }, [
        el("header", { className: "lecture-header" }, [
          el("h2", { className: "lecture-title" }, lec.title),
          el("p", { className: "lecture-desc" }, lec.summary)
        ])
      ]);

      // Slides
      lec.slides.forEach((slide, idx) => {
        const card = el("section", { className: "slide-card" }, [
          el("div", { className: "slide-title-bar" }, [
            el("div", { className: "slide-title" }, slide.title),
            el("span", { className: "slide-number" }, `Slide ${idx + 1}/${lec.slides.length}`)
          ]),
          el("div", { className: "slide-content", innerHTML: slide.content })
        ]);

        if (slide.code) {
          const codeBox = el("div", { className: "code-container" }, [
            el("div", { className: "code-header" }, [
              el("span", {}, slide.codeLang || "python")
            ]),
            el("pre", { className: "code-block" }, [
              el("code", { innerHTML: highlightPython(slide.code) })
            ])
          ]);
          card.appendChild(codeBox);
        }

        if (slide.compare) {
          const cmpGrid = el("div", { className: "compare-grid" }, [
            el("div", { className: "compare-col left" }, [
              el("div", { className: "compare-heading" }, slide.compare.leftTitle),
              el("div", { innerHTML: slide.compare.leftContent })
            ]),
            el("div", { className: "compare-col right" }, [
              el("div", { className: "compare-heading" }, slide.compare.rightTitle),
              el("div", { innerHTML: slide.compare.rightContent })
            ])
          ]);
          card.appendChild(cmpGrid);
        }

        if (slide.alert) {
          const alertBox = el("div", { className: `alert-box alert-${slide.alert.type}` }, [
            el("div", { className: "alert-title" }, slide.alert.title || slide.alert.type),
            el("div", { innerHTML: slide.alert.text })
          ]);
          card.appendChild(alertBox);
        }

        lecSec.appendChild(card);
      });

      view.appendChild(lecSec);
    });
  });

  // 3. Append Thesis Guide and Protocol Template
  const thesisSection = el("div", { className: "module-print-divider" }, [
    el("h2", {}, "Příloha: Metodologický Protokol & Šablona Textu Práce")
  ]);
  view.appendChild(thesisSection);

  const thesisGuideCard = el("section", { className: "slide-card" }, [
    el("div", { className: "slide-title-bar" }, [
      el("div", { className: "slide-title" }, "Šablona Výpočetních Metod pro Text Bakalářské Práce")
    ]),
    el("div", {
      style: { background: "#0d1117", color: "#93c5fd", padding: "12pt 16pt", borderRadius: "4pt", fontFamily: "monospace", fontSize: "9pt", lineHeight: "1.6", whiteSpace: "pre-wrap" }
    }, `### Výpočetní metody (Computational Methods)\n\nDe novo generování molekul bylo realizováno pomocí platformy DrugEx v3.4 s využitím vícekriteriálního zpětnovazebního učení (MORL). Jako výchozí generátor byl použit SequenceRNN předtrénovaný na databázi Papyrus v05.5 (~1,5 mil. sloučenin) a jemně dotrénovaný (fine-tuning) na známých ligandech po dobu 100 epoch.\n\nOptimalizační prostředí (DrugExEnvironment) integrovalo:\n1. 3D Tvarové porovnávání (ROCS): RDKitROCSScorer s metrikou TanimotoCombo a konformačním generátorem ETKDGv3 (max. 50 konformerů, 4 stereoisomery). Dělící práh byl stanoven na základě ROC analýzy a Youdenova indexu.\n2. Syntetická dostupnost: SAScore s modifikátorem SmoothClippedScore(lower_x=5.0, upper_x=3.0).\n\nVícekriteriální rovnováha byla řízena pomocí Pareto Crowding Distance. Trénink probíhal 50 epoch s exploračním poměrem epsilon = 0.2.`)
  ]);
  view.appendChild(thesisGuideCard);

  container.appendChild(view);

  // Auto-render KaTeX
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(view, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false
      });
    } catch (e) {
      console.warn("KaTeX print rendering:", e);
    }
  }
}
