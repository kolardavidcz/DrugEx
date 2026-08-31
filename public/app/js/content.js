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
      el("div", { style: { display: "flex", gap: "8px", marginTop: "8px" } }, [
        el("button", {
          className: `tb-btn ${currentStatus === "studied" ? "active" : ""}`,
          style: currentStatus === "studied" ? { background: "var(--bio-green)", color: "#000" } : {},
          onClick: () => state.markItemStatus(lectureId, "studied")
        }, "✓ Prostudováno"),
        el("button", {
          className: `tb-btn ${currentStatus === "known" ? "active" : ""}`,
          style: currentStatus === "known" ? { background: "var(--amber-warn)", color: "#000" } : {},
          onClick: () => state.markItemStatus(lectureId, "known")
        }, "↷ Znáno"),
        el("button", {
          className: "tb-btn",
          onClick: () => window.print()
        }, "🖨️ Tisknout (Ctrl+P)")
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
