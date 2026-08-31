/**
 * DrugEx Hub — Hash-Based Single Page Application Router
 */

import { state } from "./state.js";
import { renderLecture, renderThesisGuide, renderBenchmarkData, renderHandbookPrintView } from "./content.js";
import { renderDojo } from "./dojos.js";
import { renderQuiz } from "./quiz.js";
import { el, clear } from "./ui.js";

export function initRouter(viewport) {
  function handleRoute() {
    const hash = window.location.hash.slice(2) || "lecture/l1_1";
    state.activeRoute = hash;

    const parts = hash.split("/");
    const routeType = parts[0];
    const routeParam = parts[1];

    updateBreadcrumbs(routeType, routeParam);

    if (routeType === "lecture" && routeParam) {
      renderLecture(viewport, routeParam);
    } else if (routeType === "dojo" && routeParam) {
      renderDojo(viewport, routeParam);
    } else if (routeType === "quiz" && routeParam) {
      renderQuiz(viewport, routeParam);
    } else if (routeType === "thesis-guide") {
      renderThesisGuide(viewport, state.curriculum?.thesis);
    } else if (routeType === "benchmark") {
      renderBenchmarkData(viewport);
    } else if (routeType === "progress") {
      renderProgressView(viewport);
    } else if (routeType === "handbook-print") {
      renderHandbookPrintView(viewport);
    } else {
      renderLecture(viewport, "l1_1");
    }

    state.notify();
  }

  window.addEventListener("hashchange", handleRoute);
  handleRoute();
}

function updateBreadcrumbs(routeType, routeParam) {
  const crumbs = document.getElementById("breadcrumbs");
  if (!crumbs) return;
  clear(crumbs);

  crumbs.appendChild(el("span", {}, "DrugEx Hub"));
  crumbs.appendChild(el("span", { className: "crumb-sep" }, "/"));

  if (routeType === "lecture") {
    crumbs.appendChild(el("span", {}, "Přednášky"));
    crumbs.appendChild(el("span", { className: "crumb-sep" }, "/"));
    crumbs.appendChild(el("span", { className: "crumb-active" }, routeParam));
  } else if (routeType === "dojo") {
    crumbs.appendChild(el("span", {}, "Interactive Dojo"));
    crumbs.appendChild(el("span", { className: "crumb-sep" }, "/"));
    crumbs.appendChild(el("span", { className: "crumb-active" }, routeParam));
  } else if (routeType === "quiz") {
    crumbs.appendChild(el("span", {}, "Assessment Test"));
    crumbs.appendChild(el("span", { className: "crumb-sep" }, "/"));
    crumbs.appendChild(el("span", { className: "crumb-active" }, routeParam));
  } else if (routeType === "thesis-guide") {
    crumbs.appendChild(el("span", { className: "crumb-active" }, "Bakalářská Práce: Průvodce"));
  } else if (routeType === "benchmark") {
    crumbs.appendChild(el("span", { className: "crumb-active" }, "CCR2 Benchmark Datasets"));
  } else if (routeType === "handbook-print") {
    crumbs.appendChild(el("span", { className: "crumb-active" }, "📖 Kompletní Učebnice (PDF Export)"));
  }
}

function renderProgressView(container) {
  if (!container) return;
  clear(container);

  const items = state.progress.items || {};
  const quizzes = state.progress.quizzes || {};

  const studiedCount = Object.values(items).filter(s => s === "studied").length;
  const knownCount = Object.values(items).filter(s => s === "known").length;

  const view = el("div", { className: "lecture-container" }, [
    el("header", { className: "lecture-header" }, [
      el("div", { className: "lecture-meta" }, [
        el("span", { className: "item-tag core" }, "Dashboard"),
        el("span", {}, "Osobní přehled studia")
      ]),
      el("h1", { className: "lecture-title" }, "📊 Přehled Studia & Plán Práce"),
      el("p", { className: "lecture-desc" }, "Průběžné sledování absolvovaných témat, zvládnutých dovedností a výsledků testů.")
    ]),

    el("section", { className: "slide-card" }, [
      el("div", { className: "slide-title-bar" }, [
        el("div", { className: "slide-title" }, "📈 Průběh Přípravy na Bakalářskou Práci")
      ]),
      el("div", { style: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", textAlign: "center" } }, [
        el("div", { style: { background: "var(--editor)", padding: "16px", borderRadius: "var(--radius)", borderTop: "3px solid var(--bio-green)" } }, [
          el("div", { style: { fontSize: "28px", fontWeight: "700", color: "var(--bio-green)" } }, String(studiedCount)),
          el("div", { style: { fontSize: "12px", color: "var(--text-muted)", textTransform: "uppercase" } }, "Prostudováno")
        ]),
        el("div", { style: { background: "var(--editor)", padding: "16px", borderRadius: "var(--radius)", borderTop: "3px solid var(--amber-warn)" } }, [
          el("div", { style: { fontSize: "28px", fontWeight: "700", color: "var(--amber-warn)" } }, String(knownCount)),
          el("div", { style: { fontSize: "12px", color: "var(--text-muted)", textTransform: "uppercase" } }, "Znáno / Přeskočeno")
        ]),
        el("div", { style: { background: "var(--editor)", padding: "16px", borderRadius: "var(--radius)", borderTop: "3px solid var(--accent)" } }, [
          el("div", { style: { fontSize: "28px", fontWeight: "700", color: "var(--accent)" } }, `${Object.keys(quizzes).length}/6`),
          el("div", { style: { fontSize: "12px", color: "var(--text-muted)", textTransform: "uppercase" } }, "Dokončené Testy")
        ])
      ])
    ])
  ]);

  container.appendChild(view);
}
