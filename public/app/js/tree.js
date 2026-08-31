/**
 * DrugEx Hub — Explorer Sidebar Tree Navigation
 */

import { el, clear } from "./ui.js";
import { state } from "./state.js";

export function renderTree(container, curriculum, onSelect) {
  if (!container || !curriculum) return;
  clear(container);

  const filterText = (state.searchQuery || "").toLowerCase();
  const activeTag = state.selectedTag;

  // Bachelor Thesis Guide Header Item
  const thesisItem = el("a", {
    href: "#/thesis-guide",
    className: `tree-item ${state.activeRoute === "thesis-guide" ? "active" : ""}`,
    style: { borderBottom: "1px solid var(--border-subtle)", padding: "8px 12px", background: "rgba(192, 132, 252, 0.08)" }
  }, [
    el("span", { style: { color: "var(--rocs-purple)", fontWeight: "bold" } }, "🎓"),
    el("span", { className: "item-title", style: { fontWeight: "600", color: "var(--rocs-purple)" } }, "Bakalářská Práce: Průvodce"),
    el("span", { className: "item-tag legendary" }, "Thesis")
  ]);
  container.appendChild(thesisItem);

  // CCR2 Benchmark Data Item
  const benchItem = el("a", {
    href: "#/benchmark",
    className: `tree-item ${state.activeRoute === "benchmark" ? "active" : ""}`,
    style: { borderBottom: "1px solid var(--border-subtle)", padding: "8px 12px", background: "rgba(56, 189, 248, 0.06)" }
  }, [
    el("span", { style: { color: "var(--accent)" } }, "📊"),
    el("span", { className: "item-title", style: { fontWeight: "600", color: "var(--accent)" } }, "CCR2 Benchmark Data"),
    el("span", { className: "item-tag wow" }, "Data")
  ]);
  container.appendChild(benchItem);

  // Practitioner Cookbook & Hyperparameter Hub Item
  const cookbookItem = el("a", {
    href: "#/cookbook",
    className: `tree-item ${state.activeRoute === "cookbook" ? "active" : ""}`,
    style: { borderBottom: "1px solid var(--border-subtle)", padding: "8px 12px", background: "rgba(74, 222, 128, 0.06)" }
  }, [
    el("span", { style: { color: "var(--bio-green)" } }, "👨‍🔬"),
    el("span", { className: "item-title", style: { fontWeight: "600", color: "var(--bio-green)" } }, "Kuchařka & Hyperparametry"),
    el("span", { className: "item-tag core" }, "Recepty")
  ]);
  container.appendChild(cookbookItem);

  // Modules & Lectures
  curriculum.modules.forEach(mod => {
    // Filter check
    const matchingLectures = mod.lectures.filter(lec => {
      const matchText = !filterText || lec.title.toLowerCase().includes(filterText) || lec.summary.toLowerCase().includes(filterText);
      const matchTag = !activeTag || lec.tag === activeTag || mod.tag === activeTag;
      return matchText && matchTag;
    });

    const isDojoMatch = !filterText || mod.dojo.title.toLowerCase().includes(filterText);
    const isQuizMatch = !filterText || mod.quiz.title.toLowerCase().includes(filterText);

    if (matchingLectures.length === 0 && !isDojoMatch && !isQuizMatch && filterText) {
      return;
    }

    const modWrapper = el("div", { className: "tree-module" });
    const isCollapsed = state.collapsedModules && state.collapsedModules.has(mod.number);

    // Module Header with smooth accordion click
    const arrow = el("span", {
      style: {
        opacity: 0.7,
        fontSize: "9px",
        display: "inline-block",
        transform: isCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
        transition: "transform 0.2s ease"
      }
    }, "▼");

    const modTagBadge = el("span", {
      className: `item-tag ${mod.tag.toLowerCase()}`,
      title: `Filtrovat podle tagu "${mod.tag}"`,
      onClick: (e) => {
        e.stopPropagation();
        state.selectedTag = state.selectedTag === mod.tag ? null : mod.tag;
        state.notify();
      }
    }, mod.tag);

    const modHeader = el("div", {
      className: "tree-module-header",
      onClick: () => {
        if (!state.collapsedModules) state.collapsedModules = new Set();
        if (state.collapsedModules.has(mod.number)) {
          state.collapsedModules.delete(mod.number);
        } else {
          state.collapsedModules.add(mod.number);
        }
        state.notify();
      }
    }, [
      el("div", { style: { display: "flex", alignItems: "center", gap: "8px" } }, [
        arrow,
        el("span", {}, `M${mod.number}: ${mod.badge}`)
      ]),
      modTagBadge
    ]);
    modWrapper.appendChild(modHeader);

    const itemsList = el("div", {
      className: "tree-items",
      style: { display: isCollapsed ? "none" : "block" }
    });

    // Lectures
    matchingLectures.forEach(lec => {
      const status = state.getItemStatus(lec.id);
      const statusIcon = status === "studied" ? "✓" : (status === "known" ? "↷" : "○");
      const statusColor = status === "studied" ? "var(--bio-green)" : (status === "known" ? "var(--amber-warn)" : "var(--text-faint)");

      const lecTagBadge = el("span", {
        className: `item-tag ${lec.tag.toLowerCase()}`,
        title: `Filtrovat podle tagu "${lec.tag}"`,
        onClick: (e) => {
          e.stopPropagation();
          e.preventDefault();
          state.selectedTag = state.selectedTag === lec.tag ? null : lec.tag;
          state.notify();
        }
      }, lec.tag);

      const lecItem = el("a", {
        href: `#/lecture/${lec.id}`,
        className: `tree-item ${state.activeRoute === `lecture/${lec.id}` ? "active" : ""}`
      }, [
        el("span", { style: { color: statusColor, fontWeight: "bold", fontSize: "11px" } }, statusIcon),
        el("span", { className: "item-title" }, lec.title),
        lecTagBadge
      ]);
      itemsList.appendChild(lecItem);
    });

    // Dojo Item
    if (isDojoMatch || !filterText) {
      const dojoItem = el("a", {
        href: `#/dojo/${mod.dojo.id}`,
        className: `tree-item ${state.activeRoute === `dojo/${mod.dojo.id}` ? "active" : ""}`,
        style: { color: "var(--accent)" }
      }, [
        el("span", {}, "⚡"),
        el("span", { className: "item-title", style: { fontWeight: "500" } }, mod.dojo.title),
        el("span", { className: "item-tag core" }, "Dojo")
      ]);
      itemsList.appendChild(dojoItem);
    }

    // Quiz Item
    if (isQuizMatch || !filterText) {
      const quizScore = state.getQuizScore(mod.quiz.id);
      const quizScoreText = quizScore ? `${quizScore.score}/${quizScore.total}` : "Test";

      const quizItem = el("a", {
        href: `#/quiz/${mod.quiz.id}`,
        className: `tree-item ${state.activeRoute === `quiz/${mod.quiz.id}` ? "active" : ""}`,
        style: { color: "var(--bio-green)" }
      }, [
        el("span", {}, "📝"),
        el("span", { className: "item-title", style: { fontWeight: "500" } }, mod.quiz.title),
        el("span", { className: "item-tag wow" }, quizScoreText)
      ]);
      itemsList.appendChild(quizItem);
    }

    modWrapper.appendChild(itemsList);
    container.appendChild(modWrapper);
  });
}
