/**
 * DrugEx Hub — Main Application Entry Point & Module Coordinator
 */

import { state } from "./state.js";
import { initRouter } from "./router.js";
import { renderTree } from "./tree.js";
import { el, showToast } from "./ui.js";

document.addEventListener("DOMContentLoaded", async () => {
  // Apply saved theme
  document.documentElement.setAttribute("data-theme", state.theme);

  // Load Curriculum and Thesis Guide
  try {
    const curRes = await fetch("/data/curriculum.json");
    if (curRes.ok) {
      state.curriculum = await curRes.json();
    }
    const thRes = await fetch("/data/thesis_guide.json");
    if (thRes.ok) {
      state.thesisGuide = await thRes.json();
    }
  } catch (err) {
    console.warn("Failed to load curriculum manifests:", err);
  }

  const viewport = document.getElementById("contentViewport");
  const treeContainer = document.getElementById("sidebarTree");
  const filterInput = document.getElementById("filterText");
  const tagChips = document.getElementById("tagChips");
  const btnTheme = document.getElementById("btnTheme");
  const btnSidebarToggle = document.getElementById("btnSidebarToggle");
  const workbench = document.getElementById("workbench");
  const userLabel = document.getElementById("userLabel");

  if (userLabel && state.user) {
    userLabel.textContent = state.user.username;
  }

  // Render initial tree
  const updateTree = () => {
    if (state.curriculum) {
      renderTree(treeContainer, state.curriculum);
    }
  };
  updateTree();

  // Subscribe state changes
  state.subscribe(() => {
    updateTree();
    if (userLabel && state.user) {
      userLabel.textContent = state.user.username;
    }
  });

  // Filter text listener
  if (filterInput) {
    filterInput.addEventListener("input", (e) => {
      state.searchQuery = e.target.value;
      updateTree();
    });
  }

  // Tag chips filtering
  if (tagChips) {
    tagChips.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      const tag = chip.dataset.tag;
      if (state.selectedTag === tag) {
        state.selectedTag = null;
        chip.classList.remove("active");
      } else {
        tagChips.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        state.selectedTag = tag;
      }
      updateTree();
    });
  }

  // Theme toggle
  if (btnTheme) {
    btnTheme.addEventListener("click", () => {
      state.toggleTheme();
      showToast(`Barevné schéma přepnuto na ${state.theme}`, "info");
    });
  }

  // Sidebar toggle
  if (btnSidebarToggle && workbench) {
    btnSidebarToggle.addEventListener("click", () => {
      workbench.classList.toggle("sidebar-collapsed");
    });
  }

  // Activity bar view switching
  document.querySelectorAll(".activity-btn[data-view]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".activity-btn[data-view]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const view = btn.dataset.view;
      if (view === "progress") {
        window.location.hash = "#/progress";
      } else if (view === "explorer") {
        if (workbench.classList.contains("sidebar-collapsed")) {
          workbench.classList.remove("sidebar-collapsed");
        }
      }
    });
  });

  // Global Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    // Ctrl+B: Toggle Sidebar
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
      e.preventDefault();
      if (workbench) workbench.classList.toggle("sidebar-collapsed");
    }
    // Ctrl+K: Search Command Palette
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openCommandPalette();
    }
  });

  // Profile / Login Modal
  const btnProfile = document.getElementById("btnProfile");
  if (btnProfile) {
    btnProfile.addEventListener("click", openLoginModal);
  }

  // Initialize Router
  initRouter(viewport);
});

function openCommandPalette() {
  let modal = document.getElementById("cmdModal");
  if (!modal) {
    modal = el("div", { id: "cmdModal", className: "modal-overlay", onClick: (e) => {
      if (e.target === modal) modal.classList.remove("open");
    }}, [
      el("div", { className: "command-palette" }, [
        el("input", {
          id: "cmdSearchInput",
          className: "palette-input",
          placeholder: "Vyhledat přednášku, dojo nebo test (např. 'ROCS', 'Pareto', 'SMILES')...",
          onInput: filterPaletteResults,
          onKeyDown: (e) => {
            if (e.key === "Escape") modal.classList.remove("open");
          }
        }),
        el("div", { id: "cmdResults", className: "palette-results" })
      ])
    ]);
    document.body.appendChild(modal);
  }

  modal.classList.add("open");
  const inp = document.getElementById("cmdSearchInput");
  if (inp) {
    inp.value = "";
    inp.focus();
    filterPaletteResults();
  }
}

function filterPaletteResults() {
  const query = (document.getElementById("cmdSearchInput")?.value || "").toLowerCase();
  const resultsContainer = document.getElementById("cmdResults");
  if (!resultsContainer || !state.curriculum) return;
  resultsContainer.innerHTML = "";

  const items = [];
  state.curriculum.modules.forEach(m => {
    m.lectures.forEach(l => items.push({ title: l.title, hash: `#/lecture/${l.id}`, type: "Přednáška" }));
    items.push({ title: m.dojo.title, hash: `#/dojo/${m.dojo.id}`, type: "Interactive Dojo" });
    items.push({ title: m.quiz.title, hash: `#/quiz/${m.quiz.id}`, type: "Test" });
  });

  const filtered = items.filter(it => !query || it.title.toLowerCase().includes(query));

  filtered.slice(0, 10).forEach(it => {
    const itemRow = el("div", {
      className: "palette-item",
      onClick: () => {
        window.location.hash = it.hash;
        document.getElementById("cmdModal")?.classList.remove("open");
      }
    }, [
      el("span", {}, it.title),
      el("span", { style: { fontSize: "11px", opacity: 0.7 } }, it.type)
    ]);
    resultsContainer.appendChild(itemRow);
  });
}

function openLoginModal() {
  const email = prompt("Zadejte VŠCHT / univerzitní e-mail studenta:", state.user.email);
  if (email) {
    const pwd = prompt("Zadejte heslo pro lokální SHA-256 autentizaci:", "heslo123");
    if (pwd) {
      state.setUser(email, pwd);
      showToast(`Přihlášen student: ${state.user.username}`, "success");
    }
  }
}
