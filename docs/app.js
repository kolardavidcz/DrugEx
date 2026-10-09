const { useState, useMemo, useEffect, useRef } = React;

// Concentrated Thematic Libraries Structure & Reference Database (Loaded from data/*.js)
const LIBRARY_SECTIONS = window.LIBRARY_SECTIONS;
const DATABASE = window.DATABASE;

// Symbol Version / Platform Tag Badge Formatter (Matching cppreference aesthetic)
function getSymbolTag(key, item) {
  if (key.startsWith("experimental::usrcat")) return "60D";
  if (key.startsWith("experimental::electroshape")) return "4D";
  if (key.startsWith("experimental::prolif")) return "ProLIF";
  if (key.startsWith("experimental::chemprop")) return "Chemprop";
  if (key.startsWith("experimental::scaffviz")) return "Scaffviz";
  if (key.startsWith("experimental::rascore")) return "RAScore";
  if (key.startsWith("experimental::vina")) return "Vina";
  if (key.startsWith("experimental::openfe")) return "OpenFE";
  if (key.startsWith("experimental::optuna")) return "Optuna";
  if (key.startsWith("experimental::strain")) return "MMFF94";
  if (key.startsWith("rdkit::Chem::rdShapeAlign")) return "RDKit";
  if (key.startsWith("rdkit::Chem::AllChem")) return "ETKDG";
  if (key.startsWith("rdkit::Chem::BRICS")) return "BRICS";
  if (key.startsWith("rdkit::Chem::Descriptors3D")) return "3D";
  if (key.startsWith("rdkit::Chem::Scaffolds")) return "Murcko";
  if (key.startsWith("rdkit::")) return "RDKit";
  if (key.startsWith("cdpkit::")) return "CDPKit";
  if (key.startsWith("qsprpred::")) return "QSPRPred";
  if (key.includes("rocs_rdkit")) return "RDKit";
  if (key.includes("rocs_cdpkit")) return "CDPKit";
  if (key.includes("rocs_openeye")) return "OpenEye";
  if (key.includes("conformer_generators::RDKit")) return "RDKit";
  if (key.includes("conformer_generators::CDPKit")) return "CDPKit";
  if (key.includes("conformer_generators::Omega")) return "Omega";
  if (key.includes("conformer_generators::Schrodinger")) return "Schrodinger";
  if (key.includes("graph_transformer::Block")) return "Graph";
  if (key.includes("sequence_transformer::Block")) return "Seq";
  if (key.includes("GraphTransformer") || key.includes("EdgeTransformer") || key.includes("SequenceTransformer")) return "GPU";
  if (key.includes("graph_transformer")) return "Graph";
  if (key.includes("GPT2Layer")) return "Transformer";
  if (key.includes("SequenceRNN") || key.includes("CharacterRNN")) return "PyTorch";
  if (key.includes("Pareto")) return "MORL";
  if (item && item.badges && item.badges.some(b => b.includes("v3.4") || b.includes("DrugEx 3.4+"))) return "v3.4";
  return null;
}

// Helper to determine if a symbol is an abstract base class or contract interface
function isAbstractSymbol(key, item) {
  if (!item && typeof DATABASE !== "undefined") item = DATABASE[key];
  if (!item || item.kind !== "class") return false;
  if (item.isAbstract === true) return true;
  if (item.badges && item.badges.some(b => {
    const l = b.toLowerCase();
    return l.includes("abstract") || l.includes("interface") || l.includes("protocol");
  })) {
    if (key.includes("Exception") || item.title.includes("Exception")) return false;
    return true;
  }
  if (item.synopsis && (item.synopsis.includes("@abstractmethod") || /\bABC\b/.test(item.synopsis) || item.synopsis.includes("ABCMeta"))) {
    return true;
  }
  if (item.description && (
    item.description.toLowerCase().startsWith("abstract") ||
    item.description.toLowerCase().includes("root abstract") ||
    item.description.toLowerCase().includes("abstract base") ||
    item.description.toLowerCase().includes("abstract contract")
  )) {
    return true;
  }
  return false;
}

// Helper to retrieve member functions for a symbol, ordering regular methods first and magic methods second (excluding __init__)
function getEffectiveMembers(item) {
  if (!item || !item.members) return [];
  const filtered = item.members.filter(m => !m.name.startsWith("__init__"));
  const regularMethods = filtered.filter(m => !m.name.startsWith("__"));
  const magicMethods = filtered.filter(m => m.name.startsWith("__"));
  return [...regularMethods, ...magicMethods];
}

// Helper to resolve Python magic dunder methods to their special operator/syntax signpost
function getMagicMethodOperator(name) {
  if (!name) return null;
  const match = name.match(/^(__[a-zA-Z0-9_]+__)/);
  if (!match) return null;
  const dunder = match[1];
  const map = {
    "__init__": { op: "(constructor)", label: null, cls: "op-constructor" },
    "__call__": { op: "obj(...)", label: "callable", cls: "op-call" },
    "__getitem__": { op: "obj[...]", label: "operator[]", cls: "op-subscript" },
    "__setitem__": { op: "obj[...] = v", label: "operator[]=", cls: "op-subscript" },
    "__len__": { op: "len(obj)", label: "length", cls: "op-len" },
    "__iter__": { op: "iter(obj)", label: "for x in obj", cls: "op-iter" },
    "__next__": { op: "next(obj)", label: "generator step", cls: "op-iter" },
    "__repr__": { op: "repr(obj)", label: "debug", cls: "op-misc" },
    "__str__": { op: "str(obj)", label: "print", cls: "op-misc" },
    "__enter__": { op: "with obj:", label: "context enter", cls: "op-call" },
    "__exit__": { op: "with obj:", label: "context exit", cls: "op-call" },
    "__contains__": { op: "x in obj", label: "membership", cls: "op-subscript" },
    "__add__": { op: "obj + other", label: "operator+", cls: "op-len" }
  };
  return map[dunder] || null;
}

// Helper resolving interactive micro-visualization type for symbols
function getMicroVizType(key) {
  if (!key) return null;
  if (
    key === "drugex::training::rewards::ParetoCrowdingDistance" ||
    key === "drugex::utils::pareto::get_Pareto_fronts" ||
    key === "drugex::training::rewards::ParetoRewardScheme" ||
    key === "drugex::training::rewards::RewardScheme" ||
    key === "drugex::training::rewards::WeightedSum"
  ) {
    return "pareto_crowding";
  }
  if (key === "drugex::training::rewards::ParetoTanimotoDistance") {
    return "pareto_tanimoto";
  }
  if (
    key === "drugex::training::scorers::ScoreModifier" ||
    key === "drugex::training::scorers::SmoothClippedScore" ||
    key === "drugex::training::scorers::ClippedScore" ||
    key === "drugex::training::scorers::MinMaxGaussian" ||
    key === "drugex::training::scorers::modifiers::ThresholdedLinear" ||
    key === "drugex::training::scorers::modifiers::SmoothHump" ||
    key === "drugex::training::scorers::modifiers::Squared" ||
    key === "drugex::training::scorers::modifiers::AbsoluteScore" ||
    key === "drugex::training::scorers::modifiers::Chained" ||
    key === "drugex::training::scorers::Gaussian" ||
    key === "drugex::training::scorers::Linear"
  ) {
    return "modifier_curve";
  }
  if (
    key === "drugex::training::environment::DrugExEnvironment" ||
    key === "drugex::training::environment::Environment" ||
    key === "drugex::training::explorers::SequenceExplorer" ||
    key === "drugex::training::explorers::FragSequenceExplorer" ||
    key === "drugex::training::explorers::GraphExplorer" ||
    key === "drugex::training::explorers::FragGraphExplorer" ||
    key === "drugex::training::explorers::Explorer" ||
    key === "drugex::training::explorers::FragExplorer" ||
    key === "drugex::training::scorers::SmilesChecker" ||
    key === "drugex::train::Reinforce"
  ) {
    return "rl_policy_cycle";
  }
  return null;
}

// Main React Application Component
function App() {
  const [theme, setTheme] = useState("light");
  const [activeCategory, setActiveCategory] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedKey, setSelectedKey] = useState(() => {
    const hash = window.location.hash.replace("#", "");
    return (hash && hash !== "main" && hash !== "index" && DATABASE[hash]) ? hash : null;
  });
  const [copied, setCopied] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => typeof window !== "undefined" ? window.innerWidth > 1024 : true);
  const [collapsedCategories, setCollapsedCategories] = useState({});
  const searchInputRef = useRef(null);

  // Handle Theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Responsive sidebar auto-hide on <= 1024px screens
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 1024) {
        setIsSidebarOpen(false);
      } else {
        setIsSidebarOpen(true);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Auto-expand category containing active symbol; close mobile drawer upon selection
  useEffect(() => {
    if (selectedKey) {
      for (const sec of LIBRARY_SECTIONS) {
        const hasKey = sec.subsections.some(sub => sub.keys.includes(selectedKey));
        if (hasKey) {
          setCollapsedCategories(prev => ({ ...prev, [sec.id]: false }));
          break;
        }
      }
      if (window.innerWidth <= 1024) {
        setIsSidebarOpen(false);
      }
    }
  }, [selectedKey]);

  // Syntax highlighting via Prism
  useEffect(() => {
    if (window.Prism) {
      window.Prism.highlightAll();
    }
  }, [selectedKey, theme]);

  // URL Hash deep linking with active hashchange listener and category synchronization
  useEffect(() => {
    const updateFromHash = () => {
      const hash = window.location.hash.replace("#", "");
      if (hash && hash !== "main" && hash !== "index" && DATABASE[hash]) {
        setSelectedKey(hash);
        setActiveCategory(DATABASE[hash].category);
      } else {
        setSelectedKey(null);
        setActiveCategory("all");
      }
    };
    window.addEventListener("hashchange", updateFromHash);
    updateFromHash();
    return () => window.removeEventListener("hashchange", updateFromHash);
  }, []);

  // Keyboard shortcuts: '/' or 'Ctrl+K' / 'Cmd+K' to focus search, 'Escape' to clear and blur
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isK = e.key.toLowerCase() === "k";
      const isSlash = e.key === "/";
      const isModifier = e.ctrlKey || e.metaKey;
      if (((isSlash && !isModifier) || (isK && isModifier)) && document.activeElement !== searchInputRef.current && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
        e.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      } else if (e.key === "Escape" && document.activeElement === searchInputRef.current) {
        setSearchTerm("");
        searchInputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const selectItem = (key) => {
    if (DATABASE[key]) {
      setSelectedKey(key);
      setActiveCategory(DATABASE[key].category);
      window.location.hash = key;
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const goToMain = () => {
    setSelectedKey(null);
    setActiveCategory("all");
    setSearchTerm("");
    window.location.hash = "main";
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const setCategory = (cat) => {
    setActiveCategory(cat);
    if (selectedKey && cat !== "all" && DATABASE[selectedKey]?.category !== cat) {
      const firstInCat = Object.keys(DATABASE).find(k => DATABASE[k].category === cat);
      if (firstInCat) {
        setSelectedKey(firstInCat);
        window.location.hash = firstInCat;
      }
    }
  };

  const toggleCategoryCollapse = (catId) => {
    setCollapsedCategories(prev => ({
      ...prev,
      [catId]: !prev[catId]
    }));
  };

  const term = searchTerm.toLowerCase().trim();

  // Filtered keys for sidebar in detail view
  const filteredKeys = useMemo(() => {
    return Object.keys(DATABASE).filter(key => {
      const item = DATABASE[key];
      const matchCat = activeCategory === "all" || item.category === activeCategory;
      const matchSearch = !term || 
        key.toLowerCase().includes(term) || 
        item.title.toLowerCase().includes(term) ||
        item.module.toLowerCase().includes(term) ||
        item.description.toLowerCase().includes(term) ||
        item.badges.some(b => b.toLowerCase().includes(term)) ||
        (item.synopsis && item.synopsis.toLowerCase().includes(term)) ||
        getEffectiveMembers(item).some(m => m.name.toLowerCase().includes(term) || m.desc.toLowerCase().includes(term)) ||
        (item.parameters && item.parameters.some(p => p.name.toLowerCase().includes(term) || p.desc.toLowerCase().includes(term)));
      return matchCat && matchSearch;
    });
  }, [term, activeCategory]);

  const filteredKeysSet = useMemo(() => new Set(filteredKeys), [filteredKeys]);

  // Matching keys for main index filtering
  const mainIndexMatches = useMemo(() => {
    if (!term) return { count: Object.keys(DATABASE).length, matchingKeysSet: null };
    const set = new Set();
    for (const key in DATABASE) {
      const item = DATABASE[key];
      if (
        key.toLowerCase().includes(term) ||
        item.title.toLowerCase().includes(term) ||
        item.module.toLowerCase().includes(term) ||
        item.description.toLowerCase().includes(term) ||
        item.badges.some(b => b.toLowerCase().includes(term)) ||
        (item.synopsis && item.synopsis.toLowerCase().includes(term)) ||
        getEffectiveMembers(item).some(m => m.name.toLowerCase().includes(term) || m.desc.toLowerCase().includes(term)) ||
        (item.parameters && item.parameters.some(p => p.name.toLowerCase().includes(term) || p.desc.toLowerCase().includes(term)))
      ) {
        set.add(key);
      }
    }
    return { count: set.size, matchingKeysSet: set };
  }, [term]);

  const currentItem = DATABASE[selectedKey] || DATABASE["drugex::training::environment::DrugExEnvironment"];

  const copyCode = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Category counts
  const counts = useMemo(() => {
    const res = { all: Object.keys(DATABASE).length, core: 0, ecosystem: 0, experimental: 0 };
    for (const k in DATABASE) {
      res[DATABASE[k].category]++;
    }
    return res;
  }, []);

  // Filtered library sections based on activeCategory
  const visibleSections = useMemo(() => {
    if (activeCategory === "core") {
      return LIBRARY_SECTIONS.filter(s => !["ecosystem", "experimental"].includes(s.id));
    } else if (activeCategory === "ecosystem") {
      return LIBRARY_SECTIONS.filter(s => s.id === "ecosystem");
    } else if (activeCategory === "experimental") {
      return LIBRARY_SECTIONS.filter(s => s.id === "experimental");
    }
    return LIBRARY_SECTIONS;
  }, [activeCategory]);

  // Balanced columns for 3-column index layout (matching authentic cppreference density)
  let col1 = [], col2 = [], col3 = [];
  if (activeCategory === "all") {
    col1 = visibleSections.filter(s => ["morl", "explorers", "neural", "modifiers", "molecules"].includes(s.id));
    col2 = visibleSections.filter(s => ["data", "workflows"].includes(s.id));
    col3 = visibleSections.filter(s => ["scoring", "ecosystem", "experimental"].includes(s.id));
  } else if (activeCategory === "core") {
    col1 = visibleSections.filter(s => ["morl", "explorers", "neural", "modifiers"].includes(s.id));
    col2 = visibleSections.filter(s => ["scoring", "data"].includes(s.id));
    col3 = visibleSections.filter(s => ["molecules", "workflows"].includes(s.id));
  } else if (activeCategory === "ecosystem" && visibleSections.length > 0) {
    const sec = visibleSections[0];
    col1 = [{ ...sec, subsections: sec.subsections.slice(0, 2) }];
    col2 = [{ ...sec, subsections: sec.subsections.slice(2, 3) }];
    col3 = [{ ...sec, subsections: sec.subsections.slice(3) }];
  } else if (activeCategory === "experimental" && visibleSections.length > 0) {
    const sec = visibleSections[0];
    col1 = [{ ...sec, subsections: sec.subsections.slice(0, 3) }];
    col2 = [{ ...sec, subsections: sec.subsections.slice(3, 5) }];
    col3 = [{ ...sec, subsections: sec.subsections.slice(5) }];
  } else {
    col1 = visibleSections;
  }

  const renderCategoryBlock = (sec) => {
    const subMatches = sec.subsections.map(sub => {
      const matchingKeys = sub.keys.filter(k => {
        if (!mainIndexMatches.matchingKeysSet) return true;
        return mainIndexMatches.matchingKeysSet.has(k);
      });
      return { subtitle: sub.subtitle, keys: matchingKeys };
    }).filter(sub => sub.keys.length > 0);

    if (subMatches.length === 0) return null;

    return (
      <div key={sec.id} id={`sec-${sec.id}`} className="cpp-category-block">
        <div className="cpp-cat-header">{sec.title}</div>
        {subMatches.map((sub, sIdx) => (
          <div key={sub.subtitle || sIdx} className="cpp-subsection">
            {sub.subtitle && <div className="cpp-sub-title">{sub.subtitle}</div>}
            <div className="cpp-symbol-list">
              {sub.keys.map((k, idx) => {
                const item = DATABASE[k];
                if (!item) return null;
                const isAbstract = isAbstractSymbol(k, item);
                const tag = getSymbolTag(k, item);
                return (
                  <React.Fragment key={k}>
                    {idx > 0 && <span className="cpp-sym-sep">–</span>}
                    <a 
                      href={`#${k}`}
                      className={`cpp-sym-link ${isAbstract ? "is-abstract" : ""}`}
                      onClick={(e) => { e.preventDefault(); selectItem(k); }}
                      title={`${item.title} (${item.module})${isAbstract ? " [Abstract Base Class]" : ""}`}
                    >
                      {item.title}
                    </a>
                    {isAbstract ? (
                      <span className="cpp-sym-badge cpp-sym-badge-abstract">(abstract)</span>
                    ) : (
                      tag && <span className="cpp-sym-badge">({tag})</span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div>
      {/* Top Bar */}
      <header className="top-header">
        <div className="brand-title" style={{ cursor: "pointer" }} onClick={goToMain} title="Return to Main Index">
          <img src="logo.png" alt="DrugEx Logo" style={{ width: "22px", height: "22px", borderRadius: "4px", objectFit: "contain", verticalAlign: "middle" }} onError={(e) => { e.target.style.display = 'none'; }} />
          <span className="brand-text-full">DrugEx C++ & Chemoinformatics Reference</span>
          <span className="brand-text-short">DrugEx Reference</span>
          <span className="brand-pill">v3.4 Complete Reference • {counts.all} Symbols</span>
        </div>

        <div className="header-nav">
          <button 
            className={`nav-tab ${selectedKey === null && activeCategory === "all" ? "active" : ""}`}
            onClick={() => { setActiveCategory("all"); goToMain(); }}
            title="View Concentrated Main Page Reference Index"
          >
            Main Index
          </button>
          <button 
            className={`nav-tab ${activeCategory === "all" && selectedKey !== null ? "active" : ""}`}
            onClick={() => setCategory("all")}
          >
            All Topics ({counts.all})
          </button>
          <button 
            className={`nav-tab ${activeCategory === "core" ? "active" : ""}`}
            onClick={() => setCategory("core")}
          >
            DrugEx Core ({counts.core})
          </button>
          <button 
            className={`nav-tab ${activeCategory === "ecosystem" ? "active" : ""}`}
            onClick={() => setCategory("ecosystem")}
          >
            Ecosystem ({counts.ecosystem})
          </button>
          <button 
            className={`nav-tab ${activeCategory === "experimental" ? "active" : ""}`}
            onClick={() => setCategory("experimental")}
          >
            Experimental ({counts.experimental})
          </button>

          <button 
            className="theme-toggle-btn"
            onClick={() => setTheme(t => t === "light" ? "dark" : "light")}
            title="Toggle Dark / Paper Theme"
          >
            {theme === "light" ? "Dark Theme" : "Paper Theme"}
          </button>
        </div>
      </header>

      {/* Search Row */}
      <div className="search-bar-row">
        {selectedKey === null ? (
          <div className="breadcrumbs">
            <span style={{ fontWeight: 700, color: "var(--text-main)" }}>DrugEx reference</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="current">Main Index</span>
            <span style={{ fontSize: "11.5px", color: "var(--text-muted)", marginLeft: "8px" }}>
              ({counts.all} symbols across 10 libraries)
            </span>
          </div>
        ) : (
          <div className="breadcrumbs">
            <button 
              className="sidebar-toggle-btn"
              onClick={() => setIsSidebarOpen(prev => !prev)}
              title={isSidebarOpen ? "Collapse Sidebar (◨)" : "Expand Sidebar (◨)"}
              style={{ marginRight: "6px" }}
            >
              <span>◨</span>
              <span style={{ fontSize: "11px" }}>{isSidebarOpen ? "Hide" : "Sidebar"}</span>
            </button>
            <button 
              className="back-to-index-btn"
              onClick={goToMain}
              title="Return to Concentrated Main Reference Index"
              style={{ marginRight: "10px" }}
            >
              ← Main Index
            </button>
            <a href="#main" onClick={(e) => { e.preventDefault(); goToMain(); }} className="breadcrumb-link">
              DrugEx reference
            </a>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="breadcrumb-module">{currentItem.module}</span>
            <span className="breadcrumb-sep">&gt;</span>
            <span className="current">{currentItem.title}</span>
          </div>
        )}

        <div className="search-input-wrap">
          <input 
            ref={searchInputRef}
            type="text"
            className="search-input"
            placeholder={selectedKey === null ? "Search all 234 symbols across libraries... (/ or Ctrl+K)" : "Search symbol, module, member function... (/ or Ctrl+K)"}
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter") {
                if (selectedKey === null && mainIndexMatches.matchingKeysSet && mainIndexMatches.matchingKeysSet.size > 0) {
                  const first = Array.from(mainIndexMatches.matchingKeysSet)[0];
                  if (first) selectItem(first);
                } else if (selectedKey !== null && filteredKeys.length > 0) {
                  selectItem(filteredKeys[0]);
                }
              }
            }}
          />
          {searchTerm ? (
            <button
              className="clear-search-btn"
              onClick={() => setSearchTerm("")}
              title="Clear search (Esc)"
              style={{
                background: "none",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                fontSize: "13px",
                padding: "0 6px",
                fontWeight: "bold",
                lineHeight: "1"
              }}
            >
              x
            </button>
          ) : (
            <span className="search-shortcut" title="Press / to focus search">/</span>
          )}
        </div>
      </div>

      {/* Main Body */}
      {selectedKey === null ? (
        /* Concentrated Main Page Library Index (cppreference.com style) */
        <div className="cpp-main-wrapper">
          <div className="cpp-box">
            <div className="cpp-header-title">DrugEx reference</div>
            <div className="cpp-header-subtitle">
              <span className="cpp-sub-ver">DrugEx v3.4</span>
              <span>|</span>
              <span className="cpp-sub-ver">RDKit 2023+</span>
              <span>|</span>
              <span className="cpp-sub-ver">CDPKit 1.2+</span>
              <span>|</span>
              <span className="cpp-sub-ver">QSPRPred 2.5+</span>
              <span>|</span>
              <span className="cpp-sub-ver">Chemprop v2</span>
              <span>|</span>
              <span className="cpp-sub-ver">OpenFE 1.0+</span>
              <span>|</span>
              <span className="cpp-sub-ver">PyTorch 2.0+</span>
              <span>|</span>
              <span className="cpp-sub-ver">AutoDock Vina</span>
              <span>|</span>
              <span className="cpp-sub-ver">Optuna 3.0+</span>
              <span>—</span>
              <span style={{ color: "var(--text-main)", fontWeight: 500 }}>
                Multi-Objective De Novo Drug Design &amp; 3D Chemoinformatics Standard Library
              </span>
              <span style={{ color: "var(--text-muted)", fontSize: "11px", marginLeft: "8px", fontStyle: "italic" }}>
                • Abstract base classes &amp; interfaces highlighted in <span style={{ color: "#64748b", fontWeight: 600 }}>gray</span>
              </span>
            </div>
            <hr className="cpp-header-divider" />

            {searchTerm && (
              <div className="cpp-search-banner">
                <span>
                  Found <strong>{mainIndexMatches.count}</strong> symbol{mainIndexMatches.count === 1 ? "" : "s"} matching "<strong>{searchTerm}</strong>".
                </span>
                <button className="cpp-search-clear-btn" onClick={() => setSearchTerm("")}>
                  Clear Filter (Esc)
                </button>
              </div>
            )}

            {mainIndexMatches.count === 0 ? (
              <div style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-muted)" }}>
                No symbols matching "{searchTerm}". Try searching by symbol name, module, or descriptor.
              </div>
            ) : (
              <div className="cpp-grid-3col">
                {col1.length > 0 && <div className="cpp-col">{col1.map(renderCategoryBlock)}</div>}
                {col2.length > 0 && <div className="cpp-col">{col2.map(renderCategoryBlock)}</div>}
                {col3.length > 0 && <div className="cpp-col">{col3.map(renderCategoryBlock)}</div>}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Detailed Symbol Reference Page */
        <div className="main-body">
          {/* Sidebar */}
          <aside className={`sidebar ${!isSidebarOpen ? "collapsed" : ""} ${isSidebarOpen ? "drawer-open" : ""}`}>
            <div className="sidebar-section-title">
              <span>Libraries & Symbols</span>
              <span>{filteredKeys.length}</span>
            </div>

            {LIBRARY_SECTIONS.map(sec => {
              const secMatchingKeys = [];
              for (const sub of sec.subsections) {
                for (const k of sub.keys) {
                  if (filteredKeysSet.has(k)) {
                    secMatchingKeys.push(k);
                  }
                }
              }
              if (secMatchingKeys.length === 0) return null;

              const isCollapsed = term ? false : Boolean(collapsedCategories[sec.id]);
              const tierClass = sec.id === "ecosystem" ? "dot-ecosystem" : sec.id === "experimental" ? "dot-experimental" : "dot-core";

              return (
                <div key={sec.id} className="tree-category">
                  <div 
                    className="tree-category-header"
                    onClick={() => toggleCategoryCollapse(sec.id)}
                    title={`Click to ${isCollapsed ? "expand" : "collapse"} ${sec.title}`}
                  >
                    <div className="tree-cat-left">
                      <span className="tree-chevron">{isCollapsed ? "▶" : "▼"}</span>
                      <span className={`tree-tier-dot ${tierClass}`} />
                      <span className="tree-category-title">{sec.title}</span>
                    </div>
                    <span className="tree-cat-count">{secMatchingKeys.length}</span>
                  </div>

                  {!isCollapsed && (
                    <div className="tree-items-list">
                      {sec.subsections.map((sub, sIdx) => {
                        const subKeys = sub.keys.filter(k => filteredKeysSet.has(k));
                        if (subKeys.length === 0) return null;
                        return (
                          <div key={sub.subtitle || sIdx} className="tree-sub-group">
                            {sub.subtitle && (
                              <div className="tree-sub-title" title={sub.subtitle}>
                                {sub.subtitle}
                              </div>
                            )}
                            {subKeys.map(k => {
                              const item = DATABASE[k];
                              if (!item) return null;
                              const isActive = k === selectedKey;
                              const isAbstract = isAbstractSymbol(k, item);
                              return (
                                <div 
                                  key={k} 
                                  className={`sidebar-item ${isActive ? "active" : ""} ${isAbstract ? "is-abstract" : ""}`}
                                  onClick={() => selectItem(k)}
                                  title={`${item.title} (${item.module})${isAbstract ? " [Abstract Base Class]" : ""}`}
                                >
                                  <span>{item.title}</span>
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}

            {filteredKeys.length === 0 && (
              <div style={{ padding: "20px 16px", color: "var(--text-muted)", fontSize: "12px" }}>
                No symbols matching "{searchTerm}".
              </div>
            )}
          </aside>

          {/* Mobile / Half-screen Drawer Backdrop */}
          {isSidebarOpen && (
            <div className="sidebar-backdrop" onClick={() => setIsSidebarOpen(false)} />
          )}

          {/* Content Area */}
          <main className="content-area">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px", flexWrap: "wrap", marginBottom: "8px" }}>
              <div>
                <div className="entry-namespace">
                  {currentItem.module}
                  {isAbstractSymbol(selectedKey, currentItem) && (
                    <span className="entry-abstract-tag">abstract base class</span>
                  )}
                </div>
                <h1 className="entry-title" style={{ margin: "2px 0 0 0" }}>{currentItem.title}</h1>
              </div>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <button 
                  className="back-to-index-btn"
                  onClick={goToMain}
                  title="Return to Concentrated Main Reference Index"
                  style={{ padding: "6px 12px", fontSize: "12px" }}
                >
                  ← Main Index
                </button>
                <button 
                  className="copy-btn"
                  onClick={() => copyCode(window.location.origin + window.location.pathname + "#" + selectedKey)}
                  title="Copy permanent deep link to this symbol"
                  style={{ padding: "6px 12px", fontSize: "12px" }}
                >
                  Share Permalink
                </button>
              </div>
            </div>

            <div className="entry-badge-row">
              {currentItem.badges.map(badge => {
                const isAbsBadge = badge.toLowerCase().includes("abstract") || badge.toLowerCase().includes("interface") || badge.toLowerCase().includes("protocol");
                const isVersionBadge = badge.toLowerCase().includes("v3") || badge.toLowerCase().includes("3.4");
                let badgeClass = "pill-badge";
                if (isVersionBadge) badgeClass += " pill-version";
                else if (isAbsBadge) badgeClass += " pill-abstract";
                return (
                  <span key={badge} className={badgeClass}>
                    {badge}
                  </span>
                );
              })}
            </div>

            {/* Description */}
            <p className="description">{currentItem.description}</p>

            {/* Notes Box */}
            {currentItem.notes && (
              <div className="note-box">
                <strong>Practical Note:</strong> {currentItem.notes}
              </div>
            )}

            {/* Constructor / Synopsis Box */}
            <div className="synopsis-box">
              <div className="synopsis-header">Defined in header &lt;{currentItem.module}&gt;</div>
              <pre className="synopsis-code language-python" style={{ margin: 0, padding: 0, background: "transparent" }}><code className="language-python">{currentItem.synopsis}</code></pre>
            </div>

            {/* Constructor Parameters Table */}
            {currentItem.parameters && currentItem.parameters.length > 0 && (
              <div>
                <h2 className="section-header">
                  {currentItem.kind === "class" ? "Constructor Parameters" : "Parameters"}
                </h2>
                {currentItem.kind === "class" && (
                  <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "-6px", marginBottom: "10px" }}>
                    Arguments accepted by the <code>__init__</code> constructor when instantiating this class.
                  </p>
                )}
                <ul className="param-list">
                  {currentItem.parameters.map(param => (
                    <li key={param.name}>
                      <span className="param-name">{param.name}</span>
                      {" - "}
                      <span className="param-type">[{param.type}]</span>
                      {" "}: {param.desc}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Interactive Collapsible Micro-Visualization Panel */}
            {(() => {
              const vizType = getMicroVizType(selectedKey);
              return vizType ? (
                <MicroVizPanel key={selectedKey} symbolKey={selectedKey} vizType={vizType} item={currentItem} theme={theme} />
              ) : null;
            })()}

            {/* Member Functions Table */}
            {(() => {
              const effectiveMembers = getEffectiveMembers(currentItem);
              return effectiveMembers.length > 0 && (
                <div>
                  <h2 className="section-header">Member Functions</h2>
                  <table className="ref-table">
                    <thead>
                      <tr>
                        <th style={{ width: "42%" }}>Function / Operator</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {effectiveMembers.map(m => {
                        const opInfo = getMagicMethodOperator(m.name);
                        return (
                          <tr key={m.name}>
                            <td className="code-cell">
                              {opInfo ? (
                                <div className="member-fn-cell">
                                  <div className="member-op-row">
                                    <span className={`member-op-badge ${opInfo.cls}`}>{opInfo.op}</span>
                                    {opInfo.label && <span className="member-op-label">{opInfo.label}</span>}
                                  </div>
                                  <span className="member-fn-sig">{m.name}</span>
                                </div>
                              ) : (
                                <span className="member-fn-sig">{m.name}</span>
                              )}
                            </td>
                            <td>{m.desc}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              );
            })()}

            {/* Member Variables Table */}
            {currentItem.variables && currentItem.variables.length > 0 && (
              <div>
                <h2 className="section-header">Member Objects & Attributes</h2>
                <table className="ref-table">
                  <thead>
                    <tr>
                      <th style={{ width: "30%" }}>Attribute</th>
                      <th style={{ width: "25%" }}>Type</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItem.variables.map(v => (
                      <tr key={v.name}>
                        <td className="code-cell">{v.name}</td>
                        <td style={{ fontFamily: "var(--font-mono)", fontSize: "11.5px" }}>{v.type}</td>
                        <td>{v.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Code Example */}
            {currentItem.example && (
              <div>
                <h2 className="section-header">Executable Example</h2>
                <div className="code-card-wrap">
                  <div className="code-card-header">
                    <span>Python 3.10+ Executable Demonstration</span>
                    <button 
                      className="copy-btn"
                      onClick={() => copyCode(currentItem.example.code)}
                    >
                      {copied ? "Copied!" : "Copy Code"}
                    </button>
                  </div>
                  <pre className="code-content language-python"><code className="language-python">{currentItem.example.code}</code></pre>
                  {currentItem.example.output && (
                    <div className="terminal-output">
                      <div style={{ color: "#94a3b8", marginBottom: "4px", fontSize: "10.5px" }}>&gt; Standard Output:</div>
                      <pre style={{ margin: 0 }}>{currentItem.example.output}</pre>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* See Also */}
            {currentItem.seeAlso && currentItem.seeAlso.length > 0 && (
              <div>
                <h2 className="section-header">See Also</h2>
                <table className="ref-table">
                  <thead>
                    <tr>
                      <th style={{ width: "45%" }}>Symbol</th>
                      <th>Category</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItem.seeAlso.map(sa => {
                      const target = DATABASE[sa];
                      const isTargetAbstract = target ? isAbstractSymbol(sa, target) : false;
                      return (
                        <tr key={sa}>
                          <td className="code-cell">
                            <a 
                              href={`#${sa}`} 
                              className={isTargetAbstract ? "is-abstract-link" : ""}
                              onClick={(e) => { e.preventDefault(); selectItem(sa); }}
                              title={isTargetAbstract ? `${target ? target.title : sa} [Abstract Base Class]` : undefined}
                            >
                              {target ? target.title : sa}
                            </a>
                            {isTargetAbstract && <span className="see-also-abstract-badge">(abstract)</span>}
                          </td>
                          <td>{target ? `${target.category.toUpperCase()} - ${target.description.slice(0, 70)}...` : sa}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

          </main>
        </div>
      )}
    </div>
  );
}


// =========================================================================
// Micro-Visualizations Engine & Interactive SVG Components
// =========================================================================

function MicroVizPanel({ symbolKey, vizType, item, theme }) {
  const titles = {
    pareto_crowding: {
      title: "Pareto Frontier & NSGA-II Crowding Distance (2D Objective Space)",
      hint: "Interactive Non-Dominated Sorting & Isolation Cuboid Mechanics"
    },
    pareto_tanimoto: {
      title: "Chemical Diversity Tie-Breaking via Fingerprint Distance",
      hint: "Pairwise ECFP6 Tanimoto Dissimilarity for Non-Dominated Solutions"
    },
    modifier_curve: {
      title: "Continuous Desirability Transfer Function Curve: f(x) -> [0, 1]",
      hint: "Sigmoidal Logistic vs Piecewise Linear Gradients and Threshold Anchors"
    },
    rl_policy_cycle: {
      title: "RL Policy Gradient Actor-Prior Feedback Cycle",
      hint: "Sequence/Graph Rollout, Frozen Prior Mixing, SmilesChecker Gate, and REINFORCE Loss"
    }
  };

  const meta = titles[vizType] || { title: "Visual Intuition", hint: "Interactive Mechanics" };

  return (
    <details className="micro-viz-panel">
      <summary className="micro-viz-summary">
        <div className="micro-viz-summary-left">
          <span className="micro-viz-badge">Visual Intuition</span>
          <span className="micro-viz-title">{meta.title}</span>
        </div>
        <span className="micro-viz-hint">[Click to explore interactive diagram]</span>
      </summary>
      <div className="micro-viz-body">
        {vizType === "pareto_crowding" && <ParetoCrowdingViz theme={theme} symbolKey={symbolKey} />}
        {vizType === "pareto_tanimoto" && <ParetoTanimotoViz theme={theme} symbolKey={symbolKey} />}
        {vizType === "modifier_curve" && <ModifierCurvesViz theme={theme} symbolKey={symbolKey} />}
        {vizType === "rl_policy_cycle" && <RLPolicyCycleViz theme={theme} symbolKey={symbolKey} />}
      </div>
    </details>
  );
}

function ParetoCrowdingViz({ theme, symbolKey }) {
  const [activePointId, setActivePointId] = useState("B");

  const points = [
    { id: "A", name: "Candidate A (Potency Extremum)", f1: 0.94, f2: 0.35, front: 0, d: "inf (Boundary)", rank: "Rank 4 (Front 0)", reward: 0.800, note: "Boundary solution with maximal f1 potency. Assigned infinite crowding distance (d = inf) to guarantee preservation of frontier boundary extremities." },
    { id: "B", name: "Candidate B (Isolated Lead)", f1: 0.78, f2: 0.70, front: 0, d: "1.428 (High)", rank: "Rank 5 (Front 0)", reward: 1.000, note: "Interior point flanked by Cand C (0.58, 0.84) and Cand A (0.94, 0.35). Spans large bounding cuboid [0.58, 0.94] x [0.35, 0.84]. Top policy gradient reward awarded to isolated diversity." },
    { id: "C", name: "Candidate C (Crowded Analog 1)", f1: 0.58, f2: 0.84, front: 0, d: "0.245 (Low)", rank: "Rank 2 (Front 0)", reward: 0.400, note: "Closely packs with Cand D in objective space. Flanked by Cand D and Cand B. Cuboid is tightly constrained; crowding distance penalty demotes candidate." },
    { id: "D", name: "Candidate D (Crowded Analog 2)", f1: 0.52, f2: 0.87, front: 0, d: "0.210 (Low)", rank: "Rank 1 (Front 0)", reward: 0.200, note: "Redundant cluster solution adjacent to Cand C. Squeezed in objective space; tightly bounded cuboid reduces reward to discourage mode collapse." },
    { id: "E", name: "Candidate E (QED Extremum)", f1: 0.32, f2: 0.95, front: 0, d: "inf (Boundary)", rank: "Rank 3 (Front 0)", reward: 0.600, note: "Boundary solution with maximal f2 QED drug-likeness. Assigned infinite crowding distance (d = inf) to guarantee preservation of frontier boundary extremities." },
    { id: "F", name: "Candidate F (Dominated Sub-lead)", f1: 0.68, f2: 0.52, front: 1, d: "0.850", rank: "Rank 2 (Front 1)", reward: 0.125, note: "Dominated by Candidate B (0.78 > 0.68 and 0.70 > 0.52). Partitioned into secondary Pareto Front 1." },
    { id: "G", name: "Candidate G (Dominated Moderate)", f1: 0.44, f2: 0.68, front: 1, d: "0.620", rank: "Rank 1 (Front 1)", reward: 0.062, note: "Dominated by Candidate C (0.58 > 0.44 and 0.84 > 0.68) and Candidate D. Partitioned into secondary Pareto Front 1." },
    { id: "H", name: "Candidate H (Sub-optimal Baseline)", f1: 0.25, f2: 0.40, front: 2, d: "inf", rank: "Rank 1 (Front 2)", reward: 0.000, note: "Dominated across all objectives by multiple candidates. Partitioned into Front 2; receives zero reward." }
  ];

  const activePoint = points.find(p => p.id === activePointId) || points[1];

  // Coordinate mapping: 680 x 260 canvas
  const mapX = (f1) => 70 + f1 * 540;
  const mapY = (f2) => 220 - f2 * 180;

  // Front 0 curve path: E -> D -> C -> B -> A
  const f0Path = `M ${mapX(0.32)} ${mapY(0.95)} L ${mapX(0.52)} ${mapY(0.87)} L ${mapX(0.58)} ${mapY(0.84)} L ${mapX(0.78)} ${mapY(0.70)} L ${mapX(0.94)} ${mapY(0.35)}`;
  // Front 1 curve path: G -> F
  const f1Path = `M ${mapX(0.44)} ${mapY(0.68)} L ${mapX(0.68)} ${mapY(0.52)}`;

  const isDark = theme === "dark";
  const axisColor = isDark ? "#475569" : "#cbd5e1";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";
  const textColor = isDark ? "#94a3b8" : "#64748b";

  let cuboidRect = null;
  let ghostCuboid = null;

  if (activePointId === "B") {
    cuboidRect = {
      x: mapX(0.58),
      y: mapY(0.84),
      w: mapX(0.94) - mapX(0.58),
      h: mapY(0.35) - mapY(0.84),
      df1: 0.36,
      df2: 0.49,
      color: "#0284c7",
      title: "d_B = 1.428 (Isolated)"
    };
    ghostCuboid = {
      x: mapX(0.52),
      y: mapY(0.87),
      w: mapX(0.78) - mapX(0.52),
      h: mapY(0.70) - mapY(0.87),
      color: "#d97706",
      title: "d_C,D ~ 0.23 (Crowded)"
    };
  } else if (activePointId === "C") {
    cuboidRect = {
      x: mapX(0.52),
      y: mapY(0.87),
      w: mapX(0.78) - mapX(0.52),
      h: mapY(0.70) - mapY(0.87),
      df1: 0.26,
      df2: 0.17,
      color: "#d97706",
      title: "d_C = 0.245 (Crowded)"
    };
    ghostCuboid = {
      x: mapX(0.58),
      y: mapY(0.84),
      w: mapX(0.94) - mapX(0.58),
      h: mapY(0.35) - mapY(0.84),
      color: "#0284c7",
      title: "d_B = 1.428 (Isolated)"
    };
  } else if (activePointId === "D") {
    cuboidRect = {
      x: mapX(0.32),
      y: mapY(0.95),
      w: mapX(0.58) - mapX(0.32),
      h: mapY(0.84) - mapY(0.95),
      df1: 0.26,
      df2: 0.11,
      color: "#d97706",
      title: "d_D = 0.210 (Crowded)"
    };
    ghostCuboid = {
      x: mapX(0.58),
      y: mapY(0.84),
      w: mapX(0.94) - mapX(0.58),
      h: mapY(0.35) - mapY(0.84),
      color: "#0284c7",
      title: "d_B = 1.428 (Isolated)"
    };
  }

  return (
    <div>
      <div className="micro-viz-svg-wrap">
        <svg className="micro-viz-svg" viewBox="0 0 680 260" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="front0Grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0284c7" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#0d9488" stopOpacity="0.05" />
            </linearGradient>
            <marker id="paretoArrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#fb7185" />
            </marker>
          </defs>

          {/* Grid Lines */}
          {[0.2, 0.4, 0.6, 0.8, 1.0].map(v => (
            <g key={`grid-${v}`}>
              <line x1={mapX(v)} y1={40} x2={mapX(v)} y2={220} stroke={gridColor} strokeWidth="1" strokeDasharray="3,3" />
              <line x1={70} y1={mapY(v)} x2={610} y2={mapY(v)} stroke={gridColor} strokeWidth="1" strokeDasharray="3,3" />
              <text x={mapX(v)} y={235} fill={textColor} fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">{v.toFixed(1)}</text>
              <text x={62} y={mapY(v) + 3} fill={textColor} fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">{v.toFixed(1)}</text>
            </g>
          ))}

          {/* Axes */}
          <line x1={70} y1={220} x2={615} y2={220} stroke={axisColor} strokeWidth="1.5" />
          <line x1={70} y1={220} x2={70} y2={35} stroke={axisColor} strokeWidth="1.5" />
          <text x={340} y={252} fill="var(--text-main)" fontSize="11" fontWeight="600" textAnchor="middle">Objective f1: Target Potency / Activity (pIC50 Desirability)</text>
          <text x={18} y={130} fill="var(--text-main)" fontSize="11" fontWeight="600" textAnchor="middle" transform="rotate(-90, 18, 130)">Objective f2: Drug-Likeness (QED Desirability)</text>

          {/* Ghost Cuboid */}
          {ghostCuboid && (
            <g>
              <rect
                x={ghostCuboid.x}
                y={ghostCuboid.y}
                width={ghostCuboid.w}
                height={ghostCuboid.h}
                fill={ghostCuboid.color === "#0284c7" ? "rgba(2, 132, 199, 0.06)" : "rgba(245, 158, 11, 0.06)"}
                stroke={ghostCuboid.color}
                strokeWidth="1"
                strokeDasharray="2,2"
                rx="3"
                opacity="0.7"
              />
              <text x={ghostCuboid.x + 4} y={ghostCuboid.y + 11} fill={ghostCuboid.color} fontSize="8.5" fontFamily="var(--font-mono)" opacity="0.8">
                {ghostCuboid.title}
              </text>
            </g>
          )}

          {/* Active Cuboid around selected point */}
          {cuboidRect && (
            <g>
              <rect
                x={cuboidRect.x}
                y={cuboidRect.y}
                width={cuboidRect.w}
                height={cuboidRect.h}
                fill={cuboidRect.color === "#0284c7" ? "rgba(2, 132, 199, 0.16)" : "rgba(245, 158, 11, 0.14)"}
                stroke={cuboidRect.color}
                strokeWidth="2"
                strokeDasharray="4,3"
                rx="4"
              />
              <text x={cuboidRect.x + 6} y={cuboidRect.y + 14} fill={cuboidRect.color} fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)">
                {cuboidRect.title}
              </text>
              <text x={cuboidRect.x + cuboidRect.w / 2} y={cuboidRect.y + cuboidRect.h - 4} fill={cuboidRect.color} fontSize="8.5" fontFamily="var(--font-mono)" textAnchor="middle">
                Delta f1={cuboidRect.df1.toFixed(2)}, Delta f2={cuboidRect.df2.toFixed(2)}
              </text>
            </g>
          )}

          {/* Boundary Rays for Extremum points A & E */}
          {activePointId === "A" && (
            <g>
              <line x1={mapX(0.94)} y1={mapY(0.35)} x2={mapX(1.0)} y2={mapY(0.35)} stroke="#0284c7" strokeWidth="2" strokeDasharray="3,2" />
              <line x1={mapX(0.94)} y1={mapY(0.35)} x2={mapX(0.94)} y2={mapY(0.0)} stroke="#0284c7" strokeWidth="2" strokeDasharray="3,2" />
              <rect x={mapX(0.80)} y={mapY(0.20)} width={175} height={20} rx="3" fill="var(--bg-card)" stroke="#0284c7" strokeWidth="1" />
              <text x={mapX(0.80) + 6} y={mapY(0.20) + 14} fill="#0284c7" fontSize="9" fontWeight="700" fontFamily="var(--font-mono)">
                Boundary Extremum: d = inf
              </text>
            </g>
          )}
          {activePointId === "E" && (
            <g>
              <line x1={mapX(0.32)} y1={mapY(0.95)} x2={mapX(0.0)} y2={mapY(0.95)} stroke="#0284c7" strokeWidth="2" strokeDasharray="3,2" />
              <line x1={mapX(0.32)} y1={mapY(0.95)} x2={mapX(0.32)} y2={mapY(1.0)} stroke="#0284c7" strokeWidth="2" strokeDasharray="3,2" />
              <rect x={mapX(0.04)} y={mapY(0.98)} width={175} height={20} rx="3" fill="var(--bg-card)" stroke="#0284c7" strokeWidth="1" />
              <text x={mapX(0.04) + 6} y={mapY(0.98) + 14} fill="#0284c7" fontSize="9" fontWeight="700" fontFamily="var(--font-mono)">
                Boundary Extremum: d = inf
              </text>
            </g>
          )}

          {/* Domination Indicator */}
          {activePointId === "F" && (
            <g>
              <line x1={mapX(0.78)} y1={mapY(0.70)} x2={mapX(0.68)} y2={mapY(0.52)} stroke="#fb7185" strokeWidth="2" strokeDasharray="4,2" markerEnd="url(#paretoArrow)" />
              <rect x={mapX(0.55)} y={mapY(0.48)} width={215} height={20} rx="3" fill="var(--bg-card)" stroke="#fb7185" strokeWidth="1" />
              <text x={mapX(0.55) + 6} y={mapY(0.48) + 14} fill="#fb7185" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)">
                Dominated by B (0.78 &gt; 0.68, 0.70 &gt; 0.52)
              </text>
            </g>
          )}
          {activePointId === "G" && (
            <g>
              <line x1={mapX(0.58)} y1={mapY(0.84)} x2={mapX(0.44)} y2={mapY(0.68)} stroke="#fb7185" strokeWidth="2" strokeDasharray="4,2" markerEnd="url(#paretoArrow)" />
              <rect x={mapX(0.32)} y={mapY(0.62)} width={215} height={20} rx="3" fill="var(--bg-card)" stroke="#fb7185" strokeWidth="1" />
              <text x={mapX(0.32) + 6} y={mapY(0.62) + 14} fill="#fb7185" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)">
                Dominated by C (0.58 &gt; 0.44, 0.84 &gt; 0.68)
              </text>
            </g>
          )}

          {/* Front 0 Curve */}
          <path d={f0Path} fill="none" stroke="#0284c7" strokeWidth="2.5" />
          <text x={mapX(0.35)} y={mapY(0.96) - 8} fill="#0284c7" fontSize="11" fontWeight="700" fontFamily="var(--font-mono)">
            Front 0 (Optimal Frontier)
          </text>

          {/* Front 1 Dashed Curve */}
          <path d={f1Path} fill="none" stroke="#fb7185" strokeWidth="1.8" strokeDasharray="5,4" />
          <text x={mapX(0.46)} y={mapY(0.70) + 16} fill="#fb7185" fontSize="10.5" fontWeight="600" fontFamily="var(--font-mono)">
            Front 1 (Dominated Tier)
          </text>

          {/* Candidate Points */}
          {points.map(p => {
            const px = mapX(p.f1);
            const py = mapY(p.f2);
            const isActive = p.id === activePointId;
            const isF0 = p.front === 0;
            const isF1 = p.front === 1;
            const dotColor = isF0 ? "#0284c7" : isF1 ? "#fb7185" : "#64748b";

            return (
              <g
                key={p.id}
                style={{ cursor: "pointer" }}
                onClick={() => setActivePointId(p.id)}
                onMouseEnter={() => setActivePointId(p.id)}
              >
                {isActive && (
                  <circle cx={px} cy={py} r="12" fill="none" stroke={dotColor} strokeWidth="2" strokeDasharray="2,2" opacity="0.8" />
                )}
                <circle
                  cx={px}
                  cy={py}
                  r={isActive ? "7" : "5"}
                  fill={dotColor}
                  stroke="#ffffff"
                  strokeWidth="1.5"
                />
                <text
                  x={px}
                  y={py - 10}
                  fill={isActive ? "var(--text-main)" : textColor}
                  fontSize={isActive ? "11.5" : "10"}
                  fontWeight={isActive ? "700" : "500"}
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                >
                  {p.id}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="micro-viz-legend">
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#0284c7" }}></span>
          Front 0 (Non-Dominated Optimal)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#fb7185" }}></span>
          Front 1 (Dominated Tier)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#64748b" }}></span>
          Front 2 (Sub-optimal Baseline)
        </span>
        <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--text-muted)" }}>
          Click or hover any candidate point to inspect crowding box and rank
        </span>
      </div>

      <div className="micro-viz-readout">
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Selected:</span>
          <span className="micro-viz-readout-val" style={{ color: "var(--text-main)" }}>{activePoint.name}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Coordinates:</span>
          <span className="micro-viz-readout-val">f1={activePoint.f1.toFixed(2)}, f2={activePoint.f2.toFixed(2)}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Tier:</span>
          <span className="micro-viz-readout-val" style={{ color: activePoint.front === 0 ? "var(--accent-cyan)" : "var(--accent-rose)" }}>
            Pareto Front {activePoint.front}
          </span>
        </div>
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">NSGA-II Crowding Distance (d_i):</span>
          <span className="micro-viz-readout-val">{activePoint.d}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Policy Reward:</span>
          <span className="micro-viz-readout-val">R = {activePoint.reward.toFixed(3)} ({activePoint.rank})</span>
        </div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
          {activePoint.note}
        </div>
      </div>
    </div>
  );
}

function ParetoTanimotoViz({ theme, symbolKey }) {
  const [activeMolId, setActiveMolId] = useState("macro");

  const candidates = [
    { id: "macro", name: "Macrocyclic Lactam", core: "12-membered Lactam Core", f1: 0.80, f2: 0.86, dist: 0.89, sim: 0.11, rank: 5, reward: 1.000, color: "#10b981", status: "Top Structural Novelty", desc: "Highest average fingerprint dissimilarity (1 - Tc = 0.89) to all other front members. Expands chemical series into novel structural space; awarded top RL reward." },
    { id: "spiro", name: "Spiro Indane", core: "Spiro[piperidine-indane]", f1: 0.82, f2: 0.84, dist: 0.85, sim: 0.15, rank: 4, reward: 0.800, color: "#06b6d4", status: "Novel 3D Scaffold", desc: "Rigid three-dimensional spiro scaffold. High structural diversity distance (1 - Tc = 0.85); prioritized for multi-target exploration." },
    { id: "indole", name: "Indole Lead", core: "Indole-3-carboxamide", f1: 0.85, f2: 0.80, dist: 0.82, sim: 0.18, rank: 3, reward: 0.600, color: "#3b82f6", status: "Distinct Series", desc: "Established planar heteroaromatic core, structurally distant from quinazoline analogues. Maintains medium-high reward rank." },
    { id: "quin1", name: "Quinazoline Analog A", core: "Quinazoline-4-amino", f1: 0.86, f2: 0.81, dist: 0.11, sim: 0.89, rank: 2, reward: 0.400, color: "#f59e0b", status: "Redundant Cluster", desc: "High pairwise similarity (Tc = 0.89) with Quinazoline Analog B. Intra-front Tanimoto tie-breaking demotes the second cluster member to penalize duplicate exploration." },
    { id: "quin2", name: "Quinazoline Analog B", core: "Quinazoline-4-chloro", f1: 0.84, f2: 0.82, dist: 0.11, sim: 0.89, rank: 1, reward: 0.200, color: "#ef4444", status: "Redundant Cluster (Demoted)", desc: "Lowest intra-front diversity distance (1 - Tc = 0.11). Heavily penalized by ParetoTanimotoDistance to prevent generator mode collapse into a single chemotype." }
  ];

  const active = candidates.find(c => c.id === activeMolId) || candidates[0];

  const nodePositions = {
    macro:  { x: 110, y: 70 },
    spiro:  { x: 260, y: 55 },
    indole: { x: 85,  y: 180 },
    quin1:  { x: 235, y: 175 },
    quin2:  { x: 375, y: 175 }
  };

  const isDark = theme === "dark";
  const edgeRed = isDark ? "#f87171" : "#ef4444";
  const edgeGreen = isDark ? "#34d399" : "#10b981";

  const pairwiseTc = {
    "macro-spiro": 0.15, "spiro-macro": 0.15,
    "macro-indole": 0.18, "indole-macro": 0.18,
    "macro-quin1": 0.11, "quin1-macro": 0.11,
    "macro-quin2": 0.08, "quin2-macro": 0.08,
    "spiro-indole": 0.22, "indole-spiro": 0.22,
    "spiro-quin1": 0.16, "quin1-spiro": 0.16,
    "spiro-quin2": 0.14, "quin2-spiro": 0.14,
    "indole-quin1": 0.24, "quin1-indole": 0.24,
    "indole-quin2": 0.21, "quin2-indole": 0.21,
    "quin1-quin2": 0.89, "quin2-quin1": 0.89
  };

  return (
    <div>
      <div className="micro-viz-svg-wrap">
        <svg className="micro-viz-svg" viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg">
          <rect x="15" y="15" width="415" height="220" fill="none" stroke="var(--border-subtle)" strokeWidth="1" rx="6" />
          <text x="30" y="36" fill="var(--text-muted)" fontSize="10" fontWeight="700" fontFamily="var(--font-mono)">
            CHEMICAL SCAFFOLD SPACE (ECFP6 FINGERPRINTS)
          </text>

          <line
            x1={nodePositions.quin1.x}
            y1={nodePositions.quin1.y}
            x2={nodePositions.quin2.x}
            y2={nodePositions.quin2.y}
            stroke={edgeRed}
            strokeWidth="3.5"
            strokeDasharray="5,3"
          />
          <rect x="270" y="163" width="70" height="18" fill="var(--bg-card)" rx="3" stroke={edgeRed} strokeWidth="1" />
          <text x="305" y="175" fill={edgeRed} fontSize="9" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
            Tc = 0.89
          </text>

          {candidates.filter(c => c.id !== activeMolId).map(other => {
            const p1 = nodePositions[activeMolId];
            const p2 = nodePositions[other.id];
            const tcKey = `${activeMolId}-${other.id}`;
            const tc = pairwiseTc[tcKey] || 0.15;
            const isRedundant = tc >= 0.80;
            if (isRedundant) return null;

            return (
              <line
                key={`link-${activeMolId}-${other.id}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke={edgeGreen}
                strokeWidth="1.2"
                strokeDasharray="3,3"
                opacity="0.65"
              />
            );
          })}

          {candidates.map(c => {
            const pos = nodePositions[c.id];
            const isSelected = c.id === activeMolId;

            return (
              <g
                key={c.id}
                style={{ cursor: "pointer" }}
                onClick={() => setActiveMolId(c.id)}
                onMouseEnter={() => setActiveMolId(c.id)}
              >
                {isSelected && (
                  <circle cx={pos.x} cy={pos.y} r="22" fill="none" stroke={c.color} strokeWidth="2.5" strokeDasharray="3,2" opacity="0.8" />
                )}
                <circle cx={pos.x} cy={pos.y} r="16" fill={c.color} stroke="#ffffff" strokeWidth="2" />
                <text x={pos.x} y={pos.y + 4} fill="#ffffff" fontSize="10" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                  {c.rank}
                </text>
                <text x={pos.x} y={pos.y + 32} fill="var(--text-main)" fontSize="10" fontWeight={isSelected ? "700" : "500"} textAnchor="middle">
                  {c.name}
                </text>
              </g>
            );
          })}

          {/* Reward Ranking Ladder on Right */}
          <rect x="445" y="15" width="220" height="220" fill="var(--table-header)" stroke="var(--border-subtle)" strokeWidth="1" rx="6" />
          <text x="460" y="36" fill="var(--text-muted)" fontSize="10" fontWeight="700" fontFamily="var(--font-mono)">
            TANIMOTO TIE-BREAK LADDER
          </text>

          {candidates.map((c, idx) => {
            const rowY = 48 + idx * 36;
            const isSelected = c.id === activeMolId;

            return (
              <g
                key={`ladder-${c.id}`}
                style={{ cursor: "pointer" }}
                onClick={() => setActiveMolId(c.id)}
                onMouseEnter={() => setActiveMolId(c.id)}
              >
                <rect
                  x="455"
                  y={rowY}
                  width="200"
                  height="28"
                  rx="4"
                  fill={isSelected ? "rgba(2, 132, 199, 0.15)" : "transparent"}
                  stroke={isSelected ? "var(--accent-cyan)" : "transparent"}
                  strokeWidth="1"
                />
                <circle cx="470" cy={rowY + 14} r="6" fill={c.color} />
                <text x="484" y={rowY + 18} fill="var(--text-main)" fontSize="10.5" fontWeight={isSelected ? "700" : "500"}>
                  {c.name}
                </text>
                <text x="645" y={rowY + 18} fill={c.color} fontSize="10" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="end">
                  R = {c.reward.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="micro-viz-legend">
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#10b981" }}></span>
          High Diversity Novelty (R = 1.00)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#3b82f6" }}></span>
          Distinct Chemotype (R = 0.60)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#ef4444" }}></span>
          Redundant Cluster Demoted (R = 0.20)
        </span>
        <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--text-muted)" }}>
          Click any scaffold or ladder rank to inspect pairwise metrics
        </span>
      </div>

      <div className="micro-viz-readout">
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Scaffold:</span>
          <span className="micro-viz-readout-val" style={{ color: "var(--text-main)" }}>{active.name} ({active.core})</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Status:</span>
          <span className="micro-viz-readout-val" style={{ color: active.color }}>{active.status}</span>
        </div>
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Pairwise Distance (1 - Tc):</span>
          <span className="micro-viz-readout-val">{active.dist.toFixed(2)}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Nearest Tc:</span>
          <span className="micro-viz-readout-val">{active.sim.toFixed(2)}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Scalar Reward:</span>
          <span className="micro-viz-readout-val">R = {active.reward.toFixed(3)} (Rank {active.rank} of 5)</span>
        </div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
          {active.desc}
        </div>
      </div>
    </div>
  );
}

function ModifierCurvesViz({ theme, symbolKey }) {
  const getInitialTab = () => {
    if (symbolKey && symbolKey.includes("ClippedScore") && !symbolKey.includes("Smooth")) return "clipped";
    if (symbolKey && (symbolKey.includes("MinMaxGaussian") || symbolKey.includes("Gaussian"))) return "gaussian";
    if (symbolKey && (symbolKey.includes("ThresholdedLinear") || symbolKey.includes("Linear"))) return "thresholded";
    return "smooth";
  };

  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [cursorX, setCursorX] = useState(6.0);

  useEffect(() => {
    setActiveTab(getInitialTab());
  }, [symbolKey]);

  const mapX = (x) => 70 + (x / 10.0) * 540;
  const mapY = (y) => 200 - y * 160;

  const evalCurve = (x, type) => {
    if (type === "smooth") {
      const k = 1.0;
      const x0 = 6.0;
      const d = 1.0 / (1.0 + Math.exp(-k * (x - x0)));
      const grad = k * d * (1.0 - d);
      return { d, grad, formula: "d(x) = 1 / (1 + exp(-(x - 6.0)))", status: "Continuous Non-Zero Gradient" };
    } else if (type === "clipped") {
      let d = (x - 4.0) / 4.0;
      d = Math.max(0.0, Math.min(1.0, d));
      const grad = (x >= 4.0 && x <= 8.0) ? 0.25 : 0.0;
      const status = (x < 4.0 || x > 8.0) ? "Vanished Gradient (df/dx = 0)" : "Active Linear Ramp (df/dx = 0.25)";
      return { d, grad, formula: "d(x) = clip((x - 4.0) / 4.0, 0, 1)", status };
    } else if (type === "gaussian") {
      let d = 1.0;
      let grad = 0.0;
      if (x < 3.0) {
        d = Math.exp(-0.5 * Math.pow((x - 3.0) / 1.2, 2));
        grad = -((x - 3.0) / 1.44) * d;
      } else if (x > 6.0) {
        d = Math.exp(-0.5 * Math.pow((x - 6.0) / 1.2, 2));
        grad = -((x - 6.0) / 1.44) * d;
      }
      const status = (x >= 3.0 && x <= 6.0) ? "Optimal Plateau (d = 1.0)" : "Gaussian Exponential Roll-off";
      return { d, grad, formula: "d(x) = exp(-0.5 * ((x - mu) / 1.2)^2)", status };
    } else {
      const d = Math.min(x, 6.0) / 6.0;
      const grad = x <= 6.0 ? (1.0 / 6.0) : 0.0;
      const status = x > 6.0 ? "Saturated Ceiling (df/dx = 0)" : "Linear Scaling (df/dx = 0.167)";
      return { d, grad, formula: "d(x) = min(x, 6.0) / 6.0", status };
    }
  };

  const generatePath = (type) => {
    const steps = 100;
    let dStr = "";
    for (let i = 0; i <= steps; i++) {
      const xVal = (i / steps) * 10.0;
      const yVal = evalCurve(xVal, type).d;
      const px = mapX(xVal);
      const py = mapY(yVal);
      dStr += `${i === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)} `;
    }
    return dStr;
  };

  const activeEval = evalCurve(cursorX, activeTab);
  const isDark = theme === "dark";
  const axisColor = isDark ? "#475569" : "#cbd5e1";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";
  const textColor = isDark ? "#94a3b8" : "#64748b";

  const handleSvgMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const norm = (clientX / rect.width) * 680;
    const xVal = ((norm - 70) / 540) * 10.0;
    const clamped = Math.max(0.0, Math.min(10.0, xVal));
    setCursorX(clamped);
  };

  return (
    <div>
      <div className="micro-viz-tabs">
        <button
          className={`micro-viz-tab-btn ${activeTab === "smooth" ? "active" : ""}`}
          onClick={() => setActiveTab("smooth")}
        >
          SmoothClippedScore (Sigmoid)
        </button>
        <button
          className={`micro-viz-tab-btn ${activeTab === "clipped" ? "active" : ""}`}
          onClick={() => setActiveTab("clipped")}
        >
          ClippedScore (Piecewise)
        </button>
        <button
          className={`micro-viz-tab-btn ${activeTab === "gaussian" ? "active" : ""}`}
          onClick={() => setActiveTab("gaussian")}
        >
          MinMaxGaussian (Plateau)
        </button>
        <button
          className={`micro-viz-tab-btn ${activeTab === "thresholded" ? "active" : ""}`}
          onClick={() => setActiveTab("thresholded")}
        >
          ThresholdedLinear (Ramp)
        </button>
      </div>

      <div className="micro-viz-svg-wrap">
        <svg
          className="micro-viz-svg"
          viewBox="0 0 680 240"
          xmlns="http://www.w3.org/2000/svg"
          onMouseMove={handleSvgMouseMove}
          style={{ cursor: "crosshair" }}
        >
          {activeTab === "clipped" && (
            <g>
              <rect x={mapX(0)} y={mapY(1.0)} width={mapX(4.0) - mapX(0)} height={160} fill="rgba(239, 68, 68, 0.08)" />
              <rect x={mapX(8.0)} y={mapY(1.0)} width={mapX(10.0) - mapX(8.0)} height={160} fill="rgba(239, 68, 68, 0.08)" />
              <rect x={mapX(4.0)} y={mapY(1.0)} width={mapX(8.0) - mapX(4.0)} height={160} fill="rgba(16, 185, 129, 0.06)" />
              <text x={mapX(2.0)} y={120} fill="#ef4444" fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                VANISHED GRADIENT (0.0)
              </text>
              <text x={mapX(6.0)} y={120} fill="var(--accent-emerald)" fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                ACTIVE RAMP (df/dx = 0.25)
              </text>
              <text x={mapX(9.0)} y={120} fill="#ef4444" fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                SATURATED (0.0)
              </text>
              <line x1={mapX(4.0)} y1={40} x2={mapX(4.0)} y2={200} stroke="#ef4444" strokeWidth="1.5" strokeDasharray="3,3" />
              <text x={mapX(4.0)} y={32} fill="#ef4444" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                lower_x = 4.0 (Kink)
              </text>
              <line x1={mapX(8.0)} y1={40} x2={mapX(8.0)} y2={200} stroke="#ef4444" strokeWidth="1.5" strokeDasharray="3,3" />
              <text x={mapX(8.0)} y={32} fill="#ef4444" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                upper_x = 8.0 (Ceiling)
              </text>
            </g>
          )}

          {activeTab === "smooth" && (
            <g>
              <line x1={mapX(4.0)} y1={40} x2={mapX(4.0)} y2={200} stroke="#3b82f6" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
              <text x={mapX(4.0)} y={32} fill="#3b82f6" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                lower_x = 4.0
              </text>
              <line x1={mapX(6.0)} y1={40} x2={mapX(6.0)} y2={200} stroke="#0284c7" strokeWidth="1.5" strokeDasharray="4,2" />
              <rect x={mapX(6.0) - 62} y={18} width={124} height={16} rx="3" fill="var(--bg-card)" stroke="#0284c7" strokeWidth="1" />
              <text x={mapX(6.0)} y={30} fill="#0284c7" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                x0 = 6.0 (Midpoint, d=0.5)
              </text>
              <line x1={mapX(8.0)} y1={40} x2={mapX(8.0)} y2={200} stroke="#3b82f6" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
              <text x={mapX(8.0)} y={32} fill="#3b82f6" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                upper_x = 8.0
              </text>
            </g>
          )}

          {activeTab === "gaussian" && (
            <g>
              <rect x={mapX(3.0)} y={mapY(1.0)} width={mapX(6.0) - mapX(3.0)} height={160} fill="rgba(16, 185, 129, 0.08)" />
              <text x={mapX(4.5)} y={120} fill="var(--accent-emerald)" fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                OPTIMAL PLATEAU (d = 1.0)
              </text>
              <line x1={mapX(3.0)} y1={40} x2={mapX(3.0)} y2={200} stroke="var(--accent-emerald)" strokeWidth="1.5" strokeDasharray="3,3" />
              <text x={mapX(3.0)} y={32} fill="var(--accent-emerald)" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                lower_x = 3.0
              </text>
              <line x1={mapX(6.0)} y1={40} x2={mapX(6.0)} y2={200} stroke="var(--accent-emerald)" strokeWidth="1.5" strokeDasharray="3,3" />
              <text x={mapX(6.0)} y={32} fill="var(--accent-emerald)" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                upper_x = 6.0
              </text>
            </g>
          )}

          {activeTab === "thresholded" && (
            <g>
              <rect x={mapX(6.0)} y={mapY(1.0)} width={mapX(10.0) - mapX(6.0)} height={160} fill="rgba(239, 68, 68, 0.06)" />
              <text x={mapX(8.0)} y={120} fill="#ef4444" fontSize="9.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                SATURATED (df/dx = 0)
              </text>
              <line x1={mapX(6.0)} y1={40} x2={mapX(6.0)} y2={200} stroke="#0284c7" strokeWidth="1.5" strokeDasharray="3,3" />
              <text x={mapX(6.0)} y={32} fill="#0284c7" fontSize="8.5" fontWeight="700" fontFamily="var(--font-mono)" textAnchor="middle">
                Threshold = 6.0
              </text>
            </g>
          )}

          {/* Grid Lines */}
          {[0, 2, 4, 6, 8, 10].map(v => (
            <g key={`x-grid-${v}`}>
              <line x1={mapX(v)} y1={40} x2={mapX(v)} y2={200} stroke={gridColor} strokeWidth="1" strokeDasharray="3,3" />
              <text x={mapX(v)} y={215} fill={textColor} fontSize="10" fontFamily="var(--font-mono)" textAnchor="middle">{v}</text>
            </g>
          ))}
          {[0.0, 0.25, 0.5, 0.75, 1.0].map(v => (
            <g key={`y-grid-${v}`}>
              <line x1={70} y1={mapY(v)} x2={610} y2={mapY(v)} stroke={gridColor} strokeWidth="1" strokeDasharray="3,3" />
              <text x={62} y={mapY(v) + 3} fill={textColor} fontSize="10" fontFamily="var(--font-mono)" textAnchor="end">{v.toFixed(2)}</text>
            </g>
          ))}

          {/* Axes */}
          <line x1={70} y1={200} x2={615} y2={200} stroke={axisColor} strokeWidth="1.5" />
          <line x1={70} y1={200} x2={70} y2={35} stroke={axisColor} strokeWidth="1.5" />
          <text x={340} y={232} fill="var(--text-main)" fontSize="11" fontWeight="600" textAnchor="middle">Raw Physical Property x (e.g. pIC50, MW / 100, LogP + 2)</text>
          <text x={18} y={120} fill="var(--text-main)" fontSize="11" fontWeight="600" textAnchor="middle" transform="rotate(-90, 18, 120)">Desirability d(x)</text>

          {/* Comparison Ghost Curves */}
          {activeTab === "smooth" && (
            <path d={generatePath("clipped")} fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="4,4" opacity="0.6" />
          )}
          {activeTab === "clipped" && (
            <path d={generatePath("smooth")} fill="none" stroke="#0284c7" strokeWidth="1.5" strokeDasharray="4,4" opacity="0.45" />
          )}

          {/* Main Active Curve */}
          <path d={generatePath(activeTab)} fill="none" stroke="#0284c7" strokeWidth="3" />

          {/* Interactive Crosshair & Cursor Indicator */}
          <line
            x1={mapX(cursorX)}
            y1={40}
            x2={mapX(cursorX)}
            y2={200}
            stroke="#0284c7"
            strokeWidth="1.5"
            strokeDasharray="2,2"
          />
          <line
            x1={70}
            y1={mapY(activeEval.d)}
            x2={610}
            y2={mapY(activeEval.d)}
            stroke="#0284c7"
            strokeWidth="1"
            strokeDasharray="2,2"
            opacity="0.5"
          />
          <circle
            cx={mapX(cursorX)}
            cy={mapY(activeEval.d)}
            r="6"
            fill="#0284c7"
            stroke="#ffffff"
            strokeWidth="2"
          />
          <rect
            x={Math.min(mapX(cursorX) + 8, 510)}
            y={Math.max(mapY(activeEval.d) - 24, 45)}
            width={92}
            height="18"
            rx="3"
            fill="var(--bg-card)"
            stroke="#0284c7"
            strokeWidth="1"
          />
          <text
            x={Math.min(mapX(cursorX) + 12, 514)}
            y={Math.max(mapY(activeEval.d) - 12, 57)}
            fill="var(--text-main)"
            fontSize="8.5"
            fontWeight="700"
            fontFamily="var(--font-mono)"
          >
            ({cursorX.toFixed(2)}, {activeEval.d.toFixed(2)})
          </text>
        </svg>
      </div>

      <div className="micro-viz-legend">
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#0284c7" }}></span>
          Active Modifier Transfer Curve
        </span>
        {activeTab === "smooth" && (
          <span className="micro-viz-legend-item">
            <span className="micro-viz-legend-dot" style={{ background: "var(--text-muted)" }}></span>
            Piecewise ClippedScore (Vanished Contrast)
          </span>
        )}
        {activeTab === "clipped" && (
          <span className="micro-viz-legend-item">
            <span className="micro-viz-legend-dot" style={{ background: "#0284c7", opacity: 0.5 }}></span>
            SmoothClippedScore Sigmoid (Smooth Contrast)
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--text-muted)" }}>
          Move mouse over graph to probe property coordinates and policy gradient
        </span>
      </div>

      <div className="micro-viz-readout">
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Input Property x:</span>
          <span className="micro-viz-readout-val">{cursorX.toFixed(2)}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Desirability d(x):</span>
          <span className="micro-viz-readout-val">{activeEval.d.toFixed(4)}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>First Derivative:</span>
          <span className="micro-viz-readout-val" style={{ color: activeEval.grad === 0 ? "#ef4444" : "var(--accent-emerald)" }}>
            df/dx = {activeEval.grad.toFixed(4)}
          </span>
        </div>
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Formula:</span>
          <span className="micro-viz-readout-val" style={{ fontFamily: "var(--font-mono)", color: "var(--text-main)" }}>{activeEval.formula}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>RL Signal:</span>
          <span className="micro-viz-readout-val" style={{ color: activeEval.grad === 0 ? "#ef4444" : "var(--accent-emerald)" }}>{activeEval.status}</span>
        </div>
      </div>
    </div>
  );
}

function RLPolicyCycleViz({ theme, symbolKey }) {
  const [activeStepIdx, setActiveStepIdx] = useState(3);
  const isDark = theme === "dark";

  const steps = [
    {
      id: "agent",
      step: "Step 01",
      badge: "GENERATOR",
      color: "#0284c7",
      name: "Agent Policy pi_theta",
      sub: "SequenceRNN / Transformer",
      shape: "Tensor: [B, L] -> SMILES",
      math: "a_t ~ pi_theta(a_t | s_t)",
      note: "Active generative model rollout; samples batches of chemical token sequences under current policy parameters theta."
    },
    {
      id: "prior",
      step: "Step 02",
      badge: "REGULARIZER",
      color: "#06b6d4",
      name: "Frozen Prior pi_0",
      sub: "Pre-trained Language Model",
      shape: "eps*pi_0 + (1-eps)*pi_theta",
      math: "x_t ~ (1 - eps)*pi_theta + eps*pi_0",
      note: "Maintains chemical syntax and validity learned during pre-training (epsilon=0.1); penalizes mode collapse away from drug-like space."
    },
    {
      id: "smiles_checker",
      step: "Step 03",
      badge: "SANITY GATE",
      color: "#d97706",
      name: "SmilesChecker Gate",
      sub: "Valence & Substructure Filter",
      shape: "Filter: [B] valid binary",
      math: "Valid = 1 if RDKit(smi) else 0",
      note: "Syntactically invalid SMILES, valence violations, or disconnected structures are immediately filtered (score = 0.0) before scoring."
    },
    {
      id: "env",
      step: "Step 04",
      badge: "SCORING ENV",
      color: "#10b981",
      name: "DrugExEnvironment",
      sub: "Scorers + Modifiers",
      shape: "Scores: [B, M] in [0, 1]",
      math: "s_{i,m} = f_m(metric_{i,m}); Desired = prod(s_m >= th_m)",
      note: "Evaluates parallel objectives (ROCS 3D shape, QSAR potency, SA synthesis, MW) transformed into [0, 1] desirability scores."
    },
    {
      id: "reward",
      step: "Step 05",
      badge: "NSGA-II REWARD",
      color: "#8b5cf6",
      name: "RewardScheme (MORL)",
      sub: "ParetoCrowdingDistance",
      shape: "Rewards: [B, 1] in [0, 1]",
      math: "R = rank(F_0, F_1, ..., crowding_d) / B",
      note: "Decomposes candidate population into non-dominated Pareto tiers and computes crowding distance or Tanimoto diversity to break ties."
    },
    {
      id: "reinforce",
      step: "Step 06",
      badge: "POLICY GRADIENT",
      color: "#ec4899",
      name: "REINFORCE Update",
      sub: "Policy Gradient Ascent",
      shape: "Loss: L(theta) -> grad theta",
      math: "grad L = - E[(R - beta) * sum_t grad log pi_theta(a_t)]",
      note: "Backpropagates reward-weighted log-likelihood loss with moving baseline subtraction beta to update agent generative parameters."
    }
  ];

  const active = steps[activeStepIdx] || steps[3];

  const blockCoords = [
    { x: 30,  y: 24, w: 180, h: 76 },
    { x: 250, y: 24, w: 180, h: 76 },
    { x: 470, y: 24, w: 180, h: 76 },
    { x: 470, y: 136, w: 180, h: 76 },
    { x: 250, y: 136, w: 180, h: 76 },
    { x: 30,  y: 136, w: 180, h: 76 }
  ];

  return (
    <div>
      <div className="micro-viz-svg-wrap">
        <svg className="micro-viz-svg" viewBox="0 0 680 240" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#0284c7" />
            </marker>
            <marker id="arrow-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#06b6d4" />
            </marker>
            <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#d97706" />
            </marker>
            <marker id="arrow-emerald" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#10b981" />
            </marker>
            <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#8b5cf6" />
            </marker>
            <marker id="arrow-pink" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#ec4899" />
            </marker>
          </defs>

          {/* Connecting Flow Arrows */}
          <line x1={210} y1={62} x2={244} y2={62} stroke="#0284c7" strokeWidth="2" markerEnd="url(#arrow-blue)" />
          <line x1={430} y1={62} x2={464} y2={62} stroke="#06b6d4" strokeWidth="2" markerEnd="url(#arrow-cyan)" />
          <path d="M 560 100 L 560 130" fill="none" stroke="#d97706" strokeWidth="2" markerEnd="url(#arrow-amber)" />
          <line x1={470} y1={174} x2={436} y2={174} stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow-emerald)" />
          <line x1={250} y1={174} x2={216} y2={174} stroke="#8b5cf6" strokeWidth="2" markerEnd="url(#arrow-purple)" />
          <path d="M 120 136 L 120 106" fill="none" stroke="#ec4899" strokeWidth="2" markerEnd="url(#arrow-pink)" />

          {/* Step Blocks */}
          {steps.map((s, idx) => {
            const c = blockCoords[idx];
            const isSelected = idx === activeStepIdx;
            const cardBg = isSelected
              ? (isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(2, 132, 199, 0.08)")
              : "var(--bg-card)";

            return (
              <g
                key={s.id}
                style={{ cursor: "pointer" }}
                onClick={() => setActiveStepIdx(idx)}
                onMouseEnter={() => setActiveStepIdx(idx)}
              >
                <rect
                  x={c.x}
                  y={c.y}
                  width={c.w}
                  height={c.h}
                  rx="6"
                  fill={cardBg}
                  stroke={isSelected ? s.color : "var(--border-main)"}
                  strokeWidth={isSelected ? "2" : "1"}
                />
                <rect
                  x={c.x + 8}
                  y={c.y + 8}
                  width={48}
                  height={16}
                  rx="3"
                  fill={isSelected ? s.color : (isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)")}
                />
                <text
                  x={c.x + 32}
                  y={c.y + 19}
                  fill={isSelected ? "#ffffff" : s.color}
                  fontSize="9"
                  fontWeight="700"
                  fontFamily="var(--font-mono)"
                  textAnchor="middle"
                >
                  {s.step}
                </text>
                <text
                  x={c.x + 62}
                  y={c.y + 19}
                  fill="var(--text-muted)"
                  fontSize="8.5"
                  fontWeight="700"
                  fontFamily="var(--font-mono)"
                >
                  {s.badge}
                </text>
                <text x={c.x + 8} y={c.y + 38} fill="var(--text-main)" fontSize="10.5" fontWeight="700">
                  {s.name}
                </text>
                <text x={c.x + 8} y={c.y + 52} fill="var(--text-muted)" fontSize="9" fontFamily="var(--font-sans)">
                  {s.sub}
                </text>
                <text x={c.x + 8} y={c.y + 67} fill={s.color} fontSize="8.5" fontFamily="var(--font-mono)" fontWeight="600">
                  {s.shape}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="micro-viz-legend">
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#0284c7" }}></span>
          Rollout &amp; Prior (01 - 02)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#d97706" }}></span>
          SmilesChecker (03)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#10b981" }}></span>
          Scoring Env (04)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#8b5cf6" }}></span>
          Reward &amp; Pareto (05)
        </span>
        <span className="micro-viz-legend-item">
          <span className="micro-viz-legend-dot" style={{ background: "#ec4899" }}></span>
          REINFORCE Update (06)
        </span>
        <span style={{ marginLeft: "auto", fontSize: "11px", color: "var(--text-muted)" }}>
          Hover any stage to inspect tensor shapes and mathematical operations
        </span>
      </div>

      <div className="micro-viz-readout">
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">{active.step}:</span>
          <span className="micro-viz-readout-val" style={{ color: "var(--text-main)" }}>{active.name} ({active.sub})</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Phase:</span>
          <span className="micro-viz-readout-val" style={{ color: active.color, fontWeight: "700" }}>{active.badge}</span>
          <span className="micro-viz-readout-label" style={{ marginLeft: "12px" }}>Tensors / Arrays:</span>
          <span className="micro-viz-readout-val">{active.shape}</span>
        </div>
        <div className="micro-viz-readout-row">
          <span className="micro-viz-readout-label">Mathematical Operation:</span>
          <span className="micro-viz-readout-val" style={{ fontFamily: "var(--font-mono)" }}>{active.math}</span>
        </div>
        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
          {active.note}
        </div>
      </div>
    </div>
  );
}

// Render React App
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
