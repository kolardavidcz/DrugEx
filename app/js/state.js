/**
 * DrugEx Hub — Reactive State, Authentication & Deterministic Assessment Shuffler
 */

const STORAGE_KEY_USER = "drugex-hub-user-v1";
const STORAGE_KEY_PROGRESS = "drugex-hub-progress-v1";
const STORAGE_KEY_THEME = "drugex-hub-theme";

// Generic chemistry & python distractors for padding to 4 options
const GENERIC_CHEM_DISTRACTORS = [
  "Lipinski Rule of 5 Violation",
  "Tanimoto score of 0.0 due to lack of shared features",
  "Conformer generation timeout",
  "None of the above"
];

function hashFnv32(str) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export const state = {
  theme: localStorage.getItem(STORAGE_KEY_THEME) || "dark",
  user: JSON.parse(localStorage.getItem(STORAGE_KEY_USER) || "null") || {
    username: "student",
    email: "student@vscht.cz",
    role: "Student / Researcher"
  },
  progress: JSON.parse(localStorage.getItem(STORAGE_KEY_PROGRESS) || "{}"),
  curriculum: null,
  thesisGuide: null,
  activeRoute: null,
  searchQuery: "",
  selectedTag: null,
  listeners: new Set(),

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  },

  notify() {
    for (const listener of this.listeners) {
      listener(this);
    }
  },

  setTheme(theme) {
    this.theme = theme;
    localStorage.setItem(STORAGE_KEY_THEME, theme);
    document.documentElement.setAttribute("data-theme", theme);
    this.notify();
  },

  toggleTheme() {
    this.setTheme(this.theme === "dark" ? "light" : "dark");
  },

  markItemStatus(itemId, status) {
    if (!this.progress.items) this.progress.items = {};
    this.progress.items[itemId] = status; // 'studied' | 'known' | 'to-study'
    localStorage.setItem(STORAGE_KEY_PROGRESS, JSON.stringify(this.progress));
    this.notify();
  },

  getItemStatus(itemId) {
    return (this.progress.items && this.progress.items[itemId]) || "to-study";
  },

  saveQuizScore(quizId, score, total) {
    if (!this.progress.quizzes) this.progress.quizzes = {};
    this.progress.quizzes[quizId] = { score, total, date: new Date().toISOString() };
    localStorage.setItem(STORAGE_KEY_PROGRESS, JSON.stringify(this.progress));
    this.notify();
  },

  getQuizScore(quizId) {
    return this.progress.quizzes && this.progress.quizzes[quizId];
  },

  async setUser(email, password) {
    const salt = Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map(b => b.toString(16).padStart(2, "0")).join("");
    
    const enc = new TextEncoder();
    const data = enc.encode(password + salt);
    const hashBuf = await crypto.subtle.digest("SHA-256", data);
    const hashHex = Array.from(new Uint8Array(hashBuf))
      .map(b => b.toString(16).padStart(2, "0")).join("");

    const username = email.split("@")[0] || "student";
    this.user = { username, email, salt, hash: hashHex };
    localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(this.user));
    this.notify();
  }
};

/**
 * Ensures deterministic ~25% A/B/C/D option equilibrium using FNV-1a hash
 */
export function ensureShuffledOptions(q, deckKey, idx) {
  let cleanOpts = [...(q.options || [])];

  // Pad to 4 options if fewer exist
  let padIdx = 0;
  while (cleanOpts.length < 4) {
    cleanOpts.push(GENERIC_CHEM_DISTRACTORS[padIdx % GENERIC_CHEM_DISTRACTORS.length]);
    padIdx++;
  }

  const correctText = cleanOpts[q.correct] || cleanOpts[0];
  const seedStr = `${deckKey}:${q.id || idx}:${q.question || ""}:${cleanOpts.length}`;
  const hash = hashFnv32(seedStr);
  const targetIdx = hash % cleanOpts.length;

  // Move correct option to targetIdx
  const currIdx = cleanOpts.indexOf(correctText);
  if (currIdx !== -1 && currIdx !== targetIdx) {
    const temp = cleanOpts[targetIdx];
    cleanOpts[targetIdx] = cleanOpts[currIdx];
    cleanOpts[currIdx] = temp;
  }

  return {
    options: cleanOpts,
    correct: targetIdx
  };
}
