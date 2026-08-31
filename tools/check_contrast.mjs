/**
 * DrugEx Hub — WCAG 2.1 Color Contrast Audit Tool
 */

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  if (clean.length === 3) {
    return [
      parseInt(clean[0] + clean[0], 16),
      parseInt(clean[1] + clean[1], 16),
      parseInt(clean[2] + clean[2], 16)
    ];
  }
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16)
  ];
}

function getLuminance([r, g, b]) {
  const [rs, gs, bs] = [r, g, b].map(c => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

function getContrast(hex1, hex2) {
  const l1 = getLuminance(hexToRgb(hex1));
  const l2 = getLuminance(hexToRgb(hex2));
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

const TESTS = [
  // Dark Theme Tests
  { name: "Dark Surface Text", fg: "#cccccc", bg: "#1f1f1f", min: 4.5 },
  { name: "Dark Bright Text", fg: "#f0f0f0", bg: "#1f1f1f", min: 7.0 },
  { name: "Dark Accent (Cyan)", fg: "#38bdf8", bg: "#1f1f1f", min: 4.5 },
  { name: "Dark Bio Green Tag", fg: "#4ade80", bg: "#181818", min: 4.5 },
  { name: "Dark ROCS Purple Tag", fg: "#c084fc", bg: "#181818", min: 4.5 },
  { name: "Dark Amber Warn Tag", fg: "#fbbf24", bg: "#181818", min: 4.5 },
  { name: "Dark Statusbar Text", fg: "#ffffff", bg: "#007acc", min: 4.5 },

  // Light Theme Tests
  { name: "Light Surface Text", fg: "#1e293b", bg: "#ffffff", min: 4.5 },
  { name: "Light Accent Blue", fg: "#0369a1", bg: "#ffffff", min: 4.5 },
  { name: "Light Bio Green", fg: "#15803d", bg: "#ffffff", min: 4.5 },
  { name: "Light ROCS Purple", fg: "#7c3aed", bg: "#ffffff", min: 4.5 },
  { name: "Light Amber Warn", fg: "#a16207", bg: "#ffffff", min: 4.5 },
  { name: "Light Statusbar Text", fg: "#ffffff", bg: "#0369a1", min: 4.5 }
];

console.log("=================================================");
console.log(" 🧪 DrugEx Hub — WCAG 2.1 Contrast Test Suite ");
console.log("=================================================");

let passed = 0;
let failed = 0;

for (const t of TESTS) {
  const ratio = getContrast(t.fg, t.bg);
  const ok = ratio >= t.min;
  if (ok) {
    passed++;
    console.log(` ✓ [PASS] ${t.name.padEnd(26)}: ${ratio.toFixed(2)}:1 (min ${t.min}:1)`);
  } else {
    failed++;
    console.error(` ✗ [FAIL] ${t.name.padEnd(26)}: ${ratio.toFixed(2)}:1 (required ${t.min}:1)`);
  }
}

console.log("-------------------------------------------------");
console.log(`Total: ${TESTS.length} | Passed: ${passed} | Failed: ${failed}`);
if (failed > 0) {
  process.exit(1);
} else {
  console.log("✨ 100% WCAG 2.1 AA/AAA Contrast Compliance verified!");
}
