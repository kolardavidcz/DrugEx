/**
 * Test suite for Python syntax highlighter across all 18 lectures
 */

import { highlightPython } from "../app/js/format.js";
import { LECTURE_DATA } from "../app/js/lectures_content.js";

let totalSnippets = 0;
let errors = 0;

Object.entries(LECTURE_DATA).forEach(([lecId, lec]) => {
  lec.slides.forEach((slide, sIdx) => {
    if (!slide.code) return;
    totalSnippets++;

    const highlighted = highlightPython(slide.code);

    // Check for malformed tags like <span <span or unclosed/leaked attributes
    if (highlighted.includes("<span <span") || highlighted.includes("&lt;span") || /class="syn-[^"]*">[^<]*class="syn-/.test(highlighted)) {
      console.error(`❌ Syntax corruption detected in ${lecId} slide ${sIdx + 1} (${slide.title})`);
      errors++;
    }

    // Check for raw html entity leaks or mismatched spans
    const openSpans = (highlighted.match(/<span/g) || []).length;
    const closeSpans = (highlighted.match(/<\/span>/g) || []).length;
    if (openSpans !== closeSpans) {
      console.error(`❌ Mismatched span tags in ${lecId} slide ${sIdx + 1}: ${openSpans} opens vs ${closeSpans} closes`);
      errors++;
    }
  });
});

console.log("=================================================");
console.log(` 🧪 Tested ${totalSnippets} Python code showcases`);
if (errors === 0) {
  console.log(" ✨ 100% PASS: All code showcases highlight cleanly with zero HTML tag corruption!");
  process.exit(0);
} else {
  console.error(` ❌ FAIL: ${errors} syntax errors found.`);
  process.exit(1);
}
