/**
 * DrugEx Hub — Interactive Assessment & Quiz Engine
 */

import { el, clear } from "./ui.js";
import { state, ensureShuffledOptions } from "./state.js";

export async function renderQuiz(container, quizId) {
  if (!container) return;
  clear(container);

  // Map quizId (e.g. quiz_m1 -> m1)
  const modKey = quizId.replace("quiz_", "").replace("quiz-", "");
  
  let quizData = null;
  try {
    const res = await fetch(`/data/quizzes/${modKey}.json`);
    if (res.ok) {
      quizData = await res.json();
    }
  } catch (err) {
    console.warn("Fetch quiz error:", err);
  }

  if (!quizData || !quizData.questions) {
    container.innerHTML = `<div class="alert-box alert-warning">Test pro modul "${modKey}" se nepodařilo načíst.</div>`;
    return;
  }

  let userAnswers = {};
  let score = 0;

  const view = el("div", { className: "quiz-container" }, [
    el("header", { className: "quiz-header" }, [
      el("div", { className: "quiz-title-row" }, [
        el("h1", { className: "quiz-title" }, quizData.title),
        el("span", { id: "scoreBadge", className: "quiz-score-badge" }, `Skóre: 0 / ${quizData.questions.length}`)
      ]),
      el("div", { className: "quiz-progress-bar" }, [
        el("div", { id: "quizProgressFill", className: "quiz-progress-fill" })
      ])
    ])
  ]);

  const letters = ["A", "B", "C", "D"];

  quizData.questions.forEach((rawQ, qIdx) => {
    const { options, correct } = ensureShuffledOptions(rawQ, quizId, qIdx);

    const qCard = el("div", { className: "question-card", id: `qcard_${qIdx}` }, [
      el("div", { className: "question-stem" }, `${qIdx + 1}. ${rawQ.question}`)
    ]);

    const optsContainer = el("div", { className: "options-list" });
    const expBox = el("div", { className: "explanation-box", id: `exp_${qIdx}` }, rawQ.explanation || "");

    options.forEach((optText, optIdx) => {
      const btn = el("button", {
        className: "option-btn",
        id: `btn_${qIdx}_${optIdx}`,
        onClick: () => selectOption(qIdx, optIdx, correct, options.length)
      }, [
        el("span", { className: "option-letter" }, letters[optIdx] || String(optIdx + 1)),
        el("span", { style: { flex: "1" } }, optText)
      ]);
      optsContainer.appendChild(btn);
    });

    qCard.appendChild(optsContainer);
    qCard.appendChild(expBox);
    view.appendChild(qCard);
  });

  container.appendChild(view);

  function selectOption(qIdx, selectedIdx, correctIdx, numOpts) {
    if (userAnswers[qIdx] !== undefined) return; // already answered
    userAnswers[qIdx] = selectedIdx;

    const isCorrect = selectedIdx === correctIdx;
    if (isCorrect) score++;

    // Highlight options
    for (let i = 0; i < numOpts; i++) {
      const b = document.getElementById(`btn_${qIdx}_${i}`);
      if (!b) continue;
      b.disabled = true;
      if (i === correctIdx) {
        b.classList.add("correct");
      } else if (i === selectedIdx && !isCorrect) {
        b.classList.add("incorrect");
      }
    }

    // Show explanation
    const exp = document.getElementById(`exp_${qIdx}`);
    if (exp) exp.classList.add("visible");

    // Update progress & score
    const answeredCount = Object.keys(userAnswers).length;
    const totalCount = quizData.questions.length;
    const pct = Math.round((answeredCount / totalCount) * 100);

    const badge = document.getElementById("scoreBadge");
    if (badge) badge.textContent = `Skóre: ${score} / ${totalCount}`;

    const fill = document.getElementById("quizProgressFill");
    if (fill) fill.style.width = `${pct}%`;

    // Persist score
    state.saveQuizScore(quizId, score, totalCount);
  }
}
