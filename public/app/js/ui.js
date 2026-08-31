/**
 * DrugEx Hub — Micro DOM & UI Utilities
 */

export function el(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith("on") && typeof value === "function") {
      element.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "className" || key === "class") {
      element.className = value;
    } else if (key === "dataset" && typeof value === "object") {
      for (const [dKey, dVal] of Object.entries(value)) {
        element.dataset[dKey] = dVal;
      }
    } else if (key === "style" && typeof value === "object") {
      Object.assign(element.style, value);
    } else if (key === "innerHTML") {
      element.innerHTML = value;
    } else if (key === "textContent") {
      element.textContent = value;
    } else if (value !== false && value !== null && value !== undefined) {
      element.setAttribute(key, value === true ? "" : value);
    }
  }

  if (!Array.isArray(children)) {
    children = [children];
  }
  for (const child of children) {
    if (typeof child === "string" || typeof child === "number") {
      element.appendChild(document.createTextNode(String(child)));
    } else if (child instanceof Node) {
      element.appendChild(child);
    }
  }
  return element;
}

export function clear(element) {
  if (element) {
    element.innerHTML = "";
  }
}

export function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function showToast(message, type = "info") {
  let container = document.getElementById("toastContainer");
  if (!container) {
    container = el("div", { id: "toastContainer", className: "toast-container" });
    document.body.appendChild(container);
  }

  const toast = el("div", { className: `toast toast-${type}` }, [
    el("span", { className: "toast-icon" }, type === "success" ? "✓" : "ℹ"),
    el("span", { className: "toast-msg" }, message)
  ]);

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(20px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

export async function copyText(text, successMsg = "Zkopírováno do schránky!") {
  try {
    await navigator.clipboard.writeText(text);
    showToast(successMsg, "success");
  } catch (err) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    showToast(successMsg, "success");
  }
}
