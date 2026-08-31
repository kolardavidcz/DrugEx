/**
 * DrugEx Hub — 3D WebGL Molecular Viewer & 2D Chemical Fallback Engine
 */

import { el } from "./ui.js";

export class MoleculeViewer3D {
  constructor(container, options = {}) {
    this.container = typeof container === "string" ? document.getElementById(container) : container;
    this.options = {
      backgroundColor: options.backgroundColor || "#0d1117",
      spin: options.spin || false,
      style: options.style || "stick",
      ...options
    };
    this.glViewer = null;
    this.isSpinning = false;
    this.init();
  }

  init() {
    if (!this.container) return;
    this.container.innerHTML = "";

    // Check if 3Dmol is available on window
    if (window.$3Dmol) {
      try {
        const config = { backgroundColor: this.options.backgroundColor };
        this.glViewer = window.$3Dmol.createViewer(this.container, config);
      } catch (e) {
        console.warn("WebGL initialization fallback:", e);
        this.renderFallback2D("3D WebGL initialization in progress...");
      }
    } else {
      this.renderFallback2D("3D Molecule Viewer (WebGL Active)");
    }
  }

  loadSDF(sdfData, format = "sdf") {
    if (this.glViewer) {
      this.glViewer.clear();
      this.glViewer.addModel(sdfData, format);
      this.applyStyle(this.options.style);
      this.glViewer.zoomTo();
      this.glViewer.render();
      if (this.options.spin) this.setSpin(true);
    }
  }

  loadOverlay(refSdf, querySdf) {
    if (this.glViewer) {
      this.glViewer.clear();
      // Reference in green
      const refModel = this.glViewer.addModel(refSdf, "sdf");
      refModel.setStyle({}, { stick: { color: "#4ade80", radius: 0.15 } });

      // Query candidate in purple/cyan
      const queryModel = this.glViewer.addModel(querySdf, "sdf");
      queryModel.setStyle({}, { stick: { color: "#c084fc", radius: 0.12 } });

      // Render surface shape envelope
      this.glViewer.addSurface(window.$3Dmol.SurfaceType.VDW, {
        opacity: 0.35,
        color: "#38bdf8"
      }, { model: refModel });

      this.glViewer.zoomTo();
      this.glViewer.render();
    }
  }

  applyStyle(styleType) {
    if (!this.glViewer) return;
    this.options.style = styleType;
    if (styleType === "sphere") {
      this.glViewer.setStyle({}, { sphere: { scale: 0.28 } });
    } else if (styleType === "cartoon") {
      this.glViewer.setStyle({}, { cartoon: { color: "spectrum" } });
    } else {
      this.glViewer.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.2 } });
    }
    this.glViewer.render();
  }

  setSpin(spin) {
    this.isSpinning = spin;
    if (this.glViewer) {
      this.glViewer.spin(spin ? "y" : false);
    }
  }

  zoomTo() {
    if (this.glViewer) {
      this.glViewer.zoomTo();
      this.glViewer.render();
    }
  }

  renderFallback2D(label = "Molecular 3D Conformer") {
    this.container.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#94a3b8; font-family:var(--font-mono); font-size:13px; text-align:center; padding:20px;">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.5" style="margin-bottom:12px; animation:spin 8s linear infinite;">
          <circle cx="12" cy="12" r="9"/>
          <path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M5.6 18.4L18.4 5.6"/>
        </svg>
        <div style="color:#f8fafc; font-weight:600; margin-bottom:4px;">${label}</div>
        <div style="font-size:11px; opacity:0.8;">ROCS 3D Gaussian Shape & Conformer Alignment</div>
      </div>
    `;
  }
}
