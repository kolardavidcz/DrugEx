/**
 * DrugEx Hub — Publication-Grade Vector SVG Chemoinformatics Schematics
 * High-resolution, zero-blur vector diagrams for both interactive UI and publication PDF.
 */

export function createRocsGaussianSvg() {
  return `
<div class="schematic-container" data-schematic="rocs-gaussian">
  <div class="schematic-header">
    <span class="schematic-title">📐 Vektorové Schéma: 3D Gaussovský Tvarový Překryv & Barevný Farmakofor (ROCS)</span>
    <span class="schematic-badge">Analytický objem</span>
  </div>
  <div class="schematic-svg-wrap">
    <svg viewBox="0 0 760 250" class="schematic-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <!-- Gradients for Gaussian contours -->
        <radialGradient id="gaussRef" cx="40%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.5"/>
          <stop offset="60%" stop-color="#0284c7" stop-opacity="0.2"/>
          <stop offset="100%" stop-color="#0284c7" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="gaussGen" cx="60%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#34d399" stop-opacity="0.5"/>
          <stop offset="60%" stop-color="#059669" stop-opacity="0.2"/>
          <stop offset="100%" stop-color="#059669" stop-opacity="0"/>
        </radialGradient>
        <linearGradient id="overlapGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#818cf8" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#c084fc" stop-opacity="0.6"/>
        </linearGradient>
      </defs>

      <!-- Background card -->
      <rect x="2" y="2" width="756" height="246" rx="8" fill="#0b1120" stroke="#1e293b" stroke-width="1.5"/>

      <!-- Reference Gaussian Envelope A -->
      <ellipse cx="250" cy="115" rx="140" ry="75" fill="url(#gaussRef)"/>
      <ellipse cx="250" cy="115" rx="140" ry="75" fill="none" stroke="#38bdf8" stroke-width="2" stroke-dasharray="5 3"/>
      
      <!-- Generated Molecule Envelope B -->
      <ellipse cx="370" cy="115" rx="135" ry="80" fill="url(#gaussGen)"/>
      <ellipse cx="370" cy="115" rx="135" ry="80" fill="none" stroke="#34d399" stroke-width="2"/>

      <!-- Overlap Region I_AB highlight -->
      <path d="M 310,50 A 140,75 0 0,1 390,115 A 140,75 0 0,1 310,180 A 135,80 0 0,1 235,115 A 135,80 0 0,1 310,50 Z" 
            fill="url(#overlapGrad)" stroke="#a855f7" stroke-width="2.5"/>

      <!-- Center-points & Atoms Representation -->
      <!-- Ref points -->
      <circle cx="180" cy="100" r="10" fill="#0284c7" stroke="#e0f2fe" stroke-width="2"/>
      <text x="180" y="104" text-anchor="middle" font-size="9" font-weight="bold" fill="#ffffff">C</text>
      <circle cx="230" cy="140" r="10" fill="#0284c7" stroke="#e0f2fe" stroke-width="2"/>
      <text x="230" y="144" text-anchor="middle" font-size="9" font-weight="bold" fill="#ffffff">C</text>
      
      <!-- Generated points -->
      <circle cx="430" cy="95" r="10" fill="#059669" stroke="#d1fae5" stroke-width="2"/>
      <text x="430" y="99" text-anchor="middle" font-size="9" font-weight="bold" fill="#ffffff">C</text>
      <circle cx="390" cy="150" r="10" fill="#059669" stroke="#d1fae5" stroke-width="2"/>
      <text x="390" y="154" text-anchor="middle" font-size="9" font-weight="bold" fill="#ffffff">N</text>

      <!-- Color Pharmacophore Features Overlap -->
      <!-- H-Bond Donor (Cyan) -->
      <circle cx="300" cy="85" r="13" fill="#0284c7" stroke="#38bdf8" stroke-width="2.5"/>
      <text x="300" y="89" text-anchor="middle" font-size="10" font-weight="900" fill="#e0f2fe">D</text>

      <!-- H-Bond Acceptor (Rose Red) -->
      <circle cx="320" cy="145" r="13" fill="#be123c" stroke="#fb7185" stroke-width="2.5"/>
      <text x="320" y="149" text-anchor="middle" font-size="10" font-weight="900" fill="#ffe4e6">A</text>

      <!-- Hydrophobe (Amber) -->
      <circle cx="260" cy="90" r="12" fill="#b45309" stroke="#fde047" stroke-width="2"/>
      <text x="260" y="94" text-anchor="middle" font-size="10" font-weight="900" fill="#fef3c7">H</text>

      <!-- Aromatic Ring (Purple) -->
      <circle cx="340" cy="105" r="12" fill="#6b21a8" stroke="#d8b4fe" stroke-width="2"/>
      <text x="340" y="109" text-anchor="middle" font-size="10" font-weight="900" fill="#f3e8ff">Ar</text>

      <!-- Overlap Callout Annotation -->
      <line x1="310" y1="115" x2="310" y2="210" stroke="#c084fc" stroke-width="1.5" stroke-dasharray="2 2"/>
      <rect x="235" y="202" width="150" height="26" rx="4" fill="#1e1b4b" stroke="#818cf8" stroke-width="1"/>
      <text x="310" y="219" text-anchor="middle" font-size="11" font-weight="bold" fill="#e0e7ff">Objemový průnik I_AB</text>

      <!-- Math Equation Side Panel -->
      <rect x="525" y="20" width="215" height="205" rx="6" fill="#0f172a" stroke="#334155" stroke-width="1"/>
      <text x="632" y="42" text-anchor="middle" font-size="11" font-weight="bold" fill="#38bdf8">Metrika TanimotoCombo</text>
      
      <!-- Shape formula -->
      <text x="540" y="70" font-size="11" fill="#94a3b8">Tvarová shoda (Shape):</text>
      <text x="632" y="95" text-anchor="middle" font-size="12" font-weight="bold" fill="#34d399">T_shape = I_AB / (I_AA + I_BB - I_AB)</text>
      <text x="632" y="115" text-anchor="middle" font-size="9.5" fill="#64748b">kde I_AA, I_BB jsou vlastní objemy sfér</text>

      <!-- Color formula -->
      <text x="540" y="145" font-size="11" fill="#94a3b8">Barevná shoda (Color):</text>
      <text x="632" y="168" text-anchor="middle" font-size="12" font-weight="bold" fill="#f43f5e">T_color = C_AB / (C_AA + C_BB - C_AB)</text>

      <!-- Combo formula -->
      <line x1="540" y1="182" x2="725" y2="182" stroke="#334155" stroke-width="1"/>
      <text x="632" y="204" text-anchor="middle" font-size="12.5" font-weight="900" fill="#fbbf24">T_combo = T_shape + T_color ∈ [0, 2]</text>

      <!-- Legend for Left Envelope -->
      <text x="140" y="45" font-size="11" font-weight="bold" fill="#38bdf8">-- Referenční ligand (A)</text>
      <text x="400" y="45" font-size="11" font-weight="bold" fill="#34d399">── Generovaný konformer (B)</text>
    </svg>
  </div>
</div>`;
}

export function createPolicyGradientSvg() {
  return `
<div class="schematic-container" data-schematic="policy-gradient">
  <div class="schematic-header">
    <span class="schematic-title">🔁 Vektorové Schéma: Architektura Policy Gradientu & Vzájemné Působení Agent vs. Prior</span>
    <span class="schematic-badge">Tensor Flow</span>
  </div>
  <div class="schematic-svg-wrap">
    <svg viewBox="0 0 760 235" class="schematic-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- Background card -->
      <rect x="2" y="2" width="756" height="231" rx="8" fill="#0b1120" stroke="#1e293b" stroke-width="1.5"/>

      <!-- Marker defs for arrows -->
      <defs>
        <marker id="arrowBlue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#38bdf8" />
        </marker>
        <marker id="arrowGreen" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#34d399" />
        </marker>
        <marker id="arrowAmber" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#fbbf24" />
        </marker>
        <marker id="arrowPurple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#c084fc" />
        </marker>
      </defs>

      <!-- 1. Prior Network pi_0 -->
      <rect x="30" y="30" width="130" height="70" rx="6" fill="#1e293b" stroke="#475569" stroke-width="2"/>
      <text x="95" y="55" text-anchor="middle" font-size="11" font-weight="bold" fill="#e2e8f0">Prior Síť (π₀)</text>
      <text x="95" y="72" text-anchor="middle" font-size="9" fill="#94a3b8">Zmrazený model</text>
      <text x="95" y="86" text-anchor="middle" font-size="8.5" fill="#38bdf8">Papyrus v05.5</text>

      <!-- 2. Agent Network pi_theta -->
      <rect x="30" y="130" width="130" height="75" rx="6" fill="#0c4a6e" stroke="#0284c7" stroke-width="2"/>
      <text x="95" y="155" text-anchor="middle" font-size="11" font-weight="bold" fill="#e0f2fe">Agent Síť (π_θ)</text>
      <text x="95" y="172" text-anchor="middle" font-size="9" fill="#bae6fd">Trénovatelný LSTM</text>
      <text x="95" y="188" text-anchor="middle" font-size="8.5" fill="#38bdf8">θ ← θ + α ∇J(θ)</text>

      <!-- Arrow from Agent to Generator Output -->
      <line x1="160" y1="167" x2="220" y2="167" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowBlue)"/>

      <!-- 3. Sampling & Generated Batch x -->
      <rect x="225" y="135" width="120" height="65" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="1.5"/>
      <text x="285" y="157" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#e0e7ff">Vzorkování (x)</text>
      <text x="285" y="173" text-anchor="middle" font-size="9" fill="#c7d2fe">SMILES dávka</text>
      <text x="285" y="188" text-anchor="middle" font-size="8.5" fill="#818cf8">N = 1000 molekul</text>

      <!-- Arrow to Environment -->
      <line x1="345" y1="167" x2="400" y2="167" stroke="#34d399" stroke-width="2" marker-end="url(#arrowGreen)"/>

      <!-- 4. DrugExEnvironment -->
      <rect x="405" y="115" width="145" height="95" rx="6" fill="#064e3b" stroke="#059669" stroke-width="2"/>
      <text x="477" y="136" text-anchor="middle" font-size="11" font-weight="bold" fill="#ecfdf5">DrugExEnvironment</text>
      <text x="477" y="153" text-anchor="middle" font-size="9" fill="#a7f3d0">• 3D ROCS Scorer</text>
      <text x="477" y="168" text-anchor="middle" font-size="9" fill="#a7f3d0">• SAScore Modifier</text>
      <text x="477" y="183" text-anchor="middle" font-size="9" fill="#a7f3d0">• QSAR / Affinity</text>
      <text x="477" y="198" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#34d399">Paretovo řazení (PCD)</text>

      <!-- Arrow to Reward R(x) -->
      <line x1="550" y1="167" x2="595" y2="167" stroke="#fbbf24" stroke-width="2" marker-end="url(#arrowAmber)"/>

      <!-- 5. Reward Calculation Node -->
      <rect x="600" y="135" width="130" height="65" rx="6" fill="#451a03" stroke="#d97706" stroke-width="1.5"/>
      <text x="665" y="157" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#fef3c7">Odměna R(x)</text>
      <text x="665" y="173" text-anchor="middle" font-size="9" fill="#fde68a">Crowding Distance</text>
      <text x="665" y="188" text-anchor="middle" font-size="8.5" fill="#f59e0b">Skalární váha ∈ [0, 1]</text>

      <!-- KL Divergence Regularization Spring (Prior to Agent) -->
      <path d="M 95,100 C 70,115 120,115 95,130" fill="none" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="4 2"/>
      <rect x="125" y="98" width="115" height="22" rx="3" fill="#881337" stroke="#f43f5e" stroke-width="1"/>
      <text x="182" y="113" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#ffe4e6">β · D_KL(π_θ ∥ π₀)</text>

      <!-- Policy Gradient Backpropagation Path -->
      <path d="M 665,135 L 665,45 L 205,45" fill="none" stroke="#c084fc" stroke-width="2" stroke-dasharray="6 3" marker-end="url(#arrowPurple)"/>
      <rect x="350" y="32" width="220" height="26" rx="4" fill="#2e1065" stroke="#a855f7" stroke-width="1"/>
      <text x="460" y="49" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#f3e8ff">Ztráta: L(θ) = -R(x) · log π_θ(x) + β · D_KL</text>
    </svg>
  </div>
</div>`;
}

export function createParetoFrontSvg() {
  return `
<div class="schematic-container" data-schematic="pareto-front">
  <div class="schematic-header">
    <span class="schematic-title">📊 Vektorové Schéma: Paretova Optimalita & Crowding Distance v prostoru odměn</span>
    <span class="schematic-badge">MORL Architektura</span>
  </div>
  <div class="schematic-svg-wrap">
    <svg viewBox="0 0 760 250" class="schematic-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- Background card -->
      <rect x="2" y="2" width="756" height="246" rx="8" fill="#0b1120" stroke="#1e293b" stroke-width="1.5"/>

      <!-- Axes -->
      <!-- Y-axis -->
      <line x1="80" y1="210" x2="80" y2="30" stroke="#475569" stroke-width="2"/>
      <polygon points="77,32 83,32 80,24" fill="#475569"/>
      <text x="35" y="125" transform="rotate(-90 35,125)" text-anchor="middle" font-size="11" font-weight="bold" fill="#34d399">Syntetická dostupnost (f₂: SAScore)</text>

      <!-- X-axis -->
      <line x1="80" y1="210" x2="490" y2="210" stroke="#475569" stroke-width="2"/>
      <polygon points="488,207 488,213 496,210" fill="#475569"/>
      <text x="285" y="235" text-anchor="middle" font-size="11" font-weight="bold" fill="#38bdf8">3D Tvarový překryv (f₁: ROCS TanimotoCombo)</text>

      <!-- Grid lines -->
      <line x1="180" y1="210" x2="180" y2="35" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="280" y1="210" x2="280" y2="35" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="380" y1="210" x2="380" y2="35" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="80" y1="150" x2="480" y2="150" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>
      <line x1="80" y1="90" x2="480" y2="90" stroke="#1e293b" stroke-width="1" stroke-dasharray="3 3"/>

      <!-- Front 1: Non-dominated Pareto Front (Bold Blue / Cyan) -->
      <path d="M 120,60 Q 240,75 320,120 T 440,195" fill="none" stroke="#38bdf8" stroke-width="2.5"/>

      <!-- Crowding Distance Cuboid between points on Front 1 -->
      <rect x="230" y="85" width="105" height="50" fill="#0284c7" fill-opacity="0.18" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3 2"/>
      <text x="282" y="114" text-anchor="middle" font-size="9" font-weight="bold" fill="#7dd3fc">Crowding box d_i</text>

      <!-- Front 1 Points -->
      <circle cx="120" cy="60" r="6" fill="#38bdf8" stroke="#ffffff" stroke-width="2"/>
      <text x="120" y="50" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#38bdf8">d = ∞</text>

      <circle cx="230" cy="85" r="5.5" fill="#38bdf8" stroke="#ffffff" stroke-width="1.5"/>
      <text x="220" y="80" text-anchor="end" font-size="8.5" fill="#e0f2fe">i-1</text>

      <circle cx="280" cy="105" r="6.5" fill="#f59e0b" stroke="#ffffff" stroke-width="2"/>
      <text x="295" y="103" text-anchor="start" font-size="9" font-weight="bold" fill="#fbbf24">Bod i (vysoká diverzita)</text>

      <circle cx="335" cy="135" r="5.5" fill="#38bdf8" stroke="#ffffff" stroke-width="1.5"/>
      <text x="350" y="140" text-anchor="start" font-size="8.5" fill="#e0f2fe">i+1</text>

      <circle cx="440" cy="195" r="6" fill="#38bdf8" stroke="#ffffff" stroke-width="2"/>
      <text x="440" y="185" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#38bdf8">d = ∞</text>

      <!-- Front 2: Rank 2 (Emerald) -->
      <path d="M 150,110 Q 230,135 370,198" fill="none" stroke="#34d399" stroke-width="1.8" stroke-dasharray="4 2"/>
      <circle cx="150" cy="110" r="4.5" fill="#34d399"/>
      <circle cx="230" cy="135" r="4.5" fill="#34d399"/>
      <circle cx="300" cy="165" r="4.5" fill="#34d399"/>
      <circle cx="370" cy="198" r="4.5" fill="#34d399"/>

      <!-- Front 3: Dominated points (Slate) -->
      <circle cx="170" cy="165" r="4" fill="#64748b"/>
      <circle cx="200" cy="180" r="4" fill="#64748b"/>
      <circle cx="250" cy="190" r="4" fill="#64748b"/>

      <!-- Legend & Educational Key (Right Side) -->
      <rect x="515" y="25" width="225" height="200" rx="6" fill="#0f172a" stroke="#334155" stroke-width="1"/>
      <text x="627" y="48" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#f8fafc">Paretova Hierarchie & PCD</text>
      
      <!-- Legend Item 1 -->
      <circle cx="535" cy="72" r="5" fill="#38bdf8"/>
      <text x="548" y="76" font-size="10" font-weight="bold" fill="#38bdf8">Fronta 1 (Nedominované)</text>
      <text x="548" y="90" font-size="8.5" fill="#94a3b8">Žádné jiné řešení není lepší v obou osách</text>

      <!-- Legend Item 2 -->
      <circle cx="535" cy="112" r="4.5" fill="#34d399"/>
      <text x="548" y="116" font-size="10" font-weight="bold" fill="#34d399">Fronta 2 (Dominované 1. řádu)</text>
      <text x="548" y="130" font-size="8.5" fill="#94a3b8">Dominovány pouze molekulami z Fronty 1</text>

      <!-- Crowding Distance Explanation -->
      <line x1="530" y1="145" x2="725" y2="145" stroke="#334155" stroke-width="1"/>
      <text x="535" y="165" font-size="10" font-weight="bold" fill="#fbbf24">Crowding Distance (PCD):</text>
      <text x="535" y="182" font-size="8.5" fill="#e2e8f0">• Odměňuje osamocené molekuly v prostoru</text>
      <text x="535" y="196" font-size="8.5" fill="#e2e8f0">• Zabraňuje zúžení na jediný tvar (Mode Collapse)</text>
      <text x="535" y="210" font-size="8.5" fill="#38bdf8">• Hraniční body získávají d = ∞ (ochrana extrémů)</text>
    </svg>
  </div>
</div>`;
}

export function createBricsCleavageSvg() {
  return `
<div class="schematic-container" data-schematic="brics-cleavage">
  <div class="schematic-header">
    <span class="schematic-title">✂️ Vektorové Schéma: BRICS Fragmentace Molekuly & Syntonové Exit-Vektory</span>
    <span class="schematic-badge">16 Pravidel Retrosyntézy</span>
  </div>
  <div class="schematic-svg-wrap">
    <svg viewBox="0 0 760 250" class="schematic-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- Background card -->
      <rect x="2" y="2" width="756" height="246" rx="8" fill="#0b1120" stroke="#1e293b" stroke-width="1.5"/>

      <!-- Fragment 1: Aromatic Ring with Synthon [16*] -->
      <g transform="translate(40, 50)">
        <rect x="0" y="0" width="180" height="135" rx="6" fill="#1e1b4b" stroke="#6366f1" stroke-width="1.5"/>
        <text x="90" y="22" text-anchor="middle" font-size="10" font-weight="bold" fill="#a5b4fc">Fragment A (Aromatický jádrový)</text>
        
        <!-- Benzen ring schematic -->
        <polygon points="70,55 95,42 120,55 120,85 95,98 70,85" fill="none" stroke="#818cf8" stroke-width="2"/>
        <circle cx="95" cy="70" r="14" fill="none" stroke="#818cf8" stroke-width="1.5" stroke-dasharray="3 2"/>

        <!-- Exit vector [16*] -->
        <line x1="120" y1="70" x2="160" y2="70" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="3 2"/>
        <circle cx="160" cy="70" r="11" fill="#be123c" stroke="#f43f5e" stroke-width="2"/>
        <text x="160" y="74" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ffffff">[16*]</text>
        <text x="90" y="122" text-anchor="middle" font-size="9" fill="#c7d2fe">c1ccccc1-[16*]</text>
      </g>

      <!-- Cleavage Bond Cut 1 -->
      <g transform="translate(230, 85)">
        <line x1="0" y1="0" x2="20" y2="70" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="4 3"/>
        <text x="10" y="-10" text-anchor="middle" font-size="12">✂️</text>
        <text x="10" y="85" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#f43f5e">Pravidlo L1</text>
      </g>

      <!-- Fragment 2: Central Amide Linker [4*]--[8*] -->
      <g transform="translate(260, 50)">
        <rect x="0" y="0" width="220" height="135" rx="6" fill="#064e3b" stroke="#059669" stroke-width="1.5"/>
        <text x="110" y="22" text-anchor="middle" font-size="10" font-weight="bold" fill="#6ee7b7">Fragment B (Amidický Linker)</text>

        <!-- Exit vector [4*] (Carbonyl) -->
        <circle cx="25" cy="70" r="11" fill="#047857" stroke="#34d399" stroke-width="2"/>
        <text x="25" y="74" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ffffff">[4*]</text>
        <line x1="36" y1="70" x2="70" y2="70" stroke="#34d399" stroke-width="2"/>

        <!-- Carbonyl C=O -->
        <circle cx="70" cy="70" r="9" fill="#0f766e"/>
        <text x="70" y="74" text-anchor="middle" font-size="9" fill="#ffffff">C</text>
        <line x1="70" y1="61" x2="70" y2="42" stroke="#f43f5e" stroke-width="2"/>
        <line x1="74" y1="61" x2="74" y2="42" stroke="#f43f5e" stroke-width="2"/>
        <circle cx="72" cy="36" r="8" fill="#be123c"/>
        <text x="72" y="39" text-anchor="middle" font-size="8" fill="#ffffff">O</text>

        <!-- Amide N-H -->
        <line x1="79" y1="70" x2="115" y2="70" stroke="#34d399" stroke-width="2"/>
        <circle cx="115" cy="70" r="9" fill="#1d4ed8"/>
        <text x="115" y="74" text-anchor="middle" font-size="8" fill="#ffffff">NH</text>

        <!-- Spacer CH2-CH2 -->
        <line x1="124" y1="70" x2="160" y2="70" stroke="#34d399" stroke-width="2"/>
        <text x="142" y="65" text-anchor="middle" font-size="8.5" fill="#a7f3d0">CH₂</text>

        <!-- Exit vector [8*] -->
        <line x1="160" y1="70" x2="195" y2="70" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="3 2"/>
        <circle cx="195" cy="70" r="11" fill="#047857" stroke="#34d399" stroke-width="2"/>
        <text x="195" y="74" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ffffff">[8*]</text>
        <text x="110" y="122" text-anchor="middle" font-size="9" fill="#a7f3d0">[4*]C(=O)NH-CH2-[8*]</text>
      </g>

      <!-- Cleavage Bond Cut 2 -->
      <g transform="translate(490, 85)">
        <line x1="0" y1="0" x2="20" y2="70" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="4 3"/>
        <text x="10" y="-10" text-anchor="middle" font-size="12">✂️</text>
        <text x="10" y="85" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#f43f5e">Pravidlo L5</text>
      </g>

      <!-- Fragment 3: Basic Solubilizing Group (Piperazine) -->
      <g transform="translate(520, 50)">
        <rect x="0" y="0" width="200" height="135" rx="6" fill="#451a03" stroke="#d97706" stroke-width="1.5"/>
        <text x="100" y="22" text-anchor="middle" font-size="10" font-weight="bold" fill="#fde68a">Fragment C (Bazický Solubilizátor)</text>

        <!-- Exit vector [5*] -->
        <circle cx="30" cy="70" r="11" fill="#b45309" stroke="#fbbf24" stroke-width="2"/>
        <text x="30" y="74" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#ffffff">[5*]</text>
        <line x1="41" y1="70" x2="70" y2="70" stroke="#fbbf24" stroke-width="2"/>

        <!-- Piperazine ring -->
        <rect x="70" y="50" width="55" height="40" rx="4" fill="none" stroke="#f59e0b" stroke-width="2"/>
        <text x="97" y="74" text-anchor="middle" font-size="8.5" fill="#fef3c7">N-Methyl</text>
        
        <text x="100" y="122" text-anchor="middle" font-size="9" fill="#fef3c7">[5*]N1CCN(C)CC1</text>
      </g>

      <!-- Bottom Explanatory Banner -->
      <rect x="40" y="198" width="680" height="36" rx="4" fill="#0f172a" stroke="#334155" stroke-width="1"/>
      <text x="380" y="214" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#38bdf8">Klíč retrosyntézy: Izotopy [1*] až [16*] striktně kódují povolenou chemickou valenci v DrugEx síti</text>
      <text x="380" y="227" text-anchor="middle" font-size="8.5" fill="#94a3b8">Zabraňuje tvorbě nestabilních struktur (např. spojením [4*] s [4*] nevznikne nechtěný anhydrid či diketon)</text>
    </svg>
  </div>
</div>`;
}
