/**
 * DrugEx Hub — Module 6: Scaffolds, Fragmenters, CLI & HPC Automation
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M6_LECTURES = {
  // =========================================================================
  // LECTURE 6.1
  // =========================================================================
  "l6_1": {
    id: "l6_1",
    tag: "Core",
    relevance: 9,
    title: "6.1 Fragmentový návrh, BRICS štěpení & FragSequenceExplorer",
    summary: "Cílený růst z fragmentů, design linkerů pro vícedoménové proteiny a fragmentační konvertory.",
    slides: [
      {
        title: "1. Význam fragmentového a lešením řízeného (Scaffold-Based) designu",
        content: `Při navrhování ligandů pro flexibilní cíle nebo proteiny s více vazebnými doménami je často znám klíčový farmakoforový fragment (např. heteroaromatický kruh vázající se do specifické kapsy), ale je potřeba navrhnout variabilní postranní řetězce nebo linkery.
        <br><br>
        Výhody fragmentového přístupu:
        <ul>
          <li><strong>Zaručená syntetická dostupnost</strong>: Růst molekuly probíhá skládáním známých syntetických synthonů.</li>
          <li><strong>Ukotvení klíčových interakcí</strong>: Zachování stěžejních H-vazeb nebo $\pi$-interakcí z krystalografie/NMR.</li>
          <li><strong>Design bifunkčních molekul (PROTACs / Molecular Glues)</strong>: Kde generátor navrhuje optimální 3D linker spojující dvě různé vazebné jednotky.</li>
        </ul>`
      },
      {
        title: "2. BRICS pravidla fragmentace v DrugEx",
        content: `Algoritmus <strong>BRICS (Breaks on Retrosynthetically Interesting Chemical Substructures)</strong> definuje 16 typů štěpných vazeb odpovídajících robustním chemickým reakcím:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">Syntetické vazby L1 - L8</div>
            <ul>
              <li><strong>L1–L2</strong>: Amidické vazby (reakce aminu s karboxylovou kyselinou).</li>
              <li><strong>L3–L4</strong>: Esterové a sulfonamidové vazby.</li>
              <li><strong>L5–L6</strong>: C-C vazby mezi aromatickými kruhy (Suzuki-Miyaura coupling).</li>
              <li><strong>L7–L8</strong>: Etherové vazby (Williamsonova syntéza).</li>
            </ul>
          </div>
          <div class="compare-col right">
            <div class="compare-heading">Syntetické vazby L9 - L16</div>
            <ul>
              <li><strong>L9–L10</strong>: Aminové vazby (reduktivní aminace).</li>
              <li><strong>L11–L12</strong>: Močoviny a thiomočoviny.</li>
              <li><strong>L13–L14</strong>: Karbamáty a guanidiny.</li>
              <li><strong>L15–L16</strong>: C-C vazby alifatické (Grignardova reakce, alkylace).</li>
            </ul>
          </div>
        </div>
        <br>
        Modely <code>FragSequenceExplorer</code> a <code>FragGraphExplorer</code> berou na vstupu fragmentový prefix a generují pokračování molekuly se zachováním definovaného lešení.`
      }
    ]
  },

  // =========================================================================
  // LECTURE 6.2
  // =========================================================================
  "l6_2": {
    id: "l6_2",
    tag: "Core",
    relevance: 10,
    title: "6.2 Příkazová řádka DrugEx: drugex dataset, train & generate",
    summary: "Kompletní matice parametrů CLI pro automatizované skriptování a batch zpracování velkých sad dat.",
    slides: [
      {
        title: "1. Kompletní matice parametrů CLI DrugEx",
        content: `DrugEx disponuje robustním rozhraním příkazové řádky pro automatizaci na superpočítačích:
        <br><br>
        <h4>1. Příprava dat: <code>python -m drugex.dataset</code></h4>
        <ul>
          <li><code>-b, --base_dir</code>: Kořenový adresář projektu.</li>
          <li><code>-i, --input_file</code>: Vstupní soubor TSV/CSV/SDF s molekulami.</li>
          <li><code>-mc, --mol_col</code>: Název sloupce se SMILES (např. <code>SMILES</code>).</li>
          <li><code>-o, --out</code>: Prefix výstupního datasetu (např. <code>ccr2_data</code>).</li>
          <li><code>-mt, --mol_type</code>: Typ reprezentace: <code>smiles</code> nebo <code>graph</code>.</li>
          <li><code>-sf, --scaffold</code>: SMILES fixního lešení pro fragmentové modely.</li>
          <li><code>-nof, --no_filter</code>: Vypnutí základních fyzikálních filtrů pro zachování všech ligandů.</li>
        </ul>
        <br>
        <h4>2. Trénování: <code>python -m drugex.train</code></h4>
        <ul>
          <li><code>-tm, --train_mode</code>: Režim tréninku: <code>PT</code> (Pre-training), <code>FT</code> (Fine-tuning), <code>RL</code> (Reinforcement Learning).</li>
          <li><code>-ag, --agent_model</code>: Cesta k výchozímu modelu agenta.</li>
          <li><code>-pr, --prior_model</code>: Cesta k mutační síti (Prior).</li>
          <li><code>-e, --epochs</code>: Počet trénovacích epoch (např. 50 nebo 100).</li>
          <li><code>-bs, --batch_size</code>: Velikost dávky (obvykle 64 nebo 128).</li>
          <li><code>-lr, --learning_rate</code>: Rychlost učení (obvykle $10^{-4}$).</li>
          <li><code>-gpu, --gpu_id</code>: ID grafické karty (např. <code>0</code> nebo <code>cpu</code>).</li>
        </ul>`
      },
      {
        title: "2. Příklad automatizovaného batch skriptu",
        content: `Kompletní tříkrokový řetězec v Bash:`,
        code: `#!/usr/bin/env bash
set -e

WORKDIR="/home/kolar/learn_projects/drugex/workdir"
mkdir -p "$WORKDIR"

echo "=== 1. Krok: Preprocessing cílového datasetu ==="
python -m drugex.dataset \\
    -b "$WORKDIR" \\
    -i ../data/benchmarks/CCR_HUMAN_AL.tsv \\
    -mc SMILES \\
    -o ccr2_corpus \\
    -mt smiles \\
    -nof

echo "=== 2. Krok: Spuštění Fine-Tuningu ==="
python -m drugex.train \\
    -tm FT \\
    -b "$WORKDIR" \\
    -i ccr2_corpus \\
    -ag ../models/Papyrus05.5_smiles_rnn_PT.pkg \\
    -e 100 \\
    -bs 64 \\
    -gpu 0

echo "=== 3. Krok: Generování a skórování molekul ==="
python -m drugex.generate \\
    -b "$WORKDIR" \\
    -g "$WORKDIR/ccr2_corpus_FT.pkg" \\
    -n 5000 \\
    -gpu 0 \\
    --keep_undesired`
      }
    ]
  },

  // =========================================================================
  // LECTURE 6.3
  // =========================================================================
  "l6_3": {
    id: "l6_3",
    tag: "WOW",
    relevance: 9,
    title: "6.3 Škálování na GPU klastru: Slurm skripty & správa paměti",
    summary: "Šablony Slurm úloh s alokací GPU, proměnnými prostředí (OE_LICENSE) a ochranou proti přetečení RAM/VRAM.",
    slides: [
      {
        title: "1. Produkční Slurm dávkový skript pro GPU klastry (MetaCentrum / IT4Innovations)",
        content: `Výpočetní experimenty kombinující de novo generování se stovkami tisíc 3D konformačních zarovnání vyžadují akademický superpočítač. Níže je kompletní, otestovaná šablona dávkového skriptu pro plánovač úloh Slurm:`,
        code: `#!/bin/bash
# ==============================================================================
# Slurm Job Script: DrugEx MORL + 3D ROCS Shape Matching
# Bachelor's Thesis: De Novo Drug Design on Flexible Targets / IDPs
# ==============================================================================
#SBATCH --job-name=drugex_rocs_thesis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/drugex_%j.out
#SBATCH --error=logs/drugex_%j.err

set -euo pipefail

echo "========================================================="
echo " Start vypoctu na uzlu: $(hostname)"
echo " Datum a cas           : $(date)"
echo " Job ID                : $SLURM_JOB_ID"
echo " Alokovane CPU         : $SLURM_CPUS_PER_TASK jader"
echo " Alokovana GPU         : $CUDA_VISIBLE_DEVICES"
echo "========================================================="

# 1. Nacteni softwarovych modulu
module purge
module load cuda/12.1 python/3.11 cmake gcc/11.2

# 2. Aktivace virtualniho prostredi
source /storage/home/kolar/.venvs/drugex/bin/activate

# 3. Nastaveni kritickych promennych prostredi
export OE_LICENSE=/storage/projects/licenses/oe_license.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 4. Spusteni RL treninkoveho tutorialu
python tutorial/advanced/rocs/rocs_rl_tutorial.ipynb

echo "========================================================="
echo " Vypocet uspesne dokoncen: $(date)"
echo "========================================================="`
      },
      {
        title: "2. Ochrana před přetečením paměti a CPU oversubscription",
        content: `Při nasazení na klastru se často vyskytují dvě fatální chyby:
        <br><br>
        <h4>1. CPU Oversubscription (Přetížení jader)</h4>
        Knihovny NumPy, SciPy a PyTorch automaticky vytvářejí tolik podvláken (threads), kolik má server fyzických jader (např. 128). Pokud <code>RDKitROCSScorer</code> spustí 16 procesů a každý proces vytvoří 128 vláken, vznikne $16 \times 128 = 2048$ vláken, což způsobí zahlcení CPU plánovače a desetinásobné zpomalení výpočtu.
        <br><br>
        <strong>Řešení:</strong> Explicitní nastavení <code>export OMP_NUM_THREADS=1</code> a <code>num_threads=1</code> v <code>RDKitConformerGenerator</code>.
        <br><br>
        <h4>2. GPU VRAM Memory Leaks v PyTorch</h4>
        Během generování velkých batší dochází k ukládání mezilehlých tenzorů do grafické paměti. V kódu je nutné provádět vzorkování v kontextu <code>with torch.no_grad():</code> a volat <code>torch.cuda.empty_cache()</code> na konci každé epochy.`
      }
    ]
  }
};
