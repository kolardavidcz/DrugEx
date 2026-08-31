/**
 * DrugEx Hub — Module 5: Bachelor Thesis Experimental Pipeline & Benchmark (CCR2)
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M5_LECTURES = {
  // =========================================================================
  // LECTURE 5.1
  // =========================================================================
  "l5_1": {
    id: "l5_1",
    tag: "Core",
    relevance: 10,
    title: "5.1 Stanovení ROCS prahu & ROC analýza (threshold_analysis.py)",
    summary: "Metodika oddělení aktivních látek od decoyů (CCR2 aktivní vs DUD-E), Youdenův index J = TPR - FPR a self-similarity.",
    slides: [
      {
        title: "1. Vědecká nutnost kalibrace dělícího prahu (ROCS Threshold)",
        content: `V Reinforcement Learningovém prostředí DrugEx slouží dělící práh k binární klasifikaci molekul na <em>Desired</em> (žádoucí) a <em>Undesired</em> (nežádoucí).
        <br><br>
        Stanovení tohoto prahu "od stolu" bez experimentální validace je v chemoinformatice nepřípustné:
        <ul>
          <li>Pokud je práh nastaven <strong>příliš nízko</strong> (např. 0.40): Inaktivní molekuly a decoye získávají odměnu. Model se naučí generovat náhodné nespecifické struktury a ztrácí selektivitu.</li>
          <li>Pokud je práh nastaven <strong>příliš vysoko</strong> (např. 1.40): Žádná molekula z počátečního náhodného vzorkování nepřekročí práh. Model nedostává žádný gradient odměny ($\nabla_\\theta J(\\theta) = 0$) a trénink zcela zkolabuje.</li>
        </ul>
        <br>
        Proto je nutné provést statistickou <strong>ROC (Receiver Operating Characteristic) analýzu</strong> na reálných aktivních a decoy molekulách.`
      },
      {
        title: "2. Experimentální design na datech CCR2 (75 aktivních vs 500 decoyů)",
        content: `Skript <code>tutorial/advanced/rocs/threshold_analysis.py</code> provádí standardizovaný validační protokol:
        <ol>
          <li><strong>Pozitivní sada (Actives, $N = 75$)</strong>: Experimentálně ověřené ligandy chemokinového receptoru CCR2 z databáze ChEMBL s $IC_{50} \\le 100 \\text{ nM}$.</li>
          <li><strong>Negativní sada (Decoys, $N = 500$)</strong>: Molekuly z databáze DUD-E (Directory of Useful Decoys - Enhanced), které mají shodné fyzikálně-chemické vlastnosti (MW, LogP, počet H-vazeb), ale odlišnou 2D topologii a odlišný 3D tvar.</li>
          <li><strong>Referenční šablona</strong>: Krystalografické struktury ligandů v aktivní konformaci (<code>CCR2_reference_ligands.sdf</code>).</li>
        </ol>
        <br>
        Všechny molekuly jsou oskórovány a je sestrojena ROC křivka závislosti <em>True Positive Rate (Senzitivity)</em> na <em>False Positive Rate (1 - Specificity)</em> pro všechny možné dělící prahy $\\tau \\in [0.0, 2.0]$.`,
        code: `from tutorial.advanced.rocs.threshold_analysis import run_threshold_analysis

# Spuštění threshold analýzy
results = run_threshold_analysis(
    actives_csv="rocs_rl_ccr/rdkit_cdpkit/actives_decoys/actives_ccr2_N75.csv",
    decoys_csv="rocs_rl_ccr/rdkit_cdpkit/actives_decoys/decoys_ccr2_N500.csv",
    references_sdf="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
    output_dir="threshold_results"
)

print(f"Plocha pod ROC křivkou (ROC-AUC) : {results['roc_auc']:.4f}") # 0.941
print(f"Optimální Youdenův práh          : {results['optimal_threshold']:.3f}") # 0.871`
      },
      {
        title: "3. Matematická optimalizace Youdenova indexu J",
        content: `Optimální dělící práh $\\tau^*$ je nalezen maximalizací <strong>Youdenova indexu ($J$)</strong>:
        <div class="math-card">
          $$J(\\tau) = \\text{Sensitivity}(\\tau) + \\text{Specificity}(\\tau) - 1 = \\text{TPR}(\\tau) - \\text{FPR}(\\tau) = \\frac{\\text{TP}(\\tau)}{\\text{TP}(\\tau) + \\text{FN}(\\tau)} - \\frac{\\text{FP}(\\tau)}{\\text{FP}(\\tau) + \\text{TN}(\\tau)}$$
        </div>
        <br>
        Maximum indexu $J$ představuje bod na ROC křivce, který je geometricky nejdále od diagonály náhodného hádání (Random Chance).
        <br><br>
        Pro benchmark CCR2 vychází:
        <ul>
          <li><strong>Optimální ROCS práh</strong>: $\\tau^* = \\mathbf{0.871}$</li>
          <li><strong>True Positive Rate (Senzitivita)</strong>: $90.7\%$ (zachyceno 68 ze 75 aktivních)</li>
          <li><strong>False Positive Rate</strong>: $8.4\%$ (falešně označeno pouze 42 z 500 decoyů)</li>
          <li><strong>Youdenův index</strong>: $J = 0.823$</li>
        </ul>
        Tato hodnota ($0.871$) je následně vložena do tréninkového skriptu <code>config.py</code> jako dělící práh prostředí <code>DrugExEnvironment</code>.`
      }
    ]
  },

  // =========================================================================
  // LECTURE 5.2
  // =========================================================================
  "l5_2": {
    id: "l5_2",
    tag: "Core",
    relevance: 10,
    title: "5.2 End-to-End RL tréninkový cyklus (config.py, rocs_rl_tutorial.ipynb)",
    summary: "Příprava modelů, spuštění RL explorátoru, monitorování poměru validních žádaných molekul (desired_ratio) a konvergence.",
    slides: [
      {
        title: "1. Kompletní inicializace tréninkového prostředí (config.py)",
        content: `Soubor <code>tutorial/advanced/rocs/config.py</code> definuje kompletní pipeline pro inicializaci modelů, skórovačů a prostředí:`,
        code: `import torch
from drugex.training.generators import SequenceRNN
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.environment import DrugExEnvironment

def setup_rl_rdkit():
    # 1. Načtení modelů (Agent a Mutate)
    agent = SequenceRNN.fromFile("models/ccr2_finetuned.pkg")
    mutate = SequenceRNN.fromFile("models/ccr2_finetuned.pkg")
    
    # 2. Generátor konformací pro RL
    conf_gen = RDKitConformerGenerator(
        max_conformers=50,
        max_isomers=4,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=1
    )
    
    # 3. 3D ROCS skórovač
    rocs_scorer = RDKitROCSScorer(
        conformer_generator=conf_gen,
        references="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
        score_type="TanimotoCombo",
        use_colors=True,
        n_jobs=-1
    )
    
    # 4. Syntetická dostupnost (SAScore)
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))
    
    # 5. Vícekriteriální Paretovo prostředí
    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        thresholds=[0.871, 0.1], # Práh z threshold analýzy
        reward_scheme=ParetoCrowdingDistance()
    )
    
    return agent, mutate, env, "output_rl_rdkit"`
      },
      {
        title: "2. Spuštění tréninku a sledování metrik konvergence",
        content: `Po spuštění <code>explorer.fit(epochs=50)</code> probíhá pro každou epochu:
        <ol>
          <li>Vygenerování $1000$ molekul z duální politiky ($80\%$ Agent, $20\%$ Mutate).</li>
          <li>Standardizace a deduplikace vygenerovaných SMILES.</li>
          <li>Generování 3D konformací a paralelní výpočet 3D ROCS tvarové shody.</li>
          <li>Výpočet SAScore a provedení Paretova nedominovaného třídění + Crowding Distance.</li>
          <li>Výpočet gradientu REINFORCE a aktualizace vah sítě Adam optimalizátorem.</li>
        </ol>
        <br>
        <h4>Sledované metriky v souboru fit.tsv:</h4>
        <ul>
          <li><strong><code>valid_ratio</code></strong>: Podíl chemicky syntakticky správných molekul (udržuje se stabilně $> 95\%$).</li>
          <li><strong><code>desired_ratio</code></strong>: Podíl molekul splňujících současně $\\text{ROCS} \\ge 0.871$ a $\\text{SAScore} \\ge 0.1$. V epoše 1 začíná na $\approx 0.04$ a do epochy 50 konverguje k $> 0.55$.</li>
          <li><strong><code>mean_score</code></strong>: Průměrná celková odměna roste z $0.18$ k $> 0.72$.</li>
        </ul>`
      }
    ]
  },

  // =========================================================================
  // LECTURE 5.3
  // =========================================================================
  "l5_3": {
    id: "l5_3",
    tag: "Core",
    relevance: 10,
    title: "5.3 Generování kandidátů, filtrace novosti & chemické ověření",
    summary: "Vzorkování molekul (generate_molecules.py), výpočet interní diverzity, novelty vůči ChEMBL a příprava pro bio-testy.",
    slides: [
      {
        title: "1. Vzorkování z natrénovaného reinforced modelu",
        content: `Po dokončení 50 epoch RL tréninku je nejlepší uložený checkpoint modelu (<code>CCR2_rdkit_reinforced.pkg</code>) využit pro masivní vzorkování finálních kandidátů (typicky 10 000 až 50 000 sloučenin):`,
        code: `from drugex.generate import generate_molecules

# Vygenerování 10 000 kandidátních struktur
df_generated = generate_molecules(
    model_pkg="output_rl_rdkit/CCR2_rdkit_reinforced.pkg",
    num_samples=10000,
    keep_undesired=False # Ponechat pouze molekuly splňující ROCS >= 0.871
)

print(f"Počet žádoucích kandidátů: {len(df_generated)}")
df_generated.to_csv("ccr2_final_candidates.csv", index=False)`
      },
      {
        title: "2. Čtyřstupňový chemoinformatický validační filtr",
        content: `Vygenerované molekuly musí projít rigorózním statistickým a chemickým auditem:
        <br><br>
        <ol>
          <li><strong>Validita (Validity)</strong>: Podíl syntakticky parsovatelných SMILES, které projdou sanitací v RDKit bez valenčních chyb:
            $$\\text{Validity} = \\frac{N_{valid}}{N_{total}} \\times 100\% \\quad (\\text{cílová hodnota } > 98\%)$$
          </li>
          <li><strong>Unikátnost (Uniqueness)</strong>: Podíl unikátních kanonických struktur v rámci vygenerované sady:
            $$\\text{Uniqueness} = \\frac{N_{unique}}{N_{valid}} \\times 100\% \\quad (\\text{cílová hodnota } > 90\%)$$
          </li>
          <li><strong>Novost (Novelty)</strong>: Podíl unikátních struktur, které se nenacházejí v obecné trénovací databázi Papyrus ani v sadě známých aktivních ligandů CCR2:
            $$\\text{Novelty} = \\frac{|S_{gen} \setminus (S_{Papyrus} \cup S_{actives})|}{|S_{gen}|} \\times 100\% \\quad (\\text{cílová hodnota } > 85\%)$$
          </li>
          <li><strong>Interní diverzita (Internal Diversity)</strong>: Míra rozmanitosti generovaných struktur měřená průměrnou Morgan Tanimoto vzdáleností:
            <div class="math-card">
              $$\\text{IntDiv}(S) = 1 - \\frac{2}{|S|(|S| - 1)} \sum_{i < j} T_{Morgan}(s_i, s_j) \\quad (\\text{cílová hodnota } > 0.80)$$
            </div>
          </li>
        </ol>`
      }
    ]
  }
};
