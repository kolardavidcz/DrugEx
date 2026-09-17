/**
 * DrugEx Hub — Practitioner's Handbook & Hyperparameter Optimization Cookbook
 * Tailored for researchers composing multi-objective environments (QSAR, ROCS, SA, SIM, etc.)
 */

export const COOKBOOK_DATA = {
  title: "👨‍🔬 Uživatelská Kuchařka & Hyperparametrický Manuál",
  subtitle: "Praktické recepty, skládání skórovačů jako stavebnice LEGO (ROCS + QSAR + SIM + SA), ladění hyperparametrů a diagnostický strom konvergence.",
  
  // =========================================================================
  // SECTION 1: HYPERPARAMETER TUNING MATRIX
  // =========================================================================
  tuningMatrix: [
    {
      param: "learning_rate (lr)",
      defaultVal: "1e-4",
      searchRange: "5e-5 až 3e-4",
      role: "Rychlost aktualizace vah neuronového generátoru v Policy Gradient kroku.",
      tuningGuide: "Při lr >= 1e-3 dojde během 5 epoch ke kolapsu chemické gramatiky (validita < 50%). Při lr <= 1e-5 se model učí příliš pomalu a nestihne konvergovat za 50 epoch. Hodnota 1e-4 je zlatý standard.",
      risk: "Explodující gradienty vs zamrznutí učení"
    },
    {
      param: "epsilon (eps)",
      defaultVal: "0.20",
      searchRange: "0.10 až 0.35",
      role: "Poměr stochastické explorace z fixní Prior sítě (π₀) vůči učícímu se Agentovi (π_θ).",
      tuningGuide: "V 80 % kroků generuje Agent (exploatace vysokého skóre), ve 20 % mutační síť (přísun nových funkčních skupin). Pokud v tréninku nastane Mode Collapse (duplicita > 30%), zvyšte epsilon na 0.25–0.30.",
      risk: "Mode collapse vs ztráta zacílení na cíl"
    },
    {
      param: "n_samples (batch)",
      defaultVal: "1000",
      searchRange: "500 až 2000",
      role: "Počet nově vygenerovaných molekul v každé RL epoše pro výpočet odměn.",
      tuningGuide: "Pro spolehlivé rozřazení do Paretovských vrstev a výpočet Crowding Distance je potřeba alespoň 500–1000 molekul. Na výkonných GPU klastrech doporučujeme 1000–2000 pro maximální stabilitu Paretovy fronty.",
      risk: "Degenerované Paretovy fronty při malém N"
    },
    {
      param: "epochs",
      defaultVal: "50",
      searchRange: "30 až 100",
      role: "Celkový počet cyklů zpětnovazebního učení s brzkým zastavením (Early Stopping).",
      tuningGuide: "Pro SequenceRNN stačí 50 epoch (cca 50 000 vyhodnocených molekul). Pokud se 'desired_ratio' nezlepší po dobu 20 epoch (patience=20), trénink se automaticky zastaví.",
      risk: "Přetrénování (overfitting) a ztráta diverzity"
    },
    {
      param: "temperature (T)",
      defaultVal: "1.0",
      searchRange: "0.75 až 1.25",
      role: "Teplotní faktor ve funkci Softmax při autoregresním vzorkování tokenů.",
      tuningGuide: "T=0.85 pro konzervativní generování blízkých analogů s vysokou validitou (>98%). T=1.0 pro vyvážený běh. T=1.15 pro explorativní Scaffold Hopping u neznámých IDP kavit.",
      risk: "Fádní repetice vs nevalidní syntaktické chyby"
    },
    {
      param: "max_conformers",
      defaultVal: "30–50",
      searchRange: "20 až 100",
      role: "Maximální počet 3D konformací generovaných pro každou molekulu před 3D zarovnáním.",
      tuningGuide: "30 konformací pro RDKit/CDPKit poskytuje 90 % maximálního možného překryvu při polovičním výpočetním čase oproti 100 konformacím. Vhodné pro rychlé iterace.",
      risk: "Časová náročnost RL epochy"
    },
    {
      param: "num_threads",
      defaultVal: "1",
      searchRange: "Striktně 1 při n_jobs > 1",
      role: "Počet interních vláken v generátoru konformací uvnitř každého multiprocessing workeru.",
      tuningGuide: "KRITICKÉ: Při n_jobs=16 paralelních procesech MUSÍ mít každý worker nastaveno num_threads=1. Nastavení 0 (všechna jádra) způsobí 16x16=256 vláken a pád systému (CPU Thrashing)!",
      risk: "Systémový deadlock a zamrznutí"
    }
  ],

  // =========================================================================
  // SECTION 2: PRACTITIONER RECIPES
  // =========================================================================
  recipes: [
    {
      id: "recipe1",
      title: "Recept 1: Kompletní Multi-Objektivní Stack (ROCS + QSAR + SA + Fyzikální Vlastnosti)",
      badge: "Základní Recept pro Bakalářskou Práci",
      desc: "Jak v DrugExEnvironment elegantně zkombinovat 3D tvarové porovnávání (ROCS), strojově učený QSAR model afinity, syntetickou dostupnost (SAScore) a Lipinského fyzikálně-chemické filtry (MW, LogP, TPSA, QED).",
      code: `#!/usr/bin/env python3
"""
Recept 1: Plný Multi-Objektivní Stack v DrugExEnvironment
Kombinace 3D ROCS + QSAR + SAScore + MW + LogP + TPSA + QED
"""

import os
from pathlib import Path
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore, MinMaxGaussian, Gaussian
from drugex.training.scorers.properties import Property
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.qsprpred import QSPRPredScorer # nebo vlastní Scorer

def build_full_practitioner_environment():
    # 1. 3D ROCS Tvarový a Farmakoforový Skórovač (TanimotoCombo)
    conformer_engine = RDKitConformerGenerator(
        max_conformers=40,
        max_isomers=4,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=1,          # Vláknová bezpečnost v multiprocessing poolu
        show_progress=False
    )
    rocs_scorer = RDKitROCSScorer(
        conformer_generator=conformer_engine,
        references="data/CCR2_reference_ligands.sdf",
        score_type="TanimotoCombo",
        use_colors=True,
        n_jobs=-1               # Využije všechna dostupná CPU jádra
    )

    # 2. QSAR Prediktor Bioaktivity (pIC50 k cíli CCR2)
    # Požadujeme pIC50 >= 7.0 (odpovídá Ki < 100 nM)
    qsar_activity = QSPRPredScorer(model_path="models/qsar_ccr2_random_forest.pkg")
    qsar_activity.setModifier(SmoothClippedScore(
        lower_x=6.0,            # pIC50 <= 6.0 získává odměnu ~0.0
        upper_x=8.0,            # pIC50 >= 8.0 získává maximální odměnu ~1.0
        high_score=1.0,
        low_score=0.0
    ))

    # 3. Syntetická Dostupnost (SAScore)
    # Chceme nízký SAScore (1.0 = snadná syntéza, 10.0 = neproveditelné)
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(
        lower_x=4.5,            # SAScore >= 4.5 je penalizován (odměna ~0.0)
        upper_x=2.5,            # SAScore <= 2.5 je ideální (odměna ~1.0)
        high_score=1.0,
        low_score=0.0
    ))

    # 4. Fyzikálně-chemické deskriptory (Lipinski & Veber pravidla)
    # Molekulová hmotnost (MW): optimální interval 360–480 Da
    mw_scorer = Property('MW', modifier=Gaussian(mu=420.0, sigma=60.0))

    # Lipofilita (LogP): penalizace mastných kyselin (LogP > 4.0)
    logp_scorer = Property('logP', modifier=MinMaxGaussian(mu=3.0, sigma=1.2, minimize=True))

    # Polární povrch (TPSA): optimální pro permeabilitu hematoencefalické bariéry / střeva
    tpsa_scorer = Property('TPSA', modifier=SmoothClippedScore(lower_x=30.0, upper_x=90.0))

    # 5. Sestavení celého prostředí v DrugExEnvironment
    scorers = [rocs_scorer, qsar_activity, sa_scorer, mw_scorer, logp_scorer, tpsa_scorer]
    
    # Jednotné prahy pro klasifikaci žádoucnosti (Desired)
    # Díky modifikátorům jsou všechny složky škálovány do [0, 1]
    thresholds = [
        0.871,  # Youdenův optimalizovaný práh pro ROCS TanimotoCombo
        0.500,  # QSAR aktivita práh (pIC50 > 7.0)
        0.500,  # SAScore práh (SAScore < 3.5)
        0.500,  # MW práh
        0.500,  # LogP práh
        0.500   # TPSA práh
    ]

    # Nedominované třídění do Paretových front s hustotní diverzitou
    reward_scheme = ParetoCrowdingDistance()

    env = DrugExEnvironment(
        scorers=scorers,
        thresholds=thresholds,
        reward_scheme=reward_scheme
    )
    
    print(f"✓ Úspěšně vytvořeno prostředí s {len(scorers)} cíli.")
    return env

if __name__ == "__main__":
    env = build_full_practitioner_environment()
`
    },
    {
      id: "recipe2",
      title: "Recept 2: 3D Scaffold Hopping & Odměňování Strukturní Novosti (ROCS + SimSmiles Penále)",
      badge: "Patentová Novost & IDP",
      desc: "Jak donutit generátor najít zcela nové chemické jádro (jiný heterocyklus) při zachování 100% prostorového tvaru a farmakoforových bodů původního známého inhibitoru.",
      code: `#!/usr/bin/env python3
"""
Recept 2: Scaffold Hopping s penále na podobnost k původnímu ligandu
Kombinace: 3D ROCS (vysoká tvarová shoda) + SimSmiles (penalizace strukturní podobnosti)
"""

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.similarity import SimSmiles

def build_scaffold_hopping_environment(parent_smiles: str, reference_sdf: str):
    # 1. 3D ROCS Scorer: Chceme, aby nová molekula přesně kopírovala 3D tvar
    conf_gen = RDKitConformerGenerator(max_conformers=40, num_threads=1)
    rocs = RDKitROCSScorer(
        conformer_generator=conf_gen,
        references=reference_sdf,
        score_type="TanimotoCombo"
    )

    # 2. Similarity Penále (SimSmiles):
    # Spočte 2D Morgan Tanimoto podobnost vůči původnímu patentovanému ligandu
    # Použijeme klesající modifikátor: čím je podobnost NIŽŠÍ, tím je odměna VYŠŠÍ!
    sim_penalty = SimSmiles(smiles=[parent_smiles], sim_type="tanimoto")
    sim_penalty.setModifier(SmoothClippedScore(
        lower_x=0.60,           # Podobnost >= 0.60 (analog mateřské látky) -> odměna ~0.0
        upper_x=0.25,           # Podobnost <= 0.25 (zcela nový scaffold) -> odměna ~1.0
        high_score=1.0,
        low_score=0.0
    ))

    # 3. Syntetická dostupnost
    sa = Property('SA', modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))

    # 4. Sestavení prostředí
    env = DrugExEnvironment(
        scorers=[rocs, sim_penalty, sa],
        thresholds=[0.871, 0.500, 0.500],
        reward_scheme=ParetoCrowdingDistance()
    )
    return env
`
    },
    {
      id: "recipe3",
      title: "Recept 3: Selektivní Dvou-Cílový Design (Aktivace Cíle A vs Potlačení Cíle B / hERG)",
      badge: "Farmakologická Bezpečnost",
      desc: "Jak optimalizovat molekulu pro vysokou afinitu k terapeutickému cíli (CCR2) a současně penalizovat vazbu k homolognímu receptoru (CCR5) nebo kardiálnímu iontovému kanálu hERG.",
      code: `#!/usr/bin/env python3
"""
Recept 3: Selektivní multi-cílový design (On-Target vs Off-Target)
"""

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

def build_selectivity_environment():
    # 1. 3D ROCS tvarové skóre vůči cíli A (CCR2)
    rocs = RDKitROCSScorer(
        conformer_generator=RDKitConformerGenerator(max_conformers=30, num_threads=1),
        references="data/CCR2_reference_ligands.sdf",
        score_type="TanimotoCombo"
    )

    # 2. QSAR On-Target aktivita (Cíl A: CCR2) -> maximalizovat (rostoucí křivka)
    on_target = QSPRPredScorer(model_path="models/qsar_ccr2.pkg")
    on_target.setModifier(SmoothClippedScore(lower_x=6.0, upper_x=8.0))

    # 3. QSAR Off-Target toxicita (Kardiotoxický hERG kanál) -> minimalizovat (klesající křivka)
    herg_filter = QSPRPredScorer(model_path="models/qsar_herg_channel.pkg")
    herg_filter.setModifier(SmoothClippedScore(
        lower_x=6.5,            # pIC50 k hERG >= 6.5 (vysoká toxicita) -> odměna ~0.0
        upper_x=4.5,            # pIC50 k hERG <= 4.5 (bezpečné) -> odměna ~1.0
        high_score=1.0,
        low_score=0.0
    ))

    # 4. Syntetická dostupnost
    sa = Property('SA', modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))

    env = DrugExEnvironment(
        scorers=[rocs, on_target, herg_filter, sa],
        thresholds=[0.871, 0.500, 0.500, 0.500],
        reward_scheme=ParetoCrowdingDistance()
    )
    return env
`
    },
    {
      id: "recipe4",
      title: "Recept 4: Automatická Optimalizace Hyperparametrů pomocí Optuna",
      badge: "Bayesovské Ladění pro Studenta",
      desc: "Kompletní skript pro automatické nalezení optimální kombinace learning rate, epsilonu a velikosti vzorku pro maximalizaci podílu žádoucích kandidátů (Desired Ratio).",
      code: `#!/usr/bin/env python3
"""
Recept 4: Automatické ladění hyperparametrů DrugEx MORL pomocí Optuna
"""

import optuna
import torch
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from drugex.training.explorers import SequenceExplorer
from drugex.training.monitors import FileMonitor
from config import setup_rl_rdkit

def objective(trial):
    # 1. Definice vyhledávacího prostoru hyperparametrů
    lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
    epsilon = trial.suggest_float("epsilon", 0.10, 0.30, step=0.05)
    n_samples = trial.suggest_categorical("n_samples", [500, 1000])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    # 2. Inicializace sítě a prostředí
    agent, mutate, env, output_dir = setup_rl_rdkit()
    agent.optim = torch.optim.Adam(agent.parameters(), lr=lr)

    # 3. Nastavení rychlého zkušebního tréninku (15 epoch)
    trial_dir = output_dir / f"optuna_trial_{trial.number}"
    monitor = FileMonitor(str(trial_dir / "fit"), save_smiles=False)
    
    explorer = SequenceExplorer(
        agent=agent,
        mutate=mutate,
        crover=None,
        env=env,
        epsilon=epsilon,
        n_samples=n_samples,
        batch_size=batch_size
    )

    # 4. Trénink
    explorer.fit(monitor=monitor, epochs=15)
    monitor.close()

    # 5. Vyhodnocení cílové metriky (poslední desired_ratio z logu)
    # Cílem je maximalizovat poměr molekul, které splňují všechny cíle
    import pandas as pd
    df = pd.read_csv(f"{trial_dir}/fit_fit.tsv", sep="\\t")
    final_desired_ratio = df["desired_ratio"].iloc[-1]
    final_mean_score = df["mean_score"].iloc[-1]

    # Kompozitní skóre pro Optuna
    return final_desired_ratio * 0.7 + final_mean_score * 0.3

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("🚀 Spouštím 10 pokusů Optuna pro nalezení nejlepších hyperparametrů...")
    study.optimize(objective, n_trials=10)

    print("\n=================================================")
    print("✨ NEJLEPŠÍ NALEZENÉ HYPERPARAMETRY:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"  Dosažená hodnota cíle: {study.best_value:.4f}")
    print("=================================================")
`
    },
    {
      id: "recipe5",
      title: "Recept 5: Selektivní Counter-Screening (Aktivace CCR2 vs. Penalizace Toxicity hERG)",
      badge: "Bezpečnost & Antitargety",
      desc: "Jak v DrugExu optimalizovat afinitu k terapeutickému cíli (CCR2) a současně aktivně trestat vazbu k hERG iontovému kanálu (prevence prodloužení QT intervalu a kardiotoxicity).",
      code: `#!/usr/bin/env python3
"""
Recept 5: Dual-Target Selektivita v DrugExEnvironment
Aktivita k cíli CCR2 (maximalizovat) vs. hERG kardiotoxicita (penalizovat)
"""

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.qsprpred import QSPRPredScorer

def build_counter_screening_environment():
    # 1. Pozitivní cíl: afinita k CCR2 (požadujeme pIC50 >= 7.0)
    target_ccr2 = QSPRPredScorer(model_path="models/qsar_ccr2_random_forest.pkg")
    target_ccr2.setModifier(SmoothClippedScore(
        lower_x=6.0,   # pIC50 <= 6.0: odměna 0.0
        upper_x=8.0,   # pIC50 >= 8.0: odměna 1.0 (maximalizace)
        high_score=1.0, low_score=0.0
    ))

    # 2. Antitarget: hERG afinita (chceme pIC50 < 5.0, penalizujeme vyšší)
    antitarget_herg = QSPRPredScorer(model_path="models/qsar_herg_svm.pkg")
    antitarget_herg.setModifier(SmoothClippedScore(
        lower_x=6.5,   # pIC50 >= 6.5: odměna 0.0 (tvrdá penalizace kardiotoxických látek)
        upper_x=4.5,   # pIC50 <= 4.5: odměna 1.0 (bezpečné molekuly)
        high_score=1.0, low_score=0.0
    ))

    # 3. Syntetická dostupnost jako stabilizátor
    sa_scorer = Property('SA', modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))

    env = DrugExEnvironment(
        scorers=[target_ccr2, antitarget_herg, sa_scorer],
        thresholds=[0.60, 0.70, 0.50],
        reward_scheme=ParetoCrowdingDistance()
    )
    return env
`
    },
    {
      id: "recipe6",
      title: "Recept 6: Fragment Growing s Fixním Jádrem (BRICS Scaffold-Constrained RL)",
      badge: "Fragment-Based Drug Design",
      desc: "Konfigurace FragSequenceExploreru pro rozvíjení nového farmakoforového ramene ze zadaného bioaktivního syntonu s ověřením 3D tvarové komplementarity v kapse.",
      code: `#!/usr/bin/env python3
"""
Recept 6: Fragment-based generování ze známého scaffoldového jádra
Fixní fragment (synton [16*]c1ccc(NC(=O)...)) + optimalizace rozšiřujícího ramene
"""

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesFragDataSet
from drugex.training.generators import SequenceRNN
from drugex.training.explorers import FragSequenceExplorer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

def setup_fragment_growing_rl(core_scaffold_smiles: str = "c1ccc(NC(=O)[16*])cc1"):
    # 1. Inicializace fragmentového datasetu s fixním jádrem
    dataset = SmilesFragDataSet("data/papyrus_fragments.tsv")
    voc = dataset.getVoc()

    # 2. Generátor specializovaný na fragmentové napojování
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile("models/frag_generator_prior.pkg")

    # 3. Prostředí hodnotící tvarový růst celé molekuly
    conformer_engine = RDKitConformerGenerator(max_conformers=30, num_threads=1)
    rocs_scorer = RDKitROCSScorer(
        conformer_generator=conformer_engine,
        references="data/CCR2_reference_ligands.sdf",
        score_type="TanimotoCombo"
    )

    env = DrugExEnvironment(
        scorers=[rocs_scorer],
        thresholds=[0.85],
        reward_scheme=ParetoCrowdingDistance()
    )

    # 4. Explorer fixující synton a rozvíjející pouze volné exit vektory
    explorer = FragSequenceExplorer(
        agent=agent,
        mutate=None,
        env=env,
        epsilon=0.20,
        n_samples=500,
        batch_size=64
    )
    return explorer
`
    }
  ],

  // =========================================================================
  // SECTION 3: TROUBLESHOOTING DECISION TREE
  // =========================================================================
  troubleshooting: [
    {
      problem: "Desired Ratio zůstává na 0.0 % a neroste po 10 epochách",
      cause: "Příliš přísný práh u některého skórovače (např. ROCS práh nastaven na 1.40 namísto 0.871), nebo nulový gradient u skórovače.",
      solution: "1. Spusťte `threshold_analysis.py` a zkontrolujte Youdenův index.\n2. Zkontrolujte, zda všechny skórovače používají `SmoothClippedScore` (hladký gradient) namísto skokového `ClippedScore`.\n3. Zvyšte `epsilon` z 0.1 na 0.25 pro podporu explorace."
    },
    {
      problem: "Lipofilní kolaps (LogP roste nad 6.0, molekuly jsou 'mastné vosky')",
      cause: "Optimalizujete pouze afinitu nebo 3D tvar bez záporné regulace velikosti a hydrofobicity.",
      solution: "Přidejte do `DrugExEnvironment` skórovač `Property('logP', modifier=MinMaxGaussian(mu=3.0, sigma=1.2, minimize=True))` a `Property('MW', modifier=Gaussian(mu=420.0, sigma=60.0)).`"
    },
    {
      problem: "Mode Collapse (Generátor dokola navrhuje 5 stejných molekul, duplicita > 50 %)",
      cause: "Agent uvízl v lokálním optimu; chybí mutační tlak nebo je použita vážená suma (Weighted Sum).",
      solution: "1. Přepněte `reward_scheme` na `ParetoCrowdingDistance()` (odměňuje izolované molekuly na frontě).\n2. Zvyšte `epsilon` na 0.25–0.30.\n3. Zkontrolujte, zda je síť `mutate` správně zmrazena (`mutate.eval()`)."
    },
    {
      problem: "Generátor navrhuje synteticky nemožné molekuly (plné spiro-cyklů a můstků)",
      cause: "Chybí penalizace syntetické dostupnosti.",
      solution: "Přidejte `Property('SA', modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))` s prahem 0.5 do prostředí, nebo přejděte na fragmentový `FragSequenceExplorer` s pravidly BRICS."
    },
    {
      problem: "Zamrznutí konformačního generátoru (Conformer Embedding Bottleneck na makrocyklech)",
      cause: "Distanční geometrie v RDKit ETKDGv3 nedokáže u rigidních či makrocyklických scaffoldů najít konvergující 3D souřadnice a zasekne CPU worker.",
      solution: "1. Nastavte striktní timeout a `max_attempts=15` v `RDKitConformerGenerator`.\n2. Při selhání konformačního vnoření přiřaďte fallback skóre 0.0, čímž agent strukturu okamžitě opustí.\n3. Zkontrolujte strop počtu rotovatelných vazeb (`max_rotatable_bonds=15`)."
    },
    {
      problem: "Reward Hacking: Falešně pozitivní molekuly & PAINS reaktivní farmakofory",
      cause: "Optimalizace 3D tvaru odměňuje velké planární plochy (chinony, polykondenzované aromatické kruhy), které v testech vykazují nespecifickou vazbu.",
      solution: "1. Do vyhodnocovacího workflow zapojte `FilterCatalog` s pravidly PAINS a Brenk (viz Lekce 5.3).\n2. Zaveďte horní ořezání pro počet aromatických kruhů (`Property('NumAromaticRings', modifier=SmoothClippedScore(lower_x=4, upper_x=2, high_score=1.0, low_score=0.0))`)."
    },
    {
      problem: "Pád výpočtu při n_jobs > 1 (CPU Thrashing / Freeze)",
      cause: "V generátoru konformací je nastaveno `num_threads=0` nebo `num_threads > 1` uvnitř multiprocessing workerů.",
      solution: "Nastavte striktně `RDKitConformerGenerator(num_threads=1)` a před spuštěním skriptu exportujte `export OMP_NUM_THREADS=1`."
    }
  ]
};
