/**
 * DrugEx Hub — Module 4: ROCS Scorer Backends Deep-Dive (feature/rocs-scoring)
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M4_LECTURES = {
  // =========================================================================
  // LECTURE 4.1
  // =========================================================================
  "l4_1": {
    id: "l4_1",
    tag: "Core",
    relevance: 10,
    title: "4.1 RDKit ROCS Scorer: rdShapeAlign, Worker Init & Pose Invariance",
    summary: "Implementace rocs_rdkit.py: paralelní multiprocessing bez memory leaků, deduplikace SMILES a skupinové skórování.",
    slides: [
      {
        title: "1. Vnitřní architektura RDKitROCSScorer (rocs_rdkit.py)",
        content: `Soubor <code>drugex/training/scorers/rocs_rdkit.py</code> představuje hlavní referenční implementaci pro 3D tvarové hodnocení v DrugEx. Využívá C++ modul <code>rdkit.Chem.rdShapeAlign</code> pro optimalizaci prostorového zarovnání.
        <br><br>
        Základní výpočetní jednotka <code>_score_single_reference</code> provádí:
        <ol>
          <li>Pro každý vygenerovaný konformer $C_k$ kandidátní molekuly $M$ provede optimální rotaci a translaci vůči referenční molekule $R$.</li>
          <li>Spočte Shape Tanimoto a (při <code>use_colors=True</code>) Color Tanimoto.</li>
          <li>Vrátí nejvyšší dosažené skóre přes všechny konformery a stereoizomery:
            <div class="math-card">
              $$\\text{Score}(M, R) = \max_{k \\in \{1, \\dots, K\}} \\left[ T_{shape}(C_k, R) + T_{color}(C_k, R) \\right]$$
            </div>
          </li>
        </ol>`,
        code: `from rdkit.Chem import rdShapeAlign

def _score_single_reference(probe_mol, ref_mol, score_type, use_colors):
    """
    Jaderná funkce pro skórování jednoho kandidáta vůči jedné referenci.
    """
    best_score = 0.0
    for conf_id in range(probe_mol.GetNumConformers()):
        # rdShapeAlign.AlignMol provede optimální překrytí
        res = rdShapeAlign.AlignMol(
            ref_mol, 
            probe_mol, 
            probeConfId=conf_id, 
            useColors=use_colors
        )
        # res obsahuje (ShapeTanimoto, ColorTanimoto)
        shape_t, color_t = res[0], res[1] if use_colors else 0.0
        
        if score_type == "TanimotoCombo":
            combo = shape_t + color_t
        elif score_type == "shape":
            combo = shape_t
        else:
            combo = color_t
            
        if combo > best_score:
            best_score = combo
            
    return best_score`
      },
      {
        title: "2. Klíčové inženýrské optimalizace v rocs_rdkit.py",
        content: `V RL smyčce se skórují tisíce molekul v každé epoše. Neoptimalizovaný výpočet by trval hodiny na epochu. V kódu jsou proto implementovány dvě zásadní optimalizace:
        <br><br>
        <h4>1. SMILES Deduplikace (<code>_deduplicate_smiles</code>)</h4>
        Během vzorkování z neuronové sítě se často vygenerují identické molekuly. Funkce vytvoří mapu unikátních SMILES:
        <ul>
          <li>Konformace a 3D zarovnání se spočítají pouze pro unikátní struktury (např. 650 z 1000).</li>
          <li>Výsledná skóre se následně namapují zpět na původní indexy v batchi.</li>
        </ul>
        <br>
        <h4>2. Multiprocessing Worker Initializer (<code>_rdkit_worker_init</code>)</h4>
        Při standardním volání <code>multiprocessing.Pool.map</code> v Pythonu dochází k opakované serializaci (pickling) velkých referenčních molekul a nastavení v každém volání.
        <br><br>
        Initializer <code>_rdkit_worker_init</code> odešle referenční struktury do globální paměti worker procesu <strong>pouze jednou při vytvoření poolu</strong>. Do procesů se pak posílají pouze krátké SMILES řetězce, což šetří gigabajty paměťového bandwidthu.`
      },
      {
        title: "3. Kompletní inicializace RDKitROCSScorer v praxi",
        content: `Příklad konfigurace skórovače pro trénování v DrugEx:`,
        code: `from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

# 1. Konfigurace generátoru konformací
conf_gen = RDKitConformerGenerator(
    max_conformers=50,
    max_isomers=4,
    max_heavy_atoms=45,
    max_rotatable_bonds=15,
    num_threads=1
)

# 2. Inicializace RDKit ROCS skórovače
rocs_scorer = RDKitROCSScorer(
    conformer_generator=conf_gen,
    references="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
    score_type="TanimotoCombo",
    use_colors=True,
    show_progress=True,
    n_jobs=-1  # Automatická alokace všech CPU jader
)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 4.2
  // =========================================================================
  "l4_2": {
    id: "l4_2",
    tag: "WOW",
    relevance: 10,
    title: "4.2 CDPKit ROCS Scorer: CDPL.Shape, WorkerContext & Open-Source Pipeline",
    summary: "Implementace rocs_cdpkit.py: čistě open-source Gaussovské zarovnání přes PrincipalAxesAlignment a bezlicenční workflow.",
    slides: [
      {
        title: "1. Plně otevřený C++ Gaussovský engine CDPKit (rocs_cdpkit.py)",
        content: `Zatímco OpenEye ROCS vyžaduje nákladnou komerční akademickou/firemní licenci, <code>CDPKitROCSScorer</code> (v <code>drugex/training/scorers/rocs_cdpkit.py</code>) poskytuje <strong>100% open-source alternativu srovnatelné přesnosti</strong>.
        <br><br>
        Využívá vysoce optimalizovanou knihovnu CDPL (Chemical Data Processing Library) a její modul <code>CDPL.Shape</code>:
        <ul>
          <li><code>GaussianShapeGenerator</code>: Vytváří analytické Gaussovské hustoty pro molekuly.</li>
          <li><code>PrincipalAxesAlignmentStartGenerator</code>: Spočte matici momentů setrvačnosti a generuje 4 počáteční ortogonální orientace podél hlavních os setrvačnosti.</li>
          <li><code>GaussianShapeAlignment</code>: Gradientní optimalizace překryvu metodou kvazi-Newtonových kroků s omezením na maximálně 20 iterací (<code>MAX_OPTIMIZATION_ITERATIONS = 20</code>) pro zachování vysoké rychlosti.</li>
        </ul>`,
        code: `import CDPL.Chem as CDPLChem
import CDPL.Shape as CDPLShape
from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer

# Inicializace CDPKit generátoru a skórovače
cdp_conf_gen = CDPKitConformerGenerator(
    max_conformers=50,
    max_isomers=4,
    timeout=3600
)

cdp_scorer = CDPKitROCSScorer(
    conformer_generator=cdp_conf_gen,
    references="rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
    n_jobs=-1
)`
      },
      {
        title: "2. Zapouzdření kontextu workeru: CDPKitWorkerContext",
        content: `CDPKit je nativní C++ knihovna obalená v Pythonu (pomocí Boost.Python/pybind11). Přímé sdílení C++ objektů mezi Python procesy často vede k chybám typu <em>Segmentation Fault</em>.
        <br><br>
        V <code>rocs_cdpkit.py</code> je tento problém vyřešen datovou třídou <code>CDPKitWorkerContext</code>:
        <ul>
          <li>Při startu worker procesu se vytvoří izolovaný kontext obsahující vlastní instanci <code>GaussianShapeAlignment</code> a C++ referenční molekuly.</li>
          <li>Žádné C++ pointery nejsou sdíleny mezi vlákny, což garantuje 100% stabilitu při dlouhotrvajících RL výpočtech na superpočítači.</li>
        </ul>`
      }
    ]
  },

  // =========================================================================
  // LECTURE 4.3
  // =========================================================================
  "l4_3": {
    id: "l4_3",
    tag: "Core",
    relevance: 9,
    title: "4.3 OpenEye ROCS Scorer: CLI Subprocess, .sq Queries & GPU Akcelerace",
    summary: "Implementace rocs_openeye.py: řízení binárky rocs, načítání VROCS Shape Queries (.sq) a licence OpenEye.",
    slides: [
      {
        title: "1. Komerční zlatý standard OpenEye ROCS (rocs_openeye.py)",
        content: `<code>OpenEyeROCSScorer</code> integruje proprietární průmyslový standard od OpenEye Scientific Software.
        <br><br>
        Podporuje dva typy referencí:
        <ol>
          <li><strong>Standardní 3D SDF struktury</strong> (např. krystalové ligandy).</li>
          <li><strong>Shape Queries (soubory <code>.sq</code>)</strong>: Vytvořené v grafickém editoru vROCS. Umožňují manuálně přidávat, mazat nebo zesilovat váhu konkrétních farmakoforových bodů (Color Features) a nastavovat toleranční poloměry pro jednotlivé domény vazebného místa.</li>
        </ol>`,
        code: `import os
from drugex.training.scorers.conformer_generators import OmegaConformerGenerator
from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer

# Kontrola licenčního souboru
assert "OE_LICENSE" in os.environ, "Nastavte proměnnou prostředí OE_LICENSE!"

omega_gen = OmegaConformerGenerator(max_conformers=50, use_gpu=True)

oe_scorer = OpenEyeROCSScorer(
    conformer_generator=omega_gen,
    references={
        "pocket_main": "rocs_rl_ccr/rdkit_cdpkit/model1.sq",
        "pocket_sub": "rocs_rl_ccr/rdkit_cdpkit/model2.sq"
    },
    score_type="TanimotoCombo",
    color_force_field="ImplicitMillsDean"
)`
      },
      {
        title: "2. Správa dočasných souborů a CLI Subprocess řízení",
        content: `Jelikož OpenEye ROCS běží jako samostatná C++ binárka, <code>rocs_openeye.py</code> implementuje kontextový manažer <code>_managed_tmpdir()</code>:
        <ul>
          <li>Vygenerované konformery jsou zapsány do komprimovaného binárního formátu <code>.oeb.gz</code> v paměťovém RAM disku (<code>/dev/shm</code> nebo <code>/tmp</code>).</li>
          <li>Binárka <code>rocs</code> je spuštěna s přepínačem <code>-nostructs</code> (negeneruje výstupní 3D soubory, pouze textový TSV report).</li>
          <li>Skóre jsou bleskově načteny přes <code>csv.reader</code> a dočasné soubory jsou okamžitě smazány.</li>
        </ul>`
      }
    ]
  }
};
