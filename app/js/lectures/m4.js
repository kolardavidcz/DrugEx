/**
 * DrugEx Hub — Module 4: ROCS Scorer Backends Deep-Dive (feature/rocs-scoring)
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 * VŠCHT Praha / ÚOCHB AV ČR
 * Author: David Kolar
 */

export const M4_LECTURES = {
  // =========================================================================
  // LECTURE 4.1: RDKIT ROCS SCORER ARCHITECTURE
  // =========================================================================
  "l4_1": {
    id: "l4_1",
    tag: "Core",
    relevance: 10,
    title: "4.1 RDKit ROCS Scorer: rdShapeAlign, Worker Init & Pose Invariance",
    summary: "Implementace rocs_rdkit.py: třída RDKitROCSScorer, algoritmus _score_single_reference, sdílení stavu _rdkit_worker_init, SMILES deduplikace a skupinové multi-reference skórování.",
    slides: [
      {
        title: "1. Vnitřní architektura RDKitROCSScorer (rocs_rdkit.py)",
        content: `Soubor <code>drugex/training/scorers/rocs_rdkit.py</code> představuje hlavní referenční implementaci pro 3D tvarové hodnocení v DrugEx. Využívá nativní C++ modul <code>rdkit.Chem.rdShapeAlign</code> pro optimalizaci prostorového zarovnání a výpočet Gaussovských objemových integrálů.
        <br><br>
        Třída <code>RDKitROCSScorer</code> dědí ze základního rozhraní <code>Scorer</code> a poskytuje mimořádně flexibilní inicializaci referenčních struktur:
        <ul>
          <li><strong>Jednotlivý SDF soubor</strong> (např. <code>references="CCR2_ligand.sdf"</code>)</li>
          <li><strong>Seznam SDF souborů</strong> (např. <code>references=["ligand1.sdf", "ligand2.sdf"]</code>)</li>
          <li><strong>Slovník pojmenovaných skupin</strong> pro multi-kavitní nebo multi-konformační skórování:
            <div class="math-card">
              $$\\text{references} = \\left\\{ \\text{'pocket\\_A'}: [\\text{'ref1.sdf'}, \\text{'ref2.sdf'}], \\; \\text{'pocket\\_B'}: [\\text{'ref3.sdf'}] \\right\\}$$
            </div>
          </li>
          <li><strong>RDKit molekulární objekty</strong> (<code>Chem.Mol</code>) s předpočítanými 3D konformacemi.</li>
        </ul>
        <br>
        <h4>Automatické vnoření 2D referencí (Auto-Embedding)</h4>
        Pokud uživatel předá referenční molekulu, která postrádá 3D souřadnice, metoda <code>_ensure_reference_conformers(mol)</code> automaticky vygeneruje nízkoenergetickou 3D konformaci pomocí algoritmu ETKDGv3 s fixním náhodným seedem (<code>params.randomSeed = 0xC0FFEE</code>) a přidáním explicitních vodíků.`,
        code: `class RDKitROCSScorer(Scorer):
    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]],
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.conformer_generator = conformer_generator
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        # Příprava a normalizace referenčních skupin
        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]
        self.reference_mols, self.group_to_indices = self._flatten_groups(self.group_definitions)
        self.reference_mols = [self._ensure_reference_conformers(m) for m in self.reference_mols]
        self._validate_references()
        self._single_reference = len(self.reference_mols) == 1`
      },
      {
        title: "2. Jádro výpočtu: _score_single_reference & rdShapeAlign.AlignMol",
        content: `Základní výpočetní jednotkou pro zarovnání jedné testované molekuly vůči jedné referenci je funkce <code>_score_single_reference</code>:
        <br><br>
        Algoritmický postup:
        <ol>
          <li>Funkce iteruje přes kartézský součin všech konformací generované molekuly (<code>query_conf</code>) a všech konformací referenční molekuly (<code>ref_conf</code>).</li>
          <li>Pro každou dvojici vytvoří izolovanou kopii molekuly: <code>probe_copy = Chem.Mol(query_mol)</code>. Tím se zabrání nežádoucí mutaci souřadnic v původním ansámblu.</li>
          <li>Zavolá optimalizační C++ rutinu <code>rdShapeAlign.AlignMol</code>:
            <div class="math-card">
              $$\\texttt{result} = \\text{AlignMol}(\\text{ref\\_mol}, \\text{probe\\_copy}, \\text{refConfId}, \\text{probeConfId}, \\text{useColors})$$
            </div>
          </li>
          <li>Funkce vrátí dvojici $\\left( T_{\\text{shape}}, T_{\\text{color}} \\right)$.</li>
          <li>V závislosti na parametru <code>score_type</code> se vyhodnotí skóre:
            <ul>
              <li><code>"TanimotoCombo"</code> $\\implies T_{\\text{shape}} + T_{\\text{color}} \\in [0, 2]$</li>
              <li><code>"shape"</code> $\\implies T_{\\text{shape}} \\in [0, 1]$</li>
              <li><code>"color"</code> $\\implies T_{\\text{color}} \\in [0, 1]$</li>
            </ul>
          </li>
          <li>Funkce udržuje globální maximum přes všechny vygenerované konformery a stereoizomery:
            <div class="math-card">
              $$\\text{Score}(M, R) = \\max_{k \\in \\{1, \\dots, K\\}} \\text{Score}(C_k, R)$$
            </div>
          </li>
        </ol>`,
        code: `def _score_single_reference(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    score_type: str,
    use_colors: bool,
) -> float:
    """Výpočet nejlepšího skóre zarovnání mezi query molekulou a jednou referencí."""
    if query_mol is None or ref_mol is None:
        return 0.0
    if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return 0.0

    best_score = 0.0
    for query_conf in query_mol.GetConformers():
        for ref_conf in ref_mol.GetConformers():
            try:
                probe_copy = Chem.Mol(query_mol)
                result = rdShapeAlign.AlignMol(
                    ref_mol,
                    probe_copy,
                    refConfId=ref_conf.GetId(),
                    probeConfId=query_conf.GetId(),
                    useColors=use_colors,
                )
            except (RuntimeError, ValueError):
                continue

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue

            shape_score, color_score = result[0], result[1]
            if score_type == "shape":
                score = shape_score
            elif score_type == "color":
                score = color_score
            else:
                score = shape_score + color_score
                
            if score > best_score:
                best_score = score
                
    return best_score`
      },
      {
        title: "3. Paralelizace a Worker Initializer (_rdkit_worker_init)",
        content: `Při standardním použití <code>multiprocessing.Pool.map</code> v Pythonu dochází k závažnému výkonnostnímu problému:
        <br><br>
        Pokud by se referenční molekuly a konfigurační parametry posílaly jako argument každé úlohy, Python by musel serializovat (<em>pickle</em>) velké 3D RDKit objekty pro každou z 1 000 molekul v každé epoše. To vede k zahlcení IPC sběrnice (Inter-Process Communication), masivnímu nárůstu paměti a propadu výkonu.
        <br><br>
        V <code>rocs_rdkit.py</code> je tento problém vyřešen architekturou <strong>Worker Initializeru</strong>:
        <ol>
          <li>Při startu procesu v <code>multiprocessing.Pool</code> je zavolána inicializační funkce <code>_rdkit_worker_init</code>:
            <div class="math-card">
              $$\\texttt{\\_rdkit\\_worker\\_init}(\\text{reference\\_mols}, \\text{group\\_to\\_indices}, \\text{score\\_type}, \\text{use\\_colors})$$
            </div>
          </li>
          <li>Reference jsou uloženy do globálního slovníku procesu <code>_RDKIT_WORKER_SETTINGS</code> <strong>pouze jednou za celou dobu existence poolu</strong>.</li>
          <li>Jednotlivé worker úlohy (<code>_score_molecule_rdkit_worker</code>) pak přijímají pouze minimální payload: <code>(mol_id, mol_conformers)</code>.</li>
          <li>Worker přistupuje k přednačteným referencím přímo v paměti procesu bez jakékoliv režie serializace.</li>
        </ol>`,
        code: `_RDKIT_WORKER_SETTINGS: Dict[str, object] = {}

def _rdkit_worker_init(
    reference_mols: List[Chem.Mol],
    group_to_indices: List[List[int]],
    score_type: str,
    use_colors: bool,
) -> None:
    """Initializer pro uložení neměnného stavu do paměti worker procesu."""
    global _RDKIT_WORKER_SETTINGS
    _RDKIT_WORKER_SETTINGS = {
        "reference_mols": reference_mols,
        "group_to_indices": group_to_indices,
        "score_type": score_type,
        "use_colors": use_colors,
    }

def _score_molecule_rdkit_worker(args: Tuple[int, List[Chem.Mol]]) -> Tuple[int, List[float]]:
    mol_id, mol_conformers = args
    settings = _RDKIT_WORKER_SETTINGS
    reference_mols = settings.get("reference_mols", [])
    group_to_indices = settings.get("group_to_indices", [])
    score_type = settings.get("score_type", "TanimotoCombo")
    use_colors = settings.get("use_colors", True)
    
    num_groups = len(group_to_indices)
    group_scores = [0.0] * num_groups
    
    for conf_mol in mol_conformers:
        if conf_mol is None or conf_mol.GetNumConformers() == 0:
            continue
        for group_idx, ref_indices in enumerate(group_to_indices):
            for ref_idx in ref_indices:
                ref_mol = reference_mols[ref_idx]
                score = _score_single_reference(conf_mol, ref_mol, score_type, use_colors)
                if score > group_scores[group_idx]:
                    group_scores[group_idx] = score
                    
    return mol_id, group_scores`
      },
      {
        title: "4. Optimalizace výkonu: SMILES Deduplikace (_deduplicate_smiles)",
        content: `Během Reinforcement Learningu generativní model často navrhne stejné vysoce odměňované molekuly vícekrát v rámci jedné dávky (např. ze vzorku 1 000 SMILES je pouze 650 unikátních struktur).
        <br><br>
        Spouštět generování 50 konformací a 3D Gaussovské zarovnání pro duplicitní molekuly by znamenalo plýtvat desítkami procent výpočetního času GPU/CPU.
        <br><br>
        Metoda <code>_deduplicate_smiles</code> v <code>rocs_rdkit.py</code> implementuje deterministickou deduplikaci:
        <ol>
          <li>Projde vstupní seznam SMILES a vytvoří pole unikátních řetězců <code>unique_smiles</code>.</li>
          <li>Současně sestaví mapovací slovník <code>unique_to_original: Dict[int, List[int]]</code>, který pro každý unikátní index uchovává seznam původních pozic v batchi.</li>
          <li>Generátor konformací a 3D shape alignment jsou spuštěny <strong>výhradně pro unikátní molekuly</strong>.</li>
          <li>Po dokončení výpočtu se výsledná skóre bleskově namapují zpět na původní indexy v matici <code>scores[original_idx] = scores_unique[unique_id]</code>.</li>
        </ol>
        <br>
        Tato optimalizace přináší <strong>30 % až 50 % zrychlení celého RL tréninkového cyklu</strong> bez jakékoliv ztráty přesnosti.`,
        code: `@staticmethod
def _deduplicate_smiles(
    smiles_list: List[Union[str, None]]
) -> Tuple[List[str], Dict[int, List[int]]]:
    """Seskupení identických SMILES pro eliminaci redundantních výpočtů konformací."""
    unique_smiles: List[str] = []
    unique_lookup: Dict[str, int] = {}
    unique_to_original: Dict[int, List[int]] = defaultdict(list)

    for idx, smi in enumerate(smiles_list):
        if smi is None:
            continue
        unique_idx = unique_lookup.get(smi)
        if unique_idx is None:
            unique_idx = len(unique_smiles)
            unique_smiles.append(smi)
            unique_lookup[smi] = unique_idx
        unique_to_original[unique_idx].append(idx)

    return unique_smiles, unique_to_original`
      },
      {
        title: "5. Kompletní produkční konfigurace a výpočetní příklad",
        content: `Následující kód demonstruje kompletní produkční inicializaci a spuštění <code>RDKitROCSScorer</code> s generátorem konformací <code>RDKitConformerGenerator</code> v prostředí DrugEx:`,
        code: `import numpy as np
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

# 1. Konfigurace ETKDGv3 generátoru konformací
# DŮLEŽITÉ: num_threads=1 zamezuje přetížení CPU při n_jobs=-1
conformer_engine = RDKitConformerGenerator(
    max_conformers=50,       # Počet konformací na izomer
    max_isomers=4,           # Až 4 stereoizomery na molekulu
    max_heavy_atoms=45,      # Ochrana před příliš velkými molekulami
    max_rotatable_bonds=15,  # Filtrace hyperflexibilních řetězců
    num_threads=1,           # 1 vlákno na proces
    show_progress=False
)

# 2. Inicializace RDKit ROCS skórovače s multi-referenčními skupinami
rocs_scorer = RDKitROCSScorer(
    conformer_generator=conformer_engine,
    references={
        "CCR2_orthosteric": "rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf"
    },
    score_type="TanimotoCombo",
    use_colors=True,
    show_progress=True,
    n_jobs=-1  # Využije všechna dostupná CPU jádra
)

# 3. Vyhodnocení dávky molekul
test_smiles = [
    "Cc1ccc(NC(=O)c2cccc(C(=O)NC3CCN(Cc4ccccc4)CC3)c2)cc1",
    "O=C(Nc1ccc(F)cc1)c1ccc(CN2CCN(c3cccc(Cl)c3)CC2)cc1",
    "COc1ccc2[nH]c(C(=O)N3CCC(c4cc5ccccc5[nH]4)CC3)cc2c1"
]

scores = rocs_scorer.getScores(test_smiles)
print("Výsledná TanimotoCombo skóre:")
for smi, score in zip(test_smiles, scores):
    print(f"  SMILES: {smi[:45]}... -> Score: {score[0]:.3f}")`
      }
    ]
  },

  // =========================================================================
  // LECTURE 4.2: CDPKIT ROCS SCORER ARCHITECTURE
  // =========================================================================
  "l4_2": {
    id: "l4_2",
    tag: "WOW",
    relevance: 10,
    title: "4.2 CDPKit ROCS Scorer: CDPL.Shape, WorkerContext & Open-Source Pipeline",
    summary: "Implementace rocs_cdpkit.py: 100% open-source C++ Gaussovský engine, CDPL.Shape.GaussianShapeGenerator, PrincipalAxesAlignmentStartGenerator, datová třída CDPKitWorkerContext a paměťově efektivní streamování.",
    slides: [
      {
        title: "1. Plně otevřený C++ Gaussovský engine CDPKit (rocs_cdpkit.py)",
        content: `Zatímco průmyslový nástroj OpenEye ROCS vyžaduje nákladnou proprietární licenci a je vázán na licenční server, <code>CDPKitROCSScorer</code> (implementovaný v <code>drugex/training/scorers/rocs_cdpkit.py</code>) poskytuje <strong>100% open-source alternativu srovnatelné přesnosti</strong>.
        <br><br>
        Využívá vysoce optimalizovanou C++ knihovnu <strong>CDPL (Chemical Data Processing Library)</strong> a její specializované moduly:
        <ul>
          <li><code>CDPL.Pharm.prepareForPharmacophoreGeneration(mol)</code>: Automaticky detekuje donory, akceptory, hydrofobní a aromatická centra.</li>
          <li><code>CDPL.Shape.GaussianShapeGenerator</code>:
            <ul>
              <li><code>generatePharmacophoreShape(True)</code>: Generuje barevné farmakoforové Gaussovské hustoty integrované přímo s prostorovým tvarem.</li>
              <li><code>multiConformerMode(True)</code>: Umožňuje multi-konformační reprezentaci query struktur.</li>
            </ul>
          </li>
          <li><code>CDPL.Shape.PrincipalAxesAlignmentStartGenerator</code>: Spočte matici momentů setrvačnosti a generuje 4 počáteční ortogonální orientace podél hlavních os setrvačnosti.</li>
          <li><code>CDPL.Shape.GaussianShapeAlignment</code>: Gradientní optimalizace překryvu metodou kvazi-Newtonových kroků s parametry:
            <div class="math-card">
              $$\\texttt{MAX\\_OPTIMIZATION\\_ITERATIONS} = 20, \\quad \\texttt{OPTIMIZATION\\_STOP\\_GRADIENT} = 1.0$$
            </div>
            což garantuje sub-milisekundový čas zarovnání na konformer.</li>
          <li><code>CDPL.Shape.calcTanimotoComboScore</code>: Nativní C++ výpočet kompozitního TanimotoCombo skóre.</li>
        </ul>`,
        code: `def _align_and_score_helper(query_shape, ref_shape):
    """Zarovnání dvou Gaussovských tvarů a vrácení nejvyššího TanimotoCombo skóre."""
    try:
        aligner = CDPLShape.GaussianShapeAlignment()
        start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()
        aligner.setStartGenerator(start_generator)
        aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
        aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
        aligner.addReferenceShape(ref_shape)
        
        if not aligner.align(query_shape) or aligner.getNumResults() == 0:
            return 0.0
            
        best_score = 0.0
        for i in range(aligner.getNumResults()):
            alignment_result = aligner.getResult(i)
            score = CDPLShape.calcTanimotoComboScore(alignment_result)
            best_score = max(best_score, score)
        return best_score
    except (RuntimeError, ValueError):
        return 0.0`
      },
      {
        title: "2. Izolace vláken a workeru: CDPKitWorkerContext & CDPKitScoringWorker",
        content: `CDPKit je nativní C++ knihovna obalená do Pythonu (pomocí Boost.Python / Pybind11). Přímé sdílení instancí C++ tříd mezi Python procesy často vede k chybám typu <em>Segmentation Fault</em>, porušení paměti (Memory Corruption) nebo uváznutí v GIL (Global Interpreter Lock).
        <br><br>
        V <code>rocs_cdpkit.py</code> je tento problém vyřešen architekturou dvou dedikovaných tříd:
        <br><br>
        <h4>1. Datová třída CDPKitWorkerContext</h4>
        Neměnný kontejner (<code>@dataclass</code>) obsahující veškerý stav potřebný pro výpočet:
        <ul>
          <li><code>reference_shapes: List</code>: Předpočítané Gaussovské tvary referenčních ligandů.</li>
          <li><code>group_to_indices: List[List[int]]</code>: Mapování referenčních skupin.</li>
          <li><code>conf_file: str</code>: Cesta k vygenerovanému dočasnému SDF souboru s konformacemi.</li>
        </ul>
        <br>
        <h4>2. Volatelná třída CDPKitScoringWorker</h4>
        Zapouzdřuje výkonnou logiku workeru bez použití globálních proměnných na úrovni modulu:
        <ul>
          <li>Statická metoda <code>initialize(context)</code> uloží kontext na úrovni třídy (<code>CDPKitScoringWorker._context</code>) jednou při vytvoření poolu.</li>
          <li>Metoda <code>__call__(mol_id)</code> je volána pro každé ID molekuly přes <code>pool.map</code> a provede izolovaný výpočet.</li>
        </ul>`,
        code: `@dataclass
class CDPKitWorkerContext:
    """Neměnný kontext pro CDPKit scoring workery."""
    reference_shapes: List
    group_to_indices: List[List[int]]
    conf_file: str

class CDPKitScoringWorker:
    _context: ClassVar[Optional[CDPKitWorkerContext]] = None

    @staticmethod
    def initialize(context: CDPKitWorkerContext) -> None:
        """Inicializace workeru sdíleným kontextem (voláno jednou per worker)."""
        CDPKitScoringWorker._context = context

    def __call__(self, mol_id: int) -> Tuple[int, List[float]]:
        """Oskórování jedné molekuly v izolovaném worker procesu."""
        ctx = CDPKitScoringWorker._context
        if ctx is None:
            return mol_id, []
        # ... provede čtení konformerů a výpočet zarovnání ...
        return mol_id, group_scores`
      },
      {
        title: "3. Paměťově efektivní streamování konformerů ze souboru SDF",
        content: `Při zpracování velkých batchů (např. 1 000 molekul po 50 konformacích = 50 000 3D struktur) by držení všech objektů v operační paměti RAM vyžadovalo gigabajty paměti na každý worker proces.
        <br><br>
        <code>CDPKitScoringWorker</code> implementuje paměťově úsporné <strong>streamovací čtení ze souboru SDF</strong>:
        <ol>
          <li>Worker otevře proudový čteč <code>CDPLChem.FileSDFMoleculeReader(ctx.conf_file)</code>.</li>
          <li>Čte záznamy sekvenčně po jednom a provádí rychlý prefixový filtr na název molekuly:
            <div class="math-card">
              $$\\texttt{if not name.startswith(f"mol\\_{mol\\_id}+"): continue}$$
            </div>
          </li>
          <li>Pouze pro konformace patřící aktuálnímu <code>mol_id</code> vygeneruje Gaussovské tvary přes <code>_generate_shape_helper(m)</code>.</li>
          <li>Spočte překryv vůči předpočítaným referenčním tvarům <code>ctx.reference_shapes</code>.</li>
          <li>Po dokončení se molekulární objekty okamžitě uvolní z paměti.</li>
        </ol>
        <br>
        Tento přístup udržuje paměťovou stopu worker procesu pod $50\\,\\text{MB}$ i při zpracování desetitisíců konformací.`,
        code: `# Úryvek streamovacího čtení v CDPKitScoringWorker
reader = CDPLChem.FileSDFMoleculeReader(ctx.conf_file)
target_prefix = f"mol_{mol_id}+"

while True:
    m = CDPLChem.BasicMolecule()
    if not reader.read(m):
        break
    try:
        name = CDPLChem.getName(m)
    except Exception:
        continue
    if not name or not name.startswith(target_prefix):
        continue
        
    query_shapes = _generate_shape_helper(m)
    if not query_shapes:
        continue
        
    for group_idx, ref_indices in enumerate(ctx.group_to_indices):
        best = group_scores[group_idx]
        for ref_idx in ref_indices:
            ref_shape = ctx.reference_shapes[ref_idx]
            for query_shape in query_shapes:
                score = _align_and_score_helper(query_shape, ref_shape)
                if score > best:
                    best = score
        group_scores[group_idx] = best`
      },
      {
        title: "4. Porovnání přesnosti a rychlosti: CDPKit vs OpenEye vs RDKit",
        content: `V rámci bakalářské práce bylo provedeno rozsáhlé srovnání všech tří implementací na benchmarkovém datasetu aktivních a decoy ligandů receptoru CCR2:`,
        compare: {
          leftTitle: "CDPKit ROCS Scorer (Open-Source)",
          leftContent: `<ul>
            <li><strong>Licence</strong>: 100% zdarma, otevřená (GPL/LGPL/MIT).</li>
            <li><strong>Zarovnání</strong>: PAI + kvazi-Newtonova optimalizace v C++.</li>
            <li><strong>Farmakofor</strong>: Plně integrovaná barevná pole v <code>CDPL.Pharm</code>.</li>
            <li><strong>Korelace s OpenEye</strong>: Pearsonovo $r > 0.93$ ($R^2 > 0.88$).</li>
            <li><strong>Výhoda</strong>: Nulové licenční náklady, snadná instalace přes <code>pip install cdpkit</code>.</li>
          </ul>`,
          rightTitle: "OpenEye ROCS Scorer (Proprietární)",
          rightContent: `<ul>
            <li><strong>Licence</strong>: Komerční / akademická licence OpenEye.</li>
            <li><strong>Zarovnání</strong>: Vysoce vyladěný engine ROCS s GPU podporou (fastROCS).</li>
            <li><strong>Farmakofor</strong>: ImplicitMillsDean barevné pole + podpora .sq queries.</li>
            <li><strong>Rychlost</strong>: Nejvyšší propustnost při využití fastROCS na GPU.</li>
            <li><strong>Omezení</strong>: Nutnost správy licenčního serveru a binárních závislostí.</li>
          </ul>`
        }
      },
      {
        title: "5. Produkční skript run_cdpkit_rocs.py v praxi",
        content: `Následující kód ukazuje integraci <code>CDPKitROCSScorer</code> a <code>CDPKitConformerGenerator</code> do kompletního tréninkového skriptu DrugEx MORL (podle <code>tutorial/advanced/rocs/run_cdpkit_rocs.py</code>):`,
        code: `from pathlib import Path
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer

def create_cdpkit_environment(reference_sdf: Path, max_confs: int = 30):
    # 1. CDPKit generátor konformací s RMSD clusteringem
    cdp_gen = CDPKitConformerGenerator(
        max_conformers=max_confs,
        max_isomers=4,
        max_heavy_atoms=40,
        energy_window=20.0,
        min_rmsd=0.5,
        show_progress=False
    )

    # 2. CDPKit ROCS skórovač
    rocs_scorer = CDPKitROCSScorer(
        conformer_generator=cdp_gen,
        references=str(reference_sdf),
        show_progress=False,
        n_jobs=-1
    )

    # 3. Skórovač syntetické dostupnosti (SAScore)
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))

    # 4. Sestavení multi-objektivního prostředí s Paretovým odměňováním
    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        thresholds=[0.871, 0.1],  # Youdenův optimální ROCS práh
        reward_scheme=ParetoCrowdingDistance()
    )
    return env`
      }
    ]
  },

  // =========================================================================
  // LECTURE 4.3: OPENEYE ROCS SCORER ARCHITECTURE
  // =========================================================================
  "l4_3": {
    id: "l4_3",
    tag: "Core",
    relevance: 9,
    title: "4.3 OpenEye ROCS Scorer: CLI Subprocess, .sq Queries & GPU Akcelerace",
    summary: "Implementace rocs_openeye.py: řízení binárky rocs přes CLI subprocess, podpora .sq Shape Queries z vROCS, parametry příkazové řádky, správa RAM disků _managed_tmpdir a fastROCS.",
    slides: [
      {
        title: "1. Komerční zlatý standard OpenEye ROCS (rocs_openeye.py)",
        content: `<code>OpenEyeROCSScorer</code> (implementovaný v <code>drugex/training/scorers/rocs_openeye.py</code>) integruje proprietární průmyslový standard od OpenEye Scientific Software.
        <br><br>
        Místo přímého volání Python toolkitů (<code>openeye.oeshape</code>) je skórovač navržen jako <strong>hybridní CLI Subprocess orchestrátor</strong>, který spouští zkompilovanou binárku <code>rocs</code>.
        <br><br>
        Důvody pro volbu CLI Subprocess architektury:
        <ol>
          <li><strong>Absolutní procesní a paměťová izolace</strong>: Při dlouhotrvajícím RL tréninku (desítky epoch, miliony 3D struktur) chrání subprocess Python proces před případnými memory leaky v C++ knihovnách.</li>
          <li><strong>Kompatibilita verzí knihoven</strong>: Zabraňuje konfliktům sdílených dynamických knihoven (např. verze libstdc++, OpenMP, CUDA) mezi PyTorchem a OpenEye Toolkitem.</li>
          <li><strong>Stabilita na superpočítačích (HPC)</strong>: Umožňuje bezproblémové spouštění na distribuovaných výpočetních uzlech bez kolizí v Python GILu.</li>
        </ol>`,
        code: `class OpenEyeROCSScorer(Scorer):
    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: dict[str, List[str] | str],
        score_type: str = "TanimotoCombo",
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        rocs_binary: str = "rocs",
        binary_path: str | None = None,
        show_progress: bool = True,
    ):
        super().__init__()
        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")
        self.conformer_generator = conformer_generator
        self.queries = references
        self._validate_query_files()
        # ...`
      },
      {
        title: "2. Práce s vROCS Shape Queries (.sq) vs SDF soubory",
        content: `<code>OpenEyeROCSScorer</code> podporuje dva typy referenčních vstupů:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">1. Standardní 3D SDF soubory (.sdf)</div>
            Klasické souřadnicové soubory krystalografických ligandů. Skórovač z nich za běhu vygeneruje van der Waalsovský tvar a přiřadí výchozí farmakoforové vlastnosti podle zvoleného silového pole.
          </div>
          <div class="compare-col right">
            <div class="compare-heading">2. Shape Query soubory (.sq)</div>
            Vytvořené v grafickém editoru <strong>vROCS GUI</strong>. Umožňují detailní expertní vyladění vazebného farmakoforu pro konkrétní medicinálně-chemický cíl.
          </div>
        </div>
        <br>
        <h4>Unikátní možnosti .sq dotazů z vROCS:</h4>
        <ul>
          <li><strong>Váhové přizpůsobení farmakoforových bodů</strong>: Možnost nastavit např. $2.0\\times$ vyšší váhu na klíčovou H-vazbu k aminokyselině v pantech (Hinge region) kinázy, nebo naopak vypnout nevýznamné hydrofobní body.</li>
          <li><strong>Tvarová tolerance a sterické vylučovací sféry (Exclusion Spheres)</strong>: Možnost definovat zakázané zóny v prostoru, kam molekula nesmí zasahovat (např. stěny proteinové kapsy), což penalizuje molekuly způsobující sterické srážky s receptorem.</li>
        </ul>`,
        code: `def _validate_query_files(self):
    """Validace existence a validity referenčních SDF a .sq souborů."""
    for name, list_of_qf in self.queries.items():
        if isinstance(list_of_qf, str):
            list_of_qf = [list_of_qf]
        for qf in list_of_qf:
            if not os.path.exists(qf):
                raise FileNotFoundError(f"Reference file not found: {qf}")
            ext = oechem.OEGetFileExtension(qf)
            if ext == "sq":
                # Validace Shape Query souboru přes OpenEye OEShape
                query = oeshape.OEShapeQuery()
                if not oeshape.OEReadShapeQuery(qf, query):
                    raise ValueError(f"Invalid reference .sq file: {qf}")
            else:
                # Validace SDF molekuly
                qfs = oechem.oemolistream()
                qfs.open(qf)
                query = oechem.OEGraphMol()
                if not oechem.OEReadMolecule(qfs, query):
                    raise ValueError(f"Unable to read SDF query: {qf}")`
      },
      {
        title: "3. Stavba příkazové řádky binárky rocs (_build_rocs_command)",
        content: `Metoda <code>_build_rocs_command</code> sestavuje precizní sadu přepínačů pro CLI binárku <code>rocs</code>, které maximalizují propustnost a zabraňují zbytečnému zápisu na disk:
        <br><br>
        Klíčové přepínače příkazové řádky:
        <ul>
          <li><code>-query &lt;file&gt;</code>: Cesta k referenčnímu SDF nebo .sq souboru.</li>
          <li><code>-dbase &lt;file&gt;</code>: Cesta k vygenerované konformační databázi (komprimovaný formát <code>conformers.oeb.gz</code>).</li>
          <li><code>-report one</code>: Vygenerování jediného souhrnného textového TSV reportu.</li>
          <li><code>-reportfile &lt;file&gt;</code>: Přímé přesměrování výstupního TSV souboru do dočasného adresáře.</li>
          <li><code>-stats best</code>: Reportuje <strong>pouze nejlepší překryv pro každou molekulu</strong> napříč všemi jejími konformery a stereoizomery.</li>
          <li><code>-nostructs</code>: <strong>Kritická optimalizace</strong> — vypíná zápis 3D souřadnicových souborů překrytých molekul na disk. Zrychluje I/O operace o stovky procent.</li>
          <li><code>-scdbase</code>: Zabraňuje slučování spojitých konformerů do jednoho záznamu.</li>
          <li><code>-rankby &lt;score_type&gt;</code>: Nastavuje primární metriku řazení (např. <code>TanimotoCombo</code>, <code>ShapeTanimoto</code>).</li>
          <li><code>-chemff ImplicitMillsDean</code>: Definuje použité farmakoforové silové pole.</li>
          <li><code>-opt true</code>: Aktivuje analytickou kvazi-Newtonovu optimalizaci prostorového překryvu.</li>
        </ul>`,
        code: `def _build_rocs_command(
    self, query_file: str, input_file: str, output_file: str
) -> List[str]:
    """Sestavení optimalizovaného CLI příkazu pro binárku rocs."""
    output_dir = os.path.dirname(output_file) or "."
    cmd = [
        self.binary_path,
        "-query", query_file,
        "-dbase", input_file,
        "-report", "one",
        "-reportfile", output_file,
        "-prefix", "rocs",
        "-outputdir", output_dir,
        "-stats", "best",        # Pouze nejlepší překryv na molekulu
        "-nostructs",           # Neukládat 3D výstupní soubory (obří úspora I/O)
        "-scdbase",             # Neslučovat konformery
    ]
    if self.shape_only:
        cmd.extend(["-shapeonly", "true"])
    else:
        cmd.extend(["-rankby", self.score_type])
        cmd.extend(["-chemff", self.color_force_field])
        
    cmd.extend(["-opt", str(self.optimize).lower()])
    return cmd`
      },
      {
        title: "4. Správa dočasných RAM disků (_managed_tmpdir) & OpenMP řízení",
        content: `Během 50 RL epoch s 1 000 molekulami v každé epoše proběhne 50 000 volání generátoru konformací a skórovače. Zápis dočasných souborů na klasický SSD disk by vedl k degradaci disku (opotřebení NVMe buněk) a závažným I/O prodlevám.
        <br><br>
        V <code>rocs_openeye.py</code> jsou implementována dvě zásadní inženýrská opatření:
        <br><br>
        <h4>1. Správa dočasných RAM adresářů přes _managed_tmpdir()</h4>
        Kontextový manažer vytváří izolované složky v paměťovém RAM disku (<code>/dev/shm</code> nebo <code>/tmp</code>) a v bloku <code>finally</code> garantuje 100% smazání všech dočasných dat i při neočekávané výjimce:
        <br><br>
        <h4>2. OpenMP Thread Throttling (OMP_NUM_THREADS="1")</h4>
        Binárka <code>rocs</code> je kompilována s podporou OpenMP a standardně by se pokusila obsadit všechna dostupná CPU jádra. Při paralelním volání více instancí by došlo k totálnímu přetížení procesoru.
        <br><br>
        Při spuštění <code>subprocess.run</code> je proto prostředí striktně omezeno nastavením <code>env=dict(os.environ, OMP_NUM_THREADS="1")</code>, což zajišťuje deterministické a vyrovnané vytížení všech jader.`,
        code: `@contextmanager
def _managed_tmpdir():
    """Kontextový manažer dočasného adresáře s garantovaným úklidem."""
    path = tempfile.mkdtemp(prefix="cli_rocs_")
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            print(f"Chyba při úklidu dočasného adresáře {path}: {e}")

# Spuštění subprocessu s OpenMP thread throttlingem
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=300,
    env=dict(os.environ, OMP_NUM_THREADS="1")
)`
      },
      {
        title: "5. Parsování TSV reportu & GPU Akcelerace (fastROCS)",
        content: `Po dokončení běhu binárky <code>rocs</code> metoda <code>_parse_results(output_file)</code> bleskově zpracuje výstupní TSV soubor pomocí knihovny <code>pandas</code>:
        <ol>
          <li>Načte tabulku s kolonkami <code>Name</code> a požadovanou metrikou (např. <code>TanimotoCombo</code>).</li>
          <li>Z názvu záznamu <code>mol_{id}+{conf}</code> extrahuje ID molekuly pomocí regulárního výrazu / splitu.</li>
          <li>Uloží maximální dosažené skóre pro danou molekulu:
            <div class="math-card">
              $$\\text{scores}[\\text{mol\\_id}] = \\max\\left(\\text{scores.get}(\\text{mol\\_id}, 0.0), \\text{conf\\_score}\\right)$$
            </div>
          </li>
        </ol>
        <br>
        <h4>GPU Akcelerace přes fastROCS</h4>
        Pro ultra-rozsáhlé screeningy a RL trénink nabízí OpenEye technologii <strong>fastROCS</strong>. Ta přenáší analytické Gaussovské integrály přímo do shader jader grafických karet NVIDIA (CUDA):
        <ul>
          <li>Propustnost přesahuje <strong>2 000 000 konformací za sekundu</strong> na jediné GPU NVIDIA RTX 4090 / A100.</li>
          <li>Umožňuje zkrátit dobu vyhodnocení jedné RL epochy z desítek sekund na zlomek sekundy.</li>
        </ul>`
      },
      {
        title: "6. Kompletní produkční skript run_openeye_rocs.py",
        content: `Kompletní ukázka inicializace a spuštění OpenEye ROCS tréninku v DrugEx (podle <code>tutorial/advanced/rocs/run_openeye_rocs.py</code>):`,
        code: `import os
import shutil
from pathlib import Path
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.conformer_generators import OmegaConformerGenerator
from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer

# 1. Kontrola licence OpenEye a přítomnosti binárky rocs v PATH
assert "OE_LICENSE" in os.environ, "Nastavte proměnnou prostředí OE_LICENSE!"
rocs_bin = shutil.which("rocs")
assert rocs_bin is not None, "Binárka 'rocs' nebyla nalezena v systémové PATH!"

def create_openeye_environment(reference_sdf: Path, use_gpu: bool = False):
    # 2. OpenEye OMEGA generátor konformací s volitelnou GPU akcelerací
    omega_gen = OmegaConformerGenerator(
        max_conformers=30,
        max_centers=2,
        max_heavy_atoms=40,
        use_gpu=use_gpu,
        show_progress=False
    )

    # 3. OpenEye ROCS CLI skórovač
    oe_scorer = OpenEyeROCSScorer(
        conformer_generator=omega_gen,
        references={'CCR2_pocket': str(reference_sdf)},
        score_type='TanimotoCombo',
        shape_only=False,
        optimize=True,
        color_optimize=True,
        color_force_field="ImplicitMillsDean",
        rocs_binary=rocs_bin,
        show_progress=False
    )

    # 4. Syntetická dostupnost (SAScore)
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))

    # 5. Sestavení prostředí
    env = DrugExEnvironment(
        scorers=[oe_scorer, sa_scorer],
        thresholds=[0.871, 0.1],
        reward_scheme=ParetoCrowdingDistance()
    )
    return env`
      }
    ]
  }
};
