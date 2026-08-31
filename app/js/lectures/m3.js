/**
 * DrugEx Hub — Module 3: 3D Shape Matching Theory, IDPs & Conformer Engines
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 * VŠCHT Praha / ÚOCHB AV ČR
 * Author: David Kolar
 */

export const M3_LECTURES = {
  // =========================================================================
  // LECTURE 3.1: 3D MOLECULAR SIMILARITY & ROCS THEORY
  // =========================================================================
  "l3_1": {
    id: "l3_1",
    tag: "Core",
    relevance: 10,
    title: "3.1 Fyzikální podstata ROCS: Gaussovské objemy, Shape Tanimoto & Color TanimotoCombo",
    summary: "Grant-Pickupova Gaussovská reprezentace atomové hustoty, analytické uzavřené řešení objemových integrálů V(A,B), farmakoforové barevné shody (ImplicitMillsDean) a kompozitní skóre TanimotoCombo.",
    slides: [
      {
        title: "1. Selhání 2D topologických deskriptorů při 'Scaffold Hoppingu'",
        content: `Klasická chemoinformatika se po desetiletí opírala o 2D topologické otisky (např. Morgan / ECFP4 fingerprinty, Daylight cesty, MACCS klíče). Tyto metody kódují molekulu jako binární vektor přítomnosti podstruktur v topologickém okolí atomů do poloměru $r \\le 2$ vazeb.
        <br><br>
        Tento přístup má fundamentální teoretický limit: <strong>dvě molekuly se zcela odlišnými chemickými jádry (scaffoldy) mají 2D Tanimoto podobnost blízkou nule ($T_{2D} < 0.20$)</strong>, i když v trojrozměrném prostoru zaujímají <strong>identický objem a stejné prostorové uspořádání klíčových funkčních skupin</strong>.
        <br><br>
        Biologický cíl (receptor, enzym či proteinové rozhraní) nečte 2D graf ani SMILES řetězec. Vazebné rozhraní proteinu interaguje výhradně s:
        <ul>
          <li><strong>Trojrozměrnou van der Waalsovou prostorovou obálkou</strong> (sterické vyloučení a tvarové vyplnění kavity).</li>
          <li><strong>Prostorovým rozložením elektrostatického potenciálu</strong> a směrovými vektory donorů a akceptorů vodíkových vazeb.</li>
          <li><strong>Hydrofobními a aromatickými $\\pi$-$\\pi$ kontakty</strong> orientovanými v přesných 3D úhlech a vzdálenostech.</li>
        </ul>
        <br>
        Fenomén <strong>Scaffold Hopping</strong> (nalezení nových patentovatelných chemických tříd léčiv se zachováním požadované 3D vazebné afinity a selektivity) je proto na úrovni 2D otisků prakticky neřešitelný. Metoda <strong>ROCS (Rapid Overlay of Chemical Structures)</strong>, kterou v roce 1996 představili J. A. Grant, B. T. Pickup a A. Nicholls, překonává toto omezení přechodem od 2D grafů k přímému porovnávání 3D spojitých hustot.`,
        compare: {
          leftTitle: "2D Topologické Fingerprinty (ECFP4 / Morgan)",
          leftContent: `<ul>
            <li>Kódují pouze konektivitu 2D grafu a sousedství atomů do $r = 2$ vazeb.</li>
            <li>Citlivé na jakoukoliv změnu centrálního heterocyklu (nízké $T_{2D}$ při záměně jádra).</li>
            <li>Nezohledňují konformační flexibilitu ani skutečný 3D prostorový tvar molekuly.</li>
            <li>Nerozeznávají bioisosterismus mezi topologicky nepodobnými strukturami.</li>
          </ul>`,
          rightTitle: "3D Tvarové Zarovnání ROCS (Grant & Pickup)",
          rightContent: `<ul>
            <li>Modeluje molekulu jako spojitou 3D Gaussovskou elektronovou/objemovou hustotu.</li>
            <li>Umožňuje okamžitý <em>Scaffold Hopping</em> přes různé chemické rodiny.</li>
            <li>Integruje prostorový tvar ($T_{\\text{shape}}$) s farmakoforovými vlastnostmi ($T_{\\text{color}}$).</li>
            <li>Poskytuje hladkou a diferencovatelnou skórovací funkci pro Reinforcement Learning.</li>
          </ul>`
        }
      },
      {
        title: "2. Grant-Pickupova Gaussovská reprezentace atomové a molekulární hustoty",
        content: `Historické přístupy k 3D tvarovému porovnávání modelovaly atomy jako pevné neprostupné koule (<em>hard spheres</em>) s van der Waalsovými poloměry $R_i$. Výpočet objemového překryvu dvou těles $V(A \\cap B)$ však vyžadoval numerickou integraci na diskrétní 3D prostorové mřížce (voxel grid). Mřížkové metody trpěly obrovskou výpočetní náročností ($O(N^3)$ paměť i čas), chybami diskretizace a nespojitostí při rotaci.
        <br><br>
        Zásadní průlom Granta a Pickupa spočíval v nahrazení pevných koulí <strong>spojitými trojrozměrnými Gaussovskými funkcemi hustoty</strong>. Pro každý atom $i$ s polohovým vektorem jádra $\\mathbf{r}_i \\in \\mathbb{R}^3$ je atomová hustota v bodě $\\mathbf{r}$ definována jako:
        <div class="math-card">
          $$\\rho_i(\\mathbf{r}) = p_i \\exp\\left(-\\alpha_i |\\mathbf{r} - \\mathbf{r}_i|^2\\right)$$
        </div>
        kde:
        <ul>
          <li>$\\mathbf{r}_i = (x_i, y_i, z_i)^T$ jsou kartézské souřadnice jádra atomu $i$.</li>
          <li>$p_i$ je centrální hustota atomu (standardně normalizovaná na $p_i = 1.0$ pro sférický tvar).</li>
          <li>$\\alpha_i$ je koeficient rozptylu Gaussovské funkce, který přímo determinuje efektivní velikost atomu.</li>
        </ul>
        <br>
        Parametr $\\alpha_i$ je kalibrován tak, aby integrální objem Gaussovské funkce přes celý prostor $\\mathbb{R}^3$ přesně odpovídal van der Waalsovu objemu atomu $V_i = \\frac{4}{3}\\pi R_i^3$:
        <div class="math-card">
          $$V_i = \\int_{\\mathbb{R}^3} \\rho_i(\\mathbf{r}) d\\mathbf{r} = p_i \\left( \\frac{\\pi}{\\alpha_i} \\right)^{3/2} = \\frac{4}{3}\\pi R_i^3 \\implies \\alpha_i = \\frac{\\pi}{\\left(\\frac{4}{3}\\pi\\right)^{2/3} R_i^2} \\approx \\frac{1.1444 \\pi}{R_i^2}$$
        </div>
        V praktické implementaci ROCS (OpenEye, CDPKit, RDKit) se často volí zjednodušená normalizace $\\alpha_i = \\frac{\\pi}{R_i^2}$, kde $R_i$ je tabulkový van der Waalsův poloměr daného prvku (např. $R_C = 1.70\\,\\text{Å}$, $R_N = 1.55\\,\\text{Å}$, $R_O = 1.52\\,\\text{Å}$, $R_S = 1.80\\,\\text{Å}$, $R_F = 1.47\\,\\text{Å}$).
        <br><br>
        Celková prostorová hustota molekuly $A$ složené z $N_A$ atomů je dána lineární superpozicí jednotlivých atomových Gaussovských hustot:
        <div class="math-card">
          $$\\rho_A(\\mathbf{r}) = \\sum_{i \\in A} \\rho_i(\\mathbf{r}) = \\sum_{i=1}^{N_A} p_i \\exp\\left(-\\alpha_i |\\mathbf{r} - \\mathbf{r}_i|^2\\right)$$
        </div>`
      },
      {
        title: "3. Uzavřené analytické řešení objemových překryvových integrálů V(A,B)",
        content: `Míra prostorového tvarového překryvu mezi molekulou $A$ a molekulou $B$ je definována jako konvoluční integrál součinu jejich hustot přes celý prostor $\\mathbb{R}^3$:
        <div class="math-card">
          $$V(A, B) = \\int_{\\mathbb{R}^3} \\rho_A(\\mathbf{r}) \\rho_B(\\mathbf{r}) d\\mathbf{r} = \\int_{\\mathbb{R}^3} \\left( \\sum_{i \\in A} \\rho_i(\\mathbf{r}) \\right) \\left( \\sum_{j \\in B} \\rho_j(\\mathbf{r}) \\right) d\\mathbf{r} = \\sum_{i \\in A} \\sum_{j \\in B} \\int_{\\mathbb{R}^3} \\rho_i(\\mathbf{r}) \\rho_j(\\mathbf{r}) d\\mathbf{r}$$
        </div>
        <br>
        Zásadní matematickou vlastností součinu dvou trojrozměrných Gaussovských funkcí se středy v $\\mathbf{r}_i$ a $\\mathbf{r}_j$ je, že jejich integrál má <strong>přesné analytické řešení v uzavřeném tvaru</strong>:
        <div class="math-card">
          $$I_{ij} = \\int_{\\mathbb{R}^3} \\exp\\left(-\\alpha_i |\\mathbf{r} - \\mathbf{r}_i|^2\\right) \\exp\\left(-\\alpha_j |\\mathbf{r} - \\mathbf{r}_j|^2\\right) d\\mathbf{r} = \\left( \\frac{\\pi}{\\alpha_i + \\alpha_j} \\right)^{3/2} \\exp\\left( -\\frac{\\alpha_i \\alpha_j}{\\alpha_i + \\alpha_j} |\\mathbf{r}_i - \\mathbf{r}_j|^2 \\right)$$
        </div>
        <br>
        Celkový objemový překryv $V(A, B)$ je pak dán jednoduchou dvojitou sumou přes všechny dvojice atomů:
        <div class="math-card">
          $$V(A, B) = \\sum_{i \\in A} \\sum_{j \\in B} p_i p_j \\left( \\frac{\\pi}{\\alpha_i + \\alpha_j} \\right)^{3/2} \\exp\\left( -\\frac{\\alpha_i \\alpha_j}{\\alpha_i + \\alpha_j} |\\mathbf{r}_i - \\mathbf{r}_j|^2 \\right)$$
        </div>
        <br>
        Stejným způsobem se spočítají i tzv. <strong>vlastní objemy (self-volumes)</strong> molekul $A$ a $B$:
        <div class="math-card">
          $$V(A, A) = \\sum_{i \\in A} \\sum_{i' \\in A} p_i p_{i'} \\left( \\frac{\\pi}{\\alpha_i + \\alpha_{i'}} \\right)^{3/2} \\exp\\left( -\\frac{\\alpha_i \\alpha_{i'}}{\\alpha_i + \\alpha_{i'}} |\\mathbf{r}_i - \\mathbf{r}_{i'}|^2 \\right)$$
        </div>
        <br>
        <strong>Proč je analytické řešení revoluční?</strong>
        <ol>
          <li><strong>Nulová diskretizační chyba</strong>: Žádné artefakty způsobené volbou velikosti mřížky voxelů.</li>
          <li><strong>Extrémní výpočetní rychlost</strong>: Vyhodnocení jedné dvojice molekul trvá méně než $1\\,\\mu\\text{s}$ na moderním CPU.</li>
          <li><strong>Diferencovatelnost</strong>: Gradient $\\nabla_{\\mathbf{r}_j} V(A, B)$ má rovněž analytické vyjádření, což umožňuje bleskovou optimalizaci prostorové orientace pomocí kvazi-Newtonových metod (BFGS).</li>
        </ol>`
      },
      {
        title: "4. Metrika Shape Tanimoto a optimalizace prostorového zarovnání",
        content: `Znalost integrálů $V(A, B)$, $V(A, A)$ a $V(B, B)$ umožňuje definovat normalizovanou míru prostorové podobnosti — <strong>Shape Tanimoto koeficient ($T_{\\text{shape}}$)</strong>:
        <div class="math-card">
          $$T_{\\text{shape}}(A, B) = \\frac{V(A, B)}{V(A, A) + V(B, B) - V(A, B)}$$
        </div>
        <br>
        Tato metrika splňuje všechny požadované matematické vlastnosti:
        <ul>
          <li><strong>Ohraničenost</strong>: $T_{\\text{shape}}(A, B) \\in [0, 1]$.</li>
          <li><strong>Identita</strong>: $T_{\\text{shape}}(A, A) = 1.0$ (pro identické molekuly v identické konformaci).</li>
          <li><strong>Symetrie</strong>: $T_{\\text{shape}}(A, B) = $T_{\\text{shape}}(B, A)$.</li>
          <li><strong>Míra disjunkce</strong>: $T_{\\text{shape}} = 0$ pro molekuly umístěné v nekonečné vzdálenosti ($|\\mathbf{r}_i - \\mathbf{r}_j| \\to \\infty$).</li>
        </ul>
        <br>
        <h4>Optimalizace prostorového zarovnání v prostoru tuhých těles $SE(3)$</h4>
        Hodnota $T_{\\text{shape}}$ závisí na vzájemné poloze a orientaci molekul. Cílem ROCS je nalézt rigidní transformaci $\\mathbf{T} = (\\mathbf{R}, \\mathbf{t}) \\in SE(3)$ (rotaci $\\mathbf{R} \\in SO(3)$ a translaci $\\mathbf{t} \\in \\mathbb{R}^3$), která maximalizuje objemový překryv:
        <div class="math-card">
          $$\\mathbf{T}^* = \\arg\\max_{\\mathbf{T}} V\\left(A, \\mathbf{T}(B)\\right)$$
        </div>
        <br>
        Aby optimalizační algoritmus neuvízl v lokálních minimech, provádí se výpočet ve dvou krocích:
        <ol>
          <li><strong>Startovní generátor hlavních os setrvačnosti (Principal Axes of Inertia - PAI)</strong>: Pro obě molekuly se spočte tenzor momentu setrvačnosti $\\mathbf{I}$. Vlastní vektory tenzoru definují 3 ortogonální hlavní osy. Sladěním hlavních os vzniknou 4 neekvivalentní výchozí orientace.</li>
          <li><strong>Lokální gradientní optimalizace</strong>: Z každé ze 4 startovních pozic se spustí kvazi-Newtonova optimalizace překryvu s analytickými gradienty. Nejlepší nalezené maximum definuje finální $T_{\\text{shape}}$.</li>
        </ol>`
      },
      {
        title: "5. Farmakoforový Color Tanimoto & Silové pole ImplicitMillsDean",
        content: `Samotný tvar ($T_{\\text{shape}}$) pro molekulární návrh nestačí: hydrofobní naftalenový skelet má téměř identický 3D tvar jako polární chinazolin nebo isoquinolin ($T_{\\text{shape}} > 0.85$), avšak jejich vazebné vlastnosti a elektrostatické profily jsou zcela protikladné.
        <br><br>
        Proto ROCS zavádí koncept tzv. <strong>Color Features (farmakoforových center)</strong>. Každé funkční skupině je přiřazena specifická chemická barva (typ interakce). V průmyslovém standardu silového pole <code>ImplicitMillsDean</code> rozlišujeme 6 základních typů vlastností:
        <ol>
          <li><strong>Donory vodíkových vazeb (Donors)</strong>: skupiny $-\\text{OH}$, $-\\text{NH}_2$, $-\\text{NH}-$.</li>
          <li><strong>Akceptory vodíkových vazeb (Acceptors)</strong>: karbonylové kyslíky $=\\text{O}$, pyridinové dusíky $=\\text{N}-$, ethery $-\\text{O}-$.</li>
          <li><strong>Kationická centra (Positive / Cations)</strong>: protonované aminy, guanidiniové skupiny, kvarterní dusíky.</li>
          <li><strong>Anionická centra (Negative / Anions)</strong>: karboxylátové $-\\text{COO}^-$, sulfonátové $-\\text{SO}_3^-$, fosfátové skupiny.</li>
          <li><strong>Hydrofobní centra (Hydrophobes)</strong>: alifatické a cyklické uhlíkové clustery (např. terc-butyl, isobutyl, cyklohexyl).</li>
          <li><strong>Aromatické kruhy (Rings)</strong>: planární aromatické $\\pi$-systémy (benzen, thiofen, indol).</li>
        </ol>
        <br>
        Farmakoforová centra jsou modelována jako barevné Gaussovské funkce $\\rho_{i, c}(\\mathbf{r})$ umístěné v geometrických středech příslušných skupin. <strong>K barevnému překryvu dochází VÝHRADNĚ mezi centry STEJNÉHO typu</strong>:
        <div class="math-card">
          $$C(A, B) = \\sum_{c \\in \\text{Types}} \\sum_{i \\in A_c} \\sum_{j \\in B_c} w_c \\left( \\frac{\\pi}{\\alpha_{c,i} + \\alpha_{c,j}} \\right)^{3/2} \\exp\\left( -\\frac{\\alpha_{c,i} \\alpha_{c,j}}{\\alpha_{c,i} + \\alpha_{c,j}} |\\mathbf{r}_i - \\mathbf{r}_j|^2 \\right)$$
        </div>
        kde $w_c$ je váha dané vlastnosti. Normalizovaný <strong>Color Tanimoto koeficient</strong> je pak definován jako:
        <div class="math-card">
          $$T_{\\text{color}}(A, B) = \\frac{C(A, B)}{C(A, A) + C(B, B) - C(A, B)} \\in [0, 1]$$
        </div>`
      },
      {
        title: "6. Kompozitní metrika TanimotoCombo & Interpretace ve výzkumu",
        content: `Pro celkové hodnocení 3D tvarové a elektrostatické shody v generativním návrhu léčiv se používá kompozitní skóre <strong>TanimotoCombo ($T_{\\text{combo}}$)</strong>:
        <div class="math-card">
          $$T_{\\text{combo}}(A, B) = T_{\\text{shape}}(A, B) + T_{\\text{color}}(A, B) \\in [0, 2]$$
        </div>
        <br>
        V DrugEx MORL slouží $T_{\\text{combo}}$ jako primární spojitá skórovací funkce odměny. V literatuře a medicinálně-chemické praxi se hodnoty interpretují následovně:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">Klasifikace hodnot TanimotoCombo</div>
            <ul>
              <li><strong>$T_{\\text{combo}} < 0.80$</strong>: Nízká prostorová nebo barevná shoda. Molekula nepasuje do vazebné kavity nebo má špatně orientované polární skupiny.</li>
              <li><strong>$0.80 \\le T_{\\text{combo}} < 1.20$</strong>: Střední shoda. Dobrý tvarový překryv, ale chybí některé klíčové vodíkové vazby nebo nábojové interakce. Typické pro hrubé výchozí skelety.</li>
              <li><strong>$1.20 \\le T_{\\text{combo}} < 1.50$</strong>: <strong>Vysoká bioisosterní shoda</strong>. Molekula dokonale vyplňuje prostor a prezentuje klíčové farmakoforové body ve správné geometrii. Silný kandidát na <em>Scaffold Hop</em>.</li>
              <li><strong>$T_{\\text{combo}} \\ge 1.50$</strong>: Téměř identický tvar i farmakofor (často analog původního ligandu s drobnou substitucí).</li>
            </ul>
          </div>
          <div class="compare-col right">
            <div class="compare-heading">Proč TanimotoCombo v DrugEx MORL?</div>
            <ul>
              <li><strong>Hladká spojitá krajina odměn</strong>: Na rozdíl od binárních deskriptorů poskytuje $T_{\\text{combo}}$ nenulový gradient i pro částečně překryté molekuly.</li>
              <li><strong>Robustnost vůči konformačnímu šumu</strong>: Zohledňuje nejlepší konformer z vygenerovaného ansámblu.</li>
              <li><strong>Přímá korelace s $IC_{50}$</strong>: U řady flexibilních cílů a receptorů (např. CCR2) dosahuje $T_{\\text{combo}}$ špičkové diskriminační schopnosti (ROC-AUC $> 0.90$).</li>
            </ul>
          </div>
        </div>`,
        code: `from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

# Příklad výpočtu TanimotoCombo pro sadu molekul
conf_gen = RDKitConformerGenerator(max_conformers=30, max_isomers=2, num_threads=1)
rocs_scorer = RDKitROCSScorer(
    conformer_generator=conf_gen,
    references="CCR2_reference_ligands.sdf",
    score_type="TanimotoCombo",  # Shape (0..1) + Color (0..1) = Combo (0..2)
    use_colors=True,
    show_progress=False
)

# getScores vrací numpy matici tvaru (num_molecules, num_reference_groups)
# scores = rocs_scorer.getScores(generated_mols)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 3.2: FLEXIBLE TARGETS & INTRINSICALLY DISORDERED PROTEINS (IDPS)
  // =========================================================================
  "l3_2": {
    id: "l3_2",
    tag: "Legendary",
    relevance: 10,
    title: "3.2 Výzva flexibilních cílů & Intrinsically Disordered Proteins (IDP)",
    summary: "Proč rigidní krystalografické dokování selhává u IDP a dynamických kavit, Ligand-Based 3D paradigma, ansámbly vazebných konformací, konsensuální supermolekuly a Multi-Reference Grouping.",
    slides: [
      {
        title: "1. Selhání rigidního strukturního dokování u flexibilních cílů a IDP",
        content: `Tradiční strukturový de novo design léčiv (Structure-Based Drug Design - SBDD) vychází z předpokladu statického receptoru a teorie <em>zámku a klíče</em> (Lock-and-Key). Algoritmy molekulárního dokování (např. AutoDock Vina, Schrödinger Glide, Gold) vkládají ligand do tuhé krystalografické mřížky proteinu s vysokým rozlišením (< 2.0 Å).
        <br><br>
        V moderní medicinální chemii a onkologii však klíčové terapeutické cíle tomuto paradigmatu zásadně vzdorují:
        <ul>
          <li><strong>Intrinsically Disordered Proteins (IDP) a IDR domény</strong>: Proteiny (např. onkoproteiny c-Myc, p53 transaktivační doména, $\\alpha$-synuklein, tau protein), které postrádají stabilní sekundární a terciární strukturu. V roztoku existují jako dynamický konformační ansámbl na ploché energetické krajině s mělkými minimy ($\\Delta G^\\ddagger \\approx k_B T$).</li>
          <li><strong>Flexibilní membránové receptory (GPCR, iontové kanály)</strong>: Kde vazba ligandu indukuje rozsáhlé alosterické posuny transmembránových helixů a přeskupení vazebné kapsy (mechanismus <em>Induced Fit</em> a <em>Conformational Selection</em>).</li>
          <li><strong>Proteiny s kryptickými (skrytými) kavitami</strong>: Vazebné kapsy, které v nestimulovaném stavu vůbec neexistují a otevírají se teprve přítomností ligandu.</li>
          <li><strong>Modely s nízkým rozlišením (Cryo-EM > 3.5 Å, AlphaFold predikce)</strong>: Kde nepřesné rotamery postranních řetězců vedou při rigidním dokování k falešným sterickým srážkám (clashes) a vyřazení vysoce aktivních molekul.</li>
        </ul>
        <br>
        Aplikace rigidního SBDD dokování na IDP vede k naprostému selhání (masivní výskyt falešně negativních i falešně pozitivních předpovědí).`,
        compare: {
          leftTitle: "SBDD Dokování do Rigidního Receptoru",
          leftContent: `<ul>
            <li>Vyžaduje rigidní krystalovou strukturu z PDB s vysokým rozlišením.</li>
            <li>Ignoruje konformační entropii a plasticitu vazebného rozhraní.</li>
            <li>Malá nepřesnost polohy aminokyseliny ($0.5\\,\\text{Å}$) způsobí sterickou srážku a nulové skóre.</li>
            <li>U IDP a dynamických cílů zcela selhává.</li>
          </ul>`,
          rightTitle: "Ligand-Based 3D ROCS Shape Matching (LBDD)",
          rightContent: `<ul>
            <li>Využívá experimentální konformace ověřených aktivních ligandů.</li>
            <li>Aktivní ligandy představují <strong>negativní prostorový a elektrostatický otisk</strong> vazebného místa.</li>
            <li>Zachovává toleranci k flexibilitě proteinu díky Gaussovskému překryvu.</li>
            <li>Umožňuje de novo generování i pro zcela nestrukturované cíle.</li>
          </ul>`
        }
      },
      {
        title: "2. Ligand-Based 3D Drug Design (LBDD) jako rigorózní alternativa",
        content: `Pokud máme k dispozici sadu experimentálně potvrzených aktivních ligandů (získaných např. z NMR titrací, povrchové plasmonové rezonance SPR, hmotnostní spektrometrie nebo biochemických esejí), jejich 3D konformace vázaná na protein představuje <strong>dokonalý negativní prostorový a farmakoforový odlitek</strong> funkčního rozhraní proteinu.
        <br><br>
        Termodynamické zdůvodnění: Vysoce afinitní ligand ($K_d < 100\\,\\text{nM}$) stabilizuje specifický funkční konformační stav IDP nebo receptoru. Pokud generativní model navrhne novou molekulu, která:
        <ol>
          <li>Trojrozměrným tvarem přesně vyplní van der Waalsovský objem aktivních referencí ($T_{\\text{shape}} \\to 1.0$),</li>
          <li>Prezentuje donory, akceptory a náboje ve shodných prostorových koordinátách ($T_{\\text{color}} \\to 1.0$),</li>
          <li>Zachovává nízkou vnitřní pnutí a vysokou syntetickou dostupnost ($\\text{SAScore} \\le 3.0$),</li>
        </ol>
        tato nová molekula má vysokou pravděpodobnost vázat se do stejného funkčního konformačního stavu proteinu se srovnatelnou či vyšší afinitou, bez ohledu na to, zda máme k dispozici krystalovou strukturu receptoru.
        <br><br>
        Tento koncept tvoří ústřední pilíř této bakalářské práce: <strong>využití 3D ROCS shape matchingu jako objektivní odměňovací funkce v prostředí DrugEx MORL pro de novo design léčiv na flexibilní cíle</strong>.`
      },
      {
        title: "3. Ansámbly bioaktivních konformací & Experimentální data (NMR / SPR)",
        content: `U flexibilních proteinů ligand zřídka váže cíl v jediné rigidní geometrii. V roztoku se často uplatňuje konformační heterogenita:
        <ul>
          <li><strong>NMR spektroskopie v roztoku</strong>: Metody jako trNOE (transferred NOE) a měření chemických posunů (CSP) určují 3D konformaci ligandu přímo ve vázaném stavu v roztoku bez krystalizačních artefaktů.</li>
          <li><strong>Vícečetné vazebné módy (Binding Modes)</strong>: Ligand může obsazovat dvě překrývající se podkavity nebo vázat cíl v různých orientacích (např. monomer vs dimer rozhraní).</li>
          <li><strong>Cryo-EM konformační třídění</strong>: Rekonstrukce několika koexistujících stavů makromolekulárního komplexu.</li>
        </ul>
        <br>
        V DrugEx proto nelze spoléhat na porovnávání vůči jedinému statickému ligandu. Do skórovacího modulu se vkládají <strong>ansámbly bioaktivních referencí</strong>, které pokrývají celou šíři experimentálně ověřeného vazebného prostoru.`
      },
      {
        title: "4. Konsensuální supermolekuly (supermol_123.sdf)",
        content: `Jednou z klíčových technik pro reprezentaci komplexního vazebného rozhraní je vytvoření <strong>konsensuální supermolekuly (Supermolecule Template)</strong>:
        <br><br>
        Postup tvorby supermolekuly:
        <ol>
          <li>Sjednotí se souřadné systémy několika známých krystalografických nebo NMR struktur ligandů ve vázaném stavu (např. ligandy 1, 2 a 3).</li>
          <li>Strukturně se sloučí do jednoho SDF souboru (např. <code>CCR2_supermol_123.sdf</code>).</li>
          <li>Supermolekula definuje <strong>sjednocení (union) prostorových objemů</strong> všech referencí:
            <div class="math-card">
              $$V_{\\text{super}} = \\bigcup_{k} V(R_k)$$
            </div>
          </li>
          <li>Farmakoforové body jsou konsolidovány: překrývající se identické interakce jsou zesíleny, zatímco unikátní skupiny rozšiřují povolený prostor pro chemickou substituci.</li>
        </ol>
        <br>
        Výhody supermolekuly v DrugEx:
        <ul>
          <li>Jediný výpočet 3D zarovnání na kandidáta ($1\\times$ zarovnání namísto $N\\times$), což dramaticky zkracuje čas trénovací epochy.</li>
          <li>Generátor je motivován navrhovat molekuly, které propojují sub-kapsy obsazené různými referenčními ligandy (tvorba multivalentních či fragment-linked molekul).</li>
        </ul>`
      },
      {
        title: "5. Multi-Reference Group Scoring (max TanimotoCombo)",
        content: `Pokud referenční ligandy vážou protein v navzájem se vylučujících konformacích (např. orthosterické vs alosterické vazebné místo), sloučení do supermolekuly by vytvořilo nerealisticky obří tvar.
        <br><br>
        Pro tento případ implementuje DrugEx pokročilý mechanismus <strong>Multi-Reference Groupingu</strong>:
        <br><br>
        Reference jsou uspořádány do slovníku pojmenovaných skupin:
        <div class="math-card">
          $$\\text{References} = \\left\\{ \\text{'orthosteric'}: [R_1, R_2], \\; \\text{'allosteric'}: [R_3, R_4] \\right\\}$$
        </div>
        <br>
        Skórovací funkce vyhodnotí shodu kandidátní molekuly $X$ se všemi referencemi v dané skupině a vrátí <strong>maximální dosažené skóre</strong>:
        <div class="math-card">
          $$\\text{Score}_{\\text{group}}(X) = \\max_{R \\in \\text{Group}} \\left( \\max_{k \\in \\text{Conformers}(X)} T_{\\text{combo}}(C_k(X), R) \\right)$$
        </div>
        <br>
        Tento přístup umožňuje multi-cílovou optimalizaci v DrugEx MORL: generátor může být odměňován za schopnost vázat buď libovolné ze známých vazebných míst, nebo současně optimalizovat selektivitu vůči více konformačním stavům.`
      },
      {
        title: "6. Výpočetní racionále bakalářské práce (VŠCHT / ÚOCHB)",
        content: `Tato bakalářská práce spojuje nejmodernější koncepty generativní chemie a 3D modelování do uceleného výpočetního řetězce:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">Generativní AI & MORL (DrugEx)</div>
            <ul>
              <li>Autoregresní jazykové modely (SMILES RNN / GPT) pro generování chemicky validních molekul.</li>
              <li>Duální síťová architektura (Agent vs Mutate Prior) bránící <em>mode collapse</em>.</li>
              <li>Paretovo nedominované třídění s Crowding Distance pro vícekriteriální vyvažování.</li>
            </ul>
          </div>
          <div class="compare-col right">
            <div class="compare-heading">3D Tvarové Řízení (ROCS Engine)</div>
            <ul>
              <li>Analytické Gaussovské objemové integrály pro ultra-rychlé 3D hodnocení.</li>
              <li>Optimalizace shody vůči experimentálním referencím flexibilního receptoru CCR2.</li>
              <li>Penalizace syntetické nedostupnosti (SAScore) a filtrace nežádoucích reaktivních skupin.</li>
            </ul>
          </div>
        </div>
        <br>
        Výsledkem je plně automatizovaný systém schopný z nepopsaného listu navrhnout originální chemické entity se špičkovým 3D tvarem, perfektní farmakoforovou komplementaritou a vysokou syntetickou dostupností pro syntézu na ÚOCHB AV ČR.`,
        code: `# Příklad konfigurace Multi-Reference Groupingu v DrugEx
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

conf_gen = RDKitConformerGenerator(max_conformers=50, max_isomers=4, num_threads=1)

# Skórovač s definicí funkčních skupin referencí
scorer = RDKitROCSScorer(
    conformer_generator=conf_gen,
    references={
        "CCR2_pocket_A": ["references/ref_ligand_1.sdf", "references/ref_ligand_2.sdf"],
        "CCR2_pocket_B": ["references/ref_ligand_3.sdf"]
    },
    score_type="TanimotoCombo",
    use_colors=True,
    n_jobs=-1
)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 3.3: CONFORMER GENERATION DEEP-DIVE
  // =========================================================================
  "l3_3": {
    id: "l3_3",
    tag: "Tricky",
    relevance: 10,
    title: "3.3 Hloubková analýza conformer_generators.py (RDKit, OMEGA, CDPKit, Schrödinger)",
    summary: "Abstraktní rozhraní ConformerGenerator, generátory ETKDGv3, OMEGA GPU, CDPL.ConfGen a Schrödinger ConfGenX, oprava planárních souřadnic (+0.01 Å fix) a vláknová bezpečnost.",
    slides: [
      {
        title: "1. Architektura a kontrakt třídy ConformerGenerator (conformer_generators.py)",
        content: `Generování nízkokonformačních 3D geometrií z 1D SMILES je výpočetně nejnáročnějším článkem celého 3D shape-matching řetězce. Během jedné trénovací epochy v DrugEx vygeneruje neuronový model 1 000 nových molekul. Pro každou z nich je nutné vygenerovat desítky 3D konformací, což znamená vygenerovat a optimalizovat 30 000 až 50 000 3D souřadnicových struktur během několika sekund.
        <br><br>
        V souboru <code>drugex/training/scorers/conformer_generators.py</code> je definována abstraktní základní třída <code>ConformerGenerator</code> (dědící z <code>drugex.training.scorers.interfaces.ConformerGenerator</code>).
        <br><br>
        Základní kontrakt rozhraní předepisuje dvě klíčové metody:
        <ol>
          <li><code>genConformers(smiles_list: List[str], out_dir: str) -> str</code>:
            <ul>
              <li>Přijímá seznam SMILES řetězců a cestu k dočasnému výstupnímu adresáři.</li>
              <li>Provádí filtraci struktur (heavy atomy, rotovatelné vazby), enumeraci stereoizomerů a generování konformačních ansámblů.</li>
              <li>Ukládá vygenerované konformace do standardizovaného multi-molekulárního souboru (SDF nebo OEB.GZ) s pojmenováním ve formátu <code>mol_{i}+{j}</code> (kde $i$ je index vstupní molekuly a $j$ je index stereoizomeru).</li>
              <li>Vrací absolutní cestu k výslednému souboru.</li>
            </ul>
          </li>
          <li><code>write_conformers(mols: List, out_file: str) -> None</code>:
            <ul>
              <li>Pomocná metoda pro přímý zápis konformací z RDKit molekulárních objektů do cílového SDF souboru.</li>
            </ul>
          </li>
        </ol>`
      },
      {
        title: "2. RDKit backend: Algoritmus ETKDGv3 (Riniker & Landrum)",
        content: `Třída <code>RDKitConformerGenerator</code> představuje výchozí, plně otevřený a multiplatformní generátor. Využívá metodu <strong>ETKDGv3 (Experimental-Torsion Knowledge Distance Geometry version 3)</strong> vyvinutou S. Riniker a G. Landrumem (2015, 2020).
        <br><br>
        Algoritmický princip ETKDGv3:
        <ol>
          <li><strong>Matice distančních mezí (Distance Bounds Matrix)</strong>: Z 2D topologie a délek vazeb se sestaví matice minimálních a maximálních povolených vzdáleností mezi všemi dvojicemi atomů.</li>
          <li><strong>Znalostní torzní potenciály z databáze CSD</strong>: Na rotovatelné vazby jsou aplikovány statistické torzní úhlové preference odvozené z desetitisíců experimentálních krystalografických struktur malých molekul v Cambridge Structural Database (CSD).</li>
          <li><strong>Vnoření ze 4D do 3D prostoru (Embedding)</strong>: Metodou Distance Geometry s náhodným inicializačním seedem (<code>params.randomSeed = 0xc0ffee</code>) se naleznou 3D souřadnice.</li>
          <li><strong>Silová optimalizace</strong>: Geometrie je dočištěna silovým polem MMFF94 nebo UFF pro odstranění sterických srážek.</li>
          <li><strong>RMSD prořezávání (Pruning)</strong>: Konformace s geometrickou odchylkou $\\text{RMSD} < 0.5\\,\\text{Å}$ jsou zahozeny (<code>params.pruneRmsThresh = 0.5</code>), což brání redundanci.</li>
        </ol>`,
        code: `class RDKitConformerGenerator(ConformerGenerator):
    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        num_threads: int = 0,
        show_progress: bool = False,
    ):
        self.max_conformers = min(max_conformers, 200)
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.num_threads = num_threads
        self.show_progress = show_progress

    def _create_fresh_etkdg(self):
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xc0ffee
        params.numThreads = self.num_threads
        params.pruneRmsThresh = 0.5
        return params`
      },
      {
        title: "3. OpenEye backend: OMEGA pravidlové torzní vzorkování a GPU mód",
        content: `Třída <code>OmegaConformerGenerator</code> využívá komerční nástroj <strong>OpenEye OMEGA</strong> (Hawkins et al., OpenEye Scientific Software), který je v průmyslové farmacii považován za zlatý standard rychlosti a přesnosti.
        <br><br>
        Klíčové vlastnosti implementace v DrugEx:
        <ul>
          <li><strong>Fragmentový rozklad</strong>: Molekula je rozdělena na rigidní fragmenty spojené rotovatelnými vazbami. Konformace rigidních jader jsou načteny z předpočítané databáze.</li>
          <li><strong>Systematické torzní řízení (Torsion Driving)</strong>: OMEGA prohledává rotamery podle diskrétních energetických pravidel namísto stochastického vnořování.</li>
          <li><strong>Enumerace stereoizomerů přes OEFlipper</strong>: Pro neoznačená chirální centra generuje stereoizomery třída <code>oeomega.OEFlipper</code> s limitem <code>max_centers</code>.</li>
          <li><strong>GPU Akcelerace</strong>: Při volbě <code>use_gpu=True</code> a přítomnosti CUDA hardware se aktivuje GPU torzní engine přes <code>opts.GetTorDriveOptions().SetUseGPU(True)</code>, což umožňuje vzorkovat přes 5 000 konformací za sekundu na jedné GPU.</li>
        </ul>`,
        code: `class OmegaConformerGenerator(ConformerGenerator):
    def _create_fresh_omega(self):
        opts = oeomega.OEOmegaOptions()
        opts.SetMaxConfs(self.max_conformers)
        opts.SetStrictStereo(False)
        opts.SetFromCT(True)
        opts.SetFixRMS(True)
        opts.SetRMSThreshold(0.5)
        opts.SetEnumRing(True)
        opts.SetRotorOffset(False)

        if self.use_gpu and oeomega.OEOmegaIsGPUReady():
            opts.GetTorDriveOptions().SetUseGPU(True)
            opts.SetSampleHydrogens(False)
        else:
            opts.GetTorDriveOptions().SetUseGPU(False)
            opts.SetSampleHydrogens(True)

        return oeomega.OEOmega(opts)`
      },
      {
        title: "4. CDPKit backend: CDPL.ConfGen C++ pipeline",
        content: `Třída <code>CDPKitConformerGenerator</code> integruje C++ knihovnu <strong>CDPL (Chemical Data Processing Library)</strong>. Poskytuje plně otevřenou alternativu k OMEGA s energetickým oknem a RMSD shlukováním.
        <br><br>
        Architektura CDPKit pipeline:
        <ol>
          <li><code>CDPLChem.calcBasicProperties(mol)</code>: Inicializuje základní chemické deskriptory (hybridizaci, aromatické kruhy, počet H-atomů).</li>
          <li><code>CDPLChem.StereoisomerGenerator</code>: Provádí explicitní enumeraci konfigurací na atomech (<code>enumerateAtomConfig(True)</code>) i dvojných vazbách (<code>enumerateBondConfig(True)</code>) až do limitu <code>max_isomers</code>.</li>
          <li><code>CDPLConfGen.ConformerGenerator</code>:
            <ul>
              <li><code>timeout = self.timeout * 1000</code>: Časový limit v milisekundách.</li>
              <li><code>minRMSD = 0.5</code>: Minimální geometrická vzdálenost mezi konformacemi.</li>
              <li><code>energyWindow = 20.0 kcal/mol</code>: Energetické okno pro zachování nízkoenergetických stavů.</li>
              <li><code>maxNumOutputConformers = self.max_conformers</code>: Maximální počet výstupních konformací.</li>
            </ul>
          </li>
          <li><strong>Ošetření návratových kódů</strong>: Korektní obsluha stavů <code>SUCCESS</code>, <code>TOO_MUCH_SYMMETRY</code> (vysoce symetrické molekuly), <code>TIMEOUT</code> a <code>FORCEFIELD_SETUP_FAILED</code>.</li>
        </ol>`
      },
      {
        title: "5. Schrödinger backend: ConfGenX & Oprava planárních souřadnic (+0.01 Å fix)",
        content: `Třída <code>SchrodingerConformerGenerator</code> spouští proprietární nástroj <code>confgenx</code> z balíku Schrödinger Suite pomocí CLI subprocessu. Výstupní soubor Maestro <code>.maegz</code> je konvertován do SDF utilitou <code>sdconvert</code>.
        <br><br>
        <h4>Kritický inženýrský objev: Oprava singularity u planárních molekul (+0.01 Å fix)</h4>
        Během rozsáhlých výpočtů v DrugEx došlo k závažnému problému: u dokonale planárních aromatických molekul (např. ligand 8SKP) vygeneroval <code>confgenx</code> konformaci, kde všechny atomy měly přesně nulovou z-souřadnici ($z_i = 0.0000\\,\\text{Å}$).
        <br><br>
        Při následném předání do ROCS došlo k <strong>pádu výpočtu (singularity crash)</strong>:
        <ul>
          <li>Výpočet tenzoru momentu setrvačnosti a Gaussovské prostorové kovariance v $z$-ose vedl k nulovému rozptylu ($\\sigma_z^2 = 0$).</li>
          <li>Algoritmus invertoval singulární matici, což způsobilo dělení nulou a pád celého procesu.</li>
        </ul>
        <br>
        V <code>conformer_generators.py</code> byl tento problém vyřešen elegantní automatickou post-processing korekcí: k z-souřadnici každého atomu je přičten nepatrný ofset $+0.01\\,\\text{Å}$, což zachová geometrii, ale odstraní numerickou singularitu:`,
        code: `# Oprava souřadnic v SchrodingerConformerGenerator
suppl = Chem.SDMolSupplier(tmp_outfile, removeHs=False)
corrected_mols = []
for mol in suppl:
    if mol is not None:
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            # Přičtení +0.01 k ose Z zabraňuje kolapsu ROCS u planárních struktur
            new_pos = (pos.x, pos.y, pos.z + 0.01)
            mol.GetConformer().SetAtomPosition(atom.GetIdx(), new_pos)
        corrected_mols.append(mol)`
      },
      {
        title: "6. Filtrace molekul & Pravidla vláknové bezpečnosti (Multi-threading Safety)",
        content: `Generování konformací pro nekontrolované chemické struktury může snadno způsobit zamrznutí nebo pád celého RL tréninku. Proto jsou ve všech 4 generátorech implementovány striktní bezpečnostní filtry a pravidla pro paralelní běh:
        <br><br>
        <h4>1. Bezpečnostní chemické filtry</h4>
        <ul>
          <li><code>max_heavy_atoms = 35 - 45</code>: Molekuly s $> 45$ těžkými atomy (C, N, O, S...) jsou okamžitě přeskočeny a získávají nulové skóre. Zabraňuje paměťovému kolapsu při náhodném vygenerování obřích oligomerů.</li>
          <li><code>max_rotatable_bonds = 15</code>: Molekuly s $> 15$ rotovatelnými vazbami mají konformační prostor o velikosti $3^{15} \\approx 1.4 \\times 10^7$ stavů, který nelze 50 konformacemi smysluplně pokrýt. Jsou proto penalizovány.</li>
          <li><code>max_isomers = 4</code>: Pokud molekula obsahuje $k$ neoznačených chirálních center ($2^k$ izomerů), enumerace je omezena na maximálně 4 stereoizomery pro zachování rozumného výpočetního času.</li>
        </ul>
        <br>
        <h4>2. Zlaté pravidlo vláknové bezpečnosti při paralelním běhu (num_threads)</h4>
        Při trénování DrugEx na víceprocesorovém superpočítači se paralelismus realizuje na úrovni procesů přes <code>multiprocessing.Pool(n_jobs)</code>.
        <br><br>
        <div class="alert-box alert-danger">
          <div class="alert-title">POZOR: CPU Oversubscription & Deadlock</div>
          Pokud je v <code>RDKitConformerGenerator</code> nastaveno <code>num_threads = 0</code> (použít všechna dostupná jádra) a skórovač současně běží s <code>n_jobs = 32</code> procesy, každý proces se pokusí alokovat 32 OpenMP vláken. Dojde k vytvoření $32 \\times 32 = 1024$ vláken, což způsobí masivní kontextové přepínání (CPU Thrashing), propad výkonu o 95 % a zamrznutí systému.
          <br><br>
          <strong>Pravidlo:</strong> Při použití paralelního skórovače (<code>n_jobs > 1</code>) MUSÍ být v generátoru konformací striktně nastaveno <code>num_threads = 1</code>!
        </div>`
      }
    ]
  }
};
