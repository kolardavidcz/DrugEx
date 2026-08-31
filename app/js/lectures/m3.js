/**
 * DrugEx Hub — Module 3: 3D Shape Matching, IDPs & Conformer Engines
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M3_LECTURES = {
  // =========================================================================
  // LECTURE 3.1
  // =========================================================================
  "l3_1": {
    id: "l3_1",
    tag: "Core",
    relevance: 10,
    title: "3.1 Fyzikální podstata ROCS: Shape Tanimoto & Color TanimotoCombo",
    summary: "Gaussovská reprezentace atomů, objemové překryvové integrály a farmakoforové barevné shody (ImplicitMillsDean).",
    slides: [
      {
        title: "1. Selhání 2D deskriptorů při 'Scaffold Hoppingu'",
        content: `Klasické 2D otisky (např. Morgan / ECFP4 fingerprinty) jsou založeny na topologickém procházení okolí atomů. Dvě molekuly s 완전히 odlišnými chemickými jádry (např. pyridinový kruh vs bicyklický indolový systém) mají 2D Tanimoto podobnost blízkou nule ($T_{2D} < 0.2$), ačkoliv v trojrozměrném prostoru zaujímají <strong>identický objem a stejné prostorové uspořádání farmakoforových skupin</strong>.
        <br><br>
        Biologický receptor nečte 2D graf ani SMILES řetězec — receptor interaguje výhradně s <strong>3D elektrostatickým a prostorovým tvarem molekuly</strong>. Metoda ROCS (Rapid Overlay of Chemical Structures) vyvinutá Grantem, Pickupem a Nichollsem (1996) umožňuje překonat omezení 2D grafů a realizovat tzv. <em>Scaffold Hopping</em> (nalezení zcela nových strukturních tříd ligandů se zachováním požadovaného 3D tvaru).`
      },
      {
        title: "2. Gaussovský model atomové hustoty a analytický objemový integrál",
        content: `Místo nespojitých pevných koulí (hard spheres) modeluje ROCS hustotu každého atomu $i$ trojrozměrnou Gaussovskou funkcí:
        <div class="math-card">
          $$\rho_i(\mathbf{r}) = p_i \exp\left(-\alpha_i |\mathbf{r} - \mathbf{r}_i|^2\right)$$
        </div>
        kde $\mathbf{r}_i$ je polohový vektor jádra atomu, $p_i$ je centrální hustota (standardně $p_i = 1$) a parametr $\alpha_i$ souvisí s Van der Waalsovým poloměrem atomu $R_i$ vztahem $\alpha_i = \frac{\pi}{R_i^2}$.
        <br><br>
        Celková hustota molekuly $A$ je dána součtem přes všechny její atomy: $\rho_A(\mathbf{r}) = \sum_{i \in A} \rho_i(\mathbf{r})$.
        <br><br>
        Zásadní matematickou výhodou je, že konvoluční integrál překryvu dvou Gaussovských funkcí má <strong>analytické řešení v uzavřeném tvaru</strong>:
        <div class="math-card">
          $$V(A, B) = \int \rho_A(\mathbf{r}) \rho_B(\mathbf{r}) d\mathbf{r} = \sum_{i \in A} \sum_{j \in B} p_i p_j \left( \frac{\pi}{\alpha_i + \alpha_j} \right)^{3/2} \exp\left( -\frac{\alpha_i \alpha_j}{\alpha_i + \alpha_j} |\mathbf{r}_i - \mathbf{r}_j|^2 \right)$$
        </div>
        Díky analytickému řešení netrpí výpočet žádnou diskretizační chybou mřížky a je o několik řádů rychlejší než numerická integrace.`
      },
      {
        title: "3. Metriky Shape Tanimoto, Color Tanimoto & TanimotoCombo",
        content: `Na základě objemových integrálů $V(A, B)$, $V(A, A)$ a $V(B, B)$ definujeme základní tvarové skóre:
        <br><br>
        <h4>1. Shape Tanimoto ($T_{shape} \in [0, 1]$)</h4>
        <div class="math-card">
          $$T_{shape}(A, B) = \frac{V(A, B)}{V(A, A) + V(B, B) - V(A, B)}$$
        </div>
        Hodnota $T_{shape} = 1.0$ značí dokonalý prostorový překryv všech atomů.
        <br><br>
        <h4>2. Color Tanimoto ($T_{color} \in [0, 1]$)</h4>
        Samotný tvar nestačí — hydrofobní naftalenový kruh má stejný tvar jako polární chinazolin, ale zcela odlišné vazebné vlastnosti. Proto se zavádějí tzv. <strong>Color Features (farmakoforová centra)</strong>:
        <ul>
          <li>Donory vodíkových vazeb (H-bond donors)</li>
          <li>Akceptory vodíkových vazeb (H-bond acceptors)</li>
          <li>Kladně nabitá centra (Cations)</li>
          <li>Záporně nabitá centra (Anions)</li>
          <li>Aromatické kruhy (Rings)</li>
          <li>Hydrofobní skupiny (Hydrophobes)</li>
        </ul>
        Farmakoforové překryvy jsou počítány podle silového pole <code>ImplicitMillsDean</code> pouze mezi centry stejného typu:
        <div class="math-card">
          $$T_{color}(A, B) = \frac{C(A, B)}{C(A, A) + C(B, B) - C(A, B)}$$
        </div>
        <br>
        <h4>3. TanimotoCombo ($T_{combo} \in [0, 2]$)</h4>
        Celkové kompozitní skóre, které v DrugEx slouží jako primární optimalizační cíl:
        <div class="math-card">
          $$T_{combo} = T_{shape} + T_{color}$$
        </div>`
      }
    ]
  },

  // =========================================================================
  // LECTURE 3.2
  // =========================================================================
  "l3_2": {
    id: "l3_2",
    tag: "Legendary",
    relevance: 10,
    title: "3.2 Výzva flexibilních cílů & Intrinsically Disordered Proteins (IDP)",
    summary: "Když krystalové struktury chybí: Ligand-based 3D strategie, ansámbly vazebných konformací a konsensuální supermolekuly.",
    slides: [
      {
        title: "1. Selhání strukturního dokování u flexibilních cílů a IDP",
        content: `Tradiční strukturový de novo design (Structure-Based Drug Design - SBDD) spoléhá na krystalografické struktury proteinů z Protein Data Bank (PDB) s vysokým rozlišením (< 2.0 Å) a rigidní vazebnou kavitu.
        <br><br>
        V moderní medicinální chemii však roste význam cílů, pro které rigidní struktura neexistuje:
        <ul>
          <li><strong>Intrinsicky neuspořádané proteiny (Intrinsically Disordered Proteins - IDP)</strong>: Proteiny nebo proteinové domény, které postrádají stabilní sekundární a terciární strukturu a v roztoku existují jako dynamický konformační ansámbl.</li>
          <li><strong>Vysoce flexibilní membránové receptory (GPCR, iontové kanály)</strong>: Kde vazba ligandu indukuje rozsáhlé alosterické konformační změny (Induced Fit).</li>
          <li><strong>Proteiny určené s nízkým rozlišením (Cryo-EM > 4 Å)</strong>: Kde chybí přesné atomární souřadnice postranních řetězců ve vazebném místě.</li>
        </ul>
        <br>
        Při pokusu o dokování do statického modelu IDP dochází k falešným předpovědím, protože protein při interakci s malou molekulou lokálně mění svou konformaci.`
      },
      {
        title: "2. Ligand-Based 3D strategie pro bakalářskou práci",
        content: `Pokud máme k dispozici sadu experimentálně potvrzených aktivních ligandů (získaných např. z NMR titrací, SPR nebo funkčních esejí), jejich 3D konformační obálka představuje <strong>negativní prostorový a farmakoforový otisk</strong> vazebného rozhraní proteinu.
        <br><br>
        Tato bakalářská práce rozvíjí metodiku, kde je 3D ROCS shape matching využit jako <strong>objektivní skórovací funkce v prostředí DrugEx MORL</strong>. Generátor tak navrhuje molekuly, které:
        <ol>
          <li>Perfektně vyplňují 3D prostorový objem aktivních referencí ($T_{shape} \to 1.0$).</li>
          <li>Prezentují klíčové farmakoforové skupiny ve správných geometrických vzdálenostech ($T_{color} \to 1.0$).</li>
          <li>Přinášejí nové chemické skelety s vysokou syntetickou dostupností ($\text{SAScore} \le 3.0$).</li>
        </ol>`
      },
      {
        title: "3. Konsensuální supermolekuly vs Multi-Reference Grouping",
        content: `V reálném systému může ligand vázat flexibilní cíl ve více konformačních stavech. DrugEx podporuje dva pokročilé přístupy:
        <br><br>
        <div class="compare-grid">
          <div class="compare-col left">
            <div class="compare-heading">1. Konsensuální Supermolekula (supermol_123.sdf)</div>
            Strukturní sloučení několika překrytých krystalových ligandů do jedné kompozitní šablony. Výsledná supermolekula definuje maximální povolený prostorový objem a kompletní soubor farmakoforových bodů všech známých vazačů.
          </div>
          <div class="compare-col right">
            <div class="compare-heading">2. Multi-Reference Grouping</div>
            Sada referenčních ligandů je rozdělena do pojmenovaných skupin (např. <code>{'pocket_A': [ref1, ref2], 'pocket_B': [ref3]}</code>). Skórovač počítá shodu s každou referencí a vrací maximální hodnotu:
            $$\text{Score}(X) = \max_{R \in \text{Group}} T_{combo}(X, R)$$
          </div>
        </div>`
      }
    ]
  },

  // =========================================================================
  // LECTURE 3.3
  // =========================================================================
  "l3_3": {
    id: "l3_3",
    tag: "Tricky",
    relevance: 10,
    title: "3.3 Hloubková analýza conformer_generators.py (RDKit, OMEGA, CDPKit)",
    summary: "Architektura generátorů konformací, stereoisomerní enumerace, rotační vazby, heavy atom limity a vláknová bezpečnost.",
    slides: [
      {
        title: "1. Architektura conformer_generators.py v DrugEx",
        content: `Generování nízkokonformačních 3D geometrií z 1D SMILES je výpočetně nejnáročnější částí 3D shape matchingu. Modul <code>drugex/training/scorers/conformer_generators.py</code> definuje abstraktní základní třídu <code>ConformerGenerator</code> se společným rozhraním:
        <div class="math-card">
          $$\text{generate\_conformers}(\text{smiles\_list}) \to \text{List}[\text{Mol\_with\_Conformers}]$$
        </div>
        <br>
        V DrugEx jsou k dispozici 4 specializované enginy:
        <ol>
          <li><strong>RDKitConformerGenerator</strong>: Využívá metodu <code>ETKDGv3</code> (Experimental-Torsion Distance Geometry s nelineární optimalizací). Plně otevřený a stabilní standard.</li>
          <li><strong>OmegaConformerGenerator</strong>: Využívá komerční nástroj OpenEye OMEGA. Provádí systematické prohledávání torzních úhlů na základě databáze fragmentů s možností GPU akcelerace.</li>
          <li><strong>CDPKitConformerGenerator</strong>: Využívá knihovnu CDPL (Chemical Data Processing Library). Generuje konformační ansámbl s RMSD shlukováním a nastavitelným energetickým oknem (<code>energy_window=20.0 kcal/mol</code>).</li>
          <li><strong>SchrodingerConformerGenerator</strong>: Spouští nástroj <code>confgenx</code> z balíku Schrödinger Suite s automatickou opravou souřadnic (+0.01 Å z-offset) pro prevenci chyb u dokonale planárních molekul.</li>
        </ol>`
      },
      {
        title: "2. Stereoisomerní enumerace a filtrace těžkých atomů",
        content: `Během de novo generování model často vygeneruje molekuly s nejednoznačně definovanou stereochemií na chirálních centrech. Pokud by generátor konformací vzal náhodnou orientaci, mohl by aktivní stereoizomer minout.
        <br><br>
        Třída <code>RDKitConformerGenerator</code> integruje automatickou enumeraci stereoizomerů:
        <ul>
          <li><code>max_isomers=4</code>: Pokud molekula obsahuje neoznačená chirální centra, vygeneruje se až $2^k$ stereoizomerů (omezeno stropem 4).</li>
          <li><code>max_conformers=50</code>: Pro každý stereoizomer se vygeneruje až 50 konformací metodou ETKDGv3.</li>
          <li><code>max_heavy_atoms=45</code>: Molekuly s více než 45 těžkými atomy (C, N, O, S...) jsou okamžitě přeskočeny a získávají skóre 0.0 (ochrana před zamrznutím výpočtu).</li>
          <li><code>max_rotatable_bonds=15</code>: Extrémně flexibilní molekuly s $> 15$ rotovatelnými vazbami mají obrovský konformační prostor, který nelze vzorkovat 50 konformacemi — jsou proto penalizovány.</li>
        </ul>`,
        code: `from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

# Produkční konfigurace pro RL trénink
conf_gen = RDKitConformerGenerator(
    max_conformers=50,       # Počet konformací na izomer
    max_isomers=4,           # Enumerace až 4 stereoizomerů
    max_heavy_atoms=45,      # Ochrana před obřími strukturami
    max_rotatable_bonds=15,  # Ochrana před vysoce flexibilními řetězci
    num_threads=1,           # 1 thread na proces při paralelním multiprocessing Poolu
    show_progress=False
)`
      }
    ]
  }
};
