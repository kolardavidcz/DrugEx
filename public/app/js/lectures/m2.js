/**
 * DrugEx Hub — Module 2: Multi-Objective Reinforcement Learning & Pareto Optimality
 * Comprehensive Textbook-Grade Lecture Materials for Bachelor Thesis
 */

export const M2_LECTURES = {
  // =========================================================================
  // LECTURE 2.1
  // =========================================================================
  "l2_1": {
    id: "l2_1",
    tag: "Core",
    relevance: 10,
    title: "2.1 Formulace MORL: Agent vs Mutate (Prior) & Policy Gradient",
    summary: "Proč jednokriteriální RL vede k degradaci molekul, dual-network setup a epsilon-greedy směrování akcí.",
    slides: [
      {
        title: "1. Formulace generování molekul jako Markovova rozhodovacího procesu (MDP)",
        content: `Generování molekuly atom po atomu (resp. token po tokenu) lze přesně modelovat jako diskrétní <strong>Markovův rozhodovací proces (MDP)</strong> definovaný čtveřicí $(\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R})$:
        <ul>
          <li><strong>Stavový prostor $\\mathcal{S}$</strong>: Stav $s_t = (x_1, x_2, \\dots, x_t)$ představuje dosud vygenerovaný prefix SMILES sekvence v čase $t$. Počáteční stav je $s_0 = \langle \\text{START} \rangle$.</li>
          <li><strong>Prostor akcí $\\mathcal{A}$</strong>: Akce $a_t \\in \\mathcal{V}$ odpovídá výběru dalšího tokenu ze slovníku $\\mathcal{V}$ (atomy, vazby, cykly, stereochemie nebo token $\langle \\text{END} \rangle$).</li>
          <li><strong>Přechodová funkce $\\mathcal{P}$</strong>: Deterministický přechod do nového stavu $s_{t+1} = (s_t, a_t)$.</li>
          <li><strong>Funkce odměny $\\mathcal{R}$</strong>: Během generování mezilehlých tokenů je okamžitá odměna nulová ($r_t = 0$ pro $t < T$). Terminální odměna $R(X)$ je udělena teprve po vygenerování ukončovacího tokenu $\langle \\text{END} \rangle$, kdy je kompletní molekula $X$ podrobena chemoinformatickému vyhodnocení v prostředí <code>DrugExEnvironment</code>.</li>
        </ul>
        <br>
        <h4>Ztrátová funkce Policy Gradient (REINFORCE)</h4>
        Cílem je maximalizovat očekávanou odměnu $J(\\theta) = \\mathbb{E}_{X \sim \pi_\\theta} [R(X)]$. Gradient účelové funkce podle parametrů $\\theta$ generátoru je dán vztahem:
        <div class="math-card">
          $$\nabla_\\theta J(\\theta) = \\mathbb{E}_{X \sim \pi_\\theta} \\left[ \sum_{t=1}^{T} \nabla_\\theta \log \pi_\\theta(a_t \\mid s_t) \cdot \\left( R(X) - b \\right) \\right]$$
        </div>
        kde $b$ je tzv. <em>baseline</em> (např. klouzavý průměr odměn v předchozích batších), která snižuje rozptyl gradientového odhadu bez zavedení systematické chyby.`
      },
      {
        title: "2. Rizika jednokriteriálního RL a fenomén 'Reward Hacking'",
        content: `Pokud je generativní model optimalizován pouze na jedno jediné kritérium (např. maximalizaci predikované afinity QSAR modelu), generátor rychle nalezne patologické zkratky (tzv. <em>Reward Hacking</em>):
        <ul>
          <li><strong>Tvorba chemických monster</strong>: Model generuje molekuly s extrémní molekulovou hmotností ($\\text{MW} > 900 \\text{ Da}$), protože velké molekuly mají více možností pro nespecifické interakce.</li>
          <li><strong>Lipofilní kolaps</strong>: Generování nekonečných hydrofobních alifatických řetězců ($\\text{LogP} > 9$), které jsou nerozpustné ve vodě a toxické.</li>
          <li><strong>Syntetická neproveditelnost</strong>: Vznik energeticky pnutých spiro-cyklů a přemostěných systémů, které žádný chemik nedokáže syntetizovat.</li>
        </ul>
        <br>
        Proto je v DrugEx implementován <strong>Multi-Objective Reinforcement Learning (MORL)</strong>, který optimalizuje chemickou strukturu v kompromisním prostoru mnoha cílů současně.`
      },
      {
        title: "3. Duální síť: Agent vs Mutate (Prior) & Epsilon-Greedy směrování",
        content: `Dalším kritickým problémem při trénování neuronových generátorů je <em>mode collapse</em> (kolaps politiky do jediné struktury, která náhodou získala vysokou odměnu).
        <br><br>
        DrugEx řeší tento problém architekturou dvou souběžných neuronových sítí:
        <ol>
          <li><strong>Agent Network ($\pi_\\theta$)</strong>: Učící se generátor, jehož parametry $\\theta$ jsou aktualizovány gradientem odměny po každé epoše (fáze exploatace).</li>
          <li><strong>Mutate / Prior Network ($\pi_0$)</strong>: Fixní generátor (jemně dotrénovaný model z fáze Transfer Learningu), jehož váhy jsou zmrazené a nepodléhají aktualizaci.</li>
        </ol>
        <br>
        Při vzorkování každého tokenu $a_t$ v čase $t$ je pravděpodobnostní distribuce lineárně smíchána s parametrem $\\epsilon \\in [0, 1]$ (obvykle $\\epsilon = 0.2$):
        <div class="math-card">
          $$P(a_t \\mid s_t) = (1 - \\epsilon) \pi_\\theta(a_t \\mid s_t) + \\epsilon \pi_0(a_t \\mid s_t)$$
        </div>
        V $20\%$ případů tak model generuje kroky řízené stabilním obecným modelem, což udržuje vysokou strukturní diverzitu a brání zapomenutí chemické gramatiky.`,
        code: `from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles.fromFile("Papyrus05.5_smiles_voc.txt", encode_frags=False)

# 1. Učící se Agent (inicializovaný z fine-tuned modelu)
agent = SequenceRNN(voc, is_lstm=True)
agent.loadStatesFromFile("ccr2_finetuned.pkg")

# 2. Fixní Mutační síť (Prior)
mutate = SequenceRNN(voc, is_lstm=True)
mutate.loadStatesFromFile("ccr2_finetuned.pkg")
# DŮLEŽITÉ: Mutate síť zůstává ve stavu eval() a její gradienty jsou vypnuté
mutate.eval()

# 3. Inicializace explorátoru s duálním vzorkováním
explorer = SequenceExplorer(
    agent=agent,
    mutate=mutate,
    crover=None,
    env=env,
    epsilon=0.2,       # 20% explorace z mutační sítě
    n_samples=1000     # 1000 generovaných molekul na epochu
)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 2.2
  // =========================================================================
  "l2_2": {
    id: "l2_2",
    tag: "WOW",
    relevance: 10,
    title: "2.2 Paretova optimalita & Pareto Crowding Distance",
    summary: "Nedominované třídění řešení v multidimenzionálním prostoru cílů a algoritmus výpočtu Crowding Distance pro zachování diverzity.",
    slides: [
      {
        title: "1. Matematická definice Paretovy dominance v chemickém prostoru",
        content: `Při navrhování léčivých látek chceme současně optimalizovat vektor $M$ cílů $\\mathbf{f}(X) = (f_1(X), f_2(X), \\dots, f_M(X))$, kde $f_m(X) \\in [0, 1]$ představuje normalizovanou odměnu za $m$-tou vlastnost (např. 3D ROCS tvarová shoda, afinita k receptoru, syntetická dostupnost SAScore).
        <br><br>
        Vektor řešení $\\mathbf{f}(A)$ <strong>dominuje</strong> $\\mathbf{f}(B)$ (značeno $A \\succ B$), právě když:
        <div class="math-card">
          $$\\forall m \\in \{1, \\dots, M\}: f_m(A) \\ge f_m(B) \\quad \\land \\quad \\exists k \\in \{1, \\dots, M\}: f_k(A) > f_k(B)$$
        </div>
        To znamená, že molekula $A$ není v žádné vlastnosti horší než molekula $B$ a v alespoň jedné vlastnosti je striktně lepší.
        <br><br>
        Molekuly, které nejsou dominovány žádnou jinou molekulou v aktuální populaci, tvoří <strong>První Paretovu frontu (Rank 1 / Pareto Frontier)</strong>. Po jejich vyjmutí tvoří nedominované molekuly zbylé množiny Druhou Paretovu frontu (Rank 2), a tak dále.`
      },
      {
        title: "2. Algoritmus Pareto Crowding Distance",
        content: `Samotné Paretovo hodnocení (Rank) má zásadní nevýhodu: všechny molekuly na Rank 1 frontě mají stejné postavení. Pokud by model dostával odměnu pouze za Rank, shlukl by se do jediného snadno dosažitelného bodu na frontě a ztratil diverzitu.
        <br><br>
        DrugEx (Liu et al., 2021) integruje výpočet <strong>Crowding Distance (vzdálenosti shlukování)</strong> inspirovaný genetickým algoritmem NSGA-II:
        <br><br>
        Pro každou frontu $\\mathcal{F}_k$:
        <ol>
          <li>Pro každý cíl $m \\in \{1, \\dots, M\}$ seřadíme molekuly vzestupně podle hodnoty $f_m$.</li>
          <li>Krajním bodům (s minimální a maximální hodnotou $f_m$) přiřadíme nekonečnou vzdálenost: $d_1 = d_{|\\mathcal{F}_k|} = \infty$.</li>
          <li>Pro všechny vnitřní body $i \\in \{2, \\dots, |\\mathcal{F}_k| - 1\}$ přičteme normalizovanou vzdálenost jejich sousedů:
            <div class="math-card">
              $$d_i = \sum_{m=1}^{M} \\frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$$
            </div>
          </li>
        </ol>
        Konečná odměna molekuly je pak funkcí jejího Paretova ranku a Crowding Distance:
        <div class="math-card">
          $$R(X_i) = \\frac{1}{\\text{Rank}(X_i)} + \\frac{d_i}{1 + d_i}$$
        </div>
        Molekuly na okrajích fronty a v řídce osídlených regionech získávají nejvyšší odměnu, což stimuluje generátor k objevování netradičních chemických scaffoldů.`,
        code: `from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.environment import DrugExEnvironment

# Inicializace schématu Pareto Crowding Distance
reward_scheme = ParetoCrowdingDistance()

# Vytvoření prostředí DrugEx s dělícími prahy
env = DrugExEnvironment(
    scorers=[rocs_scorer, sa_scorer],
    thresholds=[0.871, 0.1],          # Prahové hodnoty pro klasifikaci Desired/Undesired
    reward_scheme=reward_scheme
)`
      }
    ]
  },

  // =========================================================================
  // LECTURE 2.3
  // =========================================================================
  "l2_3": {
    id: "l2_3",
    tag: "Core",
    relevance: 9,
    title: "2.3 Skórovací funkce & Desirability Modifiers (SmoothClippedScore)",
    summary: "Transformace fyzikálních parametrů na normalizované odměny [0, 1], syntetická dostupnost (SAScore, RAScore) a QSAR prediktory.",
    slides: [
      {
        title: "1. Přehled skórovacích funkcí v DrugEx",
        content: `V DrugEx lze jako cíle v prostředí <code>DrugExEnvironment</code> definovat libovolné chemoinformatické a biofyzikální deskriptory:
        <ul>
          <li><strong>Fyzikálně-chemické vlastnosti (<code>Property</code>)</strong>:
            Molekulová hmotnost (<code>MW</code>), rozdělovací koeficient (<code>LOGP</code>), počet donorů H-vazeb (<code>HBD</code>), počet akceptorů H-vazeb (<code>HBA</code>), polární povrch (<code>TPSA</code>), počet rotovatelných vazeb (<code>ROTB</code>), odhad lékovosti (<code>QED</code>).
          </li>
          <li><strong>Syntetická dostupnost (Synthetic Accessibility - SAScore)</strong>:
            Algoritmus podle Ertla a Schuffenhauera (2009). Kombinuje příspěvky fragmentů z databáze PubChem s nelineární penalizací za strukturní složitost (chirální centra, makrocykly, spiro-atomy). Škála 1.0 (velmi snadná syntéza) až 10.0 (extrémně obtížná).
          </li>
          <li><strong>Retrosyntetická přístupnost (RAScore)</strong>:
            Hluboký model vytrénovaný na milionech syntetických cest z nástroje AiZynthFinder, který predikuje pravděpodobnost ($0.0 \\dots 1.0$), že danou molekulu lze syntetizovat z komerčních stavebních bloků.
          </li>
          <li><strong>Bioaktivní QSAR modely</strong>:
            Modely strojového učení (Random Forest, SVM, DNN) vytrénované v balíčku <code>QSPRpred</code> na experimentálních datech $pIC_{50}$ nebo $pK_i$.
          </li>
        </ul>`
      },
      {
        title: "2. Matematické odvození Desirability Modifiers",
        content: `Aby bylo možné různorodé fyzikální vlastnosti kombinovat v Paretově prostředí, musí být transformovány na normalizovaný interval $[0, 1]$ pomocí modifikátorů (v <code>drugex/training/scorers/modifiers.py</code>):
        <br><br>
        <h4>1. SmoothClippedScore (Hladká kosínová sigmoidální transformace)</h4>
        <div class="math-card">
          $$S(x) = \\begin{cases} 
          1.0 & x \\le \\text{upper\_x} \\ 
          0.5 \\left( 1 + \cos\\left( \pi \\frac{x - \\text{upper\_x}}{\\text{lower\_x} - \\text{upper\_x}} \\right) \\right) & \\text{upper\_x} < x < \\text{lower\_x} \\ 
          0.0 & x \\ge \\text{lower\_x} 
          \\end{cases}$$
        </div>
        Výhodou <code>SmoothClippedScore</code> oproti skokovým prahům je spojitost první derivace, což poskytuje stabilní gradient pro zpětnovazební učení v přechodové oblasti.
        <br><br>
        <h4>2. ClippedScore (Lineární oříznutí)</h4>
        Lineární náběh od 0.0 do 1.0 mezi $\\text{lower\_x}$ a $\\text{upper\_x}$.
        <br><br>
        <h4>3. MinMaxScore & NormScore</h4>
        Normalizace na základě minimální a maximální pozorované hodnoty nebo centrovaná Gaussovská křivka kolem cílového optima $\mu$ se směrodatnou odchylkou $\\sigma$.`,
        code: `from drugex.training.scorers.modifiers import SmoothClippedScore, ClippedScore
from drugex.training.scorers.properties import Property

# 1. SAScore: hodnoty pod 3.0 jsou ideální (r=1.0), nad 5.0 penalizovány na 0.0
sa_scorer = Property('SA')
sa_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=3.0))

# 2. Molekulová hmotnost (MW): optimální rozsah 300 až 500 Da
mw_scorer = Property('MW')
mw_scorer.setModifier(ClippedScore(lower_x=300, upper_x=500))

# 3. LogP: penalizace vysoce hydrofobních látek (LogP > 5.0)
logp_scorer = Property('LOGP')
logp_scorer.setModifier(SmoothClippedScore(lower_x=5.0, upper_x=2.5))`
      }
    ]
  }
};
