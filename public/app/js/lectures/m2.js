/**
 * DrugEx Hub — Module 2: Multi-Objective Reinforcement Learning & Pareto Optimality
 * Comprehensive Handbook-Grade Textbook Materials for Bachelor Thesis
 * Faculty of Chemical Technology (VŠCHT Praha) / IOCB Prague (ÚOCHB AV ČR)
 */

export const M2_LECTURES = {
  // =========================================================================
  // LECTURE 2.1: Formulace MORL: Agent vs Mutate (Prior) & Policy Gradient
  // =========================================================================
  "l2_1": {
    id: "l2_1",
    tag: "Core",
    relevance: 10,
    title: "2.1 Formulace MORL: Agent vs Mutate (Prior) & Policy Gradient",
    summary: "Formulace generování molekul jako Markovova rozhodovacího procesu (MDP), matematické odvození algoritmu REINFORCE s baseline, prevence lipofilního kolapsu (Reward Hacking) a dual-network architektura s epsilon-greedy směrováním.",
    slides: [
      {
        title: "1. Formulace generování molekul jako Markovova rozhodovacího procesu (MDP)",
        content: `Generování chemické struktury atom po atomu (resp. token po tokenu v SMILES reprezentaci) lze exaktně modelovat jako diskrétní <strong>Markovův rozhodovací proces (Markov Decision Process - MDP)</strong> definovaný pěticí $(\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma)$:
        <br><br>
        <ul>
          <li><strong>Stavový prostor $\\mathcal{S}$</strong>: Stav $s_t = (x_1, x_2, \\dots, x_t) \\in \\mathcal{S}$ představuje částečně vygenerovaný prefix chemického řetězce v čase $t$. Počáteční stav je definován iniciačním tokenem $s_0 = \\langle \\text{GO} \\rangle$.</li>
          <li><strong>Prostor akcí $\\mathcal{A}$</strong>: Akce $a_t \\in \\mathcal{A} = \\mathcal{V}$ odpovídá výběru následujícího tokenu z chemického slovníku $\\mathcal{V}$ (atomy, náboje, stereochemie, závorky větvení, čísla cyklů nebo ukončovací token $\\langle \\text{EOS} \\rangle$).</li>
          <li><strong>Přechodová funkce $\\mathcal{P}$</strong>: Deterministický přechod do nového stavu $s_{t+1} = [s_t, a_t]$, kdy je vybraný token $a_t$ připojen na konec dosavadního řetězce.</li>
          <li><strong>Funkce odměny $\\mathcal{R}$ (Delayed Sparse Reward)</strong>: Během generování mezilehlých atomů je okamžitá odměna nulová ($r_t = 0$ pro $t \\lt T$). Terminální odměna $R(X) \\in [0, 1]$ je udělena teprve po vygenerování ukončovacího tokenu $\\langle \\text{EOS} \\rangle$, kdy je kompletní molekula $X$ předána prostředí <code>DrugExEnvironment</code> k chemoinformatickému a biologickému vyhodnocení.</li>
          <li><strong>Diskontní faktor $\\gamma = 1.0$</strong>: Vzhledem ke konečnému horizontu generování molekuly ($T \\le 100$) a nulovým mezilehlým odměnám se diskontování neuplatňuje.</li>
        </ul>`,
        alert: {
          type: "note",
          title: "Význam pro bakalářskou práci",
          text: "Zpožděná odměna (sparse reward) představuje hlavní výzvu: neuronová síť musí učinit desítky správných rozhodnutí (otevřít cyklus, správně umístit heteroatomy, uzavřít kruh), aniž by předem věděla, zda celá molekula získá vysoké 3D tvarové ROCS skóre!"
        }
      },
      {
        title: "2. Ztrátová funkce Policy Gradient & Matematické odvození REINFORCE",
        content: `Cílem trénování je nalézt optimální parametry $\\theta^*$ neuronového generátoru (politiky $\\pi_\\theta$), které maximalizují očekávanou kumulativní odměnu generovaných molekul:
        <div class="math-card">
          $$J(\\theta) = \\mathbb{E}_{X \\sim \\pi_\\theta} [R(X)] = \\sum_{X \\in \\mathcal{X}} P(X; \\theta) R(X)$$
        </div>
        kde $P(X; \\theta) = \\prod_{t=1}^{T} \\pi_\\theta(a_t \\mid s_t)$ je pravděpodobnost vygenerování celé sekvence $X$.
        <br><br>
        Protože prostor všech možných molekul $\\mathcal{X}$ je obrovský a odměna $R(X)$ je nespojitá (nediferencovatelná přes chemické editory), nelze gradient $\\nabla_\\theta J(\\theta)$ počítat analyticky. Aplikujeme <strong>Likelihood Ratio Trick (Log-Derivative Trick)</strong>:
        <div class="math-card">
          $$\\nabla_\\theta P(X; \\theta) = P(X; \\theta) \\frac{\\nabla_\\theta P(X; \\theta)}{P(X; \\theta)} = P(X; \\theta) \\nabla_\\theta \\log P(X; \\theta)$$
        </div>
        Dosazením do gradientu účelové funkce získáváme základní rovnici algoritmu <strong>REINFORCE</strong> (Williams, 1992):
        <div class="math-card">
          $$\\nabla_\\theta J(\\theta) = \\sum_{X} P(X; \\theta) \\nabla_\\theta \\log P(X; \\theta) R(X) = \\mathbb{E}_{X \\sim \\pi_\\theta} \\left[ \\nabla_\\theta \\log P(X; \\theta) \\cdot R(X) \\right]$$
        </div>
        <br>
        <h4>Zavedení baseline $\\beta$ pro redukci rozptylu</h4>
        Stochastický gradientní odhad trpí vysokým rozptylem (variance). V DrugEx zavádíme skalární <strong>baseline $\\beta$</strong> (standardně $\\beta = 0.0$ nebo klouzavý průměr odměn):
        <div class="math-card">
          $$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{X \\sim \\pi_\\theta} \\left[ \\sum_{t=1}^{T} \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) \\cdot \\left( R(X) - \\beta \\right) \\right]$$
        </div>
        Ztrátová funkce v PyTorch je pak implementována jako vážená záporná logaritmická věrohodnost: $\\mathcal{L}_{\\text{PG}}(\\theta) = - \\text{likelihood}(X) \\cdot (R(X) - \\beta)$.`,
        code: `import torch

# Implementace kroku Policy Gradient (REINFORCE) v SequenceExplorer
def compute_policy_gradient_loss(agent, seqs, rewards, beta=0.0):
    """
    Výpočet ztráty REINFORCE s odečtením baseline beta.
    seqs: LongTensor [batch_size, max_len]
    rewards: FloatTensor [batch_size, 1]
    """
    # 1. Výpočet log-likelihood sekvencí pod aktuální politikou agenta
    log_probs = agent.likelihood(seqs) # [batch_size, max_len]
    
    # 2. Vynásobení odměnou posunutou o baseline
    advantage = rewards - beta # [batch_size, 1]
    loss = - (log_probs * advantage).mean()
    
    return loss`
      },
      {
        title: "3. Rizika jednokriteriálního RL a fenomén 'Reward Hacking'",
        content: `Pokud bychom generátor optimalizovali pouze na jedno jediné biologické kritérium (např. maximalizaci predikované afinity z QSAR modelu nebo čistý 3D objemový překryv), generátor rychle nalezne patologické zkratky v reprezentaci (tzv. <strong>Reward Hacking</strong>):
        <br><br>
        Typické patologie jednokriteriálního učení v chemoinformatice:
        <ul>
          <li><strong>1. Lipofilní kolaps (Lipophilic Drift)</strong>:
          QSAR modely afinity často korelují s hydrofobicitou. Model začne generovat nekonečné alifatické řetězce a polycyklické aromatické uhlovodíky ($\\text{LogP} > 8.0$). Takové molekuly mají vysoké teoretické skóre, ale jsou zcela nerozpustné ve vodě a toxické.</li>
          <li><strong>2. Tvorba chemických monster (Molecular Weight Runaway)</strong>:
          Model navyšuje molekulovou hmotnost ($\\text{MW} > 900 \\text{ Da}$), protože obrovské molekuly nabízejí více kontaktních ploch pro nespecifické Van der Waalsovy interakce.</li>
          <li><strong>3. Syntetická neproveditelnost (Synthetic Infeasibility)</strong>:
          Vznik extrémně pnutých můstkových polycyklů, spiro-systémů a nestabilních peroxido-diazo vazeb, které žádný organický syntetik nedokáže připravit (SAScore $> 8.0$).</li>
        </ul>
        <br>
        Proto DrugEx striktně odmítá jednokriteriální optimalizaci a zavádí <strong>Multi-Objective Reinforcement Learning (MORL)</strong>, který balancuje 3D tvarovou shodu, selektivitu, fyzikální vlastnosti i syntetickou dostupnost.`,
        compare: {
          leftTitle: "Jednokriteriální RL (Pathological)",
          leftContent: `<ul>
            <li>Extrémní hodnoty LogP (> 8.0) a MW (> 800 Da).</li>
            <li>Nulová syntetická proveditelnost (SAScore > 7.0).</li>
            <li>Kolaps diverzity do jediné nefunkční struktury.</li>
            <li>Vysoké riziko falešně pozitivních virtuálních 'hitů'.</li>
          </ul>`,
          rightTitle: "Vícekriteriální MORL v DrugEx",
          rightContent: `<ul>
            <li>Harmonický kompromis (Pareto Frontier).</li>
            <li>Přísná kontrola MW (300–500 Da), LogP (1–4) a TPSA.</li>
            <li>Vysoká syntetická dostupnost (SAScore < 3.5).</li>
            <li>Vynikající 3D tvarový a farmakoforový překryv (ROCS).</li>
          </ul>`
        }
      },
      {
        title: "4. Dual-Network Architektura: Agent (pi_theta) vs Mutate / Prior (pi_0)",
        content: `Druhým fundamentálním úskalím posilovaného učení v diskrétním jazykovém prostoru je <strong>kolaps politiky (Mode Collapse / Policy Drift)</strong>. Jakmile agent náhodně objeví jednu sekvenci s vysokou odměnou, gradientní aktualizace prudce zvýší pravděpodobnost těchto tokenů a síť přestane prozkoumávat zbytek chemického prostoru.
        <br><br>
        DrugEx řeší tento problém unikátní <strong>duální architekturou dvou souběžných neuronových sítí</strong>:
        <br><br>
        <ol>
          <li><strong>Agent Network (Učící se generátor $\\pi_\\theta$)</strong>:
          Aktivní neuronová síť, jejíž váhy $\\theta$ jsou po každé epoše aktualizovány algoritmem Policy Gradient na základě Paretovských odměn. Zodpovídá za <em>exploataci</em> (hledání struktur s maximálním skóre).</li>
          <li><strong>Mutate / Prior Network (Fixní kotva $\\pi_0$)</strong>:
          Zmrazený model (jemně dotrénovaný checkpoint z Fáze 1.3). Je trvale přepnut do režimu <code>eval()</code> a jeho váhy nepodléhají žádným gradientovým úpravám. Zodpovídá za <em>exploraci</em> a udržování chemické gramatiky.</li>
        </ol>
        <br>
        <div class="math-card">
          $$\\text{Duální architektura}: \\quad \\pi_\\theta \\;\\text{(Učící se Agent)} \\quad \\longleftrightarrow \\quad \\pi_0 \\;\\text{(Fixní Prior / Mutate kotva)}$$
        </div>`
      },
      {
        title: "5. Epsilon-Greedy Policy Mixing a řízená chemická explorace",
        content: `Během generování každého tokenu $a_t$ v autoregresní smyčce (metoda <code>evolve</code> v <code>drugex/training/generators/sequence_rnn.py</code>) DrugEx stochasticky kombinuje predikce obou sítí pomocí parametru $\\epsilon \\in [0, 1]$ (standardně $\\epsilon = 0.2$):
        <br><br>
        V každém časovém kroku $t$ generátor vyhodnotí logity agenta $\\mathbf{z}_t^A$ i mutační sítě $\\mathbf{z}_t^M$ a převede je na pravděpodobnostní vektory:
        <div class="math-card">
          $$\\mathbf{p}_t^A = \\text{softmax}(\\mathbf{z}_t^A), \\qquad \\mathbf{p}_t^M = \\text{softmax}(\\mathbf{z}_t^M)$$
        </div>
        Následně je s pravděpodobností $\\epsilon$ vybrán krok mutační sítě:
        <div class="math-card">
          $$P(a_t \\mid s_t) = (1 - \\epsilon) \\pi_\\theta(a_t \\mid s_t) + \\epsilon \\pi_0(a_t \\mid s_t)$$
        </div>
        <br>
        <h4>Proč je hodnota $\\epsilon = 0.2$ optimální?</h4>
        <ul>
          <li>V $80\\%$ kroků model generuje podle učícího se agenta $\\pi_\\theta$, což žene populaci k vyššímu 3D tvarovému skóre a bioaktivitě.</li>
          <li>Ve $20\\%$ kroků model provede 'chemickou mutaci' řízenou obecným modelem $\\pi_0$. Tím dochází k zavádění nových substituentů, neobvyklých heterocyklů a bioisosterů, které brání uváznutí v lokálním optimu.</li>
        </ul>`,
        code: `import torch

# Ukázka interního fungování metody evolve() v SequenceRNN
def evolve_step(agent, mutate, x, hA, hM, epsilon=0.2):
    """
    Jeden krok stochastického míchání politik Agent vs Mutate.
    """
    # 1. Dopředný průchod agentem
    logitA, hA = agent(x, hA)
    proba = logitA.softmax(dim=-1)
    
    # 2. Dopředný průchod mutační sítí (bez výpočtu gradientů)
    if mutate is not None:
        with torch.no_grad():
            logitM, hM = mutate(x, hM)
            probaM = logitM.softmax(dim=-1)
        
        # 3. Epsilon-greedy maskování: náhodný výběr vzorků k mutaci
        batch_size = x.size(0)
        is_mutate = (torch.rand(batch_size, device=x.device) < epsilon)
        proba[is_mutate, :] = probaM[is_mutate, :]
        
    # 4. Vzorkování dalšího tokenu ze smíchané distribuce
    next_token = torch.multinomial(proba, num_samples=1).view(-1)
    return next_token, hA, hM`
      },
      {
        title: "6. Podpora Crover Network pro genetické křížení sekvencí v DrugEx",
        content: `Kromě mutační sítě umožňuje DrugEx zapojit také třetí neuronovou síť zvanou <strong>Crover Network (Křížící generátor $\\pi_C$)</strong>.
        <br><br>
        Crover síť reprezentuje hybridní evoluční přístup inspirovaný genetickými algoritmy:
        <ul>
          <li>Během tréninku je udržována elitní paměť molekul s nejvyšší Paretovskou odměnou z předchozích epoch.</li>
          <li>Crover síť je průběžně doučována na této elitní populaci.</li>
          <li>Při vzorkování se pravděpodobnostní rozdělení agenta a křížící sítě kombinuje s náhodným poměrem $r \\sim \\mathcal{U}(0, 1)$:
            <div class="math-card">
              $$\\mathbf{p}_t = r \\cdot \\mathbf{p}_t^A + (1 - r) \\cdot \\mathbf{p}_t^C$$
            </div>
          </li>
        </ul>
        Tento mechanismus simuluje genetické křížení (crossover) v prostoru latentních stavů neuronové sítě a umožňuje skokovou rekombinaci úspěšných farmakoforových fragmentů.`,
        alert: {
          type: "tip",
          title: "Doporučená konfigurace pro bakalářskou práci",
          text: "Pro stabilní trénink na cílech s 3D ROCS shape matchingem doporučujeme základní dvouramennou konfiguraci: Agent + Mutate (Prior) s epsilon = 0.2. Crover síť je volitelná a hodí se pro rozsáhlé vícedenní výpočetní experimenty."
        }
      },
      {
        title: "7. Stabilita tréninku, Gradient Clipping & Hyperparametry",
        content: `Trénování posilovaného učení s neuronovými generátory je náchylné k numerickým nestabilitám. V DrugEx jsou implementována následující ochranná opatření:
        <br><br>
        <ol>
          <li><strong>Ořezání normy gradientů (Gradient Clipping)</strong>:
          Maximální norma gradientů je omezena na $\\|\\mathbf{g}\\|_2 \\le 1.0$. Tím se eliminuje riziko explodujících gradientů při výskytu vzácné sekvence s extrémní chybou.</li>
          <li><strong>Velikost vzorkovacího batche ($N_{\\text{samples}} = 1000$)</strong>:
          V každé epoše model vygeneruje $1 000$ molekul. Dostatečně velká populace je nezbytná pro spolehlivý výpočet Paretovy fronty a Crowding Distance (malá populace $< 100$ vede k degenerovaným frontám).</li>
          <li><strong>Nízký learning rate ($lr = 10^{-4}$)</strong>:
          Optimalizátor Adam s konzervativním krokem zaručuje plynulý posun vah bez destrukce chemické gramatiky.</li>
          <li><strong>Patience a Early Stopping</strong>:
          Pokud se podíl žádoucích molekul (Desired Ratio) nezlepší po dobu 50 epoch, trénink je automaticky ukončen.</li>
        </ol>`
      },
      {
        title: "8. Praktická implementace: SequenceExplorer & Trénovací smyčka",
        content: `Následující ucelený Python skript demonstruje kompletní inicializaci a spuštění RL tréninku v DrugEx pomocí třídy <code>SequenceExplorer</code>:`,
        code: `import torch
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from drugex.training.explorers import SequenceExplorer
from drugex.training.monitors import FileMonitor

# 1. Načtení slovníku
voc = VocSmiles.fromFile("data/Papyrus05.5_smiles_voc.txt", encode_frags=False)

# 2. Inicializace Agenta (učící se model)
agent = SequenceRNN(voc, is_lstm=True, lr=1e-4)
agent.loadStatesFromFile("models/ccr2_finetuned_rnn.pkg")

# 3. Inicializace Prior / Mutate sítě (fixní model)
mutate = SequenceRNN(voc, is_lstm=True)
mutate.loadStatesFromFile("models/ccr2_finetuned_rnn.pkg")
mutate.eval() # Zmrazení parametrů!

# 4. Sestavení SequenceExploreru
explorer = SequenceExplorer(
    agent=agent,
    mutate=mutate,
    crover=None,
    env=env,               # DrugExEnvironment (definováno v Lekci 2.3)
    epsilon=0.2,           # 20% explorace z mutační sítě
    beta=0.0,              # Baseline pro policy gradient
    n_samples=1000,        # 1000 generovaných molekul na epochu
    batch_size=128
)

# 5. Spuštění MORL tréninku na 100 epoch
monitor = FileMonitor("logs/morl_ccr2_training")
print("Zahájení MORL optimalizace...")
explorer.fit(
    epochs=100,
    patience=30,
    monitor=monitor
)
print("✓ MORL trénink úspěšně dokončen!")`
      }
    ]
  },

  // =========================================================================
  // LECTURE 2.2: Paretova optimalita & Pareto Crowding Distance
  // =========================================================================
  "l2_2": {
    id: "l2_2",
    tag: "WOW",
    relevance: 10,
    title: "2.2 Paretova optimalita & Pareto Crowding Distance",
    summary: "Nedominované třídění řešení v multidimenzionálním prostoru cílů, matematická derivace algoritmu NSGA-II Crowding Distance pro zachování diverzity a přiřazení normalizovaných odměn.",
    slides: [
      {
        title: "1. Koncept vícekriteriální optimalizace v objevování léčiv",
        content: `Vývoj léčiva představuje ze své podstaty <strong>vysoce dimenzionální kompromisní optimalizaci</strong>. Ideální lékový kandidát musí současně splňovat celou řadu protichůdných požadavků:
        <ul>
          <li>Vysoká 3D tvarová a elektrostatická komplementarita k vazebnému místu cíle ($T_{\\text{combo}} > 1.2$).</li>
          <li>Vysoká syntetická dostupnost pro organickou přípravu v laboratoři ($\\text{SAScore} < 3.5$).</li>
          <li>Příznivé fyzikálně-chemické vlastnosti podle Lipinského pravidel ($\\text{MW} \\in [300, 500]$, $\\text{LogP} \\in [1.5, 4.0]$, $\\text{TPSA} \\in [40, 120]$).</li>
          <li>Nízká toxicita a selektivita vůči vedlejším cílům (např. iontovému kanálu hERG).</li>
        </ul>
        <br>
        <h4>Selhání jednoduchého váženého součtu (Weighted Sum Approach)</h4>
        Klasický přístup sčítající cíle do jedné skalární funkce $R = \\sum w_i f_i$ v praxi selhává:
        <ol>
          <li>Není schopen nalézt optimální řešení v nekonvexních (propadlých) oblastech prostoru vlastností.</li>
          <li>Vyžaduje subjektivní manuální ladění vah $w_i$, kde malá změna váhy může vést k úplnému potlačení jednoho z kritérií.</li>
        </ol>
        <br>
        DrugEx proto implementuje rigorózní <strong>Paretovskou optimalizaci bez nutnosti zadávání pevných vah</strong>.`
      },
      {
        title: "2. Matematická definice Paretovy dominance v multidimenzionálním prostoru",
        content: `Uvažujme $M$ objektivních skórovacích funkcí $\\mathbf{f}(X) = (f_1(X), f_2(X), \\dots, f_M(X))^T$, kde každou funkci $f_m: \\mathcal{X} \\to [0, 1]$ chceme maximalizovat.
        <br><br>
        Pro libovolné dvě molekuly $A, B \\in \\mathcal{X}$ definujeme vztah <strong>Paretovy dominance</strong> následovně:
        <div class="math-card">
          $$A \\succ B \\quad (A \\text{ dominuje } B) \\iff \\forall m \\in \\{1, \\dots, M\\}: f_m(A) \\ge f_m(B) \\quad \\land \\quad \\exists m \\in \\{1, \\dots, M\\}: f_m(A) > f_m(B)$$
        </div>
        To znamená: Molekula $A$ je ve všech sledovaných vlastnostech alespoň tak dobrá jako molekula $B$ a v alespoň jedné vlastnosti je striktně lepší.
        <br><br>
        Pokud $A \\not\\succ B$ a současně $B \\not\\succ A$, říkáme, že řešení jsou <strong>vzájemně nedominovaná ($A \\sim B$)</strong>. Žádné z nich nelze označit za jednoznačně lepší bez dodatečné subjektivní preference.
        <br><br>
        Množina všech nedominovaných řešení v celé populaci tvoří <strong>Paretovu frontu (Pareto Frontier)</strong>:
        <div class="math-card">
          $$\\mathcal{P}^* = \\{ X \\in \\mathcal{X} \\mid \\not\\exists X' \\in \\mathcal{X}: X' \\succ X \\}$$
        </div>`
      },
      {
        title: "3. Algoritmus nedominovaného třídění (Non-Dominated Sorting Layers)",
        content: `V každé trénovací epoše vygeneruje DrugEx populaci $N = 1000$ kandidátních molekul a vypočte matici skóre $\\mathbf{S} \\in \\mathbb{R}^{N \\times M}$.
        <br><br>
        Pomocí algoritmu <strong>Fast Non-Dominated Sorting (NSGA-II)</strong> je celá populace rozřazena do hierarchických vrstev (front) $\\mathcal{F}_1, \\mathcal{F}_2, \\dots, \\mathcal{F}_K$:
        <br><br>
        <ol>
          <li><strong>Vrstva 1 (Rank 1 Front $\\mathcal{F}_1$)</strong>:
          Množina všech molekul v populaci, které nejsou dominovány <em>žádnou jinou molekulou</em>. Tvoří aktuální nejlepší Paretovu frontu.</li>
          <li><strong>Vrstva 2 (Rank 2 Front $\\mathcal{F}_2$)</strong>:
          Dočasně odstraníme všechny molekuly z $\\mathcal{F}_1$. Znovu identifikujeme všechna nedominovaná řešení ze zbývající populace. Tato řešení tvoří druhou frontu $\\mathcal{F}_2$.</li>
          <li><strong>Iterativní rozklad</strong>:
          Proces se opakuje pro vrstvy $\\mathcal{F}_3, \\mathcal{F}_4, \\dots$, dokud není každé molekule $i$ v populaci přiřazena diskrétní hodnost $r_i = k$, kde $i \\in \\mathcal{F}_k$.</li>
        </ol>
        <br>
        Molekuly ve vrstvě $\\mathcal{F}_1$ mají nejvyšší možnou kvalitu, zatímco molekuly ve vyšších vrstvách jsou postupně horší.`,
        alert: {
          type: "important",
          title: "Výpočetní složitost třídění",
          text: "Standardní nedominované třídění má časovou složitost O(M · N^2). Pro populaci N = 1000 molekul a M = 4 kritéria trvá výpočet v NumPy pouhých několik milisekund na CPU!"
        }
      },
      {
        title: "4. Problém shlukování (Crowding) a potřeba diverzity v cílovém prostoru",
        content: `Samotné rozřazení do Paretovských vrstev má zásadní slabinu:
        <br><br>
        Všechny molekuly uvnitř stejné vrstvy (např. 150 molekul v Rank 1 $\\mathcal{F}_1$) mají stejnou dominanční hodnost. Pokud bychom všem těmto molekulám přiřadili stejnou odměnu:
        <ul>
          <li>Generativní model by rychle zkolaboval do jediného úzkého shluku na Paretově frontě (např. tam, kde je nejjednodušší dosáhnout vysokého skóre na úkor ostatních).</li>
          <li>Chemická rozmanitost generovaných kandidátů by prudce klesla.</li>
        </ul>
        <br>
        Aby model prozkoumával <strong>celou šíři Paretovy fronty</strong> a nabízel různorodé chemické kompromisy, zavádí DrugEx metriku <strong>Pareto Crowding Distance (Hustotní vzdálenost)</strong> podle slavného algoritmu NSGA-II (Deb et al., 2002).
        <br><br>
        Molekuly, které leží v řídce osídlených (izolovaných) oblastech Paretovy fronty, získávají vyšší odměnu než molekuly namačkané v hustém shluku.`
      },
      {
        title: "5. Matematická derivace Pareto Crowding Distance (NSGA-II)",
        content: `Výpočet Crowding Distance probíhá <strong>samostatně pro každou Paretovskou vrstvu $\\mathcal{F}_k$</strong>:
        <br><br>
        Nechť $n_k = |\\mathcal{F}_k|$ je počet molekul ve vrstvě $\\mathcal{F}_k$.
        <ol>
          <li>Pro každou molekulu $i \\in \\mathcal{F}_k$ inicializujeme vzdálenost: $d_i = 0$.</li>
          <li>Pro každé optimalizační kritérium $m \\in \\{1, 2, \\dots, M\\}$:
            <ul>
              <li>Seřadíme molekuly ve vrstvě $\\mathcal{F}_k$ vzestupně podle hodnoty kritéria $f_m$:
                $$I^{(m)} = \\text{argsort}(\\{f_m(i) \\mid i \\in \\mathcal{F}_k\\})$$
              </li>
              <li><strong>Hraničním řešením (Boundary Solutions)</strong> s minimální a maximální hodnotou kritéria přiřadíme nekonečnou vzdálenost, aby byla vždy zachována:
                <div class="math-card">
                  $$d(I_1^{(m)}) = \\infty, \\qquad d(I_{n_k}^{(m)}) = \\infty$$
                </div>
              </li>
              <li>Pro všechna vnitřní řešení $j \\in \\{2, 3, \\dots, n_k - 1\\}$ přičteme normalizovanou vzdálenost mezi sousedy:
                <div class="math-card">
                  $$d(I_j^{(m)}) \\mathrel{+}= \\frac{f_m(I_{j+1}^{(m)}) - f_m(I_{j-1}^{(m)})}{f_m^{\\max} - f_m^{\\min}}$$
                </div>
                kde $f_m^{\\max}$ a $f_m^{\\min}$ jsou maximální a minimální hodnota daného kritéria v celé vrstvě $\\mathcal{F}_k$.
              </li>
            </ul>
          </li>
          <li>Celková Crowding Distance pro molekulu $i$ je součet přes všech $M$ dimenzí: $d_i = \\sum_{m=1}^{M} d_i^{(m)}$.</li>
        </ol>`,
        alert: {
          type: "tip",
          title: "Geometrický význam",
          text: "Crowding distance d_i představuje obvod největšího hyper-kvádru v prostoru cílů, který obklopuje molekulu i a neobsahuje žádné jiné řešení z dané Paretovy vrstvy."
        }
      },
      {
        title: "6. Převod Paretových hodností a Crowding Distance na skalární odměnu R(X)",
        content: `Abychom mohli aplikovat Policy Gradient (REINFORCE), musíme Paretovské vrstvy a Crowding Distance převést na jediné spojité skalární číslo odměny $R(X) \\in [0, 1]$ pro každou molekulu:
        <br><br>
        V třídě <code>ParetoCrowdingDistance</code> (v souboru <code>drugex/training/rewards.py</code>) probíhá finální seřazení populace následujícím deterministickým postupem:
        <br><br>
        <ol>
          <li><strong>Uspořádání vrstev</strong>: Molekuly jsou primárně seřazeny podle indexu své Paretovské vrstvy (vrstva $\\mathcal{F}_1$ má přednost před $\\mathcal{F}_2$, ta před $\\mathcal{F}_3$, atd.).</li>
          <li><strong>Vnitřní uspořádání vrstvy podle diverzity</strong>: Uvnitř každé vrstvy $\\mathcal{F}_k$ jsou molekuly seřazeny <strong>sestupně podle Crowding Distance $d_i$</strong> (molekuly s $d_i = \\infty$ a nejvyšší diverzitou jsou na prvních místech).</li>
          <li><strong>Reverze pořadí</strong>: Zřetězený seznam indexů je invertován (<code>rank = rank[::-1]</code>), takže na indexu $0$ leží nejhorší molekula v populaci a na indexu $N-1$ leží nejlepší diverzifikovaná molekula z Rank 1.</li>
          <li><strong>Normalizace odměny</strong>:
            <div class="math-card">
              $$R(X_i) = \\frac{\\text{pořadí}(X_i)}{N} \\in [0, 1)$$
            </div>
          </li>
        </ol>
        Tímto způsobem získává každá molekula unikátní, spojitou a rovnoměrně rozdělenou odměnu, která perfektně řídí gradientní optimalizaci!`,
        code: `import numpy as np

# Ukázka výpočtu finální odměny v ParetoCrowdingDistance
def compute_pareto_rewards(fronts, scores):
    """
    fronts: list of arrays (indexy molekul v jednotlivých vrstvách F1, F2...)
    scores: ndarray [N, M] (matice skóre molekul)
    """
    ranked_fronts = []
    for front in fronts:
        front_scores = scores[front]
        front_size = len(front)
        crowding_distance = np.zeros(front_size)
        
        # Výpočet Crowding Distance pro každé kritérium
        for m in range(front_scores.shape[1]):
            sorted_idx = np.argsort(front_scores[:, m])
            crowding_distance[sorted_idx[0]] = np.inf
            crowding_distance[sorted_idx[-1]] = np.inf
            
            score_range = front_scores[:, m].max() - front_scores[:, m].min()
            if score_range > 0:
                for j in range(1, front_size - 1):
                    crowding_distance[sorted_idx[j]] += (
                        front_scores[sorted_idx[j+1], m] - front_scores[sorted_idx[j-1], m]
                    ) / score_range
                    
        # Seřazení uvnitř vrstvy sestupně podle crowding distance
        sorted_by_cd = np.argsort(-crowding_distance)
        ranked_fronts.append(front[sorted_by_cd])
        
    # Zřetězení a otočení: od nejhoršího (0) k nejlepšímu (N-1)
    full_rank = np.concatenate(ranked_fronts)[::-1]
    
    # Přiřazení normalizované odměny [0, 1)
    N = len(scores)
    rewards = np.zeros((N, 1))
    rewards[full_rank, 0] = np.arange(N) / N
    return rewards`
      },
      {
        title: "7. Alternativní Paretovská schémata: ParetoTanimotoDistance & Hypervolume",
        content: `Vedle standardního <code>ParetoCrowdingDistance</code> podporuje DrugEx i další pokročilá schémata odměňování:
        <br><br>
        <h4>1. ParetoTanimotoDistance</h4>
        Zatímco Crowding Distance měří vzdálenost v abstraktním prostoru vlastností $(f_1, f_2)$, <code>ParetoTanimotoDistance</code> měří diverzitu přímo v <strong>strukturním chemickém prostoru</strong>:
        <div class="math-card">
          $$d_{\\text{chem}}(i) = 1 - \\max_{j \\in \\mathcal{F}_k, j \\neq i} T_{\\text{Tanimoto}}(\\mathbf{fp}(i), \\mathbf{fp}(j))$$
        </div>
        Odměňuje molekuly, které mají odlišné chemické jádro (scaffold hopping) od ostatních molekul ve stejné Paretově vrstvě.
        <br><br>
        <h4>2. Hypervolume Indicator (HV)</h4>
        Měří celkový $M$-dimenzionální objem prostoru vlastností ohraničený Paretovou frontou $\\mathcal{P}^*$ a předem zvoleným referenčním bodem $\\mathbf{r}_{\\text{ref}} = (0, 0, \\dots, 0)^T$:
        <div class="math-card">
          $$\\text{HV}(\\mathcal{P}^*, \\mathbf{r}_{\\text{ref}}) = \\Lambda\\left( \\bigcup_{X \\in \\mathcal{P}^*} [\\mathbf{r}_{\\text{ref}}, \\mathbf{f}(X)] \\right)$$
        </div>
        Hypervolume je jedinou monomateriální metrikou, která striktně zachovává Paretovu dominanci (vyšší HV vždy značí lepší Paretovu frontu).`,
        compare: {
          leftTitle: "ParetoCrowdingDistance (Doporučeno)",
          leftContent: `<ul>
            <li>Extrémně rychlý výpočet ($O(M \\cdot N \\log N)$).</li>
            <li>Výborné rovnoměrné pokrytí celého spektra vlastností.</li>
            <li>Nativní standard v evolučním počítání (NSGA-II).</li>
          </ul>`,
          rightTitle: "ParetoTanimotoDistance",
          rightContent: `<ul>
            <li>Přímo nutí model objevovat nové chemické třídy (scaffolds).</li>
            <li>Vyžaduje výpočet matice Morganových otisků ($O(N^2)$).</li>
            <li>Skvělé pro projekty zaměřené na Scaffold Hopping u IDP.</li>
          </ul>`
        }
      },
      {
        title: "8. Praktická implementace: Výpočet Paretových front a odměn v Pythonu",
        content: `Následující plně funkční skript demonstruje výpočet Paretovy fronty a Crowding Distance pro syntetickou populaci 5 molekul se 2 protichůdnými cíli (3D Shape vs SAScore):`,
        code: `import numpy as np
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.utils import get_Pareto_fronts

# Simulace matice skóre pro 5 molekul a 2 cíle: [3D Shape TanimotoCombo, SAScore Desirability]
# Cíl 1: Tvarová shoda (vyšší = lepší)
# Cíl 2: Syntetická dostupnost (vyšší = lepší)
scores = np.array([
    [0.90, 0.30],  # Mol A: Vynikající tvar, horší syntéza
    [0.50, 0.95],  # Mol B: Horší tvar, triviální syntéza
    [0.85, 0.80],  # Mol C: Vyvážený špičkový kompromis
    [0.40, 0.40],  # Mol D: Dominována molekulou C
    [0.82, 0.78]   # Mol E: Téměř na frontě, dominována C
])

# 1. Výpočet Paretových vrstev
fronts = get_Pareto_fronts(scores)
print("Rozdělení do Paretovských vrstev:")
for k, front in enumerate(fronts, 1):
    print(f"  Front {k} (Rank {k}): indexy molekul {front}")

# 2. Výpočet finálních odměn přes ParetoCrowdingDistance
reward_scheme = ParetoCrowdingDistance()
rewards = reward_scheme(smiles=["A", "B", "C", "D", "E"], scores=scores, thresholds=[0.5, 0.5])

print("\\nPřiřazené normalizované odměny R(X):")
for i, name in enumerate(["Mol A", "Mol B", "Mol C", "Mol D", "Mol E"]):
    print(f"  {name} | Skóre: {scores[i]} | Odměna: {rewards[i, 0]:.3f}")`
      }
    ]
  },

  // =========================================================================
  // LECTURE 2.3: Skórovací funkce & Desirability Modifiers (SmoothClippedScore)
  // =========================================================================
  "l2_3": {
    id: "l2_3",
    tag: "Core",
    relevance: 9,
    title: "2.3 Skórovací funkce & Desirability Modifiers (SmoothClippedScore)",
    summary: "Architektura DrugExEnvironment, transformace fyzikálních parametrů na normalizované odměny [0, 1], syntetická dostupnost (SAScore, RAScore), QSAR prediktory a matematická odvození Desirability Modifiers.",
    slides: [
      {
        title: "1. Architektura DrugExEnvironment a role skórovačů",
        content: `Třída <code>DrugExEnvironment</code> (v souboru <code>drugex/training/environment.py</code>) představuje centrální hodnotící rozhraní (Environment) celého systému posilovaného učení.
        <br><br>
        Pipeline vyhodnocení generované batche molekul probíhá v následujících krocích:
        <ol>
          <li><strong>1. Syntaktická kontrola (<code>SmilesChecker</code>)</strong>:
          Vyhodnocení chemické validity generovaných SMILES řetězců pomocí RDKit. Nevalidním molekulám a prázdným řetězcům je přiřazen příznak <code>Valid = 0</code>.</li>
          <li><strong>2. Výpočet surových skóre (Raw Objective Scoring)</strong>:
          Paralelní vyhodnocení všech $M$ registrovaných skórovačů (3D ROCS shape matching, SAScore, QSAR modely, MW, LogP).</li>
          <li><strong>3. Aplikace modifikátorů žádoucnosti (Desirability Modifiers)</strong>:
          Každé surové fyzikální číslo (např. $\\text{MW} = 450 \\text{ Da}$ nebo $\\text{SAScore} = 2.8$) je transformováno spojitou funkcí na normalizovanou odměnu $S_m \\in [0, 1]$.</li>
          <li><strong>4. Nulování nevalidních struktur</strong>:
          Všem molekulám s <code>Valid == 0</code> jsou všechny hodnoty $S_m$ striktně přepsány na $0.0$.</li>
          <li><strong>5. Klasifikace žádoucnosti (<code>Desired</code>)</strong>:
          Molekula je označena jako žádoucí (<code>Desired = 1</code>) právě tehdy, když splňuje minimální prahové hodnoty všech dílčích cílů současně:
            <div class="math-card">
              $$\\text{Desired}(X) = \\prod_{m=1}^{M} \\mathbb{I}\\left( S_m(X) \\ge \\tau_m \\right)$$
            </div>
          </li>
        </ol>`,
        alert: {
          type: "note",
          title: "Metrika Desired Ratio",
          text: "Podíl žádoucích molekul (Desired Ratio = počet Desired / N_samples) je v DrugEx hlavní metrikou pro Early Stopping a sledování konvergence modelu."
        }
      },
      {
        title: "2. Přehled fyzikálně-chemických deskriptorů v DrugEx",
        content: `Třída <code>Property</code> (v <code>drugex/training/scorers/properties.py</code>) poskytuje přímé napojení na chemoinformatické deskriptory knihovny RDKit:
        <br><br>
        <table class="lecture-table" style="width:100%; border-collapse: collapse; font-size: 13px; margin: 10px 0;">
          <thead>
            <tr style="background: var(--editor); border-bottom: 2px solid var(--border);">
              <th style="padding: 8px; text-align: left;">Klíč</th>
              <th style="padding: 8px; text-align: left;">Fyzikální veličina</th>
              <th style="padding: 8px; text-align: left;">RDKit Implementace</th>
              <th style="padding: 8px; text-align: left;">Optimální lékový interval</th>
            </tr>
          </thead>
          <tbody>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>MW</code></td>
              <td style="padding: 8px;">Molekulová hmotnost [Da]</td>
              <td style="padding: 8px; font-family: var(--font-mono);">Descriptors.MolWt</td>
              <td style="padding: 8px; color: var(--bio-green);">300 – 500 Da</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>logP</code></td>
              <td style="padding: 8px;">Oktanol-voda rozdělovací koeficient</td>
              <td style="padding: 8px; font-family: var(--font-mono);">Crippen.MolLogP</td>
              <td style="padding: 8px; color: var(--bio-green);">1.5 – 4.0</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>HBA</code></td>
              <td style="padding: 8px;">Akceptory vodíkových vazeb (O, N)</td>
              <td style="padding: 8px; font-family: var(--font-mono);">AllChem.CalcNumLipinskiHBA</td>
              <td style="padding: 8px; color: var(--bio-green);">$\\le 10$</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>HBD</code></td>
              <td style="padding: 8px;">Donory vodíkových vazeb (OH, NH)</td>
              <td style="padding: 8px; font-family: var(--font-mono);">AllChem.CalcNumLipinskiHBD</td>
              <td style="padding: 8px; color: var(--bio-green);">$\\le 5$</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>TPSA</code></td>
              <td style="padding: 8px;">Topologický polární povrch [Å²]</td>
              <td style="padding: 8px; font-family: var(--font-mono);">AllChem.CalcTPSA</td>
              <td style="padding: 8px; color: var(--bio-green);">40 – 130 Å²</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>Rotable</code></td>
              <td style="padding: 8px;">Počet volně otočných vazeb</td>
              <td style="padding: 8px; font-family: var(--font-mono);">AllChem.CalcNumRotatableBonds</td>
              <td style="padding: 8px; color: var(--bio-green);">$\\le 8$</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-subtle);">
              <td style="padding: 8px; font-weight: 600;"><code>QED</code></td>
              <td style="padding: 8px;">Drug-likeness skóre (Bickerton)</td>
              <td style="padding: 8px; font-family: var(--font-mono);">Chem.QED.qed</td>
              <td style="padding: 8px; color: var(--bio-green);">> 0.6 (max 1.0)</td>
            </tr>
            <tr>
              <td style="padding: 8px; font-weight: 600;"><code>SA</code></td>
              <td style="padding: 8px;">Syntetická obtížnost (Ertl)</td>
              <td style="padding: 8px; font-family: var(--font-mono);">sascorer.calculateScore</td>
              <td style="padding: 8px; color: var(--bio-green);">< 3.5 (1 = snadné)</td>
            </tr>
          </tbody>
        </table>`
      },
      {
        title: "3. SAScore: Výpočet syntetické dostupnosti molekul",
        content: `Jedním z nejčastějších selhání de novo generátorů je návrh molekul, které sice mají fantastické virtuální skóre, ale jejichž syntéza by vyžadovala 40 kroků a byla by v praxi nemožná.
        <br><br>
        Metoda <strong>SAScore (Synthetic Accessibility Score)</strong> vyvinutá Peterem Ertlem a Ansgarem Schuffenhauerem (2009) modeluje syntetickou obtížnost molekuly kombinací dvou složek:
        <div class="math-card">
          $$\\text{SAScore} = \\text{FragmentScore} - \\text{ComplexityPenalty}$$
        </div>
        <br>
        <ol>
          <li><strong>Fragmentové skóre (Fragment Contribution)</strong>:
          Molekula je rozložena na cirkulární ECFP4 fragmenty (poloměr 2). Pro každý fragment je v databázi PubChem (obsahující ~13 milionů látek) zjištěna jeho statistická četnost. Často se vyskytující fragmenty (např. benzenový kruh, amidová vazba) získávají vysoké skóre snadnosti, zatímco vzácné fragmenty skóre snižují.</li>
          <li><strong>Penalizace topologické komplexity (Complexity Penalty)</strong>:
          Nelineární penalizace zohledňující strukturní rysy, které komplikují syntézu:
            <ul>
              <li>Počet stereocenter a asymetrických chirálních uhlíků.</li>
              <li>Přítomnost spiro atomů (dva cykly spojené jediným atomem).</li>
              <li>Můstkové atomy v bicyklických a polycyklických systémech (Bridgehead atoms).</li>
              <li>Makrocyklické kruhy s více než 8 atomy.</li>
              <li>Počet neobvyklých atomových kroužků a kondenzovaných systémů.</li>
            </ul>
          </li>
        </ol>
        Výsledný SAScore leží na škále <strong>1 (triviálně syntetizovatelné)</strong> až <strong>10 (extrémně náročná syntéza)</strong>.`,
        alert: {
          type: "important",
          title: "Transformace SAScore v DrugEx",
          text: "Protože v MORL maximalizujeme odměnu, ale u SAScore chceme nízkou hodnotu (snadnou syntézu), aplikujeme klesající modifikátor SmoothClippedScore(lower_x=4.5, upper_x=2.5)!"
        }
      },
      {
        title: "4. RAScore: Retrosyntetická proveditelnost pomocí strojového učení",
        content: `Zatímco SAScore hodnotí pouze lokální strukturní složitost, moderní alternativa <strong>RAScore (Retrosynthetic Accessibility Score)</strong> vyvinutá Schwallerem a Genhedenem (2021) hodnotí přímou <strong>retrosyntetickou proveditelnost</strong>:
        <br><br>
        <h4>Princip RAScore:</h4>
        <ul>
          <li>Model byl natrénován na více než <strong>200 000 kompletních retrosyntetických rozkladech</strong> provedených stromovým prohledávacím algoritmem MCTS v nástroji <em>AiZynthFinder</em>.</li>
          <li>Jako výchozí suroviny byly definovány komerčně dostupné chemikálie z katalogu ZINC / Enamine Building Blocks.</li>
          <li>Výstupem je pravděpodobnost $\\text{RAScore} \\in [0, 1]$, že molekulu lze úspěšně dekomponovat na komerčně dostupné bloky v <strong>maximálně 10 syntetických krocích</strong>.</li>
        </ul>
        <br>
        Molekula s $\\text{RAScore} > 0.8$ má vysokou pravděpodobnost, že ji organický chemik dokáže reálně nasyntetizovat z dostupných surovin během několika dní.`,
        code: `from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore

# Nastavení skórovače syntetické dostupnosti pro DrugExEnvironment
sa_scorer = Property(
    prop='SA',
    modifier=SmoothClippedScore(
        lower_x=4.5,  # Při SAScore >= 4.5 je odměna minimální (~0.0)
        upper_x=2.5,  # Při SAScore <= 2.5 je odměna maximální (~1.0)
        high_score=1.0,
        low_score=0.0
    )
)`
      },
      {
        title: "5. QSPRpred integrace: Prediktivní QSAR modely bioaktivity a selektivity",
        content: `DrugEx integruje knihovnu <strong>QSPRPred</strong> (Quantitative Structure-Property Relationship Prediction) pro rychlé zapojení modelů strojového učení predikujících biologickou aktivitu ($p\\text{IC}_{50}$, $pK_i$ nebo $-\\log_{10} \\text{EC}_{50}$):
        <br><br>
        Podporované modelové architektury v QSPRPred:
        <ul>
          <li><strong>Random Forest (RF)</strong> a <strong>Gradient Boosted Trees (LightGBM, XGBoost)</strong> na Morganových ECFP6 otiscích.</li>
          <li><strong>Support Vector Machines (SVM / SVR)</strong> s RBF jádrem.</li>
          <li><strong>Chemprop (Message Passing Neural Networks - MPNN)</strong> pro přímé hluboké učení na molekulárních grafech.</li>
        </ul>
        <br>
        <h4>Aplikabilní doména (Applicability Domain) & Konformní predikce</h4>
        Strojově učené modely mohou generovat nespolehlivé předpovědi, pokud generovaná molekula leží mimo trénovací chemický prostor. QSPRPred integruje <em>Conformal Prediction</em>, která ke každému odhadu afinity přiřazuje interval spolehlivosti. Pokud je nejistota predikce příliš vysoká, skóre je penalizováno.`,
        alert: {
          type: "tip",
          title: "Kombinace s 3D Shape Matchingem v bakalářské práci",
          text: "V této bakalářské práci je hlavní důraz kladen na 3D ROCS shape matching jako bezstrukturní geometrický cíl, který lze v DrugExEnvironment elegantně kombinovat s QSAR prediktorem selektivity proti vedlejším cílům!"
        }
      },
      {
        title: "6. Matematická odvození Desirability Modifiers v DrugEx",
        content: `Surové hodnoty z fyzikálních výpočtů a skórovačů mají různé jednotky a rozsahy (např. $T_{\\text{combo}} \\in [0, 2]$, $\\text{MW} \\in [100, 900]$, $\\text{SAScore} \\in [1, 10]$).
        <br><br>
        Modul <code>drugex/training/scorers/modifiers.py</code> implementuje rigorózní matematické transformace na interval odměn $[0, 1]$:
        <br><br>
        <h4>1. SmoothClippedScore (Hladká logistická transformace)</h4>
        Klasické oříznutí (ClippedScore) má nespojitou derivaci a nulový gradient mimo hranice, což způsobuje 'zamrznutí' posilovaného učení. <code>SmoothClippedScore</code> řeší tento problém pomocí logistické sigmoidální funkce se spojitou první derivací:
        <div class="math-card">
          $$S(x) = S_{\\text{low}} + \\frac{L}{1 + \\exp\\left( -k \\cdot (x - x_{\\text{mid}}) \\right)}$$
        </div>
        kde:
        $$k = \\frac{4}{x_{\\text{upper}} - x_{\\text{lower}}}, \\qquad x_{\\text{mid}} = \\frac{x_{\\text{upper}} + x_{\\text{lower}}}{2}, \\qquad L = S_{\\text{high}} - S_{\\text{low}}$$
        Faktor $k = 4 / \\Delta x$ zaručuje, že strmost logistické křivky v inflexním bodě $x_{\\text{mid}}$ přesně odpovídá směrnici lineárního ořezání.
        <br><br>
        <h4>2. MinMaxGaussian & Gaussian (Gaussovský zvon)</h4>
        Používá se pro veličiny, které mají optimální hodnotu uvnitř intervalu (např. $\\text{MW} = 420 \\pm 60 \\text{ Da}$):
        <div class="math-card">
          $$S(x) = \\exp\\left( -\\frac{1}{2} \\left( \\frac{x - \\mu}{\\sigma} \\right)^2 \\right)$$
        </div>
        Pro jednostrannou minimalizaci (<code>MinGaussian</code>) platí: $S(x) = 1.0$ pro $x \\le \\mu$ a klesá jako poloviční Gauss pro $x > \\mu$.`,
        code: `import numpy as np

# Matematická implementace SmoothClippedScore
class SmoothClippedScore:
    def __init__(self, upper_x: float, lower_x: float = 0.0, high_score: float = 1.0, low_score: float = 0.0):
        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score
        
        self.k = 4.0 / (upper_x - lower_x)
        self.middle_x = (upper_x + lower_x) / 2.0
        self.L = high_score - low_score
        
    def __call__(self, x):
        return self.low_score + self.L / (1.0 + np.exp(-self.k * (x - self.middle_x)))`
      },
      {
        title: "7. Kalibrace profilu odměn pro flexibilní cíle & IDP ligandy",
        content: `Správná volba prahů a modifikátorů definuje tzv. <strong>profil ideálního kandidáta (Target Candidate Profile)</strong>.
        <br><br>
        Doporučené nastavení parametrů v <code>DrugExEnvironment</code> pro bakalářskou práci na téma kombinace de novo designu a 3D shape matchingu:
        <br><br>
        <ul>
          <li><strong>3D ROCS Shape Matching (RDKitROCSScorer TanimotoCombo)</strong>:
          <code>SmoothClippedScore(lower_x=0.6, upper_x=1.2, low_score=0.0, high_score=1.0)</code>.
          Prahová hodnota: $\\tau = 0.5$ (odpovídá $T_{\\text{combo}} \\approx 0.9$).</li>
          <li><strong>Syntetická dostupnost (SAScore)</strong>:
          <code>SmoothClippedScore(lower_x=4.5, upper_x=2.5, low_score=0.0, high_score=1.0)</code>.
          Prahová hodnota: $\\tau = 0.5$ (odpovídá $\\text{SAScore} \\le 3.5$).</li>
          <li><strong>Molekulová hmotnost (MW)</strong>:
          <code>Gaussian(mu=420.0, sigma=60.0)</code>.
          Prahová hodnota: $\\tau = 0.5$ (rozsah MW 360–480 Da).</li>
          <li><strong>Lipofilita (LogP)</strong>:
          <code>MinGaussian(mu=2.8, sigma=1.2)</code>.
          Prahová hodnota: $\\tau = 0.5$ (penalizuje $\\text{LogP} > 4.0$).</li>
        </ul>`,
        alert: {
          type: "important",
          title: "Konzistence prahových hodnot",
          text: "Díky normalizaci přes modifikátory jsou všechny dílčí odměny škálovány do [0, 1]. Ve třídě DrugExEnvironment proto stačí nastavit jednotný práh thresholds = [0.5, 0.5, 0.5, 0.5]!"
        }
      },
      {
        title: "8. Kompletní kód: Sestavení DrugExEnvironment v Pythonu",
        content: `Následující plně funkční skript demonstruje kompletní sestavení prostředí <code>DrugExEnvironment</code> s více cíli, modifikátory a schématem <code>ParetoCrowdingDistance</code>:`,
        code: `from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore, Gaussian, MinMaxGaussian

# 1. Definice skórovačů s modifikátory
scorers = [
    # Cíl 1: SAScore (Syntetická dostupnost - minimalizace)
    Property(
        prop='SA',
        modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5)
    ),
    # Cíl 2: Molekulová hmotnost (MW - cílový Gaussovský zvon)
    Property(
        prop='MW',
        modifier=Gaussian(mu=420.0, sigma=60.0)
    ),
    # Cíl 3: Lipofilita (LogP - penalizace mastných kyselin)
    Property(
        prop='logP',
        modifier=MinMaxGaussian(mu=3.0, sigma=1.2, minimize=True)
    ),
    # Cíl 4: Drug-likeness (QED - maximalizace)
    Property(
        prop='QED',
        modifier=SmoothClippedScore(lower_x=0.4, upper_x=0.8)
    )
]

# 2. Prahové hodnoty pro klasifikaci 'Desired' (vše normalizováno na 0.5)
thresholds = [0.5, 0.5, 0.5, 0.5]

# 3. Paretovské schéma odměňování s Crowding Distance
reward_scheme = ParetoCrowdingDistance()

# 4. Inicializace finálního DrugExEnvironment
env = DrugExEnvironment(
    scorers=scorers,
    thresholds=thresholds,
    reward_scheme=reward_scheme
)

print(f"✓ DrugExEnvironment úspěšně inicializován s {len(scorers)} cíli.")
print(f"  Registrované skórovače: {[s.getKey() for s in scorers]}")`
      }
    ]
  }
};
