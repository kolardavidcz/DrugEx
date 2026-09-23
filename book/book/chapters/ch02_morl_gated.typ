#import "../nature_theme.typ": *

= 2. Multi-Objective Reinforcement Learning (MORL) & Architektura QSPRPred

== 2.1 Formulace generování molekul jako Markovova rozhodovacího procesu

Generování chemické struktury token po tokenu v jazykové reprezentaci SMILES lze exaktně formalizovat jako diskrétní *Markovův rozhodovací proces (Markov Decision Process — MDP)* definovaný pěticí $(cal(S), cal(A), cal(P), cal(R), gamma)$:

1. *Stavový prostor $cal(S)$*: Stav $s_t = (x_1, x_2, dots, x_t) in cal(S)$ představuje dosud vygenerovaný prefix chemického řetězce v čase $t$. Počáteční stav je determinován iniciačním řídicím tokenem $s_0 = "<GO>"$.
2. *Prostor akcí $cal(A)$*: Akce $a_t in cal(A) = cal(V)$ odpovídá volbě následujícího tokenu z chemického slovníku $cal(V)$ (chemický symbol, stereochemické označení `@`, větvicí závorka, index cyklu či ukončovací token $"<EOS>"$).
3. *Přechodová funkce $cal(P)$*: Deterministické zřetězení $s_(t+1) = [s_t, a_t]$, kdy je vybraný token připojen na konec řetězce.
4. *Funkce odměny $cal(R)$ (Delayed Sparse Reward)*: Všechny přechody mezi mezilehlými atomy mají nulovou okamžitou odměnu ($r_t = 0$ pro $t < T$). Nenulová skalární odměna $R(X) in [0, 1]$ je udělena teprve po vygenerování ukončovacího tokenu $"<EOS>"$, kdy je kompletní molekula $X$ odeslána do hodnoticího prostředí `DrugExEnvironment`.
5. *Diskontní faktor $gamma = 1.0$*: Vzhledem ke konečné délce sekvence ($T <= 100$) a nulovým mezilehlým odměnám se diskontování neuplatňuje.

#pitfall_box("Riziko Reward Hackingu v Single-Objective RL", [
  Optimalizace generátoru na jedinou izolovanou metriku (např. pouhou maximalizaci afinity z QSAR modelu) vede k okamžité patologické degeneraci chemického prostoru:
  - *Lipofilní kolaps (Lipophilic Drift)*: Model začne generovat nekonečné alifatické řetězce a polyaromatické uhlovodíky ($"LogP" > 8.0$), které vykazují vysoké teoretické skóre, ale jsou zcela nerozpustné a toxické.
  - *Expanze molekulové hmotnosti (MW Runaway)*: Nekontrolovaný růst ($"MW" > 800 "Da"$), protože objemné struktury maximalizují nespecifické Van der Waalsovy kontakty.
  - *Syntetická nedostupnost*: Tvorba extrémně pnutých můstkových a spiro-cyklů s nestabilními peroxido-diazo vazbami ($"SAScore" > 7.0$).
  Proto DrugEx striktně vyžaduje Multi-Objective Reinforcement Learning (*MORL*).
])

== 2.2 Matematické odvození Policy Gradientu (REINFORCE) s baseline

Cílem trénování je nalézt parametry $theta^*$ neuronového generátoru (politiky $pi_theta$), které maximalizují očekávanou kumulativní odměnu generovaných molekul:

$ J(theta) = bb(E)_(X ~ pi_theta) [R(X)] = sum_(X in cal(X)) P(X; theta) R(X) $

kde $P(X; theta) = product_(t=1)^T pi_theta(a_t mid s_t)$ je pravděpodobnost vygenerování celé sekvence $X$.

Protože prostor všech možných molekul $cal(X)$ je diskrétní a odměna $R(X)$ je nespojitá (nediferencovatelná přes chemické editory a orákula), nelze gradient $nabla_theta J(theta)$ počítat analyticky. Aplikujeme *Log-Derivative Trick (Likelihood Ratio)*:

$ nabla_theta P(X; theta) = P(X; theta) frac(nabla_theta P(X; theta), P(X; theta)) = P(X; theta) nabla_theta log P(X; theta) $

Dosazením do gradientu účelové funkce získáváme základní formulaci algoritmu *REINFORCE* (Williams, 1992):

$
nabla_theta J(theta) &= sum_(X in cal(X)) P(X; theta) nabla_theta log P(X; theta) R(X) \
&= bb(E)_(X ~ pi_theta) [ nabla_theta log P(X; theta) dot R(X) ] \
&= bb(E)_(X ~ pi_theta) [ sum_(t=1)^T nabla_theta log pi_theta(a_t mid s_t) dot R(X) ]
$

=== Zavedení referenční baseline ze sítě Mutate/Prior ($pi_0$)

Stochastický odhad gradientu trpí vysokým rozptylem (rozptyl Monte Carlo vzorkování). V DrugEx zavádíme do rovnice odečtení referenční základní linie $R_("baseline")(X)$ generované zmrazenou sítí Mutate/Prior ($pi_0$):

$ nabla_theta J(theta) = bb(E)_(X ~ pi_theta) [ sum_(t=1)^T nabla_theta log pi_theta(a_t mid s_t) dot (R(X) - R_("baseline")(X)) ] $

#insight_box("Proč baseline nevnáší zkreslení (Unbiased Estimator)?", [
  Odečtení libovolné funkce $B(s_t)$, která nezávisí na právě zvolené akci $a_t$, nezmění střední hodnotu gradientu:
  
  $
  bb(E)_(a_t ~ pi_theta) [ nabla_theta log pi_theta(a_t mid s_t) dot B(s_t) ] &= sum_(a_t) pi_theta(a_t mid s_t) frac(nabla_theta pi_theta(a_t mid s_t), pi_theta(a_t mid s_t)) B(s_t) \
  &= B(s_t) nabla_theta sum_(a_t) pi_theta(a_t mid s_t) = B(s_t) nabla_theta (1) = 0
  $
  
  Člen $A(X) = R(X) - R_("baseline")(X)$ představuje *výhodu (Advantage)*:
  - Pokud molekula $X$ dosáhne vyšší odměny než referenční hodnota ($A(X) > 0$), gradientní krok posílí pravděpodobnost této trajektorie tokenů.
  - Pokud molekula získá nižší odměnu ($A(X) < 0$), model je aktivně penalizován a pravděpodobnost sekvence klesá, i když samotné $R(X)$ bylo kladné.
])

== 2.3 Dual-Network Architektura: Učící se Agent ($pi_theta$) vs. Mutate Prior ($pi_0$)

Při Reinforcement Learning v diskrétním jazykovém prostoru hrozí závažný jev zvaný *kolaps modů (Mode Collapse)*: jakmile agent náhodně objeví úzkou chemickou třídu s vysokou odměnou, gradienty prudce navýší pravděpodobnost těchto tokenů a síť přestane prozkoumávat zbytek chemického prostoru.

DrugEx řeší tento problém unikátní *duální architekturou dvou souběžných neuronových sítí*:
1. *Učící se Agent ($pi_theta$)*: Optimalizovaný model adaptující své parametry ve směru maximální Paretovské odměny (exploatace).
2. *Mutační Prior ($pi_0$)*: Zmrazený model z Fáze 1.3 (předtrénovaný na Papyrus a adaptovaný na cíl). Běží trvale v režimu `eval()` bez výpočtu gradientů a reprezentuje chemickou kotvu stability (explorace).

#telemetry_box("Epsilon-Greedy směrování politik (Policy Mixing)", [
  V každém časovém kroku generování tokenu $t$ DrugEx stochasticky kombinuje predikce obou sítí:
  
  $ P(a_t mid s_t) = (1 - epsilon) pi_theta(a_t mid s_t) + epsilon pi_0(a_t mid s_t) $
  
  Standardní hodnota $epsilon = 0.20$ zajišťuje, že v 80 % kroků model exploatuje naučený gradient tvaru a bioaktivity, zatímco ve 20 % kroků provede náhodnou chemickou mutaci řízenou obecným jazykovým modelem Papyrus. To garantuje > 90% unikátnost generovaných struktur po celých 100 epoch.
])

== 2.4 Paretova optimalita a nedominované třídění (NSGA-II)

Uvažujme $M$ dílčích objektivních funkcí $bold(f)(X) = (f_1(X), f_2(X), dots, f_M(X))^T$, kde každou funkci $f_m: cal(X) -> [0, 1]$ chceme maximalizovat.

Pro dvě molekuly $A, B in cal(X)$ definujeme relaci *Paretovy dominance*:

$ A succ B quad (A "dominuje" B) <==> forall m in {1, dots, M}: f_m(A) >= f_m(B) and exists m in {1, dots, M}: f_m(A) > f_m(B) $

Množina všech vzájemně nedominovaných řešení v celé populaci tvoří *Paretovu frontu (Pareto Frontier)*:

$ cal(P)^* = { X in cal(X) mid not exists X' in cal(X): X' succ X } $

#figure(
  image("../figures/fig_pareto_front.svg", width: 95%),
  caption: [Paretova fronta a výpočet Pareto Crowding Distance v DrugEx. Populace molekul je rozřazena do hierarchických vrstev nedominovaných řešení (Fronta 1, Fronta 2, Fronta 3). Uvnitř Fronty 1 získávají hraniční řešení (tvarový a syntetický extrém) nekonečnou vzdálenost shlukování, což zabraňuje kolapsu populace do úzkého středu.]
)

=== Algoritmus Pareto Crowding Distance

Aby model nesplaskl do jediného bodu na Paretově frontě, počítá třída `ParetoCrowdingDistance` (`drugex/training/rewards.py`) hustotní vzdálenost pro každou vrstvu $cal(F)_k$:

1. Pro každou molekulu $i in cal(F)_k$ inicializujeme $d_i = 0$.
2. Pro každé kritérium $m in {1, dots, M}$ seřadíme molekuly vzestupně podle skóre $f_m$.
3. Hraničním řešením přiřadíme nekonečnou vzdálenost: $d(I_1^((m))) = infinity$, $d(I_(n_k)^((m))) = infinity$.
4. Pro vnitřní řešení přičteme normalizovanou vzdálenost sousedů:
  $ d(I_j^((m))) arrow.l d(I_j^((m))) + frac(f_m(I_(j+1)^((m))) - f_m(I_(j-1)^((m))), f_m^("max") - f_m^("min")) $
5. Výsledná spojitá odměna pro molekulu na globální pozici $"rank"(X_i) in {0, dots, N-1}$ je:
  $ R(X_i) = frac("rank"(X_i), N) in [0, 1) $

== 2.4 ParetoTanimotoDistance (PTD): Odměňování strukturní novosti pro flexibilní cíle

Zatímco klasická `ParetoCrowdingDistance` (PCD) měří hustotu řešení čistě v prostoru vypočtených vlastností $bold(f)(X) in RR^M$, třída `ParetoTanimotoDistance` (`drugex/training/rewards.py`) zavádí měření diverzity přímo v *chemickém strukturním prostoru*:

1. Pro každou molekulu $i$ uvnitř stejné Paretovy fronty $cal(F)_k$ je vypočten 2048-bitový otisk Morgan ECFP6: $bold(v)_i = "calc_fps"(X_i)$.
2. Chemická vzdálenost $d_("chem")(i, j)$ mezi molekulami je definována jako doplněk Tanimotova koeficientu:
   $ d_("chem")(i, j) = 1 - frac(bold(v)_i dot bold(v)_j, norm(bold(v)_i)_1 + norm(bold(v)_j)_1 - bold(v)_i dot bold(v)_j) $
3. Výsledné skóre strukturní řídkosti pro molekulu $i$ je dáno minimální (nebo průměrnou) vzdáleností k ostatním členům fronty:
   $ D_("PTD")(i) = min_(j in cal(F)_k, j eq.not i) d_("chem")(i, j) $

#figure_card(
  image("../figures/morl_explorer.png", width: 100%),
  caption: [DrugEx MORL Explorer: Interaktivní simulátor nedominovaného řazení a Paretovských front. V dvourozměrném prostoru vlastností (3D ROCS Shape vs. SAScore) získává Spiro-Oxazol Bioisoster B díky metrice ParetoTanimotoDistance (PTD) maximální odměnu a Advantage vystřelí vzhůru (+0.171), což posiluje politiku směrem ke strukturně novému scaffold hoppingu.]
)

#insight_box("Význam PTD pro bakalářskou práci", [
  Pokud generátor objeví molekulu se zcela novým scaffoldem (např. spiro-cyklus namísto plochého benzenového jádra), její Tanimoto podobnost k ostatním molekulám v populaci je pouhých 0.15 ($d_("chem") = 0.85$). V klasické Crowding Distance by mohla být penalizována, pokud by měla podobné skóre vlastností. V `ParetoTanimotoDistance` však získává maximální Advantage $A(X) = R(X) - beta$:
  $ "Advantage" = R(X) - beta quad (beta = 0.40 "je baseline odměny") $
  Kladná hodnota Advantage ($+0.171$) přímo posiluje logaritmus pravděpodobnosti $nabla_theta log pi_theta(a_t mid s_t)$ a učí síť generovat tento revoluční motiv častěji!
])

== 2.5 Kolaps Paretovy fronty ve vyšších dimenzích ($M >= 4$) a Gated MORL

Ačkoliv je algoritmus NSGA-II zlatým standardem ve dvou až třech dimenzích, při přímém nasazení na komplexní návrh léčiv ($M >= 4$) naráží na fatální matematické selhání.

=== Matematický důkaz kolapsu dominance

Nechť je v $M$-dimenzionálním prostoru cílů vygenerována populace $N$ nezávislých řešení s náhodně rozdělenými hodnotami kritérií.

Pravděpodobnost, že náhodný bod $A$ dominuje konkrétní bod $B$, vyžaduje, aby byl bod $A$ lepší ve všech $M$ nezávislých dimenzích současně:

$ P(A succ B) = (frac(1, 2))^M $

Pravděpodobnost, že bod $A$ *není dominován* konkrétním bodem $B$, je $1 - (1/2)^M$. Aby byl bod $A$ nedominovaný v celé populaci $N$ jedinců (a tedy náležel do Rank 1 Paretovy fronty $cal(F)_1$), nesmí být dominován žádným z ostatních $N-1$ bodů:

$ P(A in cal(F)_1) = lr(1 - (frac(1, 2))^M)^(N - 1) $

Při limitním přechodu pro rostoucí dimenzi kritérií $M -> infinity$ získáváme:

$ lim_(M -> infinity) (frac(1, 2))^M = 0 quad ==> quad lim_(M -> infinity) P(A in cal(F)_1) = lim_(M -> infinity) (1 - 0)^(N - 1) = 1 $

#table(
  columns: (1.5fr, 1.5fr, 2fr),
  align: (center, center, left),
  table.header([*Počet dimenzí $M$*], [*$P(A in cal(F)_1)$ pro $N=1000$*], [*Důsledek pro NSGA-II v DrugEx*]),
  [2 cíle], [~ 1.8 %], [Optimální selekční tlak, čistá Paretova křivka],
  [3 cíle], [~ 28.4 %], [Funkční Paretovský kompromis, fronta pokrývá třetinu populace],
  [4 cíle], [~ 93.9 %], [Kritický útlum selekce: 94 % molekul má Rank 1],
  [5 cílů], [~ 99.2 %], [Prakticky celá populace leží na Frontě 1],
  [6 cílů], [100.0 %], [*Úplný kolaps dominance*: nulový selekční gradient]
)

Pokud 100 % populace spadne do Fronty 1, nedominované třídění ztrácí jakoukoliv rozlišovací schopnost. Celý trénink degeneruje pouze na řazení metrikou Crowding Distance pro zachování diverzity, což vede k chaotickému rozptylu bez optimalizace cílových vlastností.

=== Řešení: Architektura Gated MORL / Constrained MDP

DrugEx řeší tento rozpad rozdělením cílů na dvě striktně oddělené kategorie:
1. *Tvrdá fyzikálně-chemická síta (Boolean Gating Filters)*: Vlastnosti definující základní lékovou přijatelnost (Lipinski MW, LogP, syntetická nedostupnost SAScore, optimalizaci na drug-likeness QED) nevystupují jako Paretovy osy, ale jako boolovská hradla:
  $ G(X) = bb(I)(300 <= "MW" <= 500) dot bb(I)("LogP" <= 4.0) dot bb(I)("SAScore" <= 3.5) dot bb(I)("QED" >= 0.5) $
  Pokud molekula nesplní kterékoliv síto ($G(X) = 0$), její odměna je sražena na nulu a molekula je vyřazena z Paretovského soutěžení.
2. *Aktivní Paretovy osy ($M = 2$)*: Paretovské soutěžení probíhá výhradně mezi *dvěma skutečně protichůdnými farmakologickými cíli*:
  - Osa 1: *3D Tvarová a elektrostatická shoda (ROCS $T_("combo")$)*.
  - Osa 2: *Bioaktivita a selektivita k cíli (predikce z QSPRPred orákula)*.

Tím je dimenze Paretova prostoru redukována zpět na $M = 2$, čímž je obnoven maximální selekční tlak NSGA-II při současné stoprocentní garanci fyzikálně-chemické kvality ligandů!

== 2.6 Matematika a hladký gradient Modifikátorů (SmoothClippedScore vs. ClippedScore)

Surové výstupy skórovačů mají různé škály a jednotky ($T_("combo") in [0, 2]$, $"SAScore" in [1, 10]$, $p"IC"_(50) in [4, 10]$). Pro jejich integraci do prostředí `DrugExEnvironment` slouží modifikátory žádoucnosti (*Desirability Modifiers*).

#figure(
  image("../figures/fig_modifiers_curves.svg", width: 95%),
  caption: [Srovnání modifikátorů žádoucnosti: SmoothClippedScore vs. ClippedScore. V levém panelu je vidět průběh transformace odměny $S(x)$. V pravém panelu je znázorněna první derivace $frac(d S, d x)$ představující gradientní signál pro Reinforcement Learning. ClippedScore vykazuje nulový gradient mimo prahy, zatímco SmoothClippedScore poskytuje hladký gradient v celém rozsahu.]
)

=== Analytické srovnání a odvození

Uvažujme lineární ořezání `ClippedScore`:

$ S_("clip")(x) = "clamp"lr((frac(x - x_("lower"), x_("upper") - x_("lower")), 0.0, 1.0)) $

První derivace této funkce je po částech nulová:

$ frac(d S_("clip"), d x) = cases(frac(1, x_("upper") - x_("lower")) & "pro" x in (x_("lower"), x_("upper")), 0 & "pro" x < x_("lower") or x > x_("upper")) $

Pokud generátor vygeneruje molekuly s vlastností $x < x_("lower")$ (např. nízký počáteční tvarový překryv $T_("combo") = 0.4$ při prahu 0.6), derivace odměny je identicky nulová: $frac(d S, d x) = 0$. Gradient policy gradientu se zhroutí ($nabla_theta J = 0$), agent nezíská žádnou informaci o směru zlepšení a trénink *zamrzne*.

`SmoothClippedScore` (`drugex/training/scorers/modifiers.py`) odstraňuje tento problém zavedením spojité logistické křivky (S-křivka / S-curve):

$ S_("smooth")(x) = S_("low") + frac(L, 1 + exp(-k dot (x - x_("mid")))) $

kde parametry jsou přesně kalibrovány podle zadaných prahů:

$ k = frac(4, x_("upper") - x_("lower")), quad x_("mid") = frac(x_("upper") + x_("lower"), 2), quad L = S_("high") - S_("low") $

Derivace spojitého modifikátoru je dána vztahem:

$ frac(d S_("smooth"), d x) = L dot frac(k exp(-k (x - x_("mid"))), (1 + exp(-k (x - x_("mid"))))^2) = k dot (S(x) - S_("low")) dot lr(1 - frac(S(x) - S_("low"), L)) $

V inflexním bodě $x = x_("mid")$ dosahuje derivace své maximální hodnoty:

$ lr(frac(d S_("smooth"), d x) bar)_(x = x_("mid")) = frac(k dot L, 4) = frac(4, x_("upper") - x_("lower")) dot frac(L, 4) = frac(L, x_("upper") - x_("lower")) $

Směrnice S-křivky v bodě $x_("mid")$ tedy *přesně odpovídá směrnici původního lineárního ořezání*, ale pro všechna $x in (-infinity, infinity)$ platí $frac(d S_("smooth"), d x) > 0$. Agent tak získává nenulový gradientní impuls i v suboptimálních oblastech chemického prostoru.

== 2.7 Vizuální a datová architektura integrace QSPRpred

Knihovna *QSPRPred* slouží v DrugEx jako orákulum bioaktivity a selektivity. 

#figure_card(
  image("../figures/qsprpred_ecfp_compass.png", width: 100%),
  caption: [QSPRPred jako přísný porotce v Reinforcement Learningu a featurizace přes Morganovy otisky (ECFP6). Generátor chrlí kandidáty, QSPRPred počítá afinitu pIC50. Didaktická metafora: Detektiv s kružítkem a skartovačkou rozkládá okolí atomů na poloměrech r=0 až r=3 do 2048-bitového hashovacího slotu.]
)

=== Didaktická metafora: Detektiv s kružítkem a hashovací skartovačkou

Klasické modely strojového učení (Random Forest, XGBoost, SVM) nerozumí chemickým vazbám ani obrázkům. Potřebují tabulku čísel – binární vektor. Algoritmus ECFP6 (Extended Connectivity Fingerprint, poloměr $r=3$, průměr 6 vazeb) rozloží molekulu na kruhové fragmenty:

1. *Krok $r=0$ (Centrální atom)*: Stojíš na atomu s kružítkem. Podíváš se pod nohy: _„Jsem atom dusíku s jedním vodíkem.“_ Tento atomární stav se zakóduje do celého čísla.
2. *Krok $r=1$ (První kružnice)*: Roztáhneš kružítko na 1 vazbu: _„Jsem spojen s karbonylovým uhlíkem a aromatickým kruhem.“_
3. *Krok $r=2$ a $r=3$ (Širší okolí)*: Kružítko zachytí celý amidický řetězec a rozsáhlý farmakoforový motiv.
4. *Hashovací skartovačka*: Každé nalezené okolí je vhozeno do hashovací funkce (matematické „skartovačky“), která z libovolně složitého fragmentu vygeneruje celé číslo modulo 2048 (např. index 142 nebo 879). Na této pozici ve 2048-bitovém vektoru se nastaví bitový příznak (nebo se v count-vektoru inkrementuje četnost výskytu).

#pitfall_box("Bakalářský zádrhel: Rychlost vs. Přesnost orákula v RL", [
  Generátor DrugEx v RL smyčce vyprodukuje 1000 molekul v každé epoše (při 100 epochách je to 100~000 vyhodnocení).
  - *Pokud je model orákula příliš pomalý* (např. obří ensemble hlubokých sítí počítaný na CPU), trénování zamrzne a jeden experiment poběží dny až týdny.
  - *Pokud je model moc hloupý nebo overfitnutý*, generátor okamžitě objeví jeho slepá místa (Adversarial Exploitation) – začne generovat nesmyslné chemické halucinace, které pro model vypadají jako zázračný lék s $p"IC"_(50) = 12.0$.
  Proto je ideální featurizace ECFP6 ve spojení s gradientním boostingem (LightGBM) nebo grafovou konvolucí D-MPNN!
])

#figure_card(
  image("../figures/chemprop_dmpnn_graph.png", width: 100%),
  caption: [Chemprop (D-MPNN): Přímé hluboké učení na molekulárních grafech. Výměna směrovaných zpráv mezi atomy zabraňuje falešné ozvěně (tottering effect). Skryté stavy hran se po T krocích agregují do molekulárního vektoru h_mol.]
)

=== Chemprop (D-MPNN) a metafora: Kancelářská šuškanda a Directed Messages

Zatímco ECFP6 používá fixní kružítko a hashovací skartovačku, Chemprop (Directed Message Passing Neural Network — D-MPNN) přistupuje k molekule tak, jak doopravdy existuje: jako k *matematickému grafu*, kde uzly jsou atomy a hrany jsou chemické vazby.

*Metafora kancelářské šuškandy*:
Představ si molekulu jako kancelář zaměstnanců (atomů). V kroku 0 zná každý zaměstnanec jen svou vlastní profesi (_„jsem dusík se 3 valencemi“_). V kroku 1 pošle zprávu svým kolegům přes vazby: _„Ahoj sousede, já jsem dusík.“_ V kroku 2 už soused pošle dál: _„Můj soused je dusík a já jsem karbonyl.“_

*Proč „DIRECTED“ (D-MPNN)?*
V běžném MPNN pošle atom $A$ zprávu atomu $B$ a atom $B$ ji hned pošle zpátky atomu $A$. Tím vzniká falešná nekonečná ozvěna (*echo effect / tottering*). D-MPNN posílá zprávy po *směrovaných hranách* $v -> w$ a při předávání dál *výslovně zakazuje poslat zprávu zpět proti směru, odkud přišla*!

Matematická formulace D-MPNN:

1. *Agregace příchozích zpráv (Message Passing)*:
   $ m_(v w)^(t+1) = sum_(k in cal(N)(v) \\ {w}) h_(k v)^t $
   kde $cal(N)(v) \\ {w}$ jsou všichni sousedé atomu $v$ *s vyloučením cílového atomu $w$*. Tím je exaktně zamezeno zpětné ozvěně.

2. *Aktualizace stavu vazby (Edge Hidden State)*:
   $ h_(v w)^(t+1) = tau(h_(v w)^((0)) + bold(W)_m m_(v w)^(t+1)) $
   kde $h_(v w)^((0))$ jsou počáteční vlastnosti vazby a atomu $v$, $bold(W)_m$ je trénovatelná váhová matice a $tau$ je aktivační funkce ReLU.

3. *Readout fáze (Globální molekulární vektor)*:
   Jakmile proběhne $T$ kroků výměny zpráv, všechny atomové stavy se sečtou do jediného fixního vektoru:
   $ h_("mol") = sum_(v in G) h_v, quad h_v = tau lr(bold(W)_a dot "cat"lr((x_v, sum_(w in cal(N)(v)) h_(w v)^T))) $
   Tento vektor projde standardním vícevrstvým perceptronem (Feed-Forward MLP) a vyplivne predikovanou afinitu $p"IC"_(50)$ k receptoru!

== 2.8 Produkční implementace a trénovací workflow v DrugEx

Následující ucelený Python skript demonstruje kompletní inicializaci trénovacího běhu MORL s duální architekturou Agent vs. Mutate Prior, modifikátory `SmoothClippedScore`, schématem `ParetoCrowdingDistance` a prostředím `DrugExEnvironment`:

#code_card(
  title: "Inicializace MORL SequenceExploreru a smyčky odměn v DrugEx",
  lang: "python",
  code: "from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore, Gaussian
from drugex.training.explorers import SequenceExplorer

# 1. Definice hodnoticího prostředí s hladkými modifikátory (SmoothClippedScore)
scorers = [
    Property(prop='SA', modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5)),  # syntéza
    Property(prop='MW', modifier=Gaussian(mu=420.0, sigma=60.0)),                # optimální MW
    Property(prop='QED', modifier=SmoothClippedScore(lower_x=0.4, upper_x=0.8))  # drug-likeness
]
env = DrugExEnvironment(scorers=scorers, thresholds=[0.5, 0.5, 0.5], reward_scheme=ParetoCrowdingDistance())

# 2. Inicializace průzkumníka s mutačním priorem (epsilon = 0.20 brání kolapsu modů)
explorer = SequenceExplorer(
    agent=agent, mutate=mutate, env=env,
    epsilon=0.20, n_samples=1000, batch_size=128
)
# -> Trénink MORL (100 epoch):
#    Epoch 01/100: Loss = 3.842 | Valid = 98.4% | Desired = 4.2%
#    Epoch 50/100: Loss = 1.432 | Valid = 97.5% | Desired = 41.0%
#    Epoch 100/100: Loss = 0.891 | Valid = 97.1% | Desired = 59.4% (Unikátnost = 91.2%)"
)

