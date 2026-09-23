#import "../nature_theme.typ": *

= 3. 3D Tvarové Skórování, Konformační Pnutí & Flexibilní Cíle

Klasická chemoinformatika se po desetiletí opírala o dvourozměrné topologické otisky (např. Morganovy fingerprinty, ECFP4 či MACCS klíče). Tyto metody kódují molekulu jako binární vektor přítomnosti podstruktur v topologickém okolí atomů do poloměru dvou až tří chemických vazeb. Tento přístup má však fundamentální limit: dvě molekuly se zcela odlišnými chemickými kostrami vykazují 2D topologickou podobnost blízkou nule ($T_("2D") < 0.20$), ačkoliv v trojrozměrném prostoru zaujímají identický van der Waalsovský objem a vykazují shodné prostorové uspořádání klíčových interakčních center.

Biologický cíl (receptor, enzymatická kavita či proteinové rozhraní) nečte 2D graf ani textový řetězec SMILES. Vazebné rozhraní makromolekuly interaguje výhradně s trojrozměrnou van der Waalsovou prostorovou obálkou, elektrostatickým potenciálem a směrovými vektory vodíkových vazeb. Schopnost překročit hranice známých chemických tříd při zachování požadovaného 3D tvaru – označovaná jako _Scaffold Hopping_ – je proto na úrovni 2D otisků prakticky nedosažitelná.

Tato kapitola rozebírá teoretický a biofyzikální aparát trojrozměrného tvarového skórování metodou ROCS (_Rapid Overlay of Chemical Structures_), vyvrací iluzi izolovaného konformačního minima v plynné fázi a formuluje ligand-based 3D paradigma pro flexibilní receptorové kapsy a vnitřně nestrukturované proteiny (_Intrinsically Disordered Proteins_, IDP).

== 3.1 Fyzikální podstata ROCS: Analytický Gaussovský překryv a kompozitní skóre

Historické přístupy k porovnávání 3D tvaru molekul modelovaly atomy jako pevné neprostupné koule (_hard spheres_) s van der Waalsovými poloměry $R_i$. Výpočet objemového průniku dvou molekul vyžadoval numerickou integraci na diskrétní trojrozměrné prostorové mřížce (voxel grid). Mřížkové metody však trpěly enormní výpočetní náročností ($O(N^3)$ paměťová i časová komplexita), diskretizačními chybami a nespojitostí při rotaci molekuly.

Zásadní průlom představili v roce 1996 J. A. Grant a B. T. Pickup nahrazením pevných sfér spojitými trojrozměrnými Gaussovskými funkcemi hustoty.

#figure(
  image("../figures/fig_rocs_gaussian.svg", width: 90%),
  caption: [Gaussovská reprezentace molekulárního tvaru a farmakoforových vlastností v metodě ROCS. Atomy jsou modelovány jako hladké 3D Gaussovské funkce hustoty. Prostorový tvar ($T_("shape")$) je dán objemovým překryvem všech atomů, zatímco farmakoforová složka ($T_("color")$) integruje specifická interakční centra podle silového pole ImplicitMillsDean.]
)

=== Matematické odvození Grant-Pickettova překryvového integrálu

Pro každý atom $i$ s polohovým vektorem jádra $bold(r)_i in bb(R)^3$ je atomová hustota v bodě $bold(r)$ definována jako sférická Gaussovská funkce:

$ rho_i (bold(r)) = p_i exp(-alpha_i |bold(r) - bold(r)_i|^2) $

kde $bold(r)_i = (x_i, y_i, z_i)^T$ reprezentuje kartézské souřadnice jádra atomu $i$, $p_i$ je centrální hustota atomu (standardně normalizovaná na $p_i = 1.0$) a $alpha_i$ je koeficient prostorového rozptylu, který přímo determinuje efektivní velikost atomu.

Parametr $alpha_i$ je kalibrován tak, aby integrální objem Gaussovské funkce přes celý trojrozměrný prostor $bb(R)^3$ přesně odpovídal van der Waalsovu objemu atomu $V_i = 4/3 pi R_i^3$:

$ V_i = integral_(bb(R)^3) rho_i (bold(r)) dif bold(r) = p_i (frac(pi, alpha_i))^(3/2) = frac(4, 3) pi R_i^3 $

Odtud plyne analytický vztah pro rozptylový koeficient:

$ alpha_i = frac(pi, (4/3 pi)^(2/3) R_i^2) approx frac(1.1444 pi, R_i^2) $

V praktických chemoinformatických implementacích (např. OpenEye ROCS, CDPKit, RDKit `rdShapeAlign`) se často využívá škálovaná normalizace $alpha_i = pi / R_i^2$, kde $R_i$ představuje tabulkový van der Waalsův poloměr daného chemického prvku (např. $R_("C") = 1.70 "Å"$, $R_("N") = 1.55 "Å"$, $R_("O") = 1.52 "Å"$, $R_("S") = 1.80 "Å"$, $R_("F") = 1.47 "Å"$).

Celková prostorová hustota molekuly $A$ sestávající z $N_A$ atomů je dána lineární superpozicí atomových Gaussovských příspěvků:

$ rho_A (bold(r)) = sum_(i in A) rho_i (bold(r)) = sum_(i=1)^(N_A) p_i exp(-alpha_i |bold(r) - bold(r)_i|^2) $

=== Dvojnásobný Gaussovský produktový teorém

Míra prostorového překryvu mezi molekulou $A$ a molekulou $B$ je definována jako konvoluční integrál součinu jejich hustot přes celý prostor $bb(R)^3$:

$ I_(A B) = V(A, B) = integral_(bb(R)^3) rho_A (bold(r)) rho_B (bold(r)) dif bold(r) = sum_(i in A) sum_(j in B) integral_(bb(R)^3) rho_i (bold(r)) rho_j (bold(r)) dif bold(r) $

Zásadní matematickou vlastností součinu dvou trojrozměrných Gaussovských funkcí se středy v bodech $bold(r)_i$ a $bold(r)_j$ je skutečnost, že jejich integrál má *přesné analytické řešení v uzavřeném tvaru*:

$ I_(i j) = integral_(bb(R)^3) exp(-alpha_i |bold(r) - bold(r)_i|^2) exp(-alpha_j |bold(r) - bold(r)_j|^2) dif bold(r) = (frac(pi, alpha_i + alpha_j))^(3/2) exp(- frac(alpha_i alpha_j, alpha_i + alpha_j) |bold(r)_i - bold(r)_j|^2) $

Celkový objemový překryv $V(A, B)$ je pak dán sumou přes všechny dvojice atomů:

$ V(A, B) = sum_(i in A) sum_(j in B) p_i p_j (frac(pi, alpha_i + alpha_j))^(3/2) exp(- frac(alpha_i alpha_j, alpha_i + alpha_j) |bold(r)_i - bold(r)_j|^2) $

Stejným analytickým vzorcem se spočítají i vlastní objemy (_self-volumes_) molekul $A$ a $B$:

$ V(A, A) = sum_(i in A) sum_(i' in A) p_i p_(i') (frac(pi, alpha_i + alpha_(i')))^(3/2) exp(- frac(alpha_i alpha_(i'), alpha_i + alpha_(i')) |bold(r)_i - bold(r)_(i')|^2) $

Analytické řešení přináší tři zásadní výhody:
1. *Nulová diskretizační chyba*: Výpočet netrpí žádnou prostorovou mřížkovou chybou.
2. *Extrémní výpočetní rychlost*: Vyhodnocení jedné dvojice molekul trvá méně než $1 mu"s"$ na moderním CPU.
3. *Analytická diferencovatelnost*: Gradient $nabla_(bold(r)_j) V(A, B)$ má rovněž exaktní analytické vyjádření, což umožňuje bleskovou optimalizaci orientace molekuly v prostoru tuhých těles $S E(3)$ pomocí kvazi-Newtonových metod (BFGS).

=== Metrika Shape Tanimoto a prostorové zarovnání v SE(3)

Znalost integrálů $V(A, B)$, $V(A, A)$ a $V(B, B)$ umožňuje definovat normalizovanou míru prostorové podobnosti – koeficient *Shape Tanimoto* ($T_("shape")$):

$ T_("shape")(A, B) = frac(V(A, B), V(A, A) + V(B, B) - V(A, B)) in [0, 1] $

Tato metrika splňuje axiomy míry podobnosti: pro identické molekuly v identické orientaci platí $T_("shape")(A, A) = 1.0$, pro nekonečně vzdálené molekuly $T_("shape") = 0.0$ a metrika je přísně symetrická ($T_("shape")(A, B) = T_("shape")(B, A)$).

Hodnota $T_("shape")$ závisí na vzájemné poloze a rotaci molekul. Cílem optimalizačního algoritmu je nalézt rigidní prostorovou transformaci $bold(T) = (bold(R), bold(t)) in S E(3)$ (rotaci $bold(R) in S O(3)$ a translaci $bold(t) in bb(R)^3$), která maximalizuje objemový překryv:

$ bold(T)^* = arg max_(bold(T)) V(A, bold(T)(B)) $

Aby optimalizace neuvízla v lokálním extrému, provádí se výpočet ve dvou krocích:
1. *Sladění hlavních os setrvačnosti (Principal Axes of Inertia – PAI)*: Pro obě molekuly se sestaví tenzor momentu setrvačnosti $bold(I)$. Vlastní vektory tenzoru definují tři ortogonální osy. Vzájemným sladěním těchto os vzniknou 4 neekvivalentní startovní orientace.
2. *Lokální kvazi-Newtonova optimalizace*: Z každé ze 4 startovních orientací se provede gradientní maximalizace překryvu. Globální maximum napříč všemi startovními pozicemi určuje výsledné $T_("shape")$.

=== Farmakoforový Color Tanimoto a silové pole ImplicitMillsDean

Samotný prostorový tvar k selektivnímu návrhu biologicky aktivních látek nepostačuje. Například hydrofobní naftalenový skelet a polární chinazolin vykazují téměř totožný tvar ($T_("shape") > 0.85$), avšak jejich elektrostatický profil a schopnost tvořit vodíkové vazby jsou diametrálně odlišné.

Metoda ROCS proto zavádí koncept barevných farmakoforových center (_Color Features_). Každé funkční skupině je přiřazen specifický chemický typ podle silového pole `ImplicitMillsDean`, které rozlišuje 6 základních tříd:
1. *Donory vodíkových vazeb (Donors)*: skupiny $-"OH"$, $-"NH"_2$, $-"NH"-$.
2. *Akceptory vodíkových vazeb (Acceptors)*: karbonylové kyslíky $="O"$, pyridinové dusíky $="N"-$, etherové kyslíky $-"O"-$.
3. *Kationická centra (Positive / Cations)*: protonované alifatické aminy, guanidiniové skupiny, kvarterní dusíky.
4. *Anionická centra (Negative / Anions)*: karboxylátové skupiny $-"COO"^-$, sulfonáty $-"SO"_3^-$, fosfáty.
5. *Hydrofobní centra (Hydrophobes)*: alifatické a alicyklické uhlíkové clustery (terc-butyl, isobutyl, cyklohexyl).
6. *Aromatické kruhy (Rings)*: planární aromatické $pi$-systémy (benzen, thiofen, indol, pyridin).

Tato centra jsou modelována jako barevné Gaussovské funkce $rho_(i, c)(bold(r))$ umístěné v geometrických středech příslušných skupin. Zásadním pravidlem je *ortogonalita barevných tříd*: k překryvu dochází *výhradně mezi centry téhož typu*:

$ C(A, B) = sum_(c in "Types") sum_(i in A_c) sum_(j in B_c) w_c (frac(pi, alpha_(c,i) + alpha_(c,j)))^(3/2) exp(- frac(alpha_(c,i) alpha_(c,j), alpha_(c,i) + alpha_(c,j)) |bold(r)_i - bold(r)_j|^2) $

kde $w_c$ je váhový koeficient dané vlastnosti. Normalizovaný *Color Tanimoto* koeficient ($T_("color")$) je pak definován jako:

$ T_("color")(A, B) = frac(C(A, B), C(A, A) + C(B, B) - C(A, B)) in [0, 1] $

=== Kompozitní metrika TanimotoCombo

V prostředí DrugEx MORL slouží jako primární spojitá skórovací funkce kompozitní metrika *TanimotoCombo* ($T_("combo")$), která sčítá prostorovou a farmakoforovou shodu:

$ T_("combo")(A, B) = T_("shape")(A, B) + T_("color")(A, B) in [0, 2] $

#table(
  columns: (1.5fr, 1.2fr, 3fr),
  align: (left, center, left),
  [Interval $T_("combo")$], [Klasifikace], [Interpretace v de novo designu léčiv],
  [$T_("combo") < 0.80$], [Nízká shoda], [Molekula prostorově nepasuje do kavit nebo má nesprávně orientovaná polární centra.],
  [$0.80 <= T_("combo") < 1.20$], [Střední shoda], [Dobrý základní sterický překryv, avšak chybí klíčové vodíkové vazby či iontové páry.],
  [$1.20 <= T_("combo") < 1.50$], [Vysoká shoda], [Vynikající bioisosterní analog; dokonalé vyplnění kapsy se správnou farmakoforovou geometrií.],
  [$T_("combo") >= 1.50$], [Téměř identická], [Molekula představuje blízký strukturní analog referenčního ligandu.]
)

#rule_box("Vzorkování konformerů pro ROCS v DrugEx", [
  Počet generovaných konformací volte podle počtu rotovatelných vazeb ($N_("rot")$): pro rigidní molekuly ($N_("rot") < 5$) postačuje $20$ až $30$ konformací; pro běžné lékové struktury ($N_("rot") = 6 - 10$) generujte $50$ konformací. Překročení hranice $100$ konformací nepřináší měřitelný nárůst maximálního $T_("combo")$, avšak dramaticky zpomaluje trénovací cyklus Reinforcement Learning (RL).
])

#pitfall_box("Nutnost explicitních polárních vodíků v RDKit", [
  RDKit na rozdíl od komerčního balíku OpenEye vyžaduje pro správnou detekci farmakoforových center (zejména donorů vodíkových vazeb) explicitní přítomnost polárních vodíků. Pokud do skórovače `RDKitROCSScorer` vstoupí molekula bez předchozího volání `Chem.AddHs(mol)`, barevná složka $T_("color")$ zkolabuje k nule a celkové skóre odpovídá pouze čistému tvaru.
])

== 3.2 Iluze minima v plynné fázi: Vakuum vs. Bioaktivní konformační pnutí

Standardní chemoinformatické konformační generátory (např. RDKit ETKDGv3 následovaný lokální optimalizací silovým polem MMFF94 nebo UFF) vyhledávají lokální a globální minima izolované molekuly *ve vakuu (plynné fázi, kde relativní permitivita $epsilon_r = 1$)*.

Tento přístup však v biologickém kontextu vytváří zásadní fyzikální klam.

#figure(
  image("../figures/fig_bioactive_strain.jpg", width: 95%),
  caption: [Iluze minima v plynné fázi a bioaktivní konformační pnutí. Ve vakuu ($epsilon_r = 1$) molekula kolabuje do sbalené geometrie za vzniku intramolekulárních kontaktů. V krystalografické kavitě proteinu je ligand stabilizován sítí intermolekulárních vodíkových vazeb a hydrofobním efektem, přičemž přijímá konformační pnutí $Delta E_("strain") approx 2 - 5 "kcal/mol"$.]
)

=== Proč molekuly ve vakuu kolabují do nerelevantních geometrií?

Ve vakuu neexistují molekuly rozpouštědla, které by stínily elektrostatické síly nebo soutěžily o vodíkové vazby. Molekula proto minimalizuje svou potenciální energii $E_("pot")$ tím, že sbaluje své polární skupiny dovnitř a vytváří *intramolekulární vodíkové vazby* (např. interakce karboxylu s amidickým dusíkem či protonovaným aminem). Alifatické a aromatické fragmenty se rovněž přitahují disperzními silami, čímž vzniká kompaktní, sbalená konformace s minimálním povrchem.

V reálném biologickém prostředí (vodný roztok s $epsilon_r approx 78.4$ a vazebné rozhraní proteinu) je však termodynamická situace zcela odlišná:
- Voda jako polární protické rozpouštědlo nabízí silné konkurenční vodíkové vazby, které intramolekulární kontakty ligandu rozrušují.
- V kavitě receptoru jsou funkční skupiny ligandu nuceny orientovat se směrem k aminokyselinovým zbytkům proteinu (např. aspartátům, lysinům, páteřním karbonylům), aby vytvořily vysoce příznivé *intermolekulární kontakty*.

=== Bioaktivní konformační pnutí ($Delta E_("strain")$)

Krystalografické a kryo-elektronově mikroskopické struktury komplexů protein-ligand v Protein Data Bank (PDB) opakovaně prokazují, že *ligand ve vázaném stavu téměř nikdy nezaujímá své globální minimum z plynné fáze*. Místo toho vykazuje tzv. *bioaktivní konformační pnutí*:

$ Delta E_("strain") = E_("bioactive") - E_("gas-minimum") approx 2 - 5 "kcal/mol" $

U velkých flexibilních ligandů či makrocyklů může toto pnutí dosahovat až $6 - 8 "kcal/mol"$.

Z termodynamického hlediska je toto pnutí plně kompenzováno celkovou volnou entalpií vazby ($Delta G_("bind")$):

$ Delta G_("bind") = Delta H_("bind") - T Delta S_("bind") $

Příznivý entalpický příspěvek ($Delta H_("bind") < 0$) plynoucí z tvorby sítě pevných vodíkových vazeb, solných můstků a disperzních van der Waalsových interakcí s proteinem, společně s entropickým ziskem z vytěsnění neuspořádaných molekul vody z vazebné kapsy (*hydrofobní efekt*), snadno převýší energetickou penalizaci $Delta E_("strain")$.

#insight_box("Důsledky pro 3D de novo design a ROCS skórování", [
  Pokud generátor konformací vytvoří pouze izolovaná plynná minima (MMFF94) a skórovač je porovná s experimentální bioaktivní referencí z krystalu, dojde k selhání:
  1. *Falešně negativní hodnocení*: Skutečně aktivní kandidátní molekuly, které by se v kavitě snadno rozvinuly do bioaktivní geometrie, získají nízké skóre $T_("combo")$, protože jejich plynná minima jsou uměle sbalená.
  2. *Falešně pozitivní hodnocení*: Generátor upřednostní rigidní molekuly, jejichž plynný tvar náhodně odpovídá referenci, ačkoliv nemají schopnost indukovaného přizpůsobení (_induced fit_).
])

=== Metodická řešení v DrugEx jádře

Pro překonání propasti mezi vakuovými minimy a bioaktivními tvary implementuje DrugEx čtyři provázané mechanismy:

1. *Vázaný konformační embedding (`AllChem.ConstrainedEmbed`)*: Pokud je znám referenční krystalografický skelet, vygenerované molekuly se vnořují s fixací prostorových souřadnic společného jádra. Tím je konformační geometrie nucena zachovat bioaktivní uspořádání.
2. *Znalostní torzní potenciály ETKDGv3*: Algoritmus ETKDGv3 (Riniker & Landrum) nepočítá čisté vakuové síly, nýbrž aplikuje empirické torzní distribuce odvozené z desetitisíců krystalových struktur v Cambridge Structural Database (CSD). Tím generuje geometrie odpovídající reálným krystalickým stavům.
3. *Široké energetické okno ($Delta E_("window") = 15 - 20 "kcal/mol"$)*: V generátorech `CDPKitConformerGenerator` a `OmegaConformerGenerator` se nezachovává pouze nejnižší stav, nýbrž celý ensemble konformací v energetickém okně až $20 "kcal/mol"$, což spolehlivě pokrývá bioaktivní pnutí.
4. *RMSD prořezávání (Pruning)*: Eliminace geometricky redundantních konformací s prahem $"RMSD" < 0.5 "Å"$ zajišťuje maximální diverzitu tvarového vzorkování při zachování kompaktní velikosti konformačního poolu.

== 3.3 Paradigma flexibilních cílů & Intrinsically Disordered Proteins (IDP)

Klasický racionální design léčiv založený na struktuře receptoru ([_Structure-Based Drug Design_], SBDD) vychází z Fischerova paradigmatu *zámku a klíče* ([_Lock-and-Key_]). Algoritmy molekulárního dokování (AutoDock Vina, Schrödinger Glide, GOLD) umísťují ligand do rigidní krystalografické mřížky proteinu s vysokým rozlišením.

V moderní medicinální chemii a onkologii však klíčové terapeutické cíle tomuto statickému modelu zásadně vzdorují.

#figure(
  image("../figures/fig_multigroup_rocs.svg", width: 90%),
  caption: [Architektura Multi-Reference Group Scoringu v DrugEx pro flexibilní receptorové cíle. Kandidátní molekuly jsou hodnoceny vůči nezávislým skupinám experimentálních referencí (např. ortosterická kapsa vs. alosterické místo). Skórovací funkce vrací maximální shodu napříč konformačním poolem a referenčními skupinami.]
)

=== Selhání rigidního dokování u dynamických cílů

K selhání rigidního dokování dochází zejména u následujících tříd biologických cílů:
- *Intrinsically Disordered Proteins (IDP) a IDR domény*: Onkoproteiny jako c-Myc, transaktivační doména p53, $alpha$-synuklein či tau protein zcela postrádají stabilní sekundární a terciární strukturu. V roztoku koexistují jako dynamický konformační ensemble na ploché volnoenergetické krajině s mělkými bariérami ($Delta G^dagger approx k_B T$).
- *Flexibilní membránové receptory (GPCR, iontové kanály)*: U receptorů, jako je chemokinový receptor CCR2, indukuje vazba ligandu rozsáhlé alosterické posuny transmembránových šroubovic (mechanismus [_Induced Fit_] a [_Conformational Selection_]).
- *Modely s nízkým rozlišením (kryo-EM > 3.5 Å, AlphaFold)*: Nepřesné rotamery postranních řetězců vedou při rigidním dokování k falešným sterickým srážkám ([_clashes_]), což vede k vyřazení vysoce aktivních molekul.

=== Ligand-Based 3D paradigma (LBDD) jako rigorózní alternativa

Pokud máme k dispozici sadu experimentálně potvrzených aktivních ligandů (získaných z NMR titrací, transferred NOE, SPR či afinitních esejí), jejich trojrozměrná konformace ve vázaném stavu představuje *dokonalý negativní prostorový a elektrostatický odlitek funkčního stavu vazebného místa*.

Vysoce afinitní ligand stabilizuje specifický funkční konformační stav proteinu. Pokud generativní model DrugEx navrhne novou molekulu, která:
1. Svým tvarem přesně vyplní van der Waalsovský objem aktivních referencí ($T_("shape") -> 1.0$),
2. Prezentuje donory, akceptory a náboje ve shodných prostorových souřadnicích ($T_("color") -> 1.0$),
3. Zachovává syntetickou proveditelnost ($"SAScore" <= 3.0$) a nízké vnitřní pnutí,
má tato nová sloučenina vysokou pravděpodobnost vázat se do stejného funkčního stavu receptoru se srovnatelnou afinitou, aniž bychom potřebovali rigidní krystalovou strukturu cíle.

=== Konsensuální supermolekuly a Multi-Reference Group Scoring

U flexibilních cílů nelze spoléhat na porovnání vůči jedinému referenčnímu ligandu. DrugEx proto využívá dvě komplementární strategie:

*1. Konsensuální supermolekula (`supermol.sdf`)*:
Souřadné systémy několika známých experimentálních struktur ligandů jsou sjednoceny do společného SDF souboru. Supermolekula definuje sjednocení prostorových objemů všech referencí:

$ V_("super") = union.big_k V(R_k) $

Tento přístup umožňuje jediné rychlé porovnání ($1 times$ namísto $N times$) a motivuje generátor k navrhování molekul, které propojují sub-kapsy obsazené různými ligandy.

*2. Multi-Reference Group Scoring*:
Pokud referenční ligandy vážou protein v odlišných vazebných místech (např. ortosterická vs. alosterická kapsa), sloučení do jedné supermolekuly by vytvořilo nerealisticky velký objem. DrugEx proto organizuje reference do pojmenovaných skupin a vyhodnocuje maximální dosažené skóre přes konformační pool:

$ "Score"_("group")(X) = max_(R in "Group") ( max_(k in "Conformers"(X)) T_("combo")(C_k (X), R) ) $

#telemetry_box("Multi-kavitní odezva a alosterické přepínání", [
  Během vícekavitového skórování vyhodnocuje DrugEx afinitu kandidátních struktur k nezávislým kapsám. V reálném běhu pro receptor CCR2 systém zaznamenal:
  - Kandidát #0: $T_("combo")$(ortosterická) $= 0.847$, $T_("combo")$(alosterická) $= 0.512$ $arrow.r$ výstupní skóre $= 0.847$.
  - Kandidát #1: $T_("combo")$(ortosterická) $= 0.620$, $T_("combo")$(alosterická) $= 1.140$ $arrow.r$ výstupní skóre $= 1.140$.
  Generativní agent tak spontánně objevuje odlišné vazebné módy a optimalizuje molekuly specifické pro jednotlivé funkční konformace.
])

== 3.4 Numerická stabilita: Planární singularita a korekce v ose Z

Při rozsáhlém generování konformací v DrugEx byl odhalen závažný matematický problém postihující dokonale planární aromatické molekuly (např. ligand 8SKP či planární kondenzované heterocykly).

=== Matematická podstata singularity tenzoru setrvačnosti

Pokud generátor konformací vytvoří planární molekulu ležící přesně v rovině $x y$, mají všechny atomy nulovou souřadnici v ose $z$ ($z_i = 0.0000 "Å"$). Tenzor momentu setrvačnosti $bold(I)$ má v tomto případě tvar:

$ bold(I) = mat(
  sum m_i y_i^2, -sum m_i x_i y_i, 0;
  -sum m_i x_i y_i, sum m_i x_i^2, 0;
  0, 0, sum m_i (x_i^2 + y_i^2)
) $

Ačkoliv je tenzor mechanicky definován, v Gaussovském prostorovém modelu ROCS dochází k degeneraci kovarianční matice rozptylu podél osy $z$ ($sigma_z^2 = 0$).

Při výpočtu hlavních os setrvačnosti (PAI) a následné normalizaci dochází k dělení nulou při inverzi singulární matice:

$ bold(I)^(-1) "neexistuje" quad (det(bold(I)_(2 D)) = 0) $

Tato singularita vedla k okamžitému pádu celého výpočetního procesu (`FloatingPointError` či pád C++ knihovny s chybou `SingularMatrixException`).

=== Algoritmická náprava v DrugEx (+0.01 Å fix)

V souboru `drugex/training/scorers/conformer_generators.py` řeší DrugEx pád na dokonale planárních molekulách (jako byl v praxi identifikovaný ligand 8SKP) zavedením automatické geometrické korekce: k souřadnicím osy $z$ je aplikován nepatrný střídavý ofset $plus.minus 0.01 "Å"$:

$ z_i^("opraveno") = z_i + (-1)^i dot 0.01 "Å" $

Tato úprava je z biofyzikálního hlediska zcela zanedbatelná: odchylka $0.01 "Å"$ leží hluboko pod experimentálním rozlišením krystalografie ($approx 1.5 - 2.5 "Å"$) i van der Waalsovým poloměrem atomu uhlíku ($1.70 "Å"$) a nemá žádný vliv na farmakoforové překryvy. Z biofyzikálního a matematického hlediska však řeší dva problémy najednou: eliminuje identické nuly v SDF souboru (které shazovaly konvertory OpenEye) a současně rozbíjí nulovou varianci v těžišti ($I_z > 0$), čímž odstraňuje singularitu tenzoru setrvačnosti a garantuje stoprocentní stabilitu PAI zarovnání.

#code_card(
  title: "Automatická oprava planární singularity (střídavý 0.01 Å fix)",
  lang: "python",
  code: "def fix_planar_singularity(mol: Chem.Mol, offset_z: float = 0.01) -> Chem.Mol:
    # Detekce planarity: jsou všechny Z-souřadnice v konformeru nulové?
    for conf in mol.GetConformers():
        if all(abs(conf.GetAtomPosition(i).z) < 1e-5 for i in range(mol.GetNumAtoms())):
            for idx in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(idx)
                # Střídavý ofset rozbije rovinu a vytvoří nenulovou varianci I_zz
                delta = offset_z if idx % 2 == 0 else -offset_z
                conf.SetAtomPosition(idx, (pos.x, pos.y, pos.z + delta))
    return mol

# Ověření na planárním benzenovém jádru (det(I) = 0 -> det(I) > 0)
benzene_3d = fix_planar_singularity(benzene_planar)
# -> I_z posunuto z 0.000 na 0.012 Å: matice je invertibilní, PAI zarovnání stabilní"
)
