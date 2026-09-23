#import "../nature_theme.typ": *

= 3.5 Chemoinformatický Můstek: Ztracené Klenoty z Fáze 1 (RAScore, Zvěřinec Modifikátorů & Konformační Filtry)

Tato kapitola slouží jako přímý metodologický most pro výzkumníka, který prostudoval základní teorii generativních reprezentací, SMILES syntaxe a Paretovského řazení (strany 1–40 původní příručky). *Neobsahuje žádné opakování elementárních konceptů.* Jejím jediným cílem je zachránit a zpřehlednit klíčové produkční nástroje, které propojují 2D skórování s navazujícími 3D ROCS enginy: moderní predikci syntetizovatelnosti *RAScore*, matematický aparát *Desirability Modifiers* a krystalografické filtry konformerů.

== 3.5.1 SAScore vs. RAScore: Od lokálních fragmentů k AI retrosyntéze

Při optimalizaci molekul pomocí Reinforcement Learning (RL) nestačí hodnotit pouze vazebnou afinitu k cíli. Bez přísné kontroly syntetické dostupnosti začne neuronová síť generovat geometricky dokonalé, avšak chemicky nepřipravitelné struktury. Platforma DrugEx podporuje dva komplementární přístupy k hodnocení syntézy:

1. *Klasický SAScore (Ertl & Schuffenhauer, 2009)*:
   Heuristická metoda hodnotící lokální chemickou složitost. Výsledné skóre leží na škále $[1.0, 10.0]$:
   $ "SAScore" = "FragmentScore" - "ComplexityPenalty" $
   kde `FragmentScore` odměňuje přítomnost běžných motivů z databáze PubChem, zatímco `ComplexityPenalty` penalizuje stereocentra, spiro-kruhy, makrocykly a nadměrný počet atomů v můstkových systémech.
   - *Výhoda*: Bleskový výpočet ($< 0.1$ ms na molekulu).
   - *Limit*: Je slepý ke skutečné syntetické dostupnosti – neví, zda existují komerční prekurzory.

2. *Moderní RAScore (Retrosynthetic Accessibility Score, 2021)*:
   Metoda strojového učení vyvinutá na bázi nástroje *AiZynthFinder*. Využívá hluboké neuronové sítě a stromové prohledávání Monte Carlo (MCTS) natrénované na více než 120~000 patentovaných organických reakcích.
   - Predikuje pravděpodobnost, že molekulu lze úspěšně syntetizovat v $<= 10$ reakčních krocích z komerčně dostupných chemikálií:
     $ "RAScore"(m) in [0.0, 1.0] $
   - Molekuly s $"RAScore" > 0.80$ mají 92% laboratorní úspěšnost syntézy.

#rule_box("Volba filtru syntetizovatelnosti")[
  V raných fázích RL tréninku používejte rychlý `Property('SA')` s klesajícím sigmoidálním modifikátorem `SmoothClippedScore(lower_x=4.5, upper_x=2.5)`. Pro finální filtrování a prioritizaci lead struktur v pozdních epochách nasaďte `RAScore`, který garantuje proveditelnost v reálné organické laboratoři.
]

== 3.5.2 Zvěřinec modifikátorů žádoucnosti (Desirability Modifiers)

Předání surové fyzikální veličiny (např. SAScore 1–10, molekulová hmotnost 200–800 Da, afinitní pIC50 4–10) do Paretovského třídění způsobí okamžitý kolaps gradientů. Modul `drugex.training.scorers.modifiers` převádí libovolný fyzikální parametr do unifikovaného intervalu žádoucnosti $[0.0, 1.0]$ pomocí tří základních matematických transformací:

=== 1. Hladká sigmoidální S-křivka (S-curve): `SmoothClippedScore`
Klasický ořez `ClippedScore` má nulovou derivaci mimo mezní hranice a bod nespojitosti na prahu, což zmrazí váhy sítě. `SmoothClippedScore` naproti tomu poskytuje spojitý, hladce diferencovatelný gradient:

$ f(x) = 1 / (1 + exp(- k (x - x_"mid"))) $

kde parametry strmosti $k$ a středu $x_"mid"$ jsou automaticky odvozeny z uživatelských mezí:
$ k = 4 / (|x_"upper" - x_"lower"|), wide x_"mid" = (x_"upper" + x_"lower") / 2 $

Pokud $x_"lower" < x_"upper"$, funkce roste (odměňování aktivity); pokud $x_"lower" > x_"upper"$, funkce hladce klesá (penalizace SAScore či toxicity).

=== 2. Gaussovský zvonový terč: `MinMaxGaussian` & `Gaussian`
Pro parametry, které mají optimální biologické okno (např. molekulová hmotnost MW pro prostupnost membránami či lipofilita LogP), aplikujeme Gaussovskou normalizaci s těžištěm $mu$ a tolerančním rozptylem $sigma$:

$ g(x) = exp(- (x - mu)^2 / (2 sigma^2)) $

V třídě `MinMaxGaussian(mu=3.0, sigma=1.2, minimize=True)` je odměna maximální ($1.0$) pro hodnoty $x <= mu$ a klesá jako poloviční Gauss pro $x > mu$. Tím je lipofilita bezpečně udržena pod hodnotou $3.0$, aniž bychom trestali hydrofilnější molekuly.

#insight_box("Proč nepoužívat ostré schodovité filtry?")[
  Schodovitý filtr (např. `ClippedScore`) působí na optimalizační krajinu jako propast. Pokud molekula dosáhne skóre $0.799$ při prahu $0.800$, obdrží $0.0$ a agent se nedozví, kterým směrem váhy posunout. Hladký modifikátor poskytuje směrový vektor i v suboptimální oblasti.
]

== 3.5.3 Konformační filtry & ETKDGv3 v RDKitConformerGenerator

Třetím klíčovým prvkem je generování 3D souřadnic pro následné tvarové zarovnání. Třída `RDKitConformerGenerator` kombinuje distanční geometrii s torzními preferencemi odvozenými z Cambridgeské strukturní databáze (CSD):

1. *Experimentální torzní úhly (ETKDGv3)*:
   Zohledňuje preferované konformace malých kruhů a sousedních aromatických substituentů přímo z rentgenostrukturních dat experimentálních krystalů.
2. *RMSD Prunování (Eliminace geometrických duplikátů)*:
   Algoritmus počítá vzájemnou odchylku těžkých atomů. Konformace s odchylkou $"RMSD" < 0.5 " Å"$ jsou vyřazeny, což zabraňuje zahlcení paměti RAM identickými rotamery.
3. *Thread Pinning (`num_threads=1`)*:
   Každý worker proces v multiprocessingovém poolu musí striktně běžet v jednovláknovém režimu, aby nedošlo k degradaci propustnosti CPU na vaší stanici RTX 4080 (16 workerů × 16 vláken = 256 vláken → kolaps propustnosti CPU).

#code_card(
  title: "Kompozice syntetických a konformačních filtrů v DrugEx",
  lang: "python",
  code: "from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore, MinMaxGaussian
from drugex.molecules.converters.conformer import RDKitConformerGenerator

# 1. SAScore s klesající S-křivkou (penalizace složitých struktur)
sa_scorer = Property(\"SA\", modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))
# -> SAScore 2.5 má odměnu 1.0, skóre 3.5 má odměnu 0.5, skóre 4.5 má odměnu 0.0

# 2. Gaussovský terč pro Lipofilní orální okno (LogP = 3.0 +/- 1.2)
logp_scorer = Property(\"logP\", modifier=MinMaxGaussian(mu=3.0, sigma=1.2, minimize=True))
# -> LogP 2.8 má odměnu 1.0; LogP 4.5 má odměnu 0.457 (hladký gradient)

# 3. Konformační generátor s RMSD clusteringem a thread pinningem
conf_gen = RDKitConformerGenerator(max_conformers=30, rmsd_threshold=0.5, num_threads=1)
# -> Generuje distanční geometrii ETKDGv3 bez duplikátů a bez CPU thread starve"
)
