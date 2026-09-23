#import "../nature_theme.typ": *

= Příloha B: Teorie Flexibilních Cílů, Konformačních Poolů & Biofyzika IDP

Tradiční racionální návrh léčiv založený na struktuře receptoru (_Structure-Based Drug Design_, SBDD) vychází z paradigmatu zámku a klíče (případně indukovaného přizpůsobení), formulovaného pro globulární proteiny s rigidní terciární strukturou. Tento přístup však fatálně selhává u vysoce dynamických biologických cílů, alosterických přechodových stavů a zejména u *intrinsicky neuspořádaných proteinů* (_Intrinsically Disordered Proteins_, IDP).

Tato příloha shrnuje rigorózní biofyzikální teorii konformačních krajin IDP a formuluje matematický aparát ligand-based 3D tvarového modelování, který představuje nosný pilíř této monografie.

== B.1 Biofyzika energetické krajiny: Globulární proteiny vs. IDP

Z termodynamického hlediska je konformační stav proteinu popsán volnou energií $G = H - T S$. 

#table(
  columns: (1.5fr, 2.5fr, 2.5fr),
  align: (left, left, left),
  [Vlastnost], [Globulární proteiny (Tradiční SBDD)], [Intrinsicky neuspořádané proteiny (IDP)],
  [Energetická krajina], [Hluboký nálevkovitý profil (_folding funnel_) s jediným globálním minimem volné energie.], [Mělká, silně zvlněná krajina bez hlubokého globálního minima, s mnoha degenerovanými lokálními stavy.],
  [Konformační stav], [Stabilní nativní 3D struktura s rigidními vazebnými kapsami.], [Dynamický rovnovážný *konformační pool* (ensemble) neuspořádaných řetězců.],
  [Entropický příspěvek], [Konformační entropie $S_("conf")$ je v nativním stavu silně omezena intramolekulárními kontakty.], [Vysoká konformační entropie $-T S_("conf")$ dominuje volné energii systému.],
  [Aplikovatelnost dokování], [Vysoká úspěšnost (rigidní nebo mírně flexibilní mřížka receptoru).], [Zcela selhává -- neexistuje stabilní statická vazebná kapsa.]
)

=== Proč molekulární dokování na IDP selhává?

Pokud aplikujeme klasické molekulární dokování na náhodný konformační snímek (_snapshot_) z trajektorie molekulární dynamiky IDP:
1. *Problém přizpůsobení artefaktu*: Ligand je optimalizován vůči dočasné fluktuaci, která v roztoku existuje pouze zlomky nanosekund.
2. *Entropická penalizace receptoru*: Vazba malé molekuly vyžaduje lokální znehybnění peptidového řetězce. Ztráta konformační entropie proteinu ($Delta S_("protein") << 0$) často zcela převáží enthalpický zisk z nekovalentních interakcí ($Delta H_("bind")$), což vede k falešně pozitivním predikcím s nulovou reálnou afinitou.

== B.2 Ligand-based alternativa: Tvarový otisk komplementární obálky

V situaci, kdy je proteinová struktura nestabilní, avšak máme k dispozici experimentálně potvrzené ligandy (identifikované např. NMR screeningem, povrchovou plasmonovou rezonancí či fluorescenční anizotropií), obracíme celou filosofii návrhu:

#insight_box("Ligand jako zrcadlo vazebné kapsy")[
  Pokud malá molekula vykazuje měřitelnou afinitu k IDP cíli, její bioaktivní konformace představuje fyzikální komplementární odlitek přechodného vazebného rozhraní. Namísto modelování nestabilního proteinu modelujeme *3D prostorovou a farmakoforovou obálku bioaktivního ligandu*.
]

Tento přístup využívá dva fundamentální koncepty:

=== 1. Konformační selekce vs. Indukované přizpůsobení malých molekul

Při interakci malé molekuly s IDP dochází ke vzájemnému rozpoznání typicky mechanismem *konformační selekce* (_Conformational Selection_): ligand se váže na tu minoritní frakci z konformačního poolu proteinu, která náhodnou fluktuací zaujala komplementární geometrii. Z toho plyne klíčový požadavek na generované molekuly: jejich nízkoenergetické konformace musí mít vysoký překryv s referenčním tvarem bez nutnosti překonávat vysoké vnitřní konformační pnutí ($Delta E_("strain") < 3.0$ kcal/mol).

=== 2. Konsensuální supermolekuly & Multi-Reference Grouping

Pokud je známo více aktivních ligandů s odlišnými chemotypy:
- *Konsensuální supermolekula (Supermolecule)*: Reprezentuje sjednocení sterických objemů a klíčových farmakoforových bodů všech aktivních látek do jediného referenčního tělesa. Zajišťuje, že nový kandidát může interagovat s více sub-kapsami současně.
- *Skupinové multi-referenční hodnocení (Multi-Reference Grouping)*: Pokud cíl váže molekuly ve dvou odlišných konformacích (např. vázaný stav A vs. alosterický stav B), DrugEx vyhodnocuje tvarové skóre vůči oběma referencím nezávisle a vrací vektor skóre pro Paretovskou optimalizaci.

== B.3 Matematická formulace Gaussovského tvarového a barevného překryvu

Metoda ROCS (_Rapid Overlay of Chemical Structures_) opouští reprezentaci atomů jako tvrdých koulí (van der Waalsových sfér) a modeluje každý atom jako spojitou 3D Gaussovskou funkci hustoty:

$ rho_i (bold(r)) = p_i exp (- alpha_i |bold(r) - bold(r)_i|^2) $

kde $bold(r)_i$ je polohový vektor jádra atomu $i$, $alpha_i$ je parametr určující poloměr atomu a $p_i$ je amplituda hustoty.

Sterický objem molekuly $A$ je integrálem její atomové hustoty přes celý prostor:
$ I_(A A) = integral rho_A (bold(r)) rho_A (bold(r)) dif bold(r) $

Tvarový překryv (_Shape Overlap_) mezi molekulou $A$ a referenční molekulou $B$ je dán prostorovým překryvovým integrálem:
$ I_(A B) = integral rho_A (bold(r)) rho_B (bold(r)) dif bold(r) $

=== Tvarový koeficient Tanimoto (Shape Tanimoto)

Tvarová podobnost je normalizována podle principu Tanimotova koeficientu do intervalu $[0.0, 1.0]$:

$ T_("shape") = frac(I_(A B), I_(A A) + I_(B B) - I_(A B)) $

Hodnota $T_("shape") = 1.0$ odpovídá dvěma stericky naprosto identickým molekulám v identické 3D orientaci.

=== Farmakoforový barevný koeficient (Color Tanimoto)

Pouhý sterický tvar k selektivní vazbě nestačí. ROCS definuje diskrétní množinu farmakoforových center (donory vodíkových vazeb, akceptory, anionty, kationty, hydrofobní a aromatické skupiny). Každé centrum $k$ nese farmakoforový náboj $c_k$. 

Barevný překryvový integrál $C_(A B)$ započítává pouze prostorový překryv mezi centry stejného typu (např. donor vůči donoru):

$ C_(A B) = sum_(k in "typy") integral rho_(A, k) (bold(r)) rho_(B, k) (bold(r)) dif bold(r) $

Odpovídající farmakoforová podobnost je vyjádřena jako:

$ T_("color") = frac(C_(A B), C_(A A) + C_(B B) - C_(A B)) in [0.0, 1.0] $

=== Výsledná metrika TanimotoCombo

Souhrnná míra 3D biofyzikální komplementarity v DrugEx je definována jako součet obou složek:

$ T_("combo") = T_("shape") + T_("color") in [0.0, 2.0] $

Tato hodnota je v reálném čase počítána pro všechny vygenerované konformace a předávána do Paretovské optimalizace Reinforcement Learning (RL). Tím je de novo generátor neustále směrován do konformačních a farmakoforových domén, které odpovídají reálným funkčním ligandům zkoumaného flexibilního cíle.
