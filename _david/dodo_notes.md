MOLEKULA:
C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O


PREZENTACE:

Develop integrated computational pipeline combining:
    Shape-based pharmacophore modeling
    **3D Tanimoto similarity scoring**
    Generative AI with reinforcement learning (DrugEx)
Benchmark commercial versus open-source tools for:
    Conformer generation and **3D similarity assessment**
Generate novel CCR2 allosteric modulators that are:
    Synthetically accessible
    Structurally diverse from known compounds

Objective Function:
• Maximize:** Tanimoto similarity**
• Maximize: Synthetic accessibility
• Balance: Novelty vs. similarity

==================

DISCORD:

kdyz na tu databazi chces pouzit 3D shape matching (pouzivat souradnice) tak tam je bottleneck to generovani **3D struktur (50-200) pak delat alignment**  , ale pak je jeste pristup 2D co se jmenuje **2D farmakoforove fingerprinty** ktere popisuji **farmakofor jako vektor** a pak jenom **porovnas ty dva vektor**y a spocitas **Tanimoto vzdalenost** dvou vektoru https://distancia.readthedocs.io/en/latest/Tanimoto.html
     ( A  [ A ∩ B ]  B )       |A ∩ B|        společné bity
D = 1 - ─────────────  =  1 - ───────── = 1 - ─────────────   (0 = identické, 1 = disjunktní)
        [   A ∪ B   ]          |A ∪ B|        všechny bity


tady ale musime nejak **popsat aj vazebne misto receptoru** a dostat ten vektor - interakcni profil . To znamena ze kdyz budes mit treba CCR2 receptor ktery pouzivam ja 5T1A https://www.rcsb.org/structure/5T1A a podivas se na allostericke vazebne misto (uvnitr struktury) tak tam videt primo interakce ktere jsou dulezite pro navazani molekuly a ty by se meli zachovat tak to muze byt ten pozadavek ze strany receptoru a omezeni nejak vymyslet nebo definovat (?)  a muzes na to pouzit:
- ProLif 
- PLIF
- anebo tady uplne na spodku mas rovnou ukazany ten interakcni profil VT5 ligandu co se vaze primo na allostericke misto https://plip-tool.biotec.tu-dresden.de/plip-web/plip/result/139583de-e6c1-4ac0-b45c-4fa0103ef970
