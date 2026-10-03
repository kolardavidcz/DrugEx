# [PACKAGE: rdkit] WHY: Fragment modification, hydrogen manipulation, and bond formation. WHAT: Chem Mol, EditableMol, AddHs/RemoveHs, and ReplaceSubstructs.
from rdkit import Chem

from drugex.logs import logger
from drugex.molecules.converters.interfaces import MolConverter


class dummyMolsFromFragments(MolConverter):
    """Converter synthesizing complete dummy molecules from one or more fragment SMILES."""

    @staticmethod
    def addCCToFragments(frag):
        """Append an ethyl group ('CC') to a single fragment to synthesize a dummy molecule."""
        repl = Chem.MolFromSmiles('CC')
        patt = Chem.MolFromSmarts('[#1;$([#1])]')   

        try:
            fragH = Chem.AddHs(Chem.MolFromSmiles(frag))   
            molH = Chem.ReplaceSubstructs(fragH, patt, repl, replaceAll=False)
            mol = Chem.RemoveHs(molH[0])
            return Chem.MolToSmiles(mol)
        except:
            logger.debug(f"Skipped: couldn't build a molecule from the {frag} fragment.")
            return None

    @staticmethod
    def bridgeFragments(frags):
        """Bridge multiple disconnected fragments together into a single molecule via single bonds."""
        smiles = None
        try:
            frags_rdkit = [Chem.MolFromSmiles(f) for f in frags.split('.') ]
            mol = frags_rdkit[0] # set 1st fragment as base of molecule
            
            for i in range(1,len(frags_rdkit)): # iterate over other fragments to be combined with the base molecule
                frag = frags_rdkit[i] 
                natoms_frag = frag.GetNumAtoms()
                natoms_mol = mol.GetNumAtoms()

                # iterate over bridging positions until creation of valid molecule
                for j in range(natoms_mol): 
                    for k in range(natoms_frag):
                        comb = Chem.EditableMol( Chem.CombineMols(mol, frag ))
                        mpos = natoms_mol - j -1
                        fpos = natoms_mol + k
                        comb.AddBond(mpos, fpos, order=Chem.rdchem.BondType.SINGLE)
                        smiles = Chem.MolToSmiles(comb.GetMol())
                        if Chem.MolFromSmiles(smiles) is not None: break
        except Exception as e:
            logger.warning(f"Skipped: couldn't build a molecule from the {frags} fragments. Caused by: {e}")

        return smiles
            

    def __call__(self, frag):
        """Construct dummy molecules from fragments via bridging or ethyl group substitution."""
        try:
            if '.' in frag: # multiple leaf fragments
                smiles = self.bridgeFragments(frag)
            else: # single leaf fragment
                smiles = self.addCCToFragments(frag)
            return [(frag, smiles)]
        except Exception as e:
            logger.warning(f"Skipped: couldn't build a molecule from {frag}. Caused by: {e}")
            return None     