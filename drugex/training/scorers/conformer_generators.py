"""3D Conformer generators for OpenEye Omega, Schrödinger MacroModel/confgenx, RDKit ETKDGv3, and CDPKit ConfGen."""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from drugex.training.scorers.interfaces import ConformerGenerator

try:
    from openeye import oechem, oemolprop, oeomega  # type: ignore

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
    oechem = None  # type: ignore
    oemolprop = None  # type: ignore
    oeomega = None  # type: ignore

try:
    import CDPL.Base as CDPLBase  # type: ignore
    import CDPL.Chem as CDPLChem  # type: ignore
    import CDPL.ConfGen as CDPLConfGen  # type: ignore
    import CDPL.MolProp as CDPLMolProp  # type: ignore
    from CDPL.Chem import StereoisomerGenerator  # type: ignore

    CDPL_AVAILABLE = True
except ImportError:
    CDPL_AVAILABLE = False
    CDPLChem = None  # type: ignore
    CDPLConfGen = None  # type: ignore
    CDPLBase = None  # type: ignore
    CDPLMolProp = None  # type: ignore
    StereoisomerGenerator = None  # type: ignore

logger = logging.getLogger(__name__)


class OmegaConformerGenerator(ConformerGenerator):
    """Conformer generator using OpenEye OMEGA toolkit.

    Generates low-energy 3D conformations by enumerating stereocenters with OEFlipper
    and building conformer ensembles using classical torsional driving and rule-based
    ring conformation libraries.

    Parameters
    ----------
    max_conformers : int, optional
        Maximum number of conformers to generate per molecule, capped at 200, by default 10.
    max_centers : int, optional
        Maximum number of unspecified stereocenters to enumerate, by default 4.
    max_heavy_atoms : int, optional
        Drop molecules exceeding this heavy atom count threshold, by default 35.
    max_rotatable_bonds : int, optional
        Drop molecules exceeding this rotatable bond count threshold, by default 15.
    filter : Any, optional
        Optional OpenEye OEFilter instance, filter file path, or filter type integer,
        by default None.
    use_gpu : bool, optional
        Whether to utilize GPU acceleration for torsion driving if available, by default False.
    show_progress : bool, optional
        Whether to print progress indicators during conformer generation, by default False.

    Raises
    ------
    ImportError
        If OpenEye toolkits (`openeye.oechem` or `openeye.oeomega`) are not installed.
    """

    def __init__(
        self,
        max_conformers: int = 10,
        max_centers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        filter: Any = None,
        use_gpu: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Initialize the OpenEye Omega conformer generator.

        Parameters
        ----------
        max_conformers : int, optional
            Maximum conformers per molecule, by default 10.
        max_centers : int, optional
            Maximum stereocenters to enumerate, by default 4.
        max_heavy_atoms : int, optional
            Maximum heavy atom count, by default 35.
        max_rotatable_bonds : int, optional
            Maximum rotatable bond count, by default 15.
        filter : Any, optional
            OEFilter instance or filter specification path, by default None.
        use_gpu : bool, optional
            Whether to use GPU if supported, by default False.
        show_progress : bool, optional
            Enable verbose progress dots, by default False.
        """
        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required for OmegaConformerGenerator")

        if max_conformers > 200:
            logger.warning("max_conformers > 200 may cause memory issues. Setting to 200.")
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers
        self.max_centers = max_centers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.filter = filter
        self.use_gpu = use_gpu
        self.show_progress = show_progress

    def _create_fresh_omega(self) -> Any:
        """Create a fresh OEOmega instance with conservative parameter bounds.

        Returns
        -------
        oeomega.OEOmega
            Configured OpenEye Omega conformer builder instance.
        """
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

        return oeomega.OEOmega(opts)

    def _filter_mol(self, smi: str, mol: Any) -> bool:
        """Evaluate if an OpenEye molecule passes atom count and rotatable bond filters.

        Parameters
        ----------
        smi : str
            SMILES representation for logging.
        mol : oechem.OEMolBase
            OpenEye molecule to test.

        Returns
        -------
        bool
            True if molecule should be filtered out (rejected), False if accepted.
        """
        if oechem.OECount(mol, oechem.OEIsHeavy()) > self.max_heavy_atoms:
            oechem.OEThrow.Warning(f"Skipping {smi} with > {self.max_heavy_atoms} heavy atoms")
            return True

        if oechem.OECount(mol, oechem.OEIsRotor()) > self.max_rotatable_bonds:
            oechem.OEThrow.Warning(f"Skipping {smi} with > {self.max_rotatable_bonds} rotatable bonds")
            return True

        if self.filter is not None:
            filter_type = oechem.oeifstream(self.filter) if isinstance(self.filter, str) else self.filter
            oe_filter = oemolprop.OEFilter(filter_type)
            oe_filter.SetMMFFTypeCheck(True)
            if not oe_filter(mol):
                oechem.OEThrow.Warning(f"Skipping {smi} due to: {oe_filter.GetMessage(mol)}")
                return True
        return False

    def _get_isomers(self, mol: Any) -> Iterator[Any]:
        """Enumerate stereoisomers using OpenEye OEFlipper.

        Parameters
        ----------
        mol : oechem.OEMolBase
            Input molecule.

        Yields
        ------
        oechem.OEMol
            Enumerated stereoisomer OEMol instances.
        """
        opts = oeomega.OEFlipperOptions()
        opts.SetMaxCenters(self.max_centers)
        for conf in oeomega.OEFlipper(mol, opts):
            iso = oechem.OEMol(conf)
            yield iso

    def genConformers(self, smiles_list: Sequence[str | None], out_dir: str) -> str:
        """Generate 3D conformers for input SMILES strings using OpenEye OMEGA.

        Parameters
        ----------
        smiles_list : Sequence[str | None]
            List or sequence of SMILES strings.
        out_dir : str
            Target directory to write the generated conformer archive.

        Returns
        -------
        str
            Absolute path to output `.oeb.gz` archive containing generated conformers.
        """
        os.makedirs(out_dir, exist_ok=True)
        tmp_outfile = os.path.join(out_dir, "conformers.oeb.gz")
        ofs = oechem.oemolostream()
        if not ofs.open(tmp_outfile):
            oechem.OEThrow.Fatal(f"Unable to open {tmp_outfile} for writing conformers")

        omega = self._create_fresh_omega()

        dots = None
        if self.show_progress and len(smiles_list) > 50:
            print("Generating conformers...")
            dots = oechem.OEThreadedDots(100, 50, "molecules")

        for i, smi in enumerate(smiles_list):
            if smi is None or not smi.strip():
                continue
            mol = oechem.OEMol()
            title = f"mol_{i}"
            mol.SetTitle(title)

            if not oechem.OESmilesToMol(mol, smi):
                continue

            if self._filter_mol(smi, mol):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                iso.SetTitle(f"{title}+{j}")
                ret_code = omega.Build(iso)
                if ret_code == oeomega.OEOmegaReturnCode_Success:
                    oechem.OEWriteMolecule(ofs, iso)
                else:
                    oechem.OEThrow.Warning(
                        f"{smi}: {iso.GetTitle()} {oeomega.OEGetOmegaError(ret_code)}"
                    )

            if dots:
                dots.Update()

        if dots:
            dots.Total()

        ofs.close()
        omega = None
        gc.collect()
        return tmp_outfile


class SchrodingerConformerGenerator(ConformerGenerator):
    """Conformer generator utilizing Schrödinger's `confgenx` and `sdconvert` CLI tools.

    Runs ligand preparation, stereoisomer enumeration, and torsional sampling via
    Schrödinger MacroModel/confgenx utilities. Requires the `SCHRODINGER` environment
    variable pointing to a valid installation directory.

    Parameters
    ----------
    max_conformers : int, optional
        Maximum conformers per molecule, by default 10.
    max_isomers : int, optional
        Maximum stereoisomers per molecule, by default 4.
    max_heavy_atoms : int, optional
        Maximum heavy atom count cutoff, by default 35.
    max_rotatable_bonds : int, optional
        Maximum rotatable bond count cutoff, by default 15.
    reactions : Sequence[Callable[[Chem.Mol], Chem.Mol]] | None, optional
        List of reaction/transformation callables applied to each RDKit molecule
        prior to conformer generation, by default None.

    Raises
    ------
    RuntimeError
        If `SCHRODINGER` environment variable is not defined.
    """

    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        reactions: Sequence[Callable[[Chem.Mol], Chem.Mol]] | None = None,
    ) -> None:
        """Initialize the Schrödinger conformer generator.

        Parameters
        ----------
        max_conformers : int, optional
            Maximum conformers to produce, by default 10.
        max_isomers : int, optional
            Maximum stereoisomers, by default 4.
        max_heavy_atoms : int, optional
            Heavy atom limit, by default 35.
        max_rotatable_bonds : int, optional
            Rotatable bond limit, by default 15.
        reactions : Sequence[Callable[[Chem.Mol], Chem.Mol]] | None, optional
            Pre-generation reaction functions, by default None.
        """
        if "SCHRODINGER" not in os.environ:
            raise RuntimeError("SCHRODINGER environment variable is not set")

        self.max_conformers = max_conformers
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.reactions = reactions

    def _filter_mol(self, mol: Chem.Mol | None) -> bool:
        """Filter RDKit molecules based on heavy atom and rotatable bond count thresholds.

        Parameters
        ----------
        mol : Chem.Mol | None
            RDKit molecule to inspect.

        Returns
        -------
        bool
            True if molecule should be filtered out, False otherwise.
        """
        if mol is None:
            return True

        if rdMolDescriptors.CalcNumHeavyAtoms(mol) > self.max_heavy_atoms:
            print(f"Skipping {Chem.MolToSmiles(mol)} with > {self.max_heavy_atoms} heavy atoms")
            return True

        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            print(f"Skipping {Chem.MolToSmiles(mol)} with > {self.max_rotatable_bonds} rotatable bonds")
            return True

        return False

    def apply_reactions(self, mol: Chem.Mol | None) -> Chem.Mol | None:
        """Apply pre-processing chemical reactions (e.g. beta-lactam opening).

        Parameters
        ----------
        mol : Chem.Mol | None
            RDKit molecule.

        Returns
        -------
        Chem.Mol | None
            Chemically modified molecule.
        """
        if mol is None or self.reactions is None:
            return mol

        for reaction in self.reactions:
            mol = reaction(mol)

        return mol

    def genConformers(self, smiles_list: Sequence[str | None], out_dir: str) -> str:
        """Generate 3D conformers using Schrödinger `confgenx`.

        Parameters
        ----------
        smiles_list : Sequence[str | None]
            Sequence of SMILES strings to generate conformers for.
        out_dir : str
            Directory path to write input and output files.

        Returns
        -------
        str
            Path to the output SDF file containing generated conformers.
        """
        os.makedirs(out_dir, exist_ok=True)
        currwd = os.getcwd()
        os.chdir(out_dir)

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi is not None]
        mols = [mol for mol in mols if not self._filter_mol(mol)]
        mols = [self.apply_reactions(mol) for mol in mols if mol is not None]

        tmp_infile = f"{out_dir}/input_mols.sdf"
        with Chem.SDWriter(tmp_infile) as w:
            for i, mol in enumerate(mols):
                if mol is None:
                    continue
                mol.SetProp("_Name", f"mol_{i}")
                opts = StereoEnumerationOptions(
                    tryEmbedding=False,
                    onlyUnassigned=False,
                    rand=0xF00D,
                    maxIsomers=self.max_isomers,
                )
                for j, iso in enumerate(EnumerateStereoisomers(mol, opts)):
                    iso.SetProp("_Name", f"mol_{i}+{j}")
                    iso = Chem.AddHs(iso)
                    w.write(iso)

        confgenx_cmd = [
            f"{os.environ['SCHRODINGER']}/confgenx",
            tmp_infile,
            "-NOJOBID",
            "-m",
            str(self.max_conformers),
        ]
        print("Running confgenx with command:", " ".join(confgenx_cmd))
        subprocess.run(confgenx_cmd, check=True)

        tmp_outfile = f"{out_dir}/output_conformers.sdf"
        sdconvert_cmd = [
            f"{os.environ['SCHRODINGER']}/utilities/sdconvert",
            "-imae",
            f"{os.path.basename(tmp_infile).removesuffix('.sdf')}-out.maegz",
            "-osd",
            tmp_outfile,
        ]
        subprocess.run(sdconvert_cmd, check=True)

        suppl = Chem.SDMolSupplier(tmp_outfile, removeHs=False)
        corrected_mols = []
        for mol in suppl:
            if mol is not None:
                for atom in mol.GetAtoms():
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    new_pos = (pos.x, pos.y, pos.z + 0.01)
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), new_pos)
                corrected_mols.append(mol)

        with Chem.SDWriter(tmp_outfile) as w:
            for mol in corrected_mols:
                if mol is not None:
                    w.write(mol)

        os.chdir(currwd)
        return tmp_outfile


class RDKitConformerGenerator(ConformerGenerator):
    """3D Conformer generator using RDKit's Experimental-Torsion Distance Geometry (ETKDGv3).

    Performs unassigned stereoisomer enumeration followed by multi-conformer embedding
    and RMSD pruning using RDKit's ETKDGv3 algorithm.

    Parameters
    ----------
    max_conformers : int, optional
        Maximum number of conformers to embed per stereoisomer, capped at 200, by default 10.
    max_isomers : int, optional
        Maximum number of unspecified stereoisomers to enumerate, by default 4.
    max_heavy_atoms : int, optional
        Drop molecules exceeding this heavy atom count threshold, by default 35.
    max_rotatable_bonds : int, optional
        Drop molecules exceeding this rotatable bond count threshold, by default 15.
    num_threads : int, optional
        Number of threads for ETKDG embedding (0 = all available CPU cores).
        Set to 1 when calling within multiprocessing to avoid CPU contention, by default 0.
    show_progress : bool, optional
        Whether to log progress information during conformer generation, by default False.
    """

    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        num_threads: int = 0,
        show_progress: bool = False,
    ) -> None:
        """Initialize the RDKit ETKDGv3 conformer generator.

        Parameters
        ----------
        max_conformers : int, optional
            Maximum conformers to generate per isomer, by default 10.
        max_isomers : int, optional
            Maximum stereoisomers to enumerate, by default 4.
        max_heavy_atoms : int, optional
            Heavy atom ceiling, by default 35.
        max_rotatable_bonds : int, optional
            Rotatable bond ceiling, by default 15.
        num_threads : int, optional
            Thread count for embedding (0 = all cores), by default 0.
        show_progress : bool, optional
            Whether to log generation progress, by default False.
        """
        if max_conformers > 200:
            logger.warning("max_conformers > 200 may cause memory issues. Setting to 200.")
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.num_threads = num_threads
        self.show_progress = show_progress

    def _create_fresh_etkdg(self) -> AllChem.EmbedParameters:
        """Create ETKDGv3 embedding parameters with reproducible random seed.

        Returns
        -------
        AllChem.EmbedParameters
            Configured RDKit ETKDGv3 embedding parameter object.
        """
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        params.numThreads = self.num_threads
        params.pruneRmsThresh = 0.5
        return params

    def _filter_mol(self, smi: str, mol: Chem.Mol) -> bool:
        """Check if molecule exceeds heavy atom or rotatable bond thresholds.

        Parameters
        ----------
        smi : str
            SMILES string for logging.
        mol : Chem.Mol
            RDKit molecule to test.

        Returns
        -------
        bool
            True if molecule should be filtered out, False otherwise.
        """
        if rdMolDescriptors.CalcNumHeavyAtoms(mol) > self.max_heavy_atoms:
            message = f"Skipping {smi} with > {self.max_heavy_atoms} heavy atoms"
            if self.show_progress:
                logger.warning(message)
            else:
                logger.debug(message)
            return True

        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            message = f"Skipping {smi} with > {self.max_rotatable_bonds} rotatable bonds"
            if self.show_progress:
                logger.warning(message)
            else:
                logger.debug(message)
            return True

        return False

    def _get_isomers(self, mol: Chem.Mol) -> Iterator[Chem.Mol]:
        """Enumerate unassigned stereoisomers for an RDKit molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.

        Yields
        ------
        Chem.Mol
            Enumerated stereoisomer instances.
        """
        opts = StereoEnumerationOptions()
        opts.maxIsomers = self.max_isomers
        opts.onlyUnassigned = True
        opts.tryEmbedding = False
        opts.rand = 0xC0FFEE
        for iso in EnumerateStereoisomers(mol, options=opts):
            yield iso

    def genConformers(self, smiles_list: Sequence[str | None], out_dir: str) -> str:
        """Generate 3D conformers for input SMILES using RDKit ETKDGv3 and save to SDF.

        Parameters
        ----------
        smiles_list : Sequence[str | None]
            List of SMILES strings.
        out_dir : str
            Output directory where `conformers.sdf` will be written.

        Returns
        -------
        str
            Path to the written `conformers.sdf` file.
        """
        os.makedirs(out_dir, exist_ok=True)
        tmp_outfile = os.path.join(out_dir, "conformers.sdf")
        writer = Chem.SDWriter(tmp_outfile)

        etkdg = self._create_fresh_etkdg()

        if self.show_progress and len(smiles_list) > 50:
            logger.info("Generating conformers with RDKit ETKDG")

        for i, smi in enumerate(smiles_list):
            if smi is None or not smi.strip():
                continue

            mol = Chem.MolFromSmiles(smi)
            title = f"mol_{i}"

            if mol is None:
                continue

            if self._filter_mol(smi, mol):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                iso = Chem.AddHs(iso)
                iso.SetProp("_Name", f"{title}+{j}")

                try:
                    conf_ids = AllChem.EmbedMultipleConfs(
                        iso,
                        numConfs=self.max_conformers,
                        params=etkdg,
                    )

                    if len(conf_ids) > 0:
                        for conf_id in conf_ids:
                            writer.write(iso, confId=conf_id)
                    else:
                        message = f"{smi}: {iso.GetProp('_Name')} failed conformer generation"
                        if self.show_progress:
                            logger.warning(message)
                        else:
                            logger.debug(message)

                except (RuntimeError, ValueError) as e:
                    message = f"{smi}: {iso.GetProp('_Name')} conformer generation error: {e}"
                    if self.show_progress:
                        logger.warning(message)
                    else:
                        logger.debug(message)

        writer.close()
        etkdg = None
        gc.collect()
        return tmp_outfile

    def write_conformers(self, mols: Sequence[Chem.Mol | str | None], out_file: str) -> None:
        """Write conformers for a sequence of RDKit molecules or SMILES strings to an SDF file.

        Parameters
        ----------
        mols : Sequence[Chem.Mol | str | None]
            Molecules or SMILES to process.
        out_file : str
            Target SDF file path.
        """
        smiles_list: List[str] = []
        for mol in mols:
            if mol is None:
                continue
            if isinstance(mol, str):
                smiles_list.append(mol)
            else:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi)
                except Exception:
                    continue

        output_path = Path(out_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not smiles_list:
            output_path.touch()
            return

        sdf_file = self.genConformers(smiles_list, output_path.parent.as_posix())
        if not sdf_file or not os.path.exists(sdf_file):
            output_path.touch()
            return

        if os.path.abspath(sdf_file) != str(output_path):
            shutil.move(sdf_file, output_path)


class CDPKitConformerGenerator(ConformerGenerator):
    """Conformer generator using Chemical Data Processing Toolkit (CDPKit) `CDPL.ConfGen`.

    Utilizes CDPKit's stochastic torsional sampling, force-field minimization,
    and stereoisomer enumeration following official CDPKit cookbook practices.

    Parameters
    ----------
    max_conformers : int, optional
        Maximum conformers per stereoisomer, capped at 200, by default 10.
    max_isomers : int, optional
        Maximum stereoisomers to enumerate, by default 4.
    max_heavy_atoms : int, optional
        Maximum heavy atoms allowed, by default 35.
    max_rotatable_bonds : int, optional
        Maximum rotatable bonds allowed, by default 15.
    timeout : int, optional
        Timeout limit for conformer generation per molecule in seconds, by default 3600.
    min_rmsd : float, optional
        Minimum RMSD threshold for keeping distinct conformers in Angstroms, by default 0.5.
    energy_window : float, optional
        Energy window for conformer retention above local minimum in kcal/mol, by default 20.0.
    show_progress : bool, optional
        Whether to log progress messages, by default False.

    Raises
    ------
    ImportError
        If CDPKit Python bindings (`CDPL`) are not installed.
    """

    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        timeout: int = 3600,
        min_rmsd: float = 0.5,
        energy_window: float = 20.0,
        show_progress: bool = False,
    ) -> None:
        """Initialize the CDPKit conformer generator.

        Parameters
        ----------
        max_conformers : int, optional
            Max conformers per isomer, by default 10.
        max_isomers : int, optional
            Max stereoisomers, by default 4.
        max_heavy_atoms : int, optional
            Max heavy atoms, by default 35.
        max_rotatable_bonds : int, optional
            Max rotatable bonds, by default 15.
        timeout : int, optional
            Timeout per molecule in seconds, by default 3600.
        min_rmsd : float, optional
            RMSD pruning threshold in Angstroms, by default 0.5.
        energy_window : float, optional
            Energy window in kcal/mol, by default 20.0.
        show_progress : bool, optional
            Whether to log progress, by default False.
        """
        if not CDPL_AVAILABLE:
            raise ImportError(
                "CDPKit not available. Install with: conda install -c conda-forge cdpkit "
                "or pip install cdpkit"
            )

        if max_conformers > 200:
            logger.warning("max_conformers > 200 may cause memory issues. Setting to 200.")
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers

        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.timeout = timeout
        self.min_rmsd = min_rmsd
        self.energy_window = energy_window
        self.show_progress = show_progress

    def _create_conf_generator(self) -> Any:
        """Create a CDPKit ConformerGenerator with configured parameters.

        Returns
        -------
        CDPLConfGen.ConformerGenerator
            Configured CDPKit conformer generator engine.
        """
        conf_gen = CDPLConfGen.ConformerGenerator()
        conf_gen.settings.timeout = self.timeout * 1000  # Convert to milliseconds
        conf_gen.settings.minRMSD = self.min_rmsd
        conf_gen.settings.energyWindow = self.energy_window
        conf_gen.settings.maxNumOutputConformers = self.max_conformers
        return conf_gen

    def _gen_conf_ensemble(self, mol: Any, conf_gen: Any) -> Tuple[int, int]:
        """Generate conformer ensemble for a CDPKit molecule.

        Parameters
        ----------
        mol : CDPLChem.BasicMolecule
            CDPKit molecule.
        conf_gen : CDPLConfGen.ConformerGenerator
            Configured conformer generator.

        Returns
        -------
        Tuple[int, int]
            (status_code, num_conformers_generated)
        """
        CDPLConfGen.prepareForConformerGeneration(mol)
        status = conf_gen.generate(mol)
        num_confs = conf_gen.getNumConformers()

        if status == CDPLConfGen.ReturnCode.SUCCESS or status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
            conf_gen.setConformers(mol)
        else:
            num_confs = 0

        return status, num_confs

    def _smiles_to_cdpl_mol(self, smiles: str) -> Any:
        """Parse a SMILES string into a CDPKit molecule with initialized basic properties.

        Parameters
        ----------
        smiles : str
            SMILES representation.

        Returns
        -------
        CDPLChem.BasicMolecule | None
            Parsed CDPKit molecule, or None on failure.
        """
        if not smiles:
            return None

        try:
            mol = CDPLChem.parseSMILES(smiles.strip())
            if mol is None or mol.getNumAtoms() == 0:
                return None
            CDPLChem.calcBasicProperties(mol, False)
            return mol
        except Exception:
            return None

    def _filter_mol(self, smi: str, mol: Any) -> bool:
        """Check if CDPKit molecule exceeds heavy atom or rotatable bond thresholds.

        Parameters
        ----------
        smi : str
            SMILES string for error reporting.
        mol : CDPLChem.BasicMolecule
            CDPKit molecule with initialized properties.

        Returns
        -------
        bool
            True if molecule should be filtered out, False otherwise.
        """
        if mol is None:
            return True

        try:
            heavy_atom_count = CDPLMolProp.getHeavyAtomCount(mol)
            if heavy_atom_count > self.max_heavy_atoms:
                message = f"Skipping {smi} with {heavy_atom_count} heavy atoms (max: {self.max_heavy_atoms})"
                if self.show_progress:
                    logger.warning(message)
                else:
                    logger.debug(message)
                return True

            rot_bonds = CDPLMolProp.getRotatableBondCount(mol)
            if rot_bonds > self.max_rotatable_bonds:
                message = f"Skipping {smi} with {rot_bonds} rotatable bonds (max: {self.max_rotatable_bonds})"
                if self.show_progress:
                    logger.warning(message)
                else:
                    logger.debug(message)
                return True

            return False
        except Exception:
            return True

    def _get_isomers(self, mol: Any) -> Iterator[Any]:
        """Enumerate stereoisomers for a molecule using CDPKit StereoisomerGenerator.

        Parameters
        ----------
        mol : CDPLChem.BasicMolecule
            Input CDPKit molecule.

        Yields
        ------
        CDPLChem.BasicMolecule
            Configured stereoisomer copies.
        """
        stereo_gen = StereoisomerGenerator()
        stereo_gen.enumerateAtomConfig(True)
        stereo_gen.enumerateBondConfig(True)
        stereo_gen.includeSpecifiedCenters(False)
        stereo_gen.setup(mol)

        count = 0
        while count < self.max_isomers:
            iso_mol = CDPLChem.BasicMolecule(mol)
            if not stereo_gen.generate():
                break

            atom_descriptors = stereo_gen.getAtomDescriptors()
            bond_descriptors = stereo_gen.getBondDescriptors()

            for i, desc in enumerate(atom_descriptors):
                if i < iso_mol.getNumAtoms():
                    CDPLChem.setStereoDescriptor(iso_mol.getAtom(i), desc)

            for i, desc in enumerate(bond_descriptors):
                if i < iso_mol.getNumBonds():
                    CDPLChem.setStereoDescriptor(iso_mol.getBond(i), desc)

            yield iso_mol
            count += 1

        if count == 0:
            yield mol

    def genConformers(self, smiles_list: Sequence[str | None], out_dir: str) -> str:
        """Generate 3D conformers for input SMILES using CDPKit and write to an SDF file.

        Parameters
        ----------
        smiles_list : Sequence[str | None]
            Sequence of SMILES strings to generate conformers for.
        out_dir : str
            Output directory for the conformer SDF file.

        Returns
        -------
        str
            Path to generated SDF file with conformers, or empty string if no conformers generated.
        """
        if not smiles_list:
            return ""

        os.makedirs(out_dir, exist_ok=True)
        tmp_outfile = os.path.join(out_dir, "conformers_cdpkit.sdf")

        if self.show_progress:
            logger.info("Generating conformers for %d molecules using CDPKit", len(smiles_list))

        try:
            writer = CDPLChem.FileSDFMolecularGraphWriter(tmp_outfile)
        except Exception as e:
            if self.show_progress:
                logger.error("Error creating CDPKit SDF writer: %s", e)
            return ""

        conf_gen = self._create_conf_generator()
        total_conformers = 0
        valid_molecules = 0

        status_to_str: Dict[int, str] = {
            CDPLConfGen.ReturnCode.UNINITIALIZED: "uninitialized",
            CDPLConfGen.ReturnCode.TIMEOUT: "max. processing time exceeded",
            CDPLConfGen.ReturnCode.ABORTED: "aborted",
            CDPLConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED: "force field setup failed",
            CDPLConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED: "force field structure refinement failed",
            CDPLConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET: "fragment library not available",
            CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED: "fragment conformer generation failed",
            CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT: "fragment conformer generation timeout",
            CDPLConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED: "fragment already processed",
            CDPLConfGen.ReturnCode.TORSION_DRIVING_FAILED: "torsion driving failed",
            CDPLConfGen.ReturnCode.CONF_GEN_FAILED: "conformer generation failed",
            CDPLConfGen.ReturnCode.NO_FIXED_SUBSTRUCT_COORDS: "fixed substructure atoms do not provide 3D coordinates",
        }

        for i, smi in enumerate(smiles_list):
            if not smi or smi.strip() == "":
                continue

            mol = self._smiles_to_cdpl_mol(smi)
            if mol is None:
                if self.show_progress:
                    logger.warning("Failed to parse SMILES for CDPKit: %s", smi)
                continue

            if self._filter_mol(smi, mol):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                mol_name = f"mol_{i}+{j}"
                CDPLChem.setName(iso, mol_name)

                try:
                    status, num_confs = self._gen_conf_ensemble(iso, conf_gen)

                    if (
                        status != CDPLConfGen.ReturnCode.SUCCESS
                        and status != CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY
                    ):
                        if self.show_progress:
                            error_msg = status_to_str.get(status, f"unknown status {status}")
                            logger.warning(
                                "CDPKit conformer generation failed for %s: %s",
                                mol_name,
                                error_msg,
                            )
                        continue

                    if num_confs > 0:
                        try:
                            writer.write(iso)
                            valid_molecules += 1
                            total_conformers += num_confs

                            if self.show_progress:
                                if status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
                                    logger.info(
                                        "%s: generated %d conformers (too much symmetry)",
                                        mol_name,
                                        num_confs,
                                    )
                                else:
                                    logger.info(
                                        "%s: generated %d conformer(s)",
                                        mol_name,
                                        num_confs,
                                    )
                        except Exception as e:
                            if self.show_progress:
                                logger.warning("Failed to write conformers for %s: %s", mol_name, e)
                    else:
                        if self.show_progress:
                            logger.warning("No CDPKit conformers generated for %s", mol_name)

                except Exception as e:
                    if self.show_progress:
                        logger.warning("CDPKit conformer generation error for %s: %s", mol_name, e)

        try:
            writer.close()
        except Exception:
            pass

        if self.show_progress:
            logger.info(
                "CDPKit conformer generation completed: %d molecules, %d conformers",
                valid_molecules,
                total_conformers,
            )

        gc.collect()
        return tmp_outfile if total_conformers > 0 else ""

    def write_conformers(self, mols: Sequence[Chem.Mol | str | None], out_file: str) -> None:
        """Write conformers for a sequence of molecules or SMILES to an SDF file.

        Parameters
        ----------
        mols : Sequence[Chem.Mol | str | None]
            Molecules or SMILES to generate conformers for.
        out_file : str
            Target SDF output file path.
        """
        smiles_list: List[str] = []
        for mol in mols:
            if mol is None:
                continue
            if isinstance(mol, str):
                smiles_list.append(mol)
            else:
                try:
                    smi = Chem.MolToSmiles(mol)
                    smiles_list.append(smi)
                except Exception:
                    continue

        output_path = Path(out_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not smiles_list:
            output_path.touch()
            return

        sdf_file = self.genConformers(smiles_list, output_path.parent.as_posix())
        if not sdf_file or not os.path.exists(sdf_file):
            output_path.touch()
            return

        if os.path.abspath(sdf_file) != str(output_path):
            shutil.move(sdf_file, output_path)
