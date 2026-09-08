"""OpenEye ROCS command-line interface scorer module for 3D shape and color overlay similarity."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    from openeye import oechem, oeshape  # type: ignore

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
    oechem = None  # type: ignore
    oeshape = None  # type: ignore

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer

logger = logging.getLogger(__name__)


@contextmanager
def _managed_tmpdir() -> Iterator[str]:
    """Managed temporary directory context manager with robust exception cleanup.

    Yields
    ------
    str
        Path to newly created temporary directory.
    """
    path = tempfile.mkdtemp(prefix="cli_rocs_")
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            logger.debug("Error cleaning up temporary directory %s: %s", path, e)


class OpenEyeROCSScorer(Scorer):
    """OpenEye ROCS shape and color similarity scorer invoking the official ROCS CLI binary.

    Evaluates 3D shape and chemical feature (color) similarity against one or more reference
    queries (e.g. co-crystallized ligands or shape queries in `.sq` or `.sdf` formats).

    Features:
    - Multi-reference group support with best-match scoring across files in a group.
    - Support for ShapeTanimoto, ColorTanimoto, TanimotoCombo, and ScaledColor scoring modes.
    - Automatic multi-conformer handling with molecule ID tracking.

    Parameters
    ----------
    conformer_generator : ConformerGenerator
        Conformer generator used to produce 3D conformers for input molecules.
    references : Dict[str, Union[List[str], str]]
        Dictionary mapping reference group names to file paths (or lists of file paths).
        If a group has multiple files, the best score across all files in that group is retained.
    score_type : str, optional
        ROCS scoring metric (e.g. 'TanimotoCombo', 'ShapeTanimoto', 'ColorTanimoto'),
        by default 'TanimotoCombo'.
    shape_only : bool, optional
        If True, color features are ignored and only shape overlap is computed, by default False.
    optimize : bool, optional
        Whether to perform rigid-body alignment optimization, by default True.
    color_optimize : bool, optional
        Whether to optimize color overlap during alignment, by default True.
    color_force_field : str, optional
        Color force field to use ('ImplicitMillsDean', etc.), by default 'ImplicitMillsDean'.
    rocs_binary : str, optional
        Name of ROCS executable in system PATH, by default 'rocs'.
    binary_path : str | None, optional
        Explicit path to ROCS executable (overrides `rocs_binary` lookup), by default None.
    show_progress : bool, optional
        Whether to print timing and progress messages to stdout, by default True.

    Raises
    ------
    ImportError
        If OpenEye Python toolkits are not installed.
    FileNotFoundError
        If the ROCS command-line executable or any query reference file does not exist.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Dict[str, Union[List[str], str]],
        score_type: str = "TanimotoCombo",
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        rocs_binary: str = "rocs",
        binary_path: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize OpenEyeROCSScorer.

        Parameters
        ----------
        conformer_generator : ConformerGenerator
            Conformer generator instance.
        references : Dict[str, Union[List[str], str]]
            Target reference file mapping.
        score_type : str, optional
            Scoring function name, by default 'TanimotoCombo'.
        shape_only : bool, optional
            Shape-only flag, by default False.
        optimize : bool, optional
            Optimization flag, by default True.
        color_optimize : bool, optional
            Color optimization flag, by default True.
        color_force_field : str, optional
            Force field name, by default 'ImplicitMillsDean'.
        rocs_binary : str, optional
            Executable name, by default 'rocs'.
        binary_path : str | None, optional
            Explicit binary path, by default None.
        show_progress : bool, optional
            Progress logging flag, by default True.
        """
        super().__init__()

        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required for OpenEyeROCSScorer")

        self.conformer_generator = conformer_generator
        self.queries: Dict[str, List[str]] = {}
        for name, paths in references.items():
            if isinstance(paths, str):
                self.queries[name] = [paths]
            else:
                self.queries[name] = list(paths)

        self._validate_query_files()

        self.score_type = score_type
        self.shape_only = shape_only
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.binary_path = binary_path or rocs_binary
        self.rocs_binary = rocs_binary
        self.show_progress = show_progress

        if not shutil.which(self.binary_path):
            raise FileNotFoundError(f"ROCS binary not found: {self.binary_path}")

    def _validate_query_files(self) -> None:
        """Validate that all query reference files exist and are readable by OpenEye.

        Raises
        ------
        FileNotFoundError
            If any reference file cannot be found.
        ValueError
            If an `.sq` query file fails OpenEye shape query parsing.
        """
        assert isinstance(self.queries, dict), "references must be a dictionary"

        for name, list_of_qf in self.queries.items():
            for qf in list_of_qf:
                if not os.path.exists(qf):
                    raise FileNotFoundError(f"Reference file not found: {qf}")
                ext = oechem.OEGetFileExtension(qf)
                if ext == "sq":
                    query = oeshape.OEShapeQuery()
                    if not oeshape.OEReadShapeQuery(qf, query):
                        raise ValueError(f"Invalid reference shape query file: {qf}")
                else:
                    qfs = oechem.oemolistream()
                    if not qfs.open(qf):
                        oechem.OEThrow.Fatal(f"Unable to open reference file: '{qf}'")
                    query = oechem.OEGraphMol()
                    if not oechem.OEReadMolecule(qfs, query):
                        oechem.OEThrow.Fatal(f"Unable to read query molecule from: '{qf}'")

    def _convert_to_smiles(self, mols: Sequence[Any]) -> List[str | None]:
        """Convert a sequence of diverse molecule formats into SMILES strings.

        Parameters
        ----------
        mols : Sequence[Any]
            Sequence of SMILES strings, RDKit Mols, or OpenEye OEMols.

        Returns
        -------
        List[str | None]
            Canonicalized list of SMILES strings.
        """
        smiles_list: List[str | None] = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            elif hasattr(mol, "GetTitle"):  # OpenEye molecule
                smi = oechem.OECreateSmiString(mol)
                smiles_list.append(smi)
            elif hasattr(mol, "GetNumAtoms"):  # RDKit molecule
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                raise TypeError(f"Unsupported molecule type: {type(mol)}")
        return smiles_list

    def getScores(
        self,
        mols: Sequence[Any],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Score input molecules against reference query groups using OpenEye ROCS.

        Parameters
        ----------
        mols : Sequence[Any]
            List or sequence of molecules (SMILES, RDKit Mol, or OpenEye OEMol).
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            2D numpy array of shape `(len(mols), len(queries))` with ROCS similarity scores.
        """
        if not mols:
            if self.show_progress:
                print("No molecules to score")
            return np.zeros((0, len(self.queries)), dtype=np.float32)

        timer = oechem.OEWallTimer() if self.show_progress else None
        num_input_mols = len(mols)

        if self.show_progress:
            print(f"Starting ROCS scoring for {num_input_mols} molecules...")

        smiles_list = self._convert_to_smiles(mols)

        with _managed_tmpdir() as tmpdir:
            conf_file = self.conformer_generator.genConformers(smiles_list, tmpdir)
            scores_dict = self._score(conf_file)

        result_scores = np.zeros((num_input_mols, len(self.queries)), dtype=np.float32)
        for i, scores in enumerate(scores_dict.values()):
            for mol_id, score in scores.items():
                if 0 <= mol_id < num_input_mols:
                    result_scores[mol_id, i] = score

        if self.show_progress and timer and timer.Elapsed() > 2.0:
            print(f"ROCS scoring completed in {timer.Elapsed():.1f}s")

        return result_scores

    def _build_rocs_command(self, query_file: str, input_file: str, output_file: str) -> List[str]:
        """Construct CLI arguments for the `rocs` binary.

        Parameters
        ----------
        query_file : str
            Path to reference `.sq` or molecule query file.
        input_file : str
            Path to multi-conformer query database file (`.oeb.gz` or `.sdf`).
        output_file : str
            Path where TSV results report should be written.

        Returns
        -------
        List[str]
            CLI argument list for `subprocess.run`.
        """
        output_dir = os.path.dirname(output_file) or "."
        cmd = [
            self.binary_path,
            "-query", query_file,
            "-dbase", input_file,
            "-report", "one",
            "-reportfile", output_file,
            "-prefix", "rocs",
            "-outputdir", output_dir,
            "-stats", "best",
            "-nostructs",
            "-scdbase",
        ]

        if self.shape_only:
            cmd.extend(["-shapeonly", "true"])
        else:
            cmd.extend(["-rankby", self.score_type])
            cmd.extend(["-chemff", self.color_force_field])

        cmd.extend(["-opt", str(self.optimize).lower()])
        return cmd

    def _score(self, conf_file: str) -> Dict[str, Dict[int, float]]:
        """Score conformer file against each configured reference group.

        Parameters
        ----------
        conf_file : str
            Path to generated conformer file.

        Returns
        -------
        Dict[str, Dict[int, float]]
            Mapping of group name to `{mol_id: best_score}`.
        """
        scores_dict: Dict[str, Dict[int, float]] = {}
        for name, query_files in self.queries.items():
            if len(query_files) == 1:
                scores_dict[name] = self._score_single_query(conf_file, query_files[0])
            else:
                scores_dict[name] = self._score_multi_query(conf_file, query_files)
        return scores_dict

    def _score_multi_query(self, conf_file: str, query_files: Sequence[str]) -> Dict[int, float]:
        """Score against multiple reference files within a group and retain highest score per molecule.

        Parameters
        ----------
        conf_file : str
            Path to conformer file.
        query_files : Sequence[str]
            List of reference files in this group.

        Returns
        -------
        Dict[int, float]
            Dictionary of `{mol_id: max_score}`.
        """
        best_scores: Dict[int, float] = {}
        for query_file in query_files:
            query_scores = self._score_single_query(conf_file, query_file)
            for mol_id, score in query_scores.items():
                best_scores[mol_id] = max(score, best_scores.get(mol_id, 0.0))
        return best_scores

    def _score_single_query(self, conf_file: str, query_file: str) -> Dict[int, float]:
        """Score molecules against a single reference file.

        Parameters
        ----------
        conf_file : str
            Path to conformer database file.
        query_file : str
            Path to single reference query file.

        Returns
        -------
        Dict[int, float]
            Mapping of `{mol_id: score}`.
        """
        scores: Dict[int, float] = {}
        if self.show_progress:
            print("Scoring with reference file:", query_file)

        with _managed_tmpdir() as tmpdir:
            if not conf_file or not os.path.exists(conf_file):
                if self.show_progress:
                    print("No valid molecules written to input conformer file")
                return scores

            output_file = self._execute_rocs(query_file, conf_file, tmpdir)
            if not output_file:
                if self.show_progress:
                    print("ROCS execution failed or output file not created")
                return scores

            scores = self._parse_results(output_file)

        return scores

    def _execute_rocs(self, query_file: str, input_file: str, tmpdir: str) -> str:
        """Run the ROCS executable via subprocess.

        Parameters
        ----------
        query_file : str
            Reference query file path.
        input_file : str
            Input conformer file path.
        tmpdir : str
            Temporary execution workspace directory.

        Returns
        -------
        str
            Path to generated TSV output report file.

        Raises
        ------
        RuntimeError
            If ROCS executable fails, times out, or output file is empty.
        """
        output_file = os.path.join(tmpdir, "rocs_output.tsv")
        cmd = self._build_rocs_command(query_file, input_file, output_file)

        if not os.path.exists(query_file):
            raise RuntimeError(f"Query file not found: {query_file}")
        if not os.path.exists(input_file):
            raise RuntimeError(f"Input file not found: {input_file}")
        if not shutil.which(self.binary_path):
            raise RuntimeError(f"ROCS binary not found: {self.binary_path}")

        rocs_timer = oechem.OEWallTimer() if self.show_progress else None
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=dict(os.environ, OMP_NUM_THREADS="1"),
            )

            if self.show_progress and rocs_timer and rocs_timer.Elapsed() > 2.0:
                print(f"  ROCS execution: {rocs_timer.Elapsed():.1f}s")

            if result.returncode != 0:
                raise RuntimeError(
                    f"ROCS failed with return code {result.returncode}:\n{result.stderr}"
                )

            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                raise RuntimeError(f"ROCS output file is missing or empty: {output_file}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("ROCS execution timed out after 300 seconds")

        return output_file

    def _parse_results(self, output_file: str) -> Dict[int, float]:
        """Parse ROCS TSV output report and extract maximum score per molecule.

        Parameters
        ----------
        output_file : str
            Path to ROCS TSV report.

        Returns
        -------
        Dict[int, float]
            Mapping of `{mol_id: best_conformer_score}`.
        """
        scores: Dict[int, float] = {}
        if not os.path.exists(output_file):
            return scores

        df = pd.read_csv(output_file, sep="\t")
        if df.empty or "Name" not in df.columns or self.score_type not in df.columns:
            return scores

        for _, row in df.iterrows():
            try:
                mol_id = int(str(row["Name"]).split("+")[0].split("_")[1])
            except (IndexError, ValueError):
                continue
            conf_score = float(row[self.score_type])
            scores[mol_id] = max(scores.get(mol_id, 0.0), conf_score)

        return scores

    def getKey(self) -> List[str]:
        """Return list of identifier keys for the reference groups.

        Returns
        -------
        List[str]
            List of formatted keys: `['ROCS_<group_name>', ...]`.
        """
        return [f"ROCS_{name}" for name in self.queries.keys()]
