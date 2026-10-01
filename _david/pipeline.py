from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Union

# Ensure project root and _david are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "drugex").exists() or (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

#? [NumPy] Array operations for numerical conversions and pipeline metrics.
import numpy as np
#? [Pandas] Tabular data parsing for candidate hits and leads.
import pandas as pd
#? [RDKit] Chemical structure validation, SMILES standardization, and multi-record SDF suppliers.
from rdkit import Chem
from rdkit.Chem import AllChem

from _david import prepare_query_conformer
from _david import scaffold_funnel
from _david import thesis_benchmark

DEFAULT_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
DEFAULT_EPOCHS: int = 10
DEFAULT_N_SAMPLES: int = 200
DEFAULT_BATCH_SIZE: int = 64
TEST_EPOCHS: int = 1
TEST_N_SAMPLES: int = 50
MAX_SCAFFOLDS: int = 5
RANDOM_STATE: int = 42


class MoleculePipeline:
    """End-to-End Ligand-Based De Novo Drug Design Pipeline in DrugEx.

    Skeptical Academic Architecture:
    Combines global unconstrained SMILES exploration (Sequence-RNN) with medicinal-chemistry
    chemotype filtering (Bemis-Murcko funnel, PAINS elimination, heavy-atom volume gating)
    and conditioned graph lead optimization (Graph-Transformer).

    Rejection of Invalid SBDD / 2D Approaches:
    - Rejects ProLIF: ProLIF requires 3D protein coordinates (PDB 5T1A). In 100% pure LBDD,
      no receptor structure is available. Using docked complexes introduces heavy scoring bias.
    - Rejects 2D Pharmacophore filters: Flatten 3D spatial geometry into 2D topological graph bonds,
      ignoring out-of-plane chiral arrangements and steric envelopes.
    - Adopts USRCAT & Enhanced ROCS: Employs 60-dimensional Ultrafast Shape Recognition with
      CREDO Atom Types (USRCAT) alongside 10-conformer Gaussian shape/color overlap (ROCS)
      and heavy-atom volume constraints to physically prevent fragment collapse.
    - Eliminates Mock Simulation: Synthetic random-number generators have been completely
      purged; all benchmarks reflect authentic computational evaluations.

    Attributes:
        ROOT_PATH: Repository root Path.
        MODE: Execution regime ('full', 'test', or 'benchmark_only').
        EPOCHS: RL epochs executed per stage.
        N_SAMPLES: Number of candidate trajectories sampled per epoch.
        RUN_ID: Unique ISO-style timestamp for run tracking.
        BASE_OUTPUT_DIR: Target directory for run artifacts and checkpoints.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        mode: str = "full",
        epochs: int = DEFAULT_EPOCHS,
        n_samples: int = DEFAULT_N_SAMPLES,
    ) -> None:
        """Initialize MoleculePipeline.

        Args:
            output_dir: Optional custom Path for storing all run artifacts.
            mode: Pipeline mode ('full' for production RL, 'test' for rapid 1-epoch test, 'benchmark_only').
            epochs: RL training epochs.
            n_samples: Sample count per RL epoch.
        """
        self.ROOT_PATH: Path = PROJECT_ROOT
        self.MODE: str = mode.lower()
        self.EPOCHS: int = TEST_EPOCHS if self.MODE == "test" else epochs
        self.N_SAMPLES: int = TEST_N_SAMPLES if self.MODE == "test" else n_samples
        #? [DateTime] Generates ISO-like timestamp string for run directory isolation.
        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_dir is None:
            self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "outputs" / "pipeline_runs" / self.RUN_ID
        else:
            self.BASE_OUTPUT_DIR = Path(output_dir)
        self.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.PRETRAINED_RNN: Path = self.ROOT_PATH / "data" / "models" / "pretrained" / "smiles-rnn" / "Papyrus05.5_smiles_rnn_PT"
        self.PRETRAINED_GRAPH: Path = self.ROOT_PATH / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"

    def parse_molecules(self, molecules_input: Union[str, Path, Sequence[str]]) -> List[str]:
        """Parse, validate, and standardize input query molecules from string, file, or list.

        Args:
            molecules_input: Input SMILES string, path to .sdf/.tsv/.smi file, or sequence of SMILES.

        Returns:
            List of unique, valid canonical SMILES strings.

        Raises:
            ValueError: If no chemically valid molecules can be parsed.
        """
        raw_smiles: List[str] = []

        if isinstance(molecules_input, (list, tuple)):
            for item in molecules_input:
                s = str(item).strip()
                if s:
                    raw_smiles.append(s)
        elif isinstance(molecules_input, Path) or (isinstance(molecules_input, str) and Path(molecules_input).exists()):
            file_path = Path(molecules_input)
            if file_path.suffix.lower() == ".sdf":
                #? [RDKit SDMolSupplier] Parses MDL multi-record SDF coordinates.
                suppl = Chem.SDMolSupplier(str(file_path))
                for m in suppl:
                    if m is not None:
                        raw_smiles.append(Chem.MolToSmiles(m))
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tokens = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
                        if tokens:
                            raw_smiles.append(tokens[0])
        elif isinstance(molecules_input, str):
            for part in molecules_input.replace(",", "\n").replace(";", "\n").splitlines():
                part = part.strip()
                if part:
                    raw_smiles.append(part)

        valid_molecules: List[str] = []
        for smi in raw_smiles:
            #? [RDKit] Validates chemical syntax and converts to canonical isomeric SMILES.
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                can_smi = Chem.MolToSmiles(m)
                if can_smi not in valid_molecules:
                    valid_molecules.append(can_smi)

        if not valid_molecules:
            raise ValueError(f"No valid molecules found in input: {molecules_input}")

        return valid_molecules

    def stage0_prepare_reference_conformer(self, primary_molecule: str) -> Path:
        """Generate 3D reference conformer ensemble and USRCAT moment array for query molecule.

        Args:
            primary_molecule: SMILES of the active query template.

        Returns:
            Path to the written multi-conformer reference SDF file.
        """
        ref_dir: Path = self.BASE_OUTPUT_DIR / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_sdf: Path = ref_dir / "query_molecule_3d.sdf"

        print(f"\n[Stage 0] Generating 3D reference conformer ensemble (50 confs) & USRCAT moments...")
        preparer = prepare_query_conformer.QueryConformerPreparer(references_dir=ref_dir)
        preparer.generate_reference_sdf(smiles=primary_molecule, output_path=ref_sdf)

        # Ensure shared references directory is also populated
        shared_ref = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        shared_ref.parent.mkdir(parents=True, exist_ok=True)
        if not shared_ref.exists():
            preparer.generate_reference_sdf(smiles=primary_molecule, output_path=shared_ref)

        print(f"[Stage 0] Reference conformers saved to: {ref_sdf}")
        return ref_sdf

    def stage1_sequence_rnn_hit_generation(self, primary_molecule: str, ref_sdf: Path) -> tuple[pd.DataFrame, Path]:
        """Execute Stage 1 global hit exploration using Sequence-RNN policy gradient RL.

        Args:
            primary_molecule: Query active SMILES string.
            ref_sdf: Path to 3D reference conformer SDF.

        Returns:
            Tuple containing DataFrame of generated candidate hits and Path to scored TSV.
        """
        stage1_dir: Path = self.BASE_OUTPUT_DIR / "stage1_sequence_rnn"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        scored_tsv: Path = stage1_dir / "rl_generated_candidates_scored.tsv"

        print(f"\n[Stage 1] Sequence-RNN Global Hit Discovery (Pure RL exploration)...")
        print(f"[Stage 1] Executing Sequence-RNN RL exploration (epochs={self.EPOCHS}, samples={self.N_SAMPLES})...")

        from drugex.data.corpus import vocabulary
        from drugex.training import generators
        from _david import SequenceRNN

        voc = vocabulary.VocSmiles.fromFile(self.PRETRAINED_RNN / "Papyrus05.5_smiles_rnn_PT.vocab")
        agent = generators.SequenceRNN(voc, is_lstm=True, use_gpus=SequenceRNN.GPUS)
        agent.loadStatesFromFile(self.PRETRAINED_RNN / "Papyrus05.5_smiles_rnn_PT.pkg")

        rl = SequenceRNN.RAINFORCMENT_LEARNING(finetuned=agent, output_dir=stage1_dir)
        agent, df_hits = rl.RAINFORCEMENT_LEARNING(primary_molecule, epochs=self.EPOCHS, n_samples=self.N_SAMPLES)
        if not scored_tsv.exists() and (stage1_dir / "reinforcement_learning" / "rl_generated_candidates_scored.tsv").exists():
            scored_tsv = stage1_dir / "reinforcement_learning" / "rl_generated_candidates_scored.tsv"

        print(f"[Stage 1] Completed. Generated {len(df_hits)} candidates saved to: {scored_tsv}")
        return df_hits, scored_tsv

    def stage2_bemis_murcko_funnel(self, stage1_df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
        """Process candidate hits through the Bemis-Murcko chemotype funnel.

        Filters out catechol/polyphenol PAINS, enforces heavy-atom volume bounds,
        enforces scaffold hopping distance from seed, and clusters with Butina.

        Args:
            stage1_df: DataFrame of candidate molecules from Stage 1.

        Returns:
            Tuple containing DataFrame of selected scaffold leads and Path to funnel scaffolds TSV.
        """
        stage2_dir: Path = self.BASE_OUTPUT_DIR / "stage2_scaffold_funnel"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Stage 2] Bemis-Murcko Framework Extraction & Chemotype Funnel...")
        funnel = scaffold_funnel.ScaffoldFunnel(output_dir=stage2_dir)
        scaff_df = funnel.process_candidates_dataframe(stage1_df, max_scaffolds=MAX_SCAFFOLDS)

        funnel_scaffolds_tsv: Path = stage2_dir / "funnel_scaffolds.tsv"

        # Copy to shared _david folder for downstream scripts
        shared_scaff = self.ROOT_PATH / "_david" / "funnel_scaffolds.tsv"
        scaff_df.to_csv(shared_scaff, sep="\t", index=False)
        expanded_tsv = stage2_dir / "funnel_scaffolds_expanded.tsv"
        if expanded_tsv.exists():
            shared_expanded = self.ROOT_PATH / "_david" / "funnel_scaffolds_expanded.tsv"
            pd.read_csv(expanded_tsv, sep="\t").to_csv(shared_expanded, sep="\t", index=False)

        print(f"[Stage 2] Successfully extracted {len(scaff_df)} novel non-polyphenol core frameworks:")
        for idx, row in scaff_df.iterrows():
            print(f"  • Framework #{idx+1}: {row['Scaffold']} (HAC: {row.get('Scaffold_HAC', 'N/A')}, ROCS: {row['ROCS']:.2f}, RA: {row['RAscore']:.2f})")
        print(f"[Stage 2] Scaffolds exported to: {funnel_scaffolds_tsv}")
        return scaff_df, funnel_scaffolds_tsv

    def stage3_graph_transformer_optimization(
        self, primary_molecule: str, ref_sdf: Path, scaffolds_path: Path
    ) -> tuple[pd.DataFrame, Path]:
        """Execute Stage 3 fragment-conditioned lead optimization on discovered scaffolds.

        Args:
            primary_molecule: Query active SMILES string.
            ref_sdf: Path to 3D reference conformer SDF.
            scaffolds_path: Path to funnel scaffolds TSV.

        Returns:
            Tuple containing DataFrame of optimized leads and Path to scored TSV file.
        """
        stage3_dir: Path = self.BASE_OUTPUT_DIR / "stage3_graph_transformer"
        stage3_dir.mkdir(parents=True, exist_ok=True)
        leads_tsv: Path = stage3_dir / "rl_generated_candidates_scored.tsv"

        print(f"\n[Stage 3] Graph-Transformer Lead Optimization on Discovered Scaffolds...")
        print(f"[Stage 3] Executing Graph-Transformer RL lead optimization (epochs={self.EPOCHS}, samples={self.N_SAMPLES})...")

        from drugex.data.corpus import vocabulary
        from drugex.training import generators
        from _david import GraphTransformer

        voc = vocabulary.VocGraph.fromFile(self.PRETRAINED_GRAPH / "Papyrus05.5_graph_trans_PT.vocab")
        agent = generators.GraphTransformer(voc, use_gpus=GraphTransformer.GPUS)
        agent.loadStatesFromFile(self.PRETRAINED_GRAPH / "Papyrus05.5_graph_trans_PT.pkg")

        rl = GraphTransformer.RAINFORCMENT_LEARNING(finetuned=agent, output_dir=stage3_dir)
        agent, df_leads = rl.RAINFORCEMENT_LEARNING(
            primary_molecule, epochs=self.EPOCHS, n_samples=self.N_SAMPLES, scaffolds_path=scaffolds_path
        )
        if not leads_tsv.exists() and (stage3_dir / "reinforcement_learning" / "rl_generated_candidates_scored.tsv").exists():
            leads_tsv = stage3_dir / "reinforcement_learning" / "rl_generated_candidates_scored.tsv"

        print(f"[Stage 3] Completed. Optimized leads saved to: {leads_tsv}")
        return df_leads, leads_tsv

    def stage4_benchmark_and_visualizations(
        self, stage1_tsv: Path, stage3_tsv: Path
    ) -> tuple[Path, pd.DataFrame]:
        """Execute Stage 4 publication benchmark suite on authentic candidate data.

        Args:
            stage1_tsv: Path to Stage 1 hits TSV file.
            stage3_tsv: Path to Stage 3 leads TSV file.

        Returns:
            Tuple containing benchmark output directory and summary table DataFrame.
        """
        benchmark_dir: Path = self.BASE_OUTPUT_DIR / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Stage 4] Generating Publication Benchmark & Visualizations (Empirical Only)...")
        benchmark = thesis_benchmark.ThesisBenchmark(output_dir=benchmark_dir)
        table_df = benchmark.run_benchmark(
            rnn_path=stage1_tsv,
            graph_path=None,
            funnel_path=stage3_tsv,
        )
        print(f"[Stage 4] Benchmark suite completed. Figures and tables saved to: {benchmark_dir}")
        return benchmark_dir, table_df

    def run(self, molecules: Union[str, Path, Sequence[str]]) -> Dict[str, Any]:
        """Execute the complete multi-stage drug design pipeline.

        Args:
            molecules: Query SMILES string, path to molecule file, or sequence of SMILES.

        Returns:
            Dictionary containing run summary metadata and output paths.
        """
        start_time = datetime.now()
        print("=" * 80)
        print(" DRUGEX LIGAND-BASED DE NOVO DRUG DESIGN PIPELINE")
        print(f" Run ID:    {self.RUN_ID}")
        print(f" Mode:      {self.MODE.upper()}")
        print(f" Directory: {self.BASE_OUTPUT_DIR}")
        print("=" * 80)

        # 1. Parse and validate molecules
        valid_molecules = self.parse_molecules(molecules)
        primary_molecule = valid_molecules[0]
        print(f"Input molecules loaded: {len(valid_molecules)}")
        print(f"Primary query molecule: {primary_molecule}")

        # 2. Stage 0: 3D reference conformer generation
        ref_sdf = self.stage0_prepare_reference_conformer(primary_molecule)

        if self.MODE == "benchmark_only":
            print("[Pipeline] Running in benchmark_only mode. Looking for existing run files...")
            benchmark = thesis_benchmark.ThesisBenchmark(output_dir=self.BASE_OUTPUT_DIR / "benchmark")
            table_df = benchmark.run_benchmark()
            return {"run_id": self.RUN_ID, "mode": self.MODE, "status": "completed"}

        # 3. Stage 1: Sequence-RNN Hit Generation
        df_stage1, stage1_tsv = self.stage1_sequence_rnn_hit_generation(primary_molecule, ref_sdf)

        # 4. Stage 2: Bemis-Murcko Funnel
        df_scaffolds, scaffolds_tsv = self.stage2_bemis_murcko_funnel(df_stage1)

        # 5. Stage 3: Graph-Transformer Lead Optimization
        df_stage3, stage3_tsv = self.stage3_graph_transformer_optimization(primary_molecule, ref_sdf, scaffolds_tsv)

        # 6. Stage 4: Publication Benchmark & Figures
        benchmark_dir, table_df = self.stage4_benchmark_and_visualizations(stage1_tsv, stage3_tsv)

        duration = (datetime.now() - start_time).total_seconds()

        #? [JSON] Save complete run summary metadata.
        summary_info: Dict[str, Any] = {
            "run_id": self.RUN_ID,
            "mode": self.MODE,
            "input_molecules_count": len(valid_molecules),
            "primary_molecule": primary_molecule,
            "duration_seconds": round(duration, 2),
            "output_directory": str(self.BASE_OUTPUT_DIR),
            "reference_sdf": str(ref_sdf),
            "stage1_hits_file": str(stage1_tsv),
            "stage2_scaffolds_file": str(scaffolds_tsv),
            "stage3_leads_file": str(stage3_tsv),
            "benchmark_directory": str(benchmark_dir),
        }
        with open(self.BASE_OUTPUT_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_info, f, indent=4)

        print("\n" + "=" * 80)
        print(" PIPELINE EXECUTION FINISHED SUCCESSFULLY")
        print(f" Total Duration: {duration:.2f} seconds")
        print(f" All outputs organized in: {self.BASE_OUTPUT_DIR}")
        print("=" * 80)
        return summary_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Ligand-Based De Novo Drug Design Pipeline in DrugEx")
    parser.add_argument("--molecules", "-m", type=str, default=DEFAULT_MOLECULE, help="Input SMILES string or comma-separated SMILES")
    parser.add_argument("--input-file", "-f", type=Path, default=None, help="Path to input molecules file (.smi, .tsv, .csv, .sdf)")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Target output directory for pipeline run")
    parser.add_argument("--mode", choices=["full", "test", "benchmark_only"], default="full", help="Execution mode ('full' for training, 'test' for fast verification, 'benchmark_only' for existing data)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of RL epochs per stage")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of molecules sampled per epoch")

    args = parser.parse_args()

    input_data = args.input_file if args.input_file else args.molecules
    pipeline = MoleculePipeline(output_dir=args.out_dir, mode=args.mode, epochs=args.epochs, n_samples=args.n_samples)
    pipeline.run(input_data)
