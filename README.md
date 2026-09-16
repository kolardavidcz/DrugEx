# DrugEx
 
<img src='figures/logo.png' width=20% align=right>


> [!NOTE]
> **Personal Development & Research Fork**
>
> This repository is the personal research and de novo molecular design workspace maintained by [David Kolar](https://github.com/kolardavidcz).
> It is based on [`fulopjoz/DrugEx` (`feature/rocs-scoring`)](https://github.com/fulopjoz/DrugEx/tree/feature/rocs-scoring) and the original [`CDDLeiden/DrugEx`](https://github.com/CDDLeiden/DrugEx).
>
> In addition to modernizing the DrugEx core engine with strict typing, docstrings, and `pathlib.Path` support, this branch serves as the central hub for custom target bioactivity discovery, transfer learning, and reinforcement learning pipelines.

### What Makes This Repository Special (`my_personal`)

- **Modernized DrugEx Core**: Inherits all full static type annotations (PEP 484), NumPy docstrings with IDE hover resolution, and native `pathlib.Path` / `os.PathLike` support across core classes (`FileMonitor`, `DataSet`, `Vocabulary`, `Model`).
- **`_david` End-to-End Molecular Design Pipeline**:
  - **Automated Bioactivity Auto-Discovery (`receptor_similar`)**: Automatically identifies target proteins and extracts active ligands from Papyrus for an input SMILES structure, applying progressive affinity relaxation and deduplication by highest affinity.
  - **Custom Solver & Training Pipeline (`_david/david.py`)**: End-to-end orchestration of SequenceRNN transfer learning and reinforcement learning with multi-objective Pareto scoring (QSPRPred classifiers + SAScore modifiers).
  - **Chemical Space Visualization**: Integrated TSNE manifold generation via Scaffviz and automated distribution plots.
  - **Structured Run Artifacts**: Automatically timestamps and organizes model checkpoints, generated SMILES, loss curves, and interactive chemical space plots under `_david/outputs/`.
- **Modular Import Architecture**: Explicit 1-step-back submodule aliasing (`qspr_data`, `qspr_fps`, `qspr_models`, `scaff_plot`, `manifold`) preventing namespace collisions across DrugEx, QSPRPred, and Scaffviz.