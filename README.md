# DrugEx
 
<img src='figures/logo.png' width=20% align=right>


> [!NOTE]
> **Upstream Repository & Modernization Pull Request**
>
> This branch is part of the [`kolardavidcz/DrugEx`](https://github.com/kolardavidcz/DrugEx) fork, based directly on [`fulopjoz/DrugEx` (`feature/rocs-scoring`)](https://github.com/fulopjoz/DrugEx/tree/feature/rocs-scoring) and the original [`CDDLeiden/DrugEx`](https://github.com/CDDLeiden/DrugEx).
> It provides upstream-ready modernizations focused on developer experience, static analysis, documentation, and modern path handling.

### Key Enhancements in this Branch

- **Full Static Type Annotations (PEP 484)**: Complete type hints across `drugex.training.generators`, `drugex.training.scorers`, and `drugex.data`, enabling full autocompletion, signature help, and static verification with Pyrefly, Pyright, and mypy.
- **NumPy Docstrings & IDE Hover Resolution**: Formatted all class and method docstrings to the NumPy standard. Consolidated constructor argument tables onto `__init__` methods as single sources of truth, resolving IDE hover popups for all model instantiations.
- **Modern `pathlib.Path` & `os.PathLike` Support**: Updated core serialization and dataset classes (`FileMonitor`, `DataSet`, `Vocabulary`, `Model`, `RandomTrainTestSplitter`) to seamlessly accept both modern `Path` objects and legacy string paths without type casting or runtime errors.
- **100% Behavioral & Runtime Compatibility**: Zero changes to underlying algorithmic mechanics, mathematical formulations, or model architectures. Fully verified with unit tests and linters.