"""Parallel processing and data preparation routines for DrugEx datasets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union


import numpy as np

# [PACKAGE: scikit-learn] WHY: Dataset partitioning. WHAT: Random train/test split execution.
from sklearn.model_selection import train_test_split

from drugex.data.interfaces import DataSplitter
from drugex.logs import logger
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.molecules.suppliers import StandardizedSupplier
from drugex.parallel.collectors import ListExtend
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import ParallelProcessor

if TYPE_CHECKING:
    from drugex.molecules.converters.interfaces import MolConverter
    from drugex.parallel.interfaces import ResultCollector


class Standardization(ParallelProcessor):
    """Parallel molecular standardization processor.

    Applies chemical structure cleanup routines (SMILES canonicalization, salt removal,
    charge neutralization, tautomer evaluation) across multiple parallel CPU worker processes.
    """

    class Collector(ListExtend):
        def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
            self.items.extend(result[0])

    def __init__(
        self,
        standardizer: Optional[MolConverter] = None,
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = 1000,
        chunks: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize worker processes and chemical standardizer with optional chunking configuration."""
        super().__init__(n_proc=n_proc, chunk_size=chunk_size, chunks=chunks, **kwargs)
        self.standardizer = standardizer if standardizer is not None else DefaultStandardizer()

    def apply(self, mols: Iterable[Any], collector: Optional[ResultCollector] = None) -> Optional[List[Any]]:
        """Apply standardization across input molecules in parallel, returning collected structures."""
        standardizer = ParallelSupplierEvaluator(
            StandardizedSupplier,
            kwargs={"standardizer": self.standardizer},
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            n_proc=self.nProc,
        )

        collector = collector if collector else self.Collector()
        standardizer.apply(np.asarray(list(mols)), collector, desc_string="Standardizing molecules")
        return collector.getList() if hasattr(collector, "getList") else None


class CorpusEncoder(ParallelProcessor):
    """Translates input molecules in parallel to sequence or graph representations for model training."""

    def __init__(
        self,
        corpus_class: Type[Any],
        corpus_options: Dict[str, Any],
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Configure parallel worker pool with corpus class implementation and constructor options."""
        super().__init__(n_proc, chunk_size)
        self.corpus = corpus_class
        self.options = corpus_options

    def apply(self, mols: Iterable[Any], collector: ResultCollector) -> None:
        """Encode input molecules across parallel processes using configured corpus evaluator."""
        evaluator = ParallelSupplierEvaluator(
            self.corpus, kwargs=self.options, chunk_size=self.chunkSize, chunks=self.chunks
        )
        evaluator.apply(mols, collector)


class RandomTrainTestSplitter(DataSplitter):
    """Facilitates random splitting into training and test partitions with an optional sample cap."""

    def __init__(self, test_size: float, max_test_size: float = 1e4, shuffle: bool = True) -> None:
        """Configure partition ratio, maximum test set cap, and random shuffling behavior."""
        self.testSize = test_size
        self.maxSize = max_test_size
        self.shuffle = shuffle

    def __call__(self, data: Union[Sequence[Any], np.ndarray]) -> Tuple[Any, Any]:
        """Split data into train and test subsets, capping test size if exceeding maximum threshold."""
        test_size = min(int(math.ceil(len(data) * self.testSize)), int(self.maxSize))
        if len(data) * self.testSize > int(self.maxSize):
            logger.warning(
                f"To speed up the training, the test set is reduced to a random sample of {self.maxSize} from the original test!"
            )
        splits = train_test_split(data, test_size=test_size, shuffle=self.shuffle)
        return splits[0], splits[1]
