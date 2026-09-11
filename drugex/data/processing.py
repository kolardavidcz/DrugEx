"""Parallel processing and data preparation routines for DrugEx datasets."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
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
        **kwargs: Any
    ) -> None:
        """Initialize the parallel standardization processor.

        Parameters
        ----------
        standardizer : MolConverter, optional
            The chemical standardizer used to convert input molecules (default: `DefaultStandardizer()`).
        n_proc : int, optional
            Number of CPU worker processes to initialize (default: all available CPU cores).
        chunk_size : int, optional
            Maximum number of molecules batched per process chunk to optimize memory usage
            and inter-process communication overhead (default: 1000).
        chunks : int, optional
            Number of chunks to divide the input data into. Defaults to `n_proc`.
            If both `chunk_size` and `chunks` are specified, `chunk_size` takes precedence.
        **kwargs : Any
            Additional keyword arguments forwarded to `ParallelProcessor`.
        """
        super().__init__(n_proc=n_proc, chunk_size=chunk_size, chunks=chunks, **kwargs)
        self.standardizer = standardizer if standardizer is not None else DefaultStandardizer()

    def apply(self, mols: Iterable[Any], collector: Optional[ResultCollector] = None) -> Optional[List[Any]]:
        """Apply defined standardization in parallel to an iterable of molecules.

        Parameters
        ----------
        mols : iterable
            Iterable containing molecules to transform (e.g. SMILES strings or Mol objects).
        collector : ResultCollector, optional
            Callable used to collect process results. If None, uses `Standardization.Collector()`.

        Returns
        -------
        results : list or None
            List of standardized molecule representations, or None if a custom non-list collector was supplied.
        """
        standardizer = ParallelSupplierEvaluator(
            StandardizedSupplier,
            kwargs={
                "standardizer": self.standardizer
            },
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            n_proc=self.nProc
        )

        collector = collector if collector else self.Collector()
        standardizer.apply(np.asarray(list(mols)), collector, desc_string="Standardizing molecules")
        return collector.getList() if hasattr(collector, 'getList') else None


class CorpusEncoder(ParallelProcessor):
    """Translates input molecules in parallel to sequence or graph representations for model training."""

    def __init__(
        self,
        corpus_class: Type[Any],
        corpus_options: Dict[str, Any],
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        """Initialize from a `Corpus` class and its constructor options.

        Parameters
        ----------
        corpus_class : type
            A `Corpus` implementation class used in the evaluation.
        corpus_options : dict
            Constructor keyword arguments for `corpus_class` (excluding the input data argument).
        n_proc : int, optional
            Number of CPU processes to allocate for evaluation (default: all available CPUs).
        chunk_size : int, optional
            Maximum chunk size per worker process to optimize memory usage (default: None).
        """
        super().__init__(n_proc, chunk_size)
        self.corpus = corpus_class
        self.options = corpus_options

    def apply(self, mols: Iterable[Any], collector: ResultCollector) -> None:
        """Apply the encoder across input molecules in parallel.

        Parameters
        ----------
        mols : iterable
            Collection of molecules to encode (e.g. SMILES strings or molecular representations).
        collector : ResultCollector
            Callback collector receiving `(encoded_data, corpus_instance)` tuples from workers.
        """
        evaluator = ParallelSupplierEvaluator(
            self.corpus,
            kwargs=self.options,
            chunk_size=self.chunkSize,
            chunks=self.chunks
        )
        evaluator.apply(mols, collector)


class RandomTrainTestSplitter(DataSplitter):
    """Facilitates random splitting into training and test partitions with an optional sample cap."""

    def __init__(self, test_size: float, max_test_size: float = 1e4, shuffle: bool = True) -> None:
        """Initialize the train-test splitter.

        Parameters
        ----------
        test_size : float
            Fraction of the dataset reserved for testing (e.g. 0.1 for 10%).
        max_test_size : float, optional
            Maximum number of samples in the test split to cap evaluation overhead (default: 10000).
        shuffle : bool, optional
            Whether to shuffle data prior to splitting (default: True).
        """
        self.testSize = test_size
        self.maxSize = max_test_size
        self.shuffle = shuffle

    def __call__(self, data: Sequence[Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
        """Split input dataset into train and test partitions.

        Parameters
        ----------
        data : sequence
            Input data array or list.

        Returns
        -------
        splits : tuple of (sequence, sequence)
            Partitioned `(train_split, test_split)`.
        """
        test_size = self.testSize
        if len(data) * test_size > self.maxSize:
            test_size = int(self.maxSize)
            logger.info(f"Capping test set size to {test_size} samples.")

        train, test = train_test_split(data, test_size=test_size, shuffle=self.shuffle)
        return train, test
