"""Abstract base classes and core interfaces for DrugEx data management."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from drugex.logs import logger
from drugex.parallel.interfaces import ResultCollector

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from drugex.data.corpus.interfaces import Vocabulary


class DataSplitter(ABC):
    """Abstract strategy for partitioning datasets into train, test, and validation subsets."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Partition the input data into subsets.

        Parameters
        ----------
        data : Any
            Input dataset to partition.

        Returns
        -------
        splits : Any
            Collection of partitioned data subsets.
        """
        pass


class DataToLoader(ABC):
    """Abstract converter from raw dataset arrays into PyTorch DataLoaders."""

    @abstractmethod
    def __call__(self, data: Any, batch_size: int, vocabulary: Any) -> DataLoader:
        """Convert array data into a PyTorch DataLoader.

        Parameters
        ----------
        data : Any
            Raw data matrix.
        batch_size : int
            Mini-batch size.
        vocabulary : Any
            Vocabulary used for token mapping and tensor shape constraints.

        Returns
        -------
        loader : DataLoader
            Configured PyTorch DataLoader.
        """
        pass


class DataSet(ResultCollector, ABC):
    """Abstract base class managing disk serialization, vocabulary integration, and DataLoader generation for DrugEx models."""

    def __init__(
        self,
        path: Union[str, os.PathLike, Path],
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[Union[str, os.PathLike, Path]] = None
    ) -> None:
        """Initialize the dataset instance.

        Parameters
        ----------
        path : str or Path
            Filesystem path for storing the serialized dataset.
        rewrite : bool, optional
            Whether to delete existing dataset files at `path` upon initialization (default: False).
        save_voc : bool, optional
            Whether to serialize the associated vocabulary alongside the dataset (default: True).
        voc_file : str or Path, optional
            Explicit file path for saving the vocabulary. If None, defaults to `f"{path}.vocab"`.
        """
        self.outpath: Path = Path(path)
        self.save_voc = save_voc
        self.voc_file: Optional[Path] = Path(voc_file) if voc_file is not None else None

        self.outpath.parent.mkdir(parents=True, exist_ok=True)
        self.voc = None
        try:
            self.fromFile(self.outpath)
            if rewrite:
                self.reset()
        except FileNotFoundError:
            logger.warning(
                f"Initialized empty dataset. The data set file does not exist (yet): {self.outpath}. "
                "You can add data by calling this instance with the appropriate parameters."
            )

    def reset(self) -> None:
        """Remove existing dataset and vocabulary files from disk."""
        logger.info(f"Initializing new {self.__class__.__name__} at {self.outpath}...")
        if self.outpath.exists():
            self.outpath.unlink(missing_ok=True)
            logger.info(f"Removed: {self.outpath}")

        voc_path = self.getVocPath()
        if voc_path.exists():
            voc_path.unlink(missing_ok=True)
            logger.info(f"Removed: {voc_path}")

        logger.info(f"{self} initialized.")

    def getVocPath(self) -> Path:
        """Determine the filesystem path for the vocabulary file.

        Returns
        -------
        path : Path
            Resolved vocabulary file path.
        """
        if self.voc_file:
            return self.voc_file
        else:
            return self.outpath.with_suffix(f"{self.outpath.suffix}.vocab" if self.outpath.suffix else ".vocab")

    def sendDataToFile(self, data: Sequence[Any], columns: Optional[Sequence[str]] = None) -> None:
        """Append a batch of records to the on-disk TSV dataset file.

        Parameters
        ----------
        data : sequence of Any
            Batch of records to serialize.
        columns : sequence of str, optional
            TSV header column names. If None, generated as `'Col1'`, `'Col2'`, etc.
        """
        header_written = self.outpath.is_file()
        open_mode = 'a' if header_written else 'w'
        pd.DataFrame(
            data,
            columns=columns if columns else [f'Col{x+1}' for x in range(len(data[0]))]
        ).to_csv(
            self.outpath,
            sep='\t',
            index=False,
            header=not header_written,
            mode=open_mode,
            encoding='utf-8'
        )

    def getData(self, chunk_size: Optional[int] = None) -> np.ndarray:
        """Load dataset records from disk into a NumPy array.

        Parameters
        ----------
        chunk_size : int, optional
            Size of chunks to read iteratively if dataset is large (default: None).

        Returns
        -------
        data : np.ndarray
            Loaded dataset matrix.
        """
        kwargs: Dict[str, Any] = dict()
        if chunk_size:
            kwargs['chunksize'] = chunk_size

        return pd.read_csv(self.outpath, sep='\t', header=0, **kwargs).to_numpy()

    def updateVoc(self, voc: Vocabulary) -> None:
        """Combine an incoming vocabulary into the active dataset vocabulary.

        Parameters
        ----------
        voc : Vocabulary
            Vocabulary instance to merge.
        """
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

        if self.save_voc:
            self.voc.toFile(self.getVocPath())

    def getVoc(self) -> Optional[Vocabulary]:
        """Return the active vocabulary associated with this dataset.

        Returns
        -------
        voc : Vocabulary or None
            Active token vocabulary.
        """
        return self.voc

    def setVoc(self, voc: Vocabulary) -> None:
        """Set the active vocabulary for this dataset.

        Parameters
        ----------
        voc : Vocabulary
            Vocabulary to assign.
        """
        self.voc = voc

    def fromFile(
        self,
        path: Union[str, os.PathLike[str], Path],
        vocs: Sequence[Union[str, os.PathLike[str], Path]] = tuple(),
        voc_class: Optional[Type[Vocabulary]] = None,
    ) -> None:
        """Load dataset from an existing file and initialize its vocabulary.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to existing TSV data file.
        vocs : sequence of (str or Path), optional
            Paths to vocabulary files to load.
        voc_class : type of Vocabulary, optional
            Vocabulary class to instantiate.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist on disk.
        """
        self.outpath = Path(path)
        if self.outpath.exists():
            if vocs and voc_class is not None:
                self.readVocs(vocs, voc_class)
        else:
            raise FileNotFoundError(f"The specified data file does not exist: {self.outpath}")

    def asDataLoader(
        self,
        batch_size: int,
        splitter: Optional[DataSplitter] = None,
        split_converter: Optional[Union[DataToLoader, Callable[[np.ndarray, int, Any], DataLoader]]] = None,
        n_samples: int = -1,
        n_samples_ratio: Optional[float] = None
    ) -> Union[DataLoader, List[DataLoader]]:
        """Convert on-disk data into one or more PyTorch DataLoaders.

        Parameters
        ----------
        batch_size : int
            Desired batch size for the DataLoader.
        splitter : DataSplitter, optional
            Optional partition strategy to split dataset into train/test subsets.
        split_converter : DataToLoader or callable, optional
            Custom converter mapping partition arrays into DataLoaders (defaults to `self.dataToLoader`).
        n_samples : int, optional
            Desired fixed sample size. If larger than dataset length, samples are replicated (default: -1).
        n_samples_ratio : float, optional
            Scaling factor applied to `n_samples` before partitioning.

        Returns
        -------
        loaders : DataLoader or list of DataLoader
            Single DataLoader if unsplit, or list of DataLoaders matching splitter partitions.
        """
        split_converter = split_converter if split_converter else self.dataToLoader

        data = self.getData()
        if len(data) == 0:
            raise ValueError("DataSet is not initialized. Cannot convert to data loader.")

        if n_samples_ratio:
            n_samples = int(n_samples * n_samples_ratio)

        if n_samples > 0 and n_samples > len(data):
            logger.info('Replicating original {} samples of data to have set of {} samples.'.format(len(data), n_samples))
            data = np.asarray(data)
            m = int(n_samples / data.shape[0])
            data = data.repeat(m, axis=0)

        results = []
        for split in self.createLoaders(data, batch_size, splitter=splitter, converter=split_converter):
            results.append(split)

        if len(results) == 1:
            return results[0]
        else:
            return results

    @staticmethod
    @abstractmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: Any) -> DataLoader:
        """Convert a raw data array into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Input dataset array.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : Any
            Active vocabulary instance.

        Returns
        -------
        loader : DataLoader
            Constructed DataLoader.
        """
        pass

    def createLoaders(
        self,
        data: np.ndarray,
        batch_size: int,
        splitter: Optional[DataSplitter] = None,
        converter: Optional[Union[DataToLoader, Callable[[np.ndarray, int, Any], DataLoader]]] = None
    ) -> List[Any]:
        """Split data and construct DataLoader instances for each partition.

        Parameters
        ----------
        data : np.ndarray
            Dataset array.
        batch_size : int
            Mini-batch size.
        splitter : DataSplitter, optional
            Partitioning strategy.
        converter : callable, optional
            Conversion function mapping arrays to DataLoaders.

        Returns
        -------
        loaders : list of Any
            List of DataLoaders or split arrays.
        """
        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)
        return [converter(split, batch_size, self.getVoc()) if converter else split for split in splits]

    def readVocs(
        self,
        paths: Sequence[Union[str, os.PathLike[str], Path]],
        voc_class: Type[Vocabulary],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Read vocabulary files from disk and combine them into the active vocabulary.

        Parameters
        ----------
        paths : sequence of (str or Path)
            File paths to vocabulary definitions.
        voc_class : type of Vocabulary
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """
        if not paths:
            raise ValueError(f"Invalid paths: {paths}.")

        vocs = [voc_class.fromFile(path, *args, **kwargs) for path in paths]
        if len(vocs) > 1:
            voc = sum(vocs[1:], start=vocs[0])
        else:
            voc = vocs[0]

        return self.setVoc(voc)


class FragmentPairEncoder(ABC):
    """Abstract encoder for fragment-molecule pairs used in conditioned DrugEx models."""

    @abstractmethod
    def encodeMol(self, mol: Any) -> Tuple[Optional[List[str]], Optional[List[int]]]:
        """Encode parent molecule sequence into token strings and index codes.

        Parameters
        ----------
        mol : Any
            Parent molecule representation (e.g. SMILES string).

        Returns
        -------
        result : tuple of (list of str or None, list of int or None)
            Extracted token sequence and corresponding numerical index codes.
        """
        pass

    @abstractmethod
    def encodeFrag(self, mol: Any, mol_tokens: Any, frag: Any) -> Optional[List[int]]:
        """Encode fragment within parent molecule context.

        Parameters
        ----------
        mol : Any
            Parent molecule representation.
        mol_tokens : Any
            Extracted parent molecule token sequence.
        frag : Any
            Fragment representation.

        Returns
        -------
        codes : list of int or None
            Encoded numerical representation of the fragment-molecule pair.
        """
        pass

    @abstractmethod
    def getVoc(self) -> Optional[Vocabulary]:
        """Return the active vocabulary used for encoding.

        Returns
        -------
        vocabulary : Vocabulary or None
            Active vocabulary instance.
        """
        pass
