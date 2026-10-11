"""Abstract base classes and core interfaces for DrugEx data management."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union


import numpy as np

# [PACKAGE: pandas] WHY: Tabular serialization and deserialization. WHAT: DataFrame TSV reading and writing.
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
        """Partition input data into training, testing, or validation subsets."""
        pass


class DataToLoader(ABC):
    """Abstract converter from raw dataset arrays into PyTorch DataLoaders."""

    @abstractmethod
    def __call__(self, data: Any, batch_size: int, vocabulary: Any) -> DataLoader:
        """Convert raw array data into a configured PyTorch DataLoader."""
        pass


class DataSet(ResultCollector, ABC):
    """Abstract base class managing disk serialization, vocabulary integration, and DataLoader generation for DrugEx models."""

    def __init__(
        self,
        path: Union[str, os.PathLike, Path],
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[Union[str, os.PathLike, Path]] = None,
    ) -> None:
        """Initialize dataset persistence path, directory structure, and associated vocabulary."""
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
        """Resolve the filesystem path for the serialized vocabulary file."""
        if self.voc_file:
            return self.voc_file
        else:
            return self.outpath.with_suffix(f"{self.outpath.suffix}.vocab" if self.outpath.suffix else ".vocab")

    def sendDataToFile(self, data: Sequence[Any], columns: Optional[Sequence[str]] = None) -> None:
        """Append a batch of records to the on-disk TSV dataset file."""
        header_written = self.outpath.is_file()
        open_mode = "a" if header_written else "w"
        pd.DataFrame(data, columns=columns if columns else [f"Col{x + 1}" for x in range(len(data[0]))]).to_csv(
            self.outpath, sep="\t", index=False, header=not header_written, mode=open_mode, encoding="utf-8"
        )

    def getData(self, chunk_size: Optional[int] = None) -> np.ndarray:
        """Load dataset records from disk into a NumPy array."""
        kwargs: Dict[str, Any] = dict()
        if chunk_size:
            kwargs["chunksize"] = chunk_size

        return pd.read_csv(self.outpath, sep="\t", header=0, **kwargs).to_numpy()

    def updateVoc(self, voc: Vocabulary) -> None:
        """Merge an incoming vocabulary into the active dataset vocabulary and persist if enabled."""
        if not self.voc:
            self.voc = voc
        else:
            self.voc += voc

        if self.save_voc:
            self.voc.toFile(self.getVocPath())

    def getVoc(self) -> Optional[Vocabulary]:
        """Return the active vocabulary associated with this dataset."""
        return self.voc

    def setVoc(self, voc: Vocabulary) -> None:
        """Assign the active vocabulary for this dataset."""
        self.voc = voc

    def fromFile(
        self,
        path: Union[str, os.PathLike[str], Path],
        vocs: Sequence[Union[str, os.PathLike[str], Path]] = tuple(),
        voc_class: Optional[Type[Vocabulary]] = None,
    ) -> None:
        """Load dataset from an existing file and initialize its vocabulary."""
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
        n_samples_ratio: Optional[float] = None,
    ) -> Union[DataLoader, List[DataLoader]]:
        """Convert on-disk data into one or more PyTorch DataLoaders with optional splitting/sampling."""
        split_converter = split_converter if split_converter else self.dataToLoader

        data = self.getData()
        if len(data) == 0:
            raise ValueError("DataSet is not initialized. Cannot convert to data loader.")

        if n_samples_ratio:
            n_samples = int(n_samples * n_samples_ratio)

        if n_samples > 0 and n_samples > len(data):
            logger.info(
                "Replicating original {} samples of data to have set of {} samples.".format(len(data), n_samples)
            )
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
        """Convert a raw data array into a PyTorch DataLoader."""
        pass

    def createLoaders(
        self,
        data: np.ndarray,
        batch_size: int,
        splitter: Optional[DataSplitter] = None,
        converter: Optional[Union[DataToLoader, Callable[[np.ndarray, int, Any], DataLoader]]] = None,
    ) -> List[Any]:
        """Partition data and construct DataLoader instances for each resulting subset."""
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
        **kwargs: Any,
    ) -> None:
        """Read vocabulary files from disk and combine them into the active dataset vocabulary."""
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
        """Encode parent molecule sequence into token strings and index codes."""
        pass

    @abstractmethod
    def encodeFrag(self, mol: Any, mol_tokens: Any, frag: Any) -> Optional[List[int]]:
        """Encode fragment scaffold within parent molecule context into numerical codes."""
        pass

    @abstractmethod
    def getVoc(self) -> Optional[Vocabulary]:
        """Return the active vocabulary used for encoding."""
        pass
