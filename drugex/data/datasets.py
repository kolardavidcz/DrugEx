"""Dataset implementations for sequence- and graph-based DrugEx models."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.interfaces import DataSet, DataToLoader

if TYPE_CHECKING:
    from drugex.data.interfaces import DataSplitter


class SmilesDataSet(DataSet):
    """Encoded SMILES dataset for single-network sequence-based DrugEx models (`SequenceRNN`).

    Holds tokenized and integer-encoded SMILES representations for recurrent neural network
    training, mapping chemical tokens to vocabulary indices.
    """

    columns: Tuple[str, str] = ('Smiles', 'Token')

    def __init__(self, path: str, voc: Optional[VocSmiles] = None, rewrite: bool = False) -> None:
        """Initialize the SMILES dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset file.
        voc : VocSmiles, optional
            Vocabulary mapping SMILES tokens to indices. If None, defaults to `VocSmiles(False)`.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        """
        super().__init__(path, rewrite=rewrite)
        self.setVoc(voc if voc else VocSmiles(False))

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader:
        """Convert a 2D NumPy array of token indices into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Integer matrix of encoded SMILES sequences of shape `(n_samples, max_len)`.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : VocSmiles
            Vocabulary defining sequence dimensions and padding.

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding token index tensors.
        """
        dataset = torch.from_numpy(data).long().view(len(data), vocabulary.max_len)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """Collect and append encoded sequences generated from `SequenceCorpus`.

        Parameters
        ----------
        result : tuple
            A tuple `(encoded_data, corpus_instance)` where `encoded_data` is a list of encoded
            sequences and `corpus_instance` provides the active vocabulary.
        """
        self.updateVoc(result[1].getVoc())
        self.sendDataToFile(result[0], columns=self.getColumns())

    def getColumns(self) -> List[str]:
        """Generate TSV column header names corresponding to sequence positions.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, `'C1'`, ..., `'C{max_len-1}'`.
        """
        return ['C%d' % d for d in range(self.getVoc().max_len)]

    def readVocs(self, paths: Sequence[str], voc_class: Type[VocSmiles], *args: Any, **kwargs: Any) -> None:
        """Read and combine vocabulary files into the active vocabulary.

        Parameters
        ----------
        paths : sequence of str
            Paths to vocabulary files.
        voc_class : type of VocSmiles
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """
        super().readVocs(paths, voc_class=voc_class, *args, **kwargs)


class SmilesFragDataSet(DataSet):
    """Encoded SMILES dataset for sequence-based encoder-decoder DrugEx models (`SequenceTransformer`).

    Stores paired representations of input fragment scaffolds and output molecules
    for conditioned chemical sequence generation.
    """

    columns: Tuple[str, str] = ('Input', 'Output')

    class TargetCreator(DataToLoader):
        """Historical test data creator retained for reference."""

        class TgtData(Dataset):
            def __init__(self, seqs: Sequence[Any], ix: Sequence[Any], max_len: int = 100) -> None:
                self.max_len = max_len
                self.index = np.array(ix)
                self.map = {idx: i for i, idx in enumerate(self.index)}
                self.seq = seqs

            def __getitem__(self, i: int) -> Tuple[int, Any]:
                seq = self.seq[i]
                return i, seq

            def __len__(self) -> int:
                return len(self.seq)

            def collate_fn(self, arr: Sequence[Tuple[int, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
                collated_ix = np.zeros(len(arr), dtype=int)
                collated_seq = torch.zeros(len(arr), self.max_len).long()
                for i, (ix, tgt) in enumerate(arr):
                    collated_ix[i] = ix
                    collated_seq[i, :] = tgt
                return torch.from_numpy(collated_ix), collated_seq

        def __call__(self, data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader:
            dataset = data[:, 0]
            dataset = pd.Series(dataset).drop_duplicates()
            dataset = [seq.split(' ') for seq in dataset]
            encoded = vocabulary.encode(dataset)
            tgt_dataset = self.TgtData(encoded, ix=[vocabulary.decode(seq, is_tk=False) for seq in encoded])
            return DataLoader(tgt_dataset, batch_size=batch_size, collate_fn=tgt_dataset.collate_fn)

    def __init__(
        self,
        path: str,
        voc: Optional[VocSmiles] = None,
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[str] = None
    ) -> None:
        """Initialize the fragment-molecule SMILES dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset file.
        voc : VocSmiles, optional
            Active SMILES vocabulary. If None, defaults to `VocSmiles(True)` with fragment tokens.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        save_voc : bool, optional
            Whether to persist the vocabulary to disk alongside the dataset (default: True).
        voc_file : str, optional
            Custom path to save the vocabulary file (default: None).
        """
        super().__init__(path, rewrite=rewrite, save_voc=save_voc, voc_file=voc_file)
        self.voc = voc if voc else VocSmiles(True)

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """Collect encoded fragment-molecule pairs from `FragmentCorpusEncoder`.

        Parameters
        ----------
        result : tuple
            A tuple `(pairs_data, supplier)` where `pairs_data` contains encoded fragment-molecule
            tuples and `supplier` provides the active vocabulary.
        """
        self.updateVoc(result[1].encoder.getVoc())
        self.sendDataToFile([list(chain.from_iterable(x)) for x in result[0]], columns=self.getColumns())

    def createLoaders(
        self,
        data: np.ndarray,
        batch_size: int,
        splitter: Optional[DataSplitter] = None,
        converter: Optional[Callable[[np.ndarray, int, VocSmiles], DataLoader]] = None
    ) -> List[Any]:
        """Split data and construct PyTorch DataLoaders for each partition.

        Parameters
        ----------
        data : np.ndarray
            Dataset array to partition and convert.
        batch_size : int
            Number of samples per mini-batch.
        splitter : DataSplitter, optional
            Splitter strategy to partition data (e.g. train/test).
        converter : callable, optional
            Function converting raw split arrays into DataLoaders.

        Returns
        -------
        loaders : list of DataLoader or np.ndarray
            List of created DataLoaders for each partition.
        """
        splits = []
        if splitter:
            splits = splitter(data)
        else:
            splits.append(data)
        return [converter(split, batch_size, self.getVoc()) if converter else split for split in splits]

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader:
        """Convert concatenated fragment and molecule embeddings into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Array of shape `(n_samples, 2 * max_len)` containing concatenated fragment and molecule indices.
        batch_size : int
            Batch size for DataLoader.
        vocabulary : VocSmiles
            Vocabulary defining sequence dimensions.

        Returns
        -------
        loader : DataLoader
            DataLoader yielding `(molecule_tensor, fragment_tensor)` batches.
        """
        dataset = TensorDataset(
            torch.from_numpy(data[:, :vocabulary.max_len]).long().view(len(data), vocabulary.max_len),
            torch.from_numpy(data[:, vocabulary.max_len:]).long().view(len(data), vocabulary.max_len)
        )
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader

    def getColumns(self) -> List[str]:
        """Generate TSV column header names for concatenated fragment and molecule sequences.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, ..., `'C{2*max_len-1}'`.
        """
        return ['C%d' % d for d in range(self.getVoc().max_len * 2)]

    def readVocs(self, paths: Sequence[str], voc_class: Type[VocSmiles], *args: Any, **kwargs: Any) -> None:
        """Read and combine vocabulary files into the active vocabulary.

        Parameters
        ----------
        paths : sequence of str
            Paths to vocabulary files.
        voc_class : type of VocSmiles
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """
        super().readVocs(paths, voc_class=voc_class, *args, **kwargs)


class GraphFragDataSet(DataSet):
    """Encoded molecular graph dataset for fragment-conditioned graph transformers (`GraphTransformer`).

    Stores paired fragment-molecule graph action encodings (atom types, loci, bond types)
    for step-wise autoregressive graph generation.
    """

    def __init__(
        self,
        path: str,
        voc: Optional[VocGraph] = None,
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[str] = None
    ) -> None:
        """Initialize the graph fragment dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset file.
        voc : VocGraph, optional
            Graph vocabulary defining atom, bond, and loci tokens. If None, defaults to `VocGraph()`.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        save_voc : bool, optional
            Whether to persist the vocabulary to disk alongside the dataset (default: True).
        voc_file : str, optional
            Custom path to save the vocabulary file (default: None).
        """
        super().__init__(path, rewrite=rewrite, save_voc=save_voc, voc_file=voc_file)
        self.voc = voc if voc else VocGraph()

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """Collect encoded graph actions from `FragmentCorpusEncoder`.

        Parameters
        ----------
        result : tuple
            A tuple `(pairs_data, supplier)` where `pairs_data` contains encoded graph action
            tuples and `supplier` provides the active graph vocabulary.
        """
        self.updateVoc(result[1].encoder.getVoc())
        data = [x[0] for x in result[0]]
        self.sendDataToFile(data, columns=self.getColumns())

    def getColumns(self) -> List[str]:
        """Generate TSV column header names for 5-tuple graph action sequences.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, ..., `'C{5*max_len-1}'`.
        """
        return ['C%d' % d for d in range(self.getVoc().max_len * 5)]

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocGraph) -> DataLoader:
        """Convert a 2D/3D NumPy array of graph action encodings into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Array of graph action encodings.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : VocGraph
            Vocabulary defining graph length capacity.

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding graph action tensors.
        """
        dataset = torch.from_numpy(data).long().view(len(data), vocabulary.max_len, -1)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return loader
