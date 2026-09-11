"""Encoders, suppliers, and splitters for fragment-conditioned molecular generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.interfaces import DataSplitter, FragmentPairEncoder
from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.interfaces import MolSupplier
from drugex.parallel.collectors import ListExtend
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import ParallelProcessor

if TYPE_CHECKING:
    from drugex.parallel.interfaces import ResultCollector


class SequenceFragmentEncoder(FragmentPairEncoder):
    """Fragment-molecule pair encoder for sequence-based models (`SequenceTransformer`)."""

    def __init__(
        self,
        vocabulary: VocSmiles = VocSmiles(True),
        update_voc: bool = True,
        throw: bool = False
    ) -> None:
        """Initialize the sequence fragment encoder.

        Parameters
        ----------
        vocabulary : VocSmiles, optional
            Vocabulary supporting fragment delimiter tokens (default: `VocSmiles(True)`).
        update_voc : bool, optional
            Whether to dynamically append newly encountered tokens to vocabulary (default: True).
        throw : bool, optional
            Whether to strip newly discovered tokens failing constraints (default: False).
        """
        self.vocabulary: VocSmiles = vocabulary
        self.updateVoc: bool = update_voc
        self.throw: bool = throw

    def encodeMol(self, sequence: str) -> Tuple[Optional[List[str]], Optional[List[int]]]:
        """Tokenize and integer-encode parent molecule SMILES sequence.

        Parameters
        ----------
        sequence : str
            Parent molecule SMILES string.

        Returns
        -------
        result : tuple of (list of str or None, list of int or None)
            Extracted token sequence and corresponding numerical index codes.
        """
        tokens = None
        if self.updateVoc:
            tokens = self.vocabulary.addWordsFromSeq(sequence)
        elif self.throw:
            tokens = self.vocabulary.removeIfNew(sequence)

        if tokens:
            output = self.vocabulary.encode([tokens[:-1]])
            code = output[0].reshape(-1).tolist()
            return tokens, code
        return tokens, None

    def encodeFrag(self, mol: str, mol_tokens: Sequence[str], frag: str) -> Optional[List[int]]:
        """Encode fragment scaffold within parent molecule context.

        Parameters
        ----------
        mol : str
            Parent molecule SMILES string.
        mol_tokens : sequence of str
            Tokenized parent molecule representation.
        frag : str
            Scaffold fragment SMILES string.

        Returns
        -------
        code : list of int or None
            Integer-encoded fragment sequence, or None if encoding fails.
        """
        tokens = None
        if self.updateVoc:
            tokens = self.vocabulary.addWordsFromSeq(frag, ignoreConstraints=True)
        elif self.throw:
            tokens = self.vocabulary.removeIfNew(frag, ignoreConstraints=True)

        if tokens:
            output = self.vocabulary.encode([tokens[:-1]])
            code = output[0].reshape(-1).tolist()
            return code
        return None

    def getVoc(self) -> VocSmiles:
        """Return active SMILES vocabulary.

        Returns
        -------
        voc : VocSmiles
            Active vocabulary instance.
        """
        return self.vocabulary


class GraphFragmentEncoder(FragmentPairEncoder):
    """Encodes fragment-molecule pairs for graph-based models (`GraphTransformer`)."""

    def __init__(self, vocabulary: VocGraph = VocGraph()) -> None:
        """Initialize graph fragment encoder.

        Parameters
        ----------
        vocabulary : VocGraph, optional
            Graph vocabulary defining atom, bond, and loci tokens (default: `VocGraph()`).
        """
        self.vocabulary: VocGraph = vocabulary

    def encodeMol(self, smiles: str) -> Tuple[str, str]:
        """Pass parent molecule SMILES through as raw token representations.

        Parameters
        ----------
        smiles : str
            Input parent molecule SMILES string.

        Returns
        -------
        result : tuple of (str, str)
            Pair `(smiles, smiles)` matching interface contract.
        """
        return smiles, smiles

    def encodeFrag(self, mol: str, mol_tokens: Any, frag: str) -> Optional[List[int]]:
        """Encode paired fragment scaffold and parent molecule into graph actions.

        Parameters
        ----------
        mol : str
            Parent molecule SMILES.
        mol_tokens : Any
            Parent molecule token representation.
        frag : str
            Fragment scaffold SMILES.

        Returns
        -------
        code : list of int or None
            Encoded graph action tuple indices, or None if encoding fails.
        """
        if mol == frag:
            return None
        try:
            output = self.vocabulary.encode([mol], [frag])
            f, s = self.vocabulary.decode(output)

            assert mol == s[0]
            code = output[0].reshape(-1).tolist()
            return code
        except Exception as exp:
            logger.warn(f'The following exception occured while encoding fragment {frag} for molecule {mol}: {exp}')
            return None

    def getVoc(self) -> VocGraph:
        """Return active graph vocabulary.

        Returns
        -------
        voc : VocGraph
            Active graph vocabulary instance.
        """
        return self.vocabulary


class FragmentPairsEncodedSupplier(MolSupplier):
    """Iteratively transforms raw fragment-molecule pairs into model-ready integer encodings."""

    class FragmentEncodingException(ConversionException):
        """Raised when a fragment fails to encode."""
        pass

    class MoleculeEncodingException(ConversionException):
        """Raised when a parent molecule fails to encode."""
        pass

    def __init__(self, pairs: Iterable[Tuple[str, str]], encoder: FragmentPairEncoder) -> None:
        """Initialize from fragment-molecule pairs.

        Parameters
        ----------
        pairs : iterable of (str, str)
            Sequence of `(fragment_smiles, molecule_smiles)` tuples.
        encoder : FragmentPairEncoder
            Encoder implementing `encodeMol` and `encodeFrag`.
        """
        self.encoder: FragmentPairEncoder = encoder
        self.pairs: Iterator[Tuple[str, str]] = iter(pairs)

    def next(self) -> Tuple[List[int], List[int]]:
        """Fetch and encode next fragment-molecule pair.

        Returns
        -------
        pair : tuple of (list of int, list of int)
            Encoded `(encoded_fragment, encoded_molecule)`.

        Raises
        ------
        MoleculeEncodingException
            If parent molecule fails to encode.
        FragmentEncodingException
            If fragment scaffold fails to encode.
        """
        pair = next(self.pairs)

        tokens, encoded_mol = self.encoder.encodeMol(pair[1])
        if not tokens or encoded_mol is None:
            raise self.MoleculeEncodingException(f'Failed to encode molecule: {pair[1]}')

        encoded_frag = self.encoder.encodeFrag(pair[1], tokens, pair[0])
        if not encoded_frag:
            raise self.FragmentEncodingException(f'Failed to encode fragment {pair[0]} from molecule: {pair[1]}')

        return encoded_frag, encoded_mol


class FragmentPairsSupplier(MolSupplier):
    """Generates fragment-molecule candidate pairs from input molecules via a fragmenter algorithm."""

    def __init__(self, molecules: Iterable[Any], fragmenter: Any, max_bonds: Optional[int] = None) -> None:
        """Initialize supplier with input molecules and fragmenter.

        Parameters
        ----------
        molecules : iterable
            Input molecules as list or iterator.
        fragmenter : callable
            Fragmentation strategy returning `(fragment, molecule)` pairs.
        max_bonds : int, optional
            Maximum allowed number of cut bonds (default: None).
        """
        self.molecules: Iterator[Any] = molecules if hasattr(molecules, "__next__") else iter(molecules)
        self.fragmenter: Any = fragmenter
        self.currentBatch: Optional[Iterator[Tuple[str, str]]] = None
        self.maxBonds: Optional[int] = max_bonds

    def next(self) -> Optional[Tuple[str, str]]:
        """Yield next available fragment-molecule pair.

        Returns
        -------
        pair : tuple of (str, str) or None
            A `(fragment, molecule)` pair, or None upon batch completion.
        """
        if not self.currentBatch:
            batch = None
            while not batch:
                batch = self.fragmenter(next(self.molecules))
            self.currentBatch = iter(batch)
        try:
            frags = next(self.currentBatch)
        except StopIteration:
            self.currentBatch = None
            return None
        return frags


class FragmentCorpusEncoder(ParallelProcessor):
    """Fragments and encodes chemical compounds in parallel across CPU worker processes."""

    class FragmentPairsCollector(ListExtend):
        """Collector extending an internal list with optional callback chaining."""

        def __init__(self, other: Optional[ResultCollector] = None) -> None:
            super().__init__()
            self.other: Optional[ResultCollector] = other

        def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
            self.items.extend(result[0])
            if self.other:
                self.other(result)

    def __init__(
        self,
        fragmenter: Any,
        encoder: FragmentPairEncoder,
        pairs_splitter: Optional[DataSplitter] = None,
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        """Initialize the parallel fragment corpus encoder.

        Parameters
        ----------
        fragmenter : callable
            Fragmentation strategy decomposing molecules into fragments.
        encoder : FragmentPairEncoder
            Encoder translating molecules and fragments into vocabulary tokens.
        pairs_splitter : DataSplitter, optional
            Splitter strategy partitioning generated pairs (default: None).
        n_proc : int, optional
            Number of parallel CPU worker processes (default: all available CPUs).
        chunk_size : int, optional
            Maximum chunk size per worker process (default: None).
        """
        super().__init__(n_proc, chunk_size)
        self.fragmenter: Any = fragmenter
        self.encoder: FragmentPairEncoder = encoder
        self.pairsSplitter: Optional[DataSplitter] = pairs_splitter

    def getFragmentPairs(self, mols: Iterable[str], collector: ResultCollector) -> None:
        """Decompose molecules into fragment pairs in parallel.

        Parameters
        ----------
        mols : iterable of str
            Input molecule SMILES strings.
        collector : ResultCollector
            Collector accumulating generated pairs.
        """
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsSupplier,
            kwargs={
                "fragmenter": self.fragmenter
            },
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            n_proc=self.nProc
        )
        evaluator.apply(mols, collector, desc_string="Creating fragment-molecule pairs")

    def splitFragmentPairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Sequence[Tuple[str, str]]]:
        """Partition generated fragment-molecule pairs using configured splitter.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Generated pairs.

        Returns
        -------
        splits : sequence of sequence of (str, str)
            Partitioned pair subsets.
        """
        return self.pairsSplitter(pairs) if self.pairsSplitter else [pairs]

    def encodeFragments(self, pairs: Sequence[Tuple[str, str]], collector: Optional[ResultCollector]) -> None:
        """Encode fragment-molecule pairs into numerical tokens in parallel.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Fragment-molecule pairs to encode.
        collector : ResultCollector, optional
            Collector accumulating encoded token batches.
        """
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={
                'encoder': self.encoder,
            },
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            n_proc=self.nProc
        )
        evaluator.apply(pairs, collector, desc_string="Encoding fragment-molecule pairs.")

    def apply(
        self,
        mols: Iterable[str],
        fragmentPairsCollector: Optional[ResultCollector] = None,
        encodingCollectors: Optional[Sequence[ResultCollector]] = None
    ) -> None:
        """Decompose molecules and encode fragment-molecule pairs across parallel CPU workers.

        Parameters
        ----------
        mols : iterable of str
            Molecule SMILES strings.
        fragmentPairsCollector : ResultCollector, optional
            Optional collector receiving intermediate fragment-molecule pairs.
        encodingCollectors : sequence of ResultCollector, optional
            List of collectors corresponding to each partition split.

        Raises
        ------
        RuntimeError
            If `encodingCollectors` length does not match partition count.
        """
        pairs_collector = self.FragmentPairsCollector(fragmentPairsCollector)
        self.getFragmentPairs(mols, pairs_collector)
        splits = self.splitFragmentPairs(pairs_collector.getList())
        if encodingCollectors and len(encodingCollectors) != len(splits):
            raise RuntimeError(f'The number of encoding collectors must match the number of splits: {len(encodingCollectors)} != {len(splits)}')
        for split_idx in range(len(splits)):
            self.encodeFragments(splits[split_idx], encodingCollectors[split_idx] if encodingCollectors else None)


class FragmentPairsSplitter(DataSplitter):
    """Partitions fragment-molecule pairs so that scaffold fragments do not overlap between splits."""

    def __init__(
        self,
        ratio: float = 0.2,
        max_test_samples: float = 1e4,
        train_collector: Optional[ResultCollector] = None,
        test_collector: Optional[ResultCollector] = None,
        unique_collector: Optional[ResultCollector] = None,
        make_unique: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """Initialize fragment pair splitter.

        Parameters
        ----------
        ratio : float, optional
            Fraction of unique fragments assigned to test partition (default: 0.2).
        max_test_samples : float, optional
            Maximum count threshold for test samples (default: 10000).
        train_collector : ResultCollector, optional
            Collector accumulating training pairs.
        test_collector : ResultCollector, optional
            Collector accumulating testing pairs.
        unique_collector : ResultCollector, optional
            Collector accumulating single-representative unique fragment pairs.
        make_unique : bool, optional
            Whether to return a unique fragment subset (default: False).
        seed : int, optional
            Random state seed for reproducible sampling (default: None).
        """
        self.ratio: float = ratio
        self.maxTestSamples: float = max_test_samples
        self.uniqueCollect: Optional[ResultCollector] = unique_collector
        self.trainCollect: Optional[ResultCollector] = train_collector
        self.testCollect: Optional[ResultCollector] = test_collector
        self.makeUnique: bool = make_unique
        self.seed: Optional[int] = seed

    def __call__(self, pairs: Sequence[Tuple[str, str]]) -> Union[Tuple[List[Any], List[Any], List[Any]], Tuple[List[Any], List[Any]]]:
        """Partition fragment-molecule pairs based on fragment identity.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Input `(fragment, molecule)` pairs.

        Returns
        -------
        splits : tuple
            `(test_pairs, train_pairs, unique_pairs)` if `make_unique=True`, else `(test_pairs, train_pairs)`.
        """
        df = pd.DataFrame(pairs, columns=["Frags", "Smiles"])
        frags = set(df.Frags)
        test_len = int(len(frags) * self.ratio)
        if self.seed:
            test_in = df.Frags.drop_duplicates().sort_values()
        else:
            test_in = df.Frags.drop_duplicates()
        if test_len > int(self.maxTestSamples):
            logger.warning(f'To speed up the training, the test set size was automatically capped at {self.maxTestSamples} fragments instead of the default 10% of original data, which would have been: {test_len}.')
            test_in = test_in.sample(int(self.maxTestSamples), random_state=self.seed)
        else:
            test_in = test_in.sample(test_len, random_state=self.seed)
        test = df[df.Frags.isin(test_in)].values.tolist()
        train = df[~df.Frags.isin(test_in)].values.tolist()
        unique = None
        if self.makeUnique:
            unique = df.drop_duplicates(subset="Frags").values.tolist()

        if self.trainCollect:
            self.trainCollect(train)
        if self.testCollect:
            self.testCollect(test)
        if self.uniqueCollect and unique is not None:
            self.uniqueCollect(unique)

        if unique:
            return test, train, unique
        else:
            return test, train
