"""Abstract interfaces for vocabularies and molecule corpus suppliers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier

T_Voc = TypeVar("T_Voc", bound="Vocabulary")


class Vocabulary(ABC):
    """Abstract base class representing a chemical token vocabulary.

    A vocabulary maintains a mapping between discrete tokens ('words') and integer indices,
    enabling conversion between textual/graph molecule representations and numeric tensors
    for generative model training and sampling.

    Parameters
    ----------
    words : Sequence[str]
        Initial collection of tokens forming the vocabulary.
    """

    def __init__(self, words: Sequence[str]) -> None:
        """Initialize the vocabulary with a sequence of words.

        Parameters
        ----------
        words : Sequence[str]
            Tokens included in this vocabulary.
        """
        self.words = list(words)

    def __add__(self: T_Voc, other: Vocabulary) -> T_Voc:
        """Combine tokens from two vocabularies into a new vocabulary instance.

        Parameters
        ----------
        other : Vocabulary
            Another vocabulary to merge.

        Returns
        -------
        Vocabulary
            New vocabulary containing the concatenated token list.
        """
        return type(self)(other.words + self.words)  # type: ignore

    @abstractmethod
    def encode(self, tokens: Any, frags: Any = None) -> Any:
        """Encode token sequence or molecule representation into a numeric tensor/array.

        Parameters
        ----------
        tokens : Any
            Tokens or molecule representations to encode.
        frags : Any, optional
            Fragment constraints or subgraphs, by default None.

        Returns
        -------
        Any
            Numeric array or tensor of token indices.
        """
        pass

    @abstractmethod
    def decode(self, representation: Any, *args: Any, **kwargs: Any) -> Any:
        """Decode integer indices or matrices back into human-readable SMILES or molecule representations.

        Parameters
        ----------
        representation : Any
            Tensor or array containing token indices or adjacency matrices.
        *args : Any
            Additional positional arguments (e.g. `is_tk`, `is_smiles`).
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Decoded SMILES string or tuple of (fragments, smiles).
        """
        pass

    @staticmethod
    @abstractmethod
    def fromFile(path: str, *args: Any, **kwargs: Any) -> Vocabulary:
        """Load vocabulary words from an external file.

        Parameters
        ----------
        path : str
            Path to vocabulary file.
        *args : Any
            Additional configuration parameters.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Vocabulary
            Initialized vocabulary instance.
        """
        pass

    @abstractmethod
    def toFile(self, path: str) -> None:
        """Persist vocabulary tokens to a text or TSV file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        pass


class SequenceVocabulary(Vocabulary, ABC):
    """Abstract base class for sequence-based (SMILES) vocabularies.

    Supports control tokens (`'GO'`, `'EOS'`), special delimiter characters (`'.'`),
    length filtering, and dynamic vocabulary vocabulary expansion or vocabulary pruning.

    Parameters
    ----------
    encode_frags : bool
        If True, includes fragment separation control tokens (e.g. `'_'`).
    words : Sequence[str]
        Base collection of SMILES tokens.
    max_len : int, optional
        Maximum allowable sequence length in tokens, by default 100.
    min_len : int, optional
        Minimum allowable sequence length in tokens, by default 10.
    """

    def __init__(
        self,
        encode_frags: bool,
        words: Sequence[str],
        max_len: int = 100,
        min_len: int = 10,
    ) -> None:
        """Initialize the SequenceVocabulary.

        Parameters
        ----------
        encode_frags : bool
            Whether to support fragment delimiter tokens.
        words : Sequence[str]
            Tokens defining this vocabulary.
        max_len : int, optional
            Max allowed tokens per sequence, by default 100.
        min_len : int, optional
            Min allowed tokens per sequence, by default 10.
        """
        super().__init__(words)
        if encode_frags:  # Allow fragments for fragment-based models
            self.control: Tuple[str, ...] = ("_", "GO", "EOS")  # '_' used during model fitting
            self.special: List[str] = list(self.control) + ["."]
        else:
            self.control = ("GO", "EOS")
            self.special = list(self.control)

        self.wordSet: Set[str] = set()
        if words:
            self.wordSet = set(x for x in words if x not in self.special)
        self.max_len = max_len
        self.min_len = min_len
        self.tk2ix: Dict[str, int] = {}
        self.ix2tk: Dict[int, str] = {}
        self.size: int = 0
        self.updateIndex()

    @abstractmethod
    def splitSequence(self, seq: str) -> List[str]:
        """Split a SMILES string into discrete character/bracket tokens.

        Parameters
        ----------
        seq : str
            SMILES sequence string.

        Returns
        -------
        List[str]
            List of individual tokens terminated by `'EOS'`.
        """
        pass

    def toFile(self, path: str) -> None:
        """Save vocabulary tokens (excluding control tokens) separated by newlines.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        with open(path, "w") as log:
            log.write("\n".join([x for x in self.words if x not in self.special]))

    def addWordsFromSeq(self, seq: str, ignoreConstraints: bool = False) -> Optional[List[str]]:
        """Tokenize a sequence and dynamically register any newly encountered tokens into the vocabulary.

        Parameters
        ----------
        seq : str
            SMILES sequence.
        ignoreConstraints : bool, optional
            Whether to bypass min/max length filtering, by default False.

        Returns
        -------
        Optional[List[str]]
            Extracted tokens if length criteria are satisfied, otherwise None.
        """
        token = self.splitSequence(seq)
        if ignoreConstraints or (self.min_len < len(token) <= self.max_len):
            diff = set(token) - self.wordSet
            if len(diff) > 0:
                self.wordSet.update(diff)
                self.updateIndex()
            return token
        else:
            logger.warning(
                f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). "
                f"Words found: {set(token)} (occurrence count: {len(token)}). It will be ignored."
            )
            return None

    def removeIfNew(self, seq: str, ignoreConstraints: bool = False) -> Optional[List[str]]:
        """Tokenize a sequence, rejecting and returning None if it contains out-of-vocabulary tokens.

        Parameters
        ----------
        seq : str
            SMILES sequence.
        ignoreConstraints : bool, optional
            Whether to bypass min/max length filtering, by default False.

        Returns
        -------
        Optional[List[str]]
            Extracted tokens if all tokens exist in vocabulary and meet length requirements, else None.
        """
        token = self.splitSequence(seq)
        if ignoreConstraints or (self.min_len < len(token) <= self.max_len):
            diff = set(token) - self.wordSet - set(self.special)
            if len(diff) > 0:
                logger.warning(f"Tokens: {set(diff)} do not occur in voc. Molecule: {seq} will be ignored.")
                return None
            else:
                return token
        else:
            logger.warning(
                f"Molecule does not meet min/max words requirements (min: {self.min_len}, max: {self.max_len}). "
                f"Words found: {set(token)} (occurrence count: {len(token)}). It will be ignored."
            )
            return None

    def updateIndex(self) -> None:
        """Rebuild token-to-index (`tk2ix`) and index-to-token (`ix2tk`) lookup dictionaries."""
        self.words = self.special + [x for x in sorted(self.wordSet) if x not in self.special]
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}


class Corpus(MolSupplier, ABC):
    """Abstract data supplier stream generating encoded training samples from chemical representations.

    Parameters
    ----------
    molecules : Iterable[Any] | MolSupplier
        Stream or collection of molecule inputs.
    """

    def __init__(self, molecules: Union[Iterable[Any], MolSupplier]) -> None:
        """Initialize the corpus data supplier.

        Parameters
        ----------
        molecules : Union[Iterable[Any], MolSupplier]
            Stream or iterable of input molecule representations.
        """
        super().__init__()
        self.molecules: Iterator[Any] = molecules if hasattr(molecules, "__next__") else iter(molecules)  # type: ignore

    def next(self) -> Any:
        """Retrieve next raw item from input stream.

        Returns
        -------
        Any
            Next input molecule representation.
        """
        return next(self.molecules)

    def convert(self, representation: Any) -> Any:
        """Convert a single raw representation into encoded model sample data.

        Parameters
        ----------
        representation : Any
            Raw molecule representation.

        Returns
        -------
        Any
            Processed and encoded representation, or next item on error.
        """
        try:
            ret = self.processMolecule(representation)
        except Exception as exp:
            logger.warning(f"Exception occurred when generating corpus data for molecule: {representation}. Cause:")
            logger.exception(exp)
            return next(self)
        return ret

    @abstractmethod
    def processMolecule(self, molecule: Any) -> Any:
        """Process and encode a single molecule for generative model training.

        Parameters
        ----------
        molecule : Any
            Molecule representation (e.g. SMILES string or graph object).

        Returns
        -------
        Any
            Encoded numeric tensor or list of token indices.
        """
        pass

    @abstractmethod
    def getVoc(self) -> Vocabulary:
        """Return the Vocabulary instance used by this corpus.

        Returns
        -------
        Vocabulary
            Active vocabulary instance.
        """
        pass
