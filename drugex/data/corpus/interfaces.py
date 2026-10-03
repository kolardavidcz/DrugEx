"""Abstract interfaces for vocabularies and molecule corpus suppliers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
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
    """Abstract base class representing a chemical token vocabulary."""

    def __init__(self, words: Sequence[str]) -> None:
        """Initialize vocabulary with a sequence of words."""
        self.words = list(words)

    def __add__(self: T_Voc, other: Vocabulary) -> T_Voc:
        """Combine tokens from two vocabularies into a new vocabulary instance."""
        return type(self)(other.words + self.words)  # type: ignore

    @abstractmethod
    def encode(self, tokens: Any, frags: Any = None) -> Any:
        """Encode token sequence or molecule representation into a numeric tensor/array."""
        pass

    @abstractmethod
    def decode(self, representation: Any, *args: Any, **kwargs: Any) -> Any:
        """Decode integer indices or matrices back into human-readable SMILES or molecule representations."""
        pass

    @staticmethod
    @abstractmethod
    def fromFile(path: Union[str, os.PathLike[str], Path], *args: Any, **kwargs: Any) -> Vocabulary:
        """Load vocabulary words from an external file."""
        pass

    @abstractmethod
    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """Persist vocabulary tokens to a text or TSV file."""
        pass


class SequenceVocabulary(Vocabulary, ABC):
    """Abstract base class for sequence-based (SMILES) vocabularies."""

    def __init__(
        self,
        encode_frags: bool,
        words: Sequence[str],
        max_len: int = 100,
        min_len: int = 10,
    ) -> None:
        """Initialize sequence vocabulary with control tokens, bounds, and token lookup indices."""
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
        """Split a SMILES string into discrete character/bracket tokens."""
        pass

    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """Save vocabulary tokens (excluding control tokens) separated by newlines."""
        with open(path, "w") as log:
            log.write("\n".join([x for x in self.words if x not in self.special]))

    def addWordsFromSeq(self, seq: str, ignoreConstraints: bool = False) -> Optional[List[str]]:
        """Tokenize a sequence and dynamically register any newly encountered tokens into the vocabulary."""
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
        """Tokenize a sequence, rejecting and returning None if it contains out-of-vocabulary tokens."""
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
    """Abstract data supplier stream generating encoded training samples from chemical representations."""

    def __init__(self, molecules: Union[Iterable[Any], MolSupplier]) -> None:
        """Initialize the corpus data supplier iterator."""
        super().__init__()
        self.molecules: Iterator[Any] = molecules if hasattr(molecules, "__next__") else iter(molecules)  # type: ignore

    def next(self) -> Any:
        """Retrieve next raw item from input stream."""
        return next(self.molecules)

    def convert(self, representation: Any) -> Any:
        """Convert a single raw representation into encoded model sample data, advancing on failure."""
        try:
            ret = self.processMolecule(representation)
        except Exception as exp:
            logger.warning(f"Exception occurred when generating corpus data for molecule: {representation}. Cause:")
            logger.exception(exp)
            return next(self)
        return ret

    @abstractmethod
    def processMolecule(self, molecule: Any) -> Any:
        """Process and encode a single molecule for generative model training."""
        pass

    @abstractmethod
    def getVoc(self) -> Vocabulary:
        """Return the Vocabulary instance used by this corpus."""
        pass
