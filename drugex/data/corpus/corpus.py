"""Sequence corpus implementations for converting SMILES datasets into training token streams."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set, Union

from drugex.data.corpus.interfaces import Corpus, SequenceVocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class SequenceCorpus(Corpus):
    """Corpus supplier encoding molecular sequences (SMILES) for sequence-based models (RNN and GPT-2)."""

    def __init__(
        self,
        molecules: Union[Iterable[str], MolSupplier],
        vocabulary: Optional[SequenceVocabulary] = None,
        update_voc: bool = True,
        throw: bool = False,
        check_unique: bool = True,
    ) -> None:
        """Initialize sequence corpus with molecule stream, vocabulary, and token update/uniqueness policies."""
        super().__init__(molecules)
        self.vocabulary: SequenceVocabulary = vocabulary if vocabulary is not None else VocSmiles(False)
        self.updateVoc = update_voc
        self.throw = throw
        if self.updateVoc and self.throw:
            logger.warning("update_voc and throw cannot both be true at same time, defaulting to update_voc")
        self.checkUnique = check_unique
        self._unique: Set[str] = set()

    def saveVoc(self, path: str) -> None:
        """Save current vocabulary tokens to a text file."""
        self.vocabulary.toFile(path)

    def getVoc(self) -> SequenceVocabulary:
        """Return the active SequenceVocabulary instance."""
        return self.vocabulary

    def processMolecule(self, seq: str) -> Optional[List[int]]:
        """Tokenize and integer-encode a single SMILES sequence, filtering duplicates or unseen tokens."""
        if self.checkUnique and seq in self._unique:
            return None

        tokens: Optional[List[str]] = None
        if self.updateVoc:
            tokens = self.vocabulary.addWordsFromSeq(seq)
        elif self.throw:
            tokens = self.vocabulary.removeIfNew(seq)
        else:
            tokens = self.vocabulary.splitSequence(seq)

        if tokens:
            if self.checkUnique:
                self._unique.add(seq)
            output = self.vocabulary.encode([tokens[:-1]])
            code: List[int] = output[0].reshape(-1).tolist()
            return code
        return None
