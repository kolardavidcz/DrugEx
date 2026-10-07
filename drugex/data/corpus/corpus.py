"""Sequence corpus implementations for converting SMILES datasets into training token streams."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set, Union

from drugex.data.corpus.interfaces import Corpus, SequenceVocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class SequenceCorpus(Corpus):
    """Corpus supplier encoding molecular sequences (SMILES) for sequence-based models (RNN and GPT-2).

    Parameters
    ----------
    molecules : Union[Iterable[str], MolSupplier]
        Stream or collection of SMILES strings.
    vocabulary : SequenceVocabulary | None, optional
        Vocabulary instance used for tokenization and index mapping, by default VocSmiles(False).
    update_voc : bool, optional
        If True, new tokens encountered in the dataset are dynamically added to the vocabulary,
        by default True.
    throw : bool, optional
        If True, molecules containing tokens absent from the vocabulary are skipped, by default False.
    check_unique : bool, optional
        If True, duplicate sequences in `molecules` are skipped, by default True.
    """

    def __init__(
        self,
        molecules: Union[Iterable[str], MolSupplier],
        vocabulary: Optional[SequenceVocabulary] = None,
        update_voc: bool = True,
        throw: bool = False,
        check_unique: bool = True,
    ) -> None:
        """Initialize the SequenceCorpus.

        Parameters
        ----------
        molecules : Union[Iterable[str], MolSupplier]
            Input SMILES dataset or supplier.
        vocabulary : SequenceVocabulary | None, optional
            Target vocabulary, by default VocSmiles(False).
        update_voc : bool, optional
            Whether to update vocabulary with unseen tokens, by default True.
        throw : bool, optional
            Whether to discard molecules with unseen tokens, by default False.
        check_unique : bool, optional
            Whether to enforce molecule uniqueness, by default True.
        """
        super().__init__(molecules)
        self.vocabulary: SequenceVocabulary = vocabulary if vocabulary is not None else VocSmiles(False)
        self.updateVoc = update_voc
        self.throw = throw
        if self.updateVoc and self.throw:
            logger.warning("update_voc and throw cannot both be true at same time, defaulting to update_voc")
        self.checkUnique = check_unique
        self._unique: Set[str] = set()

    def saveVoc(self, path: str) -> None:
        """Save current vocabulary tokens to a text file.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        self.vocabulary.toFile(path)

    def getVoc(self) -> SequenceVocabulary:
        """Return the active SequenceVocabulary instance.

        Returns
        -------
        SequenceVocabulary
            Currently active sequence vocabulary.
        """
        return self.vocabulary

    def processMolecule(self, seq: str) -> Optional[List[int]]:
        """Tokenize and integer-encode a single SMILES sequence.

        Parameters
        ----------
        seq : str
            Input SMILES sequence string.

        Returns
        -------
        Optional[List[int]]
            List of encoded integer token indices, or None if molecule was rejected
            (duplicate or invalid length/tokens).
        """
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
