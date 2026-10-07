"""Concrete vocabulary implementations for SMILES and molecular graph representations."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from drugex.data.corpus.interfaces import SequenceVocabulary, Vocabulary
from drugex.logs import logger
from drugex.molecules.converters.standardizers import CleanSMILES


class VocSmiles(SequenceVocabulary):
    """Vocabulary for encoding and decoding SMILES strings for `SequenceRNN` and `SequenceTransformer` models.

    Splits SMILES sequences into chemical tokens (atoms, bonds, rings, brackets),
    maps tokens to discrete integer indices, and converts tensors back into canonical SMILES.
    Also handles replacement of multi-character halogens (`'Cl'` -> `'L'`, `'Br'` -> `'R'`)
    for uniform single-token processing.

    Parameters
    ----------
    encode_frags : bool, optional
        Whether to include fragment-specific tokens (`'_'`), by default False.
    words : Sequence[str], optional
        Initial word/token list, by default `defaultWords`.
    max_len : int, optional
        Maximum sequence token length, by default 100.
    min_len : int, optional
        Minimum sequence token length, by default 10.
    """

    defaultWords: Tuple[str, ...] = (
        "#", "%", "(", ")", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "=", "B", "C", "F", "I", "L", "N", "O", "P", "R", "S", "[Ag-3]", "[As+]",
        "[As]", "[B-]", "[BH-]", "[BH2-]", "[BH3-]", "[B]", "[C+]", "[C-]", "[CH-]",
        "[CH2]", "[CH]", "[I+]", "[IH2]", "[N+]", "[N-]", "[NH+]", "[NH-]", "[NH2+]",
        "[N]", "[O+]", "[O-]", "[OH+]", "[O]", "[P+]", "[PH]", "[S+]", "[S-]", "[SH+]",
        "[SH2]", "[SH]", "[Se+]", "[SeH]", "[Se]", "[SiH2]", "[SiH]", "[Si]", "[Te]",
        "[b-]", "[c+]", "[c-]", "[cH-]", "[n+]", "[n-]", "[nH+]", "[nH]", "[o+]",
        "[s+]", "[se+]", "[se]", "[te+]", "[te]", "b", "c", "n", "o", "p", "s",
    )

    def __init__(
        self,
        encode_frags: bool = False,
        words: Sequence[str] = defaultWords,
        max_len: int = 100,
        min_len: int = 10,
    ) -> None:
        """Initialize the SMILES vocabulary.

        Parameters
        ----------
        encode_frags : bool, optional
            Whether to support fragment delimiter tokens, by default False.
        words : Sequence[str], optional
            Tokens included in vocabulary, by default defaultWords.
        max_len : int, optional
            Max allowed sequence length, by default 100.
        min_len : int, optional
            Min allowed sequence length, by default 10.
        """
        super().__init__(encode_frags, words, min_len=min_len, max_len=max_len)

    def encode(self, tokens: Sequence[Sequence[str]], frags: Any = None) -> torch.LongTensor:
        """Encode a batch of tokenized sequences into a 2D integer tensor.

        Parameters
        ----------
        tokens : Sequence[Sequence[str]]
            List of token sequences (e.g. `[['C', 'C', '(=O)', 'O'], ...]`).
        frags : Any, optional
            Fragment constraints (unused).

        Returns
        -------
        torch.LongTensor
            Tensor of shape `(len(tokens), max_len)` with token indices padded with 0.
        """
        output = torch.zeros(len(tokens), self.max_len, dtype=torch.long)
        for i, seq in enumerate(tokens):
            for j, char in enumerate(seq):
                if j < self.max_len and char in self.tk2ix:
                    output[i, j] = self.tk2ix[char]
        return output  # type: ignore

    def decode(
        self,
        tensor: Union[torch.Tensor, Sequence[int]],
        is_tk: bool = True,
        is_smiles: bool = True,
    ) -> str:
        """Decode a sequence of integer token indices back into a SMILES string.

        Parameters
        ----------
        tensor : Union[torch.Tensor, Sequence[int]]
            1D array or tensor of token indices.
        is_tk : bool, optional
            Whether elements are already token strings (True) or integer indices (False),
            by default True.
        is_smiles : bool, optional
            Whether to reconstruct halogen symbols (`'L'` -> `'Cl'`, `'R'` -> `'Br'`),
            by default True.

        Returns
        -------
        str
            Reconstructed SMILES string truncated at `'EOS'`.
        """
        tokens: List[str] = []
        for token_item in tensor:
            if not is_tk:
                idx = int(token_item)  # type: ignore
                token = self.ix2tk.get(idx, "")
            else:
                token = str(token_item)
            if token == "EOS":
                break
            if token in self.control:
                continue
            tokens.append(token)
        seqs = "".join(tokens)
        if is_smiles:
            seqs = self.parseDecoded(seqs)
        else:
            seqs = seqs.replace("|", "")
        return seqs

    def parseDecoded(self, smiles: str) -> str:
        """Substitute single-letter halogen placeholders back into valid SMILES elements.

        Parameters
        ----------
        smiles : str
            Internal SMILES containing 'L' for Cl and 'R' for Br.

        Returns
        -------
        str
            Standard SMILES with 'Cl' and 'Br'.
        """
        return smiles.replace("L", "Cl").replace("R", "Br")

    def splitSequence(self, smile: str) -> List[str]:
        """Split a SMILES string into tokens, replacing halogens and parsing bracket atoms.

        Parameters
        ----------
        smile : str
            Input SMILES string.

        Returns
        -------
        List[str]
            List of discrete tokens terminated with `'EOS'`.
        """
        regex = r"(\[[^\[\]]{1,6}\])"
        normalized_smile = smile.replace("Cl", "L").replace("Br", "R")
        tokens: List[str] = []
        for word in re.split(regex, normalized_smile):
            if not word:
                continue
            if word.startswith("["):
                tokens.append(word)
            else:
                for char in word:
                    tokens.append(char)
        return tokens + ["EOS"]

    @staticmethod
    def fromFile(
        path: Union[str, os.PathLike[str], Path],
        encode_frags: bool = False,
        min_len: int = 10,
        max_len: int = 100,
    ) -> VocSmiles:
        """Load vocabulary words from a whitespace-separated file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to vocabulary file.
        encode_frags : bool, optional
            Whether to support fragment delimiter tokens, by default False.
        min_len : int, optional
            Min sequence length, by default 10.
        max_len : int, optional
            Max sequence length, by default 100.

        Returns
        -------
        VocSmiles
            Initialized SMILES vocabulary instance.
        """
        with open(path, "r") as f:
            words = f.read().split()
            return VocSmiles(encode_frags, words, max_len=max_len, min_len=min_len)

    def calc_voc_fp(self, smiles: Sequence[str], prefix: Optional[str] = None) -> np.ndarray:
        """Calculate vocabulary token fingerprint arrays for a batch of SMILES.

        Parameters
        ----------
        smiles : Sequence[str]
            List of SMILES strings.
        prefix : Optional[str], optional
            Optional prefix token to prepend to each sequence, by default None.

        Returns
        -------
        np.ndarray
            2D integer numpy array of shape `(len(smiles), max_len)`.
        """
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        cleaner = CleanSMILES()
        for i, smile in enumerate(smiles):
            try:
                cleaned_smile = cleaner(smile)
            except Exception:
                continue
            token = self.splitSequence(cleaned_smile)
            if prefix is not None:
                token = [prefix] + token
            if len(token) > self.max_len:
                continue
            if {"C", "c"}.isdisjoint(token):
                continue
            if not {"[Na]", "[Zn]"}.isdisjoint(token):
                continue
            fps[i, :] = self.encode([token[:-1]]).numpy()[0]
        return fps


class VocNonGPT(VocSmiles):
    """Vocabulary variant for legacy sequence models (`Seq2Seq` and `EncDec`).

    Supports asymmetric source and target sequence lengths (`src_len` vs `trg_len`)
    and vertical pipe prefixes for source tokens.

    Parameters
    ----------
    words : Sequence[str]
        Base token words.
    src_len : int, optional
        Source sequence maximum length, by default 1000.
    trg_len : int, optional
        Target sequence maximum length, by default 100.
    max_len : int, optional
        Maximum length fallback, by default 100.
    min_len : int, optional
        Minimum length, by default 10.
    """

    def __init__(
        self,
        words: Sequence[str],
        src_len: int = 1000,
        trg_len: int = 100,
        max_len: int = 100,
        min_len: int = 10,
    ) -> None:
        """Initialize the VocNonGPT vocabulary.

        Parameters
        ----------
        words : Sequence[str]
            Base tokens.
        src_len : int, optional
            Source sequence length limit, by default 1000.
        trg_len : int, optional
            Target sequence length limit, by default 100.
        max_len : int, optional
            Max length fallback, by default 100.
        min_len : int, optional
            Min length, by default 10.
        """
        super().__init__(False, words, max_len=max_len, min_len=min_len)
        self.src_len = src_len
        self.trg_len = trg_len

    def encode(self, input: Sequence[Sequence[str]], is_smiles: bool = True) -> torch.LongTensor:
        """Encode sequences according to source or target mode.

        Parameters
        ----------
        input : Sequence[Sequence[str]]
            Token sequences to encode.
        is_smiles : bool, optional
            If True, uses target length and standard tokens; if False, uses source length,
            by default True.

        Returns
        -------
        torch.LongTensor
            Encoded integer tensor.
        """
        seq_len = self.trg_len if is_smiles else self.src_len
        output = torch.zeros(len(input), seq_len, dtype=torch.long)
        for i, seq in enumerate(input):
            for j, char in enumerate(seq):
                if j < seq_len:
                    token_key = char if is_smiles else ("|" + char)
                    if token_key in self.tk2ix:
                        output[i, j] = self.tk2ix[token_key]
        return output  # type: ignore

    def decode(
        self,
        matrix: Union[torch.Tensor, Sequence[int]],
        is_smiles: bool = True,
        is_tk: bool = False,
    ) -> str:
        """Decode index matrix back into a SMILES string.

        Parameters
        ----------
        matrix : Union[torch.Tensor, Sequence[int]]
            Indices to decode.
        is_smiles : bool, optional
            Whether to parse halogen substitutions, by default True.
        is_tk : bool, optional
            Whether elements are already token strings, by default False.

        Returns
        -------
        str
            Decoded SMILES string.
        """
        chars = super().decode(matrix, is_tk=is_tk, is_smiles=is_smiles)
        seqs = "".join(chars)
        if is_smiles:
            seqs = self.parseDecoded(seqs)
        else:
            seqs = seqs.replace("|", "")
        return seqs

    @staticmethod
    def fromFile(
        path: Union[str, os.PathLike[str], Path],
        src_len: int = 1000,
        trg_len: int = 100,
        max_len: int = 100,
        min_len: int = 10,
    ) -> VocNonGPT:
        """Load legacy sequence vocabulary from a text file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to vocabulary file.
        src_len : int, optional
            Source sequence length, by default 1000.
        trg_len : int, optional
            Target sequence length, by default 100.
        max_len : int, optional
            Max length, by default 100.
        min_len : int, optional
            Min length, by default 10.

        Returns
        -------
        VocNonGPT
            Initialized vocabulary instance.
        """
        with open(path, "r") as f:
            words = f.read().split()
            return VocNonGPT(words, src_len=src_len, trg_len=trg_len, max_len=max_len, min_len=min_len)


class VocGraph(Vocabulary):
    """Vocabulary for graph-based molecule representations (`GraphTransformer`).

    Encodes molecules into atom valence state tokens (e.g. `'4C'`, `'3O+'`, `'1Cl'`),
    formal charges, bond connectivity indices, and fragment attachment flags.

    Parameters
    ----------
    words : Sequence[str], optional
        Initial list of atom-valence words, by default `defaultWords`.
    max_len : int, optional
        Maximum number of graph action steps, by default 80.
    n_frags : int, optional
        Maximum number of scaffold fragment components, by default 4.
    """

    defaultWords: Tuple[str, ...] = (
        "2O", "3O+", "1O-", "4C", "3C+", "3C-", "3N", "4N+", "2N-", "1Cl", "2S", "6S",
        "4S", "3S+", "5S+", "1S-", "1F", "1I", "5I", "2I+", "1Br", "5P", "3P", "4P+",
        "2Se", "6Se", "4Se", "3Se+", "4Si", "3B", "4B-", "5As", "3As", "4As+", "2Te",
        "4Te", "3Te+",
    )

    def __init__(
        self,
        words: Sequence[str] = defaultWords,
        max_len: int = 80,
        n_frags: int = 4,
    ) -> None:
        """Initialize graph vocabulary.

        Parameters
        ----------
        words : Sequence[str], optional
            Atom-valence words, by default defaultWords.
        max_len : int, optional
            Maximum action steps, by default 80.
        n_frags : int, optional
            Max scaffold fragments, by default 4.
        """
        super().__init__(words)
        self.control: Tuple[str, ...] = ("EOS", "GO")
        cleaned_words: List[str] = [x for x in words if x not in self.control]
        words_unique: List[str] = []
        for word in cleaned_words:
            if word not in words_unique:
                words_unique.append(word)

        self.n_frags = n_frags
        self.max_len = max_len
        self.tk2ix: Dict[str, int] = {"EOS": 0, "GO": 1}
        self.ix2nr: Dict[int, int] = {0: 0, 1: 0}
        self.ix2ch: Dict[int, int] = {0: 0, 1: 0}
        self.E: Dict[int, str] = {0: "", 1: "+", -1: "-"}

        self.wordsParsed: List[Tuple[str, int, int, int, str]] = [
            self.parseWord(word) for word in words_unique
        ]
        self.words = list(self.control) + words_unique
        if "*" not in words_unique:
            self.words.append("*")
            self.wordsParsed.append(("*", 0, 0, 0, "*"))

        self.size = len(self.words)
        self.masks = torch.zeros(len(self.wordsParsed) + len(self.control), dtype=torch.long)
        for i, item in enumerate(self.wordsParsed):
            self.masks[i + len(self.control)] = item[1]
            ix = i + len(self.control)
            self.tk2ix[item[4]] = ix
            self.ix2nr[ix] = item[3]
            self.ix2ch[ix] = item[2]
        assert len(set(self.words)) == len(self.words)

    @staticmethod
    def parseWord(word: str) -> Tuple[str, int, int, int, str]:
        """Parse an atom-valence word into elemental and charge metadata.

        Parameters
        ----------
        word : str
            Atom-valence string (e.g. `'4C'`, `'3O+'`, `'*'`).

        Returns
        -------
        Tuple[str, int, int, int, str]
            `(element_with_charge, valence, charge_number, atomic_number, original_word)`
        """
        if word == "*":
            return "*", 0, 0, 0, "*"
        match_val = re.search(r"[0-9]", word)
        valence = int(match_val.group(0)) if match_val else 0
        charge_match = re.search(r"[+-]", word)
        if charge_match:
            charge = charge_match.group(0)
            charge_num = 1 if charge == "+" else -1
        else:
            charge = ""
            charge_num = 0
        match_elem = re.search(r"[a-zA-Z]+", word)
        element = match_elem.group(0) if match_elem else "C"
        atomic_num = Chem.Atom(element).GetAtomicNum()
        return element + charge, valence, charge_num, atomic_num, word

    @staticmethod
    def fromFile(
        path: Union[str, os.PathLike[str], Path],
        word_col: str = "Word",
        max_len: int = 80,
        n_frags: int = 4,
    ) -> VocGraph:
        """Load graph vocabulary from a TSV file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to TSV file.
        word_col : str, optional
            Column name containing token words, by default 'Word'.
        max_len : int, optional
            Maximum action steps, by default 80.
        n_frags : int, optional
            Fragment capacity, by default 4.

        Returns
        -------
        VocGraph
            Loaded graph vocabulary instance.
        """
        df = pd.read_table(path)
        return VocGraph.fromDataFrame(df, word_col=word_col, max_len=max_len, n_frags=n_frags)

    @staticmethod
    def fromDataFrame(
        df: pd.DataFrame,
        word_col: str = "Word",
        max_len: int = 80,
        n_frags: int = 4,
    ) -> VocGraph:
        """Instantiate graph vocabulary from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing token column.
        word_col : str, optional
            Column name, by default 'Word'.
        max_len : int, optional
            Max steps, by default 80.
        n_frags : int, optional
            Max fragments, by default 4.

        Returns
        -------
        VocGraph
            Graph vocabulary instance.
        """
        return VocGraph(df[word_col].tolist(), max_len=max_len, n_frags=n_frags)

    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """Save vocabulary words and parsed atom properties to a TSV file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Output file path.
        """
        self.toDataFrame().to_csv(path, index=False, sep="\t")

    def toDataFrame(self) -> pd.DataFrame:
        """Export parsed vocabulary words into a structured pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns `['Ele', 'Val', 'Ch', 'Nr', 'Word']`.
        """
        return pd.DataFrame(self.wordsParsed, columns=["Ele", "Val", "Ch", "Nr", "Word"])

    def get_atom_tk(self, atom: Chem.Atom) -> int:
        """Compute the vocabulary index corresponding to an RDKit Atom's valence state.

        Parameters
        ----------
        atom : Chem.Atom
            RDKit atom instance.

        Returns
        -------
        int
            Vocabulary token index.
        """
        sb = atom.GetSymbol() + self.E[atom.GetFormalCharge()]
        val = atom.GetExplicitValence() + atom.GetImplicitValence()
        tk = str(val) + sb
        return self.tk2ix[tk]

    def encode(self, smiles: Sequence[str], subs: Optional[Sequence[str]] = None) -> np.ndarray:
        """Encode molecules and scaffold fragments into 5-tuple graph construction actions.

        Parameters
        ----------
        smiles : Sequence[str]
            Sequence of SMILES strings.
        subs : Optional[Sequence[str]], optional
            Corresponding scaffold fragment SMILES strings, by default None.

        Returns
        -------
        np.ndarray
            3D integer numpy array of graph action tuples: `(atom_tk, curr_atom, prev_atom, bond_type, frag_flag)`.
        """
        if not subs:
            raise RuntimeError(f"Fragments must be specified for VocGraph.encode, got {subs} instead")

        output = np.zeros([len(smiles), self.max_len - self.n_frags - 1, 5], dtype=np.compat.long)
        connect = np.zeros([len(smiles), self.n_frags + 1, 5], dtype=np.compat.long)
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            sub = Chem.MolFromSmiles(subs[i])
            if mol is None or sub is None:
                continue

            sub_idxs = mol.GetSubstructMatches(sub)
            split_bond: Set[int] = set()
            sub_bond: List[int] = []
            sub_idx: Tuple[int, ...] = ()

            for s_idx in sub_idxs:
                sub_idx = s_idx
                sub_bond = [
                    mol.GetBondBetweenAtoms(
                        sub_idx[b.GetBeginAtomIdx()], sub_idx[b.GetEndAtomIdx()]
                    ).GetIdx()
                    for b in sub.GetBonds()
                ]
                sub_atom = [mol.GetAtomWithIdx(ix) for ix in sub_idx]
                split_bond = {
                    b.GetIdx() for a in sub_atom for b in a.GetBonds() if b.GetIdx() not in sub_bond
                }
                single = sum([int(mol.GetBondWithIdx(b).GetBondType()) for b in split_bond])
                if single == len(split_bond):
                    break

            frags = Chem.FragmentOnBonds(mol, list(split_bond))
            Chem.MolToSmiles(frags)
            rank = eval(frags.GetProp("_smilesAtomOutputOrder"))
            mol_idx = list(sub_idx) + [idx for idx in rank if idx not in sub_idx and idx < mol.GetNumAtoms()]
            frg_idx = [i_frag + 1 for i_frag, f in enumerate(Chem.GetMolFrags(sub)) for _ in f]

            Chem.Kekulize(mol)
            m: List[Tuple[int, int, int, int, int]] = [(self.tk2ix["GO"], 0, 0, 0, 1)]
            n: List[Tuple[int, int, int, int, int]] = []
            c: List[Tuple[int, int, int, int, int]] = [(self.tk2ix["GO"], 0, 0, 0, 0)]
            mol2sub = {ix: i_m for i_m, ix in enumerate(mol_idx)}

            for j, idx in enumerate(mol_idx):
                atom = mol.GetAtomWithIdx(idx)
                bonds = sorted(atom.GetBonds(), key=lambda x: mol2sub[x.GetOtherAtomIdx(idx)])
                bonds = [b for b in bonds if j > mol2sub[b.GetOtherAtomIdx(idx)]]
                n_split = sum([1 if b.GetIdx() in split_bond else 0 for b in bonds])
                tk = self.get_atom_tk(atom)
                for k, bond in enumerate(bonds):
                    ix2 = mol2sub[bond.GetOtherAtomIdx(idx)]
                    is_split = bond.GetIdx() in split_bond
                    if idx in sub_idx:
                        is_connect = is_split
                    elif len(bonds) == 1:
                        is_connect = False
                    elif n_split == len(bonds):
                        is_connect = is_split and k != 0
                    else:
                        is_connect = False

                    if bond.GetIdx() in sub_bond:
                        bin_list, f = m, frg_idx[j]
                    elif is_connect:
                        bin_list, f = c, 0
                    else:
                        bin_list, f = n, 0

                    if bond.GetIdx() in sub_bond or not is_connect:
                        tk2 = tk
                        tk = self.tk2ix["*"]
                    else:
                        tk2 = self.tk2ix["*"]
                    bin_list.append((tk2, j, ix2, int(bond.GetBondType()), f))

                if tk != self.tk2ix["*"]:
                    bin_list, f = (m, frg_idx[j]) if idx in sub_idx else (n, f)
                    bin_list.append((tk, j, j, 0, f))

            output[i, : len(m + n), :] = m + n
            if len(c) > 0:
                connect[i, : len(c)] = c

        return np.concatenate([output, connect], axis=1)

    def decode(self, matrix: Union[np.ndarray, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """Decode graph construction action matrices into reconstructed fragment and molecule SMILES.

        Parameters
        ----------
        matrix : Union[np.ndarray, torch.Tensor]
            3D matrix of action tuples `(atom, curr_atom, prev_atom, bond, frag)`.

        Returns
        -------
        Tuple[List[str], List[str]]
            `(fragments_smiles_list, molecule_smiles_list)`
        """
        frags: List[str] = []
        smiles: List[str] = []
        for _, adj in enumerate(matrix):
            emol = Chem.RWMol()
            esub = Chem.RWMol()
            try:
                for atom, curr, prev, bond, frag in adj:
                    atom, curr, prev, bond, frag = (
                        int(atom),
                        int(curr),
                        int(prev),
                        int(bond),
                        int(frag),
                    )
                    if atom == self.tk2ix["EOS"] or atom == self.tk2ix["GO"]:
                        continue
                    if atom != self.tk2ix["*"]:
                        a = Chem.Atom(self.ix2nr[atom])
                        a.SetFormalCharge(self.ix2ch[atom])
                        emol.AddAtom(a)
                        if frag != 0:
                            esub.AddAtom(a)
                    if bond != 0:
                        b = Chem.BondType(bond)
                        emol.AddBond(curr, prev, b)
                        if frag != 0:
                            esub.AddBond(curr, prev, b)
                Chem.SanitizeMol(emol)
                Chem.SanitizeMol(esub)
            except Exception as e:
                logger.error(f"Error while decoding graph matrix: {e}")
            frags.append(Chem.MolToSmiles(esub))
            smiles.append(Chem.MolToSmiles(emol))
        return frags, smiles