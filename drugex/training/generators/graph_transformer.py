"""
Graph transformer generator for fragment-conditioned molecular graph generation.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.generators.interfaces import FragGenerator
from drugex.training.generators.utils import (
    PositionalEncoding,
    PositionwiseFeedForward,
    SublayerConnection,
    tri_mask,
)
from drugex.utils import ScheduledOptim

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import DataLoader
    from drugex.data.corpus.vocabulary import VocGraph
    from drugex.training.environment import Environment


class Block(nn.Module):
    """
    Standard Transformer self-attention block for molecular graph token processing.
    """

    def __init__(self, d_model: int, n_head: int, d_inner: int) -> None:
        """
        Initialize standard Transformer self-attention block for molecular graph token processing.

        Parameters
        ----------
        d_model : int
            Dimensionality of the graph representation vectors.
        n_head : int
            Number of parallel attention heads.
        d_inner : int
            Dimensionality of the inner hidden feed-forward layer.
        """
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model) for _ in range(2)])

    def forward(
        self,
        x: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute forward computation through attention and feed-forward sublayers.

        Parameters
        ----------
        x : torch.Tensor
            Graph node/edge token tensor of shape `(seq_len, batch_size, d_model)`.
        key_mask : torch.Tensor, optional
            Padding mask of shape `(batch_size, seq_len)`.
        attn_mask : torch.Tensor, optional
            Attention masking tensor of shape `(seq_len, seq_len)`.

        Returns
        -------
        output : torch.Tensor
            Updated graph feature tensor of shape `(seq_len, batch_size, d_model)`.
        """
        x = self.connector[0](x, lambda x: self.attn(x, x, x, key_padding_mask=key_mask, attn_mask=attn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class AtomLayer(nn.Module):
    """
    Multi-layer Transformer stack for processing sequential graph construction steps.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        d_inner: int = 1024,
        n_layer: int = 12
    ) -> None:
        """
        Initialize multi-layer Transformer stack for processing sequential graph construction steps.

        Parameters
        ----------
        d_model : int, optional
            Transformer feature space dimension (default: 512).
        n_head : int, optional
            Number of attention heads (default: 8).
        d_inner : int, optional
            Dimension of intermediate feed-forward layers (default: 1024).
        n_layer : int, optional
            Number of stacked transformer blocks (default: 12).
        """
        super(AtomLayer, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.blocks = nn.ModuleList([
            Block(self.d_model, self.n_head, d_inner=d_inner)
            for _ in range(self.n_layer)
        ])

    def forward(
        self,
        x: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate features sequentially across all transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape `(seq_len, batch_size, d_model)`.
        key_mask : torch.Tensor, optional
            Padding key mask.
        attn_mask : torch.Tensor, optional
            Attention causality mask.

        Returns
        -------
        output : torch.Tensor
            Encoded graph states of shape `(seq_len, batch_size, d_model)`.
        """
        for block in self.blocks:
            x = block(x, key_mask=key_mask, attn_mask=attn_mask)
        return x


class GraphTransformer(FragGenerator):
    """
    Autoregressive Graph Transformer for fragment-conditioned molecular graph generation.

    Generates molecular graphs by sequentially growing atoms, bonds, and ring-closure
    connections onto input fragment scaffolds. Uses an atom-level attention network
    coupled with recurrent transition cells to predict 4-tuple graph actions:
    `(atom_type, current_locus, previous_locus, bond_type)`.

    Attributes
    ----------
    mol_type : str
        Representation format indicator (`'graph'`).
    voc_trg : VocGraph
        Active graph vocabulary.
    n_grows : int
        Maximum number of atom addition steps.
    n_frags : int
        Maximum number of fragment linking steps.
    emb_word : nn.Embedding
        Embedding for combined atom/bond tokens.
    emb_atom : nn.Embedding
        Embedding for atom types.
    emb_loci : nn.Embedding
        Embedding for graph node positions / loci.
    emb_site : PositionalEncoding
        Sinusoidal positional encoding for connection sites.
    attn : AtomLayer
        12-layer multi-head attention stack.
    rnn : nn.GRUCell
        Recurrent transition cell for step-wise action decoding.
    """

    def __init__(
        self,
        voc_trg: VocGraph,
        d_emb: int = 512,
        d_model: int = 512,
        n_head: int = 8,
        d_inner: int = 1024,
        n_layer: int = 12,
        pad_idx: int = 0,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None:
        """
        Initialize autoregressive Graph Transformer for fragment-conditioned molecular graph generation.

        Parameters
        ----------
        voc_trg : VocGraph
            Target graph vocabulary defining atom types, valences, and graph dimensions.
        d_emb : int, optional
            Embedding vector dimensionality for atom words and loci (default: 512).
        d_model : int, optional
            Hidden feature dimension across attention and GRU layers (default: 512).
        n_head : int, optional
            Number of attention heads in AtomLayer (default: 8).
        d_inner : int, optional
            Feed-forward inner dimension (default: 1024).
        n_layer : int, optional
            Number of stacked transformer layers (default: 12).
        pad_idx : int, optional
            Padding token index (default: 0).
        device : torch.device or str, optional
            Target hardware execution device (default: `DEFAULT_DEVICE`).
        use_gpus : sequence of int, optional
            GPU indices available for training and sampling (default: `DEFAULT_GPUS`).
        """
        super(GraphTransformer, self).__init__(device=device, use_gpus=use_gpus)
        self.mol_type = 'graph'
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_grows = voc_trg.max_len - voc_trg.n_frags - 1
        self.n_frags = voc_trg.n_frags + 1
        self.d_emb = d_emb
        self.emb_word = nn.Embedding(voc_trg.size * 4, self.d_emb, padding_idx=pad_idx)
        self.emb_atom = nn.Embedding(voc_trg.size, self.d_emb, padding_idx=pad_idx)
        self.emb_loci = nn.Embedding(self.n_grows, self.d_emb)
        self.emb_site = PositionalEncoding(self.d_emb, max_len=self.n_grows * self.n_grows)
        self.attn = AtomLayer(d_model=d_model, n_head=n_head, d_inner=d_inner, n_layer=n_layer)

        self.rnn = nn.GRUCell(self.d_model, self.d_model)
        self.prj_atom = nn.Linear(d_emb, self.voc_trg.size)
        self.prj_bond = nn.Linear(d_model, 4)
        self.prj_loci = nn.Linear(d_model, self.n_grows)
        self.init_states()

        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9),
            0.1,
            d_model
        )
        self.model_name = 'GraphTransformer'

    def init_states(self) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)

    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """
        Attach model to GPUs and bind computation device.

        Parameters
        ----------
        gpus : sequence of int
            Sequence of GPU device IDs.
        """
        self.gpus = tuple(gpus)
        self.to(self.device)

    def forward(
        self,
        src: torch.Tensor,
        is_train: bool = False
    ) -> list[torch.Tensor] | torch.Tensor:
        """
        Execute forward computation for training loss or graph-growing autoregressive sampling.

        In training mode (`is_train=True`), computes step-wise cross-entropy losses for
        predicting the next atom, bond, current locus, and previous locus.
        In inference mode (`is_train=False`), performs autoregressive growing and connection
        of atoms to generate full molecular graphs.

        Parameters
        ----------
        src : torch.Tensor
            Input graph tensor of shape `(batch_size, seq_len, 5)`, where the 5 features
            represent `[atom_type, curr_locus, prev_locus, bond_type, grow_flag]`.
        is_train : bool, optional
            If True, calculates log-likelihood losses. If False, samples novel graphs (default: False).

        Returns
        -------
        output : list of torch.Tensor or torch.Tensor
            If `is_train=True`: list of 4 loss component tensors `[out_atom, out_curr, out_prev, out_bond]`.
            If `is_train=False`: completed molecular graph tensor of shape `(batch_size, max_len, 5)`.
        """
        if is_train:
            src_in, trg = src[:, :-1, :], src[:, 1:, :]
            batch, sqlen, _ = src_in.shape
            triu = tri_mask(src_in[:, :, 0])

            emb = self.emb_word(src_in[:, :, 3] + src_in[:, :, 0] * 4)
            emb += self.emb_site(src_in[:, :, 1] * self.n_grows + src_in[:, :, 2])
            dec = self.attn(emb.transpose(0, 1), attn_mask=triu)

            dec = dec.transpose(0, 1).reshape(batch * sqlen, -1)
            out_atom = self.prj_atom(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_atom = out_atom.gather(2, trg[:, :, 0].unsqueeze(2))

            atom = self.emb_atom(trg[:, :, 0]).reshape(batch * sqlen, -1)
            dec = self.rnn(atom, dec)
            out_bond = self.prj_bond(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_bond = out_bond.gather(2, trg[:, :, 3].unsqueeze(2))

            word = self.emb_word(trg[:, :, 3] + trg[:, :, 0] * 4)
            word = word.reshape(batch * sqlen, -1)
            dec = self.rnn(word, dec)
            out_prev = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_prev = out_prev.gather(2, trg[:, :, 2].unsqueeze(2))

            curr = self.emb_loci(trg[:, :, 2]).reshape(batch * sqlen, -1)
            dec = self.rnn(curr, dec)
            out_curr = self.prj_loci(dec).log_softmax(dim=-1).view(batch, sqlen, -1)
            out_curr = out_curr.gather(2, trg[:, :, 1].unsqueeze(2))

            return [out_atom, out_curr, out_prev, out_bond]
        else:
            is_end = torch.zeros(len(src)).bool().to(src.device)
            exists = torch.zeros(len(src), self.n_grows, self.n_grows).long().to(src.device)
            vals_max = torch.zeros(len(src), self.n_grows).long().to(src.device)
            frg_ids = torch.zeros(len(src), self.n_grows).long().to(src.device)
            order = torch.LongTensor(range(len(src))).to(src.device)
            curr = torch.zeros(len(src)).long().to(src.device) - 1
            blank = torch.LongTensor(len(src)).to(src.device).fill_(self.voc_trg.tk2ix['*'])
            single = torch.ones(len(src)).long().to(src.device)
            voc_mask = self.voc_trg.masks.to(src.device)

            for step in range(1, self.n_grows):
                if is_end.all():
                    src[:, step, :] = 0
                    continue
                data = src[:, :step, :]
                triu = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
                emb += self.emb_site(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.attn(emb.transpose(0, 1), attn_mask=triu)
                dec = dec[-1, :, :]

                grow = src[:, step, 4] == 0
                mask = voc_mask.repeat(len(src), 1) < 0
                if step <= 2:
                    mask[:, -1] = True
                else:
                    judge = (vals_rom == 0) | (exists[order, curr, :] != 0)
                    judge[order, curr] = True
                    judge = judge.all(dim=1) | (vals_rom[order, curr] == 0)
                    mask[judge, -1] = True
                mask[:, 1] = True
                mask[is_end, 1:] = True
                out_atom = self.prj_atom(dec).softmax(dim=-1)
                atom = out_atom.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 0] = atom[grow]
                atom = src[:, step, 0]
                is_end |= (atom == 0) & grow
                num = (vals_max > 0).sum(dim=1)
                vals_max[order, num] = voc_mask[atom]
                vals_rom = vals_max - exists.sum(dim=1)

                bud = atom != self.voc_trg.tk2ix['*']
                curr += bud
                curr[is_end] = 0
                src[:, step, 1] = curr
                exist = exists[order, curr, :] != 0

                mask = torch.zeros(len(src), 4).bool().to(src.device)
                for i in range(1, 4):
                    judge = (vals_rom < i) | exist
                    judge[order, curr] = True
                    mask[:, i] = judge.all(dim=1) | (vals_rom[order, curr] < i)
                mask[:, 0] = False if step == 1 else True
                mask[is_end, 0] = False
                mask[is_end, 1:] = True

                atom_emb = self.emb_atom(atom)
                dec = self.rnn(atom_emb, dec)
                out_bond = self.prj_bond(dec).softmax(dim=-1)
                bond = out_bond.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 3] = bond[grow]
                bond = src[:, step, 3]

                mask = (vals_max == 0) | exist | (vals_rom < bond.unsqueeze(-1))
                mask[order, curr] = True
                if step <= 2:
                    mask[:, 0] = False
                mask[is_end, 0] = False
                mask[is_end, 1:] = True
                word_emb = self.emb_word(atom * 4 + bond)
                dec = self.rnn(word_emb, dec)
                prev_out = self.prj_loci(dec).softmax(dim=-1)
                prev = prev_out.masked_fill(mask, 0).multinomial(1).view(-1)
                src[grow, step, 2] = prev[grow]
                prev = src[:, step, 2]

                for i in range(len(src)):
                    if not grow[i]:
                        frg_ids[i, curr[i]] = src[i, step, -1]
                    elif bud[i]:
                        frg_ids[i, curr[i]] = frg_ids[i, prev[i]]
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, curr, prev] = bond
                exists[order, prev, curr] = bond
                vals_rom = vals_max - exists.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)

            src[:, -self.n_frags, 1:] = 0
            src[:, -self.n_frags, 0] = self.voc_trg.tk2ix['GO']
            is_end = torch.zeros(len(src)).bool().to(src.device)
            for step in range(self.n_grows + 1, self.voc_trg.max_len):
                data = src[:, :step, :]
                triu = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
                emb += self.emb_site(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.attn(emb.transpose(0, 1), attn_mask=triu)

                vals_rom = vals_max - exists.sum(dim=1)
                frgs_rom = torch.zeros(len(src), 8).long().to(src.device)
                for i in range(1, 8):
                    ix = frg_ids != i
                    rom = vals_rom.clone()
                    rom[ix] = 0
                    frgs_rom[:, i] = rom.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)
                is_end |= (frgs_rom != 0).sum(dim=1) <= 1
                mask = (vals_rom < 1) | (vals_max == 0)
                mask[is_end, 0] = False
                atom_emb = self.emb_word(blank * 4 + single)
                dec = self.rnn(atom_emb, dec[-1, :, :])
                out_prev = self.prj_loci(dec).softmax(dim=-1)
                prev = out_prev.masked_fill(mask, 0).multinomial(1).view(-1)

                same = frg_ids == frg_ids[order, prev].view(-1, 1)
                exist = exists[order, prev] != 0
                mask = (vals_rom < 1) | exist | (vals_max == 0) | same
                mask[is_end, 0] = False
                prev_emb = self.emb_loci(prev)
                dec = self.rnn(prev_emb, dec)
                out_curr = self.prj_loci(dec).softmax(dim=-1)
                curr = out_curr.masked_fill(mask, 0).multinomial(1).view(-1)

                src[:, step, 3] = single
                src[:, step, 2] = prev
                src[:, step, 1] = curr
                src[:, step, 0] = blank
                src[is_end, step, :] = 0

                for i in range(len(src)):
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, src[:, step, 1], src[:, step, 2]] = src[:, step, 3]
                exists[order, src[:, step, 2], src[:, step, 1]] = src[:, step, 3]
            return src

    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Train the graph transformer network for one epoch across graph losses.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding graph batches of shape `(batch_size, seq_len, 5)`.
        epoch : int
            Current epoch number.
        epochs : int
            Total planned epochs.

        Returns
        -------
        loss : float
            Final batch combined graph training loss.
        """
        net = nn.DataParallel(self, device_ids=self.gpus)
        total_steps = len(loader)
        current_step = 0
        loss_val = 0.0
        for src in tqdm(loader, desc='Iterating over training batches', leave=False):
            src = src.to(self.device)
            self.optim.zero_grad()
            loss = net(src, is_train=True)
            total_loss = sum([-l.mean() for l in loss])
            total_loss.backward()
            self.optim.step()
            loss_val = total_loss.item()
            current_step += 1
            self.monitor.saveProgress(self, current_step, epoch, total_steps, epochs, loss_val)

        return loss_val

    def validateNet(
        self,
        loader: DataLoader,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        n_samples: Optional[int] = None
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Validate graph generation accuracy and evaluate hold-out validation loss.

        Parameters
        ----------
        loader : DataLoader
            DataLoader providing validation fragment graphs.
        evaluator : Environment, optional
            Scoring environment evaluating generated compounds.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES are marked invalid (default: True).
        n_samples : int, optional
            Unused by graph transformer (reserved for interface parity).

        Returns
        -------
        valid_metrics : dict of str to float
            Validation metrics containing `'valid_ratio'`, `'accurate_ratio'`, and `'loss_valid'`.
        scores : pd.DataFrame
            DataFrame of generated molecules with validity indicators and evaluator objective scores.
        """
        valid_metrics = {}
        net = nn.DataParallel(self, device_ids=self.gpus)
        pbar = tqdm(loader, desc='Iterating over validation batches', leave=False)
        smiles, frags = self.sample(pbar)
        scores = self.evaluate(smiles, frags, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
        scores['SMILES'] = smiles
        scores['Frags'] = frags
        valid_metrics['valid_ratio'] = float(scores.Valid.mean())
        valid_metrics['accurate_ratio'] = float(scores.Accurate.mean())

        with torch.no_grad():
            valid_metrics['loss_valid'] = sum(
                [sum([-l.float().mean().item() for l in net(src, is_train=True)]) for src in loader]
            )

        return valid_metrics, scores

    def sample(self, loader: DataLoader) -> tuple[list[str], list[str]]:
        """
        Sample molecular graphs for input fragments and decode to SMILES.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding source fragment graphs.

        Returns
        -------
        smiles : list of str
            Decoded generated molecule SMILES.
        frags : list of str
            Decoded input fragment SMILES.
        """
        net = nn.DataParallel(self, device_ids=self.gpus)
        frags, smiles = [], []
        with torch.no_grad():
            for src in loader:
                trg = net(src.to(self.device))
                f, s = self.voc_trg.decode(trg)
                frags += f
                smiles += s

        return smiles, frags

    def loaderFromFrags(
        self,
        frags: list[str],
        batch_size: int = 32,
        n_proc: int = 1
    ) -> DataLoader:
        """
        Encode fragment SMILES strings into a graph DataLoader.

        Parameters
        ----------
        frags : list of str
            List of chemical fragments in SMILES format.
        batch_size : int, optional
            Batch size for DataLoader iteration (default: 32).
        n_proc : int, optional
            Number of processes for parallel fragment graph encoding (default: 1).

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding encoded fragment graph matrices.
        """
        encoder = FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=GraphFragmentEncoder(self.voc_trg),
            n_proc=n_proc
        )
        out_data = GraphFragDataSet(tempfile.NamedTemporaryFile().name)
        encoder.apply(frags, encodingCollectors=[out_data])
        loader = out_data.asDataLoader(batch_size, n_samples=batch_size)
        return loader

    def decodeLoaders(
        self,
        src: torch.Tensor,
        trg: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        """
        Decode target generated graph tensors into fragment and molecule SMILES.

        Parameters
        ----------
        src : torch.Tensor
            Input source fragment tensor.
        trg : torch.Tensor
            Output generated graph tensor.

        Returns
        -------
        frags : list of str
            Decoded fragment SMILES strings.
        smiles : list of str
            Decoded completed molecule SMILES strings.
        """
        return self.voc_trg.decode(trg)

    def iterLoader(self, loader: DataLoader) -> DataLoader:
        """
        Return the loader directly for batch iteration.

        Parameters
        ----------
        loader : DataLoader
            Input graph DataLoader.

        Returns
        -------
        loader : DataLoader
            Iterable DataLoader.
        """
        return loader