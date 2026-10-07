"""
Sequence transformer generator for fragment-based SMILES molecule generation.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, Generator as PyGenerator, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import optim
from torch.nn.init import kaiming_normal_
from tqdm.auto import tqdm

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.datasets import SmilesFragDataSet
from drugex.data.fragments import FragmentCorpusEncoder, SequenceFragmentEncoder
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.generators.interfaces import FragGenerator
from drugex.training.generators.utils import (
    PositionalEmbedding,
    PositionwiseFeedForward,
    SublayerConnection,
    pad_mask,
    tri_mask,
)
from drugex.utils import ScheduledOptim

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import DataLoader
    from drugex.data.corpus.interfaces import SequenceVocabulary
    from drugex.training.environment import Environment


class Block(nn.Module):
    """
    Standard Transformer decoder block.

    Comprises multi-head self-attention, position-wise feed-forward networks,
    and residual sublayer connections with layer normalization.
    """

    def __init__(self, d_model: int, n_head: int, d_inner: int) -> None:
        """
        Initialize standard Transformer decoder block.

        Parameters
        ----------
        d_model : int
            Dimensionality of the input and output features.
        n_head : int
            Number of parallel attention heads.
        d_inner : int
            Dimensionality of the inner hidden layer in the position-wise feed-forward network.
        """
        super(Block, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model) for _ in range(2)])

    def forward(
        self,
        x: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        atn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through attention sublayer and feed-forward sublayer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape `(seq_len, batch_size, d_model)`.
        key_mask : torch.Tensor, optional
            Boolean padding mask tensor of shape `(batch_size, seq_len)`.
        atn_mask : torch.Tensor, optional
            Causal triangular mask tensor of shape `(seq_len, seq_len)`.

        Returns
        -------
        output : torch.Tensor
            Transformed features of shape `(seq_len, batch_size, d_model)`.
        """
        x = self.connector[0](x, lambda x: self.attn(x, x, x, key_padding_mask=key_mask, attn_mask=atn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class GPT2Layer(nn.Module):
    """
    Stacked GPT-2 style autoregressive Transformer decoder.

    Combines learned token embeddings, sinusoidal positional encodings, stacked
    transformer attention blocks, layer normalization, and a linear vocabulary projection.
    """

    def __init__(
        self,
        voc: SequenceVocabulary,
        d_emb: int = 512,
        d_model: int = 512,
        n_head: int = 12,
        d_inner: int = 1024,
        n_layer: int = 12,
        pad_idx: int = 0
    ) -> None:
        """
        Initialize the stacked GPT-2 style autoregressive Transformer decoder.

        Parameters
        ----------
        voc : SequenceVocabulary
            Vocabulary defining token set and size.
        d_emb : int, optional
            Embedding vector dimensionality (default: 512).
        d_model : int, optional
            Transformer feature space dimensionality (default: 512).
        n_head : int, optional
            Number of attention heads (default: 12).
        d_inner : int, optional
            Hidden layer dimensionality in feed-forward networks (default: 1024).
        n_layer : int, optional
            Number of stacked transformer blocks (default: 12).
        pad_idx : int, optional
            Padding token index in the vocabulary (default: 0).
        """
        super(GPT2Layer, self).__init__()
        self.n_layer = n_layer
        self.d_emb = d_emb
        self.d_model = d_model
        self.n_head = n_head
        self.voc = voc
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(voc.size, self.d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(self.d_emb, max_len=voc.max_len + voc.max_len)
        self.blocks = nn.ModuleList([Block(self.d_emb, self.n_head, d_inner=d_inner) for _ in range(self.n_layer)])
        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.word_prj = nn.Linear(self.d_emb, self.voc.size)
        kaiming_normal_(self.word_prj.weight, nonlinearity="linear")

    def forward(
        self,
        input: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        atn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass converting token indices to next-token logits.

        Parameters
        ----------
        input : torch.Tensor
            Sequence token indices of shape `(seq_len, batch_size)`.
        key_mask : torch.Tensor, optional
            Padding mask of shape `(batch_size, seq_len)`.
        atn_mask : torch.Tensor, optional
            Causal attention mask of shape `(seq_len, seq_len)`.

        Returns
        -------
        logits : torch.Tensor
            Vocabulary prediction logits of shape `(seq_len, batch_size, voc.size)`.
        """
        hidden_states = self.posit_emb(input) + self.token_emb(input)
        for block in self.blocks:
            hidden_states = block(hidden_states, key_mask=key_mask, atn_mask=atn_mask)
        hidden_states = self.word_prj(hidden_states)
        return hidden_states


class SequenceTransformer(FragGenerator):
    """
    Autoregressive sequence Transformer for fragment-conditioned SMILES generation.

    Employs a causal GPT-2 decoder architecture to generate full chemical structures
    by conditioning on input fragment scaffolds or building blocks.

    Attributes
    ----------
    mol_type : str
        Representation format indicator (`'smiles'`).
    gpt2 : GPT2Layer
        Underlying causal GPT-2 decoder network.
    optim : ScheduledOptim
        Learning rate scheduler and Adam optimizer wrapper.
    """

    def __init__(
        self,
        voc_trg: SequenceVocabulary,
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
        Initialize autoregressive sequence Transformer for fragment-conditioned SMILES generation.

        Parameters
        ----------
        voc_trg : SequenceVocabulary
            Target token vocabulary for decoding and encoding SMILES sequences.
        d_emb : int, optional
            Embedding vector dimensionality (default: 512).
        d_model : int, optional
            Model hidden dimension across transformer blocks (default: 512).
        n_head : int, optional
            Number of multi-head attention heads (default: 8).
        d_inner : int, optional
            Inner feed-forward layer dimension (default: 1024).
        n_layer : int, optional
            Number of stacked transformer layers (default: 12).
        pad_idx : int, optional
            Vocabulary index corresponding to padding (default: 0).
        device : torch.device or str, optional
            Hardware device for computation (default: `DEFAULT_DEVICE`).
        use_gpus : sequence of int, optional
            GPU indices available for training and inference (default: `DEFAULT_GPUS`).
        """
        super(SequenceTransformer, self).__init__(device=device, use_gpus=use_gpus)
        self.mol_type = 'smiles'
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.gpt2 = GPT2Layer(
            self.voc_trg,
            d_emb=d_emb,
            d_model=d_model,
            n_head=n_head,
            d_inner=d_inner,
            n_layer=n_layer,
            pad_idx=pad_idx
        )
        self.init_states()
        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9),
            0.5,
            d_model
        )
        self.model_name = 'SequenceTransformer'

    def forward(
        self,
        src: torch.Tensor,
        trg: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training loss computation or autoregressive generation.

        If `trg` is provided (training mode), calculates log-likelihood losses for
        target token prediction conditioned on source fragments. If `trg` is None
        (inference mode), autoregressively samples continuation tokens step-by-step.

        Parameters
        ----------
        src : torch.Tensor
            Source fragment token tensor of shape `(batch_size, src_len)`.
        trg : torch.Tensor, optional
            Target sequence token tensor of shape `(batch_size, trg_len)`.

        Returns
        -------
        output : torch.Tensor
            In training mode (`trg` given): gathered log-probabilities of shape `(batch_size, trg_len)`.
            In inference mode (`trg` None): sampled target token sequences of shape `(batch_size, voc_trg.max_len)`.
        """
        if trg is not None:
            input_seq = torch.cat([src, trg], dim=1)
            key_mask = pad_mask(input_seq, self.pad_idx)
            atn_mask = tri_mask(input_seq)
            start, end = src.size(1) - 1, -1
            input_t = input_seq.transpose(0, 1)
            dec = self.gpt2(input_t, key_mask=key_mask, atn_mask=atn_mask)[start:end, :, :]
            dec = dec.transpose(0, 1).log_softmax(dim=-1)
            out = dec.gather(2, trg.unsqueeze(2)).squeeze(2)
        else:
            seq_len = self.voc_trg.max_len + self.voc_trg.max_len
            out = torch.zeros(len(src), seq_len).long().to(src.device)
            out[:, :src.size(1)] = src
            is_end = torch.zeros(len(src)).bool().to(src.device)
            for step in range(self.voc_trg.max_len):
                input_seq = out[:, :src.size(1) + step]
                key_mask = pad_mask(input_seq, self.pad_idx)
                atn_mask = tri_mask(input_seq)
                dec = self.gpt2(input_seq.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
                x = dec.softmax(dim=-1)[-1, :, :].multinomial(1).view(-1)
                x[is_end] = self.voc_trg.tk2ix['_']
                is_end |= x == self.voc_trg.tk2ix['EOS']
                out[:, src.size(1) + step] = x
                if is_end.all():
                    break
            out = out[:, self.voc_trg.max_len:].detach()
        return out

    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Train the Transformer model for one epoch.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding `(src, trg)` batches of encoded fragments and targets.
        epoch : int
            Current epoch number.
        epochs : int
            Total planned epochs.

        Returns
        -------
        loss : float
            Final batch negative log-likelihood loss for the epoch.
        """
        net = nn.DataParallel(self, device_ids=self.gpus)
        total_steps = len(loader)
        current_step = 0
        loss_val = 0.0
        for src, trg in tqdm(loader, desc='Iterating over training batches', leave=False):
            src, trg = src.to(self.device), trg.to(self.device)
            self.optim.zero_grad()
            loss = net(src, trg)
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            loss_val = loss.item()
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
        Validate generation accuracy and calculate hold-out validation loss.

        Parameters
        ----------
        loader : DataLoader
            DataLoader providing validation batches.
        evaluator : Environment, optional
            Environment scoring objective metrics on sampled compounds.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES are penalized as invalid (default: True).
        n_samples : int, optional
            Unused by transformer (kept for Generator interface compatibility).

        Returns
        -------
        valid_metrics : dict of str to float
            Validation metrics containing `'valid_ratio'`, `'accurate_ratio'`, and `'loss_valid'`.
        scores : pd.DataFrame
            DataFrame of generated molecules, input fragments, validity flags, and property scores.
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
                [sum([-l.mean().item() for l in net(src, trg)]) for src, trg in loader]
            )

        return valid_metrics, scores

    def sample(self, loader: DataLoader) -> tuple[list[str], list[str]]:
        """
        Sample SMILES elaborations for each fragment in `loader`.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding source fragment batches `(src, _)`.

        Returns
        -------
        smiles : list of str
            Decoded generated molecule SMILES.
        frags : list of str
            Decoded corresponding input fragment SMILES.
        """
        net = nn.DataParallel(self, device_ids=self.gpus)
        frags, smiles = [], []
        with torch.no_grad():
            for src, _ in loader:
                trg = net(src.to(self.device))
                smiles += [self.voc_trg.decode(s, is_tk=False) for s in trg]
                frags += [self.voc_trg.decode(s, is_tk=False) for s in src]

        return smiles, frags

    def loaderFromFrags(
        self,
        frags: list[str],
        batch_size: int = 32,
        n_proc: int = 1
    ) -> DataLoader:
        """
        Encode a list of fragment SMILES into a PyTorch DataLoader.

        Parameters
        ----------
        frags : list of str
            List of chemical fragments in SMILES format.
        batch_size : int, optional
            Batch size for DataLoader iteration (default: 32).
        n_proc : int, optional
            Number of worker processes for fragment encoding (default: 1).

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding encoded fragment batches.
        """
        encoder = FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=SequenceFragmentEncoder(self.voc_trg),
            n_proc=n_proc
        )
        out_data = SmilesFragDataSet(tempfile.NamedTemporaryFile().name)
        encoder.apply(frags, encodingCollectors=[out_data])
        loader = out_data.asDataLoader(batch_size, n_samples=batch_size)
        return loader

    def decodeLoaders(
        self,
        src: torch.Tensor,
        trg: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        """
        Decode encoded source fragments and target sequence tensors into SMILES strings.

        Parameters
        ----------
        src : torch.Tensor
            Batch of input fragment token tensors.
        trg : torch.Tensor
            Batch of generated molecule token tensors.

        Returns
        -------
        frags : list of str
            Decoded input fragment SMILES.
        smiles : list of str
            Decoded generated molecule SMILES.
        """
        new_smiles = [self.voc_trg.decode(s, is_tk=False) for s in trg]
        new_frags = [self.voc_trg.decode(s, is_tk=False) for s in src]
        return new_frags, new_smiles

    def iterLoader(self, loader: DataLoader) -> PyGenerator[torch.Tensor, None, None]:
        """
        Yield source fragment tensors from DataLoader batches.

        Parameters
        ----------
        loader : DataLoader
            Input DataLoader yielding `(src, ...)` batches.

        Yields
        ------
        src : torch.Tensor
            Source fragment token tensor.
        """
        for _, src in loader:
            yield src