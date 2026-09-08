"""
Neural network layers, positional embeddings, and masking utilities for generators.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn


def pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Generate boolean padding mask for sequence batches.

    Parameters
    ----------
    seq : torch.Tensor
        Batch sequence tensor of token indices.
    pad_idx : int, optional
        Token index representing padding in vocabulary (default: 0).

    Returns
    -------
    mask : torch.Tensor
        Boolean mask tensor of same shape as `seq`, where True marks padding positions.
    """
    return seq == pad_idx


def tri_mask(seq: torch.Tensor, diag: int = 1) -> torch.Tensor:
    """
    Generate causal upper-triangular mask for autoregressive attention.

    Prevents attention heads from attending to subsequent future token positions.

    Parameters
    ----------
    seq : torch.Tensor
        Sequence tensor of shape `(batch_size, seq_len)`.
    diag : int, optional
        Diagonal offset for triangular mask (default: 1).

    Returns
    -------
    masks : torch.Tensor
        Boolean tensor of shape `(seq_len, seq_len)` on `seq.device`,
        where True represents masked (forbidden) future attention positions.
    """
    sz_b, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
    return masks.bool().to(seq.device)


def unique(arr: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Find unique rows in a 2D matrix and return their sorted indices.

    Parameters
    ----------
    arr : torch.Tensor or np.ndarray
        2D matrix containing feature rows.

    Returns
    -------
    idxs : torch.Tensor or np.ndarray
        1D array/tensor containing indices of first occurrences of unique rows.
    """
    is_tensor = isinstance(arr, torch.Tensor)
    device = arr.get_device() if is_tensor else None
    if is_tensor:
        arr_np = arr.cpu().numpy()
    else:
        arr_np = arr
    arr_c = np.ascontiguousarray(arr_np).view(np.dtype((np.void, arr_np.dtype.itemsize * arr_np.shape[1])))
    _, idxs = np.unique(arr_c, return_index=True)
    idxs = np.sort(idxs)
    if is_tensor:
        return torch.LongTensor(idxs).to(device)
    return idxs


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with ReLU activation.

    Applies `Linear(d_in -> d_hid) -> ReLU -> Linear(d_hid -> d_in)`.

    Parameters
    ----------
    d_in : int
        Input and output feature dimensionality.
    d_hid : int
        Inner hidden layer dimensionality.
    """

    def __init__(self, d_in: int, d_hid: int) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation across position-wise linear projections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., d_in)`.

        Returns
        -------
        output : torch.Tensor
            Projected tensor of shape `(..., d_in)`.
        """
        y = self.w_1(x).relu()
        y = self.w_2(y)
        return y


class SublayerConnection(nn.Module):
    """
    Residual connection wrapper with Dropout and Layer Normalization.

    Computes `LayerNorm(x + Dropout(sublayer(x)))`.

    Parameters
    ----------
    size : int
        Feature dimension passed to `nn.LayerNorm`.
    dropout : float, optional
        Dropout probability (default: 0.1).
    """

    def __init__(self, size: int, dropout: float = 0.1) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Apply residual skip connection, dropout, and layer normalization to `sublayer(x)`.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape `(..., size)`.
        sublayer : callable
            Transformation function taking `x` and returning a tensor of same shape.

        Returns
        -------
        output : torch.Tensor
            Normalized residual tensor of shape `(..., size)`.
        """
        y = sublayer(x)
        y = self.dropout(y)
        return self.norm(x + y)


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding module for sequence transformers.

    Computes deterministic sine/cosine encodings across token positions.

    Parameters
    ----------
    d_model : int
        Embedding vector dimensionality (must be an even integer).
    max_len : int, optional
        Maximum sequence length capacity (default: 100).
    batch_first : bool, optional
        If True, returns output with shape `(1, seq_len, d_model)`.
        If False, returns output with shape `(seq_len, 1, d_model)` (default: False).
    """

    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None:
        super(PositionalEmbedding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve positional encoding slice corresponding to input sequence length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. If `batch_first=True`, shape is `(batch_size, seq_len)`.
            Otherwise shape is `(seq_len, batch_size)`.

        Returns
        -------
        pe : torch.Tensor
            Positional encoding tensor detached from computation graph.
        """
        if self.batch_first:
            return self.pe[:x.size(1), :].unsqueeze(0).detach()
        else:
            return self.pe[:x.size(0), :].unsqueeze(1).detach()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding lookup table for molecular graph loci.

    Parameters
    ----------
    d_model : int
        Dimensionality of the graph node/site encoding.
    max_len : int, optional
        Maximum site index capacity (default: 100).
    batch_first : bool, optional
        Format flag (default: False).
    """

    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None:
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lookup positional encodings for flattened node/loci index tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of locus indices with shape `(batch_size, seq_len)`.

        Returns
        -------
        code : torch.Tensor
            Positional encoding tensor of shape `(batch_size, seq_len, d_model)`.
        """
        bsize, sqlen = x.size()
        y = x.reshape(bsize * sqlen)
        return self.pe[y, :].view(bsize, sqlen, -1)