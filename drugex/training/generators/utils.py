"""
Neural network layers, positional embeddings, and masking utilities for generators.
"""

from __future__ import annotations

import math
from typing import Callable, Union

# [PACKAGE: numpy] WHY: Contiguous byte-view sorting for finding unique matrix rows. WHAT: Numerical computing library.
import numpy as np
# [PACKAGE: torch] WHY: Tensor indexing, boolean masks, neural network layers (Linear, LayerNorm, Dropout, Module). WHAT: Deep learning framework.
import torch
import torch.nn as nn


def pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Generate boolean padding mask for sequence batches."""
    return seq == pad_idx


def tri_mask(seq: torch.Tensor, diag: int = 1) -> torch.Tensor:
    """Generate causal upper-triangular mask for autoregressive attention."""
    sz_b, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
    return masks.bool().to(seq.device)


def unique(arr: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Find unique rows in a 2D matrix and return their sorted indices."""
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
    """Two-layer position-wise feed-forward network with ReLU activation."""

    def __init__(self, d_in: int, d_hid: int) -> None:
        """Initialize position-wise feed-forward network with two linear transformations."""
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation across position-wise linear projections."""
        y = self.w_1(x).relu()
        y = self.w_2(y)
        return y


class SublayerConnection(nn.Module):
    """Residual connection wrapper with Dropout and Layer Normalization."""

    def __init__(self, size: int, dropout: float = 0.1) -> None:
        """Initialize residual connection wrapper with Dropout and Layer Normalization."""
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply residual skip connection, dropout, and layer normalization to sublayer(x)."""
        y = sublayer(x)
        y = self.dropout(y)
        return self.norm(x + y)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding module for sequence transformers."""

    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None:
        """Initialize sinusoidal positional embedding module for sequence transformers."""
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
        """Retrieve positional encoding slice corresponding to input sequence length."""
        if self.batch_first:
            return self.pe[:x.size(1), :].unsqueeze(0).detach()
        else:
            return self.pe[:x.size(0), :].unsqueeze(1).detach()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding lookup table for molecular graph loci."""

    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None:
        """Initialize sinusoidal positional encoding lookup table for molecular graph loci."""
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
        """Lookup positional encodings for flattened node/loci index tensor."""
        bsize, sqlen = x.size()
        y = x.reshape(bsize * sqlen)
        return self.pe[y, :].view(bsize, sqlen, -1)