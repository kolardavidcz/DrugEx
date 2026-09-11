"""
Recurrent neural network sequence generator for autoregressive SMILES generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import pandas as pd
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.training.generators.interfaces import Generator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from drugex.data.corpus.vocabulary import VocSmiles
    from drugex.training.environment import Environment


class SequenceRNN(Generator):
    """
    Recurrent Neural Network (RNN) sequence generator for autoregressive SMILES generation.

    Employs a multi-layer LSTM or GRU architecture to generate chemical structures
    represented as SMILES strings token-by-token. Tokens are embedded into a continuous dense
    space, propagated through recurrent layers, and mapped to vocabulary logits to predict
    next-token probability distributions.

    Parameters
    ----------
    voc : VocSmiles
        Vocabulary mapping SMILES tokens to numerical indices and defining special tokens
        (`GO`, `EOS`, padding).
    embed_size : int, optional
        Dimensionality of the dense token embedding space (default: 128).
    hidden_size : int, optional
        Dimensionality of the recurrent hidden states across layers (default: 512).
    is_lstm : bool, optional
        If True, uses 3-layer LSTM cells (`nn.LSTM`) with separate hidden and cell states.
        If False, uses 3-layer GRU cells (`nn.GRU`) with fewer parameters (default: True).
    lr : float, optional
        Initial learning rate for the Adam optimizer (default: 1e-3).
    device : torch.device or str, optional
        Hardware execution device (default: `DEFAULT_DEVICE`).
    use_gpus : sequence of int, optional
        GPU hardware indices. SequenceRNN binds to the primary GPU in the sequence (default: `DEFAULT_GPUS`).

    Attributes
    ----------
    voc : VocSmiles
        Active token vocabulary.
    embed : nn.Embedding
        Embedding layer mapping vocabulary tokens to `embed_size` vectors.
    rnn : nn.LSTM or nn.GRU
        Three-layer recurrent network.
    linear : nn.Linear
        Output projection mapping `hidden_size` states to `voc.size` logits.
    optim : torch.optim.Adam
        Adam optimizer initialized on model parameters.
    """

    def __init__(
        self,
        voc: VocSmiles,
        embed_size: int = 128,
        hidden_size: int = 512,
        is_lstm: bool = True,
        lr: float = 1e-3,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None:
        """
        Initialize the Recurrent Neural Network sequence generator.

        Parameters
        ----------
        voc : VocSmiles
            Vocabulary mapping SMILES tokens to numerical indices and defining special tokens.
        embed_size : int, optional
            Dimensionality of the dense token embedding space (default: 128).
        hidden_size : int, optional
            Dimensionality of the recurrent hidden states across layers (default: 512).
        is_lstm : bool, optional
            If True, uses 3-layer LSTM cells (`nn.LSTM`). If False, uses 3-layer GRU cells (`nn.GRU`) (default: True).
        lr : float, optional
            Initial learning rate for the Adam optimizer (default: 1e-3).
        device : torch.device or str, optional
            Hardware execution device (default: `DEFAULT_DEVICE`).
        use_gpus : sequence of int, optional
            GPU hardware indices (default: `DEFAULT_GPUS`).
        """
        super(SequenceRNN, self).__init__(device=device, use_gpus=use_gpus)
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size

        self.embed = nn.Embedding(voc.size, embed_size)
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.rnn = rnn_layer(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.attachToGPUs(self.gpus)

        self.model_name = 'SequenceRNN'

    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """
        Attach model to specified GPU and set computation device.

        SequenceRNN utilizes single-GPU execution; only the first device ID in `gpus` is bound.

        Parameters
        ----------
        gpus : sequence of int
            Sequence of GPU indices.
        """
        self.device = torch.device(f'cuda:{gpus[0]}') if torch.cuda.is_available() and len(gpus) > 0 else torch.device('cpu')
        self.to(self.device)
        self.gpus = (gpus[0],) if len(gpus) > 0 else tuple()

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform a single step forward pass through the embedding and recurrent layers.

        Parameters
        ----------
        input : torch.Tensor
            Input token indices of shape `(batch_size,)` or `(batch_size, 1)`.
        h : torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            Recurrent state. For GRU, a tensor of shape `(3, batch_size, hidden_size)`.
            For LSTM, a tuple `(h, c)` of hidden state and cell memory tensors.

        Returns
        -------
        output : torch.Tensor
            Next-token unscaled logits of shape `(batch_size, voc.size)`.
        h_out : torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            Updated recurrent state tensor(s) for the next generation step.
        """
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize randomized recurrent hidden states for a new batch.

        Parameters
        ----------
        batch_size : int
            Number of sequences in the batch.
        labels : torch.Tensor, optional
            Optional condition vector injected into the first layer's hidden state.

        Returns
        -------
        states : torch.Tensor or tuple of (torch.Tensor, torch.Tensor)
            Initial `(h, c)` tuple for LSTM or single tensor `h` for GRU, of shape
            `(3, batch_size, hidden_size)`.
        """
        h = torch.rand(3, batch_size, self.hidden_size).to(self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(self.device)
            return (h, c)
        return h

    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log-likelihood across target tokens.

        Computes the sum of log-probabilities assigned by the model to each token in the
        target sequence batch.

        Parameters
        ----------
        target : torch.Tensor
            Long tensor of token indices with shape `(batch_size, seq_len)`.

        Returns
        -------
        scores : torch.Tensor
            Summed log-likelihood per sequence with shape `(batch_size,)`.
        """
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size).to(self.device)
        for step in range(seq_len):
            logits, h = self(x, h)
            log_probs = logits.log_softmax(dim=-1)
            scores += log_probs.gather(1, target[:, step:step+1]).squeeze()
            x = target[:, step]
        return scores

    def sample(self, batch_size: int) -> list[str]:
        """
        Autoregressively sample complete SMILES strings from the model.

        Begins from the `GO` start token, iteratively evaluates next-token probability
        distributions via multinomial sampling, masks terminated sequences upon encountering `EOS`,
        and decodes final index matrices into SMILES representations.

        Parameters
        ----------
        batch_size : int
            Number of molecules to sample in parallel.

        Returns
        -------
        smiles : list of str
            Decoded SMILES strings.
        """
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        h = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(self.device)
        isEnd = torch.zeros(batch_size).bool().to(self.device)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            proba = logit.softmax(dim=-1)
            x = torch.multinomial(proba, 1).view(-1)
            x[isEnd] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            end_token = (x == self.voc.tk2ix['EOS'])
            isEnd = torch.ge(isEnd + end_token, 1)
            if (isEnd == 1).all():
                break

        return [self.voc.decode(s, is_tk=False) for s in sequences]

    def evolve(
        self,
        batch_size: int,
        epsilon: float = 0.01,
        crover: Optional[Any] = None,
        mutate: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Evolve sequences via genetic crossover and mutation operators during reinforcement learning.

        Combines token probabilities from the base agent network, a crossover exploration network,
        and an optional mutation network governed by an epsilon-greedy mutation rate.

        Parameters
        ----------
        batch_size : int
            Number of candidate sequences to evolve.
        epsilon : float, optional
            Probability of substituting token distribution with mutation network logits (default: 0.01).
        crover : Generator, optional
            Crossover network providing alternative token distributions.
        mutate : Generator, optional
            Mutation exploration network providing exploratory token distributions.

        Returns
        -------
        sequences : torch.Tensor
            Long tensor of encoded token indices with shape `(batch_size, voc.max_len)`.
        """
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(self.device)
        hA = self.init_h(batch_size)
        hM = self.init_h(batch_size)
        hC = self.init_h(batch_size)
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(self.device)
        is_end = torch.zeros(batch_size).bool().to(self.device)

        for step in range(self.voc.max_len):
            logitA, hA = self(x, hA)
            proba = logitA.softmax(dim=-1)

            if crover is not None:
                ratio = torch.rand(batch_size, 1).to(self.device)
                logitC, hC = crover(x, hC)
                proba = proba * ratio + logitC.softmax(dim=-1) * (1 - ratio)

            if mutate is not None:
                logitM, hM = mutate(x, hM)
                is_mutate = (torch.rand(batch_size) < epsilon).to(self.device)
                proba[is_mutate, :] = logitM.softmax(dim=-1)[is_mutate, :]

            x = torch.multinomial(proba, 1).view(-1)
            is_end |= x == self.voc.tk2ix['EOS']
            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x
            if is_end.all():
                break
        return sequences

    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Train the recurrent network for one epoch using negative log-likelihood.

        Parameters
        ----------
        loader : DataLoader
            DataLoader yielding sequence training batches.
        epoch : int
            Current epoch index.
        epochs : int
            Total training epochs.

        Returns
        -------
        loss : float
            Final batch training loss for the epoch.
        """
        total_steps = len(loader)
        loss_val = 0.0
        for i, batch in enumerate(loader):
            self.optim.zero_grad()
            loss = self.likelihood(batch.to(self.device))
            loss = -loss.mean()
            loss.backward()
            self.optim.step()
            loss_val = loss.item()
            self.monitor.saveProgress(self, i, epoch, total_steps, epochs, loss_val)

        return loss_val

    def validateNet(
        self,
        loader: Optional[DataLoader] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        n_samples: int = 128
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Validate generation metrics and calculate optional validation loss.

        Samples `n_samples` SMILES strings to evaluate valid chemical structure ratio
        and property objective distributions, and computes average validation cross-entropy.

        Parameters
        ----------
        loader : DataLoader, optional
            Validation DataLoader to evaluate hold-out log-likelihood.
        evaluator : Environment, optional
            Scoring environment providing bioactivity and property scores.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / salt SMILES are marked invalid (default: True).
        n_samples : int, optional
            Number of molecules sampled for statistical validity assessment (default: 128).

        Returns
        -------
        valid_metrics : dict of str to float
            Validation metrics containing `'valid_ratio'` and optionally `'loss_valid'`.
        scores : pd.DataFrame
            DataFrame of generated SMILES with validity indicators and evaluator objective scores.
        """
        valid_metrics = {}
        smiles = self.sample(n_samples)
        scores = self.evaluate(smiles, evaluator=evaluator, no_multifrag_smiles=no_multifrag_smiles)
        scores['SMILES'] = smiles
        valid_metrics['valid_ratio'] = float(scores.Valid.mean())

        if loader is not None:
            loss_valid, size = 0.0, 0
            for j, batch in enumerate(loader):
                size += batch.size(0)
                loss_valid += -self.likelihood(batch.to(self.device)).sum().item()
            valid_metrics['loss_valid'] = loss_valid / size / self.voc.max_len

        return valid_metrics, scores

    def generate(
        self,
        num_samples: int = 100,
        batch_size: int = 32,
        n_proc: int = 1,
        drop_duplicates: bool = True,
        drop_invalid: bool = True,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        drop_undesired: bool = False,
        raw_scores: bool = True,
        progress: bool = True,
        tqdm_kwargs: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate molecules until `num_samples` valid compounds are collected.

        Samples batches iteratively, filtering for validity and duplicates,
        and optionally evaluates them against the provided objective environment.

        Parameters
        ----------
        num_samples : int, optional
            Target number of valid molecules to collect (default: 100).
        batch_size : int, optional
            Number of molecules sampled per batch iteration (default: 32).
        n_proc : int, optional
            Parallel workers placeholder for API consistency (default: 1).
        drop_duplicates : bool, optional
            If True, duplicate SMILES are filtered out (default: True).
        drop_invalid : bool, optional
            If True, chemically invalid SMILES are excluded (default: True).
        evaluator : Environment, optional
            Scoring environment evaluating generated compounds.
        no_multifrag_smiles : bool, optional
            If True, disconnected multi-fragment SMILES are penalized as invalid (default: True).
        drop_undesired : bool, optional
            If True, molecules failing minimum desirability criteria are dropped (default: False).
        raw_scores : bool, optional
            If True, returns raw unpenalized scores from scorers (default: True).
        progress : bool, optional
            If True, displays a tqdm progress bar tracking valid molecule collection (default: True).
        tqdm_kwargs : dict, optional
            Keyword arguments passed to `tqdm` progress bar (default: None).

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing generated compounds in `'SMILES'` column,
            along with objective scores if evaluator was provided.
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        if progress:
            tqdm_kwargs.update({'total': num_samples, 'desc': 'Generating molecules'})
            pbar = tqdm(**tqdm_kwargs)

        df_all = pd.DataFrame(columns=['SMILES', 'Frags'])
        while not len(df_all) >= num_samples:
            with torch.no_grad():
                new_smiles = self.sample(batch_size)
                df_new = pd.DataFrame({'SMILES': new_smiles, 'Frags': None})

                if drop_invalid:
                    df_new = self.filterNewMolecules(
                        df_all,
                        df_new,
                        with_frags=False,
                        drop_duplicates=drop_duplicates,
                        drop_undesired=drop_undesired,
                        evaluator=evaluator,
                        no_multifrag_smiles=no_multifrag_smiles
                    )

                df_all = pd.concat([df_all, df_new], axis=0, ignore_index=True)

                if progress:
                    pbar.update(len(df_new) if pbar.n + len(df_new) <= num_samples else num_samples - pbar.n)

        if progress:
            pbar.close()

        df = df_all.head(num_samples)
        if evaluator:
            df = pd.concat([
                df,
                self.evaluate(
                    df.SMILES.tolist(),
                    evaluator=evaluator,
                    no_multifrag_smiles=no_multifrag_smiles,
                    unmodified_scores=raw_scores
                )[evaluator.getScorerKeys()]
            ], axis=1)
        return df.drop('Frags', axis=1).round(decimals=3)