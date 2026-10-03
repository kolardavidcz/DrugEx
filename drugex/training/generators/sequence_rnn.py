"""
Recurrent neural network sequence generator for autoregressive SMILES generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

# [PACKAGE: pandas] WHY: Tabular organization of generated molecules and evaluation metrics. WHAT: Data analysis library.
import pandas as pd
# [PACKAGE: torch] WHY: Tensor operations, neural network modules (Embedding, LSTM, GRU, Linear), and Adam optimizer. WHAT: Deep learning framework.
import torch
from torch import nn, optim
# [PACKAGE: tqdm] WHY: Progress bar visualization during batch iterations and molecule sampling. WHAT: Progress bar utility.
from tqdm.auto import tqdm

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.training.generators.interfaces import Generator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from drugex.data.corpus.vocabulary import VocSmiles
    from drugex.training.environment import Environment


class SequenceRNN(Generator):
    """Recurrent Neural Network (RNN) sequence generator for autoregressive SMILES generation."""

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
        """Initialize the Recurrent Neural Network sequence generator."""
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
        """Attach model to specified GPU and set computation device."""
        if not gpus or gpus[0] == -1 or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            self.to(self.device)
            self.gpus = (-1,)
            return

        self.device = torch.device(f'cuda:{gpus[0]}')
        self.to(self.device)
        self.gpus = (gpus[0],) if len(gpus) > 0 else tuple()

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Perform a single step forward pass through embedding and recurrent layers."""
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Initialize randomized recurrent hidden states for a new batch."""
        h = torch.rand(3, batch_size, self.hidden_size).to(self.device)
        if labels is not None:
            h[0, batch_size, 0] = labels
        if self.is_lstm:
            c = torch.rand(3, batch_size, self.hidden_size).to(self.device)
            return (h, c)
        return h

    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        """Compute sequence log-likelihood across target tokens."""
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
        """Autoregressively sample complete SMILES strings from the model."""
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
        """Evolve sequences via genetic crossover and mutation operators during reinforcement learning."""
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
        """Train the recurrent network for one epoch using negative log-likelihood."""
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
        """Validate generation metrics and calculate optional validation loss."""
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
        """Generate molecules iteratively until target valid sample count is reached."""
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
