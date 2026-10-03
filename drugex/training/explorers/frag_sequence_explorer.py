#!/usr/bin/env python

import random
# [PACKAGE: torch] WHY: Tensor operations, Adam optimizer, TensorDataset, and DataLoader. WHAT: PyTorch deep learning framework.
import torch
from torch.optim import Adam
from drugex import utils, DEFAULT_DEVICE, DEFAULT_GPUS
from torch.utils.data import DataLoader, TensorDataset
# [PACKAGE: tqdm] WHY: Batch iteration progress meters. WHAT: Progress bar utility.
from tqdm.auto import tqdm

from drugex.logs import logger
from drugex.training.explorers.interfaces import FragExplorer
from drugex.training.generators.utils import pad_mask, tri_mask


class FragSequenceExplorer(FragExplorer):
    """Reinforcement learning explorer optimizing sequence-based fragment decorator models."""

    def __init__(self, agent, env=None, crover=None, mutate=None, batch_size=128, epsilon=0.1, beta=0.0, n_samples=-1, optim=None, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS, no_multifrag_smiles=True):
        """Initialize sequence fragment explorer with agent, environment, mutation/crossover networks, and optimizer."""
        super(FragSequenceExplorer, self).__init__(agent, env, mutate, crover, no_multifrag_smiles, batch_size, epsilon, beta, n_samples, device=device, use_gpus=use_gpus)
        self.optim = utils.ScheduledOptim(
            Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 1.0, 512) if not optim else optim

    def forward(self, src):
        """Autoregressively sample decorated SMILES tokens conditioned on input fragment prompts."""
        seq_len = self.agent.voc_trg.max_len + self.agent.voc_trg.max_len
        out = torch.zeros(len(src), seq_len).long().to(self.device)
        out[:, :src.size(1)] = src
        is_end = torch.zeros(len(src)).bool().to(self.device)

        for step in range(self.agent.voc_trg.max_len):  # decode up to max length
            sub = out[:, :src.size(1) + step]
            key_mask = pad_mask(sub, self.agent.pad_idx)
            atn_mask = tri_mask(sub)
            rand = torch.rand(1)
            if self.epsilon < rand <= 0.5 and self.crover is not None:
                dec = self.crover.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            elif rand < self.epsilon and self.mutate is not None:
                dec = self.mutate.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            else:
                dec = self.agent.gpt2(sub.transpose(0, 1), key_mask=key_mask, atn_mask=atn_mask)
            proba = dec[-1,:, :].softmax(dim=-1)

            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.agent.voc_trg.tk2ix['_']
            is_end |= x == self.agent.voc_trg.tk2ix['EOS']
            out[:, src.size(1) + step] = x
            if is_end.all(): break
        return out[:, self.agent.voc_trg.max_len:].detach()
            
    def sample_input(self, loader, is_test=False):
        """Subsample input fragment-molecule pairs into a DataLoader for training or validation."""
        encoded_in = torch.cat([batch[0] for batch in loader], 0)
        encoded_out = torch.cat([batch[1] for batch in loader], 0)
        
        n_pairs = encoded_in.shape[0]                
        n_samples = int(self.nSamples * 0.2) if is_test else self.nSamples     
        batch_size = self.batchSize * 10 if is_test else self.batchSize * 4
                
        if n_pairs > n_samples:
        
            logger.info(f"{n_samples} fragments-molecule pairs were sampled at random from original {n_pairs} pairs for {'validation' if is_test else 'training'}")
            sample_idx = torch.tensor(random.sample([ i for i in range(n_pairs)], n_samples))
            samples = [ torch.index_select(encoded_in, 0, sample_idx), torch.index_select(encoded_out, 0, sample_idx) ]
            loader = DataLoader(samples, batch_size=batch_size, drop_last=False, shuffle=True)
            
        return loader

    def getBatchOutputs(self, net, src):
        """Decode fragment and target token sequences to text and compute agent sequence cross-entropy loss."""
        src, trg = src[0].to(self.device), src[1].to(self.device)
        frags = [self.agent.voc_trg.decode(s, is_tk=False) for s in src]
        smiles = [self.agent.voc_trg.decode(s, is_tk=False) for s in trg]
        loss = net(src, trg)

        return frags, smiles, loss

    def sampleEncodedPairsToLoader(self, net, loader):
        """Sample decorated sequence completions from input fragments and wrap in a TensorDataset DataLoader."""
        srcs, trgs = [], []
        for _, src in tqdm(loader, desc='Iterating over training batches', leave=False):
            with torch.no_grad():
                trg = net(src.to(self.device))
                trgs.append(trg.detach().cpu())
                srcs.append(src.detach().cpu())
        trgs = torch.cat(trgs, dim=0)
        srcs = torch.cat(srcs, dim=0)      

        return DataLoader(TensorDataset(srcs, trgs), batch_size=self.batchSize, shuffle=True, drop_last=False)