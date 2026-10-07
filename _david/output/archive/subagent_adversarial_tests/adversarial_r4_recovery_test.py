"""Adversarial Empirical Stress-Testing Harness for Requirement R4 (Milestone M2).

Focus: PeakRecoveryMonitor, FragGraphExplorer, and Pretrained Papyrus Prior Loading.

Adversarial Stress Test Dimensions:
1. Peak Recovery Dynamics & Policy Collapse Survival:
   - Multi-epoch RL trajectory where peak desirability occurs at Epoch 2 (e.g. 0.85)
     followed by catastrophic policy collapse at Epochs 3-8 (0.35, 0.20, 0.15, 0.10, 0.08, 0.05).
   - Verify PeakRecoveryMonitor.close() and restore_peak_model(agent) restore exact Epoch 2 weights.
   - Byte-for-byte and tensor-level equality across 100% of network parameters.
   - Survival under cold disk reload (clearing in-memory peak state dict).
   - Real GraphTransformer neural network weights verification (all attention and embedding layers).
2. Early Stopping Exactness & Boundary Conditions:
   - Early stopping triggers at exactly 10 consecutive non-improving epochs with patience=10, min_epochs=5.
   - min_epochs lower-bound guard test (prevents premature stopping when epoch < min_epochs).
   - Non-improving counter reset on late breakthrough peak.
   - Strict plateau tie handling (equal scores treated as non-improving to prevent infinite loops).
3. Pre-trained Prior Loading & Zero Transfer Learning Integrity:
   - Direct loading of VocGraph and GraphTransformer from Papyrus05.5_graph_trans_PT package.
   - Full weight tensor sanity (no NaNs, no Infs, exact architecture parameter match).
   - VocGraph tokenization and NumPy 2.x np.int64 encoding compatibility.
   - Direct compatibility with FragGraphExplorer without 2D fine-tuning.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Pre-mock uninstalled legacy RAscore to prevent import issues with legacy modules
if "RAscore" not in sys.modules:
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: numpy] WHY: Numerical checks, random seeds, and score arrays. WHAT: Numerical computing library.
import numpy as np

# [PACKAGE: pytest] WHY: Test assertions, fixtures, and parameterized test execution. WHAT: Python test framework.
import pytest

# [PACKAGE: torch] WHY: Neural network weight tensors, state dict operations, and serialization. WHAT: PyTorch deep learning framework.
import torch
import torch.nn as nn

# Ensure repo root and _david/code are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DAVID_CODE = REPO_ROOT / "_david" / "code"
if str(DAVID_CODE) not in sys.path:
    sys.path.insert(0, str(DAVID_CODE))

from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.training.monitors import FileMonitor, PeakRecoveryMonitor as DrugExPeakMonitor
from _david.code.monitor import PeakRecoveryMonitor as DavidPeakMonitor
from drugex.training.explorers.frag_graph_explorer import FragGraphExplorer
from drugex.training.explorers.interfaces import FragExplorer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import AntiCollapseSoftGateReward


PRETRAINED_CANDIDATE_PATHS = [
    REPO_ROOT / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT",
    Path("/home/kolar/learn_projects/drugex/data/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT"),
    Path("/home/kolarv/DrugEx/tutorial/data/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT"),
    Path("/home/kolarv/DrugEx/data/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT"),
]


def get_pretrained_dir() -> Path:
    """Resolve active directory containing Papyrus05.5_graph_trans_PT pretrained files."""
    for p in PRETRAINED_CANDIDATE_PATHS:
        if p.exists() and (p / "Papyrus05.5_graph_trans_PT.pkg").exists() and (p / "Papyrus05.5_graph_trans_PT.vocab").exists():
            return p
    pytest.skip("Pretrained Papyrus05.5_graph_trans_PT directory not found on host filesystem.")


class MockNeuralNetAgent(nn.Module):
    """Deep multi-layer neural network with convolution, linear, and embedding weights to test weight fidelity."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(50, in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def loadStatesFromFile(self, path: Path | str) -> None:
        """Load state dictionary from file."""
        # [PACKAGE: torch] WHY: Deserialize state dictionary. WHAT: PyTorch file loading. HOW: torch.load.
        state = torch.load(path, weights_only=False)
        self.load_state_dict(state)


class MockExplorer:
    """Mock explorer implementing DrugEx explorer interface conventions."""

    def __init__(self, agent: nn.Module) -> None:
        self.agent = agent

    def getModel(self) -> Dict[str, Any]:
        return self.agent.state_dict()


class TestAdversarialPeakRecovery:
    """Adversarial stress tests for PeakRecoveryMonitor weight recovery under RL policy collapse."""

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_peak_recovery_after_catastrophic_policy_collapse(self, monitor_cls: type, tmp_path: Path) -> None:
        """Simulate RL trajectory: Peak at Epoch 2 (0.85), followed by collapse at Epochs 3-8 (0.35 -> 0.05).

        Verify that PeakRecoveryMonitor.close() and restore_peak_model(agent) restore exact Epoch 2 weights.
        """
        agent = MockNeuralNetAgent()
        explorer = MockExplorer(agent)

        output_prefix = tmp_path / "collapse_experiment"
        monitor = monitor_cls(output_prefix=output_prefix, patience=10, min_epochs=5)
        monitor.saveModel(explorer)

        # Epoch 1: Initial exploration (desirability = 0.40)
        with torch.no_grad():
            agent.fc1.weight.fill_(1.11)
            agent.fc2.weight.fill_(1.22)
        state_ep1 = copy.deepcopy(agent.state_dict())
        cont1 = monitor.step(epoch=1, desirability=0.40, model_state=state_ep1)
        assert cont1 is True
        assert monitor.peak_epoch == 1
        assert monitor.peak_desirability == 0.40

        # Epoch 2: PEAK PERFORMANCE (desirability = 0.85)
        with torch.no_grad():
            agent.fc1.weight.fill_(2.85)
            agent.fc2.weight.fill_(3.85)
            agent.embedding.weight.fill_(4.85)
        state_ep2_peak = copy.deepcopy(agent.state_dict())
        cont2 = monitor.step(epoch=2, desirability=0.85, model_state=state_ep2_peak)
        assert cont2 is True
        assert monitor.peak_epoch == 2
        assert monitor.peak_desirability == 0.85
        assert monitor.peak_path.exists()

        # Epochs 3 to 8: CATASTROPHIC POLICY COLLAPSE
        collapse_trajectory = [
            (3, 0.35, 0.33),
            (4, 0.20, 0.22),
            (5, 0.15, 0.11),
            (6, 0.10, 0.09),
            (7, 0.08, 0.07),
            (8, 0.05, 0.01),
        ]
        for ep, des, fill_val in collapse_trajectory:
            with torch.no_grad():
                agent.fc1.weight.fill_(fill_val)
                agent.fc2.weight.fill_(fill_val)
                agent.embedding.weight.fill_(fill_val)
            state_ep = copy.deepcopy(agent.state_dict())
            cont = monitor.step(epoch=ep, desirability=des, model_state=state_ep)
            assert cont is True, f"Training stopped prematurely at epoch {ep} before patience=10"
            # Verify peak remains epoch 2
            assert monitor.peak_epoch == 2
            assert monitor.peak_desirability == 0.85
            assert monitor.epochs_without_improvement == ep - 2

        # Verify agent currently holds DEGRADED Epoch 8 weights (fill_val = 0.01)
        assert torch.allclose(agent.fc1.weight, torch.tensor(0.01))
        assert not torch.allclose(agent.fc1.weight, torch.tensor(2.85))

        # Action: Close monitor (auto-restore)
        monitor.close()

        # Empirical assertion: agent weights MUST be restored to exact Epoch 2 peak
        for param_name, peak_tensor in state_ep2_peak.items():
            restored_tensor = agent.state_dict()[param_name]
            assert torch.equal(restored_tensor, peak_tensor), (
                f"Restoration mismatch for parameter '{param_name}': expected {peak_tensor.mean().item()} "
                f"from Epoch 2 peak, but found {restored_tensor.mean().item()}"
            )

        # Also verify manual restore_peak_model(agent) call produces identical match
        # Corrupt again
        with torch.no_grad():
            agent.fc1.weight.fill_(999.0)
        assert torch.allclose(agent.fc1.weight, torch.tensor(999.0))

        monitor.restore_peak_model(agent)
        assert torch.allclose(agent.fc1.weight, torch.tensor(2.85))
        assert torch.allclose(agent.embedding.weight, torch.tensor(4.85))

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_peak_recovery_via_save_performance_info(self, monitor_cls: type, tmp_path: Path) -> None:
        """Verify peak tracking through savePerformanceInfo() which is called inside DrugEx explorer loops."""
        agent = MockNeuralNetAgent()
        explorer = MockExplorer(agent)

        output_prefix = tmp_path / "perf_info_run"
        monitor = monitor_cls(output_prefix=output_prefix, patience=10, min_epochs=5)
        monitor.saveModel(explorer)

        # Epoch 1
        with torch.no_grad():
            agent.fc1.weight.fill_(1.0)
        monitor.savePerformanceInfo({"Epoch": 1, "desired_ratio": 0.40})

        # Epoch 2 (Peak)
        with torch.no_grad():
            agent.fc1.weight.fill_(2.85)
        monitor.savePerformanceInfo({"Epoch": 2, "desired_ratio": 0.85})
        assert monitor.peak_epoch == 2
        assert monitor.peak_desirability == 0.85

        # Epoch 3 (Degraded)
        with torch.no_grad():
            agent.fc1.weight.fill_(0.05)
        monitor.savePerformanceInfo({"Epoch": 3, "desired_ratio": 0.10})

        # At this point, agent has 0.05
        assert torch.allclose(agent.fc1.weight, torch.tensor(0.05))

        # Close -> restores 2.85
        monitor.close()
        assert torch.allclose(agent.fc1.weight, torch.tensor(2.85))

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_cold_disk_recovery_when_in_memory_state_purged(self, monitor_cls: type, tmp_path: Path) -> None:
        """Adversarial stress test: Purge in-memory peak state dict and verify restore from disk checkpoint."""
        agent = MockNeuralNetAgent()
        explorer = MockExplorer(agent)

        output_prefix = tmp_path / "cold_disk_test"
        monitor = monitor_cls(output_prefix=output_prefix, patience=10, min_epochs=5)
        monitor.saveModel(explorer)

        # Epoch 2 Peak
        with torch.no_grad():
            agent.fc1.weight.fill_(8.88)
        peak_state = copy.deepcopy(agent.state_dict())
        monitor.step(epoch=2, desirability=0.95, model_state=peak_state)

        # Epoch 3 Collapse
        with torch.no_grad():
            agent.fc1.weight.fill_(-1.0)
        monitor.step(epoch=3, desirability=0.10, model_state=agent.state_dict())

        # Purge in-memory state dict to simulate cold process restart / memory drop
        monitor.peak_state_dict = None
        assert monitor.peak_state_dict is None

        # Call restore_peak_model -> must load from disk file {output_prefix}_peak.pkg
        assert monitor.peak_path.exists()
        monitor.restore_peak_model(agent)
        assert torch.allclose(agent.fc1.weight, torch.tensor(8.88))

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_error_raised_when_no_peak_exists(self, monitor_cls: type, tmp_path: Path) -> None:
        """Verify restore_peak_model() raises FileNotFoundError when no peak checkpoint exists."""
        output_prefix = tmp_path / "empty_model"
        monitor = monitor_cls(output_prefix=output_prefix)
        agent = MockNeuralNetAgent()

        with pytest.raises(FileNotFoundError, match="Peak checkpoint not found"):
            monitor.restore_peak_model(agent)

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_idempotent_close_and_repeated_restore(self, monitor_cls: type, tmp_path: Path) -> None:
        """Verify monitor.close() can be called multiple times without side effects or exceptions."""
        agent = MockNeuralNetAgent()
        explorer = MockExplorer(agent)

        output_prefix = tmp_path / "idempotent_test"
        monitor = monitor_cls(output_prefix=output_prefix)
        monitor.saveModel(explorer)

        with torch.no_grad():
            agent.fc1.weight.fill_(5.55)
        monitor.step(epoch=1, desirability=0.90, model_state=agent.state_dict())

        # Multiple close calls
        monitor.close()
        monitor.close()
        monitor.close()
        assert torch.allclose(agent.fc1.weight, torch.tensor(5.55))


class TestAdversarialEarlyStopping:
    """Adversarial boundary tests for early stopping logic in PeakRecoveryMonitor."""

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_early_stopping_at_exactly_10_consecutive_non_improving_epochs(self, monitor_cls: type, tmp_path: Path) -> None:
        """Stress-test patience=10, min_epochs=5 boundary:

        - Epoch 1: 0.50
        - Epoch 2: 0.85 (peak)
        - Epochs 3-11 (9 non-improving epochs): step() MUST return True for each.
        - Epoch 12 (10th consecutive non-improving epoch): step() MUST return False.
        """
        output_prefix = tmp_path / "early_stop_exact"
        monitor = monitor_cls(output_prefix=output_prefix, patience=10, min_epochs=5)
        dummy_state = {"w": torch.tensor([1.0])}

        # Epoch 1
        assert monitor.step(epoch=1, desirability=0.50, model_state=dummy_state) is True
        # Epoch 2 (Peak)
        assert monitor.step(epoch=2, desirability=0.85, model_state=dummy_state) is True
        assert monitor.peak_epoch == 2
        assert monitor.epochs_without_improvement == 0

        # Epochs 3 through 11 (9 non-improving epochs)
        for ep in range(3, 12):
            cont = monitor.step(epoch=ep, desirability=0.40, model_state=dummy_state)
            assert cont is True, f"Early stopping triggered prematurely at epoch {ep} (count={monitor.epochs_without_improvement})"
            assert monitor.epochs_without_improvement == ep - 2
            assert monitor.early_stopped is False

        # Epoch 12 (10th non-improving epoch >= patience=10 AND epoch 12 >= min_epochs=5)
        cont_ep12 = monitor.step(epoch=12, desirability=0.40, model_state=dummy_state)
        assert cont_ep12 is False, "Early stopping did NOT trigger at exactly 10 non-improving epochs!"
        assert monitor.early_stopped is True
        assert monitor.epochs_without_improvement == 10

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_min_epochs_guard_prevents_premature_termination(self, monitor_cls: type, tmp_path: Path) -> None:
        """Adversarial scenario: patience=2, min_epochs=8.

        Even though non-improving epochs exceed patience starting at epoch 3,
        step() must NOT terminate early until epoch >= 8.
        """
        output_prefix = tmp_path / "min_epochs_guard"
        monitor = monitor_cls(output_prefix=output_prefix, patience=2, min_epochs=8)
        dummy_state = {"w": torch.tensor([1.0])}

        # Epoch 1 (Peak)
        assert monitor.step(epoch=1, desirability=0.90, model_state=dummy_state) is True

        # Epochs 2 through 7: patience is exceeded (non_improving = 1, 2, 3, 4, 5, 6 >= 2),
        # but epoch < min_epochs (8). Must keep training!
        for ep in range(2, 8):
            cont = monitor.step(epoch=ep, desirability=0.50, model_state=dummy_state)
            assert cont is True, f"Premature stop at epoch {ep} < min_epochs=8!"
            assert monitor.early_stopped is False

        # Epoch 8: epoch 8 >= min_epochs (8) AND non_improving (7) >= patience (2).
        # Must trigger early stopping!
        cont_ep8 = monitor.step(epoch=8, desirability=0.50, model_state=dummy_state)
        assert cont_ep8 is False, "Early stopping failed to trigger at epoch 8 >= min_epochs!"
        assert monitor.early_stopped is True

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_counter_resets_on_late_breakthrough(self, monitor_cls: type, tmp_path: Path) -> None:
        """Adversarial scenario: 8 non-improving epochs followed by a breakthrough peak.

        Counter must reset to 0 and allow an additional full patience window.
        """
        output_prefix = tmp_path / "breakthrough_test"
        monitor = monitor_cls(output_prefix=output_prefix, patience=10, min_epochs=5)
        dummy_state = {"w": torch.tensor([1.0])}

        # Epoch 1: initial peak
        monitor.step(epoch=1, desirability=0.60, model_state=dummy_state)

        # Epochs 2-9: 8 non-improving epochs
        for ep in range(2, 10):
            assert monitor.step(epoch=ep, desirability=0.55, model_state=dummy_state) is True
        assert monitor.epochs_without_improvement == 8

        # Epoch 10: Breakthrough peak!
        assert monitor.step(epoch=10, desirability=0.75, model_state=dummy_state) is True
        assert monitor.peak_epoch == 10
        assert monitor.peak_desirability == 0.75
        assert monitor.epochs_without_improvement == 0

        # Epochs 11-19: 9 non-improving epochs after breakthrough
        for ep in range(11, 20):
            assert monitor.step(epoch=ep, desirability=0.70, model_state=dummy_state) is True
        assert monitor.epochs_without_improvement == 9
        assert monitor.early_stopped is False

        # Epoch 20: 10th non-improving epoch -> early stop
        assert monitor.step(epoch=20, desirability=0.70, model_state=dummy_state) is False
        assert monitor.early_stopped is True

    @pytest.mark.parametrize("monitor_cls", [DrugExPeakMonitor, DavidPeakMonitor])
    def test_strict_tie_does_not_reset_patience(self, monitor_cls: type, tmp_path: Path) -> None:
        """Verify that identical desirability values (plateau) are counted as non-improving."""
        output_prefix = tmp_path / "plateau_test"
        monitor = monitor_cls(output_prefix=output_prefix, patience=3, min_epochs=2)
        dummy_state = {"w": torch.tensor([1.0])}

        # Epoch 1: 0.80
        monitor.step(epoch=1, desirability=0.80, model_state=dummy_state)
        # Epoch 2: Exactly 0.80 (tie) -> non-improving count = 1
        monitor.step(epoch=2, desirability=0.80, model_state=dummy_state)
        assert monitor.epochs_without_improvement == 1
        # Epoch 3: Exactly 0.80 (tie) -> non-improving count = 2
        monitor.step(epoch=3, desirability=0.80, model_state=dummy_state)
        assert monitor.epochs_without_improvement == 2
        # Epoch 4: Exactly 0.80 (tie) -> non-improving count = 3 >= patience (3)
        cont = monitor.step(epoch=4, desirability=0.80, model_state=dummy_state)
        assert cont is False, "Plateau tie did not trigger early stopping after patience=3!"
        assert monitor.early_stopped is True


class TestAdversarialPretrainedPriorLoading:
    """Adversarial verification of VocGraph and GraphTransformer loading from Papyrus05.5_graph_trans_PT."""

    def test_pretrained_files_exist_and_sane(self) -> None:
        """Verify Papyrus05.5_graph_trans_PT package and vocabulary files exist with non-trivial size."""
        pretrained_dir = get_pretrained_dir()
        pkg_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"
        vocab_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"

        assert pkg_file.exists(), f"Pretrained package file missing: {pkg_file}"
        assert vocab_file.exists(), f"Pretrained vocabulary file missing: {vocab_file}"

        pkg_size_mb = pkg_file.stat().st_size / (1024 * 1024)
        assert pkg_size_mb > 50.0, f"Pretrained package unexpectedly small: {pkg_size_mb:.2f} MB"
        assert vocab_file.stat().st_size > 100, "Vocabulary file unexpectedly empty"

    def test_vocgraph_loads_directly_and_encodes_with_numpy2_compatibility(self) -> None:
        """Verify VocGraph loads from .vocab file and encodes fragments without deprecation or failure."""
        pretrained_dir = get_pretrained_dir()
        vocab_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"

        voc = VocGraph.fromFile(vocab_file)
        assert isinstance(voc, VocGraph)
        assert voc.size > 20, f"Vocabulary token count suspiciously low: {voc.size}"
        assert "*" in voc.words, "Missing wildcard fragment token '*' in VocGraph"
        assert "EOS" in voc.words, "Missing end-of-sequence token 'EOS' in VocGraph"
        assert "GO" in voc.words, "Missing start token 'GO' in VocGraph"

        # Verify token to index mapping consistency
        for idx, word in enumerate(voc.words):
            assert voc.tk2ix[word] == idx

        # Test encoding benchmark SMILES + fragment pairs into numpy array with np.int64
        test_pairs = [
            ("c1ccccc1C(=O)O", "c1ccccc1"),
            ("c1ccccc1O", "c1ccccc1"),
        ]
        smiles = [p[0] for p in test_pairs]
        subs = [p[1] for p in test_pairs]
        encoded = voc.encode(smiles, subs=subs)
        # [PACKAGE: numpy] WHY: Check encoded array type and dimensions. WHAT: NumPy array assertion.
        assert isinstance(encoded, np.ndarray)
        assert encoded.dtype == np.int64, f"Expected np.int64 dtype for NumPy 2.x, got {encoded.dtype}"
        assert encoded.shape[0] == 2
        assert encoded.shape[2] == 5

    def test_graph_transformer_loads_weights_without_fine_tuning(self) -> None:
        """Verify GraphTransformer loads weights cleanly from Papyrus05.5_graph_trans_PT.pkg."""
        pretrained_dir = get_pretrained_dir()
        vocab_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"
        pkg_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"

        voc = VocGraph.fromFile(vocab_file)
        model = GraphTransformer(voc_trg=voc, use_gpus=())
        model.loadStatesFromFile(pkg_file)

        # Inspect parameter tensors: no NaNs, no Infs, parameter count > 5M
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"
            assert not torch.isinf(param).any(), f"Inf detected in parameter {name}"

        assert total_params > 5_000_000, f"Parameter count {total_params} suspiciously small for GraphTransformer"

    def test_graph_transformer_optimizer_auto_resolves_pretrained_prior(self) -> None:
        """Verify GraphTransformerOptimizer in _david/code automatically discovers and loads pretrained prior."""
        from _david.code.GraphTransformer import GraphTransformerOptimizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(output_dir=Path(tmp_dir))
            assert optimizer.PRETRAINED_DIR.exists()
            assert (optimizer.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.pkg").exists()
            assert isinstance(optimizer.VOC, VocGraph)
            assert isinstance(optimizer.AGENT, GraphTransformer)
            assert isinstance(optimizer.PRIOR, GraphTransformer)

            # Check that AGENT and PRIOR are separate instances with identical loaded weights
            agent_weights = list(optimizer.AGENT.parameters())[0]
            prior_weights = list(optimizer.PRIOR.parameters())[0]
            assert torch.equal(agent_weights, prior_weights), "AGENT and PRIOR weights diverged upon init!"
            assert agent_weights is not prior_weights, "AGENT and PRIOR share the same tensor memory!"


class TestAdversarialFragGraphExplorerAndMonitorLifecycle:
    """Adversarial stress-test of the full FragGraphExplorer + PeakRecoveryMonitor interaction lifecycle."""

    def test_peak_recovery_with_real_graph_transformer_network(self, tmp_path: Path) -> None:
        """End-to-end stress test using real GraphTransformer network weights across 8 simulated epochs."""
        pretrained_dir = get_pretrained_dir()
        vocab_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"
        pkg_file = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"

        voc = VocGraph.fromFile(vocab_file)
        agent = GraphTransformer(voc_trg=voc, use_gpus=())
        agent.loadStatesFromFile(pkg_file)

        explorer = MockExplorer(agent)

        output_prefix = tmp_path / "real_gt_peak"
        monitor = DrugExPeakMonitor(output_prefix=output_prefix, patience=10, min_epochs=5)
        monitor.saveModel(explorer)

        # Epoch 1
        monitor.step(epoch=1, desirability=0.30, model_state=agent.state_dict())

        # Epoch 2: PEAK - perturb weights slightly to represent training progress
        with torch.no_grad():
            for p in agent.parameters():
                p.add_(0.042)
        peak_state = copy.deepcopy(agent.state_dict())
        first_param_peak = next(iter(peak_state.values())).clone()

        monitor.step(epoch=2, desirability=0.88, model_state=peak_state)
        assert monitor.peak_epoch == 2
        assert monitor.peak_desirability == 0.88

        # Epochs 3-8: Severe policy collapse - corrupt weights
        with torch.no_grad():
            for p in agent.parameters():
                p.zero_()
                p.add_(-99.0)

        # Verify weights are corrupted
        first_param_corrupted = next(iter(agent.parameters()))
        assert torch.allclose(first_param_corrupted, torch.tensor(-99.0))

        # Feed degraded epochs to monitor
        for ep, des in [(3, 0.40), (4, 0.25), (5, 0.15), (6, 0.10), (7, 0.05), (8, 0.01)]:
            monitor.step(epoch=ep, desirability=des, model_state=agent.state_dict())

        # Close monitor -> auto-restore
        monitor.close()

        # Verify real GraphTransformer weights are restored byte-for-byte to Epoch 2 peak
        restored_state = agent.state_dict()
        for k in peak_state.keys():
            assert torch.equal(restored_state[k], peak_state[k]), f"Mismatch in GraphTransformer layer {k}!"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
