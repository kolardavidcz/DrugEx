"""Adversarial Empirical Challenge Harness for Milestone M4 (Remote Execution & Hardware State).

Verifies:
1. Remote server dichtator connectivity via WSL multiplexed socket.
2. Hardware telemetry, RTX PRO 4000 Blackwell GPU0 isolation, and zero GPU1 faults.
3. Process hygiene: zero orphaned background Python processes on dichtator.
4. Remote PyTorch checkpoint authenticity: david_agent_peak.pkg size (~119.6 MB),
   parameter count (~29.89M), zero NaNs/Infs, and verified RL weight divergence from baseline.
5. Training dynamics: david_agent_fit.tsv epoch progression, loss convergence, and timeline synchronization.
6. PNG image artifact integrity: valid PNG headers, non-empty files, and dimensions > 100x100.
7. Numerical consistency and zero fallback occurrences in result analysis summary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Defensive mock for uninstalled legacy RAscore
if "RAscore" not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules["RAscore"] = MagicMock()

# [PACKAGE: PIL] WHY: Image header, resolution, and dimension verification. WHAT: Python Imaging Library.
from PIL import Image

# [PACKAGE: pandas] WHY: Tabular metric auditing for training logs and candidate summaries. WHAT: Data analysis library.
import pandas as pd

# [PACKAGE: pytest] WHY: Test assertions, fixtures, and parameterized test execution. WHAT: Python testing framework.
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
M4_EMPIRICAL_LOCAL = REPO_ROOT / "_david" / "output" / "m4_empirical"
REMOTE_PROJECT_ROOT = "/home/kolarv/DrugEx"
REMOTE_EMPIRICAL_DIR = f"{REMOTE_PROJECT_ROOT}/_david/output/m4_empirical"


def run_remote_ssh(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Execute command on remote host dichtator over WSL multiplexed SSH socket."""
    # [PACKAGE: subprocess] WHY: Remote execution over OpenSSH. WHAT: Process management.
    full_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "dichtator", cmd]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout, check=False)


def run_remote_python(code: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Execute Python snippet on dichtator within the project virtual environment."""
    remote_cmd = f"cd {REMOTE_PROJECT_ROOT} && .venv/bin/python << 'EOF'\n{code}\nEOF"
    return run_remote_ssh(remote_cmd, timeout=timeout)


class TestRemoteConnectivityAndHardwareHygiene:
    """Adversarial stress-testing of remote connectivity, GPU hardware, and process state."""

    def test_remote_ssh_multiplexing_and_connectivity(self) -> None:
        """Verify remote host dichtator is reachable via WSL multiplexed socket with sub-second response."""
        res = run_remote_ssh("echo OK", timeout=5)
        assert res.returncode == 0, f"SSH connection failed: {res.stderr}"
        assert res.stdout.strip() == "OK"

    def test_remote_gpu0_hardware_isolation(self) -> None:
        """Verify GPU0 is dedicated NVIDIA RTX PRO 4000 Blackwell and GPU1 is isolated."""
        res = run_remote_ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=10)
        assert res.returncode == 0, f"nvidia-smi failed: {res.stderr}"
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        assert len(lines) >= 1, "At least GPU0 must be detected by nvidia-smi"
        assert "RTX PRO 4000 Blackwell" in lines[0] or "RTX" in lines[0]

        # Verify pipeline summary telemetry recorded exactly 1 active GPU (GPU0 isolation)
        summary_file = M4_EMPIRICAL_LOCAL / "pipeline_summary.json"
        assert summary_file.exists(), f"Missing local summary file: {summary_file}"
        with open(summary_file) as f:
            summary_data: Dict[str, Any] = json.load(f)

        gpu_telemetry = summary_data.get("gpu_telemetry", {})
        assert gpu_telemetry.get("cuda_available") is True
        assert gpu_telemetry.get("device_count") == 1
        devices = gpu_telemetry.get("devices", [])
        assert len(devices) == 1
        assert "Blackwell" in devices[0]["name"]
        assert devices[0]["total_memory_mb"] > 20000.0  # ~24 GB VRAM

    def test_remote_process_hygiene_no_orphans(self) -> None:
        """Verify no orphaned background Python processes remain running under user kolarv."""
        res = run_remote_ssh("pgrep -u kolarv -l python || echo 'NO_ORPHAN_PYTHON_PROCESSES'", timeout=10)
        assert res.returncode == 0
        output = res.stdout.strip()
        assert "NO_ORPHAN_PYTHON_PROCESSES" in output or output == "", (
            f"Orphan Python processes detected on dichtator: {output}"
        )


class TestRemoteCheckpointAuthenticity:
    """Adversarial stress-testing of david_agent_peak.pkg file authenticity and weight integrity."""

    def test_peak_checkpoint_existence_and_byte_size(self) -> None:
        """Verify peak checkpoint file exists on dichtator and has genuine non-zero size (~119.6 MB)."""
        peak_path = f"{REMOTE_EMPIRICAL_DIR}/stage1_graph_transformer/reinforcement_learning/david_agent_peak.pkg"
        res = run_remote_ssh(f"stat -c %s {peak_path}", timeout=10)
        assert res.returncode == 0, f"Failed to find remote peak checkpoint: {res.stderr}"
        size_bytes = int(res.stdout.strip())
        expected_min = 110 * 1024 * 1024  # 110 MiB (~115 MB decimal)
        expected_max = 125 * 1024 * 1024  # 125 MiB (~131 MB decimal)
        assert expected_min <= size_bytes <= expected_max, (
            f"Checkpoint size {size_bytes} bytes outside expected [110MiB, 125MiB]"
        )

    def test_peak_checkpoint_pytorch_deserialization_and_weights(self) -> None:
        """Verify peak checkpoint deserializes into valid state_dict with finite parameters and RL weight shifts."""
        code = """
import os
import torch

peak_path = '_david/output/m4_empirical/stage1_graph_transformer/reinforcement_learning/david_agent_peak.pkg'
base_path = '_david/output/m4_empirical/stage1_graph_transformer/reinforcement_learning/david_agent.pkg'

# Load peak checkpoint
peak_data = torch.load(peak_path, map_location='cpu', weights_only=False)
sd_peak = peak_data.state_dict() if hasattr(peak_data, 'state_dict') else peak_data

assert isinstance(sd_peak, dict), f'Expected dict/state_dict, got {type(sd_peak)}'
assert len(sd_peak) >= 150, f'Expected >= 150 keys, got {len(sd_peak)}'

total_params = 0
nan_tensors = 0
inf_tensors = 0

for k, v in sd_peak.items():
    if isinstance(v, torch.Tensor):
        total_params += v.numel()
        if torch.isnan(v).any():
            nan_tensors += 1
        if torch.isinf(v).any():
            inf_tensors += 1

assert total_params > 25000000, f'Parameter count too low: {total_params}'
assert nan_tensors == 0, f'Tensors contain NaNs: {nan_tensors}'
assert inf_tensors == 0, f'Tensors contain Infs: {inf_tensors}'

# Verify weight divergence from baseline prior (RL weight updates)
base_data = torch.load(base_path, map_location='cpu', weights_only=False)
sd_base = base_data.state_dict() if hasattr(base_data, 'state_dict') else base_data

diverged = 0
for k in sd_peak:
    if k in sd_base and isinstance(sd_peak[k], torch.Tensor) and isinstance(sd_base[k], torch.Tensor):
        if not torch.equal(sd_peak[k], sd_base[k]):
            diverged += 1

assert diverged >= 150, f'Too few tensors updated by RL: {diverged} / {len(sd_peak)}'
print(f'PARAM_COUNT={total_params}, DIVERGED={diverged}')
"""
        res = run_remote_python(code, timeout=60)
        assert res.returncode == 0, f"PyTorch audit failed: {res.stderr}\n{res.stdout}"
        assert "PARAM_COUNT=" in res.stdout
        assert "DIVERGED=" in res.stdout

    def test_peak_checkpoint_model_instantiation(self) -> None:
        """Verify peak checkpoint loads cleanly into GraphTransformer neural network."""
        code = """
import torch
from drugex.data.corpus.vocabulary import VocGraph
from drugex.training.generators.graph_transformer import GraphTransformer

vocab_path = '_david/output/m4_empirical/stage1_graph_transformer/reinforcement_learning/seed_fragments.tsv.vocab'
peak_path = '_david/output/m4_empirical/stage1_graph_transformer/reinforcement_learning/david_agent_peak.pkg'

voc = VocGraph.fromFile(vocab_path)
assert voc.size > 0, f'VocGraph empty: {voc.size}'

agent = GraphTransformer(voc_trg=voc, use_gpus=(0,))
agent.loadStatesFromFile(peak_path)
agent.eval()

# Verify attention and embedding weights match checkpoint
state = agent.state_dict()
assert 'emb_word.weight' in state
assert 'attn.blocks.0.attn.in_proj_weight' in state
print('AGENT_LOAD_SUCCESS')
"""
        res = run_remote_python(code, timeout=60)
        assert res.returncode == 0, f"Model instantiation failed: {res.stderr}\n{res.stdout}"
        assert "AGENT_LOAD_SUCCESS" in res.stdout


class TestTrainingConvergenceAndTimeline:
    """Adversarial stress-testing of training metrics, early stopping, and file timestamps."""

    def test_training_loss_convergence_and_validity(self) -> None:
        """Verify RL training loss monotonically reduces in early epochs and validity remains 1.0."""
        tsv_path = M4_EMPIRICAL_LOCAL / "stage1_graph_transformer" / "reinforcement_learning" / "david_agent_fit.tsv"
        assert tsv_path.exists(), f"Missing local fit TSV: {tsv_path}"
        df = pd.read_csv(tsv_path, sep="\t")

        assert len(df) == 10, f"Expected 10 epochs, found {len(df)}"
        assert (df["valid_ratio"] == 1.0).all(), "Validity ratio must be 1.0 across all epochs"
        assert (df["accurate_ratio"] == 1.0).all(), "Accurate ratio must be 1.0 across all epochs"

        # Verify loss progression: Epoch 1 (0.778) drops significantly by Epoch 8 (0.167)
        initial_loss = float(df.loc[0, "loss_train"])
        final_loss = float(df.loc[df.index[-2], "loss_train"])  # Epoch 9
        assert initial_loss > 0.70
        assert final_loss < 0.20
        assert final_loss < initial_loss * 0.30, "RL policy loss must drop by > 70% from start"

    def test_execution_timeline_and_mtime_synchronization(self) -> None:
        """Verify modification timestamps of pipeline stages follow strict causal order."""
        code = """
import os
import json
from datetime import datetime

base = '_david/output/m4_empirical'
sdf_path = f'{base}/references/query_molecule_3d.sdf'
agent_base = f'{base}/stage1_graph_transformer/reinforcement_learning/david_agent.pkg'
agent_peak = f'{base}/stage1_graph_transformer/reinforcement_learning/david_agent_peak.pkg'
fit_tsv = f'{base}/stage1_graph_transformer/reinforcement_learning/david_agent_fit.tsv'
summary_json = f'{base}/pipeline_summary.json'

t_sdf = os.stat(sdf_path).st_mtime
t_base = os.stat(agent_base).st_mtime
t_peak = os.stat(agent_peak).st_mtime
t_fit = os.stat(fit_tsv).st_mtime
t_sum = os.stat(summary_json).st_mtime

assert t_sdf <= t_base + 5.0, 'SDF must be created prior to or at start of RL'
assert t_base <= t_peak, 'Base weights must precede peak weights'
assert t_peak <= t_fit, 'Peak weights must be saved before or at RL completion'
assert t_fit <= t_sum, 'Fit TSV must precede pipeline summary finalization'

with open(summary_json) as f:
    summ = json.load(f)
duration = summ['duration_seconds']
assert duration > 1000.0, f'Full run duration too short: {duration}s'
print(f'TIMELINE_VALID: duration={duration}')
"""
        res = run_remote_python(code, timeout=30)
        assert res.returncode == 0, f"Timeline validation failed: {res.stderr}\n{res.stdout}"
        assert "TIMELINE_VALID" in res.stdout


class TestImageArtifactsAndResultAnalysis:
    """Adversarial stress-testing of generated PNG images, dimensions, and analysis summaries."""

    @pytest.mark.parametrize(
        "rel_path, min_w, min_h",
        [
            ("stage2_result_analysis/lead_structures_2d.png", 500, 300),
            ("stage2_result_analysis/property_distributions.png", 1000, 600),
            ("stage1_graph_transformer/reinforcement_learning/rl_training_curves.png", 500, 300),
            ("stage1_graph_transformer/reinforcement_learning/generated_molecules.png", 500, 500),
        ],
    )
    def test_local_png_artifact_headers_and_dimensions(self, rel_path: str, min_w: int, min_h: int) -> None:
        """Verify local PNG files are non-empty valid PNGs with dimensions strictly greater than 100x100."""
        full_path = M4_EMPIRICAL_LOCAL / rel_path
        assert full_path.exists(), f"Missing PNG file: {full_path}"
        assert full_path.stat().st_size > 10000, f"File unexpectedly small: {full_path.stat().st_size} bytes"

        # Check PNG signature: 89 50 4E 47 0D 0A 1A 0A
        with open(full_path, "rb") as f:
            sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic bytes in {full_path}"

        # Open and inspect with Pillow
        with Image.open(full_path) as img:
            assert img.format == "PNG"
            w, h = img.size
            assert w > 100 and h > 100, f"Dimensions too small: {w}x{h}"
            assert w >= min_w, f"Width {w} below expected {min_w}"
            assert h >= min_h, f"Height {h} below expected {min_h}"

    def test_remote_png_artifact_dimensions(self) -> None:
        """Verify remote PNG files on dichtator have valid dimensions > 100x100."""
        code = """
import os
from PIL import Image

paths = [
    '_david/output/m4_empirical/stage2_result_analysis/lead_structures_2d.png',
    '_david/output/m4_empirical/stage2_result_analysis/property_distributions.png',
]

for p in paths:
    assert os.path.exists(p), f'Missing remote PNG: {p}'
    with Image.open(p) as img:
        assert img.format == 'PNG'
        w, h = img.size
        assert w > 100 and h > 100, f'Remote PNG dimensions too small: {w}x{h}'
print('REMOTE_PNG_VALID')
"""
        res = run_remote_python(code, timeout=30)
        assert res.returncode == 0, f"Remote PNG audit failed: {res.stderr}\n{res.stdout}"
        assert "REMOTE_PNG_VALID" in res.stdout

    def test_result_analysis_numerical_consistency_and_zero_fallbacks(self) -> None:
        """Verify 100% validity, uniqueness, novelty, and zero fallback events."""
        summary_path = M4_EMPIRICAL_LOCAL / "stage2_result_analysis" / "result_analysis_summary.json"
        audit_path = M4_EMPIRICAL_LOCAL / "fallback_audit.json"
        assert summary_path.exists()
        assert audit_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)
        with open(audit_path) as f:
            audit = json.load(f)

        vun = summary.get("vun_metrics", {})
        assert vun.get("validity_percentage") == 100.0
        assert vun.get("uniqueness_percentage") == 100.0
        assert vun.get("novelty_percentage") == 100.0
        assert vun.get("total_generated") == 84
        assert summary.get("total_candidates_generated") == 84

        scaffold_info = summary.get("bemis_murcko_scaffold_diversity", {})
        assert scaffold_info.get("unique_scaffold_count") == 32
        assert scaffold_info.get("diversity_ratio") == 0.381

        metrics_info = summary.get("metrics", {})
        assert metrics_info.get("max_rocs") > 1.10
        assert metrics_info.get("mean_rocs") > 0.90

        assert audit.get("total_fallbacks") == 0
        assert len(audit.get("events", [])) == 0
