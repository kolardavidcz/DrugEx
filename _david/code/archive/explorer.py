from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.monitors import NullMonitor

logger = logging.getLogger(__name__)


def _save(state, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _population(metrics: dict) -> dict[str, float]:
    return {
        "valid_ratio": round(float(metrics.get("valid_ratio", 0.0)), 4),
        "unique_ratio": round(float(metrics.get("unique_ratio", 0.0)), 4),
    }


class DualCheckpointExplorer(SequenceExplorer):
    """``SequenceExplorer`` tracking ``desired_ratio`` and ``avg_gmean`` independently.

    Three checkpoints: ``desired_ratio.pkg`` (best, frozen when its patience expires),
    ``desired_ratio_at_stop.pkg`` (the live policy at that epoch) and ``gmean.pkg``. The run
    ends only once *both* windows expire, and best-state reload is deliberately absent.
    """

    def __init__(
        self,
        *args,
        desired_ratio_path: Path,
        desired_ratio_at_stop_path: Path,
        gmean_path: Path,
        training_samples_path: Path | None = None,
        n_eval: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.desired_ratio_path = Path(desired_ratio_path)
        self.desired_ratio_at_stop_path = Path(desired_ratio_at_stop_path)
        self.gmean_path = Path(gmean_path)
        # Native discards the scored policy-gradient batch, 10x the evaluation arm.
        self.training_samples_path = Path(training_samples_path) if training_samples_path else None
        self.n_eval = int(n_eval)

    def _record_training_samples(self, epoch: int) -> None:
        scored = getattr(self.env, "scores", None)
        if self.training_samples_path is None or scored is None or not len(scored):
            return
        frame = scored.copy()
        frame.insert(0, "Epoch", epoch)
        frame.to_csv(
            self.training_samples_path,
            sep="\t",
            index=False,
            mode="a",
            header=not self.training_samples_path.exists(),
        )

    def _save_at_stop(self, metrics: dict, population: dict) -> float:
        population["desired_ratio_at_stop"] = _population(metrics)
        _save(self.getModel(), self.desired_ratio_at_stop_path)
        return float(metrics["desired_ratio"])

    def _evaluate(self) -> tuple:
        smiles = self.agent.sample(self.n_eval)
        scores = self.agent.evaluate(
            smiles, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles
        )
        scores["SMILES"] = smiles
        metrics = self.getNovelMoleculeMetrics(scores)
        # Over avg_gmean's own denominator, or a dip in validity reads as a fall in shape.
        for scorer in getattr(self.env, "scorers", []):
            if hasattr(scorer, "component_frame"):
                unique = scores[scores["Valid"] == 1].drop_duplicates(subset="SMILES")
                split = scorer.component_frame(unique["SMILES"].tolist())
                for column, values in split.items():
                    metrics[f"avg_{column}"] = float(np.mean(values)) if values else 0.0
                break
        return scores, metrics

    def fit(
        self,
        train_loader=None,
        valid_loader=None,
        monitor=None,
        epochs=1000,
        patience=50,
        reload_interval=None,
        criteria=None,
        min_epochs=100,
    ):
        """``criteria`` and ``reload_interval`` are ignored."""
        self.monitor = monitor or NullMonitor()

        # -1.0: desired_ratio is legitimately 0.0 early, and each path must be written once.
        best = {"desired_ratio": -1.0, "avg_gmean": -1.0}
        last = {"desired_ratio": 0, "avg_gmean": 0}
        paths = {"desired_ratio": self.desired_ratio_path, "avg_gmean": self.gmean_path}
        population: dict[str, dict] = {}
        frozen = False
        peak_desired_ratio = -1.0

        for epoch in range(1, epochs + 1):
            smiles, seqs = self.forward()
            train_loss = self.policy_gradient(smiles, seqs, epoch=epoch, epochs=epochs)
            scores, metrics = self._evaluate()
            metrics["loss_train"] = train_loss

            # Tracked even once frozen, so a post-freeze peak is still reported.
            peak_desired_ratio = max(peak_desired_ratio, float(metrics["desired_ratio"]))

            for criterion in ("desired_ratio", "avg_gmean"):
                if criterion == "desired_ratio" and frozen:
                    continue
                if metrics[criterion] <= best[criterion]:
                    continue
                best[criterion] = float(metrics[criterion])  # JSON cannot take numpy scalars
                last[criterion] = epoch
                population[criterion] = _population(metrics)
                _save(self.getModel(), paths[criterion])

            self._record_training_samples(epoch)

            # The base class logs `best_epoch` from these.
            self.best_value = best["avg_gmean"]
            self.last_save = last["avg_gmean"]
            self.logPerformanceAndCompounds(epoch, metrics, scores)
            logger.info(
                "epoch %d/%d: desired_ratio=%.4f (best %.4f @ %d%s) avg_gmean=%.4f "
                "(best %.4f @ %d) loss=%.4f",
                epoch,
                epochs,
                metrics["desired_ratio"],
                best["desired_ratio"],
                last["desired_ratio"],
                ", frozen" if frozen else "",
                metrics["avg_gmean"],
                best["avg_gmean"],
                last["avg_gmean"],
                train_loss,
            )

            if epoch < min_epochs:
                continue
            if not frozen and epoch - last["desired_ratio"] > patience:
                frozen, frozen_epoch = True, epoch
                at_stop = self._save_at_stop(metrics, population)
                logger.info(
                    "desired_ratio checkpoint frozen at epoch %d (best %.4f @ epoch %d, "
                    "live %.4f); training continues for avg_gmean",
                    epoch,
                    best["desired_ratio"],
                    last["desired_ratio"],
                    at_stop,
                )
            if frozen and epoch - last["avg_gmean"] > patience:
                logger.info(
                    "stopping at epoch %d: both windows expired (avg_gmean last improved at "
                    "epoch %d, desired_ratio at %d)",
                    epoch,
                    last["avg_gmean"],
                    last["desired_ratio"],
                )
                break

        if not frozen:
            # The epoch cap arrived first, so the final policy plays that part.
            frozen_epoch = epoch
            at_stop = self._save_at_stop(metrics, population)
            logger.info(
                "desired_ratio never exhausted its patience window: its checkpoint holds the "
                "best value seen (%.4f @ epoch %d); at-stop is the final policy",
                best["desired_ratio"],
                last["desired_ratio"],
            )
        self.monitor.close()
        return {
            "epochs_run": epoch,
            "best_desired_ratio_before_freeze": best["desired_ratio"],
            "best_desired_ratio_epoch": last["desired_ratio"],
            "peak_desired_ratio_any_epoch": peak_desired_ratio,
            "desired_ratio_frozen": frozen,
            "desired_ratio_freeze_epoch": frozen_epoch,
            "desired_ratio_at_stop": at_stop,
            "best_avg_gmean": best["avg_gmean"],
            "best_avg_gmean_epoch": last["avg_gmean"],
            "checkpoint_population": population,
        }
