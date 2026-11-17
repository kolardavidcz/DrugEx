"""The reinforcement-learning loop: one criterion, one checkpoint.

A run trains on ``desired_ratio`` until its patience window expires, then keeps the policy
it stopped on. There is no second criterion and no best-epoch checkpoint: the model that
would be shipped is the model training ended with, and saving anything else invites reading
a score that no delivered molecule produced.
"""

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


class StopCheckpointExplorer(SequenceExplorer):
    """``SequenceExplorer`` that stops on ``desired_ratio`` and keeps the policy at stop.

    ``desired_ratio_at_stop.pkg`` is written once, after the loop, from the live agent. A
    run killed at walltime after training can therefore be scored from that checkpoint
    without retraining, and a run killed during training has no checkpoint to mistake for
    a finished policy.
    """

    def __init__(
        self,
        *args,
        checkpoint_path: Path,
        training_samples_path: Path | None = None,
        n_eval: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_path = Path(checkpoint_path)
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

    def _evaluate(self) -> tuple:
        smiles = self.agent.sample(self.n_eval)
        scores = self.agent.evaluate(
            smiles, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles
        )
        scores["SMILES"] = smiles
        metrics = self.getNovelMoleculeMetrics(scores)
        # Diagnostics only: avg_gmean's denominator is the valid unique set, so the
        # components would otherwise read as a fall in shape when validity dipped.
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
        """``criteria`` and ``reload_interval`` exist for signature compatibility only."""
        self.monitor = monitor or NullMonitor()

        # -1.0: desired_ratio is legitimately 0.0 early, and the checkpoint is written once.
        best = -1.0
        last_improved = 0
        peak = -1.0
        population: dict[str, float] = {}
        frozen = False
        metrics: dict = {}
        epoch = 0

        for epoch in range(1, epochs + 1):
            smiles, seqs = self.forward()
            train_loss = self.policy_gradient(smiles, seqs, epoch=epoch, epochs=epochs)
            scores, metrics = self._evaluate()
            metrics["loss_train"] = train_loss

            # Tracked every epoch, so a post-peak slide is visible in the summary.
            peak = max(peak, float(metrics["desired_ratio"]))
            if float(metrics["desired_ratio"]) > best:
                best = float(metrics["desired_ratio"])
                last_improved = epoch
                population = _population(metrics)

            self._record_training_samples(epoch)

            # The base class logs `best_epoch` from these.
            self.best_value = best
            self.last_save = last_improved
            self.logPerformanceAndCompounds(epoch, metrics, scores)
            logger.info(
                "epoch %d/%d: desired_ratio=%.4f (best %.4f @ %d) avg_gmean=%.4f loss=%.4f",
                epoch,
                epochs,
                metrics["desired_ratio"],
                best,
                last_improved,
                metrics["avg_gmean"],
                train_loss,
            )

            if epoch >= min_epochs and epoch - last_improved > patience:
                frozen = True
                break

        # The one checkpoint: the live policy at the epoch training ended. Written before
        # the final arms are sampled, so a walltime kill during scoring leaves it behind.
        _save(self.getModel(), self.checkpoint_path)
        at_stop = float(metrics.get("desired_ratio", 0.0))
        if frozen:
            logger.info(
                "stopping at epoch %d: desired_ratio last improved at epoch %d (best %.4f); "
                "live policy %.4f saved to %s",
                epoch,
                last_improved,
                best,
                at_stop,
                self.checkpoint_path.name,
            )
        else:
            logger.info(
                "epoch cap reached at %d: desired_ratio last improved at epoch %d (best "
                "%.4f); live policy %.4f saved to %s",
                epoch,
                last_improved,
                best,
                at_stop,
                self.checkpoint_path.name,
            )
        self.monitor.close()
        return {
            "epochs_run": epoch,
            "best_desired_ratio": best,
            "best_desired_ratio_epoch": last_improved,
            "peak_desired_ratio_any_epoch": peak,
            "desired_ratio_frozen": frozen,
            "desired_ratio_freeze_epoch": epoch,
            "desired_ratio_at_stop": at_stop,
            "checkpoint_population": {"desired_ratio_at_stop": population},
        }
