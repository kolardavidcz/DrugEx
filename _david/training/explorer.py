"""An explorer that keeps two checkpoints instead of one."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.monitors import NullMonitor

logger = logging.getLogger(__name__)


def _save_atomically(state, path: Path) -> None:
    """Write a checkpoint through a temporary file, then rename.

    PBS sends SIGTERM and the job kills the child shortly after, so a save in flight can
    be cut in half. A truncated .pkg is worse than a missing one: the copy-back presents
    it as the run's best checkpoint and it fails only when someone loads it.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


class DualCheckpointExplorer(SequenceExplorer):
    """``SequenceExplorer`` tracking ``desired_ratio`` and ``avg_gmean`` independently.

    DrugEx fits against a single ``criteria``, so a native run yields one checkpoint. This
    experiment wants both, because they answer different questions: ``desired_ratio`` is
    how much of the output clears every objective's threshold, ``avg_gmean`` is how good
    the output is on average without letting one objective collapse.

    - ``desired_ratio.pkg`` holds the best ``desired_ratio`` seen. It is frozen the first
      time that metric's patience window expires, and is never written again even if
      ``desired_ratio`` improves later;
    - ``desired_ratio_at_stop.pkg`` is the live policy at that same freeze epoch. The two
      differ by up to ``patience`` epochs of training, and reading the spec either way is
      defensible, so both are written and the thesis can report either without a rerun;
    - ``gmean.pkg`` holds the best ``avg_gmean`` seen.

    The run ends only once *both* patience windows have expired, so neither criterion is
    cut short by the other: stopping on ``avg_gmean`` alone would end a run while
    ``desired_ratio`` was still improving.

    DrugEx's periodic best-state reload is not implemented here at all. Copying a past
    checkpoint back over the live agent every n epochs would mean the policy the run ends
    on is not the policy it trained into, and neither checkpoint would be a plain record
    of an epoch.
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
        # The policy gradient scores n_samples molecules every epoch and native throws
        # them away, keeping only the n_eval evaluation arm. That is 10x more scored
        # molecules discarded than kept, and the scoring -- the expensive part -- has
        # already happened. For a de novo design project the molecules *are* the output,
        # so they are appended to disk as they are produced.
        self.training_samples_path = Path(training_samples_path) if training_samples_path else None
        # DrugEx hard-wires this arm to n_samples/10 inside its own fit(). This class
        # replaces fit(), so the size of the arm both criteria are measured on becomes a
        # setting instead of a constant: desired_ratio over 50 molecules moves in steps of
        # 2 %, which is most of why it is the noisier of the two criteria.
        self.n_eval = int(n_eval)

    def _record_training_samples(self, epoch: int) -> None:
        """Append this epoch's policy-gradient batch, already scored, to a TSV."""
        if self.training_samples_path is None:
            return
        scored = getattr(self.env, "scores", None)
        if scored is None or not len(scored):
            return
        frame = scored.copy()
        frame.insert(0, "Epoch", epoch)
        header = not self.training_samples_path.exists()
        frame.to_csv(self.training_samples_path, sep="\t", index=False, mode="a", header=header)

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
        """Train, tracking both criteria. ``criteria`` and ``reload_interval`` are ignored.

        They stay in the signature only because this overrides ``SequenceExplorer.fit``:
        this explorer tracks both criteria by construction and never reloads.
        """
        self.monitor = monitor if monitor else NullMonitor()

        # -1.0, not 0.0: desired_ratio is legitimately 0.0 for many early epochs, and a run
        # must not finish with a checkpoint path that was never written.
        best = {"desired_ratio": -1.0, "avg_gmean": -1.0}
        last = {"desired_ratio": 0, "avg_gmean": 0}
        paths = {"desired_ratio": self.desired_ratio_path, "avg_gmean": self.gmean_path}
        frozen = False
        peak_desired_ratio = -1.0
        health: dict[str, dict] = {}

        for epoch in range(1, epochs + 1):
            smiles, seqs = self.forward()
            train_loss = self.policy_gradient(smiles, seqs, epoch=epoch, epochs=epochs)

            # A held-out arm, drawn fresh from the updated policy: these molecules are
            # measured, never trained on.
            eval_smiles = self.agent.sample(self.n_eval)
            scores = self.agent.evaluate(
                eval_smiles, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles
            )
            scores["SMILES"] = eval_smiles

            metrics = self.getNovelMoleculeMetrics(scores)
            metrics["loss_train"] = train_loss
            # Diagnostics, not criteria: the pose search already computed these, so the
            # split costs nothing. Neither value is ever compared against a threshold or
            # used to choose a checkpoint.
            for scorer in getattr(self.env, "scorers", []):
                if hasattr(scorer, "component_frame"):
                    # Valid unique molecules only, exactly the denominator
                    # getNovelMoleculeMetrics uses for avg_gmean. Averaging over all
                    # n_eval with invalid molecules forced to 0.0 would make a dip in
                    # validity look like a fall in shape or colour.
                    keep = scores.assign(SMILES=eval_smiles)
                    keep = keep[keep["Valid"] == 1].drop_duplicates(subset="SMILES")
                    split = scorer.component_frame(keep["SMILES"].tolist())
                    for column, values in split.items():
                        metrics[f"avg_{column}"] = float(np.mean(values)) if values else 0.0
                    break

            # Tracked whether or not the checkpoint is frozen, so a post-freeze peak is
            # reported rather than silently discarded: the checkpoint stops updating, the
            # record of what the policy reached does not.
            peak_desired_ratio = max(peak_desired_ratio, float(metrics["desired_ratio"]))

            for criterion in ("desired_ratio", "avg_gmean"):
                if criterion == "desired_ratio" and frozen:
                    continue
                if metrics[criterion] > best[criterion]:
                    # float(), not the numpy scalar pandas hands back: the summary is
                    # serialized to JSON, where a numpy scalar is not guaranteed to encode.
                    best[criterion] = float(metrics[criterion])
                    last[criterion] = epoch
                    # avg_gmean is averaged over unique valid molecules while the ratios
                    # divide by n_eval, so an epoch that collapses to one good molecule
                    # can post a higher avg_gmean than any healthy epoch and take the
                    # checkpoint. Recording the population it was measured on makes such
                    # a checkpoint visible instead of plausible.
                    health[criterion] = {
                        "valid_ratio": round(float(metrics.get("valid_ratio", 0.0)), 4),
                        "unique_ratio": round(float(metrics.get("unique_ratio", 0.0)), 4),
                    }
                    _save_atomically(self.getModel(), paths[criterion])

            self._record_training_samples(epoch)

            # The base class logs `best_epoch` from these: point them at the criterion the
            # run actually stops on.
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
                frozen = True
                frozen_epoch = epoch
                # The other reading of "frozen when patience runs out": the policy as it
                # stands now, patience epochs past its own best. Saved alongside the best
                # so the choice between the two is a reporting decision, not a rerun.
                _save_atomically(self.getModel(), self.desired_ratio_at_stop_path)
                at_stop = float(metrics["desired_ratio"])
                health["desired_ratio_at_stop"] = {
                    "valid_ratio": round(float(metrics.get("valid_ratio", 0.0)), 4),
                    "unique_ratio": round(float(metrics.get("unique_ratio", 0.0)), 4),
                }
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
                    "stopping at epoch %d: both windows expired (avg_gmean last improved "
                    "at epoch %d, desired_ratio at %d)",
                    epoch,
                    last["avg_gmean"],
                    last["desired_ratio"],
                )
                break

        if not frozen:
            # The epoch cap arrived first, so there is no freeze epoch; the policy the run
            # ended on plays that role, and the at-stop checkpoint must exist either way.
            frozen_epoch = epoch
            at_stop = float(metrics["desired_ratio"])
            health["desired_ratio_at_stop"] = {
                "valid_ratio": round(float(metrics.get("valid_ratio", 0.0)), 4),
                "unique_ratio": round(float(metrics.get("unique_ratio", 0.0)), 4),
            }
            _save_atomically(self.getModel(), self.desired_ratio_at_stop_path)
            logger.info(
                "desired_ratio never exhausted its patience window: its checkpoint holds "
                "the best value seen (%.4f @ epoch %d); at-stop is the final policy",
                best["desired_ratio"],
                last["desired_ratio"],
            )
        self.monitor.close()
        return {
            "epochs_run": epoch,
            # Named for what it is: the best value seen *before the freeze*, which is the
            # one the checkpoint holds. `peak_desired_ratio` is the best seen at any point,
            # so a post-freeze improvement shows up in the report instead of only in
            # rl_fit.tsv, where it would contradict the summary.
            "best_desired_ratio_before_freeze": best["desired_ratio"],
            "best_desired_ratio_epoch": last["desired_ratio"],
            "peak_desired_ratio_any_epoch": peak_desired_ratio,
            "desired_ratio_frozen": frozen,
            "desired_ratio_freeze_epoch": frozen_epoch,
            "desired_ratio_at_stop": at_stop,
            "best_avg_gmean": best["avg_gmean"],
            "best_avg_gmean_epoch": last["avg_gmean"],
            "checkpoint_population": health,
        }
