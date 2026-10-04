import copy
import os
import os.path
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

# [PACKAGE: pandas] WHY: Performance metrics and SMILES table serialization. WHAT: Data manipulation library.
import pandas as pd
# [PACKAGE: torch] WHY: Model weights state dictionary checkpointing. WHAT: PyTorch deep learning framework.
import torch

from drugex.logs import logger
from drugex.training.interfaces import TrainingMonitor, Model


class NullMonitor(TrainingMonitor):
    """No-op training monitor discarding metrics and checkpoints during silent runs."""

    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        return 'all'

    def saveModel(self, model, identifier=None):
        pass

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        pass

    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        pass

    def endStep(self, step, epoch):
        pass

    def close(self):
        pass

    def getModel(self):
        pass

class FileMonitor(TrainingMonitor):
    """File-system training monitor persisting logs, tabular metrics, generated SMILES, and model checkpoints."""

    def __init__(
            self,
            path: Union[str, os.PathLike, Path],
            save_smiles: bool = False,
            save_model_option: Literal['best', 'all', 'improvement'] = 'best',
            reset_directory: bool = False,
            on_model_update: Optional[Callable[[Model], None]] = None
    ):
        """Initialize file output streams and directories for training logs, metrics TSV, and model checkpoints."""
        
        self.path: Path = Path(path)
        self.directory: Path = self.path.parent
        self.directory.mkdir(parents=True, exist_ok=True)
        if reset_directory:
            for file in self.directory.glob(f"{self.path.name}*"):
                logger.warning(f"Removing {file.name} from {self.directory}")
                file.unlink(missing_ok=True)
        self.outLog = open(self.path.parent / f"{self.path.name}_fit.log", 'w', encoding='utf-8')
        self.outDF = self.path.parent / f"{self.path.name}_fit.tsv"
        self.outSmiles = self.path.parent / f"{self.path.name}_smiles.tsv" if save_smiles else None
        self.outSmilesHeaderDone = False
        self.currentState = None
        self.saveModelOption = save_model_option
        self.onModelUpdate = on_model_update

    def saveModel(self, model, identifier=None):
        """Serialize and save model state dictionary to package file on disk."""
        self.currentState = model.getModel() 
        suffix = '_' + str(identifier) if identifier else ''
        pkg_path = self.path.parent / f"{self.path.name}{suffix}.pkg"
        torch.save(self.currentState, pkg_path)

    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs):
        """Record step/epoch progress counters and training loss to log file."""
        if self.onModelUpdate:
            self.onModelUpdate(model)
        
        txt = f"Epoch {current_epoch if current_epoch is not None else '--'}/"
        txt += f"{total_epochs if total_epochs is not None else '--'}," 
        txt += f"Step {current_step if current_step is not None else '--'}/"
        txt += f"{total_steps if total_steps is not None else '--'}\n"
        self.outLog.write(txt)
        if loss:
            self.outLog.write(f"Current training loss: {loss:.4f} \n")
        self.outLog.flush()

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        """Append epoch evaluation metrics and generated molecules to output tables."""

        df = pd.DataFrame(performance_dict, index=[0])
        self.saveEpochData(df)

        # Save smiles and their indivudual scores if requested
        if self.outSmiles and df_smiles is not None:
            self.saveMolecules(df_smiles)

    def saveEpochData(self, df):
        self.appendTableToFile(df, self.outDF)

    def saveMolecules(self, df):
        if self.outSmiles:
            self.appendTableToFile(df, self.outSmiles)

    @staticmethod
    def appendTableToFile(df, outfile):
        header_written = os.path.isfile(outfile)
        open_mode = 'a' if header_written else 'w'
        df.round(decimals=3).to_csv(
            outfile,
            sep='\t',
            index=False,
            header=not header_written,
            mode=open_mode,
            encoding='utf-8',
            na_rep='NA'
        )

    def endStep(self, step, epoch):
        super().endStep(step, epoch)
        self.outLog.flush()

    def close(self):
        super().close()
        self.outLog.close()
   
    def getModel(self):
        return self.currentState

    def getSaveModelOption(self):
        return self.saveModelOption


class PeakRecoveryMonitor(FileMonitor):
    """Training monitor with rolling desirability tracking, checkpoint persistence, and automatic peak model recovery.

    Tracks multi-objective desirability ('desired_ratio') across epochs, persists model weights
    as '{path}_peak.pkg' upon achieving a new best value, and automatically restores peak historical
    weights onto the agent neural network in close() or via restore_peak_model().
    """

    def __init__(
        self,
        path: Optional[Union[str, os.PathLike, Path]] = None,
        output_prefix: Optional[Union[str, os.PathLike, Path]] = None,
        save_smiles: bool = True,
        save_model_option: Literal['best', 'all', 'improvement'] = 'best',
        reset_directory: bool = False,
        criteria: str = 'desired_ratio',
        patience: int = 10,
        min_epochs: int = 5,
        on_model_update: Optional[Callable[[Model], None]] = None,
    ) -> None:
        """Initialize PeakRecoveryMonitor with file path, criteria, and early stopping limits."""
        resolved_path = path if path is not None else (output_prefix if output_prefix is not None else "model")
        super().__init__(
            path=resolved_path,
            save_smiles=save_smiles,
            save_model_option=save_model_option,
            reset_directory=reset_directory,
            on_model_update=on_model_update,
        )
        self.output_prefix: Path = self.path
        self.criteria: str = criteria
        self.patience: int = int(patience)
        self.min_epochs: int = int(min_epochs)
        self.peak_path: Path = self.path.parent / f"{self.path.name}_peak.pkg"
        self.peak_desirability: float = -1.0
        self.best_metric_value: float = -1.0
        self.peak_epoch: int = -1
        self.best_epoch: int = -1
        self.epochs_without_improvement: int = 0
        self.early_stopped: bool = False
        self.restored: bool = False
        self.history: List[Dict[str, Any]] = []
        self.peak_state_dict: Optional[Dict[str, Any]] = None
        self.target_model: Any = None

    def saveModel(self, model: Any, identifier: Any = None) -> None:
        """Capture target explorer/model reference and delegate serialization to FileMonitor."""
        self.target_model = model
        super().saveModel(model, identifier=identifier)

    def step(self, epoch: int, desirability: float, model_state: Any) -> bool:
        """Record epoch desirability and persist peak checkpoint on improvement.

        Parameters:
            epoch: Current training epoch index.
            desirability: Multi-objective desirability metric value (e.g. desired_ratio).
            model_state: Model state dictionary or object exposing state_dict().

        Returns:
            bool: True if training should continue, False if early stopping triggered.
        """
        self.history.append({"epoch": epoch, "desirability": float(desirability)})
        des_val = float(desirability)

        if des_val > self.peak_desirability:
            self.peak_desirability = des_val
            self.best_metric_value = des_val
            self.peak_epoch = epoch
            self.best_epoch = epoch
            self.epochs_without_improvement = 0

            # Extract state dictionary
            if isinstance(model_state, dict):
                self.peak_state_dict = copy.deepcopy(model_state)
            elif hasattr(model_state, "state_dict"):
                self.peak_state_dict = copy.deepcopy(model_state.state_dict())
            elif hasattr(model_state, "getModel"):
                self.peak_state_dict = copy.deepcopy(model_state.getModel())
            else:
                self.peak_state_dict = copy.deepcopy(model_state)

            # [PACKAGE: torch] WHY: Serialize PyTorch state dictionary to disk. WHAT: Deep learning tensor serialization. HOW: torch.save.
            torch.save(self.peak_state_dict, self.peak_path)
            logger.info(
                f"[PeakRecoveryMonitor] New peak checkpoint at Epoch {epoch} "
                f"({self.criteria} = {des_val:.4f}). Saved to {self.peak_path.name}"
            )
        else:
            self.epochs_without_improvement += 1

        if epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            self.early_stopped = True
            logger.info(
                f"[PeakRecoveryMonitor] Early stopping triggered at Epoch {epoch} "
                f"({self.epochs_without_improvement} epochs without improvement >= patience={self.patience})."
            )
            return False
        return True

    def savePerformanceInfo(self, performance_dict: Dict[str, Any], df_smiles: Any = None) -> None:
        """Track rolling performance metrics and store peak model checkpoint."""
        current_val = float(performance_dict.get(self.criteria, performance_dict.get("desired_ratio", 0.0)))
        epoch = int(performance_dict.get("Epoch", performance_dict.get("epoch", len(self.history) + 1)))

        if self.target_model is not None:
            agent_net = getattr(self.target_model, "agent", self.target_model)
            state = getattr(agent_net, "state_dict", getattr(self.target_model, "getModel", lambda: None))()
            if state is not None:
                self.step(epoch=epoch, desirability=current_val, model_state=state)
        else:
            self.history.append({"epoch": epoch, "desirability": current_val})
            if current_val > self.peak_desirability:
                self.peak_desirability = current_val
                self.best_metric_value = current_val
                self.peak_epoch = epoch
                self.best_epoch = epoch

        super().savePerformanceInfo(performance_dict, df_smiles=df_smiles)

    def restore_peak_model(self, agent: Any = None) -> Any:
        """Restore peak model weights onto agent neural network or return peak state dict."""
        target = agent or (getattr(self.target_model, "agent", self.target_model) if self.target_model else None)

        if target is not None:
            if self.peak_state_dict is not None and hasattr(target, "load_state_dict"):
                target.load_state_dict(self.peak_state_dict)
                self.restored = True
                logger.info(
                    f"[PeakRecoveryMonitor] Restored in-memory peak weights from Epoch {self.peak_epoch} "
                    f"({self.criteria} = {self.peak_desirability:.4f}) onto agent."
                )
                return self.peak_state_dict
            elif self.peak_path.exists():
                # [PACKAGE: torch] WHY: Deserialize PyTorch weights from peak checkpoint file. WHAT: Deep learning framework. HOW: torch.load.
                saved_state = torch.load(self.peak_path, weights_only=False)
                if hasattr(target, "load_state_dict"):
                    target.load_state_dict(saved_state)
                elif hasattr(target, "loadStatesFromFile"):
                    target.loadStatesFromFile(self.peak_path)
                self.restored = True
                logger.info(f"[PeakRecoveryMonitor] Restored peak weights from disk ({self.peak_path}) onto agent.")
                return saved_state

        if self.peak_path.exists():
            # [PACKAGE: torch] WHY: Deserialize PyTorch weights from peak checkpoint file. WHAT: Deep learning framework. HOW: torch.load.
            self.restored = True
            return torch.load(self.peak_path, weights_only=False)
        if self.peak_state_dict is not None:
            self.restored = True
            return self.peak_state_dict

        raise FileNotFoundError(f"Peak checkpoint not found at {self.peak_path}")

    def close(self) -> None:
        """Auto-restore peak model weights on training completion or early stopping."""
        if self.target_model is not None:
            try:
                self.restore_peak_model()
            except Exception as e:
                logger.warning(f"[PeakRecoveryMonitor] Failed to auto-restore peak model in close(): {e}")
        super().close()
