import os
import os.path
from pathlib import Path
from typing import Callable, Literal, Optional, Union

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
