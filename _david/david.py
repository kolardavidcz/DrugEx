from __future__ import annotations

# --- Standard Library ---
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Numerical Data, Cheminformatics & PyTorch ---
import matplotlib
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

# --- Plotting, QSPRPred & Chemical Space Visualization ---
import matplotlib.pyplot as plt
from qsprpred import data as qspr_data
from qsprpred.data.descriptors import fingerprints as qspr_fps
from qsprpred.models import scikit_learn as qspr_models
from scaffviz.clustering import manifold
from scaffviz.depiction import plot as scaff_plot

# --- DrugEx Core & Submodule Namespaces ---
import drugex
from drugex import data, logs, molecules, parallel, training, utils
from drugex.data import datasets, fragments, processing
from drugex.data.corpus import corpus, vocabulary
from drugex.logs import logger
from drugex.molecules import converters, mol, suppliers
from drugex.molecules.converters import default as default_converters
from drugex.molecules.converters import standardizers
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.scorers import (
    conformer_generators,
    modifiers,
    oe_color_smarts,
    properties,
    protonation,
    qsprpred,
    sascorer,
    similarity,
    smiles as smiles_scorers,
)

# --- DrugEx Direct Class Imports (convenience: namespace.Class or bare Class) ---
from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.datasets import GraphFragDataSet, SmilesDataSet, SmilesFragDataSet
from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter, Standardization
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer, FragSequenceExplorer, SequenceExplorer
from drugex.training.generators import GraphTransformer, SequenceRNN, SequenceTransformer
from drugex.training.monitors import FileMonitor, NullMonitor
from drugex.training.rewards import ParetoCrowdingDistance, ParetoRewardScheme, ParetoTanimotoDistance, WeightedSum
from drugex.training.scorers.modifiers import ClippedScore, ScoreModifier, SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.similarity import FraggleSimilarity, TverskyFingerprintSimilarity, TverskyGraphSimilarity

# --- Local Project Modules ---
from _david import receptor_similar

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

GPUS = [1]  # we will use only one GPU with ID=1, but if you have more, you can list more GPU IDs here
SAVE_PROOF = False


class MY_ERROR(RuntimeError):
    def __init__(self, message="error somewehre in david.py"):
        super().__init__(message)

class ALL_METHODS():

    N_PROCESSES = 12  # number of CPU cores to use
    CHUNK_SIZE = 1000  # largest chunk per CPU core (regulates RAM usage)

    EPOCHS = 20
    MIN_BATCH_SIZE = 10 #is couted based on accesable molecules + number of desired epochs

    LOSS_TOLERANCE = 0.02#
    EPSILON = 0.1


    def __init__(self):

        #essecnail path
        self.ROOT_PATH: Path = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "outputs"
        
        #imput
        self.QSAR_DIR : Path = self.ROOT_PATH / "tutorial/data/models/qsar"

        #output
        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_DIR: Path = self.BASE_OUTPUT_DIR / self.RUN_ID
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def molecules_save_image(generated_sample : DataFrame, out_dir : Path) -> None:
        mols = [Chem.MolFromSmiles(s) for s in generated_sample["SMILES"][:25] if Chem.MolFromSmiles(s)]
        if not mols:
            raise 
        
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[250, 250])
        img.save(str(out_dir / "generated_molecules.png"))

    
    @staticmethod
    @contextmanager
    def MY_save_plot(output_path: Path | str, **subplots_kwargs: Any) -> Generator[plt.Axes, None, None]:
        """Context manager that automatically saves and closes a matplotlib figure.

        Parameters
        ----------
        output_path : Path or str
            Destination path for the PNG file.
        **subplots_kwargs : Any
            Arguments forwarded to plt.subplots (e.g. figsize=(10, 4)).
        """
        fig, ax = plt.subplots(**subplots_kwargs)
        try:
            yield ax
        finally:
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def vizualize(finetuned : generators.SequenceRNN, model_prefix : Path, out_dir : Path) -> DataFrame:

        performance_for_epoch: DataFrame = pd.read_csv(f"{model_prefix}_fit.tsv", sep="\t")
        
        with ALL_METHODS.MY_save_plot(out_dir / "training_loss.png") as ax:
            performance_for_epoch[["loss_train", "loss_valid", "valid_ratio"]].plot.line(logy=True, ax=ax)

        generated_sample: DataFrame = finetuned.generate(num_samples=100)
        generated_sample.to_csv(out_dir / "generated_molecules.tsv", sep="\t", index=False)

        ALL_METHODS.molecules_save_image(generated_sample, out_dir)

        return generated_sample

class MY_TRANSFER_LEARNING(ALL_METHODS):
    # https://pubchem.ncbi.nlm.nih.gov/compound/Epigallocatechin-Gallate


    def __init__(self) -> None:
        
        # output
        self.MODEL: Path = self.OUTPUT_DIR / "transfer_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        # input data
        self.MODEL_PRETRAINED: Path = self.ROOT_PATH / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"

        self.VOC = vocabulary.VocSmiles.fromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.vocab")
        
        self.PRETRAINED = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)  # is_lstm = MORE PARAMETERS
        self.PRETRAINED.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.pkg")

    def __receptor_similar_molecules_get_SMILES(self, molecule_smile : str) -> tuple[list[Any], int]:

        #TODO Physicochemical profile + 2D Pharmocophore/Morgan Tanimato similarity ranking

        pipline = receptor_similar.MoleculeBioactivityPipeline()

        # Candidate Rank Range	Property Similarity Score	Chemical Profile in Papyrus
        # Top 1 – 20	        0.704→0.4000	    High resemblance: polyphenols, flavones, catechins, rich in −OH−OH groups
        # Rank 21 – 100	        0.400→0.3150	    Moderate resemblance: natural-product-like, polycyclic, polar
        # Rank 101 – 200	    0.315→0.2920	    Elbow inflection point: transitions toward synthetic scaffolds
        # Rank 201 – 500	    0.292→0.2570	    Chemical Noise: synthetic heteroaromatics, sulfonamides, basic amines
        # Rank 501 – 1,070	    0.257→0.2150	    Completely divergent chemistry (only share target binding)
        #! add noise as a decoy
        max_molecules=200
        result: DataFrame = pipline.run(molecule_smile, max_molecules).molecules

        if (self.MIN_BATCH_SIZE * self.EPOCHS < result.sum()):

            if(result.empty):
                raise MY_ERROR(f"No molecules found: result.sum ({result.sum()})")

            raise MY_ERROR(f"No eought result molecules to train: MIN_BATCH_SIZE({self.MIN_BATCH_SIZE}) * EPOCHS({self.EPOCHS}) < result({result.sum()})")
        
        BATCH_SIZE = int( result.sum() / self.EPOCHS )
        result = result[0:BATCH_SIZE+1]


        smiles: Series = result["SMILES"]

        smiles_paralel = processing.Standardization(n_proc=self.N_PROCESSES, chunk_size=self.CHUNK_SIZE) \
                                   .apply(smiles)
        if smiles_paralel is None:
            raise MY_ERROR("Standardization failed to process molecules.")

        if SAVE_PROOF:
            pd.Series(smiles_paralel, name="SMILES").to_csv(
                self.MODEL / "training_ligands.tsv", sep="\t", index=False
            )

        return smiles_paralel, BATCH_SIZE

    def __tokenization(self, smiles_train : list[Any] ) -> tuple[datasets.SmilesDataSet, processing.CorpusEncoder]:

        encoder = processing.CorpusEncoder(
            corpus.SequenceCorpus,  # The corpus CLASS (NOT OBJECT, just just as a constructor)
            {                       # implements how each SMILES string is divided into words by the vocabulary
                "vocabulary": self.VOC,
                "update_voc": False,
                "throw": True # compounds containing unknown tokens are thrown out of the resulting data set

            },
            n_proc=self.N_PROCESSES,
            chunk_size=self.CHUNK_SIZE
        )
        data_collector = datasets.SmilesDataSet(self.OUTPUT_DIR / "ligand_corpus.tsv", rewrite=True)
        
        encoder.apply(smiles_train, collector=data_collector)

        return data_collector, encoder

    def __make_test_dataset(self, data_collector : datasets.SmilesDataSet, BATCH_SIZE : int) -> tuple[DataLoader, DataLoader]:

        if SAVE_PROOF:
            splitter = processing.RandomTrainTestSplitter(0.1, 10000)
            train, test  = splitter(data_collector.getData())
            
            pd.DataFrame(train, columns=data_collector.getColumns()).to_csv(
                self.OUTPUT_DIR / "ligand_train.tsv", header=True, index=False, sep="\t"
            )
            pd.DataFrame(test, columns=data_collector.getColumns()).to_csv(
                self.OUTPUT_DIR / "ligand_test.tsv", header=True, index=False, sep="\t"
            )

            self.VOC.toFile(self.OUTPUT_DIR / "pretrained.vocab")
            
            train_loader = datasets.SmilesDataSet.dataToLoader(train, batch_size=BATCH_SIZE, vocabulary=self.VOC)
            valid_loader = datasets.SmilesDataSet.dataToLoader(test, batch_size=BATCH_SIZE, vocabulary=self.VOC)
            return train_loader, valid_loader
        ########

        loaders = data_collector.asDataLoader(
            batch_size=BATCH_SIZE,
            splitter=processing.RandomTrainTestSplitter(0.1, 10000),
        )
        if (not isinstance(loaders, (list, tuple))) or len(loaders) != 2:
                raise MY_ERROR("Expected exactly two DataLoaders \"(list, tuple)\" (train, valid) from splitter.")

        return loaders[0], loaders[1]
            

    def __transfer_learning(self, train_loader : DataLoader, valid_loader : DataLoader, model_prefix : Path) -> generators.SequenceRNN:

        finetuned = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        finetuned.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.pkg")

        reset_directory : bool =True
                                                    #save_smiles=True MAKES SAVE SAMPLE SMILES
        monitor = monitors.FileMonitor(model_prefix, save_smiles=True, reset_directory=reset_directory)
                            
        #   _______ _____            _____ _   _ _____ _   _  _____ 
        #  |__   __|  __ \     /\   |_   _| \ | |_   _| \ | |/ ____|
        #     | |  | |__) |   /  \    | | |  \| | | | |  \| | |  __ 
        #     | |  |  _  /   / /\ \   | | | . ` | | | | . ` | | |_ |
        #     | |  | | \ \  / ____ \ _| |_| |\  |_| |_| |\  | |__| |
        #     |_|  |_|  \_\/_/    \_\_____|_| \_|_____|_| \_|\_____|         
        #                           
        finetuned.fit(train_loader, valid_loader, epochs=self.EPOCHS, monitor=monitor, loss_tolerance=self.LOSS_TOLERANCE)

        if SAVE_PROOF:
            self.VOC.toFile(self.MODEL / "finetuned.vocab")

        return finetuned

    def TRANSFER_LEARNING(self, my_molecule) -> tuple[generators.SequenceRNN, DataFrame]:
        
        smiles_train, BATCH_SIZE = self.__receptor_similar_molecules_get_SMILES(my_molecule)

        data_collector, encoder = self.__tokenization(smiles_train)
        
        train_loader, valid_loader = self.__make_test_dataset(data_collector, BATCH_SIZE)

        model_prefix = self.MODEL / "david_finetuned" #wil be added ""_SOMETING.tsv" to the end
        finetuned: generators.SequenceRNN = self.__transfer_learning(train_loader, valid_loader, model_prefix)

        generated_sample = ALL_METHODS.vizualize(finetuned, model_prefix, self.MODEL)

        return finetuned, generated_sample


class RAINFORCMENT_LEARNING(ALL_METHODS):

    def __init__(self, finetuned : SequenceRNN):
        self.MODEL: Path = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        
        self.PRETRAINED: Path = self.ROOT_PATH / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        self.VOC = vocabulary.VocSmiles.fromFile(self.PRETRAINED / "Papyrus05.5_smiles_rnn_PT.vocab")

        self.MUTATE : SequenceRNN = finetuned

    def RAINFORCEMENT_LEARNING(self):

        scorers = [
            Property( prop="QED",
                modifier=SmoothClippedScore(lower_x=0.4, upper_x=0.8)),

            Property( prop="SAScore", 
                modifier=SmoothClippedScore(lower_x=4.5, upper_x=2.5))
        ]

        reward_scheme = rewards.ParetoCrowdingDistance()
        reward_scheme_v2 = rewards.ParetoTanimotoDistance()

        # # 5. Reward Scheme
        # reward_scheme = ParetoCrowdingDistance()
        # # 6. Environment
        # env = DrugExEnvironment(
        #     scorers=scorers,
        #     thresholds=[0.5, 0.5, 0.5],
        #     reward_scheme=reward_scheme
        # )
        # # 7. Explorer (RL Orchestrator)
        # explorer = SequenceExplorer(
        #     agent=agent,
        #     mutate=mutate,
        #     env=env,
        #     epsilon=0.20,
        #     beta=0.0,
        #     n_samples=1000,
        #     batch_size=128
        # )
        # # 8. Monitor & Fit
        # monitor = FileMonitor("rl_outputs/run_01", save_smiles=True)
        # explorer.fit(monitor=monitor, epochs=100, patience=20, criteria='desired_ratio')


if __name__ == "__main__":
    my_molecule : str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
    TL = MY_TRANSFER_LEARNING()
    finetuned, generated_sample = TL.TRANSFER_LEARNING(my_molecule)

    # ! fine tooning could be only molecules = 50 to 1000, with pIC50 > 6.0
    RL = RAINFORCMENT_LEARNING(finetuned)
    RL.RAINFORCEMENT_LEARNING(my_molecule)



