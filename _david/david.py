from _david import receptor_similar

from qsprpred.data import MoleculeTable
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.models.scikit_learn import SklearnModel


import drugex.training.generators.sequence_rnn
from drugex.training import explorers
from drugex.training.environment import DrugExEnvironment
from drugex.training.monitors import FileMonitor
from drugex.training.generators import SequenceRNN

from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.qsprpred import QSPRPredScorer

# PARETO DISTANCE: like comparing AI model cost vs acurrecy
# NSGA-II Crowding Distance: Model receives minimal reinforcement for generating another identical/similar molecule
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.explorers import SequenceExplorer

from drugex.data.corpus import vocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.corpus.corpus import SequenceCorpus

from drugex.data.datasets import SmilesDataSet

from drugex.data.processing import RandomTrainTestSplitter
from drugex.data.processing import Standardization
from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter

from drugex.logs import logger
logger.setLevel("ERROR")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from torch.utils.data import DataLoader

from typing import Any

import os
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scaffviz.clustering.manifold import TSNE
from scaffviz.depiction.plot import Plot


GPUS = [1] # we will use only one GPU with ID=0, but if you have more, you can list more GPU IDs here

class david:

    #https://pubchem.ncbi.nlm.nih.gov/compound/Epigallocatechin-Gallate
    MOLECULE = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

    N_PROCESSES = 12 # number of CPU cores to use
    CHUNK_SIZE = 1000 # largest chunk per CPU core (regulates RAM usage)

    EPOCHS = 200
    LOSS_TOLERANCE = 0.02
    EPSILON = 0.1
    BATCH_SIZE = 32

    def __init__(self):

        ROOT_PATH : Path = Path(__file__).resolve().parent.parent
        BASE_OUTPUT_DIR: Path = ROOT_PATH / "_david" / "outputs"

        #imput data
        MODELS_PR_PATH: Path = ROOT_PATH / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/"

        #output
        RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR: Path = BASE_OUTPUT_DIR / RUN_ID

        TRANSFER_DIR: Path = OUTPUT_DIR / "transfer_learning"
        RL_DIR : Path = OUTPUT_DIR / "reinforcement_learning"
        DATA_DIR : Path = TRANSFER_DIR / "data"

        #dir creation
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
        self.RL_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


        self.VOC = VocSmiles.fromFile((MODELS_PR_PATH / "Papyrus05.5_smiles_rnn_PT.vocab"))

        self.PRETRAINED = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS) # is_lstm = MORE PARAMETERS
        self.PRETRAINED.loadStatesFromFile(os.path.join(self.MODELS_PR_PATH, "Papyrus05.5_smiles_rnn_PT.pkg"))

    def __TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(self, molecule_smile : str) -> list[Any]:

        pipline = receptor_similar.MoleculeBioactivityPipeline()
        result: DataFrame = pipline.run(molecule_smile).papyrus_curated
        smiles: Series = result["SMILES"]

        smiles_paralel: list[Any] = Standardization(n_proc=self.N_PROCESSES, chunk_size=self.CHUNK_SIZE) \
                                                   .apply(smiles)

        pd.Series(smiles_paralel, name="SMILES") \
          .to_csv(os.path.join(self.MODEL_DIR, "training_ligands.tsv"), sep="\t", index=False)


        return smiles_paralel

    def __TRANSFER_LEARNING_tokenization(self, smiles_train : list[any] ) -> tuple[SmilesDataSet, CorpusEncoder]:

        encoder = CorpusEncoder(
            SequenceCorpus, # The corpus CLASS (NOT OBJECT, just just as a constructor)
            {               # implements how each SMILES string is divided into words by the vocabulary
                "vocabulary": self.VOC,
                "update_voc": False,
                "throw": True # compounds containing unknown tokens are thrown out of the resulting data set

            },
            n_proc=self.N_PROCESSES,
            chunk_size=self.CHUNK_SIZE
        )

        data_collector = SmilesDataSet(self.DATA_DIR / "ligand_corpus.tsv", rewrite=True)
        
        encoder.apply(smiles_train, collector=data_collector)

        return data_collector, encoder

    def __TRANSFER_LEARNING_make_test_dataset(self, data_collector : SmilesDataSet) -> tuple[DataLoader, DataLoader]:

        splitter = RandomTrainTestSplitter(0.1, 1e4)
        train, test = splitter(data_collector.getData())
        

        # tbh is just a beckup
        pd.DataFrame(train, columns=data_collector.getColumns()).to_csv(
            self.DATA_DIR / "ligand_train.tsv", header=True, index=False, sep="\t"
        )
        pd.DataFrame(test, columns=data_collector.getColumns()).to_csv(
            self.DATA_DIR / "ligand_test.tsv", header=True, index=False, sep="\t"
        )    

        #get data path (from files as it would be useful for later)
        data_set_train = SmilesDataSet(self.DATA_DIR / "ligand_train.tsv", voc=self.VOC)
        train_loader = data_set_train.asDataLoader(batch_size=self.BATCH_SIZE)

        data_set_valid = SmilesDataSet(self.DATA_DIR / "ligand_train.tsv", voc=self.VOC)
        valid_loader = data_set_train.asDataLoader(batch_size=self.BATCH_SIZE)

        return train_loader, valid_loader

    def __TRANSFER_LEARNING_transfer_learning(self, train_loader : DataLoader, valid_loader : DataLoader, destination_folder : str) -> SequenceRNN:

        ft_path = os.path.join(self.MODEL_DIR, destination_folder)
        finetuned = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        finetuned.loadStatesFromFile(os.path.join(self.MODELS_PR_PATH, "Papyrus05.5_smiles_rnn_PT.pkg"))

        reset_directory : bool =True
        monitor = FileMonitor(ft_path, save_smiles=True, reset_directory=reset_directory)
        finetuned.fit(train_loader, valid_loader, epochs=self.EPOCHS, monitor=monitor, loss_tolerance=self.LOSS_TOLERANCE)

        self.VOC.toFile(os.path.join(self.MODEL_DIR, "finetuned.VOCab"))

        return finetuned

    def __TRANSFER_LEARNING_vizualize(self, finetuned : SequenceRNN, destination_folder) -> DataFrame:
        from . import smilesToGrid #from c compiled utiles
        
        df_info = pd.read_csv(f"{self.MODEL_DIR}/{destination_folder}_fit.tsv", sep="\t")
        ax = df_info[["loss_train", "loss_valid", "valid_ratio"]].plot.line(logy=True)
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.MODEL_DIR, "training_loss.png"), bbox_inches="tight")
        plt.close(fig)

        generated_sample: DataFrame = finetuned.generate(num_samples=100)
        generated_sample.to_csv(os.path.join(self.MODEL_DIR, "generated_molecules.tsv"), sep="\t", index=False)
        try:
            smilesToGrid(generated_sample.SMILES, molsPerRow=5, n_rows=5)
        except Exception:
            pass
        return generated_sample

    def TRANSFER_LEARNING(self, my_molecule : str = MOLECULE) -> DataFrame:
        
        smiles_train = self.__TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(my_molecule)

        data_collector, encoder,  = self.__TRANSFER_LEARNING_tokenization(smiles_train)
        
        train_loader, valid_loader = self.__TRANSFER_LEARNING_make_test_dataset(data_collector)

        finetuned: SequenceRNN = self.__TRANSFER_LEARNING_transfer_learning(train_loader, valid_loader, "david_finetuned")
        generated_sample = self.__TRANSFER_LEARNING_vizualize(finetuned, "david_finetuned")
        return generated_sample

    def TRANSFER_LEARNING_vizualization(self, scores):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        scores.QSPRpred_A2AR_RandomForestClassifier.hist(ax=axes[0])
        axes[0].set_title("QSPRpred_A2AR_RandomForestClassifier")
        scores.SA.hist(ax=axes[1])
        axes[1].set_title("SA")
        fig.savefig(os.path.join(self.MODEL_DIR, "score_distributions.png"), bbox_inches="tight")
        plt.close(fig)

    def TRANSFER_LEARNING_SCORE_SHOWCASE(self, environment, my_molecule : str="C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"):
        generated_sample = self.TRANSFER_LEARNING(my_molecule)
        scores = environment.getScores(generated_sample.SMILES.to_list())
        
        # Save scored molecules
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(os.path.join(self.MODEL_DIR, "tl_generated_candidates_scored.tsv"), sep="\t", index=False)

        self.TRANSFER_LEARNING_vizualization(scores)

    def RAINFORCEMENT_LEARNING_vizualization(self, scores, generated, qsprpred_scorer):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        scores[qsprpred_scorer.getKey()].hist(ax=axes[0])
        axes[0].set_title(qsprpred_scorer.getKey())
        scores.SA.hist(ax=axes[1])
        axes[1].set_title("SA")
        fig.savefig(os.path.join(self.MODEL_DIR_RL, "rl_score_distributions.png"), bbox_inches="tight")
        plt.close(fig)

        dataset = MoleculeTable("david_agent", df=generated)
        dataset.addProperty(qsprpred_scorer.getKey(), scores[qsprpred_scorer.getKey()].values)
        dataset.addDescriptors(
            [MorganFP(radius=3, nBits=2048)]
        )

        plt_manifold = Plot(TSNE())
        fig_manifold = plt_manifold.plot(
            dataset,
            color_by=qsprpred_scorer.getKey(),
            interactive=False,
            color_continuous_scale="rdylgn"
        )
        if fig_manifold is not None:
            fig_manifold.write_html(os.path.join(self.MODEL_DIR_RL, "rl_chemical_space.html"))

    def COMPARE_vizualization(self, df, generated):
        df_joined : DataFrame = pd.concat(
            [
                pd.DataFrame(
                    {"SMILES" : df.SMILES.to_list(), "Group" : "Set"}
                ),
                pd.DataFrame(
                    {"SMILES" : generated.SMILES, "Group" : "Generated"}
                )
            ]
        )

        dataset = MoleculeTable(name="david_joined", df = df_joined)
        dataset.addDescriptors([MorganFP(radius=3, nBits=2048)])

        plt_manifold = Plot(TSNE())
        fig_manifold = plt_manifold.plot(dataset, recalculate=False, color_by="Group", interactive=False)
        if fig_manifold is not None:
            fig_manifold.write_html(os.path.join(self.MODEL_DIR_RL, "chemical_space_comparison.html"))

    def RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(self, environment : DrugExEnvironment):

        agent = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        agent.loadStatesFromFile(f"{self.MODEL_DIR_RL}/david_agent.pkg")

        generated_sample = agent.generate(num_samples=100)

        scores = environment.getScores(generated_sample.SMILES)

        # Save scored molecules
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(os.path.join(self.MODEL_DIR_RL, "rl_generated_candidates_scored.tsv"), sep="\t", index=False)

        qsprpred_scorer = [s for s in environment.scorers if isinstance(s, QSPRPredScorer)][0]
        self.RAINFORCEMENT_LEARNING_vizualization(scores, generated_sample, qsprpred_scorer)

    def RAINFORCEMENT_LEARNING(self):


        qsar_base = os.path.join(Path(__file__).resolve().parent.parent, "tutorial/data/models/qsar")
        if not os.path.exists(qsar_base):
            qsar_base = os.path.join(Path(__file__).resolve().parent.parent, "data/models/qsar")

        predictor = SklearnModel(
            name="A2AR_RandomForestClassifier",
            base_dir=qsar_base
        )

        #TODO: how about more functions?

        qsprpred_scorer = QSPRPredScorer(predictor)


        sascore = Property("SA")
        sascore.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))
        qsprpred_scorer.setModifier(ClippedScore(lower_x=0.2, upper_x=0.8))


        scorers: list[Property | QSPRPredScorer] = [
            qsprpred_scorer,
            sascore
        ]
        thresholds = [
            0.5,
            0.1
        ]

        environment = DrugExEnvironment(scorers, thresholds, reward_scheme=ParetoCrowdingDistance())


        finetuned = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        finetuned_path = os.path.join(self.MODEL_DIR, "david_finetuned.pkg")
        if not os.path.exists(finetuned_path):
            finetuned_path = os.path.join(Path(__file__).resolve().parent.parent, "tutorial/data/models/finetuned/smiles-rnn/david_finetuned.pkg")
        finetuned.loadStatesFromFile(finetuned_path)

        explorer = SequenceExplorer(
            agent = finetuned, #TODO pretrained
            env = environment,
            mutate = self.pretrained, # network introducing "random mutations" to the generated structures (rate determined by epsilon)
            epsilon = self.EPSILON,
            use_gpus = GPUS
        )

        MODEL_DIR_RL = self.MODEL_DIR_RL

        monitor = FileMonitor(os.path.join(MODEL_DIR_RL, "david_agent"), save_smiles=True, reset_directory=True)
        explorer.fit(monitor=monitor, epochs=100)

        df_info = pd.read_csv(f"{MODEL_DIR_RL}/david_agent_fit.tsv", sep="\t")
        df_info.head()
        ax = df_info[["loss_train", "valid_ratio","unique_ratio", "desired_ratio"]].plot.line()
        fig = ax.get_figure()
        fig.savefig(os.path.join(MODEL_DIR_RL, "rl_training_curves.png"), bbox_inches="tight")
        plt.close(fig)

        self.RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(environment)

if __name__ == "__main__":
    tmp = david()
    tmp.TRANSFER_LEARNING()
    tmp.RAINFORCEMENT_LEARNING()
