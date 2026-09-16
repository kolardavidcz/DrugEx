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
from torch.utils.data import DataLoader

matplotlib.use("Agg")

# --- Plotting, QSPRPred & Chemical Space Visualization ---
import matplotlib.pyplot as plt
from qsprpred import data as qspr_data
from qsprpred.data.descriptors import fingerprints as qspr_fps
from qsprpred.models import scikit_learn as qspr_models
from scaffviz.clustering import manifold
from scaffviz.depiction import plot as scaff_plot

# --- Project Modules (Local & DrugEx) ---
from _david import receptor_similar
from drugex.data import datasets, processing
from drugex.data.corpus import corpus, vocabulary
from drugex.logs import logger
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.scorers import modifiers, properties, qsprpred, similarity  # noqa: F401

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

GPUS = [1]  # we will use only one GPU with ID=1, but if you have more, you can list more GPU IDs here
SAVE_PROOF = False


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


class david:
    # https://pubchem.ncbi.nlm.nih.gov/compound/Epigallocatechin-Gallate
    MOLECULE = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

    N_PROCESSES = 12  # number of CPU cores to use
    CHUNK_SIZE = 1000  # largest chunk per CPU core (regulates RAM usage)

    EPOCHS = 200
    LOSS_TOLERANCE = 0.02
    EPSILON = 0.1
    BATCH_SIZE = 32

    def __init__(self) -> None:
        self.ROOT_PATH: Path = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "outputs"

        # input data
        self.MODEL_DIR_PR: Path = self.ROOT_PATH / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        self.QSAR_DIR : Path = self.ROOT_PATH / "tutorial/data/models/qsar"

        # output
        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_DIR: Path = self.BASE_OUTPUT_DIR / self.RUN_ID

        self.MODEL_DIR_TL: Path = self.OUTPUT_DIR / "transfer_learning"
        self.MODEL_DIR_RL: Path = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL_DIR_TL_RL: Path = self.MODEL_DIR_RL

        self.DATA_DIR: Path = self.MODEL_DIR_TL / "data"

        # dir creation
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR_TL.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR_RL.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.VOC = vocabulary.VocSmiles.fromFile(self.MODEL_DIR_PR / "Papyrus05.5_smiles_rnn_PT.vocab")

        self.PRETRAINED = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)  # is_lstm = MORE PARAMETERS
        self.PRETRAINED.loadStatesFromFile(self.MODEL_DIR_PR / "Papyrus05.5_smiles_rnn_PT.pkg")

    def __LEARNING_TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(self, molecule_smile : str) -> list[Any]:

        pipline = receptor_similar.MoleculeBioactivityPipeline()
        result: DataFrame = pipline.run(molecule_smile).papyrus_curated
        if result.empty:
            raise RuntimeError("No active ligands found...")

        smiles: Series = result["SMILES"]

        smiles_paralel = processing.Standardization(n_proc=self.N_PROCESSES, chunk_size=self.CHUNK_SIZE).apply(smiles)
        if smiles_paralel is None:
            raise RuntimeError("Standardization failed to process molecules.")

        if SAVE_PROOF:
            pd.Series(smiles_paralel, name="SMILES").to_csv(
                self.MODEL_DIR_TL / "training_ligands.tsv", sep="\t", index=False
            )

        return smiles_paralel

    def __LEARNING_TRANSFER_LEARNING_tokenization(self, smiles_train : list[Any] ) -> tuple[datasets.SmilesDataSet, processing.CorpusEncoder]:

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
        data_collector = datasets.SmilesDataSet(self.DATA_DIR / "ligand_corpus.tsv", rewrite=True)
        
        encoder.apply(smiles_train, collector=data_collector)

        return data_collector, encoder

    def __LEARNING_TRANSFER_LEARNING_make_test_dataset(self, data_collector : datasets.SmilesDataSet) -> tuple[DataLoader, DataLoader]:

        if SAVE_PROOF:
            splitter = processing.RandomTrainTestSplitter(0.1, 10000)
            train, test  = splitter(data_collector.getData())
            
            pd.DataFrame(train, columns=data_collector.getColumns()).to_csv(
                self.DATA_DIR / "ligand_train.tsv", header=True, index=False, sep="\t"
            )
            pd.DataFrame(test, columns=data_collector.getColumns()).to_csv(
                self.DATA_DIR / "ligand_test.tsv", header=True, index=False, sep="\t"
            )

            self.VOC.toFile(self.DATA_DIR / "pretrained.vocab")
            
            train_loader = datasets.SmilesDataSet.dataToLoader(train, batch_size=self.BATCH_SIZE, vocabulary=self.VOC)
            valid_loader = datasets.SmilesDataSet.dataToLoader(test, batch_size=self.BATCH_SIZE, vocabulary=self.VOC)
            return train_loader, valid_loader
        
        loaders = data_collector.asDataLoader(
            batch_size=self.BATCH_SIZE,
            splitter=processing.RandomTrainTestSplitter(0.1, 10000),
        )
        if isinstance(loaders, (list, tuple)) and len(loaders) == 2:
            return loaders[0], loaders[1]
        raise RuntimeError("Expected exactly two DataLoaders (train, valid) from splitter.")

    def __LEARNING_TRANSFER_LEARNING_transfer_learning(self, train_loader : DataLoader, valid_loader : DataLoader, model_prefix : Path) -> generators.SequenceRNN:

        finetuned = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        finetuned.loadStatesFromFile(self.MODEL_DIR_PR / "Papyrus05.5_smiles_rnn_PT.pkg")

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
            self.VOC.toFile(self.MODEL_DIR_TL / "finetuned.vocab") #NEEDED ONLY FOR NOT-PRETRAINED AI

        return finetuned

    def __molecules_save_image(self, generated_sample : DataFrame) -> None:
        mols = [Chem.MolFromSmiles(s) for s in generated_sample["SMILES"][:25] if Chem.MolFromSmiles(s)]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[250, 250])
            img.save(str(self.MODEL_DIR_TL / "generated_molecules.png"))

    def __LEARNING_TRANSFER_LEARNING_vizualize(self, finetuned : generators.SequenceRNN, model_prefix : Path) -> DataFrame:

        performance_for_epoch: DataFrame = pd.read_csv(f"{model_prefix}_fit.tsv", sep="\t")
        
        with MY_save_plot(self.MODEL_DIR_TL / "training_loss.png") as ax:
            performance_for_epoch[["loss_train", "loss_valid", "valid_ratio"]].plot.line(logy=True, ax=ax)

        generated_sample: DataFrame = finetuned.generate(num_samples=100)
        generated_sample.to_csv(self.MODEL_DIR_TL / "generated_molecules.tsv", sep="\t", index=False)

        self.__molecules_save_image(generated_sample)

        return generated_sample

    def LEARNING_TRANSFER_LEARNING(self, my_molecule : str = MOLECULE) -> tuple[generators.SequenceRNN, DataFrame]:
        
        smiles_train = self.__LEARNING_TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(my_molecule)

        data_collector, encoder = self.__LEARNING_TRANSFER_LEARNING_tokenization(smiles_train)
        
        train_loader, valid_loader = self.__LEARNING_TRANSFER_LEARNING_make_test_dataset(data_collector)

        model_prefix = self.MODEL_DIR_TL / "david_finetuned" #wil be added ""_SOMETING.tsv" to the end
        finetuned: generators.SequenceRNN = self.__LEARNING_TRANSFER_LEARNING_transfer_learning(train_loader, valid_loader, model_prefix)

        generated_sample = self.__LEARNING_TRANSFER_LEARNING_vizualize(finetuned, model_prefix)

        return finetuned, generated_sample

    # def COMPARE_vizualization(self, scores):
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #     scores.QSPRpred_A2AR_RandomForestClassifier.hist(ax=axes[0])
    #     axes[0].set_title("QSPRpred_A2AR_RandomForestClassifier")
    #     scores.SA.hist(ax=axes[1])
    #     axes[1].set_title("SA")
    #     fig.savefig(os.path.join(self.MODEL_DIR_TL, "score_distributions.png"), bbox_inches="tight")
    #     plt.close(fig)

    # def COMPARE_SCORE_SHOWCASE(self, environment, my_molecule : str="C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"):
    #     finetuned, generated_sample = self.TRANSFER_LEARNING(my_molecule)
    #     scores = environment.getScores(generated_sample.SMILES)
        
    #     # Save scored molecules
    #     scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    #     scored_df.to_csv(os.path.join(self.MODEL_DIR_TL, "tl_generated_candidates_scored.tsv"), sep="\t", index=False)

    #     self.__TRANSFER_LEARNING_vizualize()

    def LEARNING_RAINFORCEMENT_LEARNING_vizualization(self, scores, generated, qsprpred_scorer):
        ...
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        scores[qsprpred_scorer.getKey()].hist(ax=axes[0])
        axes[0].set_title(qsprpred_scorer.getKey())
        scores.SA.hist(ax=axes[1])
        axes[1].set_title("SA")
        fig.savefig(self.MODEL_DIR_RL / "rl_score_distributions.png", bbox_inches="tight")
        plt.close(fig)

        dataset = qspr_data.MoleculeTable("david_agent", df=generated)
        dataset.addProperty(qsprpred_scorer.getKey(), scores[qsprpred_scorer.getKey()].values)
        dataset.addDescriptors(
            [qspr_fps.MorganFP(radius=3, nBits=2048)]
        )

        plt_manifold = scaff_plot.Plot(manifold.TSNE())
        fig_manifold = plt_manifold.plot(
            dataset,
            color_by=qsprpred_scorer.getKey(),
            interactive=False,
            color_continuous_scale="rdylgn"
        )
        if fig_manifold is not None:
            fig_manifold.write_html(str(self.MODEL_DIR_RL / "rl_chemical_space.html"))

    def LEARNING_COMPARE_vizualization(self, df, generated):
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

        dataset = qspr_data.MoleculeTable(name="david_joined", df = df_joined)
        dataset.addDescriptors([qspr_fps.MorganFP(radius=3, nBits=2048)])

        plt_manifold = scaff_plot.Plot(manifold.TSNE())
        fig_manifold = plt_manifold.plot(dataset, recalculate=False, color_by="Group", interactive=False)
        if fig_manifold is not None:
            fig_manifold.write_html(str(self.MODEL_DIR_RL / "chemical_space_comparison.html"))

    def RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(self, environment : environment.DrugExEnvironment):

        agent = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        agent.loadStatesFromFile(self.MODEL_DIR_RL / "david_agent.pkg")

        generated_sample = agent.generate(num_samples=100)

        scores = environment.getScores(generated_sample.SMILES)

        # Save scored molecules
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(self.MODEL_DIR_RL / "rl_generated_candidates_scored.tsv", sep="\t", index=False)

        qsprpred_scorer = [s for s in environment.scorers if isinstance(s, qsprpred.QSPRPredScorer)][0]
        self.LEARNING_RAINFORCEMENT_LEARNING_vizualization(scores, generated_sample, qsprpred_scorer)

    def LEARNING_RAINFORCEMENT_LEARNING(self):

        QSAR_DIR_old_way_for_external_package : str = str(self.QSAR_DIR)

        predictor = qspr_models.SklearnModel(
            name="DAVID_RandomForestClassifier",
            base_dir=QSAR_DIR_old_way_for_external_package
        )

        #TODO: how about more functions?

        qsprpred_scorer = qsprpred.QSPRPredScorer(predictor)

        sascore = properties.Property("SA")
        #how about: 'QED': Quantitative Estimate of Drug-likeness
        sascore.setModifier(modifiers.SmoothClippedScore(lower_x=5, upper_x=3))
        qsprpred_scorer.setModifier(modifiers.ClippedScore(lower_x=0.2, upper_x=0.8))

        scorers: list[properties.Property | qsprpred.QSPRPredScorer] = [
            qsprpred_scorer,
            sascore
        ]
        thresholds = [
            0.5,
            0.1
        ]

        env = environment.DrugExEnvironment(scorers, thresholds, reward_scheme=rewards.ParetoCrowdingDistance())

        finetuned = generators.SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        finetuned_path = self.MODEL_DIR_TL / "david_finetuned.pkg"
        if not finetuned_path.exists():
            finetuned_path = self.ROOT_PATH / "tutorial/data/models/finetuned/smiles-rnn/david_finetuned.pkg"
        finetuned.loadStatesFromFile(finetuned_path)

        explorer = explorers.SequenceExplorer(
            agent = finetuned, #TODO pretrained
            env = env,
            mutate = self.PRETRAINED, # network introducing "random mutations" to the generated structures (rate determined by epsilon)
            epsilon = self.EPSILON,
            use_gpus = GPUS
        )

        monitor = monitors.FileMonitor(self.MODEL_DIR_RL / "david_agent", save_smiles=True, reset_directory=True)
        explorer.fit(monitor=monitor, epochs=100)

        df_info = pd.read_csv(self.MODEL_DIR_RL / "david_agent_fit.tsv", sep="\t")
        df_info.head()
        with MY_save_plot(self.MODEL_DIR_RL / "rl_training_curves.png") as ax:
            df_info[["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"]].plot.line(ax=ax)

        self.RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(env)

    # In DrugEx, Reinforcement Learning consists of 5 modular pieces:

    # The Agent (agent): The model being trained (starts from your david_finetuned.pkg).
    # |↑        The Mutator (mutate): A frozen baseline model (usually self.PRETRAINED). With probability epsilon (e.g. 0.1), tokens are chosen from the mutator instead of the agent, preventing the agent from getting stuck generating the exact same single molecule over and over ("mode collapse").
    # ↓|        ↓
    # Scorers & Modifiers (Scorer + ScoreModifier): Every scorer must output values scaled between 0.0 and 1.0 (desirability utility).
    # |   ↓↑
    # |   The Environment (DrugExEnvironment): Takes your scorers, matching desirability thresholds, and an objective combiner (e.g. ParetoCrowdingDistance() for multi-objective optimization).
    # ↓
    # The Explorer (SequenceExplorer): Orchestrates the generator sampling, scoring, policy gradient updates, and passes metrics to FileMonitor.
    def REAL_RAINFORCEMENT_LEARNING(self) -> None:
        # TODO: similarity.TverskyFingerprintSimilarity
        ...



if __name__ == "__main__":
    tmp = david()
    tmp.LEARNING_TRANSFER_LEARNING()
    tmp.LEARNING_RAINFORCEMENT_LEARNING()
    tmp.REAL_RAINFORCEMENT_LEARNING()



