from qsprpred.data import MoleculeTable
from scaffviz.clustering.manifold import TSNE
from scaffviz.depiction.plot import Plot
from qsprpred.data.descriptors.fingerprints import MorganFP
from drugex.training import explorers
from pandas.core.frame import DataFrame
import drugex.training.generators.sequence_rnn
from drugex.training.scorers.properties import Property
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus import vocabulary
from drugex.training.monitors import FileMonitor
from torch.utils.data import DataLoader
from drugex.data.corpus import vocabulary
from drugex.data.datasets import SmilesDataSet
from drugex.data.processing import RandomTrainTestSplitter
try:
    # pyrefly: ignore [missing-import]
    import receptor_similar
except ModuleNotFoundError:
    from _david import receptor_similarimport
import pandas as pd


#df = pd.read_csv(f'{DATASETS_PATH}/SPECIFIC_LIGANDS.tsv', sep='\t', header=0, na_values=('NA', 'nan', 'NaN'))

import os
from pathlib import Path

from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.processing import Standardization

GPUS = [0] # we will use only one GPU with ID=0, but if you have more, you can list more GPU IDs here

class david:

    GLOBAL_MODELS_PR_PATH = "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/"
    MODELS_PR_PATH = Path(__file__).resolve().parent.parent.joinpath(GLOBAL_MODELS_PR_PATH)
    DATA_DIR = "data/datasets/encoded/rnn/"

    N_PROCESSES = 12 # number of CPU cores to use
    CHUNK_SIZE = 1000 # largest chunk per CPU core (regulates RAM usage)

    EPOCHS = 200
    LOSS_TOLERANCE = 0.02
    EPSILON = 0.1
    BATCH_SIZE = 256


    def __init__(self):
        voc = VocSmiles.fromFile(os.path.join(self.MODELS_PR_PATH, "Papyrus05.5_smiles_rnn_PT.vocab"), encode_frags=False)

        pretrained = SequenceRNN(voc, is_lstm=True, use_gpus=GPUS)
        pretrained.loadStatesFromFile(os.path.join(self.MODELS_PR_PATH, "Papyrus05.5_smiles_rnn_PT.pkg"))

    #https://pubchem.ncbi.nlm.nih.gov/compound/Epigallocatechin-Gallate

    def __TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(self, source_smile : str):

        pipline = receptor_similar.MoleculeBioactivityPipeline()
        result = pipline.run(source_smile)
        smiles_train = result.papyrus_curated["smiles"].to_list()

        standardizer = Standardization(n_proc=self.N_PROCESSES, chunk_size=self.CHUNK_SIZE)
        smiles_train = standardizer.apply(smiles_train)
        #        df = pd.read_csv(f'{self.DATASETS_PATH}/{smiles_train_path}', sep='\t', header=0, na_values=('NA', 'nan', 'NaN'))
        #        smiles_train = df.SMILES
        print("="*30, "__TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES", "="*30, *smiles_train[0:10], sep='\n')

        return smiles_train

    def __TRANSFER_LEARNING_tokenization(self, smiles_train : list[any] ) -> SmilesDataSet:
        from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter
        from drugex.data.corpus.corpus import SequenceCorpus
        from drugex.data.datasets import SmilesDataSet
        from drugex.logs import logger
        logger.setLevel('ERROR')

        os.makedirs(self.DATA_DIR, exist_ok=True)

        encoder = CorpusEncoder( # CorpusEcoder uses the supplied corpus class to encode tokens for the new data set
            SequenceCorpus, # The corpus CLASS (NOT OBJECT, just just as a constructor) implements how each SMILES string is divided into words by the vocabulary
            {
                # arguments of the SequenceCorpus
                'vocabulary': self.voc, # used vocabulary
                'update_voc': False, # if False, the vocabulary stays fixed (no new tokens are added to it)
                'throw': True # compounds containing unknown tokens are thrown out of the resulting data set

            },
            n_proc=self.N_PROCESSES,
            chunk_size=self.CHUNK_SIZE
        )

        data_collector = SmilesDataSet(os.path.join(self.DATA_DIR, 'ligand_corpus.tsv'), rewrite=True)
        encoder.apply(smiles_train, collector=data_collector)

        return data_collector

    def __TRANSFER_LEARNING_make_test_dataset(self, data_collector : SmilesDataSet) -> tuple[DataLoader, DataLoader]:

        splitter = RandomTrainTestSplitter(0.1, 1e4)
        train, test = splitter(data_collector.getData())
        
        # split data
        for data, name in zip([train, test], ['train', 'test']):
            pd.DataFrame(data, columns=data_collector.getColumns()).to_csv(os.path.join(self.DATA_DIR, f'ligand_{name}.tsv'), header=True, index=False, sep='\t')

        #save updated vocabulary (but we had 'update_voc' = False)
        self.voc.toFile(os.path.join(self.DATA_DIR, 'pretrained.vocab'))
       

        #get data path (from files as it would be useful for later)
        data_set_train = SmilesDataSet(os.path.join(self.DATA_DIR, 'ligand_train.tsv'), voc=self.voc)
        train_loader = data_set_train.asDataLoader(batch_size=self.BATCH_SIZE)

        data_set_test = SmilesDataSet(os.path.join(self.DATA_DIR, 'ligand_test.tsv'), voc=self.voc)
        valid_loader = data_set_test.asDataLoader(batch_size=self.BATCH_SIZE)

        return (train_loader, valid_loader)

    def __TRANSFER_LEARNING_transfer_learning(self, train_loader : DataLoader, valid_loader : DataLoader, destination_folder : str) -> SequenceRNN:
        from drugex.training.monitors import FileMonitor

        ft_path = os.path.join(self.MODEL_DIR, destination_folder)
        finetuned = SequenceRNN(self.voc, is_lstm=True, use_gpus=GPUS)
        finetuned.loadStatesFromFile(os.path.join(self.MODELS_PR_PATH, 'Papyrus05.5_smiles_rnn_PT.pkg'))

        reset_directory : bool =True
        monitor = FileMonitor(ft_path, save_smiles=True, reset_directory=reset_directory)
        finetuned.fit(train_loader, valid_loader, epochs=self.EPOCHS, monitor=monitor, loss_tolerance=self.LOSS_TOLERANCE)

        self.voc.toFile(os.path.join(self.MODEL_DIR, 'finetuned.vocab'))

        return finetuned

    def __TRANSFER_LEARNING_vizualize(self, finetuned : SequenceRNN, destination_folder) -> DataFrame:
        from . import smilesToGrid #from c compiled utiles
        
        df_info = pd.read_csv(f"{self.MODEL_DIR}/{destination_folder}_fit.tsv", sep='\t')
        df_info[['loss_train', 'loss_valid', 'valid_ratio']].plot.line(logy=True)

        generated_sample: DataFrame = finetuned.generate(num_samples=100)
        smilesToGrid(generated_sample.SMILES, molsPerRow=5, n_rows=5)
        return generated_sample

    def TRANSFER_LEARNING(self, my_molecule : str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O") -> DataFrame:
        
        smiles_train = self.__TRANSFER_LEARNING_MY_receptor_similar_molecules_get_SMILES(my_molecule)

        data_collector : SmilesDataSet = self.__TRANSFER_LEARNING_tokenization(smiles_train)
        
        train_loader, valid_loader = self.__TRANSFER_LEARNING_make_test_dataset(data_collector)        
        finetuned: SequenceRNN = self.__TRANSFER_LEARNING_transfer_learning(train_loader, valid_loader, "david_finetuned")
        generated_sample = self.__TRANSFER_LEARNING_vizualize(finetuned, "david_finetuned")
        return generated_sample

    def TRANSFER_LEARNING_vizualization(self, scores):
        scores.QSPRpred_A2AR_RandomForestClassifier.hist()
        scores.SA.hist()

    def TRANSFER_LEARNING_SCORE_SHOWCASE(self, environment, my_molecule : str="C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"):
        generated_sample = self.TRANSFER_LEARNING( my_molecule)
        scores = environment.getScores(generated_sample.SMILES.to_list())
        self.RAINFORCEMENT_LEARNING_vizualization(scores)

    def RAINFORCEMENT_LEARNING_vizualization(self, scores, generated, qsprpred_scorer):
        scores.QSPRpred_A2AR_RandomForestClassifier.hist()
        scores.SA.hist()

        dataset = MoleculeTable("david_agent", df=generated)
        dataset.addProperty(qsprpred_scorer.getKey(), scores[qsprpred_scorer.getKey()].values)
        dataset.addDescriptors(
            [MorganFP(radius=3, nBits=2048)]
        )

        plt = Plot(TSNE())
        plt.plot(
            dataset,
            color_by=qsprpred_scorer.getKey(),
            interactive=False,
            color_continuous_scale="rdylgn"
        )
    def COMPARE_vizualization(self, df, generated):
        df_joined : DataFrame = pd.concat(
            [
                pd.DataFrame(
                    {"SMILS" : df.SMILES.to_list(), "Group" : "Set"}
                ),
                pd.DataFrame(
                    {"SMILES" : generated.SMILES, "Group" : "Generated"}
                )
            ]
        )

        dataset = MoleculeTable(name="david_joined", df = df_joined)
        dataset.addDescriptors([MorganFP(radius=3, nBits=2048)])

        plt = Plot(TSNE())
        plt.plot(dataset, recalculate=False, color_by="Group", interactive=False)

    def RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(self, environment : DrugExEnvironment):

        agent = SequenceRNN(self.voc, is_lstm=True, use_gpus=GPUS)
        agent.loadStatesFromFile(f'{self.MODEL_DIR_RL}/david_agent.pkg')

        generated_sample = agent.generate(num_samples=100)

        scores = environment.getScores(generated_sample.SMILES)
        self.RAINFORCEMENT_LEARNING_vizualization(scores)

    def RAINFORCEMENT_LEARNING(self):

        from qsprpred.models.scikit_learn import SklearnModel

        predictor = SklearnModel(
            name="david_RandomForestCLassifier",
            base_dir="../.training/data/models/qsar"
        )
        
        from drugex.training.scorers.qsprpred import QSPRPredScorer
        from drugex.training.scorers.modifiers import ClippedScore
        #TODO: how about more functions?

        qsprpred_scorer = QSPRPredScorer(predictor)
        qsprpred_scorer.setModifier(ClippedScore(lower_x=0.2, upper_x=0.8))

        from drugex.training.scorers.properties import Property
        from drugex.training.scorers.modifiers import SmoothClippedScore

        sascore = Property("SA")
        sascore.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))

        from drugex.training.environment import DrugExEnvironment
        # PARETO DISTANCE: like comparing AI model cost vs acurrecy
        # NSGA-II Crowding Distance: Model receives minimal reinforcement for generating another identical/similar molecule
        from drugex.training.rewards import ParetoCrowdingDistance

        scorers: list[Property | QSPRPredScorer] = [
            qsprpred_scorer,
            sascore
        ]
        thresholds = [
            0.5,
            0.1
        ]

        environment = DrugExEnvironment(scorers, thresholds, reward_scheme=ParetoCrowdingDistance())

        self.RAINFORCEMENT_LEARNING_SCORE_SHOWCASE(environment)

        from drugex.training.explorers import SequenceExplorer
        import warnings
        warnings.filterwarnings('ignore')

        finetuned = SequenceRNN(self.voc, is_lstm=True, use_gpus=GPUS)
        finetuned.loadStatesFromFile(f'{self.MODEL_DIR}/david_finetuned.pkg')

        explorer = SequenceExplorer(
            agent = finetuned, #TODO pretrained
            env = environment,
            mutate = self.pretrained, # network introducing "random mutations" to the generated structures (rate determined by epsilon)
            epsilon = self.EPOCHS,
            use_gpus = self.GPUS
        )

        MODEL_DIR_RL = "./../tutorial/data/models/RL/rnn"

        monitor = FileMonitor(os.path.join(MODEL_DIR_RL, 'david_agent'), save_smiles=True, reset_directory=True)
        explorer.fit(monitor=monitor, epochs=100)

        df_info = pd.read_csv(f'{MODEL_DIR_RL}/david_agent_fit.tsv', sep='\t')
        df_info.head()
        df_info[['loss_train', 'valid_ratio','unique_ratio', 'desired_ratio']].plot.line()

if __name__ == "__main__":
    tmp = david()
    tmp.TRANSFER_LEARNING()
    tmp.RAINFORCEMENT_LEARNING()
