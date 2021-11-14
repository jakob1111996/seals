# This file implements the SEALS algorithm logic
from typing import Dict, List, Tuple

import numpy as np
from alive_progress import alive_bar

from src.data_manager import DataManager
from src.data_structures import LabeledSet
from src.faiss_searcher import FaissIndex


class SEALSAlgorithm:
    """
    This class implements the main logic behind the SEALS algorithm.
    It combines everything and runs the experiments
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.faiss_index = FaissIndex()

    def run(self, repetitions: int = 5) -> List[Dict]:
        """
        Run experiments for all classes selected by the DataManager
        :param repetitions: Number of runs per class.
        :return: The scores for the different runs.
        """
        eval_classes = self.data_manager.eval_classes
        all_scores = []
        with alive_bar(
            len(eval_classes) * repetitions,
            title="Running experiments",
            force_tty=True,
        ) as bar:
            for eval_class in eval_classes:
                test_data = self.data_manager.get_test_data(eval_class)
                for rep in range(repetitions):
                    scores = self.run_one_experiment(eval_class, test_data)
                    all_scores.append(scores)
                    bar()
        return all_scores

    def run_one_experiment(self, eval_class: str, test_data: Tuple) -> Dict:
        """
        Run one SEALS experiment for the specified class.
        :param eval_class: The class to run SEALS on
        :param test_data: The test data as a tuple (embeddings, labels)
        :return: Scores for this run
        """
        scores = {}
        test_embeddings, test_labels = test_data
        labeled_set, indices = self.get_seed_set(eval_class)
        return scores

    def get_seed_set(self, eval_class: str) -> Tuple[LabeledSet, List]:
        """
        This function generates a seed set for a given class string.
        The seed set contains of the embeddings and labels for 5 positive and
        95 negative examples.
        :param eval_class: String that identifies the class to be used.
        :return: Tuple of LabeledSet object with 5 pos and 95 neg examples
            and List with the indices of the seed set.
        """
        seed_set = LabeledSet()
        positive_indices = self.data_manager.eval_class_indices[eval_class]
        selected_positives = np.random.choice(positive_indices, 5, False)
        selected_negatives = []
        while len(selected_negatives) < 95:
            randint = np.random.randint(0, self.data_manager.num_embeddings)
            if randint not in positive_indices:
                selected_negatives.append(randint)
        selected = list(selected_positives.astype(int)) + selected_negatives
        embeddings = self.data_manager.get_embedding(selected)
        seed_set.add_data(embeddings, np.array([1] * 5 + [0] * 95))
        return seed_set, selected
