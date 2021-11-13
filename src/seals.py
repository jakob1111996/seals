# This file implements the SEALS algorithm logic
from typing import Dict, List, Tuple

import numpy as np

from data_manager import DataManager
from data_structures import LabeledSet
from faiss_searcher import FaissIndex


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
        for eval_class in eval_classes:
            for rep in range(repetitions):
                scores = self.run_one_experiment(eval_class)
                all_scores.append(scores)
        return all_scores

    def run_one_experiment(self, eval_class: str) -> Dict:
        """
        Run one SEALS experiment for the specified class.
        :param eval_class: The class to run SEALS on
        :return: Scores for this run
        """
        scores = {}
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
        negative_indices = list(range(self.data_manager.num_embeddings))
        for pos_ind in positive_indices.sort(reverse=True):
            del negative_indices[pos_ind]
        selected_negatives = np.random.choice(negative_indices, 95, False)
        embeddings = self.data_manager.get_embedding(
            selected_positives + selected_negatives
        )
        seed_set.add_data(embeddings, np.array([1] * 5 + [0] * 95))
        return seed_set, selected_positives + selected_negatives
