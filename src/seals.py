# This file implements the SEALS algorithm logic
from typing import Dict, List, Tuple

import numpy as np
from alive_progress import alive_bar
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

from src.baseline_algorithms import BaseBaselineALgorithm
from src.classifier import BaseClassifier
from src.data_manager import DataManager
from src.data_structures import DataPool, LabeledSet
from src.faiss_searcher import FaissIndex
from src.selection_strategy import BaseSelectionStrategy


class SEALSAlgorithm:
    """
    This class implements the main logic behind the SEALS algorithm.
    It combines everything and runs all the experiments.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        selection: BaseSelectionStrategy,
        num_classes: int = 10,
        random_classes: bool = False,
        baseline_algorithms: List[BaseBaselineALgorithm] = None,
    ) -> None:
        """
        Initialize an instance of the SEALS algorithm
        :param classifier: The classifier to use for SEALS
        :param selection: The selection strategy to use
        :param num_classes: The number of classes to evaluate
        :param random_classes: Evaluate random classes (True) or the classes
            from the paper (False)
        :param baseline_algorithms: List of baselines to include in the
            experiment
        """
        self.data_manager = DataManager(
            num_classes, random_classes=random_classes
        )
        self.faiss_index = FaissIndex()
        self.pool = DataPool()
        self.labeled_set = LabeledSet()
        self.classifier = classifier
        self.selection = selection
        self.batch_size = 100
        self.baselines = baseline_algorithms
        self.manual_labeling = False

    def run(self, repetitions: int = 5) -> Dict[str, Dict]:
        """
        Run experiments for all classes selected by the DataManager
        :param repetitions: Number of runs per class.
        :return: The scores for the different runs.
        """
        eval_classes = self.data_manager.eval_classes
        all_scores = {}
        with alive_bar(
            len(eval_classes) * repetitions,
            title="Running experiments",
            force_tty=True,
        ) as bar:
            for eval_class in eval_classes:
                test_data = self.data_manager.get_test_data(eval_class)
                for rep in range(repetitions):
                    scores = self.run_one_experiment(eval_class, test_data)
                    all_scores[f"{eval_class}_{rep}"] = scores
                    bar()
        return all_scores

    def run_one_experiment(self, eval_class: str, test_data: Tuple) -> Dict:
        """
        Run one SEALS experiment for the specified class.
        :param eval_class: The class to run SEALS on
        :param test_data: The test data as a tuple (embeddings, labels)
        :return: Scores for this run
        """
        scores = {
            "precision": [],
            "recall": [],
            "pool_size": [],
            "average_precision": [],
            "positives": [],
        }
        self.pool = DataPool()
        self.labeled_set = self.get_seed_set(eval_class)
        self.initialize_baselines(eval_class, test_data)
        self.add_neighbors_to_pool(self.labeled_set.get_data()[0])
        while self.labeled_set.size < 2000:
            self.classifier.train(self.labeled_set)
            scores = self.compute_scores(test_data, scores)
            for i in range(self.batch_size):
                self.add_element_to_set(eval_class)
            self.update_baselines()
        self.classifier.train(self.labeled_set)
        self.update_baselines()
        scores = self.compute_scores(test_data, scores)
        scores["baselines"] = self.finish_baselines()
        return scores

    def add_element_to_set(self, eval_class: str) -> None:
        """
        This function handles the inner loop of the SEALS algorithm.
        It selects an element according to the selection strategy, then
        adds it to the labeled set and removes it from the data pool.
        It also adds the k nearest neighbors of the removed element to the pool
        :param eval_class: The class we are evaluating currently
        """
        pool_index, faiss_index, embedding = self.selection.select_element(
            self.classifier, self.pool
        )
        label = (
            1
            if faiss_index in self.data_manager.eval_class_indices[eval_class]
            else 0
        )
        embedding = embedding.reshape((1, 256))
        self.labeled_set.add_data(
            embedding, np.array([label]).reshape((1,)), [faiss_index]
        )
        self.pool.remove_element(pool_index)
        self.add_neighbors_to_pool(embedding)

    def get_seed_set(self, eval_class: str) -> LabeledSet:
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
        seed_set.add_data(embeddings, np.array([1] * 5 + [0] * 95), selected)
        return seed_set

    def add_neighbors_to_pool(self, embeddings: np.ndarray) -> None:
        """
        This function adds the k nearest neighbors of the embeddings to the
        data pool in self.pool. It removes duplicates and ignores
        :param embeddings: The embeddings to get the nearest neighbors from
        """
        indices = list(np.unique(self.faiss_index.search(embeddings)))
        indices = [x for x in indices if x not in self.labeled_set.indices]
        nn_embeddings = self.data_manager.get_embedding(indices)
        self.pool.add(nn_embeddings, indices)

    def compute_scores(self, test_data: Tuple, scores: Dict):
        """
        This function computes the scores required for plotting later.
        The scores we need are: precision, recall, pool_size
        :param test_data: The test data required to compute the scores
        :param scores: The scores dictionary to append the scores to
        :return: The scores dictionary with the appended scores
        """
        predictions, probabilities = self.classifier.predict(test_data[0])
        precision = precision_score(test_data[1], predictions)
        recall = recall_score(test_data[1], predictions)
        average_precision = average_precision_score(
            test_data[1], probabilities[:, 1]
        )
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["average_precision"].append(average_precision)
        scores["pool_size"].append(len(self.pool.indices))
        scores["positives"].append(np.sum(self.labeled_set.y))
        return scores

    def initialize_baselines(self, eval_class: str, test_data: Tuple) -> None:
        """
        This function initializes the baseline algorithms in every run of SEALS
        """
        for baseline in self.baselines:
            baseline.initialize_data(
                self.labeled_set,
                self.data_manager,
                self.faiss_index,
                eval_class,
                test_data,
            )

    def update_baselines(self) -> None:
        """
        This function updates the baseline algorithms in every batch iteration
        """
        for baseline in self.baselines:
            baseline.iteration()

    def finish_baselines(self) -> Dict[str, Dict]:
        """
        This function updates the baseline algorithms in every batch iteration
        """
        baseline_scores = {}
        for baseline in self.baselines:
            baseline_score = baseline.finish_run()
            baseline_scores[baseline.name] = baseline_score
        return baseline_scores
