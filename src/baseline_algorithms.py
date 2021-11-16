from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

from src.classifier import LogisticRegressionClassifier
from src.data_manager import DataManager
from src.data_structures import LabeledSet
from src.faiss_searcher import FaissIndex


class BaseBaselineALgorithm(ABC):
    """
    This is the base class for all baseline approaches to run alongside SEALS
    """

    def __init__(self, name: str = "base"):
        """
        Initialize the baseline algotithm
        :param name: The name of the baseline
        """
        self.name = name
        self.scores = {
            "precision": [],
            "recall": [],
            "average_precision": [],
            "positives": [],
        }
        self.labeled_set = LabeledSet()
        self.batch_size = 100
        self.data_manager = None
        self.faiss_index = None
        self.eval_class = None
        self.test_data = None
        self.new_class = True

    def initialize_data(
        self,
        initial_set: LabeledSet,
        data_manager: DataManager,
        faiss_index: FaissIndex,
        eval_class: str,
        test_data: Tuple,
        batch_size: int = 100,
    ):
        """
        Initialize the baseline algorithm with the same labeled set as SEALS
        :param initial_set: The labeled seed set to copy the values from
        :param data_manager: The data manager to get data from
        :param faiss_index: The faiss index to search for neighbors
        :param eval_class: The eval class to use for this run
        :param test_data: The test data for computing scores
        :param batch_size: The batch size for every iteration
        """
        self.labeled_set.add_data(
            initial_set.X, initial_set.y, initial_set.indices
        )
        self.data_manager = data_manager
        self.faiss_index = faiss_index
        self.batch_size = batch_size
        self.eval_class = eval_class
        self.test_data = test_data
        self.new_class = True

    @abstractmethod
    def iteration(self):
        """
        This function is called in every iteration of SEALS to get the baseline
        scores. This must be implemented in every derived class!
        """
        raise NotImplementedError()

    def finish_run(self) -> Dict[str, List]:
        scores = self.scores.copy()
        self.scores = {
            "precision": [],
            "recall": [],
            "average_precision": [],
            "positives": [],
        }
        self.labeled_set = LabeledSet()
        return scores


class MaxEntAllBaseline(BaseBaselineALgorithm):
    """
    The MaxEnt-All baseline that uses Maximum Entropy as a selecion strategy
    but searches all data points and does not restrict the candidate pool like
    MaxEnt-SEALS does.
    """

    def __init__(self):
        super().__init__("MaxEnt-All")
        self.classifier = LogisticRegressionClassifier()

    def iteration(self):
        """
        Run one iteration of the MaxEnt-All baseline
        """
        self.classifier.train(self.labeled_set)
        self.compute_scores(self.test_data)
        self.add_elements_to_set(self.eval_class)

    def compute_scores(self, test_data: Tuple):
        """
        This function computes the scores required for plotting later.
        The scores we need are: precision, recall, pool_size
        :param test_data: The test data required to compute the scores
        """
        predictions, probabilities = self.classifier.predict(test_data[0])
        precision = precision_score(test_data[1], predictions)
        recall = recall_score(test_data[1], predictions)
        average_precision = average_precision_score(
            test_data[1], probabilities[:, 1]
        )
        self.scores["precision"].append(precision)
        self.scores["recall"].append(recall)
        self.scores["average_precision"].append(average_precision)
        self.scores["positives"].append(np.sum(self.labeled_set.y))

    def add_elements_to_set(self, eval_class: str) -> None:
        """
        This function handles the inner loop of the algorithm.
        It selects batch_size elements according to the max entropy, then
        adds it to the labeled set and removes it from the data pool.
        It also adds the k nearest neighbors of the removed element to the pool
        :param eval_class: The class we are evaluating currently
        """
        embeddings = np.array(self.data_manager.embedding_mm)
        _, prob = self.classifier.predict(embeddings)
        entropies = prob[:, 1] * np.log(prob[:, 1]) + prob[:, 0] * np.log(
            prob[:, 0]
        )
        entropy_orders = np.argsort(entropies)  # Max entropy
        count = 0
        index = 0
        while count < self.batch_size:
            if entropy_orders[index] not in self.labeled_set.indices:
                label = (
                    1
                    if entropy_orders[index]
                    in self.data_manager.eval_class_indices[eval_class]
                    else 0
                )
                embedding = embeddings[entropy_orders[index], :].reshape(
                    (1, 256)
                )
                self.labeled_set.add_data(
                    embedding,
                    np.array([label]).reshape((1,)),
                    [entropy_orders[index]],
                )
                count += 1
            index += 1


class RandomAllBaseline(BaseBaselineALgorithm):
    """
    The Random-All baseline that uses Maximum Entropy as a selecion strategy
    but searches all data points and does not restrict the candidate pool like
    MaxEnt-SEALS does.
    """

    def __init__(self):
        super().__init__("Random-All")
        self.classifier = LogisticRegressionClassifier()

    def iteration(self):
        """
        Run one iteration of the MaxEnt-All baseline
        """
        self.classifier.train(self.labeled_set)
        self.compute_scores(self.test_data)
        self.add_elements_to_set(self.eval_class)

    def compute_scores(self, test_data: Tuple):
        """
        This function computes the scores required for plotting later.
        The scores we need are: precision, recall, pool_size
        :param test_data: The test data required to compute the scores
        """
        predictions, probabilities = self.classifier.predict(test_data[0])
        precision = precision_score(test_data[1], predictions)
        recall = recall_score(test_data[1], predictions)
        average_precision = average_precision_score(
            test_data[1], probabilities[:, 1]
        )
        self.scores["precision"].append(precision)
        self.scores["recall"].append(recall)
        self.scores["average_precision"].append(average_precision)
        self.scores["positives"].append(np.sum(self.labeled_set.y))

    def add_elements_to_set(self, eval_class: str) -> None:
        """
        This function handles the inner loop of the algorithm.
        It selects batch_size elements according to the max entropy, then
        adds it to the labeled set and removes it from the data pool.
        It also adds the k nearest neighbors of the removed element to the pool
        :param eval_class: The class we are evaluating currently
        """
        count = 0
        while count < self.batch_size:
            random_index = np.random.randint(
                0, self.data_manager.num_embeddings
            )
            if random_index not in self.labeled_set.indices:
                label = (
                    1
                    if random_index
                    in self.data_manager.eval_class_indices[eval_class]
                    else 0
                )
                embedding = self.data_manager.embedding_mm[
                    random_index, :
                ].reshape((1, 256))
                self.labeled_set.add_data(
                    embedding, np.array([label]).reshape((1,)), [random_index]
                )
                count += 1


class FullSupervisionBaseline(BaseBaselineALgorithm):
    """
    The Full Supervision baseline uses all labeled points as training data.
    """

    def __init__(self):
        super().__init__("FullSupervision")
        self.classifier = LogisticRegressionClassifier(solver="lbfgs")
        self.scores = {"precision": [], "recall": [], "average_precision": []}

    def iteration(self):
        """
        Run one iteration of the MaxEnt-All baseline
        """
        if self.new_class:
            labeled_set = LabeledSet()
            labels = np.zeros((self.data_manager.num_embeddings,))
            labels[self.data_manager.eval_class_indices[self.eval_class]] = 1
            labeled_set.add_data(
                np.array(self.data_manager.embedding_mm), labels, []
            )
            self.classifier.train(labeled_set)
            self.compute_scores(self.test_data)
            self.new_class = False

    def compute_scores(self, test_data: Tuple):
        """
        This function computes the scores required for plotting later.
        The scores we need are: precision, recall, pool_size
        :param test_data: The test data required to compute the scores
        """
        predictions, probabilities = self.classifier.predict(test_data[0])
        precision = precision_score(test_data[1], predictions)
        recall = recall_score(test_data[1], predictions)
        average_precision = average_precision_score(
            test_data[1], probabilities[:, 1]
        )
        self.scores["precision"] = [precision] * 20
        self.scores["recall"] = [recall] * 20
        self.scores["average_precision"] = [average_precision] * 20
