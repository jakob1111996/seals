# This class runs a SEALS experiment with manual labeling
import json
from typing import List, Tuple

import numpy as np

from src.baseline_algorithms import BaseBaselineALgorithm
from src.classifier import BaseClassifier
from src.data_manager import DataManager
from src.data_structures import DataPool, LabeledSet
from src.definitions import (
    cleaned_train_annotations_file,
    cleaned_train_uri_file,
)
from src.seals import SEALSAlgorithm
from src.selection_strategy import BaseSelectionStrategy
from src.util import get_csv_column


class SEALSManualAlgorithm(SEALSAlgorithm):
    """
    This class runs a SEALS experiment with manual labeling.
    """

    def __init__(
        self,
        classifier: BaseClassifier,
        selection: BaseSelectionStrategy,
        baseline_algorithms: List[BaseBaselineALgorithm] = None,
        eval_class: str = "/m/01bdy",
    ) -> None:
        """
        Initialize an instance of the SEALS Manual Labeling algorithm
        :param classifier: The classifier to use for SEALS
        :param selection: The selection strategy to use
            from the paper (False)
        :param baseline_algorithms: List of baselines to include in the
            experiment
        :param eval_class: The eval class to run manual labeling on
        """
        super().__init__(classifier, selection, 0, False, baseline_algorithms)
        self.manual_labeling = True
        self.uris = get_csv_column(cleaned_train_uri_file, 0)
        self.pool = DataPool()
        self.labeled_set = LabeledSet()
        self.current_index = None
        self.label = -1
        self.pool_index = None
        self.eval_class = eval_class
        self.initialize_data_manager(eval_class)
        self.scores = {
            "precision": [],
            "recall": [],
            "pool_size": [],
            "average_precision": [],
            "positives": [],
        }
        self.test_data = self.data_manager.get_test_data(eval_class)

    def initialize_data_manager(self, eval_class: str) -> None:
        """
        Overwrite the settings of the data manager for manual labeling
        :param eval_class: The class to use in the experiment
        """
        self.data_manager.eval_classes = [eval_class]
        class_counts, class_ids = DataManager.get_positives_for_classes(
            cleaned_train_annotations_file
        )
        self.data_manager.eval_class_ids[eval_class] = class_ids[eval_class]
        self.data_manager.get_indices_for_positives()

    def get_openimages_label(self, faiss_index: int):
        label = (
            1
            if faiss_index
            in self.data_manager.eval_class_indices[self.eval_class]
            else 0
        )
        return label

    def step(self) -> Tuple[str, int]:
        """
        Run one step in the algorithm until the next label is required.
        :return: The URI of the next image to be labeled and the label from OI
        """
        self.add_data_to_labeled_set()
        if self.labeled_set.size % 100 == 0 and self.labeled_set.size > 0:
            self.classifier.train(self.labeled_set)
            self.scores = self.compute_scores(self.test_data, self.scores)
        if self.labeled_set.size == 2000:
            self.scores = {f"{self.eval_class}_0": self.scores}
            with open("data/results_manual.json", "w") as fp:
                json.dump(self.scores, fp)
        if self.labeled_set.size == 100:
            self.add_neighbors_to_pool(self.labeled_set.get_data()[0])
        if self.labeled_set.size < 5:
            # Find 5 positives for seed set
            positive_indices = self.data_manager.eval_class_indices[
                self.eval_class
            ]
            index = np.random.choice(positive_indices, 1, False)[0]
            while index in self.labeled_set.indices:
                index = np.random.choice(positive_indices, 1, False)[0]
            self.current_index = index
            return self.uris[index], self.get_openimages_label(index)
        elif self.labeled_set.size < 100:
            # Find 95 negatives for seed set
            positive_indices = self.data_manager.eval_class_indices[
                self.eval_class
            ]
            index = np.random.randint(0, self.data_manager.num_embeddings)
            while (
                index in positive_indices or index in self.labeled_set.indices
            ):
                index = np.random.randint(0, self.data_manager.num_embeddings)
            self.current_index = index
            return self.uris[index], self.get_openimages_label(index)
        else:
            (
                self.pool_index,
                self.current_index,
                _,
            ) = self.selection.select_element(self.classifier, self.pool)
            return self.uris[self.current_index], self.get_openimages_label(
                self.current_index
            )

    def add_data_to_labeled_set(self):
        """
        Add the data that was labeled before to the labeled set if possible
        """
        if self.current_index is not None and self.label != -1:
            # Add to labeled data
            embedding = self.data_manager.get_embedding(
                self.current_index
            ).reshape((1, 256))
            if self.labeled_set.size < 5:
                if self.label == 1:
                    # print("Adding seed positive")
                    self.labeled_set.add_data(
                        embedding, np.array([1]), [self.current_index]
                    )
            elif self.labeled_set.size < 100:
                if self.label == 0:
                    # print("Adding seed negative")
                    self.labeled_set.add_data(
                        embedding, np.array([0]), [self.current_index]
                    )
            else:
                # print(f"Adding with {self.label}")
                self.labeled_set.add_data(
                    embedding, np.array([self.label]), [self.current_index]
                )
                self.pool.remove_element(self.pool_index)
                self.add_neighbors_to_pool(embedding)

            self.label = -1
            self.current_index = None
