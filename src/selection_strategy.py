from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.classifier import BaseClassifier
from src.data_structures import DataPool


class BaseSelectionStrategy(ABC):
    @abstractmethod
    def select_element(
        self, classifier: BaseClassifier, pool: DataPool
    ) -> Tuple[int, int, np.ndarray]:
        """
        Select an element from the pool according to the selection strategy.
        :param classifier: The trained classifier.
        :param pool: The pool of all data points that can be selected.
        :return: Tuple with three elements:
            0: The index of the selected element in the pool
            1: The index of the selected element in the faiss index
            2: The embedding of the selected element
        """
        raise NotImplementedError()


class MaxEntropySelectionStrategy(BaseSelectionStrategy):
    """
    A selection strategy that selects the element with the highest entropy.
    """

    def select_element(
        self, classifier: BaseClassifier, pool: DataPool
    ) -> Tuple[int, int, np.ndarray]:
        """
        Select the element from the pool that has the maximum entropy.
        :param classifier: The trained classifier.
        :param pool: The pool of all data points that can be selected.
        :return: Tuple with three elements:
            0: The index of the selected element in the pool
            1: The index of the selected element in the faiss index
            2: The embedding of the selected element
        """
        _, prob = classifier.predict(pool.get_all()[0])
        entropies = prob[:, 0] * np.log(prob[:, 0])
        pool_index = int(np.argmax(entropies))
        return (
            pool_index,
            pool.indices[pool_index],
            pool.embeddings[pool_index, :],
        )
