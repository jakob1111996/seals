# This file contains some data structures that are used by SEALS.
from typing import Tuple

import numpy as np


class LabeledSet:
    def __init__(self, embedding_size: int = 256):
        """
        COnstruct an empty labeled set object
        :param embedding_size: The dimension of the embeddings.
        """
        self.embedding_size = embedding_size
        self.X = np.empty((0, embedding_size))
        self.y = np.empty((0,))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function returns both X and y for easy access
        :return: Tuple with (X, y) data
        """
        return self.X, self.y

    def add_data(self, X_new, y_new) -> None:
        """
        This function can be used to add a data point to the labeled set.
        :param X_new: The embedding of the data point
        :param y_new: The binary label of the data point (0 or 1)
        """
        assert X_new.shape[1] == 256
        self.X = np.concatenate([self.X, X_new], axis=0)
        self.y = np.concatenate([self.y, y_new], axis=0)
