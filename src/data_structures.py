# This file contains some data structures that are used by SEALS.
from typing import List, Tuple

import numpy as np


class LabeledSet:
    def __init__(self, embedding_size: int = 256):
        """
        Construct an empty labeled set object
        :param embedding_size: The dimension of the embeddings.
        """
        self.size = 0
        self.embedding_size = embedding_size
        self.X = np.empty((0, embedding_size))
        self.y = np.empty((0,))
        self.indices = np.empty((0,))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function returns both X and y for easy access
        :return: Tuple with (X, y) data
        """
        return self.X, self.y

    def add_data(
        self, X_new: np.ndarray, y_new: np.ndarray, indices: List
    ) -> None:
        """
        This function can be used to add a data point to the labeled set.
        :param X_new: The embeddings of the data point
        :param y_new: The binary label of the data point (0 or 1)
        :param indices: The faiss indices of the data to add.
        """
        assert X_new.shape[1] == 256
        self.X = np.concatenate([self.X, X_new], axis=0)
        self.y = np.concatenate([self.y, y_new], axis=0)
        self.indices = np.concatenate(
            [self.indices, np.array(indices)], axis=0
        )
        self.size += X_new.shape[0]


class DataPool:
    """
    The pool of data consisting of the nearest neighbors.
    The selection strategy chooses the data to label from this pool.
    """

    def __init__(self):
        self.indices = []  # Indices of the pool elements in the faiss index
        self.indices_set = set()
        self.embeddings = np.empty((0, 256))

    def add(self, embeddings: np.ndarray, indices: List) -> None:
        """
        Add data points to the data pool
        :param embeddings: The embeddings to add to the pool
        :param indices: The corresponding indices for the embeddings
        """
        to_remove = [i for i, x in enumerate(indices) if x in self.indices_set]
        indices = [x for x in indices if x not in self.indices_set]
        for index in indices:
            self.indices_set.add(index)
        self.indices.extend(indices)
        embeddings = np.delete(embeddings, to_remove, 0)
        self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)

    def get_all(self) -> Tuple[np.ndarray, List]:
        """
        Get all the embeddings and indices from the pool
        :return: Tuple with (embeddings, indices)
        """
        return self.embeddings, self.indices

    def remove_element(self, pool_index: int) -> None:
        """
        Remove an element from the pool by its index in the pool
        :param pool_index: The index in the pool that should be removed
        """
        removed = self.indices.pop(pool_index)
        self.indices_set.remove(removed)
        self.embeddings = np.delete(self.embeddings, [pool_index], 0)
