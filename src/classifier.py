from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data_structures import LabeledSet


class BaseClassifier(ABC):
    """
    This is the base class (interface) all classifiers should inherit from
    """

    @abstractmethod
    def train(self, labeled_data: LabeledSet) -> None:
        """
        Abstract training method that all classifiers must have
        :param labeled_data: The labeled data set for training
        """
        raise NotImplementedError("Abstract method!")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract predict method all classifiers must implement
        :param X: The embeddings to make predictions for
        :return: Tuple with (predictions, confidence)
        """
        raise NotImplementedError("Abstract method!")


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression classifier wrapping sklearn LogisticRegression
    """

    def __init__(self):
        """
        Initialize the classifier
        """
        super().__init__()
        self.clf = None

    def train(self, labeled_data: LabeledSet) -> None:
        """
        Fit the logistic regression classifier to the training data
        :param labeled_data: The labeled dataset to train on.
        """
        X, y = labeled_data.get_data()
        self.clf = LogisticRegression(max_iter=1000, solver="saga")
        self.clf.fit(X, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the labels for the test embeddings using the fit LR classifier
        :param X: The embeddings to make predictions for
        :return: Tuple with (predictions, confidence)
        """
        pred, prob = self.clf.predict(X), self.clf.predict_proba(X)
        return pred, prob
