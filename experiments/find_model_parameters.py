import itertools
from typing import Tuple

import numpy as np
from alive_progress import alive_bar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score

from src.data_manager import DataManager
from src.definitions import (
    cleaned_test_annotations_file,
    cleaned_test_uri_file,
    test_images_file,
)


def get_test_data(eval_class: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns the test data for the given class
    :param eval_class: The class we are evaluating currently
    :return: Tuple of embeddings and labels from the test set:
        0: embeddings: np.ndarray in format (n, 256)
        1: labels: np.ndarray in format (n,) with entries 0 or 1
    """
    embeddings = np.array(
        np.memmap(
            "data/test_embeddings.bin",
            dtype="float32",
            mode="r",
            shape=(113_508, 256),
        )
    )
    counts, ids = DataManager.get_positives_for_classes(
        cleaned_test_annotations_file
    )
    positive_ids = ids[eval_class]
    positive_indices = DataManager.get_indices_from_ids(
        positive_ids, test_images_file, cleaned_test_uri_file
    )
    labels = np.zeros((embeddings.shape[0],))
    labels[positive_indices] = 1
    return embeddings, labels


if __name__ == "__main__":
    eval_class = "/m/0642b4"
    indices = np.load("ind.npy").astype(np.int)
    y = np.load("y.npy")

    parameters = [
        ["liblinear", "saga", "lbfgs"],
        [100, 1000, 10000],
        [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        [None, "balanced"],
    ]
    experiments = list(itertools.product(*parameters))
    X_test, y_test = get_test_data(eval_class)
    embedding_mm = np.memmap(
        "data/train_embeddings.bin",
        dtype="float32",
        mode="r",
        shape=(6_607_906, 256),
    )
    X = embedding_mm[indices, :]
    print(X.shape)
    print(y.shape)
    ap = []
    recall = []
    with alive_bar(
        len(experiments), title="Grid Search", force_tty=True
    ) as bar:
        for experiment in experiments:
            solver = experiment[0]
            max_iter = experiment[1]
            C = experiment[2]
            class_weight = experiment[3]

            classifier = LogisticRegression(
                solver=solver,
                max_iter=max_iter,
                C=C,
                class_weight=class_weight,
            )
            classifier.fit(X, y)
            pred = classifier.predict(X_test)
            prob = classifier.predict_proba(X_test)
            this_recall = recall_score(y_test, pred)
            this_ap = average_precision_score(y_test, prob[:, 1])
            recall.append(this_recall)
            if len(ap):
                if this_ap > np.max(ap):
                    print(f"Max AP: {this_ap}")
                    print(experiment)
            ap.append(this_ap)
            bar()
