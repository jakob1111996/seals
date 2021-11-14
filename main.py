# This is the main file that runs the experiment and recreates plot 1b
import json

from src.classifier import LogisticRegressionClassifier
from src.seals import SEALSAlgorithm
from src.selection_strategy import MaxEntropySelectionStrategy

if __name__ == "__main__":
    classifier = LogisticRegressionClassifier()
    selection = MaxEntropySelectionStrategy()
    seals = SEALSAlgorithm(classifier, selection, num_classes=150)
    scores = seals.run(repetitions=1)

    with open("data/results.json", "w") as fp:
        json.dump(scores, fp)
