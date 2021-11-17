# This is the main file that runs the experiment and recreates plot 1b
import json

from src.baseline_algorithms import MaxEntAllBaseline, RandomAllBaseline
from src.classifier import LogisticRegressionClassifier
from src.plotter import Plotter
from src.seals import SEALSAlgorithm
from src.selection_strategy import MaxEntropySelectionStrategy

if __name__ == "__main__":
    classifier = LogisticRegressionClassifier()
    selection = MaxEntropySelectionStrategy()
    baselines = [MaxEntAllBaseline(), RandomAllBaseline()]
    seals = SEALSAlgorithm(
        classifier,
        selection,
        num_classes=10,
        random_classes=False,
        baseline_algorithms=baselines,
    )
    scores = seals.run(repetitions=2)

    with open("data/results.json", "w") as fp:
        json.dump(scores, fp)

    Plotter.create_plots(scores)
