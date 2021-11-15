import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Class that handles all the plotting.
    """

    @staticmethod
    def create_plot(scores: Dict, ax: plt.axis, title: str, key: str):
        data = []
        for class_name, class_scores in scores.items():
            data.append(class_scores[key])
        data = np.asarray(data)
        x = range(100, 2001, 100)
        ax.plot(x, np.mean(data, axis=0), marker="o", markersize=3)
        std = np.std(data, axis=0)
        ax.fill_between(
            x,
            (np.mean(data, axis=0) - std),
            (np.mean(data, axis=0) + std),
            color="b",
            alpha=0.1,
        )
        ax.set_ylabel(title)
        ax.set_xlabel("Number of Labels")

    @staticmethod
    def create_plots(scores: Dict):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        Plotter.create_plot(scores, axes[0][0], "mAP", "average_precision")
        Plotter.create_plot(scores, axes[0][1], "Pool Size", "pool_size")
        Plotter.create_plot(scores, axes[1][0], "Recall", "recall")
        Plotter.create_plot(scores, axes[1][1], "Positives", "positives")
        plt.tight_layout()
        plt.savefig("data/results.png")
        plt.show()

    @staticmethod
    def create_plots_from_file(file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
        Plotter.create_plots(data)
