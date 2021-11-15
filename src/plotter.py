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
        ax.set_title(title)

    @staticmethod
    def create_plots(scores: Dict):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
        Plotter.create_plot(scores, axes[0], "mAP", "average_precision")
        Plotter.create_plot(scores, axes[1], "Pool Size", "pool_size")
        Plotter.create_plot(scores, axes[2], "Recall", "recall")
        plt.show()

    @staticmethod
    def create_plots_from_file(file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
        Plotter.create_plots(data)
