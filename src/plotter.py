# This file implements the plotting functionality to recreate the plot.
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Class that handles all the plotting. All methods are static.
    """

    @staticmethod
    def create_plot(
        scores: Dict, ax: plt.axis, title: str, key: str, colors: List[str]
    ) -> None:
        """
        This function creates a plot with all required lines and confidence
        intervals on one axis of the figure.
        :param scores: The scores dictionary to get the scores from.
        :param ax: The ax used for the plotting
        :param title: The title used for the y label of the plot
        :param key: The score name we are plotting in this plot
        :param colors: A list of colors to use for the plot
        """
        data = []
        for class_name, class_scores in scores.items():
            data.append(class_scores[key])
        stds = np.std(np.asarray(data), axis=0).reshape((20, 1))
        data = np.mean(np.asarray(data), axis=0).reshape((20, 1))
        data, stds, labels = Plotter.add_baseline_data(data, stds, scores, key)
        x = range(100, 2001, 100)
        for i in range(data.shape[1]):
            ax.plot(
                x,
                data[:, i],
                label=labels[i],
                marker="o",
                markersize=3,
                color=colors[i],
            )
            ax.fill_between(
                x,
                (data[:, i] - stds[:, i]),
                (data[:, i] + stds[:, i]),
                color=colors[i],
                alpha=0.1,
            )
        ax.set_ylabel(title)
        ax.set_xlabel("Number of Labels")
        ax.legend()

    @staticmethod
    def create_plots(scores: Dict):
        """
        This function creates the plot with all subplots from a given scores
        dictionary
        :param scores: The dictionary with all the scores of the runs.
        """
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        Plotter.create_plot(
            scores, axes[0][0], "mAP", "average_precision", colors
        )
        Plotter.create_plot(
            scores, axes[0][1], "Pool Size", "pool_size", colors
        )
        Plotter.create_plot(scores, axes[1][0], "Recall", "recall", colors)
        Plotter.create_plot(
            scores, axes[1][1], "Positives", "positives", colors
        )
        plt.tight_layout()
        plt.savefig("data/results.png")
        plt.show()

    @staticmethod
    def add_baseline_data(
        data: np.ndarray, stds: np.ndarray, scores: Dict, key: str
    ):
        """
        This function extracts the scores from the baseline algorithms
        for plotting them later on.
        :param data: The SEALS data ready for plotting in an array
        :param stds: The SEALS standard deviation ready for plotting
        :param scores: The scores dictionary
        :param key: The name of the score to extract right now.
        :return: Tuple with three elements:
            0: The SEALS data concatenated with the baseline data for plotting
                in shape (20, 1 + n_baselines)
            1: The SEALS stdd concatenated with the baseline stdds
            2: The labels of SEALS and the baselines for the legend.
        """
        labels = ["MaxEnt-SEALS"]

        baseline_data = {}
        for class_name, class_scores in scores.items():
            baseline_dict = class_scores["baselines"]
            for baseline_name, baseline_values in baseline_dict.items():
                if key in baseline_values:
                    if baseline_name not in baseline_data:
                        baseline_data[baseline_name] = []
                    baseline_data[baseline_name].append(baseline_values[key])
                    labels.append(baseline_name)
        if baseline_data:
            for baseline_name, values in baseline_data.items():
                baseline_std = np.std(np.asarray(values), axis=0).reshape(
                    (20, 1)
                )
                baseline_mean = np.mean(np.asarray(values), axis=0).reshape(
                    (20, 1)
                )
                data = np.concatenate([data, baseline_mean], axis=1)
                stds = np.concatenate([stds, baseline_std], axis=1)
        return data, stds, labels

    @staticmethod
    def create_plots_from_file(file_path: str):
        """
        This function reads the scores dictionary from a json file and then
        calls the main plotting function to create the plot.
        :param file_path: The path to the scores json file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        Plotter.create_plots(data)


if __name__ == "__main__":
    Plotter.create_plots_from_file("data/results.json")
