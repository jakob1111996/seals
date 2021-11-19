# This file implements the plotting functionality to recreate the plot.
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.data_manager import DataManager
from src.definitions import cleaned_train_annotations_file


class Plotter:
    """
    Class that handles all the plotting. All methods are static.
    """

    @staticmethod
    def create_plot(
        scores: Dict,
        ax: plt.axis,
        title: str,
        key: str,
        colors: List[str],
        class_counts: Dict[str, int],
    ) -> None:
        """
        This function creates a plot with all required lines and confidence
        intervals on one axis of the figure.
        :param scores: The scores dictionary to get the scores from.
        :param ax: The ax used for the plotting
        :param title: The title used for the y label of the plot
        :param key: The score name we are plotting in this plot
        :param colors: A list of colors to use for the plot
        :param class_counts: The true number of positives for each class.
        """
        is_recall = key == "recall"
        data = []
        for class_name, class_scores in scores.items():
            if is_recall:
                data.append(
                    list(
                        np.array(class_scores["positives"])
                        / class_counts[class_name[:-2]]
                    )
                )
            else:
                data.append(class_scores[key])
        data = np.mean(np.asarray(data), axis=0).reshape((20, 1))
        rep_data = {}
        for class_name, class_scores in scores.items():
            if class_name[-1:] not in rep_data:
                rep_data[class_name[-1:]] = []
            rep_data[class_name[-1:]].append(class_scores[key])
        class_means = np.empty((20, 0))
        for class_name, values in rep_data.items():
            class_means = np.concatenate(
                [
                    class_means,
                    np.mean(np.asarray(values), axis=0).reshape((20, 1)),
                ],
                axis=1,
            )
        stds = np.std(class_means, axis=1).reshape((20, 1))
        data, stds, labels = Plotter.add_baseline_data(
            data, stds, scores, key, class_counts
        )
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
        class_counts, _ = DataManager.get_positives_for_classes(
            cleaned_train_annotations_file
        )
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        Plotter.create_plot(
            scores,
            axes[0][0],
            "mAP",
            "average_precision",
            colors,
            class_counts,
        )
        Plotter.create_plot(
            scores, axes[0][1], "Pool Size", "pool_size", colors, class_counts
        )
        Plotter.create_plot(
            scores, axes[1][0], "Recall", "recall", colors, class_counts
        )
        Plotter.create_plot(
            scores, axes[1][1], "Positives", "positives", colors, class_counts
        )
        plt.tight_layout()
        plt.savefig("data/results.png")
        plt.show()

    @staticmethod
    def add_baseline_data(
        data: np.ndarray,
        stds: np.ndarray,
        scores: Dict,
        key: str,
        class_counts: Dict[str, int],
    ):
        """
        This function extracts the scores from the baseline algorithms
        for plotting them later on.
        :param data: The SEALS data ready for plotting in an array
        :param stds: The SEALS standard deviation ready for plotting
        :param scores: The scores dictionary
        :param key: The name of the score to extract right now.
        :param class_counts: The true number of positives for each class
        :return: Tuple with three elements:
            0: The SEALS data concatenated with the baseline data for plotting
                in shape (20, 1 + n_baselines)
            1: The SEALS stdd concatenated with the baseline stdds
            2: The labels of SEALS and the baselines for the legend.
        """
        is_recall = key == "recall"
        labels = ["MaxEnt-SEALS"]
        baseline_data = {}
        for class_name, class_scores in scores.items():
            baseline_dict = class_scores["baselines"]
            for baseline_name, baseline_values in baseline_dict.items():
                if baseline_name not in baseline_data:
                    baseline_data[baseline_name] = {}
                if key in baseline_values:
                    if class_name[-1:] not in baseline_data[baseline_name]:
                        baseline_data[baseline_name][class_name[-1:]] = []
                    if is_recall:
                        baseline_data[baseline_name][class_name[-1:]].append(
                            list(
                                np.array(baseline_values["positives"])
                                / class_counts[class_name[:-2]]
                            )
                        )
                    else:
                        baseline_data[baseline_name][class_name[-1:]].append(
                            baseline_values[key]
                        )
        # Get mean and std for every class separately
        baseline_means = {}
        baseline_stds = {}
        for baseline_name, baseline_values in baseline_data.items():
            baseline_means[baseline_name] = np.empty((20, 0))
            baseline_stds[baseline_name] = np.empty((20, 0))
            for repetition, values in baseline_values.items():
                baseline_mean = np.mean(np.asarray(values), axis=0).reshape(
                    (20, 1)
                )
                baseline_means[baseline_name] = np.concatenate(
                    [baseline_means[baseline_name], baseline_mean], axis=1
                )
            if baseline_means[baseline_name].shape[1] != 0:
                data = np.concatenate(
                    [
                        data,
                        np.mean(baseline_means[baseline_name], axis=1).reshape(
                            (20, 1)
                        ),
                    ],
                    axis=1,
                )
                stds = np.concatenate(
                    [
                        stds,
                        np.std(baseline_means[baseline_name], axis=1).reshape(
                            (20, 1)
                        ),
                    ],
                    axis=1,
                )
                labels.append(baseline_name)
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
