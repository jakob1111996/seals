import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Class that handles all the plotting.
    """

    @staticmethod
    def create_plot(
        scores: Dict, ax: plt.axis, title: str, key: str, colors: List[str]
    ):
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
        with open(file_path, "r") as file:
            data = json.load(file)
        Plotter.create_plots(data)


if __name__ == "__main__":
    Plotter.create_plots_from_file("data/results.json")
