"""
Script to plot learned loss weights from a multi-task experiment. Note that this script
must be moved to the root of the `meta` repository in order to run it, since we need to
use instances of `Metrics` objects which are defined in the `meta` package.
"""

import os
import argparse
import pickle
import json
import glob
from typing import List

import numpy as np
import matplotlib.pyplot as plt


DATASETS = ["NYUv2"] + [f"MTRegression{n}" for n in [2, 10, 20, 30, 40, 50]]
NUM_TASKS = {
    "NYUv2": 3,
    "MTRegression2": 2,
    "MTRegression10": 10,
    "MTRegression20": 20,
    "MTRegression30": 30,
    "MTRegression40": 40,
    "MTRegression50": 50,
}
SCALES = {
    2: [1, 10],
    **{num_tasks: list(range(1, num_tasks + 1)) for num_tasks in [10, 20, 30, 40, 50]},
}


def main(results_dir: str, methods: List[str]) -> None:
    """ Main function for plot_loss_weights.py. """

    # Check that target file doesn't already exist.
    plot_path = os.path.join(results_dir, "loss_weights_plot.png")
    if os.path.isfile(plot_path):
        print(
            f"File \"{plot_path}\" already exists."
            " Delete it in order to run this script."
        )

    # Read in metrics from results directory.
    checkpoint_paths = glob.glob(os.path.join(results_dir, "checkpoint.pkl"))
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    metrics = checkpoint["metrics"]

    # Read in config from results dictionary.
    config_paths = glob.glob(os.path.join(results_dir, "*_config.json"))
    assert len(config_paths) == 1
    config_path = config_paths[0]
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    dataset = config["base_train_config"]["dataset"]
    assert dataset in DATASETS
    num_tasks = NUM_TASKS[dataset]

    # Plot ideal loss weights, if possible.
    if dataset in SCALES:
        scales = SCALES[dataset]
        ideal_weights = 1.0 / (np.array(scales) ** 2)
        ideal_weights *= num_tasks / np.sum(ideal_weights)
        pass

    # Plot learned loss weights.
    num_steps = None
    for method in methods:
        for task in range(num_tasks):
            weight_metric = metrics[method]["mean"].metric_dict[f"loss_weight_{task}"]
            weight_vals = weight_metric.history
            
            # Check that number of steps per method is consistent.
            if num_steps is None:
                num_steps = len(weight_vals)
            else:
                assert len(weight_vals) == num_steps

            plt.plot(list(range(1, num_steps + 1)), weight_vals, label=f"{method}_{task}")

    # Save plot and exit.
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path to results directory to plot from.")
    parser.add_argument("methods", type=str, help="Comma-separated list of methods for which to plot loss weights.")
    args = parser.parse_args()

    main(args.results_dir, args.methods.split(","))
