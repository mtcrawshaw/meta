"""
Script to make plots for the loss weight quality experiments of the "Simplifying Loss
Weighting" paper.
"""

import os
import argparse
import json
import glob
from typing import Dict, Any

import matplotlib.pyplot as plt


DATASETS = [f"MTRegression{n}" for n in [2, 10, 20, 30, 40, 50]]
NUM_TASKS = {
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
METHOD_REPLACEMENTS = {
    "GradNormZero": "GN-Zero",
    "CLAW": "SLAW",
}
SETTINGS = {
    "Constant": ("black", "."),
    "IdealConstant": ("magenta", "v"),
    "GradNorm": ("red", "+"),
    "DWA": ("green", "^"),
    "CLAW": ("blue", "x"),
}
MARK_INTERVAL = 250

def file_exists(path: str) -> bool:
    """ Utility function to check whether file exists and print error message if so. """
    exists = os.path.isfile(path)
    if exists:
        print(
            f"File \"{path}\" already exists."
            " Delete it in order to run this script."
        )
    return exists


def plot_normalized_loss(metrics: Dict[str, Any], config: Dict[str, Any], plot_path: str) -> None:
    """
    Plot test normalized loss from `metrics` and `config`, saving result to `plot_path`.
    """

    methods = ["Constant", "IdealConstant", "GradNorm", "DWA", "CLAW"]
    for method in methods:
        assert method in config["methods"]

    # Plot results from each method.
    num_steps = None
    legend = []
    for method in methods:

        test_nl = metrics[method]["mean"]["eval_normal_loss"]["history"]

        # Check that number of steps per method is consistent.
        if num_steps is None:
            num_steps = len(test_nl)
        else:
            assert len(test_nl) == num_steps

        # Plot.
        color, marker = SETTINGS[method]
        label = METHOD_REPLACEMENTS[method] if method in METHOD_REPLACEMENTS else method
        plt.plot(list(range(1, num_steps + 1)), test_nl, color=color, marker=marker, label=label, markevery=MARK_INTERVAL)
        legend.append(label)

    # Title plot and axes, and create legend.
    plt.title("MTRegression10 - Test Normalized Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Normalized Loss")
    plt.legend(legend, loc="upper right")

    # Set axis ranges.
    plt.xlim([0, 300])
    plt.ylim([12, 30])

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def plot_weight_quality(metrics: Dict[str, Any], config: Dict[str, Any], plot_path: str) -> None:
    """
    Plot loss weight quality from `metrics` and `config`, saving result to `plot_path`.
    """

    methods = ["GradNorm", "DWA", "CLAW"]
    for method in methods:
        assert method in config["methods"]

    # Plot results from each method.
    num_steps = None
    legend = []
    for method in methods:

        weight_error = metrics[method]["mean"]["loss_weight_error"]["history"]

        # Check that number of steps per method is consistent.
        if num_steps is None:
            num_steps = len(weight_error)
        else:
            assert len(weight_error) == num_steps

        # Plot.
        color, marker = SETTINGS[method]
        label = METHOD_REPLACEMENTS[method] if method in METHOD_REPLACEMENTS else method
        plt.plot(list(range(1, num_steps + 1)), weight_error, color=color, marker=marker, label=label, markevery=MARK_INTERVAL)
        legend.append(label)

    # Title plot and axes, and create legend.
    plt.title("MTRegression10 - Loss Weight Error")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss Weight Error (MSE)")
    plt.legend(legend, loc="right")

    # Set axis ranges.
    plt.xlim([0, 2000])

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def main(results_dir: str, overwrite: bool = False) -> None:
    """
    Main function for make_quality_plots.py. Makes a plot showing the normalized loss
    during training and another plot showing the loss weight quality during training.
    """

    # Check that target file doesn't already exist, if necessary.
    loss_plot_path = os.path.join(results_dir, "paper_loss_plot.png")
    quality_plot_path = os.path.join(results_dir, "paper_quality_plot.png")
    paths = [loss_plot_path, quality_plot_path]
    if not overwrite:
        exists = any([file_exists(path) for path in paths])
        if exists:
            exit()

    # Read in metrics from results directory.
    metrics_paths = glob.glob(os.path.join(results_dir, "*_metrics.json"))
    assert len(metrics_paths) == 1
    metrics_path = metrics_paths[0]
    with open(metrics_path, "rb") as metrics_file:
        metrics = json.load(metrics_file)

    # Read in config from results dictionary.
    config_paths = glob.glob(os.path.join(results_dir, "*_config.json"))
    assert len(config_paths) == 1
    config_path = config_paths[0]
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    dataset = config["base_train_config"]["dataset"]
    assert dataset in DATASETS

    # Set plot font and figure size.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["figure.figsize"] = (3.2, 2.4)

    # Make plots for normalized loss and loss weight error.
    plot_normalized_loss(metrics, config, loss_plot_path)
    plot_weight_quality(metrics, config, quality_plot_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path to results directory to plot from.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing plot file, if necessary.")
    args = parser.parse_args()

    main(args.results_dir, overwrite=args.overwrite)
