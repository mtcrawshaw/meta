"""
Script to make plots for the efficiency experiments of the "Simplifying Loss Weighting"
paper.
"""

import os
import argparse
import json
import glob
from typing import Dict, List, Any

import matplotlib.pyplot as plt


NUM_TASKS = [10, 20, 30, 40, 50]
METHODS = ["Constant", "GradNorm", "CLAW"]
SETTINGS = {
    "Constant": ("black", "."),
    "GradNorm": ("red", "+"),
    "CLAW": ("blue", "x"),
}

def file_exists(path: str) -> bool:
    """ Utility function to check whether file exists and print error message if so. """
    exists = os.path.isfile(path)
    if exists:
        print(
            f"File \"{path}\" already exists."
            " Delete it in order to run this script."
        )
    return exists


def plot_train_time(training_time: Dict[str, List[float]], plot_path: str) -> None:
    """ Plot training time against number of tasks, saving result to `plot_path`. """

    # Plot results from each method in each dataset.
    legend = []
    for method in METHODS:
        color, marker = SETTINGS[method]
        plt.plot(NUM_TASKS, training_time[method], color=color, marker=marker, label=method)
        legend.append(method)

    # Title plot and axes, and create legend.
    plt.title("MTRegression - Training Time")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Time per Training Step (seconds)")
    plt.legend(legend, loc="upper left")

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def plot_normalized_loss(normalized_loss: Dict[str, List[float]], plot_path: str) -> None:
    """ Plot normalized loss against number of tasks, saving result to `plot_path`. """

    # Plot results from each method in each dataset.
    legend = []
    for method in METHODS:
        color, marker = SETTINGS[method]
        plt.plot(NUM_TASKS, normalized_loss[method], color=color, marker=marker, label=method)
        legend.append(method)

    # Title plot and axes, and create legend.
    plt.title("MTRegression - Test Normalized Loss")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Normalized Loss")
    plt.legend(legend, loc="upper left")

    # Set axis bounds.
    plt.ylim([0.0, 1.1 * max([max(normalized_loss[method]) for method in METHODS])])

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def main(results_dir_prefix: str, overwrite: bool = False) -> None:
    """ 
    Main function for make_efficiency_plots.py. Makes a plot showing the time per
    training step of each method vs. the number of tasks, and another plot showing the
    test normalized loss of each method vs. the number of tasks.
    """

    # Check that target file doesn't already exist, if necessary.
    time_plot_path = "task_time_plot.png"
    loss_plot_path = "task_loss_plot.png"
    paths = [time_plot_path, loss_plot_path]
    if not overwrite:
        exists = any([file_exists(path) for path in paths])
        if exists:
            exit()

    # Read in training time and normalized loss from each experiment.
    training_time = {method: [] for method in METHODS}
    normalized_loss = {method: [] for method in METHODS}
    for num_tasks in NUM_TASKS:

        # Construct results directory.
        results_dir = f"{results_dir_prefix}{num_tasks}"

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
        assert dataset == f"MTRegression{num_tasks}"

        # Store training time and normalized loss for each method.
        for method in METHODS:
            method_metrics = metrics[method]["mean"]
            training_time[method].append(method_metrics["train_step_time"]["mean"][-1])
            normalized_loss[method].append(method_metrics["eval_normal_loss"]["best"])

    # Set plot font and figure size.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["figure.figsize"] = (3.2, 2.4)

    # Make plots for normalized loss and loss weight error.
    plot_train_time(training_time, time_plot_path)
    plot_normalized_loss(normalized_loss, loss_plot_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir_prefix",
        type=str,
        help="Common prefix of all MTRegression results directories from experiments to"
        " plot. Each experiment's results directory should be equal to the number of"
        " tasks appended to this common prefix."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing plot file, if necessary.")
    args = parser.parse_args()

    main(args.results_dir_prefix, overwrite=args.overwrite)
