"""
Script to make plots for the efficiency experiments of the "Simplifying Loss Weighting"
paper.
"""

import os
import argparse
import json
import glob
import csv
from typing import Dict, List, Any

import matplotlib.pyplot as plt


NUM_TASKS = [32, 64, 96, 128]
NUM_DATASETS = len(NUM_TASKS)
METHODS = ["Constant", "GradNorm", "PCGrad", "SLAW"]
NUM_METHODS = len(METHODS)
SETTINGS = {
    "Constant": ("black", "."),
    "GradNorm": ("red", "+"),
    "PCGrad": ("gold", "v"),
    "SLAW": ("blue", "x"),
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
    plt.title("PCBA - Training Time")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Time per Training Step (seconds)")
    plt.legend(legend, loc="upper left")

    # Set x-ticks.
    ax = plt.gca()
    ax.set_xticks(NUM_TASKS)

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def plot_average_precision(average_precision: Dict[str, List[float]], plot_path: str) -> None:
    """ Plot average precision against number of tasks, saving result to `plot_path`. """

    # Plot results from each method in each dataset.
    legend = []
    for method in METHODS:
        color, marker = SETTINGS[method]
        plt.plot(NUM_TASKS, average_precision[method], color=color, marker=marker, label=method)
        legend.append(method)

    # Title plot and axes, and create legend.
    plt.title("PCBA - Average Precision (Test)")
    plt.xlabel("Number of Tasks")
    plt.ylabel("Average Precision")
    plt.legend(legend, loc="upper right")

    # Set axis bounds and x-ticks.
    ax = plt.gca()
    min_ap = min([min(average_precision[method]) for method in METHODS])
    max_ap = max([max(average_precision[method]) for method in METHODS])
    plt.ylim([0.9 * min_ap, 1.15 * max_ap])
    ax.set_xticks(NUM_TASKS)

    # Save plot and exit.
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def main(results_path: str, overwrite: bool = False) -> None:
    """ 
    Main function for make_efficiency_plots.py. Makes a plot showing the time per
    training step of each method vs. the number of tasks, and another plot showing the
    test average precision loss of each method vs. the number of tasks.
    """

    # Check that target file doesn't already exist, if necessary.
    time_plot_path = "task_time_plot.png"
    loss_plot_path = "task_loss_plot.png"
    paths = [time_plot_path, loss_plot_path]
    if not overwrite:
        exists = any([file_exists(path) for path in paths])
        if exists:
            exit()

    # Read in training time and average precision from each experiment.
    with open(results_path, "r") as results_file:
        results = csv.reader(results_file)
        rows = list(results)
        training_time = {METHODS[m_idx]: [float(rows[d_idx][m_idx]) for d_idx in range(NUM_DATASETS)] for m_idx in range(NUM_METHODS)}
        average_precision = {METHODS[m_idx]: [100 * float(rows[d_idx+NUM_DATASETS][m_idx]) for d_idx in range(NUM_DATASETS)] for m_idx in range(NUM_METHODS)}

    # Set plot font and figure size.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["figure.figsize"] = (3.2, 2.4)

    # Make plots for average precision and loss weight error.
    plot_train_time(training_time, time_plot_path)
    plot_average_precision(average_precision, loss_plot_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_path",
        type=str,
        help="Path to CSV file containing results that will be plotted.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing plot file, if necessary.")
    args = parser.parse_args()

    main(args.results_path, overwrite=args.overwrite)
