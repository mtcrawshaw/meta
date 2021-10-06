"""
Script to evaluate loss co-variation as an approximation to task loss gradient cosine
similarity in multi-task learning.
"""

import os
import json
import argparse
import random
from math import sqrt, floor, ceil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


BASE_CONFIG_PATH = "configs/pcba_coslaw.json"
TRIAL_CONFIG_PATH = "configs/temp_coslaw_test.json"
DATA_PATH = "data/coloss_test_data.npy"
PLOT_PATH = "data/coloss_test_plot.png"
MARKERS = [".", "v", "^", "s", "+", "x", "|", "_", "P", "1"]
NUM_TRIALS = 10
MARKER_SIZE = 12


def collect_samples():
    """
    Run training with CoLossWeighter and collect comparison of loss co-variation against
    gradient cosine similarity.
    """

    # Load base config.
    with open(BASE_CONFIG_PATH, "r") as base_config_file:
        base_config = json.load(base_config_file)

    # Run trials.
    for trial in range(NUM_TRIALS):

        # Write out trial config.
        trial_config = dict(base_config)
        trial_config["seed"] = trial
        trial_config["loss_weighter"]["save_name"] = f"coloss_test_{trial}"
        with open(TRIAL_CONFIG_PATH, "w") as f:
            json.dump(trial_config, f, indent=4)

        # Call command to run trial.
        os.system(f"python3 main.py train {TRIAL_CONFIG_PATH}")
        print("")

    # Clean up trial config.
    os.remove(TRIAL_CONFIG_PATH)


def main(reuse: bool = False):
    """ Main function for slaw_test.py. """

    # Make sure that data and log paths are clear, if necessary.
    data_exists = os.path.isfile(DATA_PATH)
    plot_exists = os.path.isfile(PLOT_PATH)
    if reuse:
        if not data_exists:
            collect_samples()
    else:
        if data_exists:
            raise ValueError(f"File '{DATA_PATH}' already exists.")
        if plot_exists:
            raise ValueError(f"File '{PLOT_PATH}' already exists.")

        collect_samples()

    # TEMP
    exit()

    # Set plot font and figure size.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["figure.figsize"] = (3.2, 2.4)

    # Plot results.
    weights = np.load(SLAW_DATA_PATH)
    weights = np.transpose(weights, [2, 0, 1])
    num_tasks = weights.shape[0]
    num_samples = weights.shape[1]
    cmap = plt.get_cmap("tab10")
    for sample in range(num_samples):
        task = random.randrange(num_tasks)
        color = cmap(task)
        marker = MARKERS[task]
        plt.scatter(weights[task, sample, 0], weights[task, sample, 1], c=color, marker=marker, s=MARKER_SIZE)

    # Title plot and axes.
    plt.title("Loss Weight Estimation by SLAW")
    plt.xlabel(r"$w_i$ (Equation 4)", usetex=True)
    plt.ylabel(r"$w_i$ (SLAW, Equation 9)", usetex=True)

    # Transform axis scales.
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    x_vals = weights[:, :, 0].flatten()
    y_vals = weights[:, :, 1].flatten()
    for axis, vals in zip(["x", "y"], [x_vals, y_vals]):
        logs = np.log(vals) / np.log(2)
        low = floor(float(logs.min()))
        high = ceil(float(logs.max()))
        ticks = [2 ** p for p in range(low, high + 1)]
        if axis == "x":
            ax.set_xticks(ticks)
        else:
            ax.set_yticks(ticks)

    # Save plot.
    plt.savefig(SLAW_PLOT_PATH, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Plot existing samples instead of creating new ones.")
    args = parser.parse_args()

    main(reuse=args.reuse)
