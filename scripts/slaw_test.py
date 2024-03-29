""" Script to evaluate SLAW as an approximation to SLW (see meta/train/loss.py). """

import os
import json
import argparse
import random
from math import sqrt, floor, ceil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


BASE_CONFIG_PATH = "configs/mtregression_slaw.json"
TRIAL_CONFIG_PATH = "configs/temp_slaw_test.json"
SLAW_DATA_PATH = "data/slaw_test_data.npy"
SLAW_LOG_PATH = "data/slaw_log.txt"
SLAW_PLOT_PATH = "data/slaw_test_plot.png"
NUM_TASKS = 10
NUM_TRIALS = 120
NUM_STEPS = 1000
START_STEP = 100
END_STEP = 1000
MARKERS = [".", "v", "^", "s", "+", "x", "|", "_", "P", "1"]
MARKER_SIZE = 12


def collect_samples():
    """ Run training with SLAW and collect estimated and true gradient norms. """

    # Load base config.
    with open(BASE_CONFIG_PATH, "r") as base_config_file:
        base_config = json.load(base_config_file)

    # Modify base config to use `SLAWTester` loss weighter.
    base_config["num_updates"] = NUM_STEPS
    base_config["loss_weighter"] = {
        "type": "SLAWTester",
        "loss_weights": [1.0] * NUM_TASKS,
        "step_bounds": [START_STEP, END_STEP],
        "slaw_data_path": SLAW_DATA_PATH,
        "slaw_log_path": SLAW_LOG_PATH,
    }

    # Run trials.
    for trial in range(NUM_TRIALS):

        # Write out trial config.
        trial_config = dict(base_config)
        trial_config["seed"] = trial
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
    data_exists = os.path.isfile(SLAW_DATA_PATH)
    log_exists = os.path.isfile(SLAW_LOG_PATH)
    plot_exists = os.path.isfile(SLAW_PLOT_PATH)
    if reuse:
        if not data_exists:
            collect_samples()
    else:
        if data_exists:
            raise ValueError(f"File '{SLAW_DATA_PATH}' already exists.")
        if log_exists:
            raise ValueError(f"File '{SLAW_LOG_PATH}' already exists.")
        if plot_exists:
            raise ValueError(f"File '{SLAW_PLOT_PATH}' already exists.")

        collect_samples()

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
