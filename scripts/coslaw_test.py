"""
Script to evaluate loss co-variation as an approximation to task loss gradient cosine
similarity in multi-task learning.
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


NUM_TRIALS = 10
NUM_TASKS = 32
NUM_STEPS = 900
NUM_SAVES = 10

BASE_CONFIG_PATH = "configs/pcba.json"
TRIAL_CONFIG_PATH = "configs/temp_coslaw_test.json"
DATA_PATH = "data/coloss_test_data.npy"
PLOT_PATH = "data/coloss_test_plot.png"

MARKER_SIZE = 12
SUBPLOT_SIZE = (4, 4)


def upper_triangle(arr: np.ndarray) -> np.ndarray:
    """ Return the flattened upper triangle of a 2-D square numpy array. """

    assert len(arr.shape) == 2
    m, n = arr.shape
    assert m == n
    flattened = np.zeros((n * (n - 1)) // 2)
    count = 0
    for i in range(n-1):
        k = n - i - 1
        flattened[count: count+k] = arr[i, i+1:]
        count += k

    return flattened


def collect_samples(lr: float = 3e-4):
    """
    Run training with CoLossWeighter and collect comparison of loss co-variation against
    gradient cosine similarity.
    """

    # Load base config.
    with open(BASE_CONFIG_PATH, "r") as base_config_file:
        base_config = json.load(base_config_file)

    # Modify base config for our case.
    save_steps = [((i+1) * NUM_STEPS) // NUM_SAVES - 1 for i in range(NUM_SAVES)]
    base_config["num_updates"] = NUM_STEPS
    base_config["loss_weighter"] = {
        "type": "CoLossTester",
        "loss_weights": None,
        "save_steps": save_steps,
        "data_path": DATA_PATH,
    }
    base_config["cuda"] = True
    base_config["lr"] = lr

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


def main(reuse: bool = False, lr: float = 3e-4):
    """ Main function for slaw_test.py. """

    # Make sure that data path is clear, if we aren't reusing old data.
    data_exists = os.path.isfile(DATA_PATH)
    plot_exists = os.path.isfile(PLOT_PATH)
    if reuse:
        if not data_exists:
            collect_samples(lr=lr)
    else:
        if data_exists:
            raise ValueError(f"File '{DATA_PATH}' already exists.")
        if plot_exists:
            raise ValueError(f"File '{PLOT_PATH}' already exists.")
        collect_samples(lr=lr)

    # Set plot font and figure size.
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["figure.figsize"] = (SUBPLOT_SIZE[0] * NUM_TRIALS, SUBPLOT_SIZE[1] * NUM_SAVES)

    # Read in collective data.
    task_similarity = np.load(DATA_PATH)
    assert task_similarity.shape[0] == NUM_TRIALS
    assert task_similarity.shape[1] == NUM_SAVES

    # Plot predicted task similarities.
    fig, ax = plt.subplots(NUM_TRIALS, NUM_SAVES)
    num_points = (NUM_TASKS * (NUM_TASKS - 1)) // 2
    for trial in range(NUM_TRIALS):
        for save in range(NUM_SAVES):
            cosine_similarity = upper_triangle(task_similarity[trial, save, 0])
            loss_covar = upper_triangle(task_similarity[trial, save, 1])
            assert cosine_similarity.shape == (num_points,)
            assert loss_covar.shape == (num_points,)
            c_iter = iter(cm.rainbow(np.linspace(0, 1, num_points)))
            c = [next(c_iter) for _ in range(num_points)]
            ax[trial, save].scatter(cosine_similarity, loss_covar, c=c, s=MARKER_SIZE)

    # Save plot.
    plt.savefig(PLOT_PATH, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Plot existing samples instead of creating new ones.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    args = parser.parse_args()

    main(reuse=args.reuse)
