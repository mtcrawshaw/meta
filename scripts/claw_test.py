""" Script to evaluate CLAW as an approximation to CLW (see meta/train/loss.py). """

import os
import json

import numpy as np
import matplotlib.pyplot as plt


BASE_CONFIG_PATH = "configs/mt_regression_claw.json"
TRIAL_CONFIG_PATH = "configs/temp_claw_test.json"
CLAW_DATA_PATH = "data/claw_test_data.npy"
CLAW_LOG_PATH = "data/claw_log.txt"
CLAW_PLOT_PATH = "data/claw_test_plot.png"
NUM_TRIALS = 100
NUM_STEPS = 1000
START_STEP = 10
END_STEP = 1000


def main():
    """ Main function for run_claw_test.py. """

    # Make sure that data and log paths are clear.
    if os.path.isfile(CLAW_DATA_PATH):
        raise ValueError(f"File '{CLAW_DATA_PATH}' already exists.")
    if os.path.isfile(CLAW_LOG_PATH):
        raise ValueError(f"File '{CLAW_LOG_PATH}' already exists.")

    # Load base config.
    with open(BASE_CONFIG_PATH, "r") as base_config_file:
        base_config = json.load(base_config_file)

    # Modify base config to use `CLAWTester` loss weighter.
    base_config["num_updates"] = NUM_STEPS
    base_config["loss_weighter"] = {
        "type": "CLAWTester",
        "loss_weights": [1.0] * 10,
        "step_bounds": [START_STEP, END_STEP],
        "claw_data_path": CLAW_DATA_PATH,
        "claw_log_path": CLAW_LOG_PATH,
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

    # Clean up trial config.
    os.remove(TRIAL_CONFIG_PATH)

    # Plot results.
    weights = np.load(CLAW_DATA_PATH)
    weights = np.transpose(weights, [0, 2, 1])
    weights = np.reshape(weights, (-1, weights.shape[-1]))
    plt.scatter(weights[:, 0], weights[:, 1])
    plt.savefig(CLAW_PLOT_PATH)


if __name__ == "__main__":
    main()
