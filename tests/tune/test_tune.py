"""
Unit tests for meta/tune/tune.py.
"""

import os
import json
import itertools
from typing import Dict, Any, Tuple

from meta.tune.tune import tune, update_config
from meta.tune.params import get_iterations
from meta.tune.utils import tune_results_equal
from meta.utils.utils import save_dir_from_name
from tests.helpers import check_results_name


RANDOM_CONFIG_PATH = os.path.join("configs", "tune_random.json")
IC_GRID_CONFIG_PATH = os.path.join("configs", "tune_IC_grid.json")


def test_tune_random_metrics() -> None:
    """
    Runs hyperparameter random search and compares metrics against a saved baseline.
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "tune_random"

    # Run training.
    tune(config)


def test_tune_random_resume_metrics() -> None:
    """
    Runs hyperparameter random search and compares metrics against a saved baseline.
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config and resume training from interrupted checkpoint.
    save_name = "tune_random_interrupt"
    config["load_from"] = save_name
    config["base_train_config"]["baseline_metrics_filename"] = save_name
    tune(config)


def test_tune_random_resume_trial_metrics() -> None:
    """
    Runs hyperparameter random search and compares metrics against a saved baseline,
    resuming from a checkpoint in which some trials were completed for a given config,
    but not all (i.e. training was interrupted during train_single_config()).
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config and resume training from interrupted checkpoint.
    save_name = "tune_random_interrupt_trial"
    config["load_from"] = save_name
    config["base_train_config"]["baseline_metrics_filename"] = save_name
    tune(config)


def test_tune_IC_grid_metrics() -> None:
    """
    Runs hyperparameter IC grid search and compares metrics against a saved baseline.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "tune_IC_grid"

    # Run training.
    tune(config)


def test_tune_resume_IC_grid_metrics() -> None:
    """
    Resumes an interrupted hyperparameter IC grid search and compares metrics against a
    saved baseline.
    """

    # Load resume hyperparameter search config and resume training.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)
    config["load_from"] = "tune_IC_grid_interrupt"
    resumed_results = tune(config)

    # Load original hyperparameter search config and run original training from scratch.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)
    original_results = tune(config)

    # Check values.
    assert tune_results_equal(resumed_results, original_results)


def test_tune_IC_grid_values() -> None:
    """
    Runs hyperparameter IC grid search and makes sure that the correct parameter
    combinations are used for training.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Construct expected param value intervals.
    param_intervals = {
        "initial_lr": [1e-5, 1e-3],
        "clip_param": [0.2, 1.0],
        "num_layers": [1, 8],
        "recurrent": [True, False],
    }

    # Run training and extract actual configs from results.
    results = tune(config)
    actual_configs = [
        config_results["config"] for config_results in results["iterations"]
    ]

    # Verify configs from training.
    iteration = 0
    best_param_vals: Dict[str, float] = {}
    for param_name, param_values in param_intervals.items():

        param_fitnesses = {}

        for param_val in param_values:

            # Check values.
            for best_param_name, best_param_val in best_param_vals.items():
                found, result = dict_search(actual_configs[iteration], best_param_name)
                assert found
                assert result == best_param_val

            # Store fitnesses for comparison.
            param_fitnesses[param_val] = results["iterations"][iteration]["fitness"]
            iteration += 1

        best_param_val = dict_argmax(param_fitnesses)
        best_param_vals[param_name] = best_param_val


def dict_argmax(d: Dict[Any, Any]) -> Any:
    """ Compute argmax for a dictionary with numerical values. """

    argmax = None
    best_val = None
    for arg, val in d.items():
        if argmax is None or val > best_val:
            argmax = arg
            best_val = val

    return argmax


def dict_search(d: Dict[str, Any], key: str) -> Tuple[bool, Any]:
    """
    Recursively search through a dictionary to find the value corresponding to ``key``.
    """

    for k, v in d.items():
        if k == key:
            return True, v
        elif isinstance(v, dict):
            found, result = dict_search(v, key)
            if found:
                return True, result

    return False, None
