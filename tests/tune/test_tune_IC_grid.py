"""
Unit tests for IC grid search in meta/tune/tune.py.
"""

import os
import json
import itertools
from typing import Dict, Any, Tuple

from meta.tune.tune import tune
from meta.tune.utils import tune_results_equal
from tests.tune.templates import resume_template


IC_GRID_CONFIG_PATH = os.path.join("configs", "tune_IC_grid.json")


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


def test_tune_IC_grid_early_stop_param() -> None:
    """
    Runs hyperparameter IC grid search until an early stop point between params.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"param_num": 2, "val_num": 0, "trials": 0}

    # Run training.
    results = tune(config)

    # Check results.
    param_iterations = (
        lambda p: p["num_values"] if "num_values" in p else len(p["choices"])
    )
    expected_iterations = 0
    param_names = list(config["search_params"].keys())
    for i in range(config["early_stop"]["param_num"]):
        expected_iterations += param_iterations(config["search_params"][param_names[i]])
    assert len(results["iterations"]) == expected_iterations
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_IC_grid_early_stop_iteration() -> None:
    """
    Runs hyperparameter IC grid search until an early stop point between iterations.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"param_num": 1, "val_num": 1, "trials": 0}

    # Run training.
    results = tune(config)

    # Check results.
    param_iterations = (
        lambda p: p["num_values"] if "num_values" in p else len(p["choices"])
    )
    expected_iterations = 0
    param_names = list(config["search_params"].keys())
    for i in range(config["early_stop"]["param_num"]):
        expected_iterations += param_iterations(config["search_params"][param_names[i]])
    expected_iterations += config["early_stop"]["val_num"]
    assert len(results["iterations"]) == expected_iterations
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_IC_grid_early_stop_trial() -> None:
    """
    Runs hyperparameter IC grid search until an early stop point between trials of an
    iteration.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"param_num": 2, "val_num": 1, "trials": 1}

    # Run training.
    results = tune(config)

    # Check results.
    param_iterations = (
        lambda p: p["num_values"] if "num_values" in p else len(p["choices"])
    )
    expected_iterations = 0
    param_names = list(config["search_params"].keys())
    for i in range(config["early_stop"]["param_num"]):
        expected_iterations += param_iterations(config["search_params"][param_names[i]])
    expected_iterations += config["early_stop"]["val_num"]
    assert len(results["iterations"]) == expected_iterations
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_IC_grid_resume_iteration() -> None:
    """
    Runs partial training, saves a checkpoint between iterations, then resumes from
    checkpoint and finishes training, comparing results against a non-interrupted
    version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_iteration"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [{"param_num": 1, "val_num": 1, "trials": 0}]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_IC_grid_resume_trial() -> None:
    """
    Runs partial training, saves a checkpoint between trials, then resumes from
    checkpoint and finishes training, comparing results against a non-interrupted
    version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_trial"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [{"param_num": 2, "val_num": 1, "trials": 1}]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_IC_grid_resume_x2_iteration_x2() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_x2_iteration_x2"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [
        {"param_num": 1, "val_num": 0, "trials": 0},
        {"param_num": 2, "val_num": 1, "trials": 0},
    ]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_IC_grid_resume_x2_trial_x2() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_x2_trial_x2"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [
        {"param_num": 0, "val_num": 1, "trials": 1},
        {"param_num": 1, "val_num": 0, "trials": 1},
    ]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_IC_grid_resume_x2_iteration_trial() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_x2_iteration_trial"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [
        {"param_num": 1, "val_num": 0, "trials": 0},
        {"param_num": 2, "val_num": 0, "trials": 1},
    ]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_IC_grid_resume_x2_trial_iteration() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_IC_grid_resume_x2_trial_iteration"
    config_path = IC_GRID_CONFIG_PATH
    early_stops = [
        {"param_num": 0, "val_num": 1, "trials": 1},
        {"param_num": 2, "val_num": 0, "trials": 0},
    ]
    baseline_name = "tune_IC_grid"
    results_name = "tune_IC_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


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
