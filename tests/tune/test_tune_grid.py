"""
Unit tests for grid search in meta/tune/tune.py.
"""

import os
import json
import itertools
from typing import Dict, Any, Tuple
from shutil import rmtree

from meta.tune.tune import tune
from meta.tune.params import update_config
from tests.tune.templates import resume_template


GRID_CONFIG_PATH = os.path.join("configs", "tune_grid.json")
GRID_VALUES_CONFIG_PATH = os.path.join("configs", "tune_grid_values.json")
GRID_TRIAL_CONFIG_PATH = os.path.join("configs", "tune_grid_trial.json")


def test_tune_grid_values() -> None:
    """
    Runs hyperparameter grid search and makes sure that the correct parameter
    combinations are used for training.
    """

    # Load hyperparameter search config.
    with open(GRID_VALUES_CONFIG_PATH, "r") as config_file:
        tune_config = json.load(config_file)

    # Construct expected parameter combinations.
    expected_configs = []
    variable_params = [
        "initial_lr",
        "final_lr",
        "clip_param",
        "recurrent",
    ]
    variable_param_combos = [
        [1e-5, 1e-6, 0.6, True],
        [1e-5, 1e-6, 0.6, False],
        [1e-4, 1e-6, 0.6, True],
        [1e-4, 1e-6, 0.6, False],
        [1e-3, 1e-6, 0.6, True],
        [1e-3, 1e-6, 0.6, False],
    ]
    for variable_param_combo in variable_param_combos:
        config = dict(tune_config["base_train_config"])
        updated_params = dict(zip(variable_params, variable_param_combo))
        config = update_config(config, updated_params)
        expected_configs.append(dict(config))

    # Run training and extract actual configs from results.
    results = tune(tune_config)
    actual_configs = [
        config_results["config"] for config_results in results["iterations"]
    ]

    # Compare actual configs with expected configs. We need to do this in a kind of
    # janky way, since we want to allow the possibility that the list of configs are in
    # a different order, but we can't make a set of dicts (dicts aren't hashable). We
    # test that the lists have the same order, that expected_configs contains unique
    # values, and that each element of expected_configs is an element of actual_configs.
    assert len(expected_configs) == len(actual_configs)
    assert all(c1 != c2 for c1, c2 in itertools.combinations(expected_configs, 2))
    for expected_config in expected_configs:
        assert expected_config in actual_configs


def test_tune_grid_metrics() -> None:
    """
    Runs hyperparameter grid search and compares metrics against a saved baseline.
    """

    # Load hyperparameter search config.
    with open(GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "tune_grid"

    # Run training.
    results = tune(config)


def test_tune_grid_early_stop_iteration() -> None:
    """
    Runs hyperparameter grid search until an early stop point between iterations.
    """

    # Load hyperparameter search config.
    with open(GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"iterations": 2, "trials": 0}

    # Run training.
    results = tune(config)

    # Check results.
    assert len(results["iterations"]) == config["early_stop"]["iterations"]
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_grid_early_stop_trial() -> None:
    """
    Runs hyperparameter grid search until an early stop point between trials of an
    iteration.
    """

    # Load hyperparameter search config.
    with open(GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"iterations": 3, "trials": 1}

    # Run training.
    results = tune(config)

    # Check results.
    assert len(results["iterations"]) == config["early_stop"]["iterations"]
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_grid_resume_iteration() -> None:
    """
    Runs partial training, saves a checkpoint between iterations, then resumes from
    checkpoint and finishes training, comparing results against a non-interrupted
    version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_iteration"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 4, "trials": 0}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_grid_resume_trial() -> None:
    """
    Runs partial training, saves a checkpoint between trials, then resumes from
    checkpoint and finishes training, comparing results against a non-interrupted
    version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_trial"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 4, "trials": 1}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_grid_resume_x2_iteration_x2() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_x2_iteration_x2"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 0}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_grid_resume_x2_trial_x2() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_x2_trial_x2"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 0}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_grid_resume_x2_iteration_trial() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_x2_iteration_trial"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 1}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_grid_resume_x2_trial_iteration() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. Then compares results against a non-interrupted version.
    """

    # Set up case.
    save_name = "test_tune_grid_resume_x2_trial_iteration"
    config_path = GRID_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 1}, {"iterations": 6, "trials": 0}]
    baseline_name = "tune_grid"
    results_name = "tune_grid"

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)
