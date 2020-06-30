"""
Unit tests for meta/hyperparameter_search.py.
"""

import os
import json
import itertools
from typing import Dict, Any

from meta.hyperparameter_search import hyperparameter_search


RANDOM_CONFIG_PATH = os.path.join("configs", "hp_random.json")
GRID_CONFIG_PATH = os.path.join("configs", "hp_grid.json")
GRID_TEST_CONFIG_PATH = os.path.join("configs", "hp_grid_test.json")
IC_GRID_CONFIG_PATH = os.path.join("configs", "hp_IC_grid.json")


def test_hp_search_random_metrics() -> None:
    """
    Runs hyperparameter random search and compares metrics against a saved baseline for
    the Cartpole environment.
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "hp_random"

    # Run training.
    hyperparameter_search(config)


def test_hp_search_grid_metrics() -> None:
    """
    Runs hyperparameter grid search and compares metrics against a saved baseline for
    the LunarLanderContinuous environment.
    """

    # Load hyperparameter search config.
    with open(GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "hp_grid"

    # Run training.
    hyperparameter_search(config)


def test_hp_search_grid_values() -> None:
    """
    Runs hyperparameter grid search and makes sure that the correct parameter
    combinations are used for training.
    """

    # Load hyperparameter search config.
    with open(GRID_TEST_CONFIG_PATH, "r") as config_file:
        hp_config = json.load(config_file)

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
        config = dict(hp_config["base_train_config"])
        updated_params = dict(zip(variable_params, variable_param_combo))
        config.update(updated_params)
        expected_configs.append(dict(config))

    # Run training and extract actual configs from results.
    results = hyperparameter_search(hp_config)
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


def test_hp_search_IC_grid_metrics() -> None:
    """
    Runs hyperparameter IC grid search and compares metrics against a saved baseline for
    the Cartpole environment.
    """

    # Load hyperparameter search config.
    with open(IC_GRID_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config.
    config["base_train_config"]["baseline_metrics_filename"] = "hp_IC_grid"

    # Run training.
    hyperparameter_search(config)


def test_hp_search_IC_grid_values() -> None:
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
    results = hyperparameter_search(config)
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
            assert actual_configs[iteration][param_name] == param_val
            for best_param_name, best_param_val in best_param_vals.items():
                assert actual_configs[iteration][best_param_name] == best_param_val

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
