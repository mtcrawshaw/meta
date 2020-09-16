"""
Unit tests for random search in meta/tune/tune.py.
"""

import os
import json

from meta.tune.tune import tune
from tests.tune.templates import resume_template


RANDOM_CONFIG_PATH = os.path.join("configs", "tune_random.json")


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


def test_tune_random_early_stop_iteration() -> None:
    """
    Runs hyperparameter random search until an early stop point between iterations.
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"iterations": 2, "trials": 0}

    # Run training.
    results = tune(config)

    # Check results.
    assert len(results["iterations"]) == config["early_stop"]["iterations"]
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_random_early_stop_trial() -> None:
    """
    Runs hyperparameter random search until an early stop point between trials of an
    iteration.
    """

    # Load hyperparameter search config.
    with open(RANDOM_CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)

    # Modify default training config to stop early.
    config["early_stop"] = {"iterations": 3, "trials": 1}

    # Run training.
    results = tune(config)

    # Check results.
    assert len(results["iterations"]) == config["early_stop"]["iterations"]
    for config_results in results["iterations"]:
        assert len(config_results["trials"]) == config["trials_per_config"]


def test_tune_random_resume_iteration() -> None:
    """
    Runs partial training, saves a checkpoint between iterations, then resumes from
    checkpoint and finishes training. We can't compare against a non-interrupted
    version, because the random decisions will be different. This test just ensures that
    saving/loading runs without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_iteration"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 1, "trials": 0}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_random_resume_trial() -> None:
    """
    Runs partial training, saves a checkpoint between trials, then resumes from
    checkpoint and finishes training. We can't compare against a non-interrupted
    version, because the random decisions will be different. This test just ensures that
    saving/loading runs without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_trial"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 1, "trials": 1}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_random_resume_x2_iteration_x2() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. We can't compare against a non-interrupted version, because the
    random decisions will be different. This test just ensures that saving/loading runs
    without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_x2_iteration_x2"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 0}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_random_resume_x2_trial_x2() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. We can't compare against a non-interrupted version, because the
    random decisions will be different. This test just ensures that saving/loading runs
    without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_x2_trial_x2"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 0}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_random_resume_x2_iteration_trial() -> None:
    """
    Runs partial training, saves checkpoint between iterations, resumes from checkpoint,
    runs more training, saves checkpoint between trials, resumes from checkpoint and
    finishes training. We can't compare against a non-interrupted version, because the
    random decisions will be different. This test just ensures that saving/loading runs
    without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_x2_iteration_trial"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 0}, {"iterations": 6, "trials": 1}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)


def test_tune_random_resume_x2_trial_iteration() -> None:
    """
    Runs partial training, saves checkpoint between trials, resumes from checkpoint,
    runs more training, saves checkpoint between iterations, resumes from checkpoint and
    finishes training. We can't compare against a non-interrupted version, because the
    random decisions will be different. This test just ensures that saving/loading runs
    without crashing.
    """

    # Set up case.
    save_name = "test_tune_random_resume_x2_trial_iteration"
    config_path = RANDOM_CONFIG_PATH
    early_stops = [{"iterations": 3, "trials": 1}, {"iterations": 6, "trials": 0}]
    baseline_name = None
    results_name = None

    # Call template.
    resume_template(save_name, config_path, early_stops, baseline_name, results_name)
