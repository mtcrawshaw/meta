""" Test templates for tests/tune. """

import os
import json
from typing import Dict, List
from shutil import rmtree

from meta.tune.tune import tune
from meta.tune.params import get_iterations, get_num_param_values
from meta.tune.utils import (
    check_name_uniqueness,
    get_experiment_names,
    tune_results_equal,
)
from meta.utils.utils import save_dir_from_name, METRICS_DIR


def resume_template(
    save_name: str,
    config_path: str,
    early_stops: List[Dict[str, int]],
    baseline_name: str,
    results_name: str,
) -> None:
    """
    Runs while stopping to save/load at a given set of checkpoints, then compares
    results against non-interrupted version.
    """

    # Load hyperparameter search config.
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    config["base_train_config"]["save_name"] = save_name

    # Set baseline to compare against throughout training.
    config["base_train_config"]["baseline_metrics_filename"] = baseline_name

    # Ensure that there are no existing saved experiments whose names coincide with the
    # experiment names used here. We do have to save and load from disk so we want to
    # make sure that we aren't overwriting any previously existing files.
    num_param_values = None
    if config["search_type"] == "IC_grid":
        num_param_values = get_num_param_values(config["search_params"])
    iterations = get_iterations(
        config["search_type"], config["search_iterations"], config["search_params"]
    )
    check_name_uniqueness(
        save_name,
        config["search_type"],
        iterations,
        config["trials_per_config"],
        num_param_values=num_param_values,
    )

    # Run until hitting each early stopping point.
    for stop_index in range(len(early_stops)):

        # Set early stopping point.
        config["early_stop"] = early_stops[stop_index]

        # Set loading point, if necessary.
        if stop_index > 0:
            config["load_from"] = save_name

        # Run partial training.
        tune(config)

    # Finish training from checkpoint.
    config["early_stop"] = None
    config["load_from"] = save_name
    resumed_results = tune(config)

    # Compare resumed results with un-interrupted results.
    if results_name is not None:
        results_path = os.path.join(METRICS_DIR, "%s.json" % results_name)
        with open(results_path, "r") as results_file:
            correct_results = json.load(results_file)
        assert tune_results_equal(resumed_results, correct_results)

    # Clean up saved results.
    experiment_names = get_experiment_names(
        save_name,
        config["search_type"],
        iterations,
        config["trials_per_config"],
        num_param_values=num_param_values,
    )
    for name in experiment_names:
        save_dir = save_dir_from_name(name)
        if os.path.isdir(save_dir):
            rmtree(save_dir)
