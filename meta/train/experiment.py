"""
Run neural network training multiple times with various settings, and compare the
results.
"""

import os
import json
from typing import Dict, Any

import numpy as np

from meta.train.train import train
from meta.utils.metrics import Metrics
from meta.utils.utils import save_dir_from_name


def experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function for experiment.py, runs multiple trainings with settings from
    `config`. The expected entries of `config` are documented below. Returns a
    dictionary with metrics from each training run.

    Parameters
    ----------
    trials_per_method : int
        Number of training trials to run for each method given. The final performance of
        each method is measured as the mean performance across all trials.
    key_metrics : List[str]
        List of metrics by which to compare performance of methods.
    base_train_config : Dict[str, Any]
        Training config that will be used as the base for all methods. This dictionary
        will be used as a config argument to `train()`, so the entries of this
        dictionary should follow the entries documented in the `train()` function. The
        only exceptions are the `seed` and `save_name` entries, which will be created
        differently for each method/trial.
    methods : Dict[str, Dict[str, Any]]
        Dictionary that describes which methods to run training with. Each key in this
        dictionary is the name of a method which will be used to refer to the
        configuration of training settings described by the value. The value is a
        dictionary whose structure should match the structure of `base_train_config`.
        That is, each key in any such dictionary should also be a key in
        `base_train_config`. If that key's value is a dictionary, then this condition
        should hold recursively for that key's value. A formal description of this
        condition is given in `add_settings()`. For each entry of `methods`, this
        function will run a number of training runs equal to `trials_per_method`, and
        aggregate the results.
    seed : int
        Random seed.
    save_name : str
        Name under which to save results. If None, no results will be saved. If not
        None, then it is required that there does not already exist a directory in the
        results directory whose name is `save_name`.
    """

    # Construct save directory, if necessary.
    if config["save_name"] is not None:

        save_dir = save_dir_from_name(config["save_name"])

        # Check for any existing results with the save names that will be used to store
        # the results of this experiment.
        save_dir_names = [save_dir]
        for method in config["methods"]:
            save_dir_names += [
                save_dir_from_name(f"{config['save_name']}_{method}_{trial}")
                for trial in range(config["trials_per_method"])
            ]
        for name in save_dir_names:
            if os.path.isdir(name):
                raise ValueError(
                    f"There already exists a results directory with the name '{name}'."
                    " Save names must be unique. Delete the existing directory or"
                    " rename your experiment/methods."
                )

        # Create save directory.
        os.makedirs(save_dir)

        # Save config.
        config_path = os.path.join(save_dir, f"{config['save_name']}_config.json")
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

    # Initialize metrics. Here we will store the training metrics from each individual
    # training run, as well as the summary of metrics over all runs.
    metrics = {}

    # Training with each method.
    for method, method_settings in config["methods"].items():

        metrics[method] = {}
        metrics[method]["trials"] = []

        # Trials for a single method.
        for trial in range(config["trials_per_method"]):

            # Construct training config from base training config.
            train_config = dict(config["base_train_config"])
            train_config = add_settings(train_config, method_settings)
            train_config["seed"] = config["seed"] + trial
            if config["save_name"] is not None:
                train_config["save_name"] = f"{config['save_name']}_{method}_{trial}"
            else:
                train_config["save_name"] = None

            # Run training.
            checkpoint = train(train_config)

            # Add metrics from single training to aggregated metrics.
            metrics[method]["trials"].append(checkpoint["metrics"])

        # Compute mean metrics over all trials.
        metrics[method]["mean"] = Metrics.mean(metrics[method]["trials"])

    # Print out results summary. Note that we are printing the mean of the best metric
    # values across training, as opposed to the best mean of the metrics values across
    # training. Mean of best is greater than best of mean.
    for i, key_metric in enumerate(config["key_metrics"]):
        print(f"\nMean {key_metric}:")
        for method in config["methods"]:
            performance = np.mean(
                [
                    trial_metrics.metric_dict[key_metric].best
                    for trial_metrics in metrics[method]["trials"]
                ]
            )
            print(f"    {method}: {performance}")

    # Save results if necessary.
    if config["save_name"] is not None:

        # Convert `metrics` into a nested dictionary.
        metrics_dict = dict(metrics)
        for method in config["methods"]:
            for trial in range(config["trials_per_method"]):
                metrics_dict[method]["trials"][trial] = metrics_dict[method]["trials"][
                    trial
                ].state()
            metrics_dict[method]["mean"] = metrics_dict[method]["mean"].state()

        # Save metrics.
        metrics_path = os.path.join(save_dir, f"{config['save_name']}_metrics.json")
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics_dict, metrics_file)

        # Plot results.
        # TODO: Add plotting.

    return metrics


def add_settings(
    base_config: Dict[str, Any], settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively add entries from a nested dictionary `settings` to a nested dictionary
    `base_config`. Any key from `settings` should also be a key in `base_config`.
    """

    new_config = dict(base_config)
    for key in settings:
        assert key in new_config
        if isinstance(settings[key], dict):
            new_config[key] = add_settings(new_config[key], settings[key])
        else:
            new_config[key] = settings[key]

    return new_config
