"""
Run neural network training multiple times with various settings, and compare the
results.
"""

import os
import pickle
import json
from collections import OrderedDict
from copy import deepcopy
from math import sqrt
from typing import Dict, Any

import numpy as np
from scipy import stats

from meta.train.train import train
from meta.utils.metrics import Metrics
from meta.utils.utils import save_dir_from_name
from meta.report.plot import plot


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

        # Compute mean/std of best metric values across training. Note that this is
        # different from the best mean across training. Mean of best is greater than
        # best mean.
        metrics[method]["summary"] = {}
        for key_metric in config["key_metrics"]:
            metrics[method]["summary"][key_metric] = {}
            bests = [
                trial_metrics.metric_dict[key_metric].best
                for trial_metrics in metrics[method]["trials"]
            ]
            mean = np.mean(bests)
            std = np.std(bests, ddof=1)
            n = len(bests)
            metrics[method]["summary"][key_metric]["mean"] = mean
            metrics[method]["summary"][key_metric]["std"] = std
            for conf in [0.99, 0.98, 0.95]:
                name = f"CI@{conf}"
                t = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
                ub = mean + t * std / sqrt(n)
                lb = mean - t * std / sqrt(n)
                metrics[method]["summary"][key_metric][name] = (lb, ub)

    # Compute results summary. Here we sort the methods by their performance on the each
    # key metric.
    metrics["summary"] = {}
    for key_metric in config["key_metrics"]:
        metrics["summary"][key_metric] = OrderedDict()
        performances = [
            (method, metrics[method]["summary"][key_metric]["mean"])
            for method in config["methods"]
        ]
        method = list(config["methods"].keys())[0]
        maximize = metrics[method]["mean"].metric_dict[key_metric].maximize
        performances = sorted(performances, key=lambda x: x[1], reverse=maximize)
        for method, performance in performances:
            conf = 0.95
            CI_name = f"CI@{conf}"
            metrics["summary"][key_metric][method] = {
                "mean_performance": performance,
                "std_performance": metrics[method]["summary"][key_metric]["std"],
                "CI": metrics[method]["summary"][key_metric][CI_name],
                "conf": conf,
            }

    # Print out results summary.
    for key_metric in config["key_metrics"]:
        print(f"\nMean, std, LB, UB {key_metric}:")
        for method, results in metrics["summary"][key_metric].items():
            mean = results["mean_performance"]
            std = results["std_performance"]
            CI = results["CI"]
            print(f"    {method}: {mean:.5f}, {std:.5f}, {CI[0]:.5f}, {CI[1]:.5f}")

    # Save results if necessary.
    if config["save_name"] is not None:

        # Save checkpoint.
        checkpoint = {}
        checkpoint["metrics"] = metrics
        checkpoint_path = os.path.join(save_dir, "checkpoint.pkl")
        with open(checkpoint_path, "wb") as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)

        # Convert `metrics` into a nested dictionary.
        metrics_dict = deepcopy(metrics)
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
        plot_path = os.path.join(save_dir, f"{config['save_name']}_plot.png")
        plot(
            {method: metrics[method]["mean"] for method in config["methods"]},
            plot_path,
            summary=metrics["summary"],
        )

    return metrics


def add_settings(
    base_config: Dict[str, Any], settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively add entries from a nested dictionary `settings` to a nested dictionary
    `base_config`.
    """

    new_config = dict(base_config)
    for key in settings:
        if isinstance(settings[key], dict):
            new_config[key] = add_settings(new_config[key], settings[key])
        else:
            new_config[key] = settings[key]

    return new_config
