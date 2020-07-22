"""
Hyperparater search functions wrapped around training. It should be noted that there is
a ton of repeated code between random_search(), grid_search(), and IC_grid_search(). If
code changes in any of these functions, similar changes should be made in the other two.
"""

import os
import random
import pickle
import json
import itertools
from functools import reduce
from typing import Dict, Any, Tuple, Callable

from meta.train.train import train
from meta.tune.mutate import mutate_train_config
from meta.tune.utils import check_name_uniqueness, strip_config
from meta.tune.params import valid_config, update_config, get_param_values
from meta.utils.utils import save_dir_from_name, aligned_tune_configs


def tune(tune_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform search over hyperparameter configurations. Only argument is ``tune_config``,
    a dictionary holding settings for training. The expected elements of this dictionary
    are documented below. This function returns a dictionary holding the results of
    training and the various parameter configurations used.

    Parameters
    ----------
    search_type : str
        Either "random", "grid", or "IC_grid", defines the search strategy to use.
    search_iterations : int
        Number of different hyperparameter configurations to try in search sequence. In
        cases where the number of configurations is determined by ``search_params``
        (such as when using grid search), the value of this variable is ignored, and the
        determined value is used instead.
    trials_per_config : int
        Number of training runs to perform for each hyperparameter configuration. The
        fitness of each training run is averaged to produce an overall fitness for each
        hyperparameter configuration.
    base_train_config : Dict[str, Any]
        Config dictionary for function train() in meta/train.py. This is used as a
        starting point for hyperparameter search. It is required that each leaf element
        of this config dictionary have a unique key, i.e. a config containing
        base_train_config["key1"]["num_layers"] and
        base_train_config["key2"]["num_layers"] is invalid. This occurrence will cause
        unexpected behavior due to the implementation of update_config().
    search_params : Dict[str, Any]
        Search specifications for each parameter, such as max/min values, etc. The
        format of this dictionary varies between different search types.
    fitness_metric_name : str
        Name of metric (key in metrics dictionary returned from train()) to use as
        fitness function for hyperparameter search. Current supported values are
        "train_reward", "eval_reward", "train_success", "eval_success".
    fitness_metric_type : str
        Either "mean" or "maximum", used to determine which value of metric given in
        tune_config["fitnesss_metric_name"] to use as fitness, either the mean value at
        the end of training or the maximum value throughout training.
    seed : int
        Random seed for hyperparameter search.
    load_from : str
        Name of results directory to resume training from.
    """

    # Extract info from config.
    search_type = tune_config["search_type"]
    iterations = tune_config["search_iterations"]
    trials_per_config = tune_config["trials_per_config"]
    base_config = tune_config["base_train_config"]
    search_params = tune_config["search_params"]
    fitness_metric_name = tune_config["fitness_metric_name"]
    fitness_metric_type = tune_config["fitness_metric_type"]
    seed = tune_config["seed"]
    load_from = tune_config["load_from"]

    # Ignore value in tune_config["search_iterations"] during grid search and IC grid
    # search, since the number of iterations is determined by
    # tune_config["search_params"].
    if search_type in ["grid", "IC_grid"]:
        num_param_values = []
        for param_settings in search_params.values():
            if "num_values" in param_settings:
                num_param_values.append(param_settings["num_values"])
            elif "choices" in param_settings:
                num_param_values.append(len(param_settings["choices"]))
            else:
                raise ValueError("Invalid ``search_params`` value in config.")
        if search_type == "grid":
            iterations = reduce(lambda a, b: a * b, num_param_values)
        elif search_type == "IC_grid":
            iterations = sum(num_param_values)
        else:
            raise NotImplementedError

    # Read in base name and make sure it is valid. Naming is slightly different for
    # different search strategies, so we do some weirdness here to make one function
    # which handles all cases.
    base_name = base_config["save_name"]
    if base_name is not None:
        if search_type == "IC_grid":
            check_name_uniqueness(
                base_name,
                search_type,
                iterations,
                trials_per_config,
                num_param_values=num_param_values,
            )
        else:
            check_name_uniqueness(base_name, search_type, iterations, trials_per_config)

    # Make results folder and save initial config.
    if base_name is not None:

        # Create save directory.
        save_dir = save_dir_from_name(base_name)
        os.makedirs(save_dir)

        # Save config.
        config_path = os.path.join(save_dir, "%s_config.json" % base_name)
        with open(config_path, "w") as config_file:
            json.dump(tune_config, config_file, indent=4)

    else:
        save_dir = None

    # Construct load directory, if necessary.
    if load_from is not None:
        load_dir = save_dir_from_name(load_from)
    else:
        load_dir = None

    # Construct fitness function.
    if fitness_metric_name not in [
        "train_reward",
        "eval_reward",
        "train_success",
        "eval_success",
    ]:
        raise ValueError("Unsupported metric name: '%s'." % fitness_metric_name)
    if fitness_metric_type == "mean":
        fitness_fn = lambda metrics: metrics[fitness_metric_name]["mean"][-1]
    elif fitness_metric_type == "maximum":
        fitness_fn = lambda metrics: metrics[fitness_metric_name]["maximum"]
    else:
        raise ValueError("Unsupported metric type: '%s'." % fitness_metric_type)

    # Set random seed. Note that this may cause reproducibility issues if the train()
    # function ever comes to use the random module.
    random.seed(seed)

    # Run the chosen search strategy.
    if tune_config["search_type"] == "random":
        search_fn = random_search
    elif tune_config["search_type"] == "grid":
        search_fn = grid_search
    elif tune_config["search_type"] == "IC_grid":
        search_fn = IC_grid_search
    results = search_fn(
        tune_config,
        base_config,
        iterations,
        trials_per_config,
        fitness_fn,
        search_params,
        save_dir,
        load_dir,
    )

    # Save results and config.
    if base_name is not None:

        # Save results.
        results_path = os.path.join(save_dir, "%s_results.json" % base_name)
        with open(results_path, "w") as results_file:
            json.dump(results, results_file, indent=4)

    return results


def random_search(
    tune_config: Dict[str, Any],
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
    save_dir: str,
    load_dir: str,
) -> Dict[str, Any]:
    """
    Perform random search over hyperparameter configurations, returning the results.
    """

    # Load in checkpoint, if necessary.
    results: Dict[str, Any] = {"iterations": []}
    config = dict(base_config)
    best_fitness = None
    best_config = None
    iteration = 0
    if load_dir is not None:
        checkpoint_filename = os.path.join(load_dir, "checkpoint.pkl")
        with open(checkpoint_filename, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        # Make sure current config and previous config line up.
        assert aligned_tune_configs(tune_config, checkpoint["tune_config"])

        results = dict(checkpoint["results"])
        config = dict(checkpoint["config"])
        best_fitness = checkpoint["best_fitness"]
        best_config = dict(checkpoint["best_config"])
        iteration = checkpoint["iteration"]

    # Training loop.
    while iteration < iterations:

        # See if we have already performed training with this configuration.
        already_trained = False
        past_iteration = None
        past_configs = [
            results["iterations"][i]["config"]
            for i in range(len(results["iterations"]))
        ]
        for i, past_config in enumerate(past_configs):
            if strip_config(config) == strip_config(past_config):
                already_trained = True
                past_iteration = i
                break

        # If so, reuse the past results. Otherwise, run training.
        if already_trained:
            fitness = float(results["iterations"][past_iteration]["fitness"])
            config_results = dict(results["iterations"][past_iteration])

        else:

            # Run training for current config.
            get_save_name = (
                lambda name: "%s_%d" % (name, iteration) if name is not None else None
            )
            config_save_name = get_save_name(base_config["save_name"])
            metrics_save_name = get_save_name(base_config["metrics_filename"])
            baseline_metrics_save_name = get_save_name(
                base_config["baseline_metrics_filename"]
            )
            fitness, config_results = train_single_config(
                config,
                trials_per_config,
                fitness_fn,
                base_config["seed"],
                config_save_name,
                metrics_save_name,
                baseline_metrics_save_name,
            )

        # Compare current step to best so far.
        new_max = False
        if best_fitness is None or fitness > best_fitness:
            new_max = True
            best_fitness = fitness
            best_config = dict(config)

        # Add maximum to config results, and add config results to overall results.
        config_results["maximum"] = new_max
        results["iterations"].append(dict(config_results))

        # Mutate config to produce a new one for next step.
        config = mutate_train_config(search_params, best_config)
        while not valid_config(config):
            config = mutate_train_config(search_params, best_config)

        # Save intermediate results, if necessary. We add one to the iteration here, so
        # that upon resumption, the first iteration will be the next one after the last
        # copmleted iteration. It is also important that we save the checkpoint AFTER
        # the config has been mutated, so that we don't repeat configs after resumption.
        if save_dir is not None:
            checkpoint = {}
            checkpoint["results"] = dict(results)
            checkpoint["config"] = dict(config)
            checkpoint["best_fitness"] = best_fitness
            checkpoint["best_config"] = dict(best_config)
            checkpoint["iteration"] = iteration + 1
            checkpoint["tune_config"] = dict(tune_config)

            checkpoint_filename = os.path.join(save_dir, "checkpoint.pkl")
            with open(checkpoint_filename, "wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)

        iteration += 1

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def grid_search(
    tune_config: Dict[str, Any],
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
    save_dir: str,
    load_dir: str,
) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameter configurations, returning the results.
    """

    # Construct set of configurations to search over.
    param_values = {}
    for param_name, param_settings in search_params.items():
        param_values[param_name] = get_param_values(param_settings)
    config_values = list(itertools.product(*list(param_values.values())))
    configs = []
    for config_value in config_values:
        config = dict(base_config)
        new_values = dict(zip(search_params.keys(), config_value))
        config = update_config(config, new_values)
        configs.append(dict(config))

    # Load in checkpoint, if necessary.
    results: Dict[str, Any] = {"iterations": []}
    best_fitness = None
    best_config = None
    iteration = 0
    if load_dir is not None:
        checkpoint_filename = os.path.join(load_dir, "checkpoint.pkl")
        with open(checkpoint_filename, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        # Make sure current config and previous config line up.
        assert aligned_tune_configs(tune_config, checkpoint["tune_config"])

        results = dict(checkpoint["results"])
        best_fitness = checkpoint["best_fitness"]
        best_config = dict(checkpoint["best_config"])
        iteration = checkpoint["iteration"]

    # Training loop.
    while iteration < len(configs):

        config = configs[iteration]

        # Run training for current config.
        get_save_name = (
            lambda name: "%s_%d" % (name, iteration) if name is not None else None
        )
        config_save_name = get_save_name(base_config["save_name"])
        metrics_save_name = get_save_name(base_config["metrics_filename"])
        baseline_metrics_save_name = get_save_name(
            base_config["baseline_metrics_filename"]
        )
        fitness, config_results = train_single_config(
            config,
            trials_per_config,
            fitness_fn,
            base_config["seed"],
            config_save_name,
            metrics_save_name,
            baseline_metrics_save_name,
        )

        # Compare current step to best so far.
        if best_fitness is None or fitness > best_fitness:
            best_fitness = fitness
            best_config = dict(config)

        # Add maximum to config results, and add config results to overall results.
        results["iterations"].append(dict(config_results))

        # Save intermediate results, if necessary. We add one to the iteration here so
        # that upon resumption, the first iteration will be the next one after the last
        # completed iteration.
        if save_dir is not None:
            checkpoint = {}
            checkpoint["results"] = dict(results)
            checkpoint["best_fitness"] = best_fitness
            checkpoint["best_config"] = dict(best_config)
            checkpoint["iteration"] = iteration + 1
            checkpoint["tune_config"] = dict(tune_config)

            checkpoint_filename = os.path.join(save_dir, "checkpoint.pkl")
            with open(checkpoint_filename, "wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)

        iteration += 1

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def IC_grid_search(
    tune_config: Dict[str, Any],
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
    save_dir: str,
    load_dir: str,
) -> Dict[str, Any]:
    """
    Perform iterated constrained grid search over hyperparameter configurations,
    returning the results. In this style of search, we only vary one parameter at a
    time. For each parameter, we do a mini grid search, where we try configurations
    where all other parameters are held fixed, and our parameter of interest varies over
    an interval. The parameter of interest is then fixed at whichever value led to the
    best fitness, and a new parameter is varied. Each parameter is varied once, in the
    order specified by search_params.
    """

    # Construct list of values for each variable parameter to vary over.
    param_values = {}
    for param_name, param_settings in search_params.items():
        param_values[param_name] = get_param_values(param_settings)

    # We set config values for all varying parameters to their median values. We do this
    # to ensure that, on each IC grid iteration, the best configuration so far is
    # included in the configurations to try. If the values of the varying parameters in
    # ``base_config`` aren't included in the intervals specified in ``search_params``,
    # then the original values of the varying parameters are never revisited. We avoid
    # this by explicitly setting the values of the varying parameters to their median
    # values in the given intervals.
    config = dict(base_config)
    median_values = {
        param_name: param_interval[len(param_interval) // 2]
        for param_name, param_interval in param_values.items()
    }
    config = update_config(config, median_values)

    # Load in checkpoint, if necessary.
    results: Dict[str, Any] = {"iterations": []}
    best_fitness = None
    best_config = None
    best_param_vals = {}
    param_num = 0
    val_num = 0
    if load_dir is not None:
        checkpoint_filename = os.path.join(load_dir, "checkpoint.pkl")
        with open(checkpoint_filename, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)

        # Make sure current config and previous config line up.
        assert aligned_tune_configs(tune_config, checkpoint["tune_config"])

        results = dict(checkpoint["results"])
        best_fitness = checkpoint["best_fitness"]
        best_config = dict(checkpoint["best_config"])
        best_param_vals = dict(checkpoint["best_param_vals"])
        param_num = checkpoint["param_num"]
        val_num = checkpoint["val_num"]

    # Training loop.
    while param_num < len(search_params):

        # Fix parameter value to that which led to highest fitness.
        config = update_config(config, best_param_vals)

        # Find best value of parameter ``param_name``.
        param_name = list(search_params.keys())[param_num]
        best_param_fitness = None

        while val_num < len(param_values[param_name]):

            # Set value of current param of interest in current config.
            param_val = param_values[param_name][val_num]
            config = update_config(config, {param_name: param_val})

            # See if we have already performed training with this configuration.
            already_trained = False
            past_iteration = None
            past_configs = [
                results["iterations"][i]["config"]
                for i in range(len(results["iterations"]))
            ]
            for i, past_config in enumerate(past_configs):
                if strip_config(config) == strip_config(past_config):
                    already_trained = True
                    past_iteration = i
                    break

            # If so, reuse the past results. Otherwise, run training.
            if already_trained:
                fitness = float(results["iterations"][past_iteration]["fitness"])
                config_results = dict(results["iterations"][past_iteration])

            else:

                # Run training for current config.
                get_save_name = (
                    lambda name: "%s_%d_%d" % (name, param_num, val_num)
                    if name is not None
                    else None
                )
                config_save_name = get_save_name(base_config["save_name"])
                metrics_save_name = get_save_name(base_config["metrics_filename"])
                baseline_metrics_save_name = get_save_name(
                    base_config["baseline_metrics_filename"]
                )
                fitness, config_results = train_single_config(
                    config,
                    trials_per_config,
                    fitness_fn,
                    base_config["seed"],
                    config_save_name,
                    metrics_save_name,
                    baseline_metrics_save_name,
                )

            # Compare current step to best so far.
            if best_fitness is None or fitness > best_fitness:
                best_fitness = fitness
                best_config = dict(config)

            # Compare current step to best among current IC grid iteration.
            if best_param_fitness is None or fitness > best_param_fitness:
                best_param_fitness = fitness
                best_param_vals[param_name] = param_val

            # Add maximum to config results, and add config results to overall results.
            results["iterations"].append(dict(config_results))

            # Save intermediate results, if necessary. We increment val_num by one here
            # (resetting to zero and incrementing param_num if necessary) so that upon
            # resumption, training starts with the first iteration not yet completed.
            if save_dir is not None:
                checkpoint = {}
                checkpoint["results"] = dict(results)
                checkpoint["best_fitness"] = best_fitness
                checkpoint["best_config"] = dict(best_config)
                checkpoint["tune_config"] = dict(tune_config)
                checkpoint["best_param_vals"] = dict(best_param_vals)

                # Increment val_num and/or param_num.
                cp_val_num = val_num + 1
                cp_param_num = param_num
                if cp_val_num == len(param_values[param_name]):
                    cp_val_num = 0
                    cp_param_num += 1
                checkpoint["val_num"] = cp_val_num
                checkpoint["param_num"] = cp_param_num

                checkpoint_filename = os.path.join(save_dir, "checkpoint.pkl")
                with open(checkpoint_filename, "wb") as checkpoint_file:
                    pickle.dump(checkpoint, checkpoint_file)

            # Update value index.
            val_num += 1

        # Update search indices.
        param_num += 1
        val_num = 0

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def train_single_config(
    train_config: Dict[str, Any],
    trials_per_config: int,
    fitness_fn: Callable,
    seed: int,
    config_save_name: str = None,
    metrics_filename: str = None,
    baseline_metrics_filename: str = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run training with a fixed config for ``trials_per_config`` trials, and return
    fitness and a dictionary holding results.
    """

    # Perform training and compute resulting fitness for multiple trials.
    fitness = 0.0
    config_results: Dict[str, Any] = {}
    config_results["trials"] = []
    config_results["config"] = dict(train_config)
    for trial in range(trials_per_config):

        trial_results = {}

        # Set trial name, seed, and metrics filenames for saving/comparison, if
        # neccessary.
        get_save_name = (
            lambda name: "%s_%d" % (name, trial) if name is not None else None
        )
        train_config["save_name"] = get_save_name(config_save_name)
        train_config["metrics_filename"] = get_save_name(metrics_filename)
        train_config["baseline_metrics_filename"] = get_save_name(
            baseline_metrics_filename
        )
        train_config["seed"] = seed + trial

        # Run training and get fitness.
        metrics = train(train_config)
        trial_fitness = fitness_fn(metrics)
        fitness += trial_fitness

        # Fill in trial results.
        trial_results["trial"] = trial
        trial_results["metrics"] = dict(metrics)
        trial_results["fitness"] = trial_fitness
        config_results["trials"].append(dict(trial_results))

    fitness /= trials_per_config
    config_results["fitness"] = fitness

    return fitness, config_results
