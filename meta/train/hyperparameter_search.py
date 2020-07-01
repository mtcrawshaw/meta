"""
Hyperparater search functions wrapped around training.
"""

import os
import random
import json
import itertools
from functools import reduce
from typing import Dict, Any, Tuple, Callable, List

from meta.train.train import train
from meta.utils.utils import save_dir_from_name


# Perturbation functions used to mutate parameters for random search.
PERTURBATIONS = {
    "geometric": lambda factor: lambda val: val * 10 ** random.uniform(-factor, factor),
    "arithmetic": lambda shift: lambda val: val * (1.0 + random.uniform(-shift, shift)),
    "increment": lambda radius: lambda val: random.randint(val - radius, val + radius),
    "discrete": (
        lambda choices, mut_p: lambda val: random.choice(choices)
        if random.random() < mut_p
        else val
    ),
}


def clip(val: float, min_value: float, max_value: float, prev_value: float) -> float:
    """
    Clips a floating point value within a given range, ensuring that the clipped
    value is not equal to either bound when the value is a float.
    """
    if val >= max_value:
        if isinstance(val, int):
            val = max_value
        else:
            val = (prev_value + max_value) / 2.0
    if val <= min_value:
        if isinstance(val, int):
            val = min_value
        else:
            val = (prev_value + min_value) / 2.0
    return val


def mutate_train_config(
    search_params: Dict[str, Any], train_config: Dict[str, Any]
) -> Dict[str, Any]:
    """ Mutates a training config by perturbing individual elements. """

    # Build up new config.
    new_config: Dict[str, Any] = {}
    for param in train_config:
        new_config[param] = train_config[param]

        # Perturb parameter, if necessary.
        if param in search_params:

            # Construct perturbation function from settings.
            param_settings = search_params[param]
            perturb_kwargs = param_settings["perturb_kwargs"]
            perturb = PERTURBATIONS[param_settings["perturb_type"]](**perturb_kwargs)
            min_value = (
                param_settings["min_value"] if "min_value" in param_settings else None
            )
            max_value = (
                param_settings["max_value"] if "max_value" in param_settings else None
            )

            # Perturb parameter.
            new_config[param] = perturb(train_config[param])

            # Clip parameter, if necessary.
            if min_value is not None and max_value is not None:
                prev_value = train_config[param]
                new_config[param] = clip(
                    new_config[param], min_value, max_value, prev_value
                )

    return new_config


def valid_config(config: Dict[str, Any]) -> bool:
    """ Determine whether or not given configuration fits requirements. """

    valid = True

    # Test for requirements on num_minibatch, rollout_length, and num_processes detailed
    # in meta/storage.py (in this file, these conditions are checked at the beginning of
    # each generator definition, and an error is raised when they are violated)
    if config["recurrent"] and config["num_processes"] < config["num_minibatch"]:
        valid = False
    if not config["recurrent"]:
        total_steps = config["rollout_length"] * config["num_processes"]
        if total_steps < config["num_minibatch"]:
            valid = False

    return valid


def check_name_uniqueness(
    base_name: str,
    search_type: str,
    iterations: int,
    trials_per_config: int,
    num_param_values: List[int] = None,
) -> None:
    """
    Check to make sure that there are no other saved experiments whose names coincide
    with the current name. This is just to make sure that the saved results don't get
    mixed up, with some trials being saved with a modified name to ensure uniqueness. We
    do some weirdness here to handle cases of different search types, since the naming
    is slightly different for IC grid.
    """

    # Build list of names to check.
    if search_type in ["grid", "random"]:
        assert num_param_values is None
        names_to_check = [base_name]
        for iteration in range(iterations):
            for trial in range(trials_per_config):
                names_to_check.append("%s_%d_%d" % (base_name, iteration, trial))
    elif search_type == "IC_grid":
        assert num_param_values is not None
        names_to_check = [base_name]
        for param_num, param_len in enumerate(num_param_values):
            for param_iteration in range(param_len):
                for trial in range(trials_per_config):
                    names_to_check.append(
                        "%s_%d_%d_%d" % (base_name, param_num, param_iteration, trial)
                    )
    else:
        raise NotImplementedError

    # Check names.
    for name in names_to_check:
        if os.path.isdir(save_dir_from_name(name)):
            raise ValueError(
                "Saved result '%s' already exists. Results of hyperparameter searches"
                " must have unique names." % name
            )


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


def get_param_values(param_settings: Dict[str, Any]) -> List[Any]:
    """
    Produce a list of parameter values from the parameter search settings. The expected
    format of the input dictionary ``param_settings`` is documented below.

    ``search_params`` can be one of two forms. An example of each is shown below.

    {
        "distribution_type": "geometric",
        "num_values": 2,
        "min_value": 1e-5,
        "max_value": 1e-3
    }

    OR

    {
        "distribution_type": "discrete",
        "choices": [true, false]
    }

    The first form is used to specify values for a numeric parameter. Values are
    computed in an interval between a min value and a max value based on a number of
    values and a distribution type. When a geometric interval is used, successive terms
    will differ by a constant ratio, whereas an arithmetic interval will yield terms
    that have a constant difference.

    The second form is used to specify values for a categorical parameter. Distribution
    type should be set to "discrete", and the value of "choices" should hold the list of
    values to draw from. This can also be used to specify a hand-coded set of numerical
    values.
    """

    param_values = []

    # Get data type from min/max value.
    if param_settings["distribution_type"] in ["geometric", "arithmetic"]:
        if type(param_settings["min_value"]) != type(param_settings["max_value"]):
            raise ValueError(
                "Conflicting data types for min/max value of parameter search settings: %s"
                % param_settings
            )
        datatype = type(param_settings["min_value"])

    # Geometric interval (numeric).
    if param_settings["distribution_type"] == "geometric":

        # If there is only one value, take geometric mean of max and min.
        if param_settings["num_values"] == 1:
            param_values = [
                datatype(
                    (param_settings["max_value"] * param_settings["min_value"]) ** (0.5)
                )
            ]
        else:
            ratio = (param_settings["max_value"] / param_settings["min_value"]) ** (
                1.0 / (param_settings["num_values"] - 1.0)
            )
            param_values = [
                datatype(param_settings["min_value"] * ratio ** i)
                for i in range(param_settings["num_values"])
            ]

    # Arithmetic interval (numeric).
    elif param_settings["distribution_type"] == "arithmetic":

        # If there is only one value, take mean of max and min.
        if param_settings["num_values"] == 1:
            param_values = [
                datatype(
                    (param_settings["max_value"] + param_settings["min_value"]) * (0.5)
                )
            ]
        else:
            shift = (param_settings["max_value"] - param_settings["min_value"]) / (
                param_settings["num_values"] - 1.0
            )
            param_values = [
                datatype(param_settings["min_value"] + shift * i)
                for i in range(param_settings["num_values"])
            ]

    # Discrete interval (categorical).
    elif param_settings["distribution_type"] == "discrete":
        param_values = list(param_settings["choices"])

    else:
        raise ValueError(
            "Unrecognized distribution type: %s" % param_settings["distribution_type"]
        )

    return param_values


def random_search(
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform random search over hyperparameter configurations, returning the results.
    """

    # Initialize results.
    results: Dict[str, Any] = {"iterations": []}

    # Helper function to compare configs.
    nonessential_params = [
        "save_name",
        "metrics_filename",
        "baseline_metrics_filename",
        "print_freq",
    ]

    def strip_config(config: Dict[str, Any]) -> Dict[str, Any]:
        stripped = dict(config)
        for param in nonessential_params:
            del stripped[param]
        return stripped

    # Training loop.
    config = dict(base_config)
    best_fitness = None
    best_config = dict(base_config)
    for iteration in range(iterations):

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

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def grid_search(
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameter configurations, returning the results.
    """

    # Initialize results.
    results: Dict[str, Any] = {"iterations": []}

    # Construct set of configurations to search over.
    param_values = {}
    for param_name, param_settings in search_params.items():
        param_values[param_name] = get_param_values(param_settings)
    config_values = list(itertools.product(*list(param_values.values())))
    configs = []
    for config_value in config_values:
        config = dict(base_config)
        config.update(dict(zip(search_params.keys(), config_value)))
        configs.append(dict(config))

    # Training loop.
    best_fitness = None
    best_config = dict(base_config)
    for iteration, config in enumerate(configs):

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

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def IC_grid_search(
    base_config: Dict[str, Any],
    iterations: int,
    trials_per_config: int,
    fitness_fn: Callable,
    search_params: Dict[str, Any],
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

    # Initialize results.
    results: Dict[str, Any] = {"iterations": []}

    # Helper function to compare configs.
    nonessential_params = ["save_name", "metrics_filename", "baseline_metrics_filename"]

    def strip_config(config: Dict[str, Any]) -> Dict[str, Any]:
        stripped = dict(config)
        for param in nonessential_params:
            del stripped[param]
        return stripped

    # Construct list of values for each variable parameter to vary over.
    param_values = {}
    for param_name, param_settings in search_params.items():
        param_values[param_name] = get_param_values(param_settings)

    # Training loop. We set config values for all varying parameters to their median
    # values. We do this to ensure that, on each IC grid iteration, the best
    # configuration so far is included in the configurations to try. If the values of
    # the varying parameters in ``base_config`` aren't included in the intervals
    # specified in ``search_params``, then the original values of the varying parameters
    # are never revisited. We avoid this by explicitly setting the values of the varying
    # parameters to their median values in the given intervals.
    config = dict(base_config)
    for param_name, param_interval in param_values.items():
        config[param_name] = param_interval[len(param_interval) // 2]
    best_fitness = None
    best_config = dict(base_config)
    for param_num, (param_name, param_settings) in enumerate(search_params.items()):

        # Find best value of parameter ``param_name``.
        best_param_fitness = None
        best_param_val = None

        for val_num, param_val in enumerate(param_values[param_name]):

            # Set value of current param of interest in current config.
            config[param_name] = param_val

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
                best_param_val = param_val

            # Add maximum to config results, and add config results to overall results.
            results["iterations"].append(dict(config_results))

        # Fix parameter value to that which led to highest fitness.
        config[param_name] = best_param_val

    # Fill results.
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    return results


def hyperparameter_search(hp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform random search over hyperparameter configurations. Only argument is
    ``hp_config``, a dictionary holding settings for training. The expected elements of
    this dictionary are documented below. This function returns a dictionary holding the
    results of training and the various parameter configurations used.

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
        starting point for hyperparameter search.
    search_params : Dict[str, Any]
        Search specifications for each parameter, such as max/min values, etc. The
        format of this dictionary varies between different search types.
    fitness_metric_name : str
        Name of metric (key in metrics dictionary returned from train()) to use as
        fitness function for hyperparameter search. Current supported values are
        "train_reward", "eval_reward", "train_success", "eval_success".
    fitness_metric_type : str
        Either "mean" or "maximum", used to determine which value of metric given in
        hp_config["fitnesss_metric_name"] to use as fitness, either the mean value at
        the end of training or the maximum value throughout training.
    seed : int
        Random seed for hyperparameter search.
    """

    # Extract info from config.
    search_type = hp_config["search_type"]
    iterations = hp_config["search_iterations"]
    trials_per_config = hp_config["trials_per_config"]
    base_config = hp_config["base_train_config"]
    search_params = hp_config["search_params"]
    fitness_metric_name = hp_config["fitness_metric_name"]
    fitness_metric_type = hp_config["fitness_metric_type"]
    seed = hp_config["seed"]

    # Ignore value in hp_config["search_iterations"] during grid search and IC grid
    # search, since the number of iterations is determined by
    # hp_config["search_params"].
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
    if hp_config["search_type"] == "random":
        search_fn = random_search
    elif hp_config["search_type"] == "grid":
        search_fn = grid_search
    elif hp_config["search_type"] == "IC_grid":
        search_fn = IC_grid_search
    results = search_fn(
        base_config, iterations, trials_per_config, fitness_fn, search_params
    )

    # Save results and config.
    if base_name is not None:

        # Create save directory.
        save_dir = save_dir_from_name(base_name)
        os.makedirs(save_dir)

        # Save config.
        config_path = os.path.join(save_dir, "%s_config.json" % base_name)
        with open(config_path, "w") as config_file:
            json.dump(hp_config, config_file, indent=4)

        # Save results.
        results_path = os.path.join(save_dir, "%s_results.json" % base_name)
        with open(results_path, "w") as results_file:
            json.dump(results, results_file, indent=4)

    return results
