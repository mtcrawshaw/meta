import os
import random
import json
from typing import Dict, Any, Tuple, Callable

from meta.train import train
from meta.utils import save_dir_from_name


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
    base_name: str, iterations: int, trials_per_config: int,
) -> None:
    """
    Check to make sure that there are no other saved experiments whose names coincide
    with the current name. This is just to make sure that the saved results don't get
    mixed up, with some trials being saved with a modified name to ensure uniqueness.
    """

    # Build list of names to check.
    names_to_check = [base_name]
    for iteration in range(iterations):
        for trial in range(trials_per_config):
            names_to_check.append("%s_%d_%d" % (base_name, iteration, trial))

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
    config_results = {}
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
        train_config["seed"] = trial

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
    results = {"iterations": []}

    # Training loop.
    config = dict(base_config)
    best_fitness = None
    best_metrics = None
    best_config = dict(base_config)
    for iteration in range(iterations):

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


def hyperparameter_search(hp_config: Dict[str, Any]) -> None:
    """
    Perform random search over hyperparameter configurations. Only argument is
    ``hp_config``, a dictionary holding settings for training. The expected elements of
    this dictionary are documented below.

    Parameters
    ----------
    search_type : str
        Either "random" or "grid", defines the search strategy to use.
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

    # Read in base name and make sure it is valid.
    base_name = base_config["save_name"]
    if base_name is not None:
        check_name_uniqueness(base_name, iterations, trials_per_config)

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

    # Set random seed.
    random.seed(seed)

    # Run random search.
    results = random_search(
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
