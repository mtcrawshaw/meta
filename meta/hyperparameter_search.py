import os
import random
import json
from typing import Dict, Any

from meta.train import train
from meta.utils import save_dir_from_name


# Perturbation specifications for each parameter. Each value in ``PERTURBATIONS`` is a
# 3-tuple consisting of a perturbation function, a minimum value, and a maximum value.
GEOMETRIC = lambda factor: lambda val: val * 10 ** random.uniform(-factor, factor)
ARITHMETIC = lambda shift: lambda val: val * (1.0 + random.uniform(-shift, shift))
INCREMENT_INT = lambda radius: lambda val: random.randint(val - radius, val + radius)
DISCRETE = (
    lambda choices, mut_p: lambda val: random.choice(choices)
    if random.random() < mut_p
    else val
)
PERTURBATIONS = {
    "num_ppo_epochs": (INCREMENT_INT(1), 1, 8),
    "num_minibatch": (INCREMENT_INT(1), 1, 8),
    "lr_schedule_type": (DISCRETE([None, "exponential", "cosine"], 0.2), None, None),
    "initial_lr": (GEOMETRIC(1), 1e-12, 1e-2),
    "final_lr": (GEOMETRIC(1), 1e-12, 1e-2),
    "eps": (GEOMETRIC(1), 1e-12, 1e-3),
    "value_loss_coeff": (ARITHMETIC(0.1), 0.05, 5.0),
    "entropy_loss_coeff": (ARITHMETIC(0.1), 0.0001, 1.0),
    "gamma": (ARITHMETIC(0.1), 0.1, 1.0),
    "gae_lambda": (ARITHMETIC(0.1), 0.1, 1.0),
    "max_grad_norm": (ARITHMETIC(0.1), 0.01, 5.0),
    "clip_param": (ARITHMETIC(0.1), 0.01, 5.0),
    "clip_value_loss": (DISCRETE([True, False], 0.2), None, None),
    "normalize_advantages": (DISCRETE([True, False], 0.2), None, None),
    "normalize_transition": (DISCRETE([True, False], 0.2), None, None),
    "num_layers": (INCREMENT_INT(1), 1, 8),
    "hidden_size": (INCREMENT_INT(16), 2, 512),
    "recurrent": (DISCRETE([True, False], 0.2), None, None),
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


def mutate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """ Mutates a training config by perturbing individual elements. """

    # Build up new config.
    new_config: Dict[str, Any] = {}
    for param in config:
        new_config[param] = config[param]

        # Perturb parameter, if necessary.
        if param in PERTURBATIONS:
            prev_value = config[param]
            perturb, min_value, max_value = PERTURBATIONS[param]
            new_config[param] = perturb(config[param])

            # Clip parameter, if necessary.
            if min_value is not None and max_value is not None:
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
    config: Dict[str, Any], base_name: str, iterations: int
) -> None:
    """
    Check to make sure that there are no other saved experiments whose names coincide
    with the current name. This is just to make sure that the saved results don't get
    mixed up, with some trials being saved with a modified name to ensure uniqueness.
    """

    # Build list of names to check.
    names_to_check = [base_name]
    for iteration in range(iterations):
        for trial in range(config["trials_per_config"]):
            names_to_check.append("%s_%d_%d" % (base_name, iteration, trial))

    # Check names.
    for name in names_to_check:
        if os.path.isdir(save_dir_from_name(name)):
            raise ValueError(
                "Saved result '%s' already exists. Results of hyperparameter searches"
                " must have unique names." % name
            )


def hyperparameter_search(hp_config: Dict[str, Any]) -> None:
    """
    Perform random search over hyperparameter configurations. Only argument is
    ``hp_config``, a dictionary holding settings for training. The expected elements of
    this dictionary are documented below.

    Parameters
    ----------
    base_train_config : Dict[str, Any]
        Config dictionary for function train() in meta/train.py. This is used as a
        starting point for hyperparameter search.
    hp_search_iterations : int
        Number of different hyperparameter configurations to try in search sequence.
    trials_per_config : int
        Number of training runs to perform for each hyperparameter configuration. The
        fitness of each training run is averaged to produce an overall fitness for each
        hyperparameter configuration.
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

    # Extract base training config and number of iterations.
    base_config = hp_config["base_train_config"]
    iterations = hp_config["hp_search_iterations"]

    # Read in base name and make sure it is valid.
    base_name = base_config["save_name"]
    if base_name is None:
        raise ValueError(
            "config['save_name'] cannot be None for hyperparameter search."
        )
    check_name_uniqueness(base_config, base_name, iterations)

    # Construct fitness as a function of metrics returned from train().
    if config["fitness_metric_name"] not in [
        "train_reward",
        "eval_reward",
        "train_success",
        "eval_success",
    ]:
        raise ValueError(
            "Unsupported metric name: '%s'." % config["fitness_metric_name"]
        )
    if config["fitness_metric_type"] == "mean":
        get_fitness = lambda metrics: metrics[config["fitness_metric_name"]]["mean"][-1]
    elif config["fitness_metric_type"] == "maximum":
        get_fitness = lambda merics: metrics[config["fitness_metric_name"]]["maximum"]
    else:
        raise ValueError(
            "Unsupported metric type: '%s'." % config["fitness_metric_type"]
        )

    # Set random seed.
    random.seed(hp_config["seed"])

    # Initialize results.
    results = {}
    results["name"] = base_name
    results["base_config"] = base_config
    results["base_name"] = base_name
    results["iterations"] = iterations
    results["seed"] = hp_config["seed"]

    # Training loop.
    config = base_config
    best_fitness = None
    best_metrics = None
    best_config = base_config
    for iteration in range(iterations):

        # Perform training and compute resulting fitness for multiple trials.
        fitness = 0.0
        iteration_results = {"trials": []}
        for trial in range(config["trials_per_config"]):

            trial_results = {}

            # Set trial name and seed.
            trial_name = "%s_%d_%d" % (base_name, iteration, trial)
            config["save_name"] = trial_name
            config["seed"] = trial

            # Run training and get fitness.
            metrics = train(config)
            trial_fitness = get_fitness(metrics)
            fitness += trial_fitness

            # Fill in trial results.
            trial_results["name"] = trial_name
            trial_results["config"] = dict(config)
            trial_results["metrics"] = dict(metrics)
            trial_results["fitness"] = trial_fitness
            iteration_results["trials"].append(dict(trial_results))

        fitness /= config["trials_per_config"]

        # Compare current step to best so far.
        new_max = False
        if best_fitness is None or fitness > best_fitness:
            new_max = True
            best_fitness = fitness
            best_config = config

        # Fill in iteration results.
        iteration_results["fitness"] = fitness
        iteration_results["maximum"] = new_max

        # Mutate config to produce a new one for next step.
        config = mutate_config(best_config)
        while not valid_config(config):
            config = mutate_config(best_config)

    # Fill results.
    results["iterations"] = dict(iteration_results)
    results["best_config"] = dict(best_config)
    results["best_fitness"] = best_fitness

    # Save results.
    save_dir = save_dir_from_name(base_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    results_path = os.path.join(save_dir, "%s_results.json" % base_name)
    with open(results_path, "w") as results_file:
        json.dump(results, results_file, indent=4)
