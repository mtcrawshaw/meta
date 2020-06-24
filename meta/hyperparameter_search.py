import random
from typing import Dict, Any

from meta.train import train


# Number of training runs to execute for each hyperparameter configuration.
TRIALS_PER_CONFIG = 3


# Fitness as a function of metrics returned from train().
get_fitness = lambda metrics: metrics["eval_reward"]["maximum"]


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


def hyperparameter_search(base_config: Dict[str, Any], iterations: int) -> None:
    """ Perform random search over hyperparameter configurations. """

    # Read in base name.
    base_name = base_config["save_name"]
    if base_name is None:
        raise ValueError(
            "config['save_name'] cannot be None for hyperparameter search."
        )

    # Set random seed.
    random.seed(base_config["seed"])

    # Training loop.
    config = base_config
    best_fitness = None
    best_config = base_config
    for iteration in range(iterations):

        # Perform training and compute resulting fitness for multiple trials.
        fitness = 0.0
        for trial in range(TRIALS_PER_CONFIG):

            # Set trial name and seed.
            trial_name = "%s_%d_%d" % (base_name, iteration, trial)
            config["save_name"] = trial_name
            config["seed"] = trial

            # Run training.

            # TEMP
            print("Trial name: %s" % trial_name)
            print("Config: %s" % config)
            # ----
            # metrics = train(config)
            # ----
            metrics = {"eval_reward": {"maximum": random.random()}}
            # ----
            print("Metrics: %s" % metrics)
            print("")

            trial_fitness = get_fitness(metrics)
            fitness += trial_fitness

        fitness /= TRIALS_PER_CONFIG

        # Compare current step to best so far.
        if best_fitness is None or fitness > best_fitness:
            print("New maximum reached: %.5f" % fitness)
            best_fitness = fitness
            best_config = config

        print("\n\n")

        # Mutate config to produce a new one for next step.
        config = mutate_config(best_config)
        while not valid_config(config):
            config = mutate_config(best_config)
