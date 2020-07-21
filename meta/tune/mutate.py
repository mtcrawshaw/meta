import random
from typing import Dict, Any


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


def mutate_param(param_settings: Dict[str, Any], value: Any) -> Any:
    """ Mutate a single parameter and return a new value. """

    perturb_kwargs = param_settings["perturb_kwargs"]
    perturb = PERTURBATIONS[param_settings["perturb_type"]](**perturb_kwargs)
    min_value = param_settings["min_value"] if "min_value" in param_settings else None
    max_value = param_settings["max_value"] if "max_value" in param_settings else None

    # Perturb parameter.
    new_value = perturb(value)

    # Clip parameter, if necessary.
    if min_value is not None and max_value is not None:
        new_value = clip(new_value, min_value, max_value, value)

    return new_value


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
            new_config[param] = mutate_param(search_params[param], train_config[param])

        # Mutate next depth level of config dictionary, if necessary.
        if isinstance(train_config[param], dict):
            new_config[param] = mutate_train_config(search_params, train_config[param])

    return new_config
