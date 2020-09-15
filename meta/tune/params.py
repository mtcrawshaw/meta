from functools import reduce
from typing import Dict, List, Any


def valid_config(config: Dict[str, Any]) -> bool:
    """ Determine whether or not given configuration fits requirements. """

    valid = True

    # Test for requirements on num_minibatch, rollout_length, and num_processes detailed
    # in meta/storage.py (in this file, these conditions are checked at the beginning of
    # each generator definition, and an error is raised when they are violated)
    if (
        config["architecture_config"]["recurrent"]
        and config["num_processes"] < config["num_minibatch"]
    ):
        valid = False
    if not config["architecture_config"]["recurrent"]:
        total_steps = config["rollout_length"] * config["num_processes"]
        if total_steps < config["num_minibatch"]:
            valid = False

    return valid


def update_config(config: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates a config with new values. The keys of ``new_values`` are the names of the
    leaf nodes in ``config`` that should be updated. For example, if
    ``config["architecture_config"]["recurrent"]`` should be updated, that will appear
    in ``new_values`` as ``new_values["recurrent"]``. This why leaf nodes must have
    unique names.
    """

    # Construct updated config.
    new_config = dict(config)

    # Loop over each parameter and check for changes, or check for recursion.
    for param_name, value in config.items():
        if param_name in new_values:
            new_config[param_name] = new_values[param_name]
        elif isinstance(value, dict):
            new_config[param_name] = update_config(new_config[param_name], new_values)

    return new_config


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


def get_iterations(
    search_type: str, iterations: int, search_params: Dict[str, Any]
) -> int:
    """
    Get number of iterations based on configuration values. If the search type is
    "random", then we just return the number of iterations passed in. With search types
    "grid" and "IC_grid", we have to compute the number of iterations from
    ``search_params``.
    """

    new_iterations = 0
    if search_type in ["grid", "IC_grid"]:
        num_param_values = get_num_param_values(search_params)

        if search_type == "grid":
            new_iterations = reduce(lambda a, b: a * b, num_param_values)
        elif search_type == "IC_grid":
            new_iterations = sum(num_param_values)
        else:
            raise NotImplementedError

    elif search_type == "random":
        new_iterations = iterations

    else:
        raise NotImplementedError

    return new_iterations


def get_num_param_values(search_params: Dict[str, Any]) -> List[int]:
    """
    Given ``search_params`` from a tune configuration (this entry doesn't exist for
    random searches), returns the number of parameter values to iterate over for each
    parameter.
    """

    num_param_values = []
    for param_settings in search_params.values():
        if "num_values" in param_settings:
            num_param_values.append(param_settings["num_values"])
        elif "choices" in param_settings:
            num_param_values.append(len(param_settings["choices"]))
        else:
            raise ValueError("Invalid ``search_params`` value in config.")

    return num_param_values
