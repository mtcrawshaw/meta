""" Utility/helper methods specifically for meta/tune. """

import os
from typing import List, Dict, Any

from meta.utils.utils import save_dir_from_name


def get_experiment_names(
    base_name: str,
    search_type: str,
    iterations: int,
    trials_per_config: int,
    start_pos: Dict[str, int] = None,
    num_param_values: List[int] = None,
) -> List[str]:
    """
    Construct list of names of experiments that will be run during this hyperparameter
    tuning run. We do some weirdness here to handle cases of different search types,
    since the naming is slightly different for IC grid.
    """

    if search_type in ["grid", "random"]:
        assert num_param_values is None
        names_to_check = [base_name]

        # Handle case of `start_pos = None`.
        if start_pos is None:
            start_iteration = 0
            start_trial = 0
        else:
            start_iteration = start_pos["iteration"]
            start_trial = start_pos["trial"]

        # Iterate over iteration, trial.
        for iteration in range(start_iteration, iterations):
            start = start_trial if iteration == start_iteration else 0
            for trial in range(start, trials_per_config):
                names_to_check.append("%s_%d_%d" % (base_name, iteration, trial))

    elif search_type == "IC_grid":
        assert num_param_values is not None
        names_to_check = [base_name]

        # Handle case of `start_pos = None`.
        if start_pos is None:
            start_param = 0
            start_val = 0
            start_trial = 0
        else:
            start_param = start_pos["param"]
            start_val = start_pos["val"]
            start_trial = start_pos["trial"]

        # Iterate over param_num, param_iteration, trial.
        for param_num in range(start_param, len(num_param_values)):
            param_len = num_param_values[param_num]
            start_1 = start_val if param_num == start_param else 0
            for param_iteration in range(start_1, param_len):
                start_2 = (
                    start_trial
                    if param_num == start_param and param_iteration == start_1
                    else 0
                )
                for trial in range(start_2, trials_per_config):
                    names_to_check.append(
                        "%s_%d_%d_%d" % (base_name, param_num, param_iteration, trial)
                    )
    else:
        raise NotImplementedError

    return names_to_check


def check_name_uniqueness(
    base_name: str,
    search_type: str,
    iterations: int,
    trials_per_config: int,
    start_pos: Dict[str, int] = None,
    exempt_base: bool = False,
    num_param_values: List[int] = None,
) -> None:
    """
    Check to make sure that there are no other saved experiments whose names coincide
    with the current name. This is just to make sure that the saved results don't get
    mixed up, with some trials being saved with a modified name to ensure uniqueness.
    """

    # Build list of names to check.
    names_to_check = get_experiment_names(
        base_name,
        search_type,
        iterations,
        trials_per_config,
        start_pos,
        num_param_values,
    )

    # Check names.
    for name in names_to_check:
        if exempt_base and name == base_name:
            continue

        if os.path.isdir(save_dir_from_name(name)):
            raise ValueError(
                "Saved result '%s' already exists. Results of hyperparameter searches"
                " must have unique names." % name
            )


def get_start_pos(search_type: str, checkpoint: Dict[str, Any]) -> Dict[str, int]:
    """
    Computes starting position from a previous checkpoint. For example, with grid
    search, this will compute the iteration and trial index to start on when resuming
    from ``checkpoint``.
    """

    # Set default values before loading from checkpoint.
    if search_type in ["grid", "random"]:
        start_pos = {"iteration": 0, "trial": 0}
    elif search_type == "IC_grid":
        start_pos = {"param": 0, "val": 0, "trial": 0}

    # Load start position from checkpoint, if necessary.
    if checkpoint is not None:

        if search_type in ["grid", "random"]:
            start_pos["iteration"] = checkpoint["iteration"]
        else:
            start_pos["param"] = checkpoint["param_num"]
            start_pos["val"] = checkpoint["val_num"]

        if checkpoint["config_checkpoint"] is not None:
            start_pos["trial"] = checkpoint["config_checkpoint"]["trial"]

    return start_pos


def strip_config(config: Dict[str, Any], strip_seed: bool = False) -> Dict[str, Any]:
    """ Helper function to compare configs. """

    nonessential_params = ["save_name", "metrics_filename", "baseline_metrics_filename"]
    stripped = dict(config)
    for param in nonessential_params:
        del stripped[param]
    if strip_seed:
        del stripped["seed"]
    return stripped


def tune_results_equal(results1: Dict[str, Any], results2: Dict[str, Any]) -> bool:
    """
    Compares results returned from tune(), and returns true if the results are equal and
    false otherwise.
    """

    # This is to ensure that this function doesn't fail silently if the layout of tune
    # results ever changes. Instead, we fail loudly!
    for result in [results1, results2]:
        if set(result.keys()) != set(["best_config", "best_fitness", "iterations"]):
            raise NotImplementedError

        for iteration in result["iterations"]:
            if set(iteration.keys()) != set(["config", "fitness", "trials"]):
                raise NotImplementedError

            for trial in iteration["trials"]:
                if set(trial.keys()) != set(["fitness", "metrics", "trial"]):
                    raise NotImplementedError

    # Check best configs and best fitnesses.
    equal = True
    equal = equal and strip_config(
        results1["best_config"], strip_seed=True
    ) == strip_config(results2["best_config"], strip_seed=True)
    equal = equal and results1["best_fitness"] == results2["best_fitness"]

    # Check iteration results.
    equal = equal and len(results1["iterations"]) == len(results2["iterations"])
    if equal:
        for iteration in range(len(results1["iterations"])):
            iteration1 = results1["iterations"][iteration]
            iteration2 = results2["iterations"][iteration]

            # Check config and fitness.
            equal = equal and strip_config(
                iteration1["config"], strip_seed=True
            ) == strip_config(iteration2["config"], strip_seed=True)
            equal = equal and iteration1["fitness"] == iteration2["fitness"]

            # Check trial results.
            equal = equal and len(iteration1["trials"]) == len(iteration2["trials"])
            if not equal:
                break

            for trial in range(len(iteration1["trials"])):
                trial1 = iteration1["trials"][trial]
                trial2 = iteration2["trials"][trial]

                # Check fitness, metrics, and trial.
                equal = equal and trial1["fitness"] == trial2["fitness"]
                equal = equal and trial1["metrics"] == trial2["metrics"]
                equal = equal and trial1["trial"] == trial2["trial"]

            if not equal:
                break

    return equal
