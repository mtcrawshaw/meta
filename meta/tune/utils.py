import os
from typing import List, Dict, Any

from meta.utils.utils import save_dir_from_name


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


def strip_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """ Helper function to compare configs. """

    nonessential_params = ["save_name", "metrics_filename", "baseline_metrics_filename"]
    stripped = dict(config)
    for param in nonessential_params:
        del stripped[param]
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
    equal = equal and strip_config(results1["best_config"]) == strip_config(
        results2["best_config"]
    )
    equal = equal and results1["best_fitness"] == results2["best_fitness"]

    # Check iteration results.
    equal = equal and len(results1["iterations"]) == len(results2["iterations"])
    if equal:
        for iteration in range(len(results1["iterations"])):
            iteration1 = results1["iterations"][iteration]
            iteration2 = results2["iterations"][iteration]

            # Check config and fitness.
            equal = equal and strip_config(iteration1["config"]) == strip_config(
                iteration2["config"]
            )
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