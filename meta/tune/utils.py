import os
from typing import List


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
