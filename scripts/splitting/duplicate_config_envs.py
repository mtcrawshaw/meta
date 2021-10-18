"""
Script to duplicate training config file to create copies with different random seeds
and environment names.
"""

import os
import argparse
import json
import glob
from datetime import date


ENV_NAMES = [
    "reach-v1",
    "push-v1",
    "pick-place-v1",
    "door-open-v1",
    "drawer-open-v1",
    "drawer-close-v1",
    "button-press-topdown-v1",
    "peg-insert-side-v1",
    "window-open-v1",
    "window-close-v1",
]


def get_copy_path(path: str, save_name: str) -> str:
    """ Construct a path to a duplicate config file from the original path. """

    # Get a stripped filename of the original config (filename without extension).
    filename = os.path.basename(path)
    end_pos = filename.rindex(".")
    extension = filename[end_pos:]

    # Construct copy path.
    new_filename = "%s%s" % (save_name, extension)
    new_path = os.path.join(os.path.dirname(path), new_filename)

    return new_path


def main(args) -> None:
    """ Main function of script. """

    # Read in config.
    with open(args.target, "r") as config_file:
        config = json.load(config_file)

    # Get current date for save names.
    today = date.today()
    date_str = today.strftime("%m%d%y")

    # Create copies of config for each environment.
    for env_name in ENV_NAMES:
        for i in range(args.num_copies):
            save_name = "%s_%s_%d" % (date_str, env_name, i)
            config_copy = dict(config)
            config_copy["env_name"] = env_name
            config_copy["seed"] = i
            config_copy["save_name"] = save_name
            with open(get_copy_path(args.target, save_name), "w") as copy_file:
                json.dump(config_copy, copy_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target", type=str, help="Path to config file to duplicate.",
    )
    parser.add_argument(
        "num_copies", type=int, help="Number of copies to create of each config."
    )
    args = parser.parse_args()
    main(args)
