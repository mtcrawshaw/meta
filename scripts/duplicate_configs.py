"""
Script to duplicate training config files to create copies with different random seeds,
and deleting the originals.
"""

import os
import argparse
import json
import glob


def get_copy_path(path: str, index: int) -> str:
    """ Construct a path to a duplicate config file from the original path. """

    # Get a stripped filename of the original config (filename without extension).
    filename = os.path.basename(path)
    end_pos = filename.rindex(".")
    filename_stripped = filename[:end_pos]
    extension = filename[end_pos:]

    # Construct copy path.
    new_filename = "%s_%d%s" % (filename_stripped, index, extension)
    new_path = os.path.join(os.path.dirname(path), new_filename)

    return new_path


def main(args):
    """ Main function of script. """

    # Get paths to configs.
    original_config_paths = glob.glob(args.targets)

    for config_path in original_config_paths:

        # Read in config.
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # Create copies of config at ``config_path``.
        for i in range(args.num_copies):
            config_copy = dict(config)
            config_copy["seed"] = i
            with open(get_copy_path(config_path, i), "w") as copy_file:
                json.dump(config_copy, copy_file, indent=4)

        # Delete original file.
        os.remove(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        type=str,
        help="Literal or wildcard expression giving the path to config files to duplicate.",
    )
    parser.add_argument(
        "num_copies", type=int, help="Number of copies to create of each config."
    )
    args = parser.parse_args()
    main(args)
